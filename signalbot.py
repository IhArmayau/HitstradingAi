from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import logging
import os
import joblib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from sklearn.ensemble import RandomForestClassifier
import san
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sqlalchemy
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, update, delete, func

# -----------------------------
# Database Setup
# -----------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

Base = declarative_base()

class SignalModel(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String)
    symbol = Column(String)
    market_type = Column(String)
    contract_address = Column(String, nullable=True)
    signal = Column(String)
    entry = Column(Float)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    confidence = Column(Float)
    safety_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=50.0)
    funding = Column(Float, default=0.0)
    open_interest = Column(Float, default=0.0)
    is_cluster = Column(Integer, default=0)
    status = Column(String, default='open')
    model_version = Column(String)
    vol_liq_ratio = Column(Float, default=0.0)
    time_to_close = Column(Integer, nullable=True)

class TrackedWallet(Base):
    __tablename__ = "tracked_wallets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=True)
    added_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class BotSetting(Base):
    __tablename__ = "bot_settings"
    key = Column(String, primary_key=True)
    value = Column(Text)

# -----------------------------
# Configs & Environment
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "Eo6zp2wemnkb4cui_thgwsepbufktb4qz")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuikPulseAI")

if SANTIMENT_API_KEY:
    san.ApiConfig.api_key = SANTIMENT_API_KEY

@dataclass
class TradeConfig:
    enabled: bool = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
    max_position_size_usd: float = float(os.getenv("MAX_POS_SIZE", 50.0))
    min_safety_score: float = float(os.getenv("MIN_SAFETY_SCORE", 70.0))
    min_sentiment_score: float = 60.0
    max_funding_threshold: float = 0.05
    min_win_rate_threshold: float = 0.40
    signal_cooldown_minutes: int = 60

@dataclass
class IndicatorsConfig:
    ema_short: int = int(os.getenv("EMA_SHORT", 9))
    ema_medium: int = int(os.getenv("EMA_MEDIUM", 21))
    adx_threshold: int = int(os.getenv("ADX_THRESHOLD", 25))
    atr_period: int = 14
    atr_tp_mult: float = float(os.getenv("ATR_TP_MULT", 3.0))
    atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", 1.5))

@dataclass
class BotConfig:
    enable_cex: bool = os.getenv("ENABLE_CEX_MONITOR", "true").lower() == "true"
    enable_dex: bool = os.getenv("ENABLE_DEX_MONITOR", "true").lower() == "true"
    symbols: List[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT").split(',')])
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    higher_timeframe: str = os.getenv("HIGHER_TIME_FRAME", "1h")
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    dex_poll_interval: int = 60
    cluster_window_minutes: int = 30
    min_insider_buy_sol: float = float(os.getenv("MIN_INSIDER_BUY_SOL", 2.0))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v5.9-alpha-hunter")
    tracked_wallets: List[str] = field(default_factory=lambda: [w.strip() for w in os.getenv("TRACKED_WALLETS", "").split(',') if w.strip()])

# -----------------------------
# Database Store
# -----------------------------
class SignalStore:
    def __init__(self, db_url: str):
        # Update: Limit connections to prevent Aiven SUPERUSER exhaustion
        self.engine = create_async_engine(
            db_url, 
            pool_size=2,          # Max 2 persistent connections
            max_overflow=0,       # No extra overflow connections
            pool_pre_ping=True,   # Check connection health before use
            pool_recycle=1800     # Refresh connections every 30 mins
        )
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.symbol_locks = {}

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database Schema Synchronized.")

    async def save_setting(self, key: str, value: Any):
        async with self.async_session() as session:
            await session.merge(BotSetting(key=key, value=str(value)))
            await session.commit()

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            s.setdefault('funding', 0.0)
            s.setdefault('open_interest', 0.0)
            new_sig = SignalModel(**s)
            session.add(new_sig)
            await session.commit()

    async def get_latest_signals(self, limit: int = 20):
        async with self.async_session() as session:
            result = await session.execute(select(SignalModel).order_by(SignalModel.id.desc()).limit(limit))
            return result.scalars().all()

    async def has_open_signal(self, iden: str):
        async with self.async_session() as session:
            q = select(SignalModel).where((SignalModel.symbol == iden) | (SignalModel.contract_address == iden)).where(SignalModel.status == 'open')
            result = await session.execute(q)
            return result.first() is not None

    async def add_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            try:
                session.add(TrackedWallet(address=address))
                await session.commit()
                return True
            except: return False

    async def remove_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
            await session.commit()

    async def get_all_tracked_wallets_detailed(self) -> Dict[str, str]:
        async with self.async_session() as session:
            res = await session.execute(select(TrackedWallet.address, TrackedWallet.label))
            return {row[0]: row[1] or row[0][:6] for row in res.all()}

    def get_symbol_lock(self, s):
        # FIX: Convert 's' to string to avoid "unhashable type: dict" error
        lock_key = str(s)
        if lock_key not in self.symbol_locks: self.symbol_locks[lock_key] = asyncio.Lock()
        return self.symbol_locks[lock_key]

# -----------------------------
# Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, helius_key: str, session: aiohttp.ClientSession, cfg: BotConfig):
        self.helius_key, self.session, self.cfg = helius_key, session, cfg
        self.known_exchanges = ["Binance", "Kraken", "Coinbase", "OKX", "Bybit", "KuCoin", "Gate.io"]
    async def is_whale_funded(self, wallet_address: str) -> bool:
        if not self.helius_key or not wallet_address: return False
        url = f"https://api.helius.xyz/v1/identities?api-key={self.helius_key}"
        try:
            async with self.session.post(url, json={"query": {"addresses": [wallet_address]}}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("identities", []):
                        if any(ex.lower() in item.get("name", "").lower() for ex in self.known_exchanges): return True
        except: pass
        return False

class SocialSentinel:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self.api_key, self.session = api_key, session
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self.api_key: return {"score": 50, "label": "Neutral"}
        slug = symbol.split('/')[0].lower()
        try:
            data = await asyncio.to_thread(san.get, "sentiment_balance_per_asset", slug=slug, from_date="now-1d", to_date="now", interval="1h")
            score = 50 if data.empty else max(0, min(100, int(((data.iloc[-1][0] + 5) / 10) * 100)))
            return {"score": score, "label": "Bullish" if score > 60 else "Bearish" if score < 40 else "Neutral"}
        except: return {"score": 50, "label": "Neutral"}

class DexEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    async def get_price_data(self, address: str) -> Dict[str, Any]:
        try:
            async with self.session.get(f"https://api.dexscreener.com/latest/dex/tokens/{address}") as resp:
                data = await resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    p = pairs[0]
                    return {"price": float(p.get('priceUsd', 0)), "symbol": p.get('baseToken', {}).get('symbol', 'UNK'), "vol24": float(p.get('volume', {}).get('h24', 0)), "liq": float(p.get('liquidity', {}).get('usd', 0))}
        except: pass
        return {"price": 0, "symbol": "UNK", "vol24": 0, "liq": 0}

class ClusterEngine:
    def __init__(self, window_mins: int):
        self.window, self.history = timedelta(minutes=window_mins), {}
    def record_and_check(self, mint: str) -> int:
        now = datetime.now(timezone.utc)
        if mint not in self.history: self.history[mint] = []
        self.history[mint].append(now)
        self.history[mint] = [t for t in self.history[mint] if now - t <= self.window]
        return len(self.history[mint])

class SecurityEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    async def get_safety_report(self, address: str, vol_24h: float, liq: float) -> Dict[str, Any]:
        vl_ratio = vol_24h / liq if liq > 0 else 999.0
        return {"safety_score": 80, "is_rugged": vl_ratio > 10.0, "vl_ratio": vl_ratio}

# -----------------------------
# Signal Generation
# -----------------------------
class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.dex, self.security = DexEngine(session), SecurityEngine(session)
        self.hunter = DiscoveryHunter(HELIUS_API_KEY, session, cfg)

    async def check_htf_trend(self, symbol: str) -> str:
        """Confirms 1H trend: BUY if price > 200 EMA, SELL if price < 200 EMA."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.higher_timeframe, limit=201)
            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            ema_200 = df['c'].ewm(span=200).mean().iloc[-1]
            last_price = df['c'].iloc[-1]
            return "BUY" if last_price > ema_200 else "SELL"
        except: return "NEUTRAL"

    async def generate_cex_signal(self, symbol: str, btc_bullish: bool):
        if not self.cfg.enable_cex or not self.cfg.trade.enabled: return
        async with self.store.get_symbol_lock(symbol):
            try:
                # 1. Fetch data & Indicators
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                
                # EMAs
                df['ema_s'] = df['c'].ewm(span=self.cfg.indicators.ema_short).mean()
                df['ema_m'] = df['c'].ewm(span=self.cfg.indicators.ema_medium).mean()
                
                # ADX (Trend Strength)
                adx_obj = ta.trend.ADXIndicator(df['h'], df['l'], df['c'], window=14)
                df['adx'] = adx_obj.adx()
                
                # ATR (Volatility for Anti-Stop Hunt)
                atr_obj = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c'], window=self.cfg.indicators.atr_period)
                df['atr'] = atr_obj.average_true_range()
                
                last = df.iloc[-1]
                adx_val = last['adx']
                volatility = last['atr']
                entry_price = last['c']

                # 2. EMA Crossover Signal
                stype = "BUY" if last['ema_s'] > last['ema_m'] else "SELL" if last['ema_s'] < last['ema_m'] else None

                # 3. Apply Professional Filters & Logic
                if stype and not await self.store.has_open_signal(symbol):
                    # Filter: ADX (Trend Strength)
                    if adx_val < self.cfg.indicators.adx_threshold:
                        logger.info(f"Skipping {symbol}: Weak Trend (ADX: {adx_val:.2f})")
                        return

                    # Filter: HTF Alignment
                    htf_trend = await self.check_htf_trend(symbol)
                    if stype != htf_trend:
                        logger.info(f"Skipping {symbol}: HTF Mismatch (1H: {htf_trend}, 5M: {stype})")
                        return

                    # Filter: Social Sentiment
                    social = await self.sentinel.get_sentiment(symbol)
                    if stype == "BUY" and social['score'] < self.cfg.trade.min_sentiment_score:
                        return

                    # Dynamic Anti-Stop Hunt SL/TP Calculation
                    if stype == "BUY":
                        sl = entry_price - (volatility * self.cfg.indicators.atr_sl_mult)
                        tp = entry_price + (volatility * self.cfg.indicators.atr_tp_mult)
                    else: # SELL
                        sl = entry_price + (volatility * self.cfg.indicators.atr_sl_mult)
                        tp = entry_price - (volatility * self.cfg.indicators.atr_tp_mult)

                    # 4. Market Stats
                    funding_rate = 0.0
                    open_interest = 0.0
                    try:
                        funding_data = await self.exchange.fetch_funding_rate(symbol)
                        funding_rate = float(funding_data.get('fundingRate', 0.0))
                        oi_data = await self.exchange.fetch_open_interest(symbol)
                        open_interest = float(oi_data.get('openInterestAmount') or oi_data.get('baseVolume', 0.0))
                    except: pass

                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "signal": stype,
                        "market_type": "CEX",
                        "entry": entry_price,
                        "sl": round(sl, 6),
                        "tp": round(tp, 6),
                        "confidence": 75.0 + (5.0 if adx_val > 40 else 0.0),
                        "sentiment_score": social['score'],
                        "funding": funding_rate,
                        "open_interest": open_interest,
                        "model_version": self.cfg.model_version
                    }
                    await self.store.insert_signal(sig)
                    await notify_new_signal(sig, self.session, self.cfg)
            except Exception as e:
                logger.error(f"CEX Signal Gen Error for {symbol}: {e}")

# -----------------------------
# FastAPI App
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

cfg = BotConfig()
store = SignalStore(DATABASE_URL)
cluster_map = ClusterEngine(cfg.cluster_window_minutes)
exchange = ccxt.kucoinfutures({"enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index(request: Request):
    try:
        signals = await store.get_latest_signals(limit=25)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "signals": signals,
            "bot_status": "ONLINE",
            "version": cfg.model_version
        })
    except Exception as e:
        logger.error(f"Index Error: {e}")
        return HTMLResponse(f"<html><body><h1>QuikPulse Dashboard</h1><p>Syncing signals... Error: {e}</p></body></html>")

@app.get("/health")
async def health_check():
    """Endpoint for UptimeRobot to keep the service awake."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/webhook")
async def helius_webhook_handler(request: Request):
    try:
        data = await request.json()
        db_wallet_map = await store.get_all_tracked_wallets_detailed()
        master_tracked = list(set(cfg.tracked_wallets + list(db_wallet_map.keys())))

        for event in data:
            if event.get("type") != "SWAP": continue
            swap_info = event.get("events", {}).get("swap", {})
            token_address = swap_info.get("tokenOutMint")
            buyer_wallet = event.get("feePayer")
            if not token_address: continue

            is_whale = await generator.hunter.is_whale_funded(buyer_wallet)
            is_sniper = buyer_wallet in master_tracked
            sniper_label = db_wallet_map.get(buyer_wallet, "Tracked Wallet")

            cluster_count = cluster_map.record_and_check(token_address)
            dex_data = await generator.dex.get_price_data(token_address)
            security = await generator.security.get_safety_report(token_address, dex_data['vol24'], dex_data['liq'])

            if (is_sniper or is_whale or cluster_count >= 2) and security['safety_score'] >= cfg.trade.min_safety_score:
                if not await store.has_open_signal(token_address):
                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": dex_data['symbol'],
                        "market_type": "DEX",
                        "contract_address": token_address,
                        "signal": "BUY",
                        "entry": dex_data['price'],
                        "confidence": 100.0 if is_sniper else 90.0,
                        "safety_score": security['safety_score'],
                        "vol_liq_ratio": security['vl_ratio'],
                        "is_cluster": 1 if cluster_count >= 2 else 0,
                        "model_version": cfg.model_version
                    }
                    await store.insert_signal(sig)
                    await notify_new_signal(sig, session, cfg, is_whale=is_whale, is_sniper=is_sniper, label=sniper_label)
        return {"status": "success"}
    except: return {"status": "error"}

@app.post("/tg-webhook")
async def telegram_command_handler(request: Request):
    try:
        data = await request.json()
        if "message" not in data: return {"ok": True}
        text = data["message"].get("text", "")
        chat_id = data["message"]["chat"]["id"]
        if str(chat_id) != cfg.telegram_chat_id: return {"ok": True}

        if text == "/status":
            msg = (f"🤖 **QuikPulse Status**\nCEX Monitor: `ONLINE`\nDEX Sniper: `ACTIVE`\nAuto-Trade: `{cfg.trade.enabled}`\nVersion: `{cfg.model_version}`")
            await send_direct_tg(msg)
        elif text == "/pnl":
            signals = await store.get_latest_signals(limit=5)
            pnl_msg = "📈 **Performance Analytics**\n\n"
            for s in signals: pnl_msg += f"• {s.symbol}: Entry ${s.entry} ({s.status})\n"
            await send_direct_tg(pnl_msg if signals else "No trade history found.")
        elif text == "/list":
            wallets = await store.get_all_tracked_wallets_detailed()
            msg = f"🔍 **CEX Pairs:**\n`{', '.join(cfg.symbols)}`\n\n🎯 **DEX Snipers:**\n"
            for addr, lbl in wallets.items(): msg += f"• {lbl}: `{addr[:8]}...`\n"
            await send_direct_tg(msg)
        elif text.startswith("/pair "):
            parts = text.split(" ")
            if len(parts) >= 3:
                cmd, target = parts[1], parts[2]
                if "/" in target:
                    if cmd == "add": cfg.symbols.append(target)
                    elif cmd == "rm" and target in cfg.symbols: cfg.symbols.remove(target)
                    await store.save_setting("symbols", ",".join(cfg.symbols))
                else:
                    if cmd == "add": await store.add_tracked_wallet(target)
                    elif cmd == "rm": await store.remove_tracked_wallet(target)
                await send_direct_tg(f"✅ Pairs list updated.")
        elif text.startswith("/set "):
            parts = text.split(" ")
            if len(parts) == 3:
                key, val = parts[1], parts[2]
                if hasattr(cfg.indicators, key):
                    setattr(cfg.indicators, key, float(val))
                    await store.save_setting(key, val)
                    await send_direct_tg(f"⚙️ `{key}` updated to `{val}`")
        elif text == "/resume":
            cfg.trade.enabled = True
            await store.save_setting("auto_trade", "true")
            await send_direct_tg("🟢 **Auto-Trading Resumed.**")
        elif text == "/help":
            help_txt = ("📖 **QuikPulse Commands**\n• `/pair add [Symbol/Wallet]`\n• `/set [Key] [Value]`\n• `/status`\n• `/pnl`\n• `/list`")
            await send_direct_tg(help_txt)
        return {"ok": True}
    except: return {"ok": True}

@app.on_event("startup")
async def startup():
    global session, generator
    await store.init_db()
    session = aiohttp.ClientSession()
    sentinel = SocialSentinel(SANTIMENT_API_KEY, session)
    generator = SignalGenerator(cfg, store, exchange, sentinel, cluster_map, session)
    asyncio.create_task(background_monitor())
    logger.info("QuikPulse AI Live on Render.")

async def background_monitor():
    while True:
        try:
            for s in cfg.symbols:
                await generator.generate_cex_signal(str(s), True)
                await asyncio.sleep(1)
        except: pass
        await asyncio.sleep(cfg.poll_interval)

async def notify_new_signal(sig, session, cfg, is_whale=False, is_sniper=False, label=None):
    if not cfg.telegram_bot_token: return
    prefix = f"🎯 *SNIPER ({label})*" if is_sniper else "🐋 *WHALE*" if is_whale else "🚀 *NEW*"

    sl_tp_info = ""
    if sig.get('sl') and sig.get('tp'):
        sl_tp_info = f"\n🛡️ *Protection:*\n└ TP: `${sig['tp']}`\n└ SL: `${sig['sl']}`"

    extra = ""
    if sig.get('market_type') == "CEX":
        f_rate = sig.get('funding', 0) * 100
        oi = sig.get('open_interest', 0)
        extra = f"\n📊 *Market Stats:*\n└ Funding: `{f_rate:.4f}%` \n└ OI: `{oi:,.0f}`"

    msg = f"{prefix} SIGNAL\nPair: `{sig['symbol']}`\nType: {sig['signal']}\nPrice: `${sig['entry']}`{sl_tp_info}{extra}"
    try:
        await session.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
                          json={"chat_id": cfg.telegram_chat_id, "text": msg, "parse_mode": "Markdown"})
    except: pass

async def send_direct_tg(text: str):
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = {"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"}
    async with session.post(url, json=payload) as resp: return await resp.json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
