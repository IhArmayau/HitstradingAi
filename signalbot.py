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
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, update, delete, func

# -----------------------------
# Database Setup (PostgreSQL Ready)
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
    is_cluster = Column(Integer, default=0)
    status = Column(String, default='open')
    model_version = Column(String)
    vol_liq_ratio = Column(Float, default=0.0)
    time_to_close = Column(Integer, nullable=True)

class TrackedWallet(Base):
    """Table to store wallets added via Telegram commands for persistence"""
    __tablename__ = "tracked_wallets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=True)
    added_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

# -----------------------------
# Configs & Environment
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "Eo6zp2wemnkb4cui_thgwsepbufktb4qz")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
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
        self.engine = create_async_engine(db_url, pool_pre_ping=True)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.symbol_locks = {}

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database Schema Synchronized.")

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            new_sig = SignalModel(**s)
            session.add(new_sig)
            await session.commit()

    async def update_signal_status(self, sig_id: int, status: str, duration: int = 0):
        async with self.async_session() as session:
            q = update(SignalModel).where(SignalModel.id == sig_id).values(status=status, time_to_close=duration)
            await session.execute(q)
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
                new_w = TrackedWallet(address=address)
                session.add(new_w)
                await session.commit()
                return True
            except: return False

    async def remove_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
            await session.commit()

    async def get_all_tracked_wallets(self) -> List[str]:
        async with self.async_session() as session:
            res = await session.execute(select(TrackedWallet.address))
            return list(res.scalars().all())

    def get_symbol_lock(self, s):
        if s not in self.symbol_locks: self.symbol_locks[s] = asyncio.Lock()
        return self.symbol_locks[s]

# -----------------------------
# Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, helius_key: str, session: aiohttp.ClientSession, cfg: BotConfig):
        self.helius_key = helius_key
        self.session = session
        self.cfg = cfg
        self.known_exchanges = ["Binance", "Kraken", "Coinbase", "OKX", "Bybit", "KuCoin", "Gate.io"]

    async def is_whale_funded(self, wallet_address: str) -> bool:
        if not self.helius_key or not wallet_address: return False
        url = f"https://api.helius.xyz/v1/identities?api-key={self.helius_key}"
        try:
            async with self.session.post(url, json={"query": {"addresses": [wallet_address]}}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("identities", []):
                        if any(ex.lower() in item.get("name", "").lower() for ex in self.known_exchanges):
                            logger.info(f"🐋 WHALE DETECTED: {wallet_address} funded by {item.get('name')}")
                            return True
        except Exception as e:
            logger.error(f"Whale Check Error: {e}")
        return False

    async def scan_for_insiders(self, token_address: str):
        pass 

class SocialSentinel:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self.api_key, self.session = api_key, session

    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self.api_key: return {"score": 50, "label": "Neutral", "funding": 0.0}
        slug = symbol.split('/')[0].lower()
        try:
            data = await asyncio.to_thread(san.get, "sentiment_balance_per_asset", slug=slug, from_date="now-1d", to_date="now", interval="1h")
            score = 50
            if not data.empty:
                val = data.iloc[-1][0]
                score = max(0, min(100, int(((val + 5) / 10) * 100)))
            return {"score": score, "label": "Bullish" if score > 60 else "Bearish" if score < 40 else "Neutral", "funding": 0.0, "slug": slug}
        except: return {"score": 50, "label": "Neutral", "funding": 0.0}

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
        self.window = timedelta(minutes=window_mins)
        self.history = {}
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
        score = 80
        return {"safety_score": score, "is_rugged": vl_ratio > 10.0, "vl_ratio": vl_ratio}

# -----------------------------
# Signal Generation
# -----------------------------
class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.dex, self.security = DexEngine(session), SecurityEngine(session)
        self.hunter = DiscoveryHunter(HELIUS_API_KEY, session, cfg)

    async def generate_cex_signal(self, symbol: str, btc_bullish: bool):
        now = datetime.now(timezone.utc)
        if not self.cfg.enable_cex or not self.cfg.trade.enabled: return
        async with self.store.get_symbol_lock(symbol):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=50)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                df['ema_s'] = df['c'].ewm(span=self.cfg.indicators.ema_short).mean()
                df['ema_m'] = df['c'].ewm(span=self.cfg.indicators.ema_medium).mean()
                last = df.iloc[-1]
                stype = "BUY" if last['ema_s'] > last['ema_m'] else "SELL" if last['ema_s'] < last['ema_m'] else None
                if stype and not await self.store.has_open_signal(symbol):
                    social = await self.sentinel.get_sentiment(symbol)
                    sig = {
                        "timestamp": now.isoformat(), "symbol": symbol, "signal": stype, "market_type": "CEX",
                        "entry": last['c'], "confidence": 75.0, "sentiment_score": social['score'],
                        "model_version": self.cfg.model_version
                    }
                    await self.store.insert_signal(sig)
                    await notify_new_signal(sig, self.session, self.cfg)
            except: pass

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
cfg = BotConfig()
store = SignalStore(DATABASE_URL)
cluster_map = ClusterEngine(cfg.cluster_window_minutes)
exchange = ccxt.kucoinfutures({"enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        signals = await store.get_latest_signals(limit=20)
        return templates.TemplateResponse("index.html", {"request": request, "signals": signals})
    except Exception as e:
        return HTMLResponse("<html><body><h1>Database Syncing...</h1></body></html>")

@app.post("/webhook")
async def helius_webhook_handler(request: Request):
    try:
        data = await request.json()
        db_wallets = await store.get_all_tracked_wallets()
        master_tracked = list(set(cfg.tracked_wallets + db_wallets))

        for event in data:
            tx_type = event.get("type")
            if tx_type != "SWAP": continue
            
            swap_info = event.get("events", {}).get("swap", {})
            token_address = swap_info.get("tokenOutMint")
            buyer_wallet = event.get("feePayer")
            
            if not token_address: continue

            is_whale = await generator.hunter.is_whale_funded(buyer_wallet)
            is_manual_sniper = buyer_wallet in master_tracked
            
            cluster_count = cluster_map.record_and_check(token_address)
            dex_data = await generator.dex.get_price_data(token_address)
            security_report = await generator.security.get_safety_report(token_address, dex_data['vol24'], dex_data['liq'])

            trigger_signal = False
            confidence = 85.0
            
            if is_manual_sniper:
                trigger_signal, confidence = True, 100.0
            elif is_whale:
                trigger_signal, confidence = True, 95.0
            elif cluster_count >= 2:
                trigger_signal = True

            if trigger_signal and security_report['safety_score'] >= cfg.trade.min_safety_score:
                if not await store.has_open_signal(token_address):
                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": dex_data['symbol'], "market_type": "DEX", "contract_address": token_address,
                        "signal": "BUY", "entry": dex_data['price'], "confidence": confidence,
                        "safety_score": security_report['safety_score'], "vol_liq_ratio": security_report['vl_ratio'],
                        "is_cluster": 1 if cluster_count >= 2 else 0, "model_version": cfg.model_version
                    }
                    await store.insert_signal(sig)
                    await notify_new_signal(sig, session, cfg, is_whale=is_whale, is_sniper=is_manual_sniper)

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return {"status": "error"}

@app.post("/tg-webhook")
async def telegram_command_handler(request: Request):
    """Processes /add, /remove, and /list commands from Telegram Webhook"""
    try:
        data = await request.json()
        if "message" not in data: return {"ok": True}
        
        text = data["message"].get("text", "")
        chat_id = data["message"]["chat"]["id"]

        if str(chat_id) != cfg.telegram_chat_id: return {"ok": True}

        if text.startswith("/add "):
            wallet = text.split(" ")[1].strip()
            success = await store.add_tracked_wallet(wallet)
            msg = f"✅ Added to Sniper List:\n`{wallet}`" if success else "⚠️ Already being tracked."
            await send_direct_tg(msg)

        elif text.startswith("/remove "):
            wallet = text.split(" ")[1].strip()
            await store.remove_tracked_wallet(wallet)
            await send_direct_tg(f"❌ Removed from Sniper List:\n`{wallet}`")

        elif text == "/list":
            wallets = await store.get_all_tracked_wallets()
            if not wallets:
                await send_direct_tg("📭 Your Sniper List is currently empty.")
            else:
                list_msg = "🎯 **Active Sniper Wallets:**\n\n" + "\n".join([f"• `{w}`" for w in wallets])
                await send_direct_tg(list_msg)

        return {"ok": True}
    except:
        return {"ok": True}

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
                await generator.generate_cex_signal(s, True)
                await asyncio.sleep(1)
        except: pass
        await asyncio.sleep(cfg.poll_interval)

async def notify_new_signal(sig, session, cfg, is_whale=False, is_sniper=False):
    if not cfg.telegram_bot_token: return
    prefix = "🎯 *SNIPER SIGNAL*" if is_sniper else "🐋 *WHALE SIGNAL*" if is_whale else "🚀 *NEW SIGNAL*"
    msg = f"{prefix}\nPair: `{sig['symbol']}`\nType: {sig['signal']}\nPrice: `${sig['entry']}`\nConfidence: {sig['confidence']}%"
    try: await session.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage", json={"chat_id": cfg.telegram_chat_id, "text": msg, "parse_mode": "Markdown"})
    except: pass

async def send_direct_tg(text: str):
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = {"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"}
    async with session.post(url, json=payload) as resp:
        return await resp.json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
