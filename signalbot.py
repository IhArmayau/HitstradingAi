from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import logging
import os
import json
import pickle
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from sklearn.ensemble import RandomForestClassifier
import san
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, delete

# -----------------------------
# Logging Configuration (Optimized for Render/Stdout)
# -----------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuikPulseAI")

# -----------------------------
# Database Setup
# -----------------------------
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
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "Eo6zp2wemnkb4cui_thgwsepbufktb4qz")

if SANTIMENT_API_KEY:
    san.ApiConfig.api_key = SANTIMENT_API_KEY

@dataclass
class TradeConfig:
    enabled: bool = False
    max_position_size_usd: float = float(os.getenv("MAX_POS_SIZE", 50.0))
    min_safety_score: float = float(os.getenv("MIN_SAFETY_SCORE", 70.0))
    min_sentiment_score: float = 60.0
    max_funding_threshold: float = 0.05
    min_win_rate_threshold: float = 0.40
    signal_cooldown_minutes: int = int(os.getenv("SIGNAL_COOLDOWN", 60))

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
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/latest_model.pkl")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = "v6.1-pro-prod-signal"
    tracked_wallets: List[str] = field(default_factory=lambda: [w.strip() for w in os.getenv("TRACKED_WALLETS", "").split(',') if w.strip()])

# -----------------------------
# Production Utility: API Resilience
# -----------------------------
async def request_with_retry(session: aiohttp.ClientSession, method: str, url: str, retries: int = 3, **kwargs):
    for i in range(retries):
        try:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 429:
                    wait = (i + 1) * 5
                    logger.warning(f"Rate limited on {url}. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                return await resp.json()
        except Exception as e:
            if i == retries - 1: raise e
            await asyncio.sleep(2 ** i)

# -----------------------------
# Database Store
# -----------------------------
class SignalStore:
    def __init__(self, db_url: str):
        # FIX: Removed pool_size/max_overflow which can cause issues with some asyncpg drivers on cloud hosts
        self.engine = create_async_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
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
            new_sig = SignalModel(**s)
            session.add(new_sig)
            await session.commit()

    async def get_latest_signals(self, limit: int = 25):
        async with self.async_session() as session:
            try:
                result = await session.execute(select(SignalModel).order_by(SignalModel.id.desc()).limit(limit))
                rows = result.scalars().all()
                return [{
                    "id": int(r.id), "symbol": str(r.symbol), "signal": str(r.signal),
                    "entry": float(r.entry or 0.0), "sl": float(r.sl) if r.sl else None,
                    "tp": float(r.tp) if r.tp else None, "confidence": float(r.confidence or 0.0),
                    "market_type": str(r.market_type), "status": str(r.status),
                    "funding": float(r.funding or 0.0), "open_interest": float(r.open_interest or 0.0),
                    "sentiment_score": float(r.sentiment_score or 50.0),
                    "vol_liq_ratio": float(r.vol_liq_ratio or 0.0)
                } for r in rows]
            finally:
                await session.close()

    async def has_open_signal(self, iden: str):
        async with self.async_session() as session:
            q = select(SignalModel).where((SignalModel.symbol == iden) | (SignalModel.contract_address == iden)).where(SignalModel.status == 'open')
            result = await session.execute(q)
            return result.scalars().first() is not None

    def get_symbol_lock(self, s):
        lock_key = str(s)
        if lock_key not in self.symbol_locks:
            self.symbol_locks[lock_key] = asyncio.Lock()
        return self.symbol_locks[lock_key]

    async def get_all_tracked_wallets_detailed(self) -> Dict[str, str]:
        async with self.async_session() as session:
            res = await session.execute(select(TrackedWallet.address, TrackedWallet.label))
            return {str(row[0]): (str(row[1]) if row[1] else str(row[0])[:6]) for row in res.all()}

# -----------------------------
# Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, helius_key: str, session: aiohttp.ClientSession, cfg: BotConfig):
        self.helius_key, self.session, self.cfg = helius_key, session, cfg
        self.known_exchanges = ["Binance", "Kraken", "Coinbase", "OKX", "Bybit", "KuCoin"]

    async def is_whale_funded(self, wallet_address: str) -> bool:
        if not self.helius_key or not wallet_address: return False
        url = f"https://api.helius.xyz/v1/identities?api-key={self.helius_key}"
        try:
            data = await request_with_retry(self.session, "POST", url, json={"query": {"addresses": [wallet_address]}})
            for item in data.get("identities", []):
                if any(ex.lower() in item.get("name", "").lower() for ex in self.known_exchanges): return True
        except: return False
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
            return {"score": score}
        except: return {"score": 50}

class DexEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def get_price_data(self, address: str) -> Dict[str, Any]:
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
            async with self.session.get(url, timeout=10) as resp:
                data = await resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    p = pairs[0]
                    return {
                        "price": float(p.get('priceUsd', 0)),
                        "symbol": p.get('baseToken', {}).get('symbol', 'UNK'),
                        "vol24": float(p.get('volume', {}).get('h24', 0)),
                        "liq": float(p.get('liquidity', {}).get('usd', 0))
                    }
        except Exception as e:
            logger.error(f"DexEngine Fetch Error: {e}")
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
        vl_ratio = vol_24h / liq if liq > 0 else 0.0
        return {"safety_score": 80 if (0 < vl_ratio < 5) else 40, "is_rugged": vl_ratio > 10.0, "vl_ratio": vl_ratio}

# -----------------------------
# Production Execution Engine
# -----------------------------
class TradeExecutor:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    async def execute_trade(self, sig: dict):
        logger.info(f"📣 [SIGNAL] {sig['market_type']} | {sig['symbol']} | {sig['signal']} @ {sig['entry']}")

class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.dex, self.security = DexEngine(session), SecurityEngine(session)
        self.hunter = DiscoveryHunter(HELIUS_API_KEY, session, cfg)
        self.executor = TradeExecutor(cfg)
        self.active_monitors_data: Dict[str, float] = {} 
        self.cooldown_cache = {}
        self.prev_oi = {}
        self.ml_model = self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.cfg.ml_model_path):
                with open(self.cfg.ml_model_path, 'rb') as f: return pickle.load(f)
            else:
                logger.warning(f"ML Model file missing at {self.cfg.ml_model_path}. Defaulting to heuristic confidence.")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
        return None

    def predict_confidence(self, features: list) -> float:
        if self.ml_model:
            try: return float(self.ml_model.predict_proba([features])[0][1] * 100)
            except: return 50.0
        return 70.0

    async def get_btc_trend_filter(self) -> str:
        try:
            ohlcv = await self.exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=50)
            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            ema = df['c'].ewm(span=20).mean().iloc[-1]
            return "BULLISH" if df['c'].iloc[-1] > ema else "BEARISH"
        except: return "NEUTRAL"

    async def analyze_funding_squeeze(self, symbol: str):
        try:
            f_data = await self.exchange.fetch_funding_rate(symbol)
            oi_data = await self.exchange.fetch_open_interest(symbol)
            funding = float(f_data.get('fundingRate', 0.0))
            oi = float(oi_data.get('openInterestAmount') or 0.0)
            last_oi = self.prev_oi.get(symbol, 0)
            oi_growth = (oi > last_oi * 1.05) if last_oi > 0 else False
            self.prev_oi[symbol] = oi
            return funding, oi, (funding < -0.01 and oi_growth)
        except: return 0.0, 0.0, False

    async def generate_cex_signal(self, symbol: str):
        if self.cooldown_cache.get(symbol) and (datetime.now() - self.cooldown_cache[symbol]) < timedelta(minutes=self.cfg.trade.signal_cooldown_minutes):
            return
        
        async with self.store.get_symbol_lock(symbol):
            try:
                btc_trend = await self.get_btc_trend_filter()
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                df['adx'] = ta.trend.ADXIndicator(df['h'], df['l'], df['c']).adx()
                if df['adx'].iloc[-1] < self.cfg.indicators.adx_threshold: return

                df['ema_s'] = df['c'].ewm(span=self.cfg.indicators.ema_short).mean()
                df['ema_m'] = df['c'].ewm(span=self.cfg.indicators.ema_medium).mean()
                df['atr'] = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c']).average_true_range()
                last = df.iloc[-1]
                stype = "BUY" if last['ema_s'] > last['ema_m'] else "SELL" if last['ema_s'] < last['ema_m'] else None

                if stype and not await self.store.has_open_signal(symbol):
                    funding, oi, is_squeeze = await self.analyze_funding_squeeze(symbol)
                    if stype == "BUY" and btc_trend == "BEARISH" and not is_squeeze: return

                    social = await self.sentinel.get_sentiment(symbol)
                    entry_price = float(last['c'])
                    sl = entry_price - (last['atr'] * self.cfg.indicators.atr_sl_mult) if stype == "BUY" else entry_price + (last['atr'] * self.cfg.indicators.atr_sl_mult)
                    tp = entry_price + (last['atr'] * self.cfg.indicators.atr_tp_mult) if stype == "BUY" else entry_price - (last['atr'] * self.cfg.indicators.atr_tp_mult)

                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol, "signal": stype, "market_type": "CEX",
                        "entry": entry_price, "sl": round(sl, 6), "tp": round(tp, 6),
                        "confidence": self.predict_confidence([funding, social['score']]),
                        "sentiment_score": social['score'], "funding": funding, "open_interest": oi,
                        "model_version": self.cfg.model_version,
                        "vol_liq_ratio": 0.0 
                    }
                    await self.store.insert_signal(sig)
                    self.cooldown_cache[symbol] = datetime.now()
                    await self.executor.execute_trade(sig)
                    await notify_new_signal(sig, self.session, self.cfg, is_squeeze=is_squeeze)
            except Exception as e: logger.error(f"CEX Error: {e}")

# -----------------------------
# FastAPI Service
# -----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
cfg = BotConfig()
store = SignalStore(DATABASE_URL)
cluster_map = ClusterEngine(cfg.cluster_window_minutes)
exchange = ccxt.kucoinfutures({"enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None
background_tasks = set()

@app.get("/health")
async def health():
    return {"status": "online", "monitors": len(background_tasks), "version": cfg.model_version}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "bot_status": "ONLINE",
        "version": cfg.model_version
    })

@app.get("/signals", response_class=HTMLResponse)
async def get_signals_partial(request: Request):
    try:
        raw_signals = await store.get_latest_signals()
        return templates.TemplateResponse("signals_partial.html", {
            "request": request,
            "signals": raw_signals
        })
    except Exception as e:
        logger.error(f"Partial Render Error: {e}")
        return HTMLResponse(content="<tr><td colspan='5' class='py-10 text-center text-red-500'>Backend Error</td></tr>", status_code=500)

@app.post("/webhook")
async def helius_webhook(request: Request):
    try:
        # PRODUCTION FIX: Full request visibility
        body = await request.body()
        data = json.loads(body)
        logger.info(f"🔗 [WEBHOOK] Incoming Hit: {len(data)} events detected.")
        
        db_wallets = await store.get_all_tracked_wallets_detailed()
        for event in data:
            if event.get("type") != "SWAP": continue
            swap = event.get("events", {}).get("swap", {})
            mint, buyer = swap.get("tokenOutMint"), event.get("feePayer")
            
            logger.info(f"🔍 [SCAN] Processing swap: {mint} | Wallet: {buyer}")
            
            is_whale = await generator.hunter.is_whale_funded(buyer)
            is_sniper = buyer in db_wallets

            if (is_sniper or is_whale) and not await store.has_open_signal(mint):
                logger.info(f"🎯 [TARGET] Valid Signal for {mint} (Whale={is_whale}, Sniper={is_sniper})")
                dex_data = await generator.dex.get_price_data(mint)
                safety = await generator.security.get_safety_report(mint, dex_data['vol24'], dex_data['liq'])
                
                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": dex_data['symbol'], "market_type": "DEX", "contract_address": mint,
                    "signal": "BUY", "entry": dex_data['price'], "confidence": 95.0, 
                    "model_version": cfg.model_version,
                    "vol_liq_ratio": safety['vl_ratio'],
                    "safety_score": safety['safety_score']
                }
                await store.insert_signal(sig)
                await generator.executor.execute_trade(sig)
                await notify_new_signal(sig, session, cfg, is_whale=is_whale, is_sniper=is_sniper)
                
                if mint not in generator.active_monitors_data:
                    generator.active_monitors_data[mint] = float(dex_data['price'])
                    
        return JSONResponse(content={"status": "success"}, status_code=200)
    except Exception as e: 
        logger.error(f"❌ [WEBHOOK ERROR] {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

async def centralized_dex_watcher():
    while True:
        try:
            tokens_to_check = list(generator.active_monitors_data.items())
            for mint, entry in tokens_to_check:
                data = await generator.dex.get_price_data(mint)
                if data['price'] <= 0: continue
                
                if data['price'] >= entry * 1.5:
                    await send_direct_tg(f"💰 **DEX TP**\nToken: `{data['symbol']}`\nGain: `+50%`")
                    generator.active_monitors_data.pop(mint, None)
                elif data['price'] <= entry * 0.8:
                    await send_direct_tg(f"⚠️ **DEX SL**\nToken: `{data['symbol']}`\nLoss: `-20%`")
                    generator.active_monitors_data.pop(mint, None)
                
                await asyncio.sleep(2)
                
            await asyncio.sleep(60)
        except asyncio.CancelledError: break
        except Exception as e:
            logger.error(f"DEX Watcher Error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    global session, generator
    await store.init_db()
    session = aiohttp.ClientSession()
    sentinel = SocialSentinel(SANTIMENT_API_KEY, session)
    generator = SignalGenerator(cfg, store, exchange, sentinel, cluster_map, session)
    
    monitor_task = asyncio.create_task(background_monitor())
    dex_watcher_task = asyncio.create_task(centralized_dex_watcher())
    
    background_tasks.add(monitor_task)
    background_tasks.add(dex_watcher_task)
    logger.info(f"✅ QuikPulse AI {cfg.model_version} Startup Complete.")

@app.on_event("shutdown")
async def shutdown():
    for task in background_tasks: task.cancel()
    if session: await session.close()
    await exchange.close()

async def background_monitor():
    while True:
        try:
            for s in cfg.symbols:
                await generator.generate_cex_signal(s)
                await asyncio.sleep(5)
            await asyncio.sleep(cfg.poll_interval)
        except asyncio.CancelledError: break
        except Exception as e:
            logger.error(f"Monitor Error: {e}")
            await asyncio.sleep(60)

async def notify_new_signal(sig, session, cfg, is_whale=False, is_sniper=False, is_squeeze=False):
    if not cfg.telegram_bot_token or not session: return
    prefix = "🚨 *SQUEEZE*" if is_squeeze else "🎯 *SNIPER*" if is_sniper else "🐋 *WHALE*" if is_whale else "🚀 *SIGNAL*"
    msg = f"{prefix}\nPair: `{sig['symbol']}`\nAction: {sig['signal']}\nEntry: `${sig['entry']}`"
    await send_direct_tg(msg)

async def send_direct_tg(text: str):
    if not session or not cfg.telegram_bot_token: return
    try:
        url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
        await request_with_retry(session, "POST", url, json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")

if __name__ == "__main__":
    # PRODUCTION FIX: Log Level to INFO for Render visibility
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        access_log=True
    )
