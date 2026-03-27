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
import san
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, delete, func

# Import for .h5 support
try:
    from tensorflow.keras.models import load_model as load_keras_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# -----------------------------
# Logging Configuration
# -----------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

file_handler = logging.FileHandler("quikpulse_audit.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuikPulseAI")
logger.addHandler(file_handler)

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
    symbol = Column(String, index=True)
    market_type = Column(String)
    contract_address = Column(String, nullable=True, index=True)
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
    priority = Column(Integer, default=0) 

class TrackedWallet(Base):
    __tablename__ = "tracked_wallets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, unique=True, nullable=False, index=True)
    label = Column(String, nullable=True)
    added_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class WalletCandidate(Base):
    __tablename__ = "wallet_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, nullable=False, index=True)
    token_mint = Column(String, nullable=False)
    entry_price = Column(Float)
    timestamp = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    is_win = Column(Integer, default=0)

class MonitoredPair(Base):
    __tablename__ = "monitored_pairs"
    symbol = Column(String, primary_key=True)
    added_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class BotSetting(Base):
    __tablename__ = "bot_settings"
    key = Column(String, primary_key=True)
    value = Column(Text)

# -----------------------------
# Configs & Environment
# -----------------------------
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "") 
ALCHEMY_AUTH_TOKEN = os.getenv("ALCHEMY_AUTH_TOKEN", "") 
ALCHEMY_WEBHOOK_ID = os.getenv("ALCHEMY_WEBHOOK_ID", "")
ALCHEMY_RPC_URL = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "Eo6zp2wemnkb4cui_thgwsepbufktb4qz")

if SANTIMENT_API_KEY:
    san.ApiConfig.api_key = SANTIMENT_API_KEY

@dataclass
class TradeConfig:
    enabled: bool = os.getenv("TRADE_ENABLED", "true").lower() == "true"
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
    min_hunter_profit_mult: float = 3.0
    min_hunter_wins_required: int = 3
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/lstm_model.h5")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = "v7.7-alchemy-prod"

# -----------------------------
# Utility: API Resilience
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
        self.engine = create_async_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.symbol_locks = {}
        self.wallet_cache = {}

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self.refresh_wallet_cache()
        logger.info("Database Schema Synchronized & Wallet Cache Hydrated.")

    async def refresh_wallet_cache(self):
        self.wallet_cache = await self.get_all_tracked_wallets_detailed()

    async def save_setting(self, key: str, value: Any):
        async with self.async_session() as session:
            await session.merge(BotSetting(key=key, value=str(value)))
            await session.commit()

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            new_sig = SignalModel(**s)
            session.add(new_sig)
            await session.commit()
            logger.info(f"AUDIT: Signal Stored: {s['symbol']} - {s['signal']}")

    async def add_tracked_wallet(self, address: str, label: str = "Manual"):
        async with self.async_session() as session:
            await session.merge(TrackedWallet(address=address, label=label))
            await session.commit()
            await self.refresh_wallet_cache()
            await sync_alchemy_webhook(list(self.wallet_cache.keys()))

    async def remove_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
            await session.commit()
            await self.refresh_wallet_cache()
            await sync_alchemy_webhook(list(self.wallet_cache.keys()))

    async def remove_cex_pair(self, symbol: str):
        async with self.async_session() as session:
            await session.execute(delete(MonitoredPair).where(MonitoredPair.symbol == symbol))
            await session.commit()

    async def insert_candidate(self, address: str, mint: str, price: float):
        async with self.async_session() as session:
            session.add(WalletCandidate(address=address, token_mint=mint, entry_price=price))
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
                    "sentiment_score": float(r.sentiment_score or 50.0), "vol_liq_ratio": float(r.vol_liq_ratio or 0.0)
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
# Alchemy Sync Utility
# -----------------------------
async def sync_alchemy_webhook(addresses: List[str]):
    if not ALCHEMY_AUTH_TOKEN or not ALCHEMY_WEBHOOK_ID: return
    url = f"https://dashboard.alchemy.com/api/update-webhook-addresses"
    headers = {"X-Alchemy-Token": ALCHEMY_AUTH_TOKEN, "Content-Type": "application/json"}
    payload = {"webhook_id": ALCHEMY_WEBHOOK_ID, "addresses_to_add": addresses, "addresses_to_remove": []}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.patch(url, json=payload, headers=headers) as resp:
                logger.info(f"Alchemy Sync Status: {resp.status}")
    except Exception as e:
        logger.error(f"Alchemy Sync Failed: {e}")

# -----------------------------
# Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, helius_key: str, session: aiohttp.ClientSession, cfg: BotConfig):
        self.helius_key, self.session, self.cfg = helius_key, session, cfg

    async def is_whale_funded(self, wallet_address: str) -> bool:
        if not ALCHEMY_API_KEY or not wallet_address: return False
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_address]}
        try:
            data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=payload)
            balance = int(data.get("result", {}).get("value", 0))
            return balance > 50_000_000_000
        except: return False

class SocialSentinel:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self.api_key, self.session = api_key, session
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self.api_key: return {"score": 50}
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
        for attempt in range(3):
            try:
                url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
                async with self.session.get(url, timeout=10) as resp:
                    if resp.status != 200: continue
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
                if attempt == 2: logger.error(f"Dex Error: {e}")
                await asyncio.sleep(1)
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
        if not self.cfg.trade.enabled:
            logger.info(f"🚫 [READ-ONLY] {sig['symbol']} Signal Detected.")
            return
        
        pos_size = self.cfg.trade.max_position_size_usd
        if sig.get('priority') == 2:
            pos_size = pos_size * 1.5 
            
        logger.info(f"📣 [EXECUTION] {sig['market_type']} | {sig['symbol']} | {sig['signal']} @ {sig['entry']} | Size: ${pos_size}")

class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.dex, self.security = DexEngine(session), SecurityEngine()
        self.hunter = DiscoveryHunter(HELIUS_API_KEY, session, cfg)
        self.executor = TradeExecutor(cfg)
        self.active_monitors_data: Dict[str, float] = {}
        self.cooldown_cache = {}
        self.prev_oi = {}
        self.ml_model = self._load_model()

    def _load_model(self):
        path = self.cfg.ml_model_path
        try:
            if os.path.exists(path) and os.path.getsize(path) > 100:
                if path.endswith('.h5') and HAS_TF: return load_keras_model(path)
                with open(path, 'rb') as f: return pickle.load(f)
        except: pass
        return None

    def predict_confidence(self, features: list) -> float:
        if self.ml_model:
            try:
                if self.cfg.ml_model_path.endswith('.h5'):
                    pred = self.ml_model.predict(np.array([features]), verbose=0)
                    return float(pred[0][0] * 100)
                return float(self.ml_model.predict_proba([features])[0][1] * 100)
            except: return 50.0
        return 70.0

    async def get_market_sentiment_index(self) -> str:
        try:
            assets = {"BTC/USDT:USDT": 0.6, "ETH/USDT:USDT": 0.2, "SOL/USDT:USDT": 0.2}
            score = 0
            for symbol, weight in assets.items():
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=50)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                ema = df['c'].ewm(span=20).mean().iloc[-1]
                if df['c'].iloc[-1] > ema:
                    score += weight
            if score >= 0.8: return "BULLISH"
            if score <= 0.2: return "BEARISH"
            return "NEUTRAL"
        except: return "NEUTRAL"

    async def analyze_funding_squeeze(self, symbol: str):
        try:
            f_data = await self.exchange.fetch_funding_rate(symbol)
            oi_data = await self.exchange.fetch_open_interest(symbol)
            funding, oi = float(f_data.get('fundingRate', 0.0)), float(oi_data.get('openInterestAmount') or 0.0)
            last_oi = self.prev_oi.get(symbol, 0)
            oi_growth = (oi > last_oi * 1.05) if last_oi > 0 else False
            self.prev_oi[symbol] = oi
            return funding, oi, (funding < -0.01 and oi_growth)
        except: return 0.0, 0.0, False

    async def generate_cex_signal(self, symbol: str):
        logger.info(f"🔍 CEX Scan: Checking {symbol} for signal conditions...")
        
        if self.cooldown_cache.get(symbol) and (datetime.now() - self.cooldown_cache[symbol]) < timedelta(minutes=self.cfg.trade.signal_cooldown_minutes):
            return
        async with self.store.get_symbol_lock(symbol):
            try:
                market_bias = await self.get_market_sentiment_index()
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                df['adx'] = ta.trend.ADXIndicator(df['h'], df['l'], df['c']).adx()
                if df['adx'].iloc[-1] < self.cfg.indicators.adx_threshold: return
                df['ema_s'], df['ema_m'] = df['c'].ewm(span=self.cfg.indicators.ema_short).mean(), df['c'].ewm(span=self.cfg.indicators.ema_medium).mean()
                df['atr'] = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c']).average_true_range()
                last = df.iloc[-1]
                stype = "BUY" if last['ema_s'] > last['ema_m'] else "SELL" if last['ema_s'] < last['ema_m'] else None

                if stype and not await self.store.has_open_signal(symbol):
                    funding, oi, is_squeeze = await self.analyze_funding_squeeze(symbol)
                    if stype == "BUY" and market_bias == "BEARISH" and not is_squeeze: return
                    if stype == "SELL" and market_bias == "BULLISH": return
                    social = await self.sentinel.get_sentiment(symbol)
                    entry_price = float(last['c'])
                    sl = entry_price - (last['atr'] * self.cfg.indicators.atr_sl_mult) if stype == "BUY" else entry_price + (last['atr'] * self.cfg.indicators.atr_sl_mult)
                    tp = entry_price + (last['atr'] * self.cfg.indicators.atr_tp_mult) if stype == "BUY" else entry_price - (last['atr'] * self.cfg.indicators.atr_tp_mult)
                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "signal": stype,
                        "market_type": "CEX", "entry": entry_price, "sl": round(sl, 6), "tp": round(tp, 6),
                        "confidence": self.predict_confidence([funding, social['score']]),
                        "sentiment_score": social['score'], "funding": funding, "open_interest": oi,
                        "model_version": self.cfg.model_version, "vol_liq_ratio": 0.0, "priority": 0
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

# PRODUCTION FIX: Background initialization to prevent Port Scan Timeout
@app.on_event("startup")
async def startup():
    global session, generator
    # 1. Faster startup: Initialize Database schema only
    await store.init_db()
    
    # 2. Run heavy loading (Exchanges, RPCs, Logic) in a non-blocking task
    # This lets Uvicorn open the port IMMEDIATELY so Render sees it.
    asyncio.create_task(run_background_initialization())
    
    logger.info(f"🚀 QuikPulse {cfg.model_version} Port Listener Started.")

async def run_background_initialization():
    global session, generator
    try:
        logger.info("📡 Starting Background Service Initialization...")
        
        # Load monitored pairs from DB
        async with store.async_session() as session_db:
            result = await session_db.execute(select(MonitoredPair.symbol))
            db_pairs = result.scalars().all()
            for p in db_pairs:
                if p not in cfg.symbols:
                    cfg.symbols.append(p)

        session = aiohttp.ClientSession()
        sentinel = SocialSentinel(SANTIMENT_API_KEY, session)
        
        logger.info("⚙️ Mounting Signal Generator Logic...")
        generator = SignalGenerator(cfg, store, exchange, sentinel, cluster_map, session)

        # Telegram Webhook Setup
        public_url = os.getenv("RENDER_EXTERNAL_URL")
        if public_url and cfg.telegram_bot_token:
            webhook_url = f"{public_url}/tg-webhook"
            setup_url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/setWebhook?url={webhook_url}"
            async with session.get(setup_url) as resp:
                logger.info(f"Telegram Webhook Status: {await resp.json()}")

        # Start background loops
        logger.info("📈 Launching CEX Polling Monitor...")
        background_tasks.add(asyncio.create_task(background_monitor()))
        
        logger.info("🎯 Launching DEX Watcher (Position Tracking)...")
        background_tasks.add(asyncio.create_task(centralized_dex_watcher()))
        
        logger.info("🧬 Launching Wallet Refresh Loop...")
        background_tasks.add(asyncio.create_task(wallet_refresh_loop()))
        
        logger.info("🕵️ Launching Hunter Audit Engine (DEX Sniper Logic)...")
        background_tasks.add(asyncio.create_task(hunting_audit_loop()))
        
        logger.info("✅ All background hunting engines are now LIVE and scanning.")
    except Exception as e:
        logger.error(f"CRITICAL: Background initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    for t in background_tasks: t.cancel()
    if session: await session.close()
    await exchange.close()

@app.post("/webhook")
async def combined_webhook_handler(request: Request):
    # Added heartbeat for incoming webhooks
    logger.info("📡 Webhook Event Received: Incoming DEX Signal Data...")
    try:
        data = await request.json()
        db_wallets = store.wallet_cache
        if isinstance(data, dict) and "event" in data:
            for act in data.get("event", {}).get("activity", []):
                buyer = act.get("fromAddress")
                mint = act.get("rawContract", {}).get("address")
                if not mint or not buyer: continue
                if await store.has_open_signal(mint): continue
                dex_info = await generator.dex.get_price_data(mint)
                await store.insert_candidate(buyer, mint, dex_info['price'])
                is_whale = await generator.hunter.is_whale_funded(buyer)
                is_sniper = buyer in db_wallets
                if is_sniper or is_whale:
                    await process_dex_signal(mint, buyer, is_whale, is_sniper)
        elif isinstance(data, list):
            for event in data:
                if event.get("type") != "SWAP": continue
                swap = event.get("events", {}).get("swap", {})
                mint, buyer = swap.get("tokenOutMint"), event.get("feePayer")
                if await store.has_open_signal(mint): continue
                is_whale, is_sniper = await generator.hunter.is_whale_funded(buyer), buyer in db_wallets
                if is_sniper or is_whale:
                    await process_dex_signal(mint, buyer, is_whale, is_sniper)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook Failure: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

async def process_dex_signal(mint: str, buyer: str, is_whale: bool, is_sniper: bool):
    priority_level = 0
    buyer_label = store.wallet_cache.get(buyer, "")
    if "Expert-Hunter" in buyer_label: priority_level = 2
    elif is_sniper or is_whale: priority_level = 1

    dex_data = await generator.dex.get_price_data(mint)
    if dex_data['price'] <= 0: return
    safety = await generator.security.get_safety_report(mint, dex_data['vol24'], dex_data['liq'])
    sig = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": dex_data['symbol'],
        "market_type": "DEX", "contract_address": mint, "signal": "BUY", "entry": dex_data['price'],
        "confidence": 98.0 if priority_level == 2 else 95.0, 
        "model_version": cfg.model_version, "vol_liq_ratio": safety['vl_ratio'],
        "safety_score": safety['safety_score'], "priority": priority_level
    }
    await store.insert_signal(sig)
    await generator.executor.execute_trade(sig)
    await notify_new_signal(sig, session, cfg, is_whale=is_whale, is_sniper=is_sniper, priority=priority_level)
    generator.active_monitors_data[mint] = float(dex_data['price'])

@app.post("/tg-webhook")
async def telegram_command_handler(request: Request):
    try:
        data = await request.json()
        msg = data.get("message", {})
        text = msg.get("text", "").strip()
        parts = text.split()
        if not parts: return JSONResponse({"status": "ok"})
        cmd = parts[0].lower()

        if cmd == "/status":
            cex_txt = ", ".join([f"`{s}`" for s in cfg.symbols]) if cfg.symbols else "None"
            res = (
                f"📊 **QuikPulse Dashboard**\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🤖 **Status:** `LIVE` 🟢\n"
                f"📈 **CEX Pairs:** {cex_txt}\n"
                f"🎯 **DEX Active:** `{len(generator.active_monitors_data)}` tokens\n"
                f"🧬 **Wallets:** `{len(store.wallet_cache)}` tracked\n"
                f"━━━━━━━━━━━━━━━"
            )
            await send_direct_tg(res)
        elif cmd == "/list":
            wallets = store.wallet_cache
            if not wallets: await send_direct_tg("📭 No insiders currently tracked.")
            else:
                msg = "🎯 **Tracked Insiders**\n\n"
                for i, (addr, label) in enumerate(wallets.items(), 1):
                    msg += f"{i}. `{addr}`\n   └ Label: *{label}*\n"
                await send_direct_tg(msg)
        elif cmd == "/hunt":
            async with store.async_session() as session_db:
                total_c = await session_db.execute(select(func.count(WalletCandidate.id)))
                wins = await session_db.execute(select(func.count(WalletCandidate.id)).where(WalletCandidate.is_win == 1))
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                stmt = select(WalletCandidate).where(WalletCandidate.timestamp >= cutoff).limit(20)
                res = await session_db.execute(stmt)
                candidates = res.scalars().all()
                leaderboard = []
                for c in candidates:
                    curr = await generator.dex.get_price_data(c.token_mint)
                    if curr['price'] > 0 and c.entry_price > 0:
                        roi = ((curr['price'] - c.entry_price) / c.entry_price) * 100
                        leaderboard.append((c.address, roi, curr['symbol']))
                leaderboard.sort(key=lambda x: x[1], reverse=True)
                top_3 = leaderboard[:3]
                msg = (f"🧬 **Hunter Audit Status**\n━━━━━━━━━━━━━━━\n🕵️ Candidates: `{total_c.scalar()}`\n🏆 Confirmed Wins: `{wins.scalar()}`\n\n🔥 **Top 24h Performers:**\n")
                if not top_3: msg += "_No active performers found._"
                else:
                    for i, (addr, roi, sym) in enumerate(top_3, 1):
                        msg += f"{i}. `{addr[:6]}...` | *{sym}* | **+{roi:.1f}%**\n"
                msg += f"\n━━━━━━━━━━━━━━━\n⏱️ Window: `24h` | Mult: `{cfg.min_hunter_profit_mult}x`"
                await send_direct_tg(msg)
        elif cmd == "/pair":
            if len(parts) > 1:
                pair = parts[1].upper()
                await exchange.load_markets()
                if pair in exchange.markets:
                    async with store.async_session() as session_db:
                        await session_db.merge(MonitoredPair(symbol=pair))
                        await session_db.commit()
                        if pair not in cfg.symbols: cfg.symbols.append(pair)
                        await send_direct_tg(f"✅ CEX Pair `{pair}` enabled.")
                else: await send_direct_tg(f"❌ `{pair}` not found.")
        elif cmd == "/addwallet":
            if len(parts) > 1:
                await store.add_tracked_wallet(parts[1], "Manual")
                await send_direct_tg(f"✅ Now tracking: `{parts[1]}`")
        elif cmd == "/remwallet":
            if len(parts) > 1:
                await store.remove_tracked_wallet(parts[1])
                await send_direct_tg(f"❌ Stopped tracking: `{parts[1]}`")
        elif cmd == "/resume":
            cfg.trade.enabled = True
            await send_direct_tg("🚀 Trading Engine **RESUMED**")
        elif cmd == "/pause":
            cfg.trade.enabled = False
            await send_direct_tg("🛑 Trading Engine **PAUSED** (Read-Only)")
        elif cmd == "/help":
            help_text = ("📖 **QuikPulse Guide**\n/status - System health\n/list - Show tracked wallets\n/hunt - ROI Leaderboard & Audit stats\n/pair [SYMBOL] - Monitor CEX pair\n/addwallet [ADDR] - Track DEX address\n/resume - Start auto-trading\n/pause - Disable auto-trading")
            await send_direct_tg(help_text)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"TG Error: {e}")
        return JSONResponse({"status": "error"})

async def hunting_audit_loop():
    while True:
        try:
            await asyncio.sleep(3600)
            async with store.async_session() as session_db:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                q = select(WalletCandidate).where(WalletCandidate.timestamp >= cutoff).where(WalletCandidate.is_win == 0)
                res = await session_db.execute(q)
                candidates = res.scalars().all()
                for c in candidates:
                    try:
                        curr = await generator.dex.get_price_data(c.token_mint)
                        if curr['price'] >= (c.entry_price * cfg.min_hunter_profit_mult):
                            c.is_win = 1
                            logger.info(f"Hunter: Candidate {c.address[:6]} hit profit target.")
                    except: continue
                await session_db.commit()
                consistency_query = (select(WalletCandidate.address, func.count(WalletCandidate.id)).where(WalletCandidate.is_win == 1).group_by(WalletCandidate.address).having(func.count(WalletCandidate.id) >= cfg.min_hunter_wins_required))
                winners = await session_db.execute(consistency_query)
                for addr, win_count in winners.all():
                    if addr not in store.wallet_cache:
                        await store.add_tracked_wallet(addr, label=f"Expert-Hunter-{win_count}W")
                        await send_direct_tg(f"🧬 **PRO-INSIDER HUNTED**\nWallet `{addr[:6]}` added.")
                cleanup_cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
                await session_db.execute(delete(WalletCandidate).where(WalletCandidate.timestamp < cleanup_cutoff))
                await session_db.commit()
        except Exception as e: logger.error(f"Hunter Loop Error: {e}")

async def centralized_dex_watcher():
    while True:
        try:
            for mint, entry in list(generator.active_monitors_data.items()):
                data = await generator.dex.get_price_data(mint)
                if data['price'] >= entry * 1.5:
                    await send_direct_tg(f"💰 **DEX TP (+50%)**\nToken: `{data['symbol']}`")
                    generator.active_monitors_data.pop(mint, None)
                elif data['price'] <= entry * 0.8:
                    await send_direct_tg(f"⚠️ **DEX SL (-20%)**\nToken: `{data['symbol']}`")
                    generator.active_monitors_data.pop(mint, None)
                await asyncio.sleep(2)
            await asyncio.sleep(60)
        except: await asyncio.sleep(60)

async def background_monitor():
    while True:
        try:
            logger.info(f"🔄 Starting CEX Polling Cycle for {len(cfg.symbols)} pairs...")
            for s in list(cfg.symbols):
                await generator.generate_cex_signal(s)
                await asyncio.sleep(5)
            await asyncio.sleep(cfg.poll_interval)
        except: await asyncio.sleep(60)

async def wallet_refresh_loop():
    while True:
        await asyncio.sleep(1800)
        await store.refresh_wallet_cache()

async def notify_new_signal(sig, session, cfg, is_whale=False, is_sniper=False, is_squeeze=False, priority=0):
    if priority == 2: prefix = "⚡ *EXPERT SNIPE*"
    elif is_squeeze: prefix = "🚨 *SQUEEZE*"
    elif is_sniper: prefix = "🎯 *SNIPER*"
    elif is_whale: prefix = "🐋 *WHALE*"
    else: prefix = "🚀 *SIGNAL*"
    msg = (f"{prefix}\nPair: `{sig['symbol']}`\nAction: {sig['signal']}\nEntry: `${sig['entry']}`")
    if sig.get("sl"): msg += f"\nSL: `${sig['sl']}`"
    if sig.get("tp"): msg += f"\nTP: `${sig['tp']}`"
    await send_direct_tg(msg)

async def send_direct_tg(text: str):
    if not session or not cfg.telegram_bot_token: return
    try:
        url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
        await asyncio.sleep(0.05)
        await request_with_retry(session, "POST", url, json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True})
    except: pass

@app.get("/health")
async def health():
    return {
        "status": "online",
        "bot_version": cfg.model_version,
        "uptime_snapshot": str(datetime.now(timezone.utc)),
        "cex_active": len(cfg.symbols),
        "dex_active": len(generator.active_monitors_data) if generator else 0,
        "db_engine": "ready" if store.engine else "not_initialized"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
