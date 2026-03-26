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
from sqlalchemy import Column, Integer, String, Float, Text, select, delete

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
    triggering_wallet = Column(String, nullable=True)

class TrackedWallet(Base):
    __tablename__ = "tracked_wallets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=True)
    hits = Column(Integer, default=0)
    misses = Column(Integer, default=0)
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
ALCHEMY_RPC_URL = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
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
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/lstm_model.h5")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = "v6.3-pro-resilient-alpha"

# -----------------------------
# API Resilience
# -----------------------------
async def request_with_retry(session: aiohttp.ClientSession, method: str, url: str, retries: int = 3, **kwargs):
    for i in range(retries):
        try:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 429:
                    wait = (i + 1) * 5
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

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            new_sig = SignalModel(**s)
            session.add(new_sig)
            await session.commit()

    async def update_wallet_score(self, wallet_address: str, success: bool):
        if not wallet_address: return
        async with self.async_session() as session:
            try:
                stmt = select(TrackedWallet).where(TrackedWallet.address == wallet_address)
                result = await session.execute(stmt)
                wallet = result.scalars().first()
                if wallet:
                    if success: wallet.hits += 1
                    else: wallet.misses += 1
                    
                    total = wallet.hits + wallet.misses
                    win_rate = (wallet.hits / total) * 100
                    if total >= 5 and win_rate < 30.0:
                        logger.warning(f"🗑️ PRUNING: Removing {wallet_address} ({win_rate:.1f}% Win Rate)")
                        await session.delete(wallet)
                        await send_direct_tg(f"🗑️ **Auto-Pruned**\nWallet `{wallet_address[:8]}` removed (Win Rate: {win_rate:.1f}%)")
                    await session.commit()
                    await self.refresh_wallet_cache()
            except Exception as e:
                logger.error(f"Score Update Error: {e}")
                await session.rollback()

    async def add_tracked_wallet(self, address: str, label: str):
        async with self.async_session() as session:
            try:
                new_w = TrackedWallet(address=address, label=label)
                session.add(new_w)
                await session.commit()
                await self.refresh_wallet_cache()
                return True
            except:
                await session.rollback()
                return False

    async def get_latest_signals(self, limit: int = 25):
        async with self.async_session() as session:
            result = await session.execute(select(SignalModel).order_by(SignalModel.id.desc()).limit(limit))
            rows = result.scalars().all()
            return [{"id": r.id, "symbol": r.symbol, "signal": r.signal, "entry": r.entry, "status": r.status} for r in rows]

    async def has_open_signal(self, iden: str):
        async with self.async_session() as session:
            q = select(SignalModel).where((SignalModel.symbol == iden) | (SignalModel.contract_address == iden)).where(SignalModel.status == 'open')
            result = await session.execute(q)
            return result.scalars().first() is not None

    async def get_all_tracked_wallets_detailed(self) -> Dict[str, str]:
        async with self.async_session() as session:
            res = await session.execute(select(TrackedWallet.address, TrackedWallet.label))
            return {str(row[0]): (str(row[1]) if row[1] else str(row[0])[:6]) for row in res.all()}

    def get_symbol_lock(self, s):
        if s not in self.symbol_locks: self.symbol_locks[s] = asyncio.Lock()
        return self.symbol_locks[s]

# -----------------------------
# Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, session: aiohttp.ClientSession, cfg: BotConfig):
        self.session, self.cfg = session, cfg

    async def get_wallet_performance(self, wallet: str) -> bool:
        """Verifies if a wallet is an active alpha trader. Falls back to Alchemy if Helius is out of credits."""
        if not HELIUS_API_KEY: return True
        try:
            url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions?api-key={HELIUS_API_KEY}"
            async with self.session.get(url) as resp:
                if resp.status == 429:
                    logger.warning(f"Helius credits exhausted. Falling back to Alchemy for wallet audit: {wallet[:8]}")
                    return await self.is_whale_funded(wallet) or True
                txs = await resp.json()
                return len(txs) > 5 
        except: return True

    async def is_whale_funded(self, wallet: str) -> bool:
        if not ALCHEMY_API_KEY: return False
        try:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet]}
            data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=payload)
            return data.get("result", {}).get("value", 0) > 50_000_000_000 # 50 SOL
        except: return False

    async def find_insiders_for_token(self, mint: str) -> List[str]:
        try:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress", "params": [mint, {"limit": 100}]}
            data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=payload)
            signatures = data.get("result", [])
            if not signatures: return []
            
            oldest_sig = signatures[-1]['signature']
            tx_payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction", "params": [oldest_sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}]}
            tx_data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=tx_payload)
            
            buyer = tx_data.get("result", {}).get("transaction", {}).get("message", {}).get("accountKeys", [None])[0]
            if buyer and await self.get_wallet_performance(buyer):
                return [buyer]
            return []
        except Exception as e:
            logger.error(f"Insider search failed: {e}")
            return []

class SocialSentinel:
    def __init__(self, api_key: str):
        self.api_key = api_key
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self.api_key: return {"score": 50}
        try:
            slug = symbol.split('/')[0].lower()
            data = await asyncio.to_thread(san.get, "sentiment_balance_per_asset", slug=slug, from_date="now-1d", to_date="now")
            return {"score": 50 if data.empty else int(((data.iloc[-1][0] + 5) / 10) * 100)}
        except: return {"score": 50}

class DexEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    async def get_price_data(self, address: str) -> Dict[str, Any]:
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
            async with self.session.get(url) as resp:
                data = await resp.json()
                pair = data.get('pairs', [{}])[0]
                return {
                    "price": float(pair.get('priceUsd', 0)), "symbol": pair.get('baseToken', {}).get('symbol', 'UNK'),
                    "vol24": float(pair.get('volume', {}).get('h24', 0)), "liq": float(pair.get('liquidity', {}).get('usd', 0))
                }
        except: return {"price": 0, "symbol": "UNK", "vol24": 0, "liq": 0}

    async def get_top_gainers(self) -> List[str]:
        try:
            async with self.session.get("https://api.dexscreener.com/token-boosts/top/v1") as resp:
                data = await resp.json()
                return [t.get('tokenAddress') for t in data[:5] if t.get('tokenAddress')]
        except: return []

# -----------------------------
# Signal Engine
# -----------------------------
class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.session = cfg, store, ex, sentinel, session
        self.dex, self.hunter = DexEngine(session), DiscoveryHunter(session, cfg)
        self.active_monitors: Dict[str, Dict] = {} 
        self.cooldowns = {}
        self.ml_model = self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.cfg.ml_model_path) and HAS_TF: return load_keras_model(self.cfg.ml_model_path)
        except: pass
        return None

    async def generate_cex_signal(self, symbol: str):
        if self.cooldowns.get(symbol) and datetime.now() < self.cooldowns[symbol]: return
        async with self.store.get_symbol_lock(symbol):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=50)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                ema_s = df['c'].ewm(span=self.cfg.indicators.ema_short).mean().iloc[-1]
                ema_m = df['c'].ewm(span=self.cfg.indicators.ema_medium).mean().iloc[-1]
                
                if (ema_s > ema_m) and not await self.store.has_open_signal(symbol):
                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "signal": "BUY",
                        "market_type": "CEX", "entry": float(df['c'].iloc[-1]), "confidence": 75.0,
                        "model_version": self.cfg.model_version, "status": "open"
                    }
                    await self.store.insert_signal(sig)
                    await notify_new_signal(sig, self.session, self.cfg)
                    self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cfg.trade.signal_cooldown_minutes)
            except: pass

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
cfg = BotConfig()
store = SignalStore(DATABASE_URL)
exchange = ccxt.kucoinfutures({"enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None
background_tasks = set()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "bot_status": "ONLINE", "version": cfg.model_version})

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if isinstance(data, list): # Helius Swap
        for event in data:
            if event.get("type") != "SWAP": continue
            mint, buyer = event["events"]["swap"]["tokenOutMint"], event["feePayer"]
            if (buyer in store.wallet_cache or await generator.hunter.is_whale_funded(buyer)) and not await store.has_open_signal(mint):
                dex_data = await generator.dex.get_price_data(mint)
                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": dex_data['symbol'],
                    "market_type": "DEX", "contract_address": mint, "signal": "BUY",
                    "entry": dex_data['price'], "confidence": 90.0, "model_version": cfg.model_version,
                    "triggering_wallet": buyer, "status": "open"
                }
                await store.insert_signal(sig)
                generator.active_monitors[mint] = {"entry": dex_data['price'], "wallet": buyer}
                await notify_new_signal(sig, session, cfg, is_sniper=(buyer in store.wallet_cache))
    return {"status": "success"}

@app.post("/tg-webhook")
async def tg_webhook(request: Request):
    data = await request.json()
    text = data.get("message", {}).get("text", "").strip().split()
    if not text: return {"status": "ignored"}
    cmd = text[0].lower()
    
    if cmd == "/status":
        await send_direct_tg(f"📊 **QuikPulse**\nActive DEX Monitors: `{len(generator.active_monitors)}`")
    elif cmd == "/leaderboard":
        async with store.async_session() as s:
            query = select(TrackedWallet).where((TrackedWallet.hits + TrackedWallet.misses) > 0).order_by(((TrackedWallet.hits * 1.0) / (TrackedWallet.hits + TrackedWallet.misses)).desc())
            res = await s.execute(query)
            wallets = res.scalars().all()
            if not wallets: await send_direct_tg("Leaderboard is empty.")
            else:
                msg = "🏆 **Alpha Leaderboard**\n"
                for i, w in enumerate(wallets[:10], 1):
                    msg += f"{i}. `{w.label or w.address[:6]}`: {((w.hits/(w.hits+w.misses))*100):.1f}% WR\n"
                await send_direct_tg(msg)
    elif cmd == "/discover" and len(text) > 1:
        mint = text[1]
        await send_direct_tg(f"🕵️ Tracing `{mint[:8]}`...")
        insiders = await generator.hunter.find_insiders_for_token(mint)
        for w in insiders: await store.add_tracked_wallet(w, f"Alpha-{mint[:4]}")
        await send_direct_tg(f"✅ Found and verified `{len(insiders)}` alpha wallet(s).")
    return {"status": "success"}

async def centralized_dex_watcher():
    while True:
        try:
            mints = list(generator.active_monitors.items())
            for mint, meta in mints:
                data = await generator.dex.get_price_data(mint)
                if data['price'] >= meta['entry'] * 1.5:
                    await store.update_wallet_score(meta['wallet'], True)
                    await send_direct_tg(f"💰 **TP HIT**\n`{data['symbol']}` +50%\nSource: `{meta['wallet'][:6]}`")
                    generator.active_monitors.pop(mint)
                elif data['price'] <= meta['entry'] * 0.8:
                    await store.update_wallet_score(meta['wallet'], False)
                    await send_direct_tg(f"⚠️ **SL HIT**\n`{data['symbol']}` -20%")
                    generator.active_monitors.pop(mint)
                await asyncio.sleep(1)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Watcher Error: {e}")
            await asyncio.sleep(60)

async def insider_loop():
    while True:
        try:
            mints = await generator.dex.get_top_gainers()
            for m in mints:
                found = await generator.hunter.find_insiders_for_token(m)
                for w in found:
                    if w not in store.wallet_cache: await store.add_tracked_wallet(w, f"AutoAlpha-{m[:4]}")
                await asyncio.sleep(5)
            await asyncio.sleep(3600)
        except: await asyncio.sleep(300)

@app.on_event("startup")
async def startup():
    global session, generator
    await store.init_db()
    session = aiohttp.ClientSession()
    sentinel = SocialSentinel(SANTIMENT_API_KEY)
    generator = SignalGenerator(cfg, store, exchange, sentinel, session)
    background_tasks.add(asyncio.create_task(centralized_dex_watcher()))
    background_tasks.add(asyncio.create_task(insider_loop()))
    logger.info("🚀 QuikPulse Active.")

@app.on_event("shutdown")
async def shutdown():
    for t in background_tasks: t.cancel()
    if session: await session.close()
    await exchange.close()

async def notify_new_signal(sig, session, cfg, is_sniper=False):
    if not cfg.telegram_bot_token: return
    icon = "🎯" if is_sniper else "🚀"
    msg = f"{icon} **NEW SIGNAL**\nPair: `{sig['symbol']}`\nEntry: `${sig['entry']}`"
    await send_direct_tg(msg)

async def send_direct_tg(text: str):
    if not session or not cfg.telegram_bot_token: return
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    await session.post(url, json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"})

if __name__ == "__main__":
    uvicorn.run("signalbot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
