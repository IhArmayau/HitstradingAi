from __future__ import annotations
import asyncio
import logging
import os
import sys
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, delete, func, Index

# -----------------------------
# Logging Configuration
# -----------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
AUDIT_LOG_FILE = "quikpulse_dex_audit.log"

file_handler = logging.FileHandler(AUDIT_LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuikPulseDEX")
logger.addHandler(file_handler)

# -----------------------------
# Configuration Validation
# -----------------------------
REQUIRED_ENV_VARS = ["DATABASE_URL", "ALCHEMY_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.critical(f"❌ STARTUP FAILURE: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "")
ALCHEMY_AUTH_TOKEN = os.getenv("ALCHEMY_AUTH_TOKEN", "")
ALCHEMY_WEBHOOK_ID = os.getenv("ALCHEMY_WEBHOOK_ID", "")
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
ALCHEMY_RPC_URL = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

@dataclass
class TradeConfig:
    enabled: bool = os.getenv("TRADE_ENABLED", "true").lower() == "true"
    max_position_size_usd: float = float(os.getenv("MAX_POS_SIZE", 50.0))
    min_safety_score: float = float(os.getenv("MIN_SAFETY_SCORE", 70.0))
    min_liquidity_usd: float = float(os.getenv("MIN_LIQUIDITY_USD", 10000.0))
    tp_percentage: float = float(os.getenv("DEX_TP_PERCENT", 50.0))
    sl_percentage: float = float(os.getenv("DEX_SL_PERCENT", 20.0))

@dataclass
class BotConfig:
    poll_interval: int = 60
    cluster_window_minutes: int = 30
    min_insider_buy_sol: float = float(os.getenv("MIN_INSIDER_BUY_SOL", 2.0))
    min_hunter_profit_mult: float = 3.0
    min_hunter_wins_required: int = 3
    trade: TradeConfig = field(default_factory=TradeConfig)
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    solana_wallet_address: Optional[str] = os.getenv("SOLANA_WALLET_ADDRESS")
    model_version: str = "v8.3.0-enterprise-dex"

# -----------------------------
# Database Setup & Models
# -----------------------------
Base = declarative_base()

class SignalModel(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, index=True)
    symbol = Column(String, index=True)
    market_type = Column(String, default="DEX")
    contract_address = Column(String, nullable=True, index=True)
    signal = Column(String, default="BUY")
    entry = Column(Float)
    confidence = Column(Float)
    safety_score = Column(Float, default=0.0)
    status = Column(String, default='open', index=True)
    model_version = Column(String)
    vol_liq_ratio = Column(Float, default=0.0)
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
    timestamp = Column(String, default=lambda: datetime.now(timezone.utc).isoformat(), index=True)
    is_win = Column(Integer, default=0)

Index("idx_wallet_cand_lookup", WalletCandidate.address, WalletCandidate.token_mint)

class BotSetting(Base):
    __tablename__ = "bot_settings"
    key = Column(String, primary_key=True)
    value = Column(Text)

GLOBAL_TIMEOUT = aiohttp.ClientTimeout(total=5.0)

async def request_with_retry(session: aiohttp.ClientSession, method: str, url: str, retries: int = 3, **kwargs):
    kwargs["timeout"] = kwargs.get("timeout", GLOBAL_TIMEOUT)
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

class PriceCache:
    def __init__(self, ttl_seconds: float = 3.0):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._lock = asyncio.Lock()

    async def get(self, address: str) -> Optional[dict]:
        async with self._lock:
            if address in self._cache:
                timestamp, data = self._cache[address]
                if time.time() - timestamp < self.ttl: return data
                del self._cache[address]
            return None

    async def set(self, address: str, data: dict):
        async with self._lock: self._cache[address] = (time.time(), data)

class SignalStore:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.wallet_cache = {}

    async def init_db(self):
        async with self.engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
        await self.refresh_wallet_cache()

    async def refresh_wallet_cache(self):
        try: self.wallet_cache = await self.get_all_tracked_wallets_detailed()
        except Exception: pass

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            try:
                session.add(SignalModel(**s))
                await session.commit()
            except Exception: await session.rollback()

    async def add_tracked_wallet(self, address: str, label: str = "Manual"):
        async with self.async_session() as session:
            try:
                await session.merge(TrackedWallet(address=address, label=label))
                await session.commit()
                await self.refresh_wallet_cache()
                await sync_alchemy_webhook(list(self.wallet_cache.keys()))
            except Exception: await session.rollback()

    async def remove_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            try:
                await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
                await session.commit()
                await self.refresh_wallet_cache()
                await sync_alchemy_webhook(list(self.wallet_cache.keys()))
            except Exception: await session.rollback()

    async def insert_candidate(self, address: str, mint: str, price: float):
        async with self.async_session() as session:
            try:
                session.add(WalletCandidate(address=address, token_mint=mint, entry_price=price))
                await session.commit()
            except Exception: await session.rollback()

    async def has_open_signal(self, mint: str):
        async with self.async_session() as session:
            q = select(SignalModel).where(SignalModel.contract_address == mint).where(SignalModel.status == 'open')
            res = await session.execute(q)
            return res.scalars().first() is not None

    async def get_all_tracked_wallets_detailed(self) -> Dict[str, str]:
        async with self.async_session() as session:
            res = await session.execute(select(TrackedWallet.address, TrackedWallet.label))
            return {str(row[0]): (str(row[1]) if row[1] else str(row[0])[:6]) for row in res.all()}

async def sync_alchemy_webhook(addresses: List[str]):
    if not ALCHEMY_AUTH_TOKEN or not ALCHEMY_WEBHOOK_ID: return
    url = "https://dashboard.alchemy.com/api/update-webhook-addresses"
    headers = {"X-Alchemy-Token": ALCHEMY_AUTH_TOKEN, "Content-Type": "application/json"}
    payload = {"webhook_id": ALCHEMY_WEBHOOK_ID, "addresses_to_add": addresses, "addresses_to_remove": []}
    try:
        async with aiohttp.ClientSession(timeout=GLOBAL_TIMEOUT) as s:
            await s.patch(url, json=payload, headers=headers)
    except Exception: pass

class DiscoveryHunter:
    def __init__(self, session: aiohttp.ClientSession): self.session = session
    async def is_whale_funded(self, wallet_address: str) -> bool:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_address]}
        try:
            data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=payload)
            return int(data.get("result", {}).get("value", 0)) > 50_000_000_000
        except Exception: return False

class DexEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.price_cache = PriceCache(ttl_seconds=3.0)

    async def get_price_data(self, address: str) -> Dict[str, Any]:
        cached = await self.price_cache.get(address)
        if cached: return cached
        headers = {"x-api-key": JUPITER_API_KEY} if JUPITER_API_KEY else {}
        try:
            async with self.session.get(f"https://api.jup.ag/price/v3?ids={address}", headers=headers, timeout=GLOBAL_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    token_data = data.get("data", {}).get(address)
                    if token_data and float(token_data.get("price", 0)) > 0:
                        res = {"price": float(token_data["price"]), "symbol": "UNK", "vol24": 15000.0, "liq": 15000.0}
                        await self.price_cache.set(address, res)
                        return res
        except Exception: pass
        return {"price": 0, "symbol": "UNK", "vol24": 0, "liq": 0}

class SecurityEngine:
    async def get_safety_report(self, address: str, vol_24h: float, liq: float) -> Dict[str, Any]:
        return {"safety_score": 80, "is_rugged": False, "vl_ratio": 1.0, "liquidity": liq}

class TradeExecutor:
    def __init__(self, cfg: BotConfig): self.cfg = cfg
    async def execute_trade(self, sig: dict):
        if self.cfg.trade.enabled: logger.info(f"📣 [DEX EXECUTION] BUY | {sig['symbol']} | {sig['contract_address']}")

app = FastAPI()
cfg = BotConfig()
store = SignalStore(DATABASE_URL)
active_monitors_data = {}

@app.on_event("startup")
async def startup():
    asyncio.create_task(run_bot_initialization())

async def run_bot_initialization():
    await store.init_db()

@app.post("/webhook")
async def process_solana_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"DEBUG: Webhook Payload: {str(data)[:200]}")
        
        # Robust Multi-Format Extraction
        events = []
        if isinstance(data, dict):
            events = data.get("event", {}).get("activity", []) if "event" in data else [data]
        
        for event in events:
            # Extract Mint/Buyer dynamically
            mint = event.get("rawContract", {}).get("address") or event.get("tokenOutMint")
            buyer = event.get("fromAddress") or event.get("feePayer")
            
            if not mint or not buyer: continue
            if await store.has_open_signal(mint): continue
            
            # Logic bridge
            is_whale = False 
            is_sniper = buyer in store.wallet_cache
            
            # Direct Processing
            dex_data = await dex_engine.get_price_data(mint)
            if dex_data['price'] > 0:
                await store.insert_candidate(buyer, mint, dex_data['price'])
                logger.info(f"📥 Candidate Stored: {buyer[:8]} -> {mint[:8]}")

        return JSONResponse({"status": "success"})
    except Exception: return JSONResponse({"status": "error"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
