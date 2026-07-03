from __future__ import annotations
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, select, delete, Index

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

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql+asyncpg://", 1)
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

GLOBAL_TIMEOUT = aiohttp.ClientTimeout(total=5.0)

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

app = FastAPI()
cfg = BotConfig()
store = SignalStore(DATABASE_URL)

# Global variables for async dependencies
http_session: Optional[aiohttp.ClientSession] = None
dex_engine: Optional[DexEngine] = None

@app.on_event("startup")
async def startup():
    global http_session, dex_engine
    # Initialize async dependencies here to ensure they exist within a running event loop
    http_session = aiohttp.ClientSession()
    dex_engine = DexEngine(http_session)
    await store.init_db()

@app.on_event("shutdown")
async def shutdown():
    if http_session:
        await http_session.close()

@app.post("/webhook")
async def process_solana_webhook(request: Request):
    if not dex_engine:
        return JSONResponse({"status": "error", "message": "Engine starting"}, status_code=503)
    try:
        data = await request.json()
        transactions = data.get("event", {}).get("transaction", [])
        
        for tx in transactions:
            mint = tx.get("tokenOutMint") or tx.get("rawContract", {}).get("address")
            buyer = tx.get("fromAddress") or tx.get("feePayer")

            if not mint or not buyer: continue
            if await store.has_open_signal(mint): continue

            dex_data = await dex_engine.get_price_data(mint)
            if dex_data['price'] > 0:
                await store.insert_candidate(buyer, mint, dex_data['price'])
                logger.info(f"📥 Candidate Stored: {buyer[:8]} -> {mint[:8]}")

        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
