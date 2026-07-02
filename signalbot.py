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

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
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

# Performance Composite Index for wallet-token pairing lookups
Index("idx_wallet_cand_lookup", WalletCandidate.address, WalletCandidate.token_mint)

class BotSetting(Base):
    __tablename__ = "bot_settings"
    key = Column(String, primary_key=True)
    value = Column(Text)

# -----------------------------
# API Resilience & Global Timeouts
# -----------------------------
GLOBAL_TIMEOUT = aiohttp.ClientTimeout(total=5.0)

async def request_with_retry(session: aiohttp.ClientSession, method: str, url: str, retries: int = 3, **kwargs):
    kwargs["timeout"] = kwargs.get("timeout", GLOBAL_TIMEOUT)
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
            if i == retries - 1:
                raise e
            await asyncio.sleep(2 ** i)

# -----------------------------
# High-Performance In-Memory Cache Engine
# -----------------------------
class PriceCache:
    def __init__(self, ttl_seconds: float = 3.0):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._lock = asyncio.Lock()

    async def get(self, address: str) -> Optional[dict]:
        async with self._lock:
            if address in self._cache:
                timestamp, data = self._cache[address]
                if time.time() - timestamp < self.ttl:
                    return data
                del self._cache[address]
            return None

    async def set(self, address: str, data: dict):
        async with self._lock:
            self._cache[address] = (time.time(), data)

# -----------------------------
# Database Store Middleware
# -----------------------------
class SignalStore:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.wallet_cache = {}

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self.refresh_wallet_cache()
        logger.info("DEX Database Schema Synchronized & Wallet Cache Hydrated.")

    async def refresh_wallet_cache(self):
        try:
            self.wallet_cache = await self.get_all_tracked_wallets_detailed()
        except Exception:
            logger.exception("Failed to refresh wallet cache from database records.")

    async def insert_signal(self, s: dict):
        async with self.async_session() as session:
            try:
                new_sig = SignalModel(**s)
                session.add(new_sig)
                await session.commit()
                logger.info(f"AUDIT: DEX Signal Stored: {s['symbol']} ({s['contract_address']})")
            except Exception:
                await session.rollback()
                logger.exception("Failed to insert trading signal into database.")

    async def add_tracked_wallet(self, address: str, label: str = "Manual"):
        async with self.async_session() as session:
            try:
                await session.merge(TrackedWallet(address=address, label=label))
                await session.commit()
                await self.refresh_wallet_cache()
                await sync_alchemy_webhook(list(self.wallet_cache.keys()))
            except Exception:
                await session.rollback()
                logger.exception(f"Failed to add tracked wallet: {address}")

    async def remove_tracked_wallet(self, address: str):
        async with self.async_session() as session:
            try:
                await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
                await session.commit()
                await self.refresh_wallet_cache()
                await sync_alchemy_webhook(list(self.wallet_cache.keys()))
            except Exception:
                await session.rollback()
                logger.exception(f"Failed to remove tracked wallet: {address}")

    async def insert_candidate(self, address: str, mint: str, price: float):
        async with self.async_session() as session:
            try:
                existing_q = select(WalletCandidate).where(
                    WalletCandidate.address == address,
                    WalletCandidate.token_mint == mint
                )
                existing_res = await session.execute(existing_q)
                if existing_res.scalars().first() is not None:
                    return

                session.add(WalletCandidate(address=address, token_mint=mint, entry_price=price))
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception("Failed to handle candidate insertion.")

    async def has_open_signal(self, mint: str):
        async with self.async_session() as session:
            try:
                q = select(SignalModel).where(SignalModel.contract_address == mint).where(SignalModel.status == 'open')
                result = await session.execute(q)
                return result.scalars().first() is not None
            except Exception:
                logger.exception(f"Error checking open signals for mint: {mint}")
                return False

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
        async with aiohttp.ClientSession(timeout=GLOBAL_TIMEOUT) as s:
            async with s.patch(url, json=payload, headers=headers) as resp:
                logger.info(f"Alchemy Webhook Sync Status: {resp.status}")
    except Exception:
        logger.exception("Alchemy Sync Interface Protocol Failure.")

# -----------------------------
# On-Chain Intelligence Engines
# -----------------------------
class DiscoveryHunter:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def is_whale_funded(self, wallet_address: str) -> bool:
        if not ALCHEMY_API_KEY or not wallet_address: return False
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_address]}
        try:
            data = await request_with_retry(self.session, "POST", ALCHEMY_RPC_URL, json=payload)
            balance = int(data.get("result", {}).get("value", 0))
            return balance > 50_000_000_000  # 50 SOL threshold
        except Exception:
            logger.exception(f"Error analyzing whale criteria balance status for: {wallet_address}")
            return False

class DexEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.price_cache = PriceCache(ttl_seconds=3.0)

    async def get_price_data(self, address: str) -> Dict[str, Any]:
        cached_val = await self.price_cache.get(address)
        if cached_val:
            return cached_val

        headers = {"x-api-key": JUPITER_API_KEY} if JUPITER_API_KEY else {}
        url_jupiter = f"https://api.jup.ag/price/v3?ids={address}"
        fallback_symbol = f"SOL-{address[:4]}"
        try:
            async with self.session.get(url_jupiter, headers=headers, timeout=GLOBAL_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    token_data = data.get(address)
                    if token_data and float(token_data.get("usdPrice", 0)) > 0:
                        res = {
                            "price": float(token_data.get("usdPrice")),
                            "symbol": token_data.get("extraInfo", {}).get("quotedMint", fallback_symbol),
                            "vol24": float(token_data.get("liquidity", 15000.0)),
                            "liq": float(token_data.get("liquidity", 15000.0))
                        }
                        await self.price_cache.set(address, res)
                        return res
        except Exception as e:
            logger.debug(f"Jupiter pricing lookup pass-through: {e}")

        for attempt in range(3):
            try:
                url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
                async with self.session.get(url, timeout=GLOBAL_TIMEOUT) as resp:
                    if resp.status != 200: continue
                    data = await resp.json()
                    pairs = data.get('pairs', [])
                    if pairs:
                        p = pairs[0]
                        res = {
                            "price": float(p.get('priceUsd', 0)),
                            "symbol": p.get('baseToken', {}).get('symbol', 'UNK'),
                            "vol24": float(p.get('volume', {}).get('h24', 0)),
                            "liq": float(p.get('liquidity', {}).get('usd', 0))
                        }
                        await self.price_cache.set(address, res)
                        return res
            except Exception:
                await asyncio.sleep(1)

        error_fallback = {"price": 0, "symbol": "UNK", "vol24": 0, "liq": 0}
        return error_fallback

class SecurityEngine:
    async def get_safety_report(self, address: str, vol_24h: float, liq: float, min_liq: float = 10000.0) -> Dict[str, Any]:
        vl_ratio = vol_24h / liq if liq > 0 else 0.0
        is_safe_liquidity = liq >= min_liq
        safety_score = 80 if (0 < vl_ratio < 5 and is_safe_liquidity) else 40
        return {
            "safety_score": safety_score,
            "is_rugged": vl_ratio > 10.0 or not is_safe_liquidity,
            "vl_ratio": vl_ratio,
            "liquidity": liq
        }

# -----------------------------
# DEX Execution Engine
# -----------------------------
class TradeExecutor:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
    async def execute_trade(self, sig: dict):
        if not self.cfg.trade.enabled:
            logger.info(f"🚫 [READ-ONLY] On-Chain DEX Setup Detected for {sig['symbol']}.")
            return

        pos_size = self.cfg.trade.max_position_size_usd
        if sig.get('priority') == 2:
            pos_size = pos_size * 1.5

        logger.info(f"📣 [DEX EXECUTION] BUY | {sig['symbol']} | Mint: {sig['contract_address']} | Size: ${pos_size}")

# -----------------------------
# FastAPI Webhook Router
# -----------------------------
app = FastAPI()
cfg = BotConfig()
store = SignalStore(DATABASE_URL)
session: Optional[aiohttp.ClientSession] = None
dex_engine: Optional[DexEngine] = None
security_engine = SecurityEngine()
hunter_engine: Optional[DiscoveryHunter] = None
executor: Optional[TradeExecutor] = None

active_monitors_data: Dict[str, float] = {}
background_tasks = set()

async def run_bot_initialization():
    """Asynchronously triggers database syncing and engine instantiation without blocking port listeners."""
    global session, dex_engine, hunter_engine, executor
    try:
        logger.info("📡 Running engine database migrations and workspace hydration...")
        await store.init_db()

        session = aiohttp.ClientSession(timeout=GLOBAL_TIMEOUT)
        dex_engine = DexEngine(session)
        hunter_engine = DiscoveryHunter(session)
        executor = TradeExecutor(cfg)

        public_url = os.getenv("RENDER_EXTERNAL_URL")
        if public_url and cfg.telegram_bot_token:
            webhook_url = f"{public_url}/tg-webhook"
            setup_url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/setWebhook?url={webhook_url}"
            try:
                async with session.get(setup_url) as resp:
                    logger.info(f"Telegram Webhook Status Handshake: {resp.status}")
            except Exception:
                logger.exception("Failed connecting system webhook router directly with Telegram networks.")

        background_tasks.add(asyncio.create_task(centralized_dex_watcher()))
        background_tasks.add(asyncio.create_task(hunting_audit_loop()))
        background_tasks.add(asyncio.create_task(wallet_refresh_loop()))
        logger.info(f"🚀 Pure-DEX Machine {cfg.model_version} Live on Network Infrastructure.")
    except Exception as e:
        logger.critical(f"⚠️ Core processing loop hit initialization friction but keeping web services running: {e}")

@app.on_event("startup")
async def startup():
    # Immediate execution extraction so Uvicorn opens ports instantly for Render scans
    asyncio.create_task(run_bot_initialization())

@app.on_event("shutdown")
async def shutdown():
    for t in background_tasks:
        t.cancel()
    if session:
        await session.close()

@app.get("/")
async def root():
    return {"message": "QuikPulse Pure-DEX Engine Running Successfully."}

@app.get("/health")
async def health():
    return {
        "status": "online",
        "bot_version": cfg.model_version,
        "monitored_tokens_count": len(active_monitors_data),
        "tracked_wallets_count": len(store.wallet_cache)
    }

@app.post("/webhook")
async def process_solana_webhook(request: Request):
    try:
        data = await request.json()
        db_wallets = store.wallet_cache

        if isinstance(data, dict) and "event" in data:
            for act in data.get("event", {}).get("activity", []):
                buyer = act.get("fromAddress")
                mint = act.get("rawContract", {}).get("address")

                if not mint and "log" in act:
                    mints = re.findall(r'[1-9A-HJ-NP-Za-km-z]{32,44}', str(act))
                    if mints: mint = mints[0]

                if not mint or not buyer: continue
                if await store.has_open_signal(mint): continue

                is_whale = await hunter_engine.is_whale_funded(buyer) if hunter_engine else False
                is_sniper = buyer in db_wallets

                await process_dex_signal(mint, buyer, is_whale, is_sniper)

        elif isinstance(data, list):
            for event in data:
                if event.get("type") != "SWAP": continue
                swap = event.get("events", {}).get("swap", {})
                mint = swap.get("tokenOutMint")
                buyer = event.get("feePayer")

                if not mint or not buyer: continue
                if await store.has_open_signal(mint): continue

                is_whale = await hunter_engine.is_whale_funded(buyer) if hunter_engine else False
                is_sniper = buyer in db_wallets

                await process_dex_signal(mint, buyer, is_whale, is_sniper)

        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook Ingestion Execution Failure: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

async def process_dex_signal(mint: str, buyer: str, is_whale: bool, is_sniper: bool):
    if not dex_engine or not store or not executor:
        logger.warning("Engine components are still hydrating. Skipping inbound match evaluation.")
        return

    if not is_sniper and not is_whale:
        dex_data = await dex_engine.get_price_data(mint)
        if dex_data['price'] > 0:
            await store.insert_candidate(buyer, mint, dex_data['price'])
            logger.info(f"📥 Wallet Candidate Stored: `{buyer[:8]}` tracking asset `{mint[:8]}`")
        return

    priority_level = 0
    buyer_label = store.wallet_cache.get(buyer, "")
    if "Expert-Hunter" in buyer_label: priority_level = 2
    elif is_sniper or is_whale: priority_level = 1

    dex_data = await dex_engine.get_price_data(mint)
    if dex_data['price'] <= 0: return

    safety = await security_engine.get_safety_report(mint, dex_data['vol24'], dex_data['liq'], cfg.trade.min_liquidity_usd)
    if safety['is_rugged']:
        logger.info(f"🚫 Skipped {dex_data['symbol']} due to high-risk risk checks (${dex_data['liq']} Liq)")
        return

    sig = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": dex_data['symbol'],
        "market_type": "DEX", "contract_address": mint, "signal": "BUY", "entry": dex_data['price'],
        "confidence": 98.0 if priority_level == 2 else 95.0,
        "model_version": cfg.model_version, "vol_liq_ratio": safety['vl_ratio'],
        "safety_score": safety['safety_score'], "priority": priority_level
    }
    await store.insert_signal(sig)
    await executor.execute_trade(sig)
    await notify_new_signal(sig, is_whale=is_whale, is_sniper=is_sniper, priority=priority_level)
    active_monitors_data[mint] = float(dex_data['price'])

# -----------------------------
# Active Position Watcher (TP/SL)
# -----------------------------
async def evaluation_worker(mint: str, entry: float, tp_factor: float, sl_factor: float):
    if not dex_engine: return
    data = await dex_engine.get_price_data(mint)
    if data['price'] <= 0: return

    current_price = data['price']
    if current_price >= entry * tp_factor:
        await send_direct_tg(f"💰 **DEX TAKE PROFIT (+{cfg.trade.tp_percentage}%)**\nToken: `{data['symbol']}`\nMint: `{mint}`")
        active_monitors_data.pop(mint, None)
    elif current_price <= entry * sl_factor:
        await send_direct_tg(f"⚠️ **DEX STOP LOSS (-{cfg.trade.sl_percentage}%)**\nToken: `{data['symbol']}`\nMint: `{mint}`")
        active_monitors_data.pop(mint, None)

async def centralized_dex_watcher():
    while True:
        try:
            if not active_monitors_data:
                await asyncio.sleep(2)
                continue

            tp_factor = 1 + (cfg.trade.tp_percentage / 100)
            sl_factor = 1 - (cfg.trade.sl_percentage / 100)

            execution_tasks = [
                evaluation_worker(mint, entry, tp_factor, sl_factor)
                for mint, entry in list(active_monitors_data.items())
            ]
            await asyncio.gather(*execution_tasks)
            await asyncio.sleep(1)
        except Exception:
            logger.exception("Watcher Core Thread Framework Exception caught.")
            await asyncio.sleep(5)

# -----------------------------
# Insider Grading & Hunting Diagnostics Loops
# -----------------------------
async def hunting_audit_loop():
    while True:
        try:
            await asyncio.sleep(3600)
            if not dex_engine: continue
            async with store.async_session() as session_db:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                q = select(WalletCandidate).where(WalletCandidate.timestamp >= cutoff).where(WalletCandidate.is_win == 0)
                res = await session_db.execute(q)
                candidates = res.scalars().all()

                for c in candidates:
                    curr = await dex_engine.get_price_data(c.token_mint)
                    if curr['price'] >= (c.entry_price * cfg.min_hunter_profit_mult):
                        c.is_win = 1
                await session_db.commit()

                consistency = (select(WalletCandidate.address, func.count(WalletCandidate.id))
                               .where(WalletCandidate.is_win == 1)
                               .group_by(WalletCandidate.address)
                               .having(func.count(WalletCandidate.id) >= cfg.min_hunter_wins_required))
                winners = await session_db.execute(consistency)

                for addr, win_count in winners.all():
                    if addr not in store.wallet_cache:
                        await store.add_tracked_wallet(addr, label=f"Expert-Hunter-{win_count}W")
                        await send_direct_tg(f"🧬 **PRO-INSIDER AUTOMATICALLY UNLOCKED**\nWallet `{addr[:6]}...` upgraded.")

                cleanup = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
                await session_db.execute(delete(WalletCandidate).where(WalletCandidate.timestamp < cleanup))
                await session_db.commit()
        except Exception:
            logger.exception("Hunting Audit Loop Diagnostics Task Exception.")
            await asyncio.sleep(60)

async def wallet_refresh_loop():
    while True:
        try:
            await asyncio.sleep(1800)
            await store.refresh_wallet_cache()
        except Exception:
            logger.exception("Static background wallet database lookup frame synchronization failure.")

# -----------------------------
# Telegram Messaging Interface
# -----------------------------
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
            res = (f"📊 **QuikPulse DEX Engine**\n━━━━━━━━━━━━━━━\n🤖 **Status:** `PURE DEX MODE` 🟢\n🎯 **Active Tracks:** `{len(active_monitors_data)}` tokens\n🧬 **Database Insiders:** `{len(store.wallet_cache)}` tracked\n━━━━━━━━━━━━━━━")
            await send_direct_tg(res)

        elif cmd == "/addwallet" and len(parts) > 1:
            await store.add_tracked_wallet(parts[1], "Manual")
            await send_direct_tg(f"✅ Target Locked on Wallet: `{parts[1]}`")

        elif cmd == "/remwallet" and len(parts) > 1:
            await store.remove_tracked_wallet(parts[1])
            await send_direct_tg(f"❌ Target Dropped for Wallet: `{parts[1]}`")

        elif cmd == "/listwallets":
            wallets = store.wallet_cache
            if not wallets:
                await send_direct_tg(" 📁 No wallets currently tracking inside active database storage maps.")
            else:
                lines = [f"• `{addr[:8]}...` ({label})" for addr, label in list(wallets.items())[:30]]
                await send_direct_tg(f"🧬 **Tracked Insiders (Top 30):**\n" + "\n".join(lines))

        elif cmd == "/signals":
            async with store.async_session() as db:
                q = select(SignalModel).order_by(SignalModel.id.desc()).limit(10)
                res = await db.execute(q)
                sigs = res.scalars().all()
                if not sigs:
                    await send_direct_tg("No execution tracking history signals found inside records.")
                else:
                    lines = [f"• {s.symbol} | Entry: ${s.entry} | Status: `{s.status}`" for s in sigs]
                    await send_direct_tg(f"🎯 **Recent Core Generated Signals:**\n" + "\n".join(lines))

        elif cmd == "/stats":
            async with store.async_session() as db:
                cand_count = await db.execute(select(func.count(WalletCandidate.id)))
                sig_count = await db.execute(select(func.count(SignalModel.id)))
                await send_direct_tg(
                    f"📈 **System Performance Analytics Metrics:**\n"
                    f"• Evaluated Candidates: `{cand_count.scalar()}`\n"
                    f"• Dispatched Active Signals: `{sig_count.scalar()}`\n"
                    f"• Monitoring Cache Memory Size: `{len(active_monitors_data)}` entries"
                )

        elif cmd == "/help":
            commands = [
                "/status - Check engine status",
                "/addwallet [addr] - Track specialized wallet instance",
                "/remwallet [addr] - Halt tracking instance targets",
                "/listwallets - View active watchlists",
                "/signals - Display past execution arrays",
                "/stats - Ingestion architecture database counts"
            ]
            await send_direct_tg("📖 **DEX Bot Command Control System Manual:**\n" + "\n".join(commands))

        return JSONResponse({"status": "ok"})
    except Exception:
        logger.exception("Error routing incoming Telegram command structures.")
        return JSONResponse({"status": "error"})

async def notify_new_signal(sig, is_whale=False, is_sniper=False, priority=0):
    if priority == 2: prefix = "⚡ *EXPERT METRIC SNIPE*"
    elif is_sniper: prefix = "🎯 *INSIDER WALLET BUY*"
    elif is_whale: prefix = "🐋 *WHALE INFLOW*"
    else: prefix = "🚀 *DEX SETUP*"

    msg = (f"{prefix}\nAsset: `{sig['symbol']}`\nMint: `{sig['contract_address']}`\nPrice: `${sig['entry']}`\nSafety: `{sig['safety_score']}/100`")
    await send_direct_tg(msg)

async def send_direct_tg(text: str):
    if not session or not cfg.telegram_bot_token: return
    try:
        url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
        await request_with_retry(session, "POST", url, json={
            "chat_id": cfg.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        })
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
