from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import aiosqlite
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
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import san

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "Eo6zp2wemnkb4cui_thgwsepbufktb4qz")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot_production.log")]
)
logger = logging.getLogger("QuikPulseAI")

if SANTIMENT_API_KEY:
    san.ApiConfig.api_key = SANTIMENT_API_KEY

# -----------------------------
# Configs
# -----------------------------
@dataclass
class TradeConfig:
    enabled: bool = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
    max_position_size_usd: float = float(os.getenv("MAX_POS_SIZE", 50.0))
    min_safety_score: float = float(os.getenv("MIN_SAFETY_SCORE", 70.0))
    min_sentiment_score: float = 60.0

@dataclass
class IndicatorsConfig:
    ema_short: int = int(os.getenv("EMA_SHORT", 9))
    ema_medium: int = int(os.getenv("EMA_MEDIUM", 21))
    ema_long: int = int(os.getenv("EMA_LONG", 50))
    ema_filter: int = int(os.getenv("EMA_FILTER", 200))
    adx_threshold: int = int(os.getenv("ADX_THRESHOLD", 25))
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    atr_period: int = int(os.getenv("ATR_PERIOD", 14))
    bb_period: int = int(os.getenv("BB_PERIOD", 20))
    bb_std: float = float(os.getenv("BB_STD", 2.0))
    atr_tp_mult: float = float(os.getenv("ATR_TP_MULT", 3.0)) 
    atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", 1.5)) 
    max_spread_pct: float = float(os.getenv("MAX_SPREAD_PCT", 0.15))

@dataclass
class BotConfig:
    enable_cex: bool = os.getenv("ENABLE_CEX_MONITOR", "true").lower() == "true"
    enable_dex: bool = os.getenv("ENABLE_DEX_MONITOR", "true").lower() == "true"
    enable_whale: bool = os.getenv("ENABLE_WHALE_MONITOR", "true").lower() == "true"
    symbols: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT").split(',')
    ])
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    higher_timeframe: str = os.getenv("HIGHER_TIME_FRAME", "1h")
    limit: int = int(os.getenv("LIMIT", 1000))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    dex_poll_interval: int = 60
    social_poll_interval: int = 900
    insider_poll_interval: int = 120
    cluster_window_minutes: int = 30
    min_insider_usd: float = float(os.getenv("MIN_INSIDER_USD", 5000.0))
    min_insider_buy_sol: float = float(os.getenv("MIN_INSIDER_BUY_SOL", 2.0))
    vl_ratio_max: float = 10.0
    min_liquidity_usd: float = 15000.0
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v4-prod")
    tracked_wallets: List[str] = field(default_factory=lambda: [
        w.strip() for w in os.getenv("TRACKED_WALLETS", "").split(',') if w.strip()
    ])

# -----------------------------
# Intelligence Engines
# -----------------------------

class SocialSentinel:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self.api_key, self.session = api_key, session
        if self.api_key: san.ApiConfig.api_key = self.api_key

    def _get_slug(self, symbol: str) -> str:
        slug_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
        clean = symbol.split('/')[0].split(':')[0].upper()
        return slug_map.get(clean, clean.lower())

    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self.api_key: return {"score": 50, "label": "No API Key", "funding": 0.0}
        slug = self._get_slug(symbol)
        try:
            data = await asyncio.to_thread(san.get, "sentiment_balance_per_asset", slug=slug, from_date="now-1d", to_date="now", interval="1h")
            funding = await asyncio.to_thread(san.get, "funding_rates_aggregated_by_exchange", slug=slug, from_date="now-8h", to_date="now")

            score = 50
            if not data.empty:
                val = data.iloc[-1][0]
                score = max(0, min(100, int(((val + 5) / 10) * 100)))

            f_rate = round(funding.iloc[-1][0], 5) if not funding.empty else 0.0
            label = "Bullish" if score > 60 else "Bearish" if score < 40 else "Neutral"
            return {"score": score, "label": label, "funding": f_rate}
        except Exception as e:
            logger.error(f"Santiment Error for {slug}: {e}")
            return {"score": 50, "label": "Neutral", "funding": 0.0}

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

# -----------------------------
# Core Logic Engines
# -----------------------------

class ModelTrainer:
    def __init__(self, db_path: str, model_dir: str):
        self.db_path, self.model_dir = db_path, model_dir
        if not os.path.exists(model_dir): os.makedirs(model_dir)

    async def retrain_model(self, symbol: str = "GLOBAL"):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = "SELECT entry, safety_score, sentiment_score, confidence, status FROM signals WHERE status IN ('win', 'loss')"
                if symbol != "GLOBAL": query += f" AND symbol = '{symbol}'"
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    if len(rows) < 20: return
                    df = pd.DataFrame(rows, columns=['entry', 'safety_score', 'sentiment_score', 'confidence', 'status'])
            X, y = df[['entry', 'safety_score', 'sentiment_score', 'confidence']], df['status'].apply(lambda x: 1 if x == 'win' else 0)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_model.joblib"))
        except Exception as e: logger.error(f"Trainer Error: {e}")

class ClusterEngine:
    def __init__(self, window_mins: int):
        self.window = timedelta(minutes=window_mins)
        self.history: Dict[str, List[datetime]] = {}
    def record_and_check(self, mint: str) -> int:
        if not mint: return 1
        now = datetime.now(timezone.utc)
        if mint not in self.history: self.history[mint] = []
        self.history[mint].append(now)
        self.history[mint] = [t for t in self.history[mint] if now - t <= self.window]
        return len(self.history[mint])

class HeliusEngine:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self.api_key, self.session = api_key, session
        self.url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.webhook_api = f"https://api.helius-rpc.com/v0/webhooks?api-key={api_key}"

    async def setup_webhooks(self, webhook_url: str, addresses: List[str]):
        if not self.api_key or not webhook_url or not addresses: return
        full_url = f"{webhook_url.rstrip('/')}/webhook/helius"
        try:
            async with self.session.get(self.webhook_api) as resp:
                webhooks = await resp.json()
                for wh in webhooks:
                    if wh['webhookURL'] == full_url: return
            payload = {"webhookURL": full_url, "transactionTypes": ["SWAP"], "accountAddresses": addresses, "webhookType": "enhanced"}
            await self.session.post(self.webhook_api, json=payload)
        except: pass

    async def get_wallet_activity(self, wallet: str) -> List[Dict]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress", "params": [wallet, {"limit": 5}]}
        try:
            async with self.session.post(self.url, json=payload) as resp:
                data = await resp.json()
                return data.get('result', [])
        except: return []

    async def get_funding_source(self, wallet_address: str) -> str:
        if not self.api_key: return "Unknown"
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress", "params": [wallet_address, {"limit": 5}]}
        try:
            async with self.session.post(self.url, json=payload) as resp:
                data = await resp.json()
                sigs = data.get('result', [])
                if not sigs: return "Fresh"
                tx_payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction", "params": [sigs[-1]['signature'], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]}
                async with self.session.post(self.url, json=tx_payload) as tx_resp:
                    res = str(await tx_resp.json()).lower()
                    for cex in ['binance', 'coinbase', 'okx', 'bybit']:
                        if cex in res: return "CEX Funded"
                return "Private"
        except: return "Unknown"

    async def get_token_risk_profile(self, address: str) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": "risk", "method": "getAsset", "params": {"id": address}}
        try:
            async with self.session.post(self.url, json=payload, timeout=5) as resp:
                data = await resp.json()
                t = data.get('result', {}).get('token_info', {})
                safe = (t.get('mint_authority') is None) and (t.get('freeze_authority') is None)
                return {"is_safe": safe, "reason": "Renounced" if safe else "Auth Enabled"}
        except: return {"is_safe": False, "reason": "Lookup Failed"}

class SecurityEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    async def get_safety_report(self, address: str, vol_24h: float, liq: float, helius: HeliusEngine) -> Dict[str, Any]:
        vl_ratio = vol_24h / liq if liq > 0 else 999.0
        score = 0
        try:
            async with self.session.get(f"https://api.rugcheck.xyz/v1/tokens/{address}/report", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    score = max(0, 100 - (data.get('score', 1000) / 10))
        except: pass
        hr = await helius.get_token_risk_profile(address)
        rugged = score < 60 or not hr['is_safe'] or vl_ratio > BotConfig.vl_ratio_max
        return {"safety_score": score, "is_rugged": rugged, "helius_reason": hr['reason']}

class SignalStore:
    def __init__(self, db_path: str):
        self.db_path, self.conn, self.symbol_locks = db_path, None, {}
    async def init_db(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, market_type TEXT, contract_address TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, confidence REAL, safety_score REAL, sentiment_score REAL, funding REAL DEFAULT 0.0, is_cluster INTEGER DEFAULT 0, status TEXT DEFAULT 'open', model_version TEXT)")
        await self.conn.execute("CREATE TABLE IF NOT EXISTS discovered_wallets (address TEXT PRIMARY KEY, total_pnl REAL DEFAULT 0.0, auto_copy INTEGER DEFAULT 0, is_blacklisted INTEGER DEFAULT 0)")
        await self.conn.commit()
    async def insert_signal(self, s: dict):
        await self.conn.execute("INSERT INTO signals(timestamp, symbol, market_type, contract_address, signal, entry, sl, tp, confidence, safety_score, sentiment_score, funding, is_cluster, model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s['timestamp'], s['symbol'], s['market_type'], s.get('contract_address'), s['signal'], s['entry'], s.get('sl'), s.get('tp'), s['confidence'], s.get('safety_score', 0), s.get('sentiment_score', 50), s.get('funding', 0.0), s.get('is_cluster', 0), s.get('model_version')))
        await self.conn.commit()
    async def update_signal_status(self, sig_id: int, status: str):
        await self.conn.execute("UPDATE signals SET status = ? WHERE id = ?", (status, sig_id))
        await self.conn.commit()
    async def has_open_signal(self, iden: str):
        async with self.conn.execute("SELECT 1 FROM signals WHERE (symbol=? OR contract_address=?) AND status='open'", (iden, iden)) as c: return await c.fetchone() is not None
    async def get_blacklist(self):
        async with self.conn.execute("SELECT address FROM discovered_wallets WHERE is_blacklisted=1") as c: return [r[0] for r in await c.fetchall()]
    async def get_recent_signals(self, limit: int):
        async with self.conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)) as c:
            cols = [column[0] for column in c.description]
            return [dict(zip(cols, row)) for row in await c.fetchall()]
    def get_symbol_lock(self, s):
        if s not in self.symbol_locks: self.symbol_locks[s] = asyncio.Lock()
        return self.symbol_locks[s]

class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.helius = HeliusEngine(HELIUS_API_KEY, session)
        self.dex = DexEngine(session)
        self.security = SecurityEngine(session)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)

    async def get_ml_confidence(self, symbol: str, features: list) -> float:
        path = os.path.join(self.cfg.ml_model_path, f"{symbol.replace('/', '_')}_model.joblib")
        if not os.path.exists(path): path = os.path.join(self.cfg.ml_model_path, "GLOBAL_model.joblib")
        if not os.path.exists(path): return 70.0
        try:
            model = joblib.load(path)
            return round(model.predict_proba([features])[0][1] * 100, 2)
        except: return 70.0

    async def get_btc_health(self) -> bool:
        try:
            ohlcv = await self.exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe=self.cfg.higher_timeframe, limit=2)
            return float(ohlcv[-1][4]) > float(ohlcv[-2][4])
        except: return True

    async def generate_cex_signal(self, symbol: str, btc_bullish: bool):
        if not self.cfg.enable_cex: return
        async with self.semaphore:
            async with self.store.get_symbol_lock(symbol):
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=100)
                    df = add_indicators(pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]), self.cfg.indicators)
                    if df.empty or await self.store.has_open_signal(symbol): return
                    last = df.iloc[-1]
                    if last['adx'] < self.cfg.indicators.adx_threshold: return

                    stype = "BUY" if last['ema_short'] > last['ema_medium'] else "SELL" if last['ema_short'] < last['ema_medium'] else None
                    if not stype or (stype == "BUY" and not btc_bullish) or (stype == "SELL" and btc_bullish): return

                    price, atr = float(last['close']), float(last['atr'])
                    sl_dist, tp_dist = atr * self.cfg.indicators.atr_sl_mult, atr * self.cfg.indicators.atr_tp_mult

                    sl = price - sl_dist if stype == "BUY" else price + sl_dist
                    tp = price + tp_dist if stype == "BUY" else price - tp_dist

                    social = await self.sentinel.get_sentiment(symbol)
                    ml_conf = await self.get_ml_confidence(symbol, [price, 0, social['score'], 70.0])

                    sig = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "signal": stype, "market_type": "CEX",
                        "entry": price, "sl": round(sl, 6), "tp": round(tp, 6), "confidence": ml_conf,
                        "sentiment_score": social['score'], "funding": social.get('funding', 0.0), "model_version": self.cfg.model_version
                    }
                    await self.store.insert_signal(sig)
                    await notify_new_signal(sig, self.session, self.cfg)
                except Exception as e: logger.error(f"CEX Gen Error: {e}")

    async def fetch_dex_alpha(self):
        while True:
            if self.cfg.enable_dex:
                try:
                    async with self.session.get("https://api.dexscreener.com/latest/dex/search?q=solana") as resp:
                        data = await resp.json()
                        for pair in data.get('pairs', [])[:10]:
                            addr = pair.get('baseToken', {}).get('address')
                            if await self.store.has_open_signal(addr): continue
                            report = await self.security.get_safety_report(addr, float(pair.get('volume', {}).get('h24', 0)), float(pair.get('liquidity', {}).get('usd', 0)), self.helius)
                            if report['is_rugged']: continue
                            if float(pair.get('priceChange', {}).get('m5', 0)) > 5.0:
                                social = await self.sentinel.get_sentiment(pair['baseToken']['symbol'])
                                entry = float(pair.get('priceUsd', 0))
                                ml_conf = await self.get_ml_confidence("GLOBAL", [entry, report['safety_score'], social['score'], 85.0])
                                sig = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": pair['baseToken']['symbol'], "market_type": "DEX", "contract_address": addr,
                                    "signal": f"BUY (GEM - {social['label']})", "entry": entry, "confidence": ml_conf,
                                    "safety_score": report['safety_score'], "sentiment_score": social['score']
                                }
                                await self.store.insert_signal(sig)
                                await notify_new_signal(sig, self.session, self.cfg)
                except: pass
            await asyncio.sleep(self.cfg.dex_poll_interval)

    async def fetch_insider_signals(self):
        # Note: Handled by Webhooks in real-time version.
        while True:
            await asyncio.sleep(self.cfg.insider_poll_interval)

# -----------------------------
# Bot Lifecycle
# -----------------------------

async def telegram_command_listener(session: aiohttp.ClientSession, cfg: BotConfig):
    last_id = 0
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getUpdates"
    while True:
        try:
            async with session.get(url, params={"offset": last_id + 1, "timeout": 20}) as resp:
                data = await resp.json()
                for update in data.get("result", []):
                    last_id = update["update_id"]
                    msg = update.get("message", {})
                    text, chat_id = msg.get("text", ""), str(msg.get("chat", {}).get("id", ""))
                    if chat_id != cfg.telegram_chat_id: continue
                    if text == "/status":
                        await send_tg_msg(session, cfg, f"🚀 *QuikPulse Control Center*\nCEX: {cfg.enable_cex}\nDEX: {cfg.enable_dex}\nWhale: {cfg.enable_whale}")
                    elif text == "/history": await handle_history_command(session, cfg)
                    elif text.startswith("/add "):
                        pair = text.split(" ")[1].upper()
                        if pair not in cfg.symbols: cfg.symbols.append(pair)
                        await send_tg_msg(session, cfg, f"➕ Monitoring `{pair}`")
        except: pass
        await asyncio.sleep(4)

async def handle_history_command(session: aiohttp.ClientSession, cfg: BotConfig):
    signals = await store.get_recent_signals(10)
    if not signals: return
    report = "📊 *Recent Performance*\n\n"
    for s in signals: report += f"#{s['id']} `{s['symbol']}`: {s['signal']}\n"
    await send_tg_msg(session, cfg, report)

async def send_tg_msg(session: aiohttp.ClientSession, cfg: BotConfig, text: str):
    if not cfg.telegram_bot_token: return
    try: await session.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage", json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

async def background_monitor():
    while True:
        try:
            h = await generator.get_btc_health()
            await asyncio.gather(*(generator.generate_cex_signal(s, h) for s in cfg.symbols))
        except: pass
        await asyncio.sleep(cfg.poll_interval)

async def continuous_learning_loop(trainer: ModelTrainer):
    while True:
        await asyncio.sleep(86400)
        await trainer.retrain_model("GLOBAL")

def add_indicators(df, cfg_ind):
    df['ema_short'] = df['close'].ewm(span=cfg_ind.ema_short).mean()
    df['ema_medium'] = df['close'].ewm(span=cfg_ind.ema_medium).mean()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    return df

async def notify_new_signal(sig, session, cfg_bot):
    if not cfg_bot.telegram_bot_token: return
    prefix = "🚨 *CLUSTER BUY DETECTED!* 🚨\n" if sig.get('is_cluster') else "🚀 *NEW SIGNAL*\n"
    ca_link = f"https://dexscreener.com/solana/{sig.get('contract_address')}" if sig.get('contract_address') else "#"

    funding = sig.get('funding', 0.0)
    f_emoji = "⚠️" if abs(funding) > 0.01 else "✅"

    msg = (
        f"{prefix}Pair: `{sig['symbol']}`\nType: {sig['signal']}\n"
        f"Price: `${sig['entry']}`\n"
        f"SL: `{sig.get('sl', 'N/A')}` | TP: `{sig.get('tp', 'N/A')}`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Social: `{sig.get('sentiment_score', 50)}` | Funding: `{funding}` {f_emoji}\n"
        f"Safety: `{sig.get('safety_score', 0)}` | Conf: `{sig['confidence']}%`\n"
        f"[View Chart]({ca_link})"
    )
    try: await session.post(f"https://api.telegram.org/bot{cfg_bot.telegram_bot_token}/sendMessage", json={"chat_id": cfg_bot.telegram_chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True})
    except: pass

app = FastAPI()
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
cluster_map = ClusterEngine(cfg.cluster_window_minutes)
exchange = ccxt.kucoinfutures({"apiKey": os.getenv("API_KEY"), "secret": os.getenv("API_SECRET"), "password": os.getenv("API_PASS"), "enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    signals = await store.get_recent_signals(30)
    html = "<h1>QuikPulse Dashboard</h1><table><tr><th>ID</th><th>Symbol</th><th>Type</th><th>Entry</th><th>SL</th><th>TP</th><th>Social</th><th>Conf</th></tr>"
    for s in signals: html += f"<tr><td>{s['id']}</td><td>{s['symbol']}</td><td>{s['signal']}</td><td>{s['entry']}</td><td>{s.get('sl','-')}</td><td>{s.get('tp','-')}</td><td>{s['sentiment_score']}</td><td>{s['confidence']}%</td></tr>"
    return html + "</table>"

@app.post("/webhook/helius")
async def helius_webhook_handler(request: Request):
    data = await request.json()
    logger.info(f"Helius Webhook Received: {len(data)} events")

    for activity in data:
        events = activity.get('events', {})
        swap = events.get('swap', {})
        
        if swap:
            raw_amount = float(swap.get('nativeInput', {}).get('amount', 0))
            amount_sol = raw_amount / 1_000_000_000 
            
            if amount_sol < cfg.min_insider_buy_sol:
                continue 

            token_addr = swap.get('tokenOutMint')
            if await store.has_open_signal(token_addr):
                continue

            price_data = await generator.dex.get_price_data(token_addr)
            if price_data['price'] > 0:
                social = await generator.sentinel.get_sentiment(price_data['symbol'])
                report = await generator.security.get_safety_report(token_addr, price_data['vol24'], price_data['liq'], generator.helius)
                
                hit_count = generator.cluster.record_and_check(token_addr)
                is_cluster = 1 if hit_count >= 2 else 0

                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": price_data['symbol'],
                    "market_type": "INSIDER (WEBHOOK)",
                    "contract_address": token_addr,
                    "signal": f"BUY ({'CLUSTER' if is_cluster else 'WHALE MOVE'})",
                    "entry": price_data['price'],
                    "confidence": 95.0 if is_cluster else 85.0,
                    "safety_score": report['safety_score'],
                    "sentiment_score": social['score'],
                    "is_cluster": is_cluster
                }

                await store.insert_signal(sig)
                await notify_new_signal(sig, session, cfg)
                logger.info(f"🚀 Webhook Signal Fired: {price_data['symbol']} ({amount_sol} SOL)")

    return {"status": "success"}

@app.on_event("startup")
async def startup():
    global session, generator
    await store.init_db()
    session = aiohttp.ClientSession()
    sentinel = SocialSentinel(SANTIMENT_API_KEY, session)
    generator = SignalGenerator(cfg, store, exchange, sentinel, cluster_map, session)
    trainer = ModelTrainer(cfg.sqlite_db, cfg.ml_model_path)
    
    if WEBHOOK_URL and cfg.tracked_wallets: 
        await generator.helius.setup_webhooks(WEBHOOK_URL, cfg.tracked_wallets)
        
    asyncio.create_task(background_monitor())
    asyncio.create_task(generator.fetch_dex_alpha())
    # Polling disabled in favor of Helius Webhooks
    # asyncio.create_task(generator.fetch_insider_signals())
    asyncio.create_task(telegram_command_listener(session, cfg))
    asyncio.create_task(continuous_learning_loop(trainer))
    logger.info("QuikPulse: Production Ready with Helius Webhook & Whale Filter.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
