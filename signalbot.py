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
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SignalBotAI")

# -----------------------------
# Configs
# -----------------------------
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
    symbols: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT").split(',')
    ])
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    higher_timeframe: str = os.getenv("HIGHER_TIME_FRAME", "1h")
    limit: int = int(os.getenv("LIMIT", 1000))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v4")
    min_train_samples: int = 20

# -----------------------------
# Database / SignalStore
# -----------------------------
class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self.symbol_locks: Dict[str, asyncio.Lock] = {}

    async def init_db(self):
        if self.conn: return
        db_dir = os.path.dirname(self.db_path) or "."
        if db_dir != ".": os.makedirs(db_dir, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("PRAGMA synchronous=NORMAL;")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, signal TEXT,
                entry REAL, sl REAL, tp REAL, confidence REAL,
                rr REAL, status TEXT DEFAULT 'open',
                feature_json TEXT, model_version TEXT, executed_price REAL
            )
        """)
        await self.conn.commit()

    async def get_recent_signals(self, limit: int = 5):
        """NEW: Fetches the latest signals for the landing page."""
        if not self.conn: await self.init_db()
        query = "SELECT symbol, signal, entry, confidence, status, timestamp FROM signals ORDER BY id DESC LIMIT ?"
        async with self.conn.execute(query, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]

    async def has_open_signal(self, symbol: str) -> bool:
        async with self.conn.execute("SELECT 1 FROM signals WHERE symbol=? AND status='open' LIMIT 1", (symbol,)) as cursor:
            return await cursor.fetchone() is not None

    async def insert_signal(self, sig: dict):
        await self.conn.execute("""
            INSERT INTO signals(timestamp, symbol, signal, entry, sl, tp, confidence, rr, feature_json, model_version, executed_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig['timestamp'], sig['symbol'], sig['signal'], sig['entry'], sig['sl'], sig['tp'],
            sig['confidence'], sig['rr'], json.dumps(sig['features']), sig['model_version'], sig['executed_price']
        ))
        await self.conn.commit()

    async def update_signal_status(self, symbol: str, last_price: float):
        async with self.conn.execute("SELECT id, signal, sl, tp FROM signals WHERE symbol=? AND status='open'", (symbol,)) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                sid, stype, sl, tp = row
                new_status = None
                if stype == "BUY":
                    if last_price >= tp: new_status = 'take_profit'
                    elif last_price <= sl: new_status = 'stop_loss'
                else: 
                    if last_price <= tp: new_status = 'take_profit'
                    elif last_price >= sl: new_status = 'stop_loss'

                if new_status:
                    await self.conn.execute("UPDATE signals SET status=? WHERE id=?", (new_status, sid))
                    await self.conn.commit()
                    async with self.conn.execute("SELECT * FROM signals WHERE id=?", (sid,)) as c2:
                        res = await c2.fetchone()
                        if res: await notify_signal_update(dict(zip([col[0] for col in c2.description], res)))

    async def get_training_data(self, symbol: Optional[str] = None):
        query = "SELECT feature_json, status FROM signals WHERE status IN ('take_profit', 'stop_loss')"
        params = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
            
        async with self.conn.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            X, y = [], []
            for feat_json, status in rows:
                feat_dict = json.loads(feat_json)
                X.append([feat_dict.get(f, 0.0) for f in FEATURE_LIST])
                y.append(1 if status == 'take_profit' else 0)
            return np.array(X), np.array(y)

    async def close(self):
        if self.conn: await self.conn.close()

    def get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self.symbol_locks: self.symbol_locks[symbol] = asyncio.Lock()
        return self.symbol_locks[symbol]

# -----------------------------
# ML & Generator
# -----------------------------
class MLModelManager:
    def __init__(self, path: str, version: str = "v4"):
        self.path, self.version = path, version
        self.models, self.scalers = {}, {}
        if not os.path.exists(path): os.makedirs(path)

    def load_model(self, symbol: str) -> bool:
        for s in [symbol, "GLOBAL"]:
            mf = os.path.join(self.path, f"{s.replace('/', '_')}_{self.version}.pkl")
            sf = os.path.join(self.path, f"{s.replace('/', '_')}_{self.version}_scaler.pkl")
            try:
                if os.path.exists(mf) and os.path.exists(sf):
                    self.models[symbol] = joblib.load(mf)
                    self.scalers[symbol] = joblib.load(sf)
                    return True
            except: continue
        return False

    def save_model(self, symbol: str, model, scaler):
        mf = os.path.join(self.path, f"{symbol.replace('/', '_')}_{self.version}.pkl")
        sf = os.path.join(self.path, f"{symbol.replace('/', '_')}_{self.version}_scaler.pkl")
        joblib.dump(model, mf)
        joblib.dump(scaler, sf)
        self.models[symbol] = model
        self.scalers[symbol] = scaler

class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ml_mgr: MLModelManager, ex: ccxt.Exchange):
        self.cfg, self.store, self.ml_mgr, self.exchange = cfg, store, ml_mgr, ex
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.oi_cache: Dict[str, float] = {} 

    async def fetch_candles(self, sym: str, tf: str) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe=tf, limit=self.cfg.limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {sym}: {e}")
            return pd.DataFrame()

    async def fetch_market_stats(self, sym: str) -> Dict[str, float]:
        stats = {"funding": 0.0, "oi_change": 0.0, "liquidity": 0.0}
        try:
            funding = await self.exchange.fetch_funding_rate(sym)
            oi_data = await self.exchange.fetch_open_interest(sym)
            stats["funding"] = float(funding.get('fundingRate', 0))
            current_oi = float(oi_data.get('openInterestAmount', 0))
            prev_oi = self.oi_cache.get(sym, current_oi)
            if prev_oi > 0:
                stats["oi_change"] = ((current_oi - prev_oi) / prev_oi) * 100
            self.oi_cache[sym] = current_oi
            ob = await self.exchange.fetch_order_book(sym, limit=5)
            if ob['bids'] and ob['asks']:
                bid, ask = ob['bids'][0][0], ob['asks'][0][0]
                stats["liquidity"] = ((ask - bid) / bid) * 100
        except: pass
        return stats

    async def get_btc_health(self) -> bool:
        df_btc = await self.fetch_candles("BTC/USDT:USDT", self.cfg.higher_timeframe)
        if df_btc.empty: return True
        ema_200 = df_btc['close'].ewm(span=self.cfg.indicators.ema_filter, adjust=False).mean()
        return float(df_btc['close'].iloc[-1]) > float(ema_200.iloc[-1])

    async def generate_signal(self, symbol: str, btc_bullish: bool):
        async with self.semaphore:
            lock = self.store.get_symbol_lock(symbol)
            async with lock:
                df = add_indicators(await self.fetch_candles(symbol, self.cfg.timeframe), self.cfg.indicators)
                if df.empty or await self.store.has_open_signal(symbol): return
                m_stats = await self.fetch_market_stats(symbol)
                if m_stats["liquidity"] > self.cfg.indicators.max_spread_pct: return

                last = df.iloc[-1]
                if last['adx'] < self.cfg.indicators.adx_threshold: return
                if last['ema_short'] > last['ema_medium']: stype = "BUY"
                elif last['ema_short'] < last['ema_medium']: stype = "SELL"
                else: return

                if stype == "BUY" and not btc_bullish: return
                if stype == "SELL" and btc_bullish: return

                entry, atr = float(last['close']), float(last['atr'] or 0)
                sl = entry - atr * self.cfg.indicators.atr_sl_mult if stype == "BUY" else entry + atr * self.cfg.indicators.atr_sl_mult
                tp = entry + atr * self.cfg.indicators.atr_tp_mult if stype == "BUY" else entry - atr * self.cfg.indicators.atr_tp_mult

                features = {f: last[f] for f in FEATURE_LIST if f not in ['funding_rate', 'oi_5m_change', 'liquidity_score']}
                features['funding_rate'] = m_stats["funding"]
                features['oi_5m_change'] = m_stats["oi_change"]
                features['liquidity_score'] = m_stats["liquidity"]
                
                confidence = 50.0
                if symbol in self.ml_mgr.models:
                    try:
                        X = np.array([features[f] for f in FEATURE_LIST]).reshape(1, -1)
                        X_scaled = self.ml_mgr.scalers[symbol].transform(X)
                        confidence = self.ml_mgr.models[symbol].predict_proba(X_scaled)[0][1] * 100
                    except: pass

                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "signal": stype,
                    "entry": entry, "sl": sl, "tp": tp, "confidence": confidence, "rr": 2.0,
                    "features": features, "model_version": self.cfg.model_version, "executed_price": entry,
                }
                await self.store.insert_signal(sig)
                await notify_new_signal(sig)

# -----------------------------
# Helpers & Notifications
# -----------------------------
FEATURE_LIST = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'atr', 'adx', 'bb_trend', 'vol_ok', 'funding_rate', 'oi_5m_change', 'liquidity_score']

def add_indicators(df: pd.DataFrame, ind_cfg: IndicatorsConfig) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ema_short'] = df['close'].ewm(span=ind_cfg.ema_short, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=ind_cfg.ema_medium, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind_cfg.ema_long, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind_cfg.rsi_period).rsi().fillna(50)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind_cfg.atr_period).average_true_range().fillna(0)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind_cfg.atr_period).adx().fillna(0)
    bb = ta.volatility.BollingerBands(df['close'], ind_cfg.bb_period, ind_cfg.bb_std)
    df['bb_trend'] = np.where(df['close'] > bb.bollinger_mavg(), 1, -1)
    df['vol_ok'] = (df['volume'] > df['volume'].rolling(20).mean().fillna(df['volume'].mean())).astype(int)
    return df

async def set_bot_commands(token: str):
    url = f"https://api.telegram.org/bot{token}/setMyCommands"
    commands = [
        {"command": "status", "description": "📊 View dashboard & live stats"},
        {"command": "pair", "description": "🔍 Manage trading pairs (add/remove/list)"},
        {"command": "set", "description": "⚙️ Update indicators or timeframes"},
        {"command": "help", "description": "❓ Show user guide & commands"}
    ]
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json={"commands": commands}) as resp:
                if resp.status == 200: logger.info("Telegram command menu registered.")
        except: pass

async def send_help_guide(token, cid):
    guide = (
        "👋 *SignalBotAI SaaS Guide*\n\n"
        "Control your AI signal engine using these commands:\n\n"
        "📊 *Monitoring*\n"
        "• `/status` : System health & current config\n\n"
        "🔍 *Trading Pairs*\n"
        "• `/pair list` : Current active assets\n"
        "• `/pair add BTC/USDT:USDT` : Start scanning a new asset\n"
        "• `/pair remove ETH/USDT:USDT` : Stop scanning\n\n"
        "⚙️ *Live Configuration*\n"
        "• `/set timeframe 15m` : Change candle entry TF\n"
        "• `/set poll 60` : Change scan frequency\n"
        "• `/set spread 0.2` : Max allow bid/ask gap %\n"
        "• `/set adx 30` : Min trend strength requirement"
    )
    await send_tg(token, cid, guide)

async def send_heartbeat(generator: SignalGenerator, cfg: BotConfig):
    btc_bullish = await generator.get_btc_health()
    market_filter = "🟢 BULLISH" if btc_bullish else "🔴 BEARISH"
    msg = (
        f"🤖 *SignalBotAI Master Dashboard*\n"
        f"------------------------\n"
        f"Market: {market_filter}\n"
        f"Pairs Active: {len(cfg.symbols)}\n"
        f"Entry TF: {cfg.timeframe} | Conf TF: {cfg.higher_timeframe}\n"
        f"Scan Every: {cfg.poll_interval}s\n"
        f"Spread Max: {cfg.indicators.max_spread_pct}%\n"
        f"ADX Min: {cfg.indicators.adx_threshold}\n"
        f"------------------------\n"
        f"Use /help to see command formats."
    )
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)

async def telegram_polling_loop(generator: SignalGenerator, cfg: BotConfig):
    if not cfg.telegram_bot_token: return
    await set_bot_commands(cfg.telegram_bot_token)
    offset = 0
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getUpdates"
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url, params={"offset": offset, "timeout": 30}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            offset = update["update_id"] + 1
                            msg_obj = update.get("message", {})
                            raw_text = msg_obj.get("text", "").lower().strip()
                            chat_id = str(msg_obj.get("chat", {}).get("id", ""))
                            text = raw_text.replace("/", "")

                            if chat_id == str(cfg.telegram_chat_id):
                                if text in ["start", "help"]: await send_help_guide(cfg.telegram_bot_token, chat_id)
                                elif text == "status": await send_heartbeat(generator, cfg)
                                elif text.startswith("set"):
                                    parts = raw_text.split(" ")
                                    if len(parts) == 3:
                                        param, val = parts[1], parts[2]
                                        try:
                                            if param == "timeframe":
                                                if val in ['1m', '3m', '5m', '15m', '1h', '4h']:
                                                    cfg.timeframe = val
                                                    await send_tg(cfg.telegram_bot_token, chat_id, f"✅ Entry TF updated to {val}")
                                            elif param == "poll":
                                                cfg.poll_interval = int(val)
                                                await send_tg(cfg.telegram_bot_token, chat_id, f"✅ Poll interval set to {val}s")
                                            elif param == "spread":
                                                cfg.indicators.max_spread_pct = float(val)
                                                await send_tg(cfg.telegram_bot_token, chat_id, "✅ Max spread limit updated")
                                            elif param == "adx":
                                                cfg.indicators.adx_threshold = int(val)
                                                await send_tg(cfg.telegram_bot_token, chat_id, "✅ Min ADX threshold updated")
                                        except: await send_tg(cfg.telegram_bot_token, chat_id, "❌ Parameter error.")
                                elif text.startswith("pair"):
                                    parts = raw_text.split(" ")
                                    if len(parts) == 2 and parts[1] == "list":
                                        await send_tg(cfg.telegram_bot_token, chat_id, "📜 *Active Pairs*:\n" + "\n".join(cfg.symbols))
                                    elif len(parts) == 3:
                                        action, symbol = parts[1], parts[2].upper()
                                        if action == "add":
                                            markets = await generator.exchange.load_markets()
                                            if symbol in markets:
                                                if symbol not in cfg.symbols:
                                                    cfg.symbols.append(symbol)
                                                    generator.ml_mgr.load_model(symbol)
                                                    await send_tg(cfg.telegram_bot_token, chat_id, f"✅ {symbol} added.")
                                        elif action == "remove":
                                            if symbol in cfg.symbols:
                                                cfg.symbols.remove(symbol)
                                                await send_tg(cfg.telegram_bot_token, chat_id, f"❌ Removed {symbol}.")
            except Exception as e: logger.error(f"Telegram Loop Error: {e}")
            await asyncio.sleep(2)

async def notify_new_signal(sig):
    if not cfg.telegram_bot_token: return
    msg = (f"🚀 *New AI Trading Signal*\n\nPair: `{sig['symbol']}`\nDirection: *{sig['signal']}*\nEntry: {sig['entry']}\nAI Confidence: {sig['confidence']:.1f}%")
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)

async def notify_signal_update(sig):
    if not cfg.telegram_bot_token: return
    emoji = "✅" if sig['status'] == 'take_profit' else "❌"
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, f"{emoji} *Signal Closed*: {sig['symbol']} hit {sig['status'].upper().replace('_', ' ')}")

async def send_tg(token, cid, msg):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        try: await session.post(url, json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"})
        except: pass

async def train_models_task(generator, cfg):
    while True:
        await asyncio.sleep(86400)
        X_glob, y_glob = await generator.store.get_training_data()
        if len(y_glob) >= cfg.min_train_samples:
            try:
                scaler, model = StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42)
                X_scaled = scaler.fit_transform(X_glob)
                model.fit(X_scaled, y_glob)
                generator.ml_mgr.save_model("GLOBAL", model, scaler)
            except: pass

# -----------------------------
# Lifecycle & Web Routes
# -----------------------------
app = FastAPI()
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
ml_mgr = MLModelManager(cfg.ml_model_path)
exchange = ccxt.kucoinfutures({"enableRateLimit": True, "options": {"defaultType": "future"}})
generator = SignalGenerator(cfg, store, ml_mgr, exchange)

# Set up templates folder
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    """SaaS Landing Page with Live Signal Feed."""
    recent_signals = await store.get_recent_signals(5)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_pairs": len(cfg.symbols),
        "timeframe": cfg.timeframe,
        "signals": recent_signals
    })

@app.get("/health")
async def health():
    return {"status": "healthy"}

async def background_monitor():
    while True:
        try:
            btc_bullish = await generator.get_btc_health()
            await asyncio.gather(*(generator.generate_signal(s, btc_bullish) for s in cfg.symbols), return_exceptions=True)
            for s in cfg.symbols:
                df = await generator.fetch_candles(s, cfg.timeframe)
                if not df.empty: await store.update_signal_status(s, df['close'].iloc[-1])
        except Exception as e: logger.error(f"Monitor error: {e}")
        await asyncio.sleep(cfg.poll_interval)

@app.on_event("startup")
async def startup():
    await store.init_db()
    for _ in range(3):
        try: 
            await exchange.load_markets()
            break
        except: await asyncio.sleep(5)
    for s in cfg.symbols: ml_mgr.load_model(s)
    
    asyncio.create_task(background_monitor())
    asyncio.create_task(telegram_polling_loop(generator, cfg))
    asyncio.create_task(train_models_task(generator, cfg))
    logger.info("SignalBotAI SaaS: Online.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
