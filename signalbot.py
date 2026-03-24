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
from fastapi import FastAPI
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

@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv(
            "SYMBOLS",
            "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,ADA/USDT:USDT,XRP/USDT:USDT"
        ).split(',')
    ])
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "1h")
    limit: int = int(os.getenv("LIMIT", 1000))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    heartbeat_interval: int = 14400  # 4 Hours
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v2")
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

    async def get_training_data(self, symbol: str):
        async with self.conn.execute(
            "SELECT feature_json, status FROM signals WHERE symbol=? AND status IN ('take_profit', 'stop_loss')", 
            (symbol,)
        ) as cursor:
            rows = await cursor.fetchall()
            X, y = [], []
            for feat_json, status in rows:
                X.append(list(json.loads(feat_json).values()))
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
    def __init__(self, path: str, version: str = "v2"):
        self.path, self.version = path, version
        self.models, self.scalers = {}, {}
        if not os.path.exists(path): os.makedirs(path)

    def load_model(self, symbol: str) -> bool:
        mf = os.path.join(self.path, f"{symbol.replace('/', '_')}_{self.version}.pkl")
        sf = os.path.join(self.path, f"{symbol.replace('/', '_')}_{self.version}_scaler.pkl")
        try:
            if os.path.exists(mf): self.models[symbol] = joblib.load(mf)
            if os.path.exists(sf): self.scalers[symbol] = joblib.load(sf)
            return symbol in self.models
        except: return False

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

    async def fetch_candles(self, sym: str, tf: str) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe=tf, limit=self.cfg.limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except: return pd.DataFrame()

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

                last = df.iloc[-1]
                if last['adx'] < self.cfg.indicators.adx_threshold: return

                stype = "BUY" if last['ema_short'] > last['ema_medium'] else "SELL" if last['ema_short'] < last['ema_medium'] else None
                if stype == "BUY" and not btc_bullish: return
                if not stype: return

                entry, atr = float(last['close']), float(last['atr'] or 0)
                sl = entry - atr * self.cfg.indicators.atr_sl_mult if stype == "BUY" else entry + atr * self.cfg.indicators.atr_sl_mult
                tp = entry + atr * self.cfg.indicators.atr_tp_mult if stype == "BUY" else entry - atr * self.cfg.indicators.atr_tp_mult

                features = {f: last[f] for f in FEATURE_LIST}
                confidence = 50.0 
                if symbol in self.ml_mgr.models:
                    try:
                        X = np.array(list(features.values())).reshape(1, -1)
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
# Notification & Heartbeat
# -----------------------------
async def send_heartbeat(generator: SignalGenerator, cfg: BotConfig):
    btc_bullish = await generator.get_btc_health()
    market_filter = "🟢 BULLISH (Active)" if btc_bullish else "🔴 BEARISH (Paused)"
    
    # Check volatility on first symbol
    df = await generator.fetch_candles(cfg.symbols[0], cfg.timeframe)
    df_ind = add_indicators(df, cfg.indicators)
    adx = df_ind['adx'].iloc[-1] if not df_ind.empty else 0
    vol_status = "⚡ Trending" if adx >= cfg.indicators.adx_threshold else "💤 Choppy"

    msg = (
        f"🤖 *Bot Status Report*\n"
        f"────────────────────────\n"
        f"✅ *Status:* `Running`\n"
        f"📈 *Market:* {market_filter}\n"
        f"📊 *Volatility:* {vol_status} `({adx:.1f})`\n"
        f"🔍 *Pairs:* `{len(cfg.symbols)}` active\n"
        f"────────────────────────\n"
        f"_Time: {datetime.now().strftime('%H:%M:%S')} UTC_"
    )
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)

async def heartbeat_loop(generator: SignalGenerator, cfg: BotConfig):
    await asyncio.sleep(60) 
    while True:
        try:
            await send_heartbeat(generator, cfg)
        except Exception as e: logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(cfg.heartbeat_interval)

async def telegram_polling_loop(generator: SignalGenerator, cfg: BotConfig):
    """Listens for /status command using Long Polling."""
    if not cfg.telegram_bot_token: return
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
                            message = update.get("message", {})
                            text = message.get("text", "")
                            chat_id = str(message.get("chat", {}).get("id", ""))
                            
                            if chat_id == cfg.telegram_chat_id and text.lower() == "/status":
                                await send_heartbeat(generator, cfg)
            except Exception as e:
                logger.error(f"Telegram polling error: {e}")
            await asyncio.sleep(2)

# -----------------------------
# Training Task
# -----------------------------
async def train_models_periodically():
    while True:
        logger.info("Starting scheduled ML retraining...")
        for symbol in cfg.symbols:
            X, y = await store.get_training_data(symbol)
            if len(y) >= cfg.min_train_samples:
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_scaled, y)
                    ml_mgr.save_model(symbol, model, scaler)
                    logger.info(f"Retrained model for {symbol} with {len(y)} samples.")
                except Exception as e: logger.error(f"Training failed for {symbol}: {e}")
        await asyncio.sleep(86400)

# -----------------------------
# Standard Logic & Helpers
# -----------------------------
FEATURE_LIST = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'atr', 'adx', 'bb_trend', 'vol_ok']

def add_indicators(df: pd.DataFrame, ind_cfg: IndicatorsConfig) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ema_short'] = df['close'].ewm(span=ind_cfg.ema_short, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=ind_cfg.ema_medium, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind_cfg.ema_long, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind_cfg.rsi_period).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind_cfg.atr_period).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind_cfg.atr_period).adx()
    bb = ta.volatility.BollingerBands(df['close'], ind_cfg.bb_period, ind_cfg.bb_std)
    df['bb_trend'] = np.where(df['close'] > bb.bollinger_mavg(), 1, -1)
    df['vol_ok'] = (df['volume'] > df['volume'].rolling(20).mean().fillna(df['volume'].mean())).astype(int)
    return df

async def notify_new_signal(sig):
    if not cfg.telegram_bot_token: return
    msg = f"*🚀 New AI Signal*\n*Pair:* `{sig['symbol']}`\n*Signal:* `{sig['signal']}`\n*AI Confidence:* `{sig['confidence']:.1f}%`"
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)

async def notify_signal_update(sig):
    if not cfg.telegram_bot_token: return
    emoji = "✅" if sig['status'] == 'take_profit' else "❌"
    msg = f"*{emoji} Signal Closed*\n*Pair:* `{sig['symbol']}`\n*Outcome:* `{sig['status'].upper()}`"
    await send_tg(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)

async def send_tg(token, cid, msg):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        await session.post(url, json={"chat_id": cid, "text": msg, "parse_mode": "MarkdownV2"})

# -----------------------------
# Lifecycle
# -----------------------------
app = FastAPI()
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
ml_mgr = MLModelManager(cfg.ml_model_path)
exchange = ccxt.kucoinfutures({"enableRateLimit": True, "options": {"defaultType": "future"}})
generator = SignalGenerator(cfg, store, ml_mgr, exchange)

async def background_monitor():
    while True:
        try:
            df_btc = await generator.fetch_candles("BTC/USDT:USDT", cfg.higher_timeframe)
            btc_bullish = float(df_btc['close'].iloc[-1]) > float(df_btc['close'].ewm(span=200).mean().iloc[-1]) if not df_btc.empty else True
            await asyncio.gather(*(generator.generate_signal(s, btc_bullish) for s in cfg.symbols), return_exceptions=True)
            for s in cfg.symbols:
                df = await generator.fetch_candles(s, cfg.timeframe)
                if not df.empty: await store.update_signal_status(s, df['close'].iloc[-1])
        except Exception as e: logger.error(f"Monitor error: {e}")
        await asyncio.sleep(cfg.poll_interval)

@app.on_event("startup")
async def startup():
    await store.init_db()
    await exchange.load_markets()
    for s in cfg.symbols: ml_mgr.load_model(s)
    asyncio.create_task(background_monitor())
    asyncio.create_task(train_models_periodically())
    asyncio.create_task(heartbeat_loop(generator, cfg))
    asyncio.create_task(telegram_polling_loop(generator, cfg))
    logger.info("SignalBotAI Online: Filters, Heartbeat (4h), and /status Active.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
