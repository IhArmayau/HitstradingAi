from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import aiosqlite
import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import aiohttp
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Optional TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
    print("TensorFlow available")
except Exception:
    TF_AVAILABLE = False
    print("TensorFlow not available")

# Load environment variables
load_dotenv()

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SignalBotAI")

# -----------------------------
# Configs
# -----------------------------
@dataclass
class IndicatorsConfig:
    ema_short: int = int(os.getenv("EMA_SHORT", 9))
    ema_medium: int = int(os.getenv("EMA_MEDIUM", 21))
    ema_long: int = int(os.getenv("EMA_LONG", 50))
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
    timeframe: str = os.getenv("TIMEFRAME", "5m")  # Entry TF
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "1h")  # Confirmation 1H
    htf_4h: str = os.getenv("HTF_4H", "4h")  # Confirmation 4H
    limit: int = int(os.getenv("LIMIT", 1000))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 70.0))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v2")
    expected_slippage_pct: float = float(os.getenv("EXPECTED_SLIPPAGE_PCT", "0.0005"))
    retrain_interval_minutes: int = int(os.getenv("RETRAIN_INTERVAL_MINUTES", 60))
    min_samples_for_retrain: int = int(os.getenv("MIN_SAMPLES_FOR_RETRAIN", 500))
    onchain_api_url: Optional[str] = os.getenv("ONCHAIN_API_URL")
    simulate_execution: bool = os.getenv("SIMULATE_EXECUTION", "true").lower() in ("1", "true", "yes")

# -----------------------------
# Database (signals + training dataset + performance)
# -----------------------------
class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self.symbol_locks: Dict[str, asyncio.Lock] = {}

    async def init_db(self):
        if self.conn:
            return
        db_dir = os.path.dirname(self.db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("PRAGMA synchronous=NORMAL;")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                confidence REAL,
                rr REAL,
                outcome TEXT,
                status TEXT DEFAULT 'open',
                pred_prob REAL,
                model_version TEXT,
                executed_price REAL
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                features TEXT,
                label INTEGER,
                used_in_model INTEGER DEFAULT 0
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS perf (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT,
                value REAL,
                ts TEXT
            )
        """)
        await self.conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    async def has_open_signal(self, symbol: str) -> bool:
        async with self.conn.execute(
            "SELECT 1 FROM signals WHERE symbol=? AND status='open' LIMIT 1", (symbol,)
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def insert_signal(self, sig: dict):
        await self.conn.execute("""
            INSERT INTO signals(timestamp, symbol, signal, entry, sl, tp, confidence, rr, pred_prob, model_version, executed_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig['timestamp'], sig['symbol'], sig['signal'], sig['entry'], sig['sl'], sig['tp'],
            sig['confidence'], sig['rr'], sig['pred_prob'], sig['model_version'], sig['executed_price']
        ))
        await self.conn.commit()
        logger.debug("Inserted signal for %s: %s", sig['symbol'], sig['signal'])

    async def update_signal_status(self, symbol: str, last_price: float):
        async with self.conn.execute(
            "SELECT id, sl, tp, status FROM signals WHERE symbol=? AND status='open'", (symbol,)
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                signal_id, sl, tp, status = row
                new_status = None
                if last_price >= tp:
                    new_status = 'take_profit'
                elif last_price <= sl:
                    new_status = 'stop_loss'
                if new_status:
                    await self.conn.execute(
                        "UPDATE signals SET status=? WHERE id=?", (new_status, signal_id)
                    )
                    logger.debug("Signal %s for %s updated to %s", signal_id, symbol, new_status)
        await self.conn.commit()

    async def fetch_recent(self, limit: int = 50):
        async with self.conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ) as cursor:
            return [dict(zip([c[0] for c in cursor.description], row)) async for row in cursor]

    async def fetch_training_samples(self, min_samples: int):
        async with self.conn.execute(
            "SELECT id, symbol, features, label FROM training WHERE used_in_model=0 ORDER BY timestamp ASC LIMIT ?", (min_samples,)
        ) as cursor:
            return [dict(zip([c[0] for c in cursor.description], row)) async for row in cursor]

    async def mark_training_used(self, ids: List[int]):
        if not ids:
            return
        placeholders = ','.join(['?']*len(ids))
        await self.conn.execute(
            f"UPDATE training SET used_in_model=1 WHERE id IN ({placeholders})", ids
        )
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = asyncio.Lock()
        return self.symbol_locks[symbol]

# -----------------------------
# Helpers + Messaging
# -----------------------------
def rr_bar(rr_value: float, max_rr: float = 5.0, length: int = 10) -> str:
    try:
        blocks = min(max(int((rr_value / max_rr) * length), 0), length)
    except Exception:
        blocks = 0
    return "🟩" * max(0, blocks) + "⬜" * (length - max(0, blocks))

def fmt_price(x: float) -> str:
    if x is None:
        return "0"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}"

def escape_telegram_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!\\"
    return ''.join(f"\\{c}" if c in escape_chars else c for c in text)

def format_signal_message(signals: List[Dict]) -> str:
    if not signals:
        return "*No new signals.*"
    lines = ["*📊 SignalBotAI New Signals*"]
    for s in signals:
        icon = "🟢" if s['signal'] == "BUY" else "🔴"
        rr_visual = rr_bar(s.get("rr", 0.0))
        lines.append(f"{icon} *{s['signal']}* `{escape_telegram_markdown(s['symbol'])}`")
        lines.append(f"• Entry: `{fmt_price(s['entry'])}`  SL: `{fmt_price(s['sl'])}`  TP: `{fmt_price(s['tp'])}`")
        lines.append(f"• R:R: `{s.get('rr', 0):.2f}` {rr_visual}  Confidence: `{s['confidence']:.2f}%`")
        lines.append("━━━━━━━━━━━━━━")
    return "\n".join(lines)

async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.warning("Telegram send failed %s %s", resp.status, text)
                else:
                    logger.debug("Telegram send OK")
        except Exception as e:
            logger.exception("Telegram error: %s", e)

# -----------------------------
# Feature Engineering
# -----------------------------
FEATURE_LIST = [
    'ema_short', 'ema_medium', 'ema_long', 'rsi', 'atr', 'adx', 'bb_trend',
    'vol_ok', 'htf_trend', 'spread_pct', 'ob_imbalance', 'bid_depth', 'ask_depth',
    'funding_rate', 'open_interest', 'onchain_flow',
]

def add_indicators(df: pd.DataFrame, ind_cfg: IndicatorsConfig, df_htf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # EMA
    df['ema_short'] = df['close'].ewm(span=ind_cfg.ema_short, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=ind_cfg.ema_medium, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind_cfg.ema_long, adjust=False).mean()

    # RSI, ATR, ADX
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind_cfg.rsi_period).rsi()
    except Exception:
        df['rsi'] = np.nan

    try:
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind_cfg.atr_period).average_true_range()
    except Exception:
        df['atr'] = np.nan

    try:
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind_cfg.atr_period).adx()
    except Exception:
        df['adx'] = np.nan

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(df['close'], ind_cfg.bb_period, ind_cfg.bb_std)
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)
    except Exception:
        df['bb_trend'] = 0

    # Volume check
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = (df['volume'] > df['vol_avg']).astype(int)

    # Higher timeframe trend
    if df_htf is not None and not df_htf.empty:
        try:
            df['htf_trend'] = np.where(df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1], 1, -1)
        except Exception:
            df['htf_trend'] = 0
    else:
        df['htf_trend'] = 0
        logger.debug("Higher timeframe data missing. HTF trend set to 0.")

    return df.dropna().reset_index(drop=True)

# -----------------------------
# ML / Model Utilities
# -----------------------------
class MLModelManager:
    def __init__(self, model_path: str, feature_list: List[str], model_version: str = "v2"):
        self.model_path = model_path
        self.feature_list = feature_list
        self.model_version = model_version
        self.models: Dict[str, Any] = {}  # key = symbol
        self.scalers: Dict[str, StandardScaler] = {}

    def _get_model_file(self, symbol: str) -> str:
        os.makedirs(self.model_path, exist_ok=True)
        return os.path.join(self.model_path, f"{symbol.replace('/', '_')}_{self.model_version}.pkl")

    def _get_scaler_file(self, symbol: str) -> str:
        os.makedirs(self.model_path, exist_ok=True)
        return os.path.join(self.model_path, f"{symbol.replace('/', '_')}_{self.model_version}_scaler.pkl")

    def load_model(self, symbol: str) -> bool:
        model_file = self._get_model_file(symbol)
        scaler_file = self._get_scaler_file(symbol)
        try:
            if os.path.exists(model_file):
                self.models[symbol] = joblib.load(model_file)
            if os.path.exists(scaler_file):
                self.scalers[symbol] = joblib.load(scaler_file)
            return symbol in self.models
        except Exception as e:
            logger.warning("Failed to load model for %s: %s", symbol, e)
            return False

    def save_model(self, symbol: str):
        try:
            model_file = self._get_model_file(symbol)
            scaler_file = self._get_scaler_file(symbol)
            if symbol in self.models:
                joblib.dump(self.models[symbol], model_file)
            if symbol in self.scalers:
                joblib.dump(self.scalers[symbol], scaler_file)
            logger.info("Saved model & scaler for %s", symbol)
        except Exception as e:
            logger.error("Failed to save model for %s: %s", symbol, e)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        df_f = df.copy()
        features = df_f[self.feature_list].fillna(0)
        return features.values

    def scale_features(self, symbol: str, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if symbol not in self.scalers or fit:
            self.scalers[symbol] = StandardScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
        else:
            X_scaled = self.scalers[symbol].transform(X)
        return X_scaled

    def predict_proba(self, symbol: str, X: np.ndarray) -> np.ndarray:
        if symbol not in self.models:
            logger.debug("No model loaded for %s, returning 50/50 probability", symbol)
            return np.array([[0.5, 0.5]])
        model = self.models[symbol]
        try:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            elif TF_AVAILABLE and isinstance(model, tf.keras.Model):
                proba = model.predict(X, verbose=0)
                if proba.shape[1] == 1:
                    return np.hstack([1 - proba, proba])
                return proba
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", symbol, e)
        return np.array([[0.5, 0.5]])

    async def retrain_models_if_needed(self, store: SignalStore, cfg: BotConfig):
        for symbol in cfg.symbols:
            samples = await store.fetch_training_samples(cfg.min_samples_for_retrain)
            if len(samples) < cfg.min_samples_for_retrain:
                continue

            X = []
            y = []
            ids = []

            for s in samples:
                try:
                    feats = json.loads(s['features'])
                    X.append([feats.get(f, 0) for f in FEATURE_LIST])
                    y.append(int(s['label']))
                    ids.append(s['id'])
                except Exception:
                    continue

            if not X:
                continue

            X = np.array(X)
            y = np.array(y)

            X_scaled = self.scale_features(symbol, X, fit=True)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models[symbol] = model
            self.save_model(symbol)
            await store.mark_training_used(ids)
            logger.info("Retrained model for %s using %d samples", symbol, len(ids))

# -----------------------------
# Signal Generation
# -----------------------------
class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ml_mgr: MLModelManager, exchange: ccxt.kucoinfutures):
        self.cfg = cfg
        self.store = store
        self.ml_mgr = ml_mgr
        self.exchange = exchange
        self.gen_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)

    async def fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=self.cfg.limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.warning("Failed to fetch candles %s %s: %s", symbol, timeframe, e)
            return pd.DataFrame()

    async def fetch_orderbook_features(self, symbol: str) -> dict:
        # Returns order book imbalance, bid/ask depth, spread percentage
        return {'spread_pct': 0.0, 'ob_imbalance': 0.0, 'bid_depth': 0.0, 'ask_depth': 0.0}

    async def fetch_funding_and_oi(self, symbol: str) -> tuple[float, float]:
        """Fetch real funding rate and open interest using CCXT"""
        try:
            info = await self.exchange.futures_get_funding_rate({'symbol': symbol.replace('/', '')})
            funding_rate = float(info[0]['fundingRate']) if info else 0.0
        except Exception:
            funding_rate = 0.0
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            open_interest = float(ticker.get('openInterest', 0.0))
        except Exception:
            open_interest = 0.0
        return funding_rate, open_interest

    async def fetch_onchain_flow(self, symbol: str) -> float:
        # Placeholder if using onchain metrics
        return 0.0

    def simulate_execution_price(self, entry: float, ob_feats: dict, signal_type: str) -> float:
        slippage = entry * self.cfg.expected_slippage_pct
        return entry + slippage if signal_type == "BUY" else entry - slippage

    async def generate_signal(self, symbol: str):
        async with self.gen_semaphore:
            lock = self.store.get_symbol_lock(symbol)
            async with lock:
                # Update open signals with latest price
                df_latest = await self.fetch_candles(symbol, self.cfg.timeframe)
                if not df_latest.empty:
                    last_price = df_latest['close'].iloc[-1]
                    await self.store.update_signal_status(symbol, last_price)

                # Skip if open signal exists
                if await self.store.has_open_signal(symbol):
                    logger.debug("Open signal exists for %s. Skipping.", symbol)
                    return

                # Fetch higher TF for confirmation
                df_1h = await self.fetch_candles(symbol, self.cfg.higher_timeframe)
                df_4h = await self.fetch_candles(symbol, self.cfg.htf_4h)

                df = add_indicators(df_latest, self.cfg.indicators, df_1h)
                if df.empty:
                    return
                last = df.iloc[-1]

                confirm_1h = df_1h['close'].iloc[-1] > df_1h['open'].iloc[-1] if not df_1h.empty else True
                confirm_4h = df_4h['close'].iloc[-1] > df_4h['open'].iloc[-1] if not df_4h.empty else True

                signal_type = None
                if last['close'] > last['ema_short'] > last['ema_medium'] and confirm_1h and confirm_4h:
                    signal_type = "BUY"
                elif last['close'] < last['ema_short'] < last['ema_medium'] and not confirm_1h and not confirm_4h:
                    signal_type = "SELL"

                if not signal_type:
                    return

                # Fetch additional features
                ob_feats = await self.fetch_orderbook_features(symbol)
                funding, oi = await self.fetch_funding_and_oi(symbol)
                onchain = await self.fetch_onchain_flow(symbol)

                last_features = {f: last.get(f, 0) for f in FEATURE_LIST}
                last_features.update({
                    'spread_pct': ob_feats.get('spread_pct', 0),
                    'ob_imbalance': ob_feats.get('ob_imbalance', 0),
                    'bid_depth': ob_feats.get('bid_depth', 0),
                    'ask_depth': ob_feats.get('ask_depth', 0),
                    'funding_rate': funding,
                    'open_interest': oi,
                    'onchain_flow': onchain
                })

                X_input = np.array([[last_features[f] for f in FEATURE_LIST]])
                X_scaled = self.ml_mgr.scale_features(symbol, X_input)
                prob = self.ml_mgr.predict_proba(symbol, X_scaled)[0][1]
                confidence = prob * 100
                if confidence < self.cfg.confidence_threshold:
                    return

                entry = float(last['close'])
                atr = float(last['atr'] or 0)
                if atr <= 0:
                    return

                entry_exec = self.simulate_execution_price(entry, ob_feats, signal_type) \
                    if self.cfg.simulate_execution else entry

                if signal_type == "BUY":
                    sl = entry_exec - atr * self.cfg.indicators.atr_sl_mult
                    tp = entry_exec + atr * self.cfg.indicators.atr_tp_mult
                else:
                    sl = entry_exec + atr * self.cfg.indicators.atr_sl_mult
                    tp = entry_exec - atr * self.cfg.indicators.atr_tp_mult

                rr = abs((tp - entry_exec) / abs(entry_exec - sl)) if entry_exec != sl else 0

                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "signal": signal_type,
                    "entry": entry_exec,
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                    "rr": rr,
                    "pred_prob": prob,
                    "model_version": self.cfg.model_version,
                    "executed_price": entry_exec,
                }

                await self.store.insert_signal(sig)
                logger.info("Generated signal %s %s conf=%.2f%% rr=%.2f", symbol, signal_type, confidence, rr)

                # Send to Telegram
                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    msg = f"""*📊 New Signal from SignalBotAI*
*Pair:* `{escape_telegram_markdown(sig['symbol'])}`
*Signal:* {'🟢 BUY' if signal_type=='BUY' else '🔴 SELL'}
*Entry:* `{fmt_price(sig['entry'])}`
*SL:* `{fmt_price(sig['sl'])}` | *TP:* `{fmt_price(sig['tp'])}`
*R:R:* `{sig['rr']:.2f}` {rr_bar(sig['rr'], length=10)}
*Confidence:* `{sig['confidence']:.2f}%`
────────────────────────"""
                    await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

# -----------------------------
# FastAPI App & Heartbeat
# -----------------------------
app = FastAPI()
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
ml_mgr = MLModelManager(cfg.ml_model_path, FEATURE_LIST)
generator = SignalGenerator(cfg, store, ml_mgr, ccxt.kucoinfutures({"enableRateLimit": True}))

@app.on_event("startup")
async def startup_event():
    await store.init_db()
    await generator.exchange.load_markets()
    logger.info("App startup completed")
    asyncio.create_task(monitor_signals())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown initiated")
    try:
        await store.close()
        await generator.exchange.close()
    except Exception:
        logger.exception("Error during shutdown")
    logger.info("Shutdown complete")

@app.get("/")
async def root():
    return {"status": "alive", "model_version": cfg.model_version, "tensorflow": TF_AVAILABLE}

@app.get("/signals")
async def get_signals(limit: int = 50):
    try:
        rows = await store.fetch_recent(limit=limit)
        return {"count": len(rows), "signals": rows}
    except Exception:
        logger.exception("Failed to fetch signals")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

@app.post("/trigger")
async def trigger_symbol(symbol: str):
    try:
        await generator.generate_signal(symbol)
        return {"status": "triggered", "symbol": symbol}
    except Exception:
        logger.exception("Manual trigger failed for %s", symbol)
        raise HTTPException(status_code=500, detail="Trigger failed")

# Heartbeat endpoint (GET + HEAD)
@app.get("/heartbeat")
@app.head("/heartbeat")
async def heartbeat():
    return JSONResponse(content={"status": "ok", "message": "SignalBotAI is alive"})

# -----------------------------
# Background signal monitor
# -----------------------------
async def monitor_signals():
    while True:
        try:
            for symbol in cfg.symbols:
                await generator.generate_signal(symbol)
            await asyncio.sleep(cfg.poll_interval)
        except Exception:
            logger.exception("Error in background monitor loop")
            await asyncio.sleep(cfg.poll_interval)

# -----------------------------
# Run Uvicorn if executed directly
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
