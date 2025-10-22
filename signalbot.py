#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import pandas as pd
import ta
import aiosqlite
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone
import numpy as np
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Response, HTTPException
import uvicorn
import ccxt.async_support as ccxt

# TensorFlow / LSTM
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

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
    entry_timeframe: str = os.getenv("ENTRY_TIMEFRAME", "5m")
    confirm_timeframe1: str = os.getenv("CONFIRM_TIMEFRAME1", "1h")
    confirm_timeframe2: str = os.getenv("CONFIRM_TIMEFRAME2", "4h")
    limit: int = int(os.getenv("LIMIT", 500))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 50.0))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    model_version: str = os.getenv("MODEL_VERSION", "v2")
    expected_slippage_pct: float = float(os.getenv("EXPECTED_SLIPPAGE_PCT", "0.0005"))
    retrain_interval_minutes: int = int(os.getenv("RETRAIN_INTERVAL_MINUTES", 60))
    min_samples_for_retrain: int = int(os.getenv("MIN_SAMPLES_FOR_RETRAIN", 500))
    simulate_execution: bool = os.getenv("SIMULATE_EXECUTION", "true").lower() in ("1", "true", "yes")

# -----------------------------
# SQLite Database
# -----------------------------
class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()  # ensure single DB write at a time

    async def init_db(self):
        if self.conn:
            return
        db_dir = os.path.dirname(self.db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
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
        await self.conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    async def has_open_signal(self, symbol: str) -> bool:
        if not self.conn:
            await self.init_db()
        async with self._lock:
            async with self.conn.execute(
                "SELECT 1 FROM signals WHERE symbol=? AND status='open' LIMIT 1", (symbol,)
            ) as cursor:
                row = await cursor.fetchone()
                return row is not None

    async def insert_signal(self, signal_data: dict):
        if not self.conn:
            await self.init_db()
        async with self._lock:
            try:
                keys = ", ".join(signal_data.keys())
                placeholders = ", ".join("?" * len(signal_data))
                values = list(signal_data.values())
                await self.conn.execute(f"INSERT INTO signals ({keys}) VALUES ({placeholders})", values)
                await self.conn.commit()
                logger.debug("Signal inserted: %s", signal_data)
            except Exception as e:
                logger.exception("Failed to insert signal: %s", e)

# -----------------------------
# Helper functions
# -----------------------------
def rr_bar(rr_value: float, max_rr: float = 5.0, length: int = 10) -> str:
    try:
        blocks = min(int((rr_value / max_rr) * length), length)
    except Exception:
        blocks = 0
    return "🟩" * max(0, blocks) + "⬜" * (length - max(0, blocks))

def fmt_price(x: float) -> str:
    if x is None:
        return "0"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}"

def format_signal_message(signals: List[Dict]) -> str:
    if not signals:
        return "*No new signals.*"
    lines = ["*📊 SignalBotAI New Signals*"]
    for s in signals:
        icon = "🟢" if s['signal'] == "BUY" else "🔴"
        rr_visual = rr_bar(s.get("rr", 0.0))
        lines.append(f"{icon} *{s['signal']}* `{s['symbol']}`")
        lines.append(f"• Entry: `{fmt_price(s['entry'])}`  SL: `{fmt_price(s['sl'])}`  TP: `{fmt_price(s['tp'])}`")
        lines.append(f"• R:R: `{s.get('rr',0):.2f}` {rr_visual}  Confidence: `{s['confidence']:.2f}%`")
        lines.append("────────")
    return "\n".join(lines)

async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    import aiohttp
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
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
# Indicators & Features
# -----------------------------
FEATURE_LIST = ["ema_short", "ema_medium", "ema_long", "rsi", "atr", "bb_upper", "bb_lower"]

def add_indicators(df: pd.DataFrame, cfg: IndicatorsConfig, df_htf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        logger.warning("add_indicators: received empty dataframe")
        return df
    try:
        # EMA
        df['ema_short'] = df['close'].ewm(span=cfg.ema_short, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=cfg.ema_medium, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=cfg.ema_long, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(cfg.rsi_period).mean()
        avg_loss = loss.rolling(cfg.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(cfg.atr_period).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(cfg.bb_period).mean()
        df['bb_std'] = df['close'].rolling(cfg.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + cfg.bb_std * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - cfg.bb_std * df['bb_std']

        # Drop NaNs to ensure signal generation
        df = df.dropna(subset=FEATURE_LIST)

        return df
    except Exception as e:
        logger.exception("add_indicators failed: %s", e)
        return df
# -----------------------------
# SignalBot class
# -----------------------------
class SignalBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.store = SignalStore(cfg.sqlite_db)
        self.gen_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.running_event = asyncio.Event()
        self.running_event.set()
        self._bg_tasks: List[asyncio.Task] = []
        # KuCoin Perpetual Futures
        self.exchange = ccxt.kucoinfutures({
            "enableRateLimit": True,
            "options": {"defaultType": "future"}  # perpetual futures
        })

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.exception("Failed to fetch OHLCV for %s: %s", symbol, e)
            return pd.DataFrame()

    async def generate_signal(self, symbol: str, df: Optional[pd.DataFrame] = None):
        async with self.gen_semaphore:
            try:
                # Check for existing open signal
                if await self.store.has_open_signal(symbol):
                    logger.info("Open signal exists for %s, skipping generation", symbol)
                    return None

                if df is None:
                    df = await self.fetch_ohlcv(symbol, self.cfg.entry_timeframe, self.cfg.limit)
                if df.empty:
                    logger.debug("No OHLCV data for %s", symbol)
                    return None

                df_entry = add_indicators(df, self.cfg.indicators)
                if df_entry.empty:
                    logger.debug("Indicators could not be computed for %s", symbol)
                    return None

                last_entry = df_entry.iloc[-1]
                logger.debug("Last entry row for %s: %s", symbol, last_entry.to_dict())

                if last_entry[FEATURE_LIST].isna().any():
                    logger.debug("NaNs present in last entry for %s", symbol)
                    return None

                # Entry timeframe signal
                signal_type = None
                if last_entry['ema_short'] > last_entry['ema_medium'] > last_entry['ema_long']:
                    signal_type = "BUY"
                elif last_entry['ema_short'] < last_entry['ema_medium'] < last_entry['ema_long']:
                    signal_type = "SELL"
                else:
                    logger.debug("Trend unclear for %s", symbol)
                    return None

                # Multi-timeframe confirmation
                higher_timeframes = [self.cfg.confirm_timeframe1, self.cfg.confirm_timeframe2]
                for tf in higher_timeframes:
                    df_htf = await self.fetch_ohlcv(symbol, tf, limit=max(200, self.cfg.indicators.ema_long*2))
                    if df_htf.empty:
                        continue
                    df_htf = add_indicators(df_htf, self.cfg.indicators)
                    last_htf = df_htf.iloc[-1]
                    if last_htf[FEATURE_LIST].isna().any():
                        continue

                    if signal_type == "BUY" and not (last_htf['ema_short'] > last_htf['ema_medium'] > last_htf['ema_long']):
                        logger.debug("HTF trend mismatch for %s on %s", symbol, tf)
                        return None
                    elif signal_type == "SELL" and not (last_htf['ema_short'] < last_htf['ema_medium'] < last_htf['ema_long']):
                        logger.debug("HTF trend mismatch for %s on %s", symbol, tf)
                        return None

                # ATR-based SL/TP
                if signal_type == "BUY":
                    sl = last_entry['close'] - last_entry['atr'] * self.cfg.indicators.atr_sl_mult
                    tp = last_entry['close'] + last_entry['atr'] * self.cfg.indicators.atr_tp_mult
                else:
                    sl = last_entry['close'] + last_entry['atr'] * self.cfg.indicators.atr_sl_mult
                    tp = last_entry['close'] - last_entry['atr'] * self.cfg.indicators.atr_tp_mult

                rr = abs(tp - last_entry['close']) / max(abs(last_entry['close'] - sl), 1e-6)

                # -----------------------------
                # TensorFlow ML model prediction
                # -----------------------------
                confidence = 100.0  # default
                model_path = os.path.join(self.cfg.ml_model_path, "lstm_model.h5")
                if TF_AVAILABLE and os.path.exists(model_path):
                    try:
                        model = tf.keras.models.load_model(model_path)
                        X = df_entry[FEATURE_LIST].values[-50:]  # last 50 rows
                        X = np.expand_dims(X, axis=0)            # batch dimension
                        pred = model.predict(X)
                        confidence = float(pred[0][0] * 100)
                        logger.debug("TensorFlow prediction for %s: %.2f%%", symbol, confidence)
                    except Exception as e:
                        logger.exception("TensorFlow model failed for %s: %s", symbol, e)

                signal_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "signal": signal_type,
                    "entry": last_entry['close'],
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                    "rr": rr,
                    "pred_prob": confidence / 100.0,
                    "model_version": self.cfg.model_version,
                }

                await self.store.insert_signal(signal_data)

                # Send Telegram message if configured
                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    msg = format_signal_message([signal_data])
                    await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

                logger.info("Signal generated for %s: %s", symbol, signal_type)
                return signal_data

            except Exception as e:
                logger.exception("Error generating signal for %s: %s", symbol, e)
                return None
# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()
bot_cfg = BotConfig()
bot = SignalBot(bot_cfg)

@app.on_event("startup")
async def startup_event():
    await bot.store.init_db()

@app.get("/signals")
async def get_signals():
    if not bot.store.conn:
        await bot.store.init_db()
    async with bot.store.conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 50") as cursor:
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

@app.get("/generate/{symbol}")
async def generate(symbol: str):
    try:
        sig = await bot.generate_signal(symbol)
        if sig:
            return sig
        return {"status": "No signal generated (maybe an open signal exists or trend unclear)"}
    except Exception as e:
        logger.exception("Failed to generate signal for %s", symbol)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_all")
async def generate_all():
    tasks = [bot.generate_signal(symbol) for symbol in bot_cfg.symbols]
    if tasks:
        signals = await asyncio.gather(*tasks)
        results = [s for s in signals if s is not None]
    else:
        results = []

    if not results:
        return {"status": "No signals generated (all symbols may have open signals or trends unclear)"}
    return results

@app.get("/heartbeat")
async def heartbeat():
    status = "running" if bot.running_event.is_set() else "stopped"
    return {"status": status, "timestamp": datetime.now(timezone.utc).isoformat()}

# -----------------------------
# Run FastAPI / Uvicorn
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
