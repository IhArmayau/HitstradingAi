from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import aiosqlite
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import numpy as np
import aiohttp
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
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
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "1h")
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
    simulate_execution: bool = os.getenv("SIMULATE_EXECUTION", "true").lower() in ("1", "true", "yes")
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")


# -----------------------------
# Database / SignalStore
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

    async def fetch_recent(self, limit: int = 50) -> List[Dict]:
        async with self.conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def update_signal_status(self, symbol: str, last_price: float):
        async with self.conn.execute(
            "SELECT id, signal, sl, tp, status FROM signals WHERE symbol=? AND status='open'", (symbol,)
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                signal_id, signal_type, sl, tp, status = row
                new_status = None
                if signal_type == "BUY":
                    if last_price >= tp:
                        new_status = 'take_profit'
                    elif last_price <= sl:
                        new_status = 'stop_loss'
                else:
                    if last_price <= tp:
                        new_status = 'take_profit'
                    elif last_price >= sl:
                        new_status = 'stop_loss'
                if new_status:
                    await self.conn.execute(
                        "UPDATE signals SET status=? WHERE id=?", (new_status, signal_id)
                    )
                    # Notify Telegram about status change
                    async with self.conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)) as c2:
                        full_signal = await c2.fetchone()
                        if full_signal:
                            columns = [column[0] for column in c2.description]
                            signal_dict = dict(zip(columns, full_signal))
                            await notify_signal_update(signal_dict)
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
# Helpers
# -----------------------------
def rr_bar(rr_value: float, max_rr: float = 5.0, length: int = 10) -> str:
    blocks = min(max(int((rr_value / max_rr) * length), 0), length)
    return "🟩" * blocks + "⬜" * (length - blocks)

def fmt_price(x: float) -> str:
    if x is None:
        return "0"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}"

def escape_telegram_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!\\"
    return ''.join(f"\\{c}" if c in escape_chars else c for c in text)

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
        except Exception as e:
            logger.exception("Telegram error: %s", e)


async def notify_signal_update(signal: dict):
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        msg = (
            f"*🔄 Signal Update*\n"
            f"*Pair:* `{escape_telegram_markdown(signal['symbol'])}`\n"
            f"*Signal:* {'🟢 BUY' if signal['signal']=='BUY' else '🔴 SELL'}\n"
            f"*Entry:* `{fmt_price(signal['entry'])}`\n"
            f"*SL:* `{fmt_price(signal['sl'])}` | *TP:* `{fmt_price(signal['tp'])}`\n"
            f"*R:R:* `{signal['rr']:.2f} {rr_bar(signal['rr'])}`\n"
            f"*Confidence:* `{signal['confidence']:.2f}%`\n"
            f"*Status:* `{signal.get('status', 'open')}`\n"
            f"────────────────────────"
        )
        await send_telegram_message(cfg.telegram_bot_token, cfg.telegram_chat_id, msg)


# -----------------------------
# Feature list
# -----------------------------
FEATURE_LIST = [
    'ema_short', 'ema_medium', 'ema_long', 'rsi', 'atr', 'adx', 'bb_trend',
    'vol_ok', 'htf_trend', 'spread_pct', 'ob_imbalance', 'bid_depth', 'ask_depth',
    'funding_rate', 'open_interest', 'onchain_flow',
]


# -----------------------------
# Add Indicators
# -----------------------------
def add_indicators(df: pd.DataFrame, ind_cfg: IndicatorsConfig, df_htf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['ema_short'] = df['close'].ewm(span=ind_cfg.ema_short, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=ind_cfg.ema_medium, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind_cfg.ema_long, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind_cfg.rsi_period).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind_cfg.atr_period).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind_cfg.atr_period).adx()
    bb = ta.volatility.BollingerBands(df['close'], ind_cfg.bb_period, ind_cfg.bb_std)
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)
    df['vol_avg'] = df['volume'].rolling(20, min_periods=1).mean().fillna(df['volume'].mean())
    df['vol_ok'] = (df['volume'] > df['vol_avg']).astype(int)
    if df_htf is not None and not df_htf.empty:
        df['htf_trend'] = np.where(df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1], 1, -1)
    else:
        df['htf_trend'] = 0
    return df.reset_index(drop=True)

# -----------------------------
# ML Model Manager
# -----------------------------
class MLModelManager:
    def __init__(self, model_path: str, feature_list: List[str], model_version: str = "v2"):
        self.model_path = model_path
        self.feature_list = feature_list
        self.model_version = model_version
        self.models: Dict[str, Any] = {}
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

    def scale_features(self, symbol: str, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if symbol not in self.scalers or fit:
            self.scalers[symbol] = StandardScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
        else:
            X_scaled = self.scalers[symbol].transform(X)
        return X_scaled
# -----------------------------
# Signal Generator
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
            logger.warning("Failed to fetch candles for %s: %s", symbol, e)
            return pd.DataFrame()

    async def generate_signal(self, symbol: str):
        async with self.gen_semaphore:
            lock = self.store.get_symbol_lock(symbol)
            async with lock:
                df_latest = await self.fetch_candles(symbol, self.cfg.timeframe)
                df_htf = await self.fetch_candles(symbol, self.cfg.higher_timeframe)
                df = add_indicators(df_latest, self.cfg.indicators, df_htf)
                if df.empty:
                    return

                last = df.iloc[-1]

                if await self.store.has_open_signal(symbol):
                    return

                # EMA crossover signal
                signal_type = None
                if last['ema_short'] > last['ema_medium']:
                    signal_type = "BUY"
                elif last['ema_short'] < last['ema_medium']:
                    signal_type = "SELL"
                if not signal_type:
                    return

                entry = float(last['close'])
                atr = float(last['atr'] or 0)
                if atr <= 0:
                    return

                sl = entry - atr * self.cfg.indicators.atr_sl_mult if signal_type == "BUY" else entry + atr * self.cfg.indicators.atr_sl_mult
                tp = entry + atr * self.cfg.indicators.atr_tp_mult if signal_type == "BUY" else entry - atr * self.cfg.indicators.atr_tp_mult
                rr = abs((tp - entry) / abs(entry - sl)) if entry != sl else 0

                pred_prob = 0.75
                if symbol in self.ml_mgr.models:
                    try:
                        missing_features = [f for f in FEATURE_LIST if f not in df.columns]
                        if not missing_features:
                            features = df[FEATURE_LIST].iloc[-1].values.reshape(1, -1)
                            X_scaled = self.ml_mgr.scale_features(symbol, features)
                            pred_prob = float(self.ml_mgr.models[symbol].predict_proba(X_scaled)[0][1])
                    except Exception:
                        pred_prob = 0.75

                sig = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "signal": signal_type,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "confidence": pred_prob * 100,
                    "rr": rr,
                    "pred_prob": pred_prob,
                    "model_version": self.cfg.model_version,
                    "executed_price": entry,
                }

                await self.store.insert_signal(sig)

                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    msg = (
                        f"*📊 New Signal from SignalBotAI*\n"
                        f"*Pair:* `{escape_telegram_markdown(sig['symbol'])}`\n"
                        f"*Signal:* {'🟢 BUY' if signal_type=='BUY' else '🔴 SELL'}\n"
                        f"*Entry:* `{fmt_price(sig['entry'])}`\n"
                        f"*SL:* `{fmt_price(sig['sl'])}` | *TP:* `{fmt_price(sig['tp'])}`\n"
                        f"*R:R:* `{sig['rr']:.2f} {rr_bar(sig['rr'])}`\n"
                        f"*Confidence:* `{sig['confidence']:.2f}%`\n"
                        f"────────────────────────"
                    )
                    await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)


# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
ml_mgr = MLModelManager(cfg.ml_model_path, FEATURE_LIST)
exchange = ccxt.kucoinfutures({"enableRateLimit": True})
generator = SignalGenerator(cfg, store, ml_mgr, exchange)


# -----------------------------
# Background Monitor
# -----------------------------
async def background_monitor():
    fetch_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)

    async def safe_fetch_candles(symbol: str, timeframe: str) -> pd.DataFrame:
        async with fetch_semaphore:
            try:
                df = await generator.fetch_candles(symbol, timeframe)
                if df.empty:
                    logger.warning("No candles fetched for %s", symbol)
                return df
            except Exception as e:
                logger.warning("Failed to fetch candles for %s: %s", symbol, e)
                return pd.DataFrame()

    while True:
        try:
            # Generate signals
            await asyncio.gather(*(generator.generate_signal(sym) for sym in cfg.symbols), return_exceptions=True)

            # Fetch latest candles
            fetch_tasks = {sym: asyncio.create_task(safe_fetch_candles(sym, cfg.timeframe)) for sym in cfg.symbols}
            results = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)

            # Update signal statuses
            update_tasks = []
            for idx, symbol in enumerate(fetch_tasks.keys()):
                df_latest = results[idx]
                if isinstance(df_latest, pd.DataFrame) and not df_latest.empty:
                    last_price = df_latest['close'].iloc[-1]
                    update_tasks.append(store.update_signal_status(symbol, last_price))
            await asyncio.gather(*update_tasks, return_exceptions=True)

        except Exception:
            logger.exception("Unexpected error in background monitor loop")

        await asyncio.sleep(cfg.poll_interval)


# -----------------------------
# Startup / Shutdown Events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    await store.init_db()
    await exchange.load_markets()
    for symbol in cfg.symbols:
        ml_mgr.load_model(symbol)
    asyncio.create_task(background_monitor())
    logger.info("SignalBotAI started successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    await store.close()
    await exchange.close()
    logger.info("SignalBotAI shutdown complete.")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"status": "alive", "model_version": cfg.model_version}


@app.get("/signals")
async def get_signals(limit: int = 50):
    rows = await store.fetch_recent(limit=limit)
    return {"count": len(rows), "signals": rows}


# -----------------------------
# Run Uvicorn
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Adjust if your script filename is different
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        reload=True
    )
