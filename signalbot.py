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
from fastapi.staticfiles import StaticFiles
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import san
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
    max_funding_threshold: float = 0.05
    min_win_rate_threshold: float = 0.40
    signal_cooldown_minutes: int = 60

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
    model_version: str = os.getenv("MODEL_VERSION", "v5.2-prod")
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
        slug_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "BNB": "binance-coin", "ADA": "cardano", "XRP": "ripple",
            "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "polygon",
            "LTC": "litecoin", "LINK": "chainlink", "SHIB": "shiba-inu"
        }
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
            return {"score": score, "label": label, "funding": f_rate, "slug": slug}
        except Exception as e:
            logger.error(f"Santiment Error for {slug}: {e}")
            return {"score": 50, "label": "Neutral", "funding": 0.0, "slug": slug}

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
                query = """
                    SELECT entry, safety_score, sentiment_score, confidence, vol_liq_ratio, funding, status
                    FROM signals
                    WHERE status IN ('win', 'loss')
                """
                if symbol != "GLOBAL": query += f" AND symbol = '{symbol}'"
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    if len(rows) < 30: return
                    df = pd.DataFrame(rows, columns=['entry', 'safety_score', 'sentiment_score', 'confidence', 'vol_liq_ratio', 'funding', 'status'])
            X = df.drop('status', axis=1).fillna(0)
            y = df['status'].apply(lambda x: 1 if x == 'win' else 0)
            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
            model.fit(X, y)
            joblib.dump(model, os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_model.joblib"))
            logger.info(f"🧠 Model Retrained for {symbol}")
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
        full_url = f"{webhook_url.rstrip('/')}/webhook"
        try:
            async with self.session.get(self.webhook_api) as resp:
                webhooks = await resp.json()
                for wh in webhooks:
                    if wh['webhookURL'] == full_url: return
            payload = {"webhookURL": full_url, "transactionTypes": ["SWAP"], "accountAddresses": addresses, "webhookType": "enhanced"}
            await self.session.post(self.webhook_api, json=payload)
        except: pass

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
        rugged = score < 60 or not hr['is_safe'] or vl_ratio > 10.0
        return {"safety_score": score, "is_rugged": rugged, "helius_reason": hr['reason'], "vl_ratio": vl_ratio}

class SignalStore:
    def __init__(self, db_path: str):
        self.db_path, self.conn, self.symbol_locks = db_path, None, {}
    async def init_db(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, market_type TEXT, contract_address TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, confidence REAL, safety_score REAL, sentiment_score REAL, funding REAL DEFAULT 0.0, is_cluster INTEGER DEFAULT 0, status TEXT DEFAULT 'open', model_version TEXT, vol_liq_ratio REAL, time_to_close INTEGER)")
        await self.conn.commit()
    async def insert_signal(self, s: dict):
        await self.conn.execute("INSERT INTO signals(timestamp, symbol, market_type, contract_address, signal, entry, sl, tp, confidence, safety_score, sentiment_score, funding, is_cluster, model_version, vol_liq_ratio) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s['timestamp'], s['symbol'], s['market_type'], s.get('contract_address'), s['signal'], s['entry'], s.get('sl'), s.get('tp'), s['confidence'], s.get('safety_score', 0), s.get('sentiment_score', 50), s.get('funding', 0.0), s.get('is_cluster', 0), s.get('model_version'), s.get('vol_liq_ratio', 0.0)))
        await self.conn.commit()
    async def update_signal_status(self, sig_id: int, status: str, duration: int = 0):
        await self.conn.execute("UPDATE signals SET status = ?, time_to_close = ? WHERE id = ?", (status, duration, sig_id))
        await self.conn.commit()
    async def get_latest_signals(self, limit: int = 20):
        self.conn.row_factory = aiosqlite.Row
        async with self.conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)) as cursor:
            return await cursor.fetchall()
    async def has_open_signal(self, iden: str):
        async with self.conn.execute("SELECT 1 FROM signals WHERE (symbol=? OR contract_address=?) AND status='open'", (iden, iden)) as c: return await c.fetchone() is not None
    def get_symbol_lock(self, s):
        if s not in self.symbol_locks: self.symbol_locks[s] = asyncio.Lock()
        return self.symbol_locks[s]

    async def get_pnl_stats(self, timeframe_hours: int = None):
        query = "SELECT status, COUNT(*), SUM(CASE WHEN status='win' THEN 1 ELSE 0 END) FROM signals WHERE status IN ('win', 'loss')"
        params = []
        if timeframe_hours:
            since = (datetime.now(timezone.utc) - timedelta(hours=timeframe_hours)).isoformat()
            query += " AND timestamp > ?"
            params.append(since)

        async with self.conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if not row or row[1] == 0: return None
            total, wins = row[1], row[2]
            return {"total": total, "wins": wins, "losses": total - wins, "win_rate": round((wins/total)*100, 2)}

class SignalGenerator:
    def __init__(self, cfg: BotConfig, store: SignalStore, ex: ccxt.Exchange, sentinel: SocialSentinel, cluster: ClusterEngine, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.sentinel, self.cluster, self.session = cfg, store, ex, sentinel, cluster, session
        self.helius = HeliusEngine(HELIUS_API_KEY, session)
        self.dex = DexEngine(session)
        self.security = SecurityEngine(session)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.last_signal_time: Dict[str, datetime] = {}

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
        if not self.cfg.enable_cex or not self.cfg.trade.enabled: return
        now = datetime.now(timezone.utc)

        if symbol in self.last_signal_time and now - self.last_signal_time[symbol] < timedelta(minutes=self.cfg.trade.signal_cooldown_minutes):
            return

        async with self.semaphore:
            async with self.store.get_symbol_lock(symbol):
                if symbol in self.last_signal_time and now - self.last_signal_time[symbol] < timedelta(minutes=self.cfg.trade.signal_cooldown_minutes):
                    return
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=100)
                    df = add_indicators(pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]), self.cfg.indicators)
                    if df.empty or await self.store.has_open_signal(symbol): return
                    last = df.iloc[-1]
                    if last['adx'] < self.cfg.indicators.adx_threshold: return
                    stype = "BUY" if last['ema_short'] > last['ema_medium'] else "SELL" if last['ema_short'] < last['ema_medium'] else None
                    if not stype or (stype == "BUY" and not btc_bullish) or (stype == "SELL" and btc_bullish): return

                    social = await self.sentinel.get_sentiment(symbol)
                    funding = social.get('funding', 0.0)
                    if abs(funding) > self.cfg.trade.max_funding_threshold: return

                    price, atr = float(last['close']), float(last['atr'])
                    sl_dist, tp_dist = atr * self.cfg.indicators.atr_sl_mult, atr * self.cfg.indicators.atr_tp_mult
                    sl = price - sl_dist if stype == "BUY" else price + sl_dist
                    tp = price + tp_dist if stype == "BUY" else price - tp_dist
                    ml_conf = await self.get_ml_confidence(symbol, [price, 0, social['score'], 70.0, 0.0, funding])

                    sig = {
                        "timestamp": now.isoformat(), "symbol": symbol, "signal": stype, "market_type": "CEX",
                        "entry": price, "sl": round(sl, 6), "tp": round(tp, 6), "confidence": ml_conf,
                        "sentiment_score": social['score'], "funding": funding, "model_version": self.cfg.model_version, "vol_liq_ratio": 0.0
                    }
                    await self.store.insert_signal(sig)
                    self.last_signal_time[symbol] = now
                    await notify_new_signal(sig, self.session, self.cfg)
                except Exception as e: logger.error(f"CEX Gen Error: {e}")

    async def fetch_dex_alpha(self):
        while True:
            if self.cfg.enable_dex and self.cfg.trade.enabled:
                try:
                    async with self.session.get("https://api.dexscreener.com/latest/dex/search?q=solana") as resp:
                        data = await resp.json()
                        for pair in data.get('pairs', [])[:10]:
                            addr = pair.get('baseToken', {}).get('address')
                            now = datetime.now(timezone.utc)
                            if addr in self.last_signal_time and now - self.last_signal_time[addr] < timedelta(minutes=self.cfg.trade.signal_cooldown_minutes): continue
                            if await self.store.has_open_signal(addr): continue
                            report = await self.security.get_safety_report(addr, float(pair.get('volume', {}).get('h24', 0)), float(pair.get('liquidity', {}).get('usd', 0)), self.helius)
                            if report['is_rugged']: continue
                            if float(pair.get('priceChange', {}).get('m5', 0)) > 5.0:
                                social = await self.sentinel.get_sentiment(pair['baseToken']['symbol'])
                                entry = float(pair.get('priceUsd', 0))
                                ml_conf = await self.get_ml_confidence("GLOBAL", [entry, report['safety_score'], social['score'], 85.0, report['vl_ratio'], social.get('funding', 0.0)])
                                sig = {
                                    "timestamp": now.isoformat(), "symbol": pair['baseToken']['symbol'], "market_type": "DEX", "contract_address": addr,
                                    "signal": f"BUY (GEM - {social['label']})", "entry": entry, "confidence": ml_conf,
                                    "safety_score": report['safety_score'], "sentiment_score": social['score'],
                                    "funding": social.get('funding', 0.0), "vol_liq_ratio": report['vl_ratio']
                                }
                                await self.store.insert_signal(sig)
                                self.last_signal_time[addr] = now
                                await notify_new_signal(sig, self.session, self.cfg)
                except: pass
            await asyncio.sleep(self.cfg.dex_poll_interval)

# -----------------------------
# Risk & Monitoring
# -----------------------------

class PositionMonitor:
    def __init__(self, cfg: BotConfig, store: SignalStore, exchange: ccxt.Exchange, session: aiohttp.ClientSession):
        self.cfg, self.store, self.exchange, self.session = cfg, store, exchange, session
        self.dex = DexEngine(session)

    async def watch_signals(self):
        while True:
            try:
                async with self.store.conn.execute("SELECT id, symbol, market_type, contract_address, signal, entry, sl, tp, timestamp FROM signals WHERE status='open'") as cursor:
                    rows = await cursor.fetchall()
                for row in rows:
                    sig_id, symbol, mtype, addr, side, entry, sl, tp, ts = row
                    current_price = 0.0
                    if mtype == "CEX":
                        ticker = await self.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                    elif "DEX" in mtype or "INSIDER" in mtype:
                        p_data = await self.dex.get_price_data(addr)
                        current_price = p_data['price']

                    if current_price == 0: continue
                    is_win, is_loss = False, False
                    if "BUY" in side:
                        if tp and current_price >= tp: is_win = True
                        if sl and current_price <= sl: is_loss = True
                    elif "SELL" in side:
                        if tp and current_price <= tp: is_win = True
                        if sl and current_price >= sl: is_loss = True

                    if is_win or is_loss:
                        status = "win" if is_win else "loss"
                        duration = int((datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 60)
                        pnl_pct = ((current_price - entry) / entry * 100) if "BUY" in side else ((entry - current_price) / entry * 100)

                        await self.store.update_signal_status(sig_id, status, duration)

                        emoji = "✅" if is_win else "❌"
                        msg = (f"{emoji} *Signal Closed: {status.upper()}*\n"
                               f"Pair: `{symbol}`\n"
                               f"Reason: `{'TP Hit' if is_win else 'SL Hit'}`\n"
                               f"Entry: `${entry}`\n"
                               f"Exit: `${current_price}`\n"
                               f"PnL: `{pnl_pct:+.2f}%`\n"
                               f"Duration: `{duration} mins`")
                        await send_tg_msg(self.session, self.cfg, msg)

            except Exception as e: logger.error(f"Monitor Error: {e}")
            await asyncio.sleep(30)

class RiskManager:
    def __init__(self, cfg: BotConfig, store: SignalStore, session: aiohttp.ClientSession):
        self.cfg, self.store, self.session = cfg, store, session
    async def check_performance_safety(self):
        while True:
            try:
                async with self.store.conn.execute("SELECT status FROM signals WHERE status IN ('win', 'loss') ORDER BY id DESC LIMIT 20") as cursor:
                    rows = await cursor.fetchall()
                if len(rows) >= 10:
                    wr = sum(1 for r in rows if r[0] == 'win') / len(rows)
                    if wr < self.cfg.trade.min_win_rate_threshold and self.cfg.trade.enabled:
                        self.cfg.trade.enabled = False
                        await send_tg_msg(self.session, self.cfg, f"⚠️ *KILL-SWITCH* - Win Rate: `{wr*100}%` - Auto-trading DISABLED.")
            except: pass
            await asyncio.sleep(300)

# -----------------------------
# Bot Lifecycle & Telegram
# -----------------------------

async def send_daily_report(session: aiohttp.ClientSession, cfg: BotConfig, store: SignalStore):
    stats = await store.get_pnl_stats(timeframe_hours=24)
    if stats:
        report = (f"📅 *Daily Performance Report*\n"
                  f"Period: Last 24 Hours\n\n"
                  f"✅ Wins: `{stats['wins']}`\n"
                  f"❌ Losses: `{stats['losses']}`\n"
                  f"📈 Win Rate: `{stats['win_rate']}%`\n\n"
                  f"🤖 Bot Status: `{'ACTIVE' if cfg.trade.enabled else 'PAUSED'}`\n"
                  f"🔥 Model: `{cfg.model_version}`")
        await send_tg_msg(session, cfg, report)

async def telegram_command_listener(session: aiohttp.ClientSession, cfg: BotConfig, sentinel: SocialSentinel, store: SignalStore):
    last_id = 0
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getUpdates"
    while True:
        try:
            async with session.get(url, params={"offset": last_id + 1, "timeout": 20}) as resp:
                data = await resp.json()
                for update in data.get("result", []):
                    last_id = update["update_id"]

                    # Handle Button Taps (Callback Queries)
                    cb_query = update.get("callback_query", {})
                    if cb_query:
                        cb_data = cb_query.get("data")
                        chat_id = str(cb_query.get("message", {}).get("chat", {}).get("id", ""))
                        if chat_id != cfg.telegram_chat_id: continue

                        if cb_data == "toggle_cex":
                            cfg.enable_cex = not cfg.enable_cex
                            msg = f"CEX Monitoring: {'✅ ENABLED' if cfg.enable_cex else '❌ DISABLED'}"
                        elif cb_data == "toggle_dex":
                            cfg.enable_dex = not cfg.enable_dex
                            msg = f"DEX Monitoring: {'✅ ENABLED' if cfg.enable_dex else '❌ DISABLED'}"
                        elif cb_data == "toggle_trade":
                            cfg.trade.enabled = not cfg.trade.enabled
                            msg = f"Auto-Trading: {'▶️ ACTIVE' if cfg.trade.enabled else '⏸️ PAUSED'}"

                        await send_tg_msg(session, cfg, f"⚙️ {msg}")
                        # Acknowledge the callback to Telegram
                        await session.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/answerCallbackQuery", json={"callback_query_id": cb_query['id']})
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id != cfg.telegram_chat_id: continue

                    if text == "/status":
                        # Interactive Toggle Buttons
                        keyboard = {
                            "inline_keyboard": [
                                [
                                    {"text": f"CEX: {'✅' if cfg.enable_cex else '❌'}", "callback_data": "toggle_cex"},
                                    {"text": f"DEX: {'✅' if cfg.enable_dex else '❌'}", "callback_data": "toggle_dex"}
                                ],
                                [
                                    {"text": f"Auto-Trade: {'▶️' if cfg.trade.enabled else '⏸️'}", "callback_data": "toggle_trade"}
                                ]
                            ]
                        }

                        status_text = (f"🚀 *QuikPulse Control Center*\n"
                                     f"Model: `{cfg.model_version}`\n"
                                     f"EMA Strategy: `{cfg.indicators.ema_short}/{cfg.indicators.ema_medium}`")

                        await session.post(
                            f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
                            json={
                                "chat_id": cfg.telegram_chat_id,
                                "text": status_text,
                                "parse_mode": "Markdown",
                                "reply_markup": keyboard
                            }
                        )

                    elif text == "/pnl":
                        stats = await store.get_pnl_stats()
                        if stats:
                            pnl_msg = (f"💰 *Overall Performance*\n"
                                      f"Total Trades: `{stats['total']}`\n"
                                      f"✅ Wins: `{stats['wins']}`\n"
                                      f"❌ Losses: `{stats['losses']}`\n"
                                      f"📈 Win Rate: `{stats['win_rate']}%`")
                            await send_tg_msg(session, cfg, pnl_msg)
                        else: await send_tg_msg(session, cfg, "📊 No trade history found.")

                    elif text == "/list":
                        try:
                            markets = await exchange.load_markets()
                            usdt_pairs = [s for s in markets.keys() if ':USDT' in s]
                            pair_list = "\n".join([f"`{p}`" for p in usdt_pairs[:40]])
                            await send_tg_msg(session, cfg, f"📊 *Available Symbols:*\n{pair_list}\n\n_Showing top 40..._")
                        except Exception as e: await send_tg_msg(session, cfg, f"❌ Exchange Error: `{e}`")

                    elif text.startswith("/pair"):
                        parts = text.split()
                        if len(parts) == 1:
                            await send_tg_msg(session, cfg, f"🔍 *Active Monitoring:*\n`{', '.join(cfg.symbols)}`")
                        elif len(parts) == 3:
                            action, target = parts[1].lower(), parts[2].upper()
                            if action == "add":
                                if target not in cfg.symbols:
                                    try:
                                        await exchange.load_markets()
                                        if target in exchange.markets:
                                            social = await sentinel.get_sentiment(target)
                                            cfg.symbols.append(target)
                                            await send_tg_msg(session, cfg, f"✅ Added `{target}`\nSlug: `{social['slug']}` | Sentiment: `{social['score']}`")
                                        else: await send_tg_msg(session, cfg, f"❌ `{target}` not on KuCoin.")
                                    except: await send_tg_msg(session, cfg, "❌ Validation Error.")
                                else: await send_tg_msg(session, cfg, f"ℹ️ `{target}` already active.")
                            elif action == "remove":
                                if target in cfg.symbols:
                                    cfg.symbols.remove(target)
                                    await send_tg_msg(session, cfg, f"🗑️ Removed `{target}`.")

                    elif text.startswith("/set"):
                        parts = text.split()
                        if len(parts) == 3:
                            param, val = parts[1], parts[2]
                            if hasattr(cfg.indicators, param):
                                try:
                                    current_type = type(getattr(cfg.indicators, param))
                                    setattr(cfg.indicators, param, current_type(val))
                                    await send_tg_msg(session, cfg, f"⚙️ Updated `{param}` to `{val}`")
                                except: await send_tg_msg(session, cfg, "❌ Invalid value.")

                    elif text == "/resume":
                        cfg.trade.enabled = True
                        await send_tg_msg(session, cfg, "▶️ Trading manually resumed.")

                    elif text in ["/help", "/start"]:
                        help_msg = ("💡 *QuikPulse AI Commands:*\n"
                                   "• `/status` - Control center dashboard\n"
                                   "• `/pnl` - Performance analytics\n"
                                   "• `/list` - Show available symbols\n"
                                   "• `/pair add [SYMBOL]` - Monitor new pair\n"
                                   "• `/pair remove [SYMBOL]` - Stop monitoring\n"
                                   "• `/set [param] [val]` - Update indicator settings\n"
                                   "• `/resume` - Start auto-trading")
                        await send_tg_msg(session, cfg, help_msg)
        except Exception as e: logger.error(f"TG Error: {e}")
        await asyncio.sleep(3)

async def send_tg_msg(session: aiohttp.ClientSession, cfg: BotConfig, text: str):
    if not cfg.telegram_bot_token: return
    try: await session.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage", json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

async def background_monitor():
    while True:
        try:
            h = await generator.get_btc_health()
            for s in cfg.symbols:
                await generator.generate_cex_signal(s, h)
                await asyncio.sleep(2)
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
    prefix = "🚨 *CLUSTER!* 🚨\n" if sig.get('is_cluster') else "🚀 *NEW SIGNAL*\n"
    ca_link = f"https://dexscreener.com/solana/{sig.get('contract_address')}" if sig.get('contract_address') else "#"
    msg = f"{prefix}Pair: `{sig['symbol']}`\nType: {sig['signal']}\nPrice: `${sig['entry']}`\nConf: `{sig['confidence']}%` | Social: `{sig.get('sentiment_score', 50)}`\n[Chart]({ca_link})"
    try: await session.post(f"https://api.telegram.org/bot{cfg_bot.telegram_bot_token}/sendMessage", json={"chat_id": cfg_bot.telegram_chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True})
    except: pass

# -----------------------------
# FastAPI App & Webhook Handler
# -----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Globals
cfg = BotConfig()
store = SignalStore(cfg.sqlite_db)
cluster_map = ClusterEngine(cfg.cluster_window_minutes)
exchange = ccxt.kucoinfutures({"apiKey": os.getenv("API_KEY"), "secret": os.getenv("API_SECRET"), "password": os.getenv("API_PASS"), "enableRateLimit": True})
session: Optional[aiohttp.ClientSession] = None
generator: Optional[SignalGenerator] = None

# Web Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        signals = await store.get_latest_signals(limit=20)
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "signals": signals if signals is not None else []
        })
    except Exception as e:
        logger.error(f"Dashboard Route Error: {e}")
        return HTMLResponse("Database Initializing... Please refresh in a moment.")

@app.post("/webhook")
async def helius_webhook_handler(request: Request):
    if not generator: return {"status": "starting"}
    data = await request.json()
    for activity in data:
        events = activity.get('events', {})
        swap = events.get('swap', {})
        if swap:
            native_input = swap.get('nativeInput') or {}
            amount_sol = float(native_input.get('amount', 0)) / 1_000_000_000
            if amount_sol < cfg.min_insider_buy_sol: continue
            token_addr = swap.get('tokenOutMint')
            if not token_addr or await store.has_open_signal(token_addr): continue
            now = datetime.now(timezone.utc)
            if token_addr in generator.last_signal_time and now - generator.last_signal_time[token_addr] < timedelta(minutes=cfg.trade.signal_cooldown_minutes): continue

            price_data = await generator.dex.get_price_data(token_addr)
            if price_data['price'] > 0:
                social = await generator.sentinel.get_sentiment(price_data['symbol'])
                report = await generator.security.get_safety_report(token_addr, price_data['vol24'], price_data['liq'], generator.helius)
                is_cluster = 1 if generator.cluster.record_and_check(token_addr) >= 2 else 0
                funding = social.get('funding', 0.0)
                ml_conf = await generator.get_ml_confidence("GLOBAL", [price_data['price'], report['safety_score'], social['score'], 85.0, report['vl_ratio'], funding])

                sig = {
                    "timestamp": now.isoformat(), "symbol": price_data['symbol'], "market_type": "INSIDER (WEBHOOK)",
                    "contract_address": token_addr, "signal": f"BUY ({'CLUSTER' if is_cluster else 'WHALE MOVE'})",
                    "entry": price_data['price'], "confidence": ml_conf, "safety_score": report['safety_score'],
                    "sentiment_score": social['score'], "funding": funding, "is_cluster": is_cluster,
                    "vol_liq_ratio": report['vl_ratio'], "model_version": cfg.model_version
                }
                await store.insert_signal(sig)
                generator.last_signal_time[token_addr] = now
                await notify_new_signal(sig, session, cfg)
    return {"status": "success"}

@app.on_event("startup")
async def startup():
    global session, generator
    await store.init_db()
    session = aiohttp.ClientSession()
    sentinel = SocialSentinel(SANTIMENT_API_KEY, session)
    generator = SignalGenerator(cfg, store, exchange, sentinel, cluster_map, session)
    trainer = ModelTrainer(cfg.sqlite_db, cfg.ml_model_path)
    monitor = PositionMonitor(cfg, store, exchange, session)
    risk_mgmt = RiskManager(cfg, store, session)

    if WEBHOOK_URL and cfg.tracked_wallets:
        await generator.helius.setup_webhooks(WEBHOOK_URL, cfg.tracked_wallets)

    # Scheduler for Daily Summary
    scheduler = AsyncIOScheduler()
    scheduler.add_job(send_daily_report, 'cron', hour=8, minute=0, args=[session, cfg, store])
    scheduler.start()

    asyncio.create_task(background_monitor())
    asyncio.create_task(generator.fetch_dex_alpha())
    asyncio.create_task(monitor.watch_signals())
    asyncio.create_task(risk_mgmt.check_performance_safety())
    asyncio.create_task(telegram_command_listener(session, cfg, sentinel, store))
    asyncio.create_task(continuous_learning_loop(trainer))
    logger.info("QuikPulse: Production Ready.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
