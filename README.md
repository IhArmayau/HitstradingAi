# SignalBotAI SignalBotAI
 
SignalBotAI is an intelligent cryptocurrency signal generation bot using technical indicators and machine learning. It supports automated signal generation, Telegram notifications, and background monitoring for multiple trading pairs.
  
## 📂 Project Structure
 `~/SignalBotAI ├── data/                # SQLite DB & historical OHLCV data ├── logs/                # Bot logs (e.g., signalbot.log) ├── models/              # Saved ML models & scalers ├── requirements.txt     # Python dependencies ├── signalbot.py         # Main bot logic: indicators, ML, signal generation ├── telegram.py          # Telegram messaging utilities └── start.sh             # Script to launch the bot `  
## 🔥 Features
 
 
- Multi-symbol signal generation (BTC, ETH, SOL, ADA, XRP)
 
- Configurable timeframes: 
 
  - Entry: 5m
 
  - Confirmation: 1h & 4h
 

 
 
- Technical indicators: 
 
  - EMA (Short, Medium, Long)
 
  - RSI, ATR, ADX
 
  - Bollinger Bands
 
  - Volume & higher timeframe trend
 

 
 
- ML-based confidence scoring (RandomForest default)
 
- Automatic risk/reward calculation (SL/TP)
 
- Telegram notifications with formatted messages
 
- SQLite database for signals, training data, and performance metrics
 
- Background monitoring and automatic retraining
 
- Simulation mode for testing without real trades
 
- Optional TensorFlow/Keras support
 

  
## 📦 Installation
 
### Prerequisites
 
 
- Python 3.11+
 
- Git
 
- Optional: Virtual environment
 

 `git clone https://github.com/yourusername/SignalBotAI.git cd SignalBotAI python -m venv venv source venv/bin/activate   # Linux/macOS venv\Scripts\activate      # Windows pip install --upgrade pip pip install -r requirements.txt ` 
Optional for TensorFlow models:
 `pip install tensorflow `  
## ⚙️ Configuration
 
Create a `.env` file in the project root (or copy `.env.example`) and configure:
 `SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT TIMEFRAME=5m HIGHER_TIMEFRAME=1h HTF_4H=4h LIMIT=1000 POLL_INTERVAL=300 SQLITE_DB=data/signals.db CONFIDENCE_THRESHOLD=70.0 MAX_CONCURRENT_TASKS=5 TELEGRAM_BOT_TOKEN=<YOUR_BOT_TOKEN> TELEGRAM_CHAT_ID=<YOUR_CHAT_ID> MODEL_VERSION=v2 EXPECTED_SLIPPAGE_PCT=0.0005 SIMULATE_EXECUTION=true ` 
 
 SIMULATE_EXECUTION=true → signals generated without executing trades.
 
- SQLITE_DB → points to SQLite database stored in `data/`.
 

  
## 💻 Running the Bot
 
### Option 1: Using `start.sh` script
 `chmod +x start.sh ./start.sh ` 
 
- Exports environment variables from `.env`.
 
- Launches the bot (`signalbot.py`) with FastAPI server and background monitoring.
 

 
### Option 2: Directly with Python
 `python signalbot.py `  
## 🌐 API Endpoints
 
  
 
Endpoint
 
Method
 
Description
 
   
 
`/`
 
GET
 
Health check & model info
 
 
 
`/signals?limit=50`
 
GET
 
Fetch recent signals
 
 
 
`/trigger?symbol=BTC/USDT`
 
POST
 
Generate signal manually for a symbol
 
 
 
`/heartbeat`
 
GET/HEAD
 
Heartbeat for uptime monitoring
 
  
  
## 🧠 Machine Learning
 
 
- Uses features such as EMA, RSI, ATR, ADX, Bollinger Bands, order book metrics, funding rate, open interest, and on-chain flow.
 
- Models saved per symbol in `models/`.
 
- Automatic retraining when enough new training samples are collected.
 
- Scales features per symbol with `StandardScaler`.
 

  
## 📊 Example Telegram Signal
 `📊 New Signal from SignalBotAI Pair: BTC/USDT Signal: 🟢 BUY Entry: 29300.12 SL: 29250.50 | TP: 29450.00 R:R: 3.0 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜ Confidence: 85.0% `  
## 🛠 Dependencies
 
 
- `fastapi`
 
- `uvicorn`
 
- `pandas`
 
- `numpy`
 
- `scikit-learn`
 
- `ta` (technical analysis)
 
- `ccxt` (exchange API)
 
- `aiosqlite`
 
- `python-dotenv`
 
- `aiohttp`
 
- `joblib`
 
- Optional: `tensorflow` / `keras`
 

  
## ⚡ Logs
 
 
- Logs stored in `logs/` directory (e.g., `logs/bot.log`)
 
- Configure log level via `.env` (`LOG_LEVEL=INFO`)
 

  
## 🌐 Screenshots / Demo
 
SignalBotAI Telegram Signal Example Signals sent directly to Telegram with R:R and confidence visualization.
  
## 🤝 Contribution
 
 
1. Fork the repository
 
2. Create a feature branch (`git checkout -b feature-name`)
 
3. Commit your changes (`git commit -am 'Add new feature'`)
 
4. Push to the branch (`git push origin feature-name`)
 
5. Open a Pull Request
 

  
📜 License
 
MIT License © 2025 SignalBotAI
 
