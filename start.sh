#!/bin/bash
# SignalBotAI start script

# Make Python output unbuffered for logs
export PYTHONUNBUFFERED=1

# Ensure necessary folders exist
mkdir -p data
mkdir -p models
mkdir -p logs

# Use Python module to run uvicorn (avoids "command not found")
python -m uvicorn signalbot:app --host 0.0.0.0 --port $PORT --log-level info
