#!/bin/bash
# SignalBotAI start script for Render

# Make sure Python prints logs immediately
export PYTHONUNBUFFERED=1

# Create required folders inside project
mkdir -p data
mkdir -p models
mkdir -p logs

# Run the FastAPI app using the PORT assigned by Render
uvicorn signalbot:app --host 0.0.0.0 --port $PORT --log-level info
