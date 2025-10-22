#!/bin/bash
export PYTHONUNBUFFERED=1
uvicorn signalbot:app --host 0.0.0.0 --port $PORT
