#!/usr/bin/env python3
import requests
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ Make sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env")
    exit(1)

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "✅ Telegram test message from Termux!"}

try:
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code == 200:
        print("✅ Message sent successfully!")
    else:
        print(f"❌ Failed to send message: {resp.status_code} {resp.text}")
except requests.exceptions.RequestException as e:
    print(f"❌ Error sending message: {e}")
