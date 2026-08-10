import os
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

data = {
    "chat_id": CHAT_ID,
    "text": "🟢 Score Hunter Bot فعال شد"
}

response = requests.post(url, data=data, timeout=10)

print("Status:", response.status_code)
print("Telegram:", response.text)

response.raise_for_status()
