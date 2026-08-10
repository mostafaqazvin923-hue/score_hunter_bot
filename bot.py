import os
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(url)

print(response.status_code)
print(response.text)
