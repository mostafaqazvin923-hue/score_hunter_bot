import os
import time
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "APTUSDT"]

TG_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"

last_alert = {}


def send_message(text):
    response = requests.post(
        TG_URL,
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=10
    )
    response.raise_for_status()


def get_prices():
    response = requests.get(BINANCE_URL, timeout=10)
    response.raise_for_status()

    data = response.json()
    return {
        item["symbol"]: float(item["price"])
        for item in data
        if item["symbol"] in SYMBOLS
    }


def check_market():
    prices = get_prices()

    for symbol in SYMBOLS:
        price = prices.get(symbol)

        if price is None:
            continue

        previous = last_alert.get(symbol)

        if previous is None:
            last_alert[symbol] = price
            continue

        change = ((price - previous) / previous) * 100

        if abs(change) >= 1:
            direction = "🟢 LONG" if change > 0 else "🔴 SHORT"

            message = (
                f"🚨 SCORE HUNTER ALERT\n\n"
                f"{direction}\n"
                f"Symbol: {symbol}\n"
                f"Price: {price}\n"
                f"Move: {change:+.2f}%\n\n"
                f"⚠️ حرکت ۱٪ شناسایی شد"
            )

            send_message(message)
            last_alert[symbol] = price


send_message("🟢 Score Hunter Bot شروع به کار کرد")

for _ in range(1):
    try:
        check_market()
        print("Market check completed successfully.")
    except Exception as e:
        print("ERROR:", e)
