import os
import json
import time
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = "2090120004"

CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v1/simple/price"

COINS = {
    "ETH": 1027,
    "SOL": 5426,
    "XRP": 52,
    "APT": 21794,
}

THRESHOLD = 1.0
COOLDOWN = 30 * 60
STATE_FILE = "state.json"


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    print("Telegram:", response.status_code)
    print(response.text)


def get_prices():

    ids = ",".join(str(x) for x in COINS.values())

    response = requests.get(
        CMC_URL,
        params={
            "ids": ids,
            "convert": "USD"
        },
        timeout=20
    )

    print("CoinMarketCap:", response.status_code)

    response.raise_for_status()

    data = response.json()["data"]

    prices = {}

    # مهم:
    # CoinMarketCap اینجا LIST برمی‌گرداند
    for item in data:

        coin_id = int(item["id"])
        price = float(item["price"])

        for symbol, expected_id in COINS.items():

            if coin_id == expected_id:
                prices[symbol] = price

    return prices


def main():

    print("🟢 SCORE HUNTER SCANNING")

    state = load_state()
    prices = get_prices()

    now = int(time.time())

    for symbol, price in prices.items():

        print(f"{symbol}: {price}")

        if symbol not in state:

            state[symbol] = {
                "price": price,
                "time": now,
                "last_alert": 0
            }

            print(f"{symbol}: first price saved")
            continue

        old_price = float(state[symbol]["price"])

        change = ((price - old_price) / old_price) * 100

        print(
            f"{symbol}: "
            f"{old_price} -> {price} "
            f"CHANGE: {change:+.2f}%"
        )

        last_alert = int(
            state[symbol].get("last_alert", 0)
        )

        if abs(change) >= THRESHOLD:

            if now - last_alert >= COOLDOWN:

                if change > 0:
                    direction = "🟢 LONG"
                else:
                    direction = "🔴 SHORT"

                message = (
                    "🚨 SCORE HUNTER 🚨\n\n"
                    f"💰 {symbol}\n"
                    f"📊 {direction}\n"
                    f"💵 Price: {price:.8f}\n"
                    f"📈 Move: {change:+.2f}%\n\n"
                    "⚠️ Manage risk."
                )

                send_telegram(message)

                state[symbol]["last_alert"] = now

        state[symbol]["price"] = price
        state[symbol]["time"] = now

    save_state(state)

    print("✅ Scan completed.")


if __name__ == "__main__":
    main()
