import os
import json
import time
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = "2090120004"

STATE_FILE = "state.json"
CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v1/simple/price"

COINS = {
    "ETH": 1027,
    "SOL": 5426,
    "XRP": 52,
    "APT": 21794,
}

THRESHOLD = 1.0
COOLDOWN = 30 * 60


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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

    response.raise_for_status()

    data = response.json()["data"]

    prices = {}

    for name, coin_id in COINS.items():
        prices[name] = float(data[str(coin_id)]["price"])

    return prices


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=20
    )

    print("Telegram:", response.status_code)
    print(response.text)


def main():

    print("🟢 SCORE HUNTER SCANNING")

    state = load_state()
    prices = get_prices()

    now = int(time.time())

    for symbol, price in prices.items():

        print(f"{symbol}: {price}")

        old = state.get(symbol, {}).get("price")

        if old is None:
            state[symbol] = {
                "price": price,
                "time": now,
                "last_alert": 0
            }

            print(f"{symbol}: first price saved")
            continue

        change = ((price - old) / old) * 100

        print(
            f"{symbol}: "
            f"{old} -> {price} "
            f"({change:+.2f}%)"
        )

        last_alert = state[symbol].get("last_alert", 0)

        if abs(change) >= THRESHOLD and now - last_alert >= COOLDOWN:

            if change >= THRESHOLD:
                direction = "🟢 LONG"
            else:
                direction = "🔴 SHORT"

            message = (
                "🚨 SCORE HUNTER\n\n"
                f"💰 {symbol}\n"
                f"📊 {direction}\n"
                f"💵 Price: {price:.8f}\n"
                f"📈 5m Move: {change:+.2f}%\n\n"
                "⚠️ Signal only — manage risk."
            )

            send_telegram(message)

            state[symbol]["last_alert"] = now

        state[symbol]["price"] = price
        state[symbol]["time"] = now

    save_state(state)

    print("✅ Scan completed.")


if __name__ == "__main__":
    main()
