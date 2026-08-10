import os
import json
import time
import subprocess
import requests

# =========================
# Telegram
# =========================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = "2090120004"

# =========================
# CoinMarketCap
# =========================

CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v1/simple/price"

COINS = {
    "ETH": 1027,
    "SOL": 5426,
    "XRP": 52,
    "APT": 21794,
}

# هشدار در حرکت 1 درصدی
THRESHOLD = 1.0

# جلوگیری از هشدار تکراری برای همان ارز
COOLDOWN = 30 * 60

STATE_FILE = "state.json"


# =========================
# State
# =========================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("State load error:", e)
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# =========================
# Telegram
# =========================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=20
        )

        print("Telegram status:", response.status_code)
        print("Telegram response:", response.text)

    except Exception as e:
        print("Telegram error:", e)


# =========================
# Get prices
# =========================

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

    print("CoinMarketCap status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    data = result["data"]

    prices = {}

    # CoinMarketCap در این endpoint
    # data را به صورت LIST برمی‌گرداند.
    for item in data:

        coin_id = int(item["id"])
        price = float(item["price"])

        for symbol, expected_id in COINS.items():

            if coin_id == expected_id:
                prices[symbol] = price

    return prices


# =========================
# Save state to GitHub
# =========================

def commit_state():

    try:

        subprocess.run(
            ["git", "config", "user.name", "Score Hunter Bot"],
            check=True
        )

        subprocess.run(
            ["git", "config", "user.email", "score-hunter-bot@users.noreply.github.com"],
            check=True
        )

        subprocess.run(
            ["git", "add", STATE_FILE],
            check=True
        )

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"]
        )

        # اگر تغییری وجود نداشته باشد
        if result.returncode == 0:
            print("No state changes to commit.")
            return

        subprocess.run(
            ["git", "commit", "-m", "Update price state"],
            check=True
        )

        subprocess.run(
            ["git", "push"],
            check=True
        )

        print("✅ State saved to GitHub.")

    except Exception as e:

        print("State commit error:", e)


# =========================
# Main
# =========================

def main():

    print("🟢 SCORE HUNTER SCANNING")

    state = load_state()

    prices = get_prices()

    now = int(time.time())

    for symbol, price in prices.items():

        print(f"{symbol}: {price}")

        # -------------------------
        # First run
        # -------------------------

        if symbol not in state:

            state[symbol] = {
                "price": price,
                "time": now,
                "last_alert": 0
            }

            print(f"{symbol}: first price saved.")
            continue

        old_price = float(state[symbol]["price"])

        old_time = int(state[symbol].get("time", now))

        last_alert = int(
            state[symbol].get("last_alert", 0)
        )

        # -------------------------
        # Percentage movement
        # -------------------------

        change = ((price - old_price) / old_price) * 100

        elapsed = now - old_time

        print(
            f"{symbol} | "
            f"Old: {old_price} | "
            f"New: {price} | "
            f"Change: {change:+.2f}% | "
            f"Elapsed: {elapsed}s"
        )

        # -------------------------
        # Alert
        # -------------------------

        if abs(change) >= THRESHOLD:

            if now - last_alert >= COOLDOWN:

                if change >= THRESHOLD:

                    direction = "🟢 LONG"

                else:

                    direction = "🔴 SHORT"

                message = (
                    "🚨 SCORE HUNTER ALERT 🚨\n\n"
                    f"💰 {symbol}\n"
                    f"📊 {direction}\n"
                    f"💵 Price: {price:.8f}\n"
                    f"📈 Movement: {change:+.2f}%\n"
                    f"⏱️ Period: {elapsed // 60} min\n\n"
                    "⚠️ Signal only — manage risk."
                )

                send_telegram(message)

                state[symbol]["last_alert"] = now

            else:

                print(
                    f"{
