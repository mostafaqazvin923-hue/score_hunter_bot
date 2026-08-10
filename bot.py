import os
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "APTUSDT"]

TELEGRAM_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
BINANCE_URL = "https://api.binance.com/api/v3/klines"


def send_message(text):
    response = requests.post(
        TELEGRAM_URL,
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=10
    )

    print("Telegram:", response.status_code)
    response.raise_for_status()


def get_candle(symbol):
    params = {
        "symbol": symbol,
        "interval": "5m",
        "limit": 1
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    candles = response.json()

    if not candles:
        return None

    candle = candles[0]

    return {
        "open_time": int(candle[0]),
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": float(candle[4])
    }


def check_symbol(symbol):
    candle = get_candle(symbol)

    if candle is None:
        return

    open_price = candle["open"]
    current_price = candle["close"]

    change = ((current_price - open_price) / open_price) * 100

    print(
        f"{symbol} | "
        f"Open: {open_price} | "
        f"Price: {current_price} | "
        f"Move: {change:+.2f}%"
    )

    if change >= 1:
        direction = "🟢 LONG"

    elif change <= -1:
        direction = "🔴 SHORT"

    else:
        return

    candle_time = datetime.fromtimestamp(
        candle["open_time"] / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    message = (
        "🚨 SCORE HUNTER ALERT\n\n"
        f"{direction}\n"
        f"Symbol: {symbol}\n"
        f"Entry reference: {current_price}\n"
        f"5m Open: {open_price}\n"
        f"Move: {change:+.2f}%\n"
        f"Candle: {candle_time}\n\n"
        "⚠️ این هشدار فقط شناسایی حرکت ۱٪ است و سیگنال قطعی معامله نیست."
    )

    send_message(message)


def main():
    print("Score Hunter scanning...")

    for symbol in SYMBOLS:
        try:
            check_symbol(symbol)
        except Exception as error:
            print(f"{symbol} ERROR:", error)

    print("Scan completed.")


if __name__ == "__main__":
    main()
