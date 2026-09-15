import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V75
# Adaptive Risk + Anti Loss Cluster Engine
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}


REMOVED_COINS = {
    "NEAR","OP","HYPE","HBAR","AVAX","SUI",
    "PENDLE","TIA","FET","SEI","ARB",
    "DOT","ETC","SHIB","STX","RUNE",
    "MKR","APT","LTC","AR","IMX",
    "PEPE","BONK"
}


SYMBOLS = {
    k:v for k,v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


LOOKBACK_DAYS = 365
TIMEFRAME = "4h"


MAX_POSITIONS = 5


# هزینه‌ها
SLIPPAGE = 0.0003
FEE_RATE = 0.0007


# ATR
ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.2


# مدیریت ضرر
TIMEOUT_CANDLES = 35

# بعد از ضرر روی یک کوین
COOLDOWN_CANDLES = 12


EMA_WARMUP = 200


INITIAL_CAPITAL = 1000
TRADE_MARGIN = 100
LEVERAGE = 80



start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp()*1000)



print("="*70)
print("🚀 HUNTER-V75 Adaptive Risk Engine")
print("="*70)



processed_data = {}



def fetch_symbol_data(symbol):

    candles = []

    current_since = since_timestamp
    last_seen = None


    while current_since < exchange.milliseconds():

        batch = None

        for attempt in range(3):

            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000
                )
                break

            except:

                if attempt == 2:
                    return None


        if not batch:
            break


        last_ts = batch[-1][0]


        if last_seen and last_ts <= last_seen:
            return None


        candles.extend(batch)

        last_seen = last_ts

        current_since = last_ts + 1


        if len(batch)<1000:
            break



    if not candles:
        return None



    df = pd.DataFrame(
        candles,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )


    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms"
    )


    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]


    df.drop_duplicates(
        "Date",
        keep="last",
        inplace=True
    )


    df.sort_values(
        "Date",
        inplace=True
    )


    df.reset_index(
        drop=True,
        inplace=True
    )


    if len(df)<EMA_WARMUP+50:
        return None



    # ATR

    tr1 = df["High"]-df["Low"]

    tr2 = abs(
        df["High"]-
        df["Close"].shift(1)
    )

    tr3 = abs(
        df["Low"]-
        df["Close"].shift(1)
    )


    df["ATR"] = pd.concat(
        [tr1,tr2,tr3],
        axis=1
    ).max(axis=1).rolling(
        ATR_PERIOD
    ).mean()



    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()


    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()


    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()



    df["Mom_Short"] = (
        df["Close"]-
        df["Close"].shift(10)
    ) / df["Close"].shift(10)



    df["Mom_Long"] = (
        df["Close"]-
        df["Close"].shift(30)
    ) / df["Close"].shift(30)



    df.set_index(
        "Date",
        inplace=True
    )


    return df



for name,symbol in SYMBOLS.items():

    data = fetch_symbol_data(symbol)

    if data is not None:
        processed_data[name]=data



print(
    f"✅ Valid symbols: {len(processed_data)}"
)
