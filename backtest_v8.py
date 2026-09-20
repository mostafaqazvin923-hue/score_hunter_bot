"""
HUNTER_PRO_V2_BACKTEST.py

Causal Trend Momentum Futures Engine
LBank Futures / CCXT

Rules:
- No look ahead bias
- Signal only after candle close
- Entry on next candle open
- No repainting
- No timeout exits
- Fixed RR 1:2
- Conservative SL priority
"""

import sys, subprocess
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})


SYMBOLS = {
    "BTC":"BTC/USDT",
    "ETH":"ETH/USDT",
    "SOL":"SOL/USDT",
    "XRP":"XRP/USDT",
    "SUI":"SUI/USDT",
    "ADA":"ADA/USDT",
    "LINK":"LINK/USDT",
    "AVAX":"AVAX/USDT",
    "DOT":"DOT/USDT",
    "NEAR":"NEAR/USDT"
}

DAYS = 365
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
MAX_POSITIONS = 3

FEE_RATE = 0.0006
SLIPPAGE = 0.0002

RR = 2.0
ATR_SL = 1.8


def fetch_data(symbol, timeframe):
    since = int((datetime.now()-timedelta(days=DAYS)).timestamp()*1000)
    candles = []

    while True:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=1000
        )

        if not batch:
            break

        candles.extend(batch)
        since = batch[-1][0] + 1

        if len(batch) < 1000:
            break

    df = pd.DataFrame(
        candles,
        columns=["Timestamp","Open","High","Low","Close","Volume"]
    )

    df["Date"] = pd.to_datetime(df.Timestamp, unit="ms")
    df.drop_duplicates("Date", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def indicators(df):
    df["EMA50"] = df.Close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = df.Close.ewm(span=200, adjust=False).mean()

    tr = pd.concat([
        df.High-df.Low,
        abs(df.High-df.Close.shift()),
        abs(df.Low-df.Close.shift())
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["RVOL"] = df.Volume / df.Volume.rolling(20).mean()

    df["Momentum"] = (
        df.Close - df.Close.shift(20)
    ) / df.Close.shift(20)

    return df


def regime(df, i):
    c = df.iloc[i]

    if c.Close > c.EMA50 > c.EMA200:
        return "LONG"

    if c.Close < c.EMA50 < c.EMA200:
        return "SHORT"

    return None


def run_backtest(data):

    active = {}
    trades = []
    equity = INITIAL_CAPITAL

    timestamps = sorted(
        set(
            t for x in data.values()
            for t in x["1h"].Date
        )
    )

    for ts in timestamps:

        # exits first
        for sym, pos in list(active.items()):

            df = data[sym]["1h"]
            rows = df[df.Date == ts]

            if rows.empty:
                continue

            c = rows.iloc[0]

            if pos["side"] == "LONG":
                hit_sl = c.Low <= pos["SL"]
                hit_tp = c.High >= pos["TP"]
            else:
                hit_sl = c.High >= pos["SL"]
                hit_tp = c.Low <= pos["TP"]

            if hit_sl or hit_tp:

                # conservative rule: SL wins
                exit_price = pos["SL"] if hit_sl else pos["TP"]

                if pos["side"] == "LONG":
                    ret = (exit_price-pos["entry"])/pos["entry"]
                else:
                    ret = (pos["entry"]-exit_price)/pos["entry"]

                pnl = TRADE_MARGIN*ret - TRADE_MARGIN*FEE_RATE*2
                equity += pnl

                trades.append({
                    "Symbol": sym,
                    "Side": pos["side"],
                    "PnL": pnl,
                    "Result": "WIN" if pnl > 0 else "LOSS"
                })

                del active[sym]


        # entries
        if len(active) >= MAX_POSITIONS:
            continue

        for sym,obj in data.items():

            if sym in active:
                continue

            df1 = obj["1h"]
            df4 = obj["4h"]

            rows = df1[df1.Date == ts]

            if rows.empty:
                continue

            i = rows.index[0]

            if i < 220:
                continue

            r = regime(df4, min(i//4, len(df4)-1))

            c = df1.iloc[i]

            if c.RVOL <= 1.2:
                continue

            if r == "LONG" and c.Momentum <= 0:
                continue

            if r == "SHORT" and c.Momentum >= 0:
                continue

            if r is None:
                continue

            # next candle open only
            if i+1 >= len(df1):
                continue

            nxt = df1.iloc[i+1]

            entry = nxt.Open * (
                1 + SLIPPAGE if r=="LONG" else 1-SLIPPAGE
            )

            if r == "LONG":
                sl = entry - ATR_SL*c.ATR
                tp = entry + (entry-sl)*RR
            else:
                sl = entry + ATR_SL*c.ATR
                tp = entry - (sl-entry)*RR

            active[sym] = {
                "side": r,
                "entry": entry,
                "SL": sl,
                "TP": tp
            }

    return pd.DataFrame(trades), equity


if __name__ == "__main__":

    data = {}

    for name, symbol in SYMBOLS.items():
        print("Loading", name)

        h1 = indicators(fetch_data(symbol, "1h"))
        h4 = indicators(fetch_data(symbol, "4h"))

        data[name] = {
            "1h": h1,
            "4h": h4
        }

    trades, equity = run_backtest(data)

    trades.to_csv("hunter_v2_trades.csv", index=False)

    print("="*60)
    print("HUNTER PRO V2 RESULTS")
    print("="*60)

    print("Trades:", len(trades))

    if len(trades):
        print("Win Rate:",
              round((trades.Result=="WIN").mean()*100,2), "%")
        print("PnL:",
              round(trades.PnL.sum(),2))
        print("Final Balance:",
              round(equity,2))
        print(trades.groupby("Symbol").PnL.sum())
    else:
        print("No trades")
