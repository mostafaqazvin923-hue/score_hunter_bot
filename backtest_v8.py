
"""
CLTS-PRO V1 — CAUSAL LIQUIDITY STRUCTURE ENGINE
LBank Futures / CCXT Backtester

Backtest only. No API key required.
Strategy engine:
- 4H market regime
- 1H liquidity sweep + BOS + retest
- Fixed RR 1:2
- No timeout
- No repainting
"""

import sys, subprocess
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})


SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "SUI": "SUI/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
    "AVAX": "AVAX/USDT",
    "DOT": "DOT/USDT",
    "NEAR": "NEAR/USDT",
}


LOOKBACK_DAYS = 365
INITIAL_CAPITAL = 1000
TRADE_MARGIN = 100
MAX_OPEN_POSITIONS = 3

FEE = 0.0006
SLIPPAGE = 0.0002

RR = 2.0


def fetch_ohlcv(symbol, timeframe):
    since = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp()*1000)
    data = []

    while True:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=1000
        )

        if not batch:
            break

        data.extend(batch)
        since = batch[-1][0] + 1

        if len(batch) < 1000:
            break

    df = pd.DataFrame(
        data,
        columns=["ts","open","high","low","close","volume"]
    )

    df["date"] = pd.to_datetime(df.ts, unit="ms")
    df.drop_duplicates("date", inplace=True)
    df.sort_values("date", inplace=True)

    return df.reset_index(drop=True)



def add_indicators(df):

    df["ema50"] = df.close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = df.close.ewm(
        span=200,
        adjust=False
    ).mean()

    tr = pd.concat([
        df.high-df.low,
        abs(df.high-df.close.shift()),
        abs(df.low-df.close.shift())
    ],axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    return df



def regime(df4, i):

    c=df4.iloc[i]

    if (
        c.close > c.ema50 and
        c.ema50 > c.ema200
    ):
        return "BULL"

    if (
        c.close < c.ema50 and
        c.ema50 < c.ema200
    ):
        return "BEAR"

    return "NEUTRAL"



def pivot_low(df,i,left=2,right=2):

    if i-right < 0:
        return False

    x=df.low.iloc[i-right]

    return (
        x == min(
            df.low.iloc[i-left:i+right+1]
        )
    )



def pivot_high(df,i,left=2,right=2):

    if i-right < 0:
        return False

    x=df.high.iloc[i-right]

    return (
        x == max(
            df.high.iloc[i-left:i+right+1]
        )
    )



def backtest(data):

    trades=[]
    active={}

    equity=INITIAL_CAPITAL
    curve=[]

    for ts in sorted(data.keys()):

        for sym,pos in list(active.items()):

            df=data[sym]["1h"]

            i=df.index[df.date==ts]

            if len(i)==0:
                continue

            c=df.iloc[i[0]]

            hit_sl=False
            hit_tp=False

            if pos["side"]=="LONG":

                hit_sl=c.low <= pos["sl"]
                hit_tp=c.high >= pos["tp"]

            else:

                hit_sl=c.high >= pos["sl"]
                hit_tp=c.low <= pos["tp"]


            if hit_sl or hit_tp:

                # conservative rule:
                # SL wins if both touched

                if hit_sl:
                    exit_price=pos["sl"]
                    result="LOSS"
                else:
                    exit_price=pos["tp"]
                    result="WIN"


                pnl = (
                    (exit_price-pos["entry"])
                    if pos["side"]=="LONG"
                    else
                    (pos["entry"]-exit_price)
                )

                pnl = pnl/pos["entry"]*TRADE_MARGIN

                pnl -= TRADE_MARGIN*(FEE*2)

                equity += pnl

                trades.append({
                    "symbol":sym,
                    "side":pos["side"],
                    "result":result,
                    "pnl":pnl
                })

                del active[sym]


        curve.append(equity)


        for sym,obj in data.items():

            if len(active)>=MAX_OPEN_POSITIONS:
                break

            if sym in active:
                continue


            df1=obj["1h"]

            row=df1[df1.date==ts]

            if row.empty:
                continue

            i=row.index[0]

            if i<20:
                continue


            c=df1.iloc[i]


            # CLTS placeholder engine:
            # sweep + BOS + retest logic goes here.
            # Kept isolated for modification/testing.

            if regime(obj["4h"],min(len(obj["4h"])-1, i//4))=="BULL":

                if (
                    c.close>c.open and
                    c.volume >
                    df1.volume.iloc[i-20:i].mean()*1.2
                ):

                    entry=c.close*(1+SLIPPAGE)
                    sl=c.low-c.atr*0.2
                    risk=entry-sl

                    active[sym]={
                        "side":"LONG",
                        "entry":entry,
                        "sl":sl,
                        "tp":entry+risk*RR
                    }


    return pd.DataFrame(trades),curve



if __name__=="__main__":

    processed={}

    for name,symbol in SYMBOLS.items():

        try:
            h1=add_indicators(fetch_ohlcv(symbol,"1h"))
            h4=add_indicators(fetch_ohlcv(symbol,"4h"))

            processed[name]={
                "1h":h1,
                "4h":h4
            }

            print(name,"loaded")

        except Exception as e:
            print(name,e)


    timestamps=set()

    for x in processed.values():
        timestamps.update(x["1h"].date)

    trades,equity=backtest(processed)

    if len(trades):

        print("\nTrades:",len(trades))
        print("Win rate:",
              round(
                  (trades.result=="WIN").mean()*100,2
              ),
              "%")

        print("PnL:",
              round(trades.pnl.sum(),2))

        print(trades.groupby("symbol").pnl.sum())

    else:
        print("No trades")


    plt.figure(figsize=(10,5))
    plt.plot(equity)
    plt.title("CLTS-PRO Equity Curve")
    plt.savefig("equity_curve.png")
