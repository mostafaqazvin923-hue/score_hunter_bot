# HUNTER-V147 — Liquidity Sweep + FVG Retest
# Live-safe backtest engine
# RR 1:2 | No lookahead | No repaint

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {"defaultType": "swap"}
})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
    "AVAX": "AVAX/USDT",
    "DOT": "DOT/USDT",
    "SUI": "SUI/USDT",
    "NEAR": "NEAR/USDT",
}

DAYS = 365
MARGIN = 100
LEVERAGE = 50
FEE = 0.0007
SLIPPAGE = 0.0003


def fetch_data(symbol, start, end):
    rows = []
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while since < end_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe="15m",
                since=since,
                limit=1000
            )
        except Exception:
            break

        if not batch:
            break

        rows.extend(batch)
        new_since = batch[-1][0] + 1

        if new_since <= since:
            break

        since = new_since

        if len(batch) < 1000:
            break

        time.sleep(0.2)

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["ts","Open","High","Low","Close","Volume"]
    )

    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates("Date")
    df = df.set_index("Date")
    return df.loc[start:end][[
        "Open","High","Low","Close","Volume"
    ]]


def prepare(df):

    x = df.copy()

    x["ATR"] = (
        (x["High"] - x["Low"])
        .rolling(14)
        .mean()
    )

    x["VOL_AVG"] = (
        x["Volume"]
        .rolling(20)
        .mean()
    )

    # confirmed liquidity from past only
    x["Prev_High"] = (
        x["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    x["Prev_Low"] = (
        x["Low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # Fair Value Gap using closed candles only
    x["Bull_FVG"] = (
        x["Low"] > x["High"].shift(2)
    )

    x["Bear_FVG"] = (
        x["High"] < x["Low"].shift(2)
    )

    return x


def get_signal(df, i):

    if i < 50:
        return None

    p = df.iloc[i-1]
    c = df.iloc[i]

    bull = (
        p["Low"] < p["Prev_Low"]
        and
        c["Close"] > p["High"]
        and
        c["Close"] > c["Open"]
        and
        c["Volume"] > c["VOL_AVG"]
        and
        df.iloc[i-2]["Bull_FVG"]
    )

    bear = (
        p["High"] > p["Prev_High"]
        and
        c["Close"] < p["Low"]
        and
        c["Close"] < c["Open"]
        and
        c["Volume"] > c["VOL_AVG"]
        and
        df.iloc[i-2]["Bear_FVG"]
    )

    if bull:
        return "LONG"

    if bear:
        return "SHORT"

    return None


def backtest(df, symbol):

    trades=[]

    for i in range(50, len(df)-2):

        side=get_signal(df,i)

        if not side:
            continue

        entry=df.iloc[i+1]["Open"]
        atr=df.iloc[i]["ATR"]

        if not np.isfinite(atr):
            continue

        if side=="LONG":
            sl=entry-atr
            tp=entry+(atr*2)

        else:
            sl=entry+atr
            tp=entry-(atr*2)


        result=None
        exit_price=None

        for j in range(i+2,len(df)):

            h=df.iloc[j]["High"]
            l=df.iloc[j]["Low"]

            if side=="LONG":

                if l<=sl:
                    result="LOSS"
                    exit_price=sl
                    break

                if h>=tp:
                    result="WIN"
                    exit_price=tp
                    break

            else:

                if h>=sl:
                    result="LOSS"
                    exit_price=sl
                    break

                if l<=tp:
                    result="WIN"
                    exit_price=tp
                    break


        if result:

            ret = (
                (exit_price-entry)/entry
                if side=="LONG"
                else
                (entry-exit_price)/entry
            )

            pnl = (
                ret *
                MARGIN *
                LEVERAGE
            ) - (
                MARGIN *
                LEVERAGE *
                FEE *
                2
            )

            trades.append({
                "Symbol":symbol,
                "Side":side,
                "Outcome":result,
                "PnL":pnl
            })


    return trades


def main():

    end=datetime.utcnow()
    start=end-timedelta(days=DAYS)

    trades=[]

    for name,symbol in SYMBOLS.items():

        print("Loading",name)

        df=fetch_data(symbol,start,end)

        if df is not None:
            trades.extend(
                backtest(
                    prepare(df),
                    name
                )
            )


    if not trades:
        print("No trades generated")
        return


    t=pd.DataFrame(trades)

    win=t[t.Outcome=="WIN"]
    loss=t[t.Outcome=="LOSS"]

    print("="*70)
    print("HUNTER-V147 REPORT")
    print("Trades:",len(t))
    print("Win Rate:",round(len(win)/len(t)*100,2))
    print("PnL:",round(t.PnL.sum(),2))

    pf = (
        win.PnL.sum() /
        abs(loss.PnL.sum())
        if len(loss)
        else 0
    )

    print("Profit Factor:",round(pf,2))
    print(t.groupby("Symbol").PnL.sum())
    print("="*70)


if __name__=="__main__":
    main()
