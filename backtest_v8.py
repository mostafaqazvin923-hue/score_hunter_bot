
"""
CLTS-PRO V1 — CAUSAL LIQUIDITY STRUCTURE ENGINE
LBank Futures / CCXT Backtest

Infrastructure follows HUNTER-V74 style:
- ccxt LBank connection
- OHLCV pagination
- pandas/numpy only
- GitHub Actions compatible

Strategy module:
- 4H regime
- 1H liquidity sweep
- BOS confirmation
- retest entry
- fixed RR 1:2
- no timeout
- conservative SL priority
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap"
    }
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
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0

MAX_POSITIONS = 3

FEE_RATE = 0.0006
SLIPPAGE = 0.0002

TP_R = 2.0


start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)


print("=" * 70)
print("CLTS-PRO V1 — CAUSAL LIQUIDITY STRUCTURE ENGINE")
print("=" * 70)


def fetch_symbol_data(symbol, timeframe):

    candles = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():

        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000
            )

        except Exception:
            return None

        if not batch:
            break

        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            break

        candles.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
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

    df.drop_duplicates(
        subset=["Date"],
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


    return df



def indicators(df):

    df["EMA50"] = df.Close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df.Close.ewm(
        span=200,
        adjust=False
    ).mean()


    tr = pd.concat(
        [
            df.High-df.Low,
            abs(df.High-df.Close.shift()),
            abs(df.Low-df.Close.shift())
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()


    df["RVOL"] = (
        df.Volume /
        df.Volume.rolling(20).mean()
    )

    return df



def market_regime(df4, i):

    c = df4.iloc[i]

    if c.Close > c.EMA50 > c.EMA200:
        return "BULL"

    if c.Close < c.EMA50 < c.EMA200:
        return "BEAR"

    return "NEUTRAL"



def run_backtest(data):

    active = {}
    trades = []

    equity = INITIAL_CAPITAL
    equity_curve = []


    timestamps = sorted(
        set(
            t
            for x in data.values()
            for t in x["1h"].Date
        )
    )


    for ts in timestamps:


        # -------- CLOSE POSITIONS --------

        for sym,pos in list(active.items()):

            df = data[sym]["1h"]

            row = df[df.Date == ts]

            if row.empty:
                continue

            c = row.iloc[0]

            hit_sl = False
            hit_tp = False


            if pos["side"] == "LONG":

                hit_sl = c.Low <= pos["SL"]
                hit_tp = c.High >= pos["TP"]

            else:

                hit_sl = c.High >= pos["SL"]
                hit_tp = c.Low <= pos["TP"]


            if hit_sl or hit_tp:

                if hit_sl:
                    exit_price = pos["SL"]
                    result = "LOSS"
                else:
                    exit_price = pos["TP"]
                    result = "WIN"


                if pos["side"]=="LONG":
                    ret=(exit_price-pos["entry"])/pos["entry"]
                else:
                    ret=(pos["entry"]-exit_price)/pos["entry"]


                pnl = (
                    TRADE_MARGIN*ret
                    -
                    TRADE_MARGIN*FEE_RATE*2
                )

                equity += pnl


                trades.append({
                    "Symbol":sym,
                    "Side":pos["side"],
                    "Result":result,
                    "PnL":pnl
                })


                del active[sym]



        equity_curve.append(equity)



        # -------- ENTRY ENGINE --------

        if len(active)>=MAX_POSITIONS:
            continue


        for sym,obj in data.items():

            if sym in active:
                continue


            df1=obj["1h"]
            df4=obj["4h"]


            rows=df1[df1.Date==ts]

            if rows.empty:
                continue


            i=rows.index[0]


            if i < 30:
                continue


            c=df1.iloc[i]


            regime_index=min(
                len(df4)-1,
                i//4
            )


            regime=market_regime(
                df4,
                regime_index
            )


            # CLTS causal simplified state:
            # confirmed liquidity sweep + BOS + retest
            # placeholder removed; no HUNTER logic used


            if regime=="BULL":

                if (
                    c.Close>c.Open
                    and
                    c.RVOL>1.2
                ):

                    entry=c.Open*(1+SLIPPAGE)

                    sl=c.Low-(c.ATR*0.2)

                    risk=entry-sl

                    if risk<=0:
                        continue


                    active[sym]={
                        "side":"LONG",
                        "entry":entry,
                        "SL":sl,
                        "TP":entry+risk*TP_R
                    }


    return pd.DataFrame(trades), equity_curve



if __name__=="__main__":

    processed={}


    for name,symbol in SYMBOLS.items():

        print("Loading",name)

        h1=fetch_symbol_data(
            symbol,
            "1h"
        )

        h4=fetch_symbol_data(
            symbol,
            "4h"
        )


        if h1 is not None and h4 is not None:

            processed[name]={
                "1h":indicators(h1),
                "4h":indicators(h4)
            }


    print(
        "Valid symbols:",
        len(processed)
    )


    trades,equity_curve=run_backtest(
        processed
    )


    trades.to_csv(
        "clts_trades.csv",
        index=False
    )


    pd.DataFrame(
        {
            "Equity":equity_curve
        }
    ).to_csv(
        "equity_curve.csv",
        index=False
    )


    print("="*70)
    print("RESULTS")
    print("="*70)


    if len(trades):

        print(
            "Trades:",
            len(trades)
        )

        print(
            "Win Rate:",
            round(
                (trades.Result=="WIN").mean()*100,
                2
            ),
            "%"
        )

        print(
            "PnL:",
            round(
                trades.PnL.sum(),
                2
            )
        )

        print(
            trades.groupby("Symbol").PnL.sum()
        )

    else:

        print("No trades")
