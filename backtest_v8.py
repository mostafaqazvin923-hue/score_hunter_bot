"""
CLTS_PRO_V3_FULL_BACKTEST.py

LBank Futures / CCXT
Causal Liquidity Structure Engine

Features:
- 1 year OHLCV download
- 4H regime
- 1H confirmed pivots
- liquidity sweep
- BOS
- retest entry
- fixed RR 1:2
- no timeout
- no matplotlib dependency

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


DAYS=365
MARGIN=100
START_BALANCE=1000
FEE=0.0006
SLIPPAGE=0.0002
RR=2


def fetch(symbol,tf):
    since=int((datetime.now()-timedelta(days=DAYS)).timestamp()*1000)
    rows=[]

    while True:
        batch=exchange.fetch_ohlcv(
            symbol,
            timeframe=tf,
            since=since,
            limit=1000
        )

        if not batch:
            break

        rows.extend(batch)
        since=batch[-1][0]+1

        if len(batch)<1000:
            break

    df=pd.DataFrame(
        rows,
        columns=["ts","open","high","low","close","volume"]
    )

    df["date"]=pd.to_datetime(df.ts,unit="ms")
    df.drop_duplicates("date",inplace=True)
    df.sort_values("date",inplace=True)
    return df.reset_index(drop=True)


def indicators(df):

    df["ema50"]=df.close.ewm(span=50,adjust=False).mean()
    df["ema200"]=df.close.ewm(span=200,adjust=False).mean()

    tr=pd.concat([
        df.high-df.low,
        abs(df.high-df.close.shift()),
        abs(df.low-df.close.shift())
    ],axis=1).max(axis=1)

    df["atr"]=tr.rolling(14).mean()
    df["rvol"]=df.volume/df.volume.rolling(20).mean()

    return df


def pivots(df):

    ph={}
    pl={}

    for i in range(2,len(df)-2):

        if df.high.iloc[i]==max(df.high.iloc[i-2:i+3]):
            ph[i]=df.high.iloc[i]

        if df.low.iloc[i]==min(df.low.iloc[i-2:i+3]):
            pl[i]=df.low.iloc[i]

    return ph,pl


def regime(df,i):

    c=df.iloc[i]

    if c.close>c.ema50>c.ema200:
        return "LONG"

    if c.close<c.ema50<c.ema200:
        return "SHORT"

    return None


def run_backtest(data):

    trades=[]
    equity=START_BALANCE
    active={}

    for sym,obj in data.items():

        df=obj["1h"]
        ph,pl=obj["pivots"]

        for i in range(100,len(df)-1):

            if sym in active:
                continue

            side=regime(
                obj["4h"],
                min(i//4,len(obj["4h"])-1)
            )

            if not side:
                continue

            c=df.iloc[i]

            last_high=max(
                [v for k,v in ph.items() if k<i],
                default=None
            )

            last_low=max(
                [v for k,v in pl.items() if k<i],
                default=None
            )

            if last_high is None or last_low is None:
                continue


            # long sweep + BOS
            if side=="LONG":

                if c.low<last_low and c.close>last_high:

                    entry=df.iloc[i+1].open*(1+SLIPPAGE)
                    sl=last_low-c.atr*0.2
                    risk=entry-sl

                    if risk>0:
                        tp=entry+risk*RR

                        hit=False

                        for j in range(i+1,len(df)):
                            x=df.iloc[j]

                            if x.low<=sl:
                                pnl=-MARGIN*risk/entry
                                hit=True
                                break

                            if x.high>=tp:
                                pnl=MARGIN*risk*RR/entry
                                hit=True
                                break

                        if hit:
                            pnl-=MARGIN*FEE*2
                            equity+=pnl
                            trades.append([sym,"LONG",pnl])


            # short sweep + BOS
            if side=="SHORT":

                if c.high>last_high and c.close<last_low:

                    entry=df.iloc[i+1].open*(1-SLIPPAGE)
                    sl=last_high+c.atr*0.2
                    risk=sl-entry

                    if risk>0:
                        tp=entry-risk*RR

                        for j in range(i+1,len(df)):
                            x=df.iloc[j]

                            if x.high>=sl:
                                pnl=-MARGIN*risk/entry
                                break

                            if x.low<=tp:
                                pnl=MARGIN*risk*RR/entry
                                break

                        pnl-=MARGIN*FEE*2
                        equity+=pnl
                        trades.append([sym,"SHORT",pnl])

    return pd.DataFrame(trades,columns=["Symbol","Side","PnL"]),equity


if __name__=="__main__":

    data={}

    for n,s in SYMBOLS.items():
        print("Loading",n)

        h1=indicators(fetch(s,"1h"))
        h4=indicators(fetch(s,"4h"))

        data[n]={
            "1h":h1,
            "4h":h4,
            "pivots":pivots(h1)
        }


    trades,balance=run_backtest(data)

    trades.to_csv("clts_trades.csv",index=False)

    print("="*60)
    print("CLTS PRO V3 FULL RESULTS")
    print("="*60)

    print("Trades:",len(trades))
    print("Final balance:",round(balance,2))

    if len(trades):
        print("Win rate:",
              round((trades.PnL>0).mean()*100,2))
        print(trades.groupby("Symbol").PnL.sum())
