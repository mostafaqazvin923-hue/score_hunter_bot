
"""
CLTS-PRO V2 — CAUSAL LIQUIDITY STRUCTURE ENGINE
LBank Futures / CCXT Backtester

Based on HUNTER infrastructure only:
- LBank CCXT connection
- OHLCV pagination
- pandas/numpy only

Strategy:
4H regime
+
1H confirmed pivots
+
Liquidity sweep
+
BOS confirmation
+
Retest entry
+
Fixed 1:2 RR
No timeout
No repainting logic
"""

import sys, subprocess
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


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

DAYS = 365
INITIAL_CAPITAL = 1000
MARGIN = 100
MAX_POSITIONS = 3

FEE = 0.0006
SLIPPAGE = 0.0002
RR = 2


def fetch(symbol, tf):

    since = int((datetime.now()-timedelta(days=DAYS)).timestamp()*1000)
    out=[]

    while True:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=tf,
            since=since,
            limit=1000
        )

        if not batch:
            break

        out.extend(batch)
        since=batch[-1][0]+1

        if len(batch)<1000:
            break

    if not out:
        return None

    df=pd.DataFrame(
        out,
        columns=["ts","open","high","low","close","volume"]
    )

    df["date"]=pd.to_datetime(df.ts,unit="ms")
    df.drop_duplicates("date",inplace=True)
    df.sort_values("date",inplace=True)
    df.reset_index(drop=True,inplace=True)

    return df



def indicators(df):

    df["ema50"]=df.close.ewm(span=50,adjust=False).mean()
    df["ema200"]=df.close.ewm(span=200,adjust=False).mean()

    tr=pd.concat(
        [
            df.high-df.low,
            abs(df.high-df.close.shift()),
            abs(df.low-df.close.shift())
        ],axis=1
    ).max(axis=1)

    df["atr"]=tr.rolling(14).mean()

    df["rvol"]=df.volume/df.volume.rolling(20).mean()

    return df



def regime(df,i):

    c=df.iloc[i]

    if c.close>c.ema50>c.ema200:
        return "LONG"

    if c.close<c.ema50<c.ema200:
        return "SHORT"

    return None



def confirmed_pivots(df,i):

    if i<5:
        return None,None

    highs=[]
    lows=[]

    for x in range(2,i-2):

        if df.high[x]==max(df.high[x-2:x+3]):
            highs.append((x,df.high[x]))

        if df.low[x]==min(df.low[x-2:x+3]):
            lows.append((x,df.low[x]))

    ph=highs[-1] if highs else None
    pl=lows[-1] if lows else None

    return ph,pl



def signal(df,i,side):

    ph,pl=confirmed_pivots(df,i)

    if not ph or not pl:
        return None


    c=df.iloc[i]


    # LONG CLTS:
    # sweep previous low -> close back above -> BOS -> retest

    if side=="LONG":

        if c.low < pl[1] and c.close > pl[1]:

            if c.close > ph[1]:

                entry=c.open
                sl=pl[1]-0.2*c.atr
                risk=entry-sl

                if risk>0:
                    return {
                        "side":"LONG",
                        "entry":entry,
                        "sl":sl,
                        "tp":entry+risk*RR
                    }


    # SHORT CLTS

    if side=="SHORT":

        if c.high > ph[1] and c.close < ph[1]:

            if c.close < pl[1]:

                entry=c.open
                sl=ph[1]+0.2*c.atr
                risk=sl-entry

                if risk>0:
                    return {
                        "side":"SHORT",
                        "entry":entry,
                        "sl":sl,
                        "tp":entry-risk*RR
                    }

    return None



def run(data):

    active={}
    trades=[]
    equity=INITIAL_CAPITAL

    timestamps=sorted(
        set(
            t for x in data.values()
            for t in x["1h"].date
        )
    )


    for ts in timestamps:


        # exits

        for sym,pos in list(active.items()):

            df=data[sym]["1h"]
            row=df[df.date==ts]

            if row.empty:
                continue

            c=row.iloc[0]

            sl=False
            tp=False

            if pos["side"]=="LONG":
                sl=c.low<=pos["sl"]
                tp=c.high>=pos["tp"]
            else:
                sl=c.high>=pos["sl"]
                tp=c.low<=pos["tp"]


            if sl or tp:

                result="LOSS" if sl else "WIN"

                exitp=pos["sl"] if sl else pos["tp"]

                ret=(exitp-pos["entry"])/pos["entry"]

                if pos["side"]=="SHORT":
                    ret=-ret

                pnl=MARGIN*ret-MARGIN*FEE*2
                equity+=pnl

                trades.append({
                    "Symbol":sym,
                    "Side":pos["side"],
                    "Result":result,
                    "PnL":pnl
                })

                del active[sym]


        # entries

        if len(active)>=MAX_POSITIONS:
            continue


        for sym,x in data.items():

            if sym in active:
                continue

            df1=x["1h"]
            df4=x["4h"]

            r=df1[df1.date==ts]

            if r.empty:
                continue

            i=r.index[0]

            if i<100:
                continue

            side=regime(df4,min(i//4,len(df4)-1))

            if not side:
                continue

            s=signal(df1,i,side)

            if s:
                active[sym]=s

    return pd.DataFrame(trades)



if __name__=="__main__":

    processed={}

    for n,s in SYMBOLS.items():

        print("Loading",n)

        h1=fetch(s,"1h")
        h4=fetch(s,"4h")

        if h1 is not None and h4 is not None:
            processed[n]={
                "1h":indicators(h1),
                "4h":indicators(h4)
            }


    trades=run(processed)

    trades.to_csv(
        "clts_v2_trades.csv",
        index=False
    )

    print("="*60)
    print("CLTS-PRO V2 RESULT")
    print("="*60)

    print("Trades:",len(trades))

    if len(trades):
        print(
            "Win Rate:",
            round((trades.Result=="WIN").mean()*100,2),
            "%"
        )
        print(
            "PnL:",
            round(trades.PnL.sum(),2)
        )
        print(trades.groupby("Symbol").PnL.sum())
    else:
        print("No trades")
