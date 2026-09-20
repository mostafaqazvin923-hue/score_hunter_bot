# CLTS_PRO_V3_Optimized_backtest.py
# Optimized LBank CCXT backtest base
# Pivot pre-calculation version

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

def fetch_ohlcv(symbol, timeframe):
    since = int((datetime.now()-timedelta(days=DAYS)).timestamp()*1000)
    data=[]
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
        since=batch[-1][0]+1
        if len(batch)<1000:
            break
    df=pd.DataFrame(data,columns=["ts","open","high","low","close","volume"])
    df["date"]=pd.to_datetime(df.ts,unit="ms")
    return df.sort_values("date").reset_index(drop=True)

def add_indicators(df):
    df["ema50"]=df.close.ewm(span=50,adjust=False).mean()
    df["ema200"]=df.close.ewm(span=200,adjust=False).mean()
    tr=pd.concat([
        df.high-df.low,
        abs(df.high-df.close.shift()),
        abs(df.low-df.close.shift())
    ],axis=1).max(axis=1)
    df["atr"]=tr.rolling(14).mean()
    return df

def calculate_pivots(df):
    highs={}
    lows={}
    for i in range(2,len(df)-2):
        if df.high.iloc[i]==max(df.high.iloc[i-2:i+3]):
            highs[i]=df.high.iloc[i]
        if df.low.iloc[i]==min(df.low.iloc[i-2:i+3]):
            lows[i]=df.low.iloc[i]
    return highs,lows

if __name__=="__main__":
    print("CLTS-PRO V3 OPTIMIZED")
    for name,symbol in SYMBOLS.items():
        print("Loading",name)
        h1=add_indicators(fetch_ohlcv(symbol,"1h"))
        piv=calculate_pivots(h1)
        print(name,"pivots",len(piv[0])+len(piv[1]))
