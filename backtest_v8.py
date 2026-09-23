
import time
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt


# ============================================================
# HUNTER-V150
# ICHIMOKU + SMART MONEY HYBRID ENGINE
# Causal / No Lookahead / Fixed RR 1:2
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {"defaultType": "swap"}
})

SYMBOLS = {
    "BTC":"BTC/USDT",
    "ETH":"ETH/USDT",
    "SOL":"SOL/USDT",
    "BNB":"BNB/USDT",
    "XRP":"XRP/USDT",
    "ADA":"ADA/USDT",
    "DOGE":"DOGE/USDT",
    "AVAX":"AVAX/USDT",
    "LINK":"LINK/USDT",
    "DOT":"DOT/USDT",
}

DAYS = 365
MARGIN = 100
LEVERAGE = 50
FEE = 0.0007
SLIPPAGE = 0.0003
MAX_POSITIONS = 3


def load_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=DAYS)
    since = int(start.timestamp()*1000)
    data=[]

    while True:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe="15m",
            since=since,
            limit=1000
        )
        if not batch:
            break
        data.extend(batch)
        last=batch[-1][0]
        if last <= since or len(batch)<1000:
            break
        since=last+1
        time.sleep(.2)

    if not data:
        return None

    df=pd.DataFrame(
        data,
        columns=["ts","Open","High","Low","Close","Volume"]
    )
    df["Date"]=pd.to_datetime(df.ts,unit="ms")
    df=df.set_index("Date")[["Open","High","Low","Close","Volume"]]
    df=df[~df.index.duplicated()]
    return df


def ichimoku(df):
    high9=df.High.rolling(9).max()
    low9=df.Low.rolling(9).min()

    high26=df.High.rolling(26).max()
    low26=df.Low.rolling(26).min()

    high52=df.High.rolling(52).max()
    low52=df.Low.rolling(52).min()

    df["tenkan"]=(high9+low9)/2
    df["kijun"]=(high26+low26)/2
    df["span_a"]=((df.tenkan+df.kijun)/2).shift(26)
    df["span_b"]=((high52+low52)/2).shift(26)

    return df


def prepare(df):

    df=df.copy()

    df["ATR"]=(df.High-df.Low).rolling(14).mean()

    # Volume confirmation
    df["vol_ma"]=df.Volume.rolling(30).mean()

    # Market structure
    df["swing_high"]=df.High.rolling(20).max().shift(1)
    df["swing_low"]=df.Low.rolling(20).min().shift(1)

    df["bos_up"]=df.Close>df.swing_high
    df["bos_down"]=df.Close<df.swing_low

    df=ichimoku(df)

    # Liquidity sweep
    df["sweep_low"]=(df.Low<df.swing_low) & (df.Close>df.swing_low)
    df["sweep_high"]=(df.High>df.swing_high) & (df.Close<df.swing_high)

    return df


def signal(df,i):

    r=df.iloc[i]

    if not np.isfinite(r.ATR):
        return None

    cloud_bull = (
        r.Close > r.span_a and
        r.Close > r.span_b and
        r.tenkan > r.kijun
    )

    cloud_bear = (
        r.Close < r.span_a and
        r.Close < r.span_b and
        r.tenkan < r.kijun
    )

    volume = r.Volume > r.vol_ma*1.2

    long = cloud_bull and r.sweep_low and r.bos_up and volume
    short = cloud_bear and r.sweep_high and r.bos_down and volume

    if long:
        return "LONG"
    if short:
        return "SHORT"
    return None


def backtest_symbol(symbol,df):

    trades=[]

    for i in range(100,len(df)-10):

        side=signal(df,i)

        if not side:
            continue

        entry=df.Close.iloc[i]

        atr=df.ATR.iloc[i]

        if side=="LONG":
            sl=entry-1.5*atr
            tp=entry+3*atr
        else:
            sl=entry+1.5*atr
            tp=entry-3*atr

        result="LOSS"
        exit_price=sl

        for j in range(i+1,len(df)):

            h=df.High.iloc[j]
            l=df.Low.iloc[j]

            if side=="LONG":
                if l<=sl:
                    break
                if h>=tp:
                    result="WIN"
                    exit_price=tp
                    break

            else:
                if h>=sl:
                    break
                if l<=tp:
                    result="WIN"
                    exit_price=tp
                    break

        pnl=((exit_price-entry)/entry if side=="LONG"
             else (entry-exit_price)/entry)

        pnl*=MARGIN*LEVERAGE
        pnl-=MARGIN*LEVERAGE*FEE*2

        trades.append({
            "Symbol":symbol,
            "Side":side,
            "Result":result,
            "PnL":pnl
        })

    return trades


def main():

    all_trades=[]

    for name,symbol in SYMBOLS.items():

        print("Loading",name)

        df=load_data(symbol)

        if df is None:
            continue

        df=prepare(df)

        all_trades += backtest_symbol(name,df)


    if not all_trades:
        print("No trades")
        return

    t=pd.DataFrame(all_trades)

    wins=t[t.Result=="WIN"]
    total=len(t)

    print("="*60)
    print("HUNTER-V150 REPORT")
    print("Trades:",total)
    print("Win Rate:",round(len(wins)/total*100,2))
    print("PnL:",round(t.PnL.sum(),2))
    print("="*60)

    print(t.groupby("Symbol").PnL.sum())


if __name__=="__main__":
    main()
