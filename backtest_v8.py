# HUNTER-V148
# Ichimoku Professional Ranking Engine
# 30 Symbols | RR 1:2 | No lookahead

import time
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options":{"defaultType":"swap"}
})

SYMBOLS = {
"BTC":"BTC/USDT","ETH":"ETH/USDT","SOL":"SOL/USDT","XRP":"XRP/USDT",
"BNB":"BNB/USDT","ADA":"ADA/USDT","DOGE":"DOGE/USDT","AVAX":"AVAX/USDT",
"LINK":"LINK/USDT","DOT":"DOT/USDT","SUI":"SUI/USDT","NEAR":"NEAR/USDT",
"APT":"APT/USDT","ATOM":"ATOM/USDT","LTC":"LTC/USDT","BCH":"BCH/USDT",
"ETC":"ETC/USDT","FIL":"FIL/USDT","ICP":"ICP/USDT","ARB":"ARB/USDT",
"OP":"OP/USDT","INJ":"INJ/USDT","TIA":"TIA/USDT","AAVE":"AAVE/USDT",
"UNI":"UNI/USDT","MATIC":"MATIC/USDT","MKR":"MKR/USDT",
"RUNE":"RUNE/USDT","TRX":"TRX/USDT","XLM":"XLM/USDT"
}

DAYS=365
RR=2
MARGIN=100
LEVERAGE=50
FEE=0.0007


def fetch(symbol,start,end):
    data=[]
    since=int(start.timestamp()*1000)
    while True:
        try:
            batch=exchange.fetch_ohlcv(symbol,"15m",since=since,limit=1000)
        except:
            break
        if not batch:
            break
        data+=batch
        nxt=batch[-1][0]+1
        if nxt<=since or len(batch)<1000:
            break
        since=nxt
        time.sleep(.2)

    if not data:
        return None

    df=pd.DataFrame(data,columns=["ts","Open","High","Low","Close","Volume"])
    df["Date"]=pd.to_datetime(df.ts,unit="ms")
    df=df.drop_duplicates("Date").set_index("Date")
    return df.loc[start:end]


def ichimoku(df):
    x=df.copy()
    x["tenkan"]=(x.High.rolling(9).max()+x.Low.rolling(9).min())/2
    x["kijun"]=(x.High.rolling(26).max()+x.Low.rolling(26).min())/2
    x["senkou_a"]=((x.tenkan+x.kijun)/2).shift(26)
    x["senkou_b"]=((x.High.rolling(52).max()+x.Low.rolling(52).min())/2).shift(26)
    x["chikou"]=x.Close.shift(26)
    x["atr"]=(x.High-x.Low).rolling(14).mean()
    x["vol"]=x.Volume/x.Volume.rolling(20).mean()
    return x


def score(row):
    s=0
    if row.Close>max(row.senkou_a,row.senkou_b): s+=2
    if row.tenkan>row.kijun: s+=2
    if row.chikou>row.Close: s+=1
    if row.vol>1: s+=1
    return s


def run(df,symbol):
    trades=[]
    for i in range(60,len(df)-2):
        r=df.iloc[i-1]

        if score(r)<5:
            continue

        side="LONG" if r.tenkan>r.kijun else "SHORT"
        entry=df.iloc[i].Open
        atr=r.atr

        if not np.isfinite(atr):
            continue

        sl=entry-atr if side=="LONG" else entry+atr
        tp=entry+atr*RR if side=="LONG" else entry-atr*RR

        result=None
        price=None

        for j in range(i+1,len(df)):
            c=df.iloc[j]

            if side=="LONG":
                if c.Low<=sl:
                    result="LOSS";price=sl;break
                if c.High>=tp:
                    result="WIN";price=tp;break
            else:
                if c.High>=sl:
                    result="LOSS";price=sl;break
                if c.Low<=tp:
                    result="WIN";price=tp;break

        if result:
            pnl=((price-entry)/entry if side=="LONG" else (entry-price)/entry)*MARGIN*LEVERAGE
            pnl-=MARGIN*LEVERAGE*FEE*2
            trades.append([symbol,side,result,pnl])

    return trades


def main():
    end=datetime.utcnow()
    start=end-timedelta(days=DAYS)
    all=[]

    for n,s in SYMBOLS.items():
        print("Loading",n)
        df=fetch(s,start,end)
        if df is not None:
            all+=run(ichimoku(df),n)

    if not all:
        print("No trades")
        return

    t=pd.DataFrame(all,columns=["Symbol","Side","Outcome","PnL"])
    print("="*70)
    print("HUNTER-V148 REPORT")
    print("Trades:",len(t))
    print("Win Rate:",round((t.Outcome=="WIN").mean()*100,2))
    print("PnL:",round(t.PnL.sum(),2))
    print(t.groupby("Symbol").PnL.sum())
    print("="*70)


if __name__=="__main__":
    main()
