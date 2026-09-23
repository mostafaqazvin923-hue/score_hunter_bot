# HUNTER-V149 - Ichimoku Institutional Pullback Backtest
# Single file version

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt

exchange = ccxt.lbank({
    'enableRateLimit': True,
    'timeout': 20000,
    'options': {'defaultType':'swap'}
})

SYMBOLS = {
'BTC':'BTC/USDT','ETH':'ETH/USDT','SOL':'SOL/USDT','BNB':'BNB/USDT',
'XRP':'XRP/USDT','ADA':'ADA/USDT','AVAX':'AVAX/USDT',
'LINK':'LINK/USDT','DOGE':'DOGE/USDT','DOT':'DOT/USDT'
}

DAYS=365
RR=2
ATR_MULT=1.5
MARGIN=100
LEV=50
FEE=0.0007


def fetch(symbol,start,end):
    data=[]
    since=int((start-timedelta(days=10)).timestamp()*1000)
    while since < int(end.timestamp()*1000):
        try:
            batch=exchange.fetch_ohlcv(symbol,'15m',since=since,limit=1000)
        except:
            break
        if not batch: break
        data.extend(batch)
        since=batch[-1][0]+1
        if len(batch)<1000: break
        time.sleep(.2)

    if not data:return None

    df=pd.DataFrame(data,columns=['ts','Open','High','Low','Close','Volume'])
    df['Date']=pd.to_datetime(df.ts,unit='ms')
    df=df.drop(columns=['ts']).drop_duplicates('Date').set_index('Date').sort_index()
    return df[(df.index>=start)&(df.index<=end)]


def ichi(df):
    x=df.copy()
    x['tenkan']=(x.High.rolling(9).max()+x.Low.rolling(9).min())/2
    x['kijun']=(x.High.rolling(26).max()+x.Low.rolling(26).min())/2
    x['span_a']=((x.tenkan+x.kijun)/2).shift(26)
    x['span_b']=((x.High.rolling(52).max()+x.Low.rolling(52).min())/2).shift(26)
    return x


def prepare(df):
    h1=df.resample('1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4=df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    h1=ichi(h1)
    h4=ichi(h4)

    df['ATR']=(df.High-df.Low).rolling(14).mean()
    df['VOLAVG']=df.Volume.rolling(20).mean()

    return df,h1,h4


def get_signal(t,df,h1,h4):
    a=h1[h1.index<t]
    b=h4[h4.index<t]

    if len(a)<60 or len(b)<60:return None

    c=a.iloc[-1]
    p=df.iloc[df.index.get_loc(t)-1]

    bull=c.Close>max(c.span_a,c.span_b) and c.tenkan>c.kijun
    bear=c.Close<min(c.span_a,c.span_b) and c.tenkan<c.kijun

    long_pull=p.Low<=c.kijun and p.Close>p.Open and p.Volume>p.VOLAVG if 'VOLAVG' in p else False
    short_pull=p.High>=c.kijun and p.Close<p.Open

    if bull and long_pull:return 'LONG'
    if bear and short_pull:return 'SHORT'
    return None


def run(dataset):
    trades=[]

    for sym,(df,h1,h4) in dataset.items():
        for i in range(100,len(df)-2):
            t=df.index[i]
            side=get_signal(t,df,h1,h4)

            if not side:continue

            entry=df.iloc[i+1].Open
            atr=df.iloc[i].ATR
            if not np.isfinite(atr):continue

            if side=='LONG':
                sl=entry-ATR_MULT*atr
                tp=entry+ATR_MULT*atr*RR
            else:
                sl=entry+ATR_MULT*atr
                tp=entry-ATR_MULT*atr*RR

            result='LOSS'
            price=sl

            for j in range(i+1,len(df)):
                r=df.iloc[j]
                if side=='LONG':
                    if r.Low<=sl:break
                    if r.High>=tp:
                        result='WIN';price=tp;break
                else:
                    if r.High>=sl:break
                    if r.Low<=tp:
                        result='WIN';price=tp;break

            ret=(price-entry)/entry if side=='LONG' else (entry-price)/entry
            pnl=MARGIN*LEV*ret-MARGIN*LEV*FEE*2

            trades.append([sym,side,result,pnl])

    return pd.DataFrame(trades,columns=['Symbol','Side','Result','PnL'])


def main():
    end=datetime.now()
    start=end-timedelta(days=DAYS)

    data={}

    for k,v in SYMBOLS.items():
        print('Loading',k)
        df=fetch(v,start,end)
        if df is not None:
            data[k]=prepare(df)

    out=run(data)

    print('='*60)
    print('HUNTER-V149 REPORT')
    print('Trades:',len(out))

    if len(out):
        print('Win Rate:',round((out.Result=='WIN').mean()*100,2))
        print('PnL:',round(out.PnL.sum(),2))
        print(out.groupby('Symbol').PnL.sum())

if __name__=='__main__':
    main()
