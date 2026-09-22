# HUNTER-V146 — Liquidity Sweep + 15m Structure Shift
# Full source generated in this response.
import sys, subprocess, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ccxt'])
    import ccxt

exchange = ccxt.lbank({'enableRateLimit': True, 'timeout': 20000, 'options': {'defaultType': 'swap'}})
SYMBOLS = {'CRV':'CRV/USDT','DOGE':'DOGE/USDT','ICP':'ICP/USDT','APT':'APT/USDT','PENDLE':'PENDLE/USDT','WIF':'WIF/USDT','ONDO':'ONDO/USDT','NEAR':'NEAR/USDT','SEI':'SEI/USDT','XLM':'XLM/USDT','ADA':'ADA/USDT','BNB':'BNB/USDT','SOL':'SOL/USDT','ETH':'ETH/USDT'}
DAYS=365; WARMUP_DAYS=15; SLIPPAGE=.0003; FEE_RATE=.0007; TRADE_MARGIN=100.; LEVERAGE=50.; RR=2.; MAX_OPEN_POSITIONS=3
H1_PIVOT_LEFT=2; H1_PIVOT_RIGHT=2; MICRO_LOOKBACK=8; SL_BUFFER_ATR=.15; MIN_BODY_ATR=.20; MIN_RVOL=.90; LOSS_STREAK_LIMIT=4

def fetch_chunk_data(symbol,start_dt,end_dt):
    since=int((start_dt-timedelta(days=WARMUP_DAYS)).timestamp()*1000); end=int(end_dt.timestamp()*1000); rows=[]; cur=since
    try:
        while cur<end:
            batch=exchange.fetch_ohlcv(symbol,timeframe='15m',since=cur,limit=1000)
            if not batch: break
            rows.extend(batch); last=batch[-1][0]
            if last<=cur: break
            cur=last+1
            if len(batch)<1000 or last>=end: break
            time.sleep(.2)
    except Exception as e:
        print(f'  ERROR {symbol}: {e}'); return None
    if not rows: return None
    df=pd.DataFrame(rows,columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Date']=pd.to_datetime(df['Timestamp'],unit='ms'); df=df[['Date','Open','High','Low','Close','Volume']].dropna().drop_duplicates('Date',keep='last').sort_values('Date').set_index('Date')
    tf_ms=15*60*1000; last_complete=(exchange.milliseconds()//tf_ms)*tf_ms-tf_ms
    df=df[df.index<=pd.to_datetime(last_complete,unit='ms')]
    return df[(df.index>=start_dt)&(df.index<=end_dt)]

def atr(df,p=14):
    pc=df['Close'].shift(1); tr=pd.concat([df['High']-df['Low'],(df['High']-pc).abs(),(df['Low']-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,adjust=False,min_periods=p).mean()

def adx(df,p=14):
    h,l,c=df['High'],df['Low'],df['Close']; pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    up=h.diff(); dn=-l.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=df.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=df.index)
    a=1/p; ats=tr.ewm(alpha=a,adjust=False,min_periods=p).mean(); ps=plus.ewm(alpha=a,adjust=False,min_periods=p).mean(); ms=minus.ewm(alpha=a,adjust=False,min_periods=p).mean()
    pdi=100*ps/ats.replace(0,np.nan); mdi=100*ms/ats.replace(0,np.nan); dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=a,adjust=False,min_periods=p).mean()

def prepare_data(df):
    df=df.copy(); df['ATR']=atr(df); df['BODY']=(df['Close']-df['Open']).abs(); df['RVOL']=df['Volume']/df['Volume'].rolling(20,min_periods=20).mean()
    df['Prev_Micro_High']=df['High'].rolling(MICRO_LOOKBACK,min_periods=MICRO_LOOKBACK).max().shift(1)
    df['Prev_Micro_Low']=df['Low'].rolling(MICRO_LOOKBACK,min_periods=MICRO_LOOKBACK).min().shift(1)
    h1=df.resample('1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(); h1['ATR']=atr(h1); h1['ADX']=adx(h1)
    w=H1_PIVOT_LEFT+H1_PIVOT_RIGHT+1
    rh=h1['High'].rolling(w,min_periods=w).max(); rl=h1['Low'].rolling(w,min_periods=w).min()
    ph=h1['High'].eq(rh)&h1['High'].gt(h1['High'].shift(1))&h1['High'].gt(h1['High'].shift(2))&h1['High'].ge(h1['High'].shift(-1))&h1['High'].ge(h1['High'].shift(-2))
    pl=h1['Low'].eq(rl)&h1['Low'].lt(h1['Low'].shift(1))&h1['Low'].le(h1['Low'].shift(-1))&h1['Low'].le(h1['Low'].shift(-2))
    h1['Last_Swing_High']=h1['High'].where(ph).shift(H1_PIVOT_RIGHT).ffill(); h1['Last_Swing_Low']=h1['Low'].where(pl).shift(H1_PIVOT_RIGHT).ffill()
    h4=df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(); h4['EMA50']=h4['Close'].ewm(span=50,adjust=False,min_periods=50).mean(); h4['EMA200']=h4['Close'].ewm(span=200,adjust=False,min_periods=200).mean(); h4['ADX']=adx(h4)
    h4['BULL']=(h4.Close>h4.EMA200)&(h4.EMA50>h4.EMA200)&(h4.ADX>=15); h4['BEAR']=(h4.Close<h4.EMA200)&(h4.EMA50<h4.EMA200)&(h4.ADX>=15)
    return df,h1,h4

def completed(h1,h4,t):
    a=h1[h1.index+pd.Timedelta(hours=1)<=t]; b=h4[h4.index+pd.Timedelta(hours=4)<=t]
    return (a.iloc[-1],b.iloc[-1]) if len(a) and len(b) else (None,None)

def generate_signals(symbol,df,h1,h4,start_dt,end_dt):
    out=[]; sweep=None; last_seen_h1=None
    for i in range(60,len(df)):
        t=df.index[i]
        if t<start_dt or t>end_dt: continue
        c=df.iloc[i-1]
        if not all(np.isfinite(c[x]) for x in ['ATR','RVOL','Prev_Micro_High','Prev_Micro_Low']): continue
        h1c,h4c=completed(h1,h4,t)
        if h1c is None: continue
        hc=h1[h1.index+pd.Timedelta(hours=1)<=t]
        htime=hc.index[-1]
        if htime!=last_seen_h1:
            last_seen_h1=htime; support=h1c['Last_Swing_Low']; resistance=h1c['Last_Swing_High']
            if pd.notna(support) and h1c['Low']<support and h1c['Close']>support: sweep={'direction':'LONG','time':htime,'extreme':float(h1c['Low'])}
            elif pd.notna(resistance) and h1c['High']>resistance and h1c['Close']<resistance: sweep={'direction':'SHORT','time':htime,'extreme':float(h1c['High'])}
        if sweep is None or t<=sweep['time']: continue
        long_ok=sweep['direction']=='LONG' and c['Close']>c['Prev_Micro_High'] and c['Close']>c['Open']
        short_ok=sweep['direction']=='SHORT' and c['Close']<c['Prev_Micro_Low'] and c['Close']<c['Open']
        if not(long_ok or short_ok) or c['BODY']<MIN_BODY_ATR*c['ATR'] or c['RVOL']<MIN_RVOL: continue
        side='LONG' if long_ok else 'SHORT'; raw=float(df.iloc[i]['Open']); entry=raw*(1+SLIPPAGE) if side=='LONG' else raw*(1-SLIPPAGE)
        sl=sweep['extreme']-SL_BUFFER_ATR*c['ATR'] if side=='LONG' else sweep['extreme']+SL_BUFFER_ATR*c['ATR']; risk=entry-sl if side=='LONG' else sl-entry
        if risk<=0: continue
        tp=entry+RR*risk if side=='LONG' else entry-RR*risk
        out.append({'Timestamp':t,'Symbol':symbol,'Side':side,'Entry':entry,'SL':sl,'TP':tp,'DF':df,'StartIdx':i,'SweepTime':sweep['time']}); sweep=None
    return out

def simulate(s):
    df=s['DF']; start=s['StartIdx']; side=s['Side']; entry=s['Entry']; sl=s['SL']; tp=s['TP']; outcome='OPEN_AT_END'; exitp=entry; ex=len(df)-1
    for j in range(start+1,len(df)):
        hi,lo=float(df.iloc[j].High),float(df.iloc[j].Low)
        if side=='LONG':
            if lo<=sl: outcome='LOSS'; exitp=sl; ex=j; break
            if hi>=tp: outcome='WIN'; exitp=tp; ex=j; break
        else:
            if hi>=sl: outcome='LOSS'; exitp=sl; ex=j; break
            if lo<=tp: outcome='WIN'; exitp=tp; ex=j; break
    pnl=0.0 if outcome=='OPEN_AT_END' else TRADE_MARGIN*LEVERAGE*((exitp-entry)/entry if side=='LONG' else (entry-exitp)/entry)-TRADE_MARGIN*LEVERAGE*FEE_RATE*2
    return {'Timestamp':s['Timestamp'],'ExitTimestamp':df.index[ex],'Symbol':s['Symbol'],'Side':side,'Entry':entry,'SL':sl,'TP':tp,'Outcome':outcome,'Dollar_PnL':pnl,'SweepTime':s['SweepTime']}

def run_portfolio(data,start_dt,end_dt):
    sigs=[]
    for sym,(df,h1,h4) in data.items(): sigs+=generate_signals(sym,df,h1,h4,start_dt,end_dt)
    sigs.sort(key=lambda x:(x['Timestamp'],x['Symbol'])); trades=[]; active=[]; streak=0
    for s in sigs:
        t=s['Timestamp']; active=[p for p in active if p['ExitTimestamp']>t]
        if any(p['Symbol']==s['Symbol'] for p in active) or len(active)>=MAX_OPEN_POSITIONS or streak>=LOSS_STREAK_LIMIT: continue
        tr=simulate(s); trades.append(tr)
        if tr['Outcome']=='LOSS': streak+=1
        elif tr['Outcome']=='WIN': streak=0
        if tr['Outcome']!='OPEN_AT_END': active.append(tr)
    return trades

def max_streak(seq):
    m=c=0
    for x in seq:
        if x=='LOSS': c+=1; m=max(m,c)
        elif x=='WIN': c=0
    return m

def report(trades):
    d=pd.DataFrame(trades); closed=d[d.Outcome.isin(['WIN','LOSS'])]
    print('='*80); print('HUNTER-V146 PERFORMANCE REPORT'); print('='*80); print(f'Total Signals Accepted: {len(d)}'); print(f'Closed Trades: {len(closed)}'); print(f'Open At Dataset End: {len(d)-len(closed)}')
    if closed.empty: return
    w=closed[closed.Outcome=='WIN']; l=closed[closed.Outcome=='LOSS']; gp=w.Dollar_PnL.sum(); gl=abs(l.Dollar_PnL.sum()); pf=gp/gl if gl else float('inf'); eq=closed.Dollar_PnL.cumsum(); dd=eq-eq.cummax()
    print(f'Win Rate: {100*len(w)/len(closed):.2f}%'); print(f'Loss Rate: {100*len(l)/len(closed):.2f}%'); print(f'Net PnL: ${closed.Dollar_PnL.sum():,.2f}'); print(f'Profit Factor: {pf:.2f}'); print(f'Average Win: ${w.Dollar_PnL.mean():,.2f}' if len(w) else 'Average Win: $0.00'); print(f'Average Loss: ${l.Dollar_PnL.mean():,.2f}' if len(l) else 'Average Loss: $0.00'); print(f'Trades / Day: {len(d)/DAYS:.2f}'); print(f'Maximum Consecutive Losses: {max_streak(closed.Outcome.tolist())}'); print(f'Max Drawdown: ${dd.min():,.2f}')
    r=closed.groupby('Symbol').agg(Trades=('Outcome','count'),Wins=('Outcome',lambda x:(x=='WIN').sum()),Losses=('Outcome',lambda x:(x=='LOSS').sum()),Net_PnL=('Dollar_PnL','sum')); r['Win_Rate_%']=(100*r.Wins/r.Trades).round(2); print('\n--- BY SYMBOL ---'); print(r.to_string())
    q=closed.groupby('Side').agg(Trades=('Outcome','count'),Wins=('Outcome',lambda x:(x=='WIN').sum()),Net_PnL=('Dollar_PnL','sum')); q['Win_Rate_%']=(100*q.Wins/q.Trades).round(2); print('\n--- BY DIRECTION ---'); print(q.to_string()); print('='*80)

def main():
    print('='*80); print('HUNTER-V146 BACKTEST START'); print('='*80); end=datetime.utcnow(); start=end-timedelta(days=DAYS); data={}
    for name,sym in SYMBOLS.items():
        print(f'Loading {name}...'); df=fetch_chunk_data(sym,start,end)
        if df is None or len(df)<1000: print(f'  Skipping {name}: insufficient data'); continue
        data[name]=prepare_data(df)
    if not data: print('No valid market data.'); return
    print('Running engine...'); trades=run_portfolio(data,start,end)
    if not trades: print('No trades generated.'); return
    report(trades)

if __name__=='__main__': main()
