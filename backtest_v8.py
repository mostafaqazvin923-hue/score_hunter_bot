# V153 LBank Futures Research Engine
# Causal research: no lookahead, no centered windows, no timeout.
import time
from pathlib import Path
from collections import defaultdict
import ccxt, numpy as np, pandas as pd

DAYS=365; WARMUP_DAYS=70; TIMEFRAME='15m'; LIMIT=1000
SYMBOLS=['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT','ADA/USDT:USDT','AVAX/USDT:USDT','LINK/USDT:USDT','DOGE/USDT:USDT','DOT/USDT:USDT']
TRADE_MARGIN=100.0; LEVERAGE=50; FEE_RATE=0.0007; SLIPPAGE=0.0003; RR=2.0; MAX_OPEN_POSITIONS=3
ATR_N=14; EMA_FAST=20; EMA_SLOW=50; EMA_TREND=200; VOL_N=20; BREAKOUT_LOOKBACK=20
FAMILIES=['TREND_BREAKOUT','BREAKOUT_PULLBACK','MOMENTUM_CONTINUATION','VOL_EXPANSION']; OUT=Path('.')

def exchange(): return ccxt.lbank({'enableRateLimit':True,'timeout':20000,'options':{'defaultType':'swap'}})
def fetch(ex,symbol,start_ms,end_ms):
    rows=[]; since=start_ms
    while since<end_ms:
        b=ex.fetch_ohlcv(symbol,TIMEFRAME,since=since,limit=LIMIT)
        if not b: break
        rows.extend(b); last=int(b[-1][0])
        if last<=since: break
        since=last+1; time.sleep(.2)
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
    d['timestamp']=pd.to_datetime(d['timestamp'],unit='ms',utc=True)
    d=d.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    tf=15*60*1000; last_complete=(ex.milliseconds()//tf)*tf-tf
    d=d[d.index<=pd.to_datetime(last_complete,unit='ms',utc=True)]
    return d.astype(float)

def atr(d,n=14):
    p=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def enrich(d):
    x=d.copy(); x['atr']=atr(x,ATR_N); x['atr_pct']=x.atr/x.close
    x['ema20']=x.close.ewm(span=EMA_FAST,adjust=False).mean(); x['ema50']=x.close.ewm(span=EMA_SLOW,adjust=False).mean(); x['ema200']=x.close.ewm(span=EMA_TREND,adjust=False).mean()
    x['vol_ma']=x.volume.rolling(VOL_N,min_periods=VOL_N).mean(); x['rvol']=x.volume/x.vol_ma
    x['range']=x.high-x.low; x['body']=(x.close-x.open).abs(); x['body_ratio']=x.body/x['range'].replace(0,np.nan)
    x['prior_high_20']=x.high.rolling(BREAKOUT_LOOKBACK,min_periods=BREAKOUT_LOOKBACK).max().shift(1); x['prior_low_20']=x.low.rolling(BREAKOUT_LOOKBACK,min_periods=BREAKOUT_LOOKBACK).min().shift(1)
    x['range_ma']=x['range'].rolling(20,min_periods=20).mean().shift(1); x['ema50_prev']=x.ema50.shift(4)
    return x

def candidates(symbol,d):
    x=enrich(d); test_start=d.index.max()-pd.Timedelta(days=DAYS); rows=[]
    for i in range(max(EMA_TREND+5,BREAKOUT_LOOKBACK+5),len(x)-1):
        ts=x.index[i]
        if ts<test_start: continue
        r=x.iloc[i]; p=x.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0: continue
        tl=r.close>r.ema50>r.ema200 and r.ema50>r.ema50_prev; ts_=r.close<r.ema50<r.ema200 and r.ema50<r.ema50_prev
        rl=r.close>r.prior_high_20; rs=r.close<r.prior_low_20
        q=np.isfinite(r.rvol) and r.rvol>=1 and np.isfinite(r.body_ratio) and r.body_ratio>=.45
        ml=r.close>r.open and r.body_ratio>=.55 and np.isfinite(r.range_ma) and r.range>=.90*r.range_ma
        ms=r.close<r.open and r.body_ratio>=.55 and np.isfinite(r.range_ma) and r.range>=.90*r.range_ma
        ve=np.isfinite(r.range_ma) and r.range>=1.25*r.range_ma and np.isfinite(r.rvol) and r.rvol>=1.15
        ll=p.prior_high_20; ls=p.prior_low_20
        pl=np.isfinite(ll) and p.close>ll and r.low<=ll*1.0015 and r.close>ll and r.close>r.open
        ps=np.isfinite(ls) and p.close<ls and r.high>=ls*.9985 and r.close<ls and r.close<r.open
        specs={'TREND_BREAKOUT':(tl and rl,ts_ and rs),'BREAKOUT_PULLBACK':(tl and pl,ts_ and ps),'MOMENTUM_CONTINUATION':(tl and ml,ts_ and ms),'VOL_EXPANSION':((tl or r.close>r.ema200) and ve and ml,(ts_ or r.close<r.ema200) and ve and ms)}
        raws={'TREND_BREAKOUT':(rl,rs),'BREAKOUT_PULLBACK':(rl or pl,rs or ps),'MOMENTUM_CONTINUATION':(tl,ts_),'VOL_EXPANSION':(ve,ve)}
        for fam in FAMILIES:
            for side,sig,raw in [('LONG',specs[fam][0],raws[fam][0]),('SHORT',specs[fam][1],raws[fam][1])]:
                if not raw: continue
                ei=i+1; entry=float(x.iloc[ei].open)
                if side=='LONG': sl=min(float(r.low),float(p.low))-.15*float(r.atr); risk=entry-sl; tp=entry+RR*risk
                else: sl=max(float(r.high),float(p.high))+.15*float(r.atr); risk=sl-entry; tp=entry-RR*risk
                if risk<=0 or not .001<=risk/entry<=.04: continue
                rows.append({'Family':fam,'Symbol':symbol,'SignalTime':ts,'EntryTime':x.index[ei],'Side':side,'Entry':entry,'SL':sl,'TP':tp,'RiskPct':risk/entry,'ATRpct':float(r.atr_pct),'RVOL':float(r.rvol) if np.isfinite(r.rvol) else np.nan,'RegimePass':int(tl if side=='LONG' else ts_),'QualityPass':int(q),'MomentumPass':int(ml if side=='LONG' else ms)})
    return pd.DataFrame(rows)

def execute(c,candles):
    if c.empty:return pd.DataFrame()
    c=c.sort_values(['EntryTime','Symbol']).reset_index(drop=True); active={}; done=[]; last_close={}; em=defaultdict(list)
    for i,r in c.iterrows(): em[r.EntryTime].append(i)
    times=sorted(set(t for d in candles.values() for t in d.index))
    for t in times:
        for sym in list(active):
            tr=active[sym]
            if t<=tr['EntryTime'] or t not in candles[sym].index: continue
            b=candles[sym].loc[t]
            slhit=float(b.low)<=tr['SL'] if tr['Side']=='LONG' else float(b.high)>=tr['SL']; tphit=float(b.high)>=tr['TP'] if tr['Side']=='LONG' else float(b.low)<=tr['TP']
            if slhit or tphit:
                result='LOSS' if slhit else 'WIN'; ep=tr['SL'] if slhit else tr['TP']
                gross=((ep-tr['Entry'])/tr['Entry'] if tr['Side']=='LONG' else (tr['Entry']-ep)/tr['Entry'])*TRADE_MARGIN*LEVERAGE
                fee=(tr['Entry']+ep)*TRADE_MARGIN*LEVERAGE/tr['Entry']*FEE_RATE; pnl=gross-fee-TRADE_MARGIN*LEVERAGE*SLIPPAGE*2
                z=dict(tr); z.update({'ExitTime':t,'Exit':ep,'Result':result,'PnL':pnl,'HoldingHours':(t-tr['EntryTime']).total_seconds()/3600}); done.append(z); del active[sym]; last_close[sym]=t
        for i in em.get(t,[]):
            r=c.iloc[i]; sym=r.Symbol
            if sym in active or last_close.get(sym)==t or len(active)>=MAX_OPEN_POSITIONS: continue
            active[sym]=r.to_dict()
    for sym,tr in active.items():
        z=dict(tr); z.update({'ExitTime':pd.NaT,'Exit':np.nan,'Result':'OPEN_AT_END','PnL':np.nan,'HoldingHours':np.nan}); done.append(z)
    return pd.DataFrame(done)

def summarize(t):
    if t is None or t.empty:return {'Trades':0,'Wins':0,'Losses':0,'WinRatePct':0,'NetPnL':0,'PF':0,'AvgWin':0,'AvgLoss':0,'MaxDD':0,'MaxConsecutiveLosses':0,'LossStreaks':'','TradesPerDay':0,'OpenAtEnd':0}
    c=t[t.Result.isin(['WIN','LOSS'])].copy(); op=int((t.Result=='OPEN_AT_END').sum())
    if c.empty: z=summarize(pd.DataFrame()); z['OpenAtEnd']=op; return z
    w=c[c.Result=='WIN']; l=c[c.Result=='LOSS']; p=c.PnL.astype(float); eq=p.cumsum(); dd=eq-eq.cummax(); cur=0; st=[]
    for r in c.Result:
        if r=='LOSS':cur+=1
        elif cur:st.append(cur);cur=0
    if cur:st.append(cur)
    elapsed=max(1,(c.EntryTime.max()-c.EntryTime.min()).total_seconds()/86400); gw=w.PnL.sum(); gl=l.PnL.sum()
    return {'Trades':len(c),'Wins':len(w),'Losses':len(l),'WinRatePct':100*len(w)/len(c),'NetPnL':p.sum(),'PF':gw/abs(gl) if gl<0 else np.inf,'AvgWin':w.PnL.mean() if len(w) else 0,'AvgLoss':l.PnL.mean() if len(l) else 0,'MaxDD':dd.min(),'MaxConsecutiveLosses':max(st) if st else 0,'LossStreaks':','.join(map(str,st)),'TradesPerDay':len(c)/elapsed,'OpenAtEnd':op}

def main():
    ex=exchange(); ex.load_markets(); now=pd.Timestamp.now(tz='UTC'); test_start=now-pd.Timedelta(days=DAYS); fetch_start=test_start-pd.Timedelta(days=WARMUP_DAYS); sm=int(fetch_start.timestamp()*1000); em=int(now.timestamp()*1000)
    candles={}; byfam={f:[] for f in FAMILIES}
    for sym in SYMBOLS:
        print('\nFetching',sym)
        try:
            d=fetch(ex,sym,sm,em)
            if d.empty: print(' NO DATA'); continue
            candles[sym]=d; cc=candidates(sym,d); print(' candles=',len(d))
            for f in FAMILIES:
                z=cc[cc.Family==f].copy(); byfam[f].append(z); print(' ',f,len(z))
        except Exception as e: print(' ERROR',type(e).__name__,e)
    if not candles: raise RuntimeError('NO LBank DATA RECEIVED')
    summaries=[]; stages=[]; frames=[]
    for f in FAMILIES:
        c=pd.concat(byfam[f],ignore_index=True) if byfam[f] else pd.DataFrame(); stages.append({'Family':f,'Candidates':len(c),'RegimePass':int(c.RegimePass.sum()) if not c.empty else 0,'QualityPass':int(c.QualityPass.sum()) if not c.empty else 0,'MomentumPass':int(c.MomentumPass.sum()) if not c.empty else 0})
        tr=execute(c,candles)
        if not tr.empty: frames.append(tr.assign(Family=f))
        s=summarize(tr); s['Family']=f; summaries.append(s); print('\n',f,s)
    summary=pd.DataFrame(summaries); stage=pd.DataFrame(stages); alltr=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    summary.to_csv('v153_summary.csv',index=False); stage.to_csv('v153_stage_counts.csv',index=False)
    if not alltr.empty:
        alltr.to_csv('v153_trades.csv',index=False); closed=alltr[alltr.Result.isin(['WIN','LOSS'])].copy()
        rows=[]
        for (f,sym),g in closed.groupby(['Family','Symbol']): z=summarize(g); z.update({'Family':f,'Symbol':sym}); rows.append(z)
        pd.DataFrame(rows).to_csv('v153_symbol_stats.csv',index=False)
        rows=[]
        for (f,side),g in closed.groupby(['Family','Side']): z=summarize(g); z.update({'Family':f,'Side':side}); rows.append(z)
        pd.DataFrame(rows).to_csv('v153_direction_stats.csv',index=False)
        closed['Month']=closed.ExitTime.dt.to_period('M').astype(str); rows=[]
        for (f,m),g in closed.groupby(['Family','Month']): z=summarize(g); z.update({'Family':f,'Month':m}); rows.append(z)
        pd.DataFrame(rows).to_csv('v153_monthly.csv',index=False)
    print('\nV153 COMPLETE | RR=1:2 | TIMEOUT=DISABLED | LOOKAHEAD=NONE | NEXT-OPEN ENTRY | MAX OPEN=3')
    print(summary.to_string(index=False)); print('\nSTAGES'); print(stage.to_string(index=False))

if __name__=='__main__': main()
