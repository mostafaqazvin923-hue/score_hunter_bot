#!/usr/bin/env python3
"""HUNTER-V6.1 - causal XT USDT-M Futures liquidity exhaustion/reclaim backtest."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SYMBOLS=['btc_usdt','eth_usdt','sol_usdt','sui_usdt','avax_usdt','near_usdt','ada_usdt','bnb_usdt','apt_usdt','crv_usdt','ondo_usdt','pendle_usdt','icp_usdt','wif_usdt']
DATA_DIR=Path('data/xt_futures_v6'); TOTAL_DAYS=365; WARMUP_DAYS=60
INITIAL_CAPITAL=1000.; MARGIN_PER_TRADE=100.; LEVERAGE=50.; RR=2.
FEE_RATE=.0007; SLIPPAGE=.0003; INTERVAL_MS=900_000
URL='https://fapi.xt.com/future/market/v1/public/q/kline'; LIMIT=1500
PIVOT_LEFT=2; PIVOT_RIGHT=2; SWEEP_MIN=.10; SWEEP_MAX=1.; SL_BUF=.20; VOL_MIN=1.05

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--data-dir',default=str(DATA_DIR)); return p.parse_args()

def rows_from(x):
 if isinstance(x,list): return x
 if isinstance(x,dict):
  for k in ('result','data','rows','list'):
   if k in x:
    y=rows_from(x[k])
    if y is not None:return y
 return None

def parse_row(k):
 if isinstance(k,(list,tuple)):
  if len(k)<6: raise ValueError('short OHLCV list')
  ts,o,h,l,c,v=k[:6]
 elif isinstance(k,dict):
  ts=k.get('t',k.get('time',k.get('timestamp'))); o=k.get('o',k.get('open')); h=k.get('h',k.get('high')); l=k.get('l',k.get('low')); c=k.get('c',k.get('close')); v=k.get('v',k.get('volume',k.get('baseVolume',k.get('quoteVolume'))))
  if v is None: raise ValueError('ambiguous/missing volume')
 else: raise ValueError('unsupported row')
 ts=int(ts); o,h,l,c,v=map(float,(o,h,l,c,v))
 if ts<=0 or not np.isfinite([o,h,l,c,v]).all() or min(o,h,l,c)<=0 or h<max(o,c) or l>min(o,c) or l>h or v<0: raise ValueError('invalid OHLCV')
 return [ts,o,h,l,c,v]

def validate(df,symbol,start,end):
 if df.empty: raise RuntimeError(f'{symbol}: empty dataset')
 df=df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
 a=df.timestamp.to_numpy(np.int64)
 if not np.all(np.diff(a)>0): raise RuntimeError(f'{symbol}: timestamps not strictly increasing')
 gaps=int(np.sum(np.diff(a)>INTERVAL_MS))
 if a[0]>start+2*INTERVAL_MS: raise RuntimeError(f'{symbol}: insufficient start coverage')
 if a[-1]<end-2*INTERVAL_MS: raise RuntimeError(f'{symbol}: stale end coverage')
 if gaps>50: raise RuntimeError(f'{symbol}: excessive gaps={gaps}')
 if not np.isfinite(df[['open','high','low','close','volume']].to_numpy(float)).all(): raise RuntimeError(f'{symbol}: nonfinite values')
 span=(a[-1]-a[0])/86400000
 print(f'Validation {symbol.upper()}: rows={len(df)}, first={pd.to_datetime(a[0],unit="ms",utc=True)}, last={pd.to_datetime(a[-1],unit="ms",utc=True)}, span={span:.1f}d, gaps={gaps}')
 return df

def fetch(symbol,data_dir,session):
 p=data_dir/f'{symbol.upper()}_15m.csv'; now=int(time.time()*1000); start=now-int((TOTAL_DAYS+WARMUP_DAYS)*86400000); end=(now//INTERVAL_MS)*INTERVAL_MS-INTERVAL_MS
 if p.exists() and p.stat().st_size>1000:
  try:
   d=pd.read_csv(p); d['timestamp']=pd.to_numeric(d.timestamp,errors='raise').astype('int64'); d=d[['timestamp','open','high','low','close','volume']]
   return validate(d,symbol,start,end)
  except Exception as e: print(f'[CACHE] invalid {symbol.upper()}: {e}; refetching')
 data_dir.mkdir(parents=True,exist_ok=True); cur=start; out=[]; page=0
 while cur<=end:
  page+=1; win=min(end,cur+LIMIT*INTERVAL_MS-1); params={'symbol':symbol,'interval':'15m','startTime':cur,'endTime':win,'limit':LIMIT}; last=None
  for attempt in range(4):
   try:
    r=session.get(URL,params=params,headers={'User-Agent':'HUNTER-V6.1'},timeout=20); r.raise_for_status(); raw=rows_from(r.json());
    if raw is None: raise RuntimeError('unexpected XT response envelope')
    break
   except Exception as e:
    last=e
    if attempt<3: time.sleep(1.5*(attempt+1))
  else: raise RuntimeError(f'{symbol}: API failure: {last}')
  if not raw: raise RuntimeError(f'{symbol}: empty page {page} before target end')
  parsed=[]; raw_ts=[]; failures=0
  for k in raw:
   try:
    x=parse_row(k); raw_ts.append(x[0]);
    if cur<=x[0]<=win: parsed.append(x)
   except Exception: failures+=1
  if not parsed:
   mx=max(raw_ts) if raw_ts else None
   if mx is not None and mx<cur: break
   raise RuntimeError(f'{symbol}: page {page} has no valid rows in requested range (failures={failures}, raw_max={mx}, cursor={cur})')
  out.extend(parsed); mx=max(x[0] for x in parsed); nxt=mx+INTERVAL_MS
  if nxt<=cur: raise RuntimeError(f'{symbol}: pagination stalled at page {page}')
  cur=nxt; time.sleep(.08)
 if not out: raise RuntimeError(f'{symbol}: no data fetched')
 d=pd.DataFrame(out,columns=['timestamp','open','high','low','close','volume']); d=d[(d.timestamp>=start)&(d.timestamp<=end)].copy(); d=validate(d,symbol,start,end); d.to_csv(p,index=False); return d

def resample(df,rule,n):
 x=df.copy(); x.index=pd.to_datetime(x.pop('timestamp'),unit='ms',utc=True); x=x.sort_index().loc[~x.index.duplicated(keep='last')]
 y=x.resample(rule,closed='left',label='right',origin='epoch').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),n=('close','count'))
 return y[y.n==n].drop(columns='n').dropna()

def build(df):
 h1=resample(df,'1h',4); h4=resample(df,'4h',16); d1=resample(df,'1d',96)
 prev=h1.close.shift(1); tr=pd.concat([h1.high-h1.low,(h1.high-prev).abs(),(h1.low-prev).abs()],axis=1).max(axis=1)
 h1['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); rng=(h1.high-h1.low).replace(0,np.nan); clv=((h1.close-h1.low)-(h1.high-h1.close))/rng; h1['svp']=clv.fillna(0)*h1.volume; h1['svp4']=h1.svp.shift(1).rolling(4,min_periods=4).sum(); h1['high24']=h1.high.shift(1).rolling(24,min_periods=24).max(); h1['low24']=h1.low.shift(1).rolling(24,min_periods=24).min(); med=h1.volume.shift(1).rolling(24,min_periods=24).median(); h1['volshock']=h1.volume.shift(1)/med.replace(0,np.nan)
 d1['ema50']=d1.close.shift(1).ewm(span=50,adjust=False,min_periods=50).mean(); d1['ema200']=d1.close.shift(1).ewm(span=200,adjust=False,min_periods=200).mean(); d1['slope']=d1.ema50.diff()
 hi=h4.high.to_numpy(float); lo=h4.low.to_numpy(float); ph=np.full(len(h4),np.nan); pl=np.full(len(h4),np.nan)
 for i in range(PIVOT_LEFT,len(h4)-PIVOT_RIGHT):
  if hi[i]>hi[i-PIVOT_LEFT:i].max() and hi[i]>=hi[i+1:i+1+PIVOT_RIGHT].max(): ph[i+PIVOT_RIGHT]=hi[i]
  if lo[i]<lo[i-PIVOT_LEFT:i].min() and lo[i]<=lo[i+1:i+1+PIVOT_RIGHT].min(): pl[i+PIVOT_RIGHT]=lo[i]
 h4['last_ph']=pd.Series(ph,index=h4.index).ffill(); h4['last_pl']=pd.Series(pl,index=h4.index).ffill(); return {'15m':df,'1h':h1,'4h':h4,'1d':d1}

def daily_at(d1,ts):
 i=d1.index.searchsorted(ts,side='right')-1; return None if i<0 else d1.iloc[i]
def struct(h4,ts):
 i=h4.index.searchsorted(ts,side='right')-1
 if i<0:return None
 a,b=h4.iloc[i].last_ph,h4.iloc[i].last_pl
 return (float(a),float(b)) if np.isfinite(a) and np.isfinite(b) else None

def backtest(all_data,test_start):
 times=sorted(set().union(*(d['1h'].index for d in all_data.values()))); arr={s:{'h1':d['1h'],'h4':d['4h'],'d1':d['1d']} for s,d in all_data.items()}; active=None; trades=[]; cooldown={s:None for s in arr}
 for ts in times:
  if active is not None and ts>=active['entry_ts'] and ts in arr[active['symbol']]['h1'].index:
   b=arr[active['symbol']]['h1'].loc[ts]; side=active['side']; sl,tp,e=active['sl'],active['tp'],active['entry']; hit_sl=(b.low<=sl if side=='LONG' else b.high>=sl); hit_tp=(b.high>=tp if side=='LONG' else b.low<=tp)
   if hit_sl or hit_tp:
    loss=hit_sl or (hit_sl and hit_tp); exitp=sl if loss else tp; notional=MARGIN_PER_TRADE*LEVERAGE; gross=((exitp-e)/e if side=='LONG' else (e-exitp)/e)*notional; pnl=gross-notional*FEE_RATE*2; trades.append({'symbol':active['symbol'],'side':side,'entry_ts':active['entry_ts'],'exit_ts':ts,'outcome':'LOSS' if loss else 'WIN','pnl':float(pnl),'entry':e,'exit':exitp}); cooldown[active['symbol']]=ts+pd.Timedelta(hours=3); active=None; continue
  if active is not None or ts<test_start: continue
  cand=[]
  for s,z in arr.items():
   h=z['h1'];
   if ts not in h.index or (cooldown[s] is not None and ts<cooldown[s]): continue
   i=h.index.get_loc(ts); b=h.iloc[i]; d=daily_at(z['d1'],ts); st=struct(z['h4'],ts)
   if i<30 or d is None or st is None: continue
   ema50,ema200,slope=d.ema50,d.ema200,d.slope; atr,svp,vs=b.atr,b.svp4,b.volshock
   if not np.isfinite([ema50,ema200,slope,atr,svp,vs]).all() or atr<=0: continue
   long=d.close>ema200 and ema50>ema200 and slope>0; short=d.close<ema200 and ema50<ema200 and slope<0
   sh,slv=st; low,high,close=float(b.low),float(b.high),float(b.close)
   if long and low<slv-SWEEP_MIN*atr and low>=slv-SWEEP_MAX*atr and close>slv and svp>0 and vs>=VOL_MIN: cand.append((min(vs-1,3)+(slv-low)/atr,s,'LONG',low-SL_BUF*atr))
   elif short and high>sh+SWEEP_MIN*atr and high<=sh+SWEEP_MAX*atr and close<sh and svp<0 and vs>=VOL_MIN: cand.append((min(vs-1,3)+(high-sh)/atr,s,'SHORT',high+SL_BUF*atr))
  if not cand: continue
  _,s,side,sl=cand[0] if len(cand)==1 else sorted(cand,key=lambda x:(-x[0],x[1]))[0]; h=arr[s]['h1']; future=h.index[h.index>ts]
  if len(future)==0: continue
  et=future[0]; raw=float(h.loc[et,'open']); e=raw*(1+SLIPPAGE if side=='LONG' else 1-SLIPPAGE); risk=e-sl if side=='LONG' else sl-e
  if not np.isfinite(risk) or risk<=0: continue
  active={'symbol':s,'side':side,'signal_ts':ts,'entry_ts':et,'entry':e,'sl':sl,'tp':e+RR*risk if side=='LONG' else e-RR*risk}
 return trades,active

def stats(ts):
 n=len(ts); w=sum(x['outcome']=='WIN' for x in ts); l=n-w; gp=sum(x['pnl'] for x in ts if x['pnl']>0); gl=-sum(x['pnl'] for x in ts if x['pnl']<0); eq=peak=INITIAL_CAPITAL; dd=ddp=0; streak=mxs=0
 for x in sorted(ts,key=lambda q:pd.Timestamp(q['exit_ts'])):
  eq+=x['pnl']; peak=max(peak,eq); dd=max(dd,peak-eq); ddp=max(ddp,(peak-eq)/peak*100 if peak>0 else 0); streak=streak+1 if x['outcome']=='LOSS' else 0; mxs=max(mxs,streak)
 return {'n':n,'w':w,'l':l,'wr':100*w/n if n else 0,'gp':gp,'gl':gl,'pf':gp/gl if gl else (float('inf') if gp else 0),'net':sum(x['pnl'] for x in ts),'dd':dd,'ddp':ddp,'streak':mxs,'avgw':gp/w if w else 0,'avgl':-gl/l if l else 0,'exp':sum(x['pnl'] for x in ts)/n if n else 0,'med':float(np.median([x['pnl'] for x in ts])) if ts else 0}

def main():
 a=parse_args(); data_dir=Path(a.data_dir); data_dir.mkdir(parents=True,exist_ok=True); sess=requests.Session(); all_data={}
 for s in SYMBOLS:
  try: all_data[s]=build(fetch(s,data_dir,sess))
  except Exception as e: print(f'[ABORT] {s.upper()}: {e}'); sys.exit(1)
 common_end=min(d['15m'].index.max() for d in all_data.values()); test_start=common_end-pd.Timedelta(days=TOTAL_DAYS); print(f'[TIMELINE] common_end={common_end}; test_start={test_start}; warmup=all prior data')
 trades,openp=backtest(all_data,test_start); r=stats(trades); print('\n'+'='*30+' PER-SYMBOL '+'='*30)
 for s in SYMBOLS:
  q=stats([x for x in trades if x['symbol']==s]); print(f'{s.upper():<12} trades={q["n"]:<4} WR={q["wr"]:6.2f}% PF={q["pf"]:5.2f} PnL=${q["net"]:9.2f}')
 print('\n'+'='*30+' PORTFOLIO '+'='*30); print(f'Total Trades         : {r["n"]}\nWins                 : {r["w"]}\nLosses               : {r["l"]}\nWin Rate             : {r["wr"]:.2f}%\nGross Profit         : ${r["gp"]:,.2f}\nGross Loss           : ${r["gl"]:,.2f}\nProfit Factor        : {r["pf"]:.2f}\nNet PnL              : ${r["net"]:,.2f}\nMax Drawdown ($)     : ${r["dd"]:,.2f}\nMax Drawdown (%)     : {r["ddp"]:.2f}%\nMax Consecutive Loss : {r["streak"]}\nTrades Per Day       : {r["n"]/TOTAL_DAYS:.4f}\nAverage Win          : ${r["avgw"]:,.2f}\nAverage Loss         : ${r["avgl"]:,.2f}\nExpectancy / Trade   : ${r["exp"]:,.2f}\nMedian Trade PnL     : ${r["med"]:,.2f}\nOpen Positions At End: {1 if openp else 0}')
 ok=r['wr']>50 and r['pf']>1.2 and r['net']>0 and r['streak']<=4 and r['n']>=150; print('\nACCEPTED:',ok)
if __name__=='__main__': main()
