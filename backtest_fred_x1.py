#!/usr/bin/env python3
"""FRED-X1 research backtest. Past-only signals; next-hour-open entry; conservative intrabar exits."""
from __future__ import annotations
import argparse, time, heapq
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
import requests

BASE='https://fapi.binance.com'
SYMBOLS=['BTCUSDT','ETHUSDT','SOLUSDT','SUIUSDT','AVAXUSDT','NEARUSDT','ADAUSDT','BNBUSDT','APTUSDT','CRVUSDT','ONDOUSDT','PENDLEUSDT','ICPUSDT','WIFUSDT']
INITIAL_EQUITY=1000.0; MARGIN=100.0; LEVERAGE=50.0; NOTIONAL=MARGIN*LEVERAGE
RR=2.0; FEE=0.0007; SLIPPAGE=0.0003; MAX_GROSS_RISK=75.0
MIN_STOP=0.002; MAX_STOP=0.008; BASIS_WINDOW=24*90; FUNDING_WINDOW=90*3
MIN_FUNDING_HISTORY=30; TIMEOUT_BARS=72; TRAIN_FRAC=.60; VAL_FRAC=.20; MIN_OOS_TRADES=30
S=requests.Session(); S.headers['User-Agent']='FRED-X1-research/1.0'

def get(path, params, retries=5):
    for n in range(retries):
        try:
            r=S.get(BASE+path,params=params,timeout=30)
            if r.status_code in (418,429): time.sleep(2+n*2); continue
            r.raise_for_status(); return r.json()
        except Exception:
            if n==retries-1: raise
            time.sleep(1+n)
    raise RuntimeError('API retries exhausted')

def fetch_klines(path,key,symbol,start_ms,end_ms):
    rows=[]; cursor=start_ms; step=3600000
    while cursor<=end_ms:
        batch=get(path,{key:symbol,'interval':'1h','startTime':cursor,'endTime':end_ms,'limit':1500})
        if not batch: break
        rows.extend(batch); nxt=int(batch[-1][0])+step
        if nxt<=cursor: raise RuntimeError(f'No pagination progress {symbol} {path}')
        cursor=nxt
        if len(batch)<1500: break
        time.sleep(.08)
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows,columns=['open_ms','open','high','low','close','volume','close_ms','qv','trades','tbb','tbq','ignore'])
    d=d.drop_duplicates('open_ms').sort_values('open_ms')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['open_time']=pd.to_datetime(d.open_ms,unit='ms',utc=True); d['close_time']=pd.to_datetime(d.close_ms,unit='ms',utc=True)
    return d[['open_time','close_time','open','high','low','close','volume']].dropna()

def fetch_funding(symbol,start_ms,end_ms):
    rows=[]; cursor=start_ms
    while cursor<=end_ms:
        b=get('/fapi/v1/fundingRate',{'symbol':symbol,'startTime':cursor,'endTime':end_ms,'limit':1000})
        if not b: break
        rows.extend(b); nxt=int(b[-1]['fundingTime'])+1
        if nxt<=cursor: raise RuntimeError(f'No funding pagination progress {symbol}')
        cursor=nxt
        if len(b)<1000: break
        time.sleep(.08)
    if not rows: return pd.DataFrame(columns=['funding_time','funding_rate'])
    d=pd.DataFrame(rows).drop_duplicates('fundingTime').sort_values('fundingTime')
    d['funding_time']=pd.to_datetime(d.fundingTime.astype('int64'),unit='ms',utc=True); d['funding_rate']=pd.to_numeric(d.fundingRate,errors='coerce')
    return d[['funding_time','funding_rate']].dropna()

def past_percentile(values,window,min_periods):
    out=np.full(len(values),np.nan)
    for i,v in enumerate(values):
        hist=values[max(0,i-window):i]; hist=hist[np.isfinite(hist)]
        if len(hist)>=min_periods and np.isfinite(v): out[i]=np.mean(hist<=v)
    return out

def features(price,premium,funding):
    if price.empty or premium.empty or funding.empty: return pd.DataFrame()
    p=price.set_index('open_time').sort_index(); prem=premium.set_index('open_time').sort_index()
    x=p.join(prem[['close']].rename(columns={'close':'premium_index'}),how='inner').sort_index()
    x=x[~x.index.duplicated(keep='last')]; x['basis']=x.premium_index
    x['basis_pct']=past_percentile(x.basis.to_numpy(float),BASIS_WINDOW,500)
    x['basis_prev']=x.basis.shift(1)
    x['stop_pct']=x.close.pct_change().abs().shift(1).rolling(168,min_periods=72).quantile(.90).mul(2).clip(MIN_STOP,MAX_STOP)
    f=funding.sort_values('funding_time').reset_index(drop=True)
    f['funding_pct']=past_percentile(f.funding_rate.to_numpy(float),FUNDING_WINDOW,MIN_FUNDING_HISTORY)
    left=x.reset_index().rename(columns={'open_time':'bar_time'}).sort_values('bar_time')
    left=pd.merge_asof(left,f[['funding_time','funding_rate','funding_pct']].sort_values('funding_time'),left_on='bar_time',right_on='funding_time',direction='backward',allow_exact_matches=True)
    left=left.set_index('bar_time').sort_index(); left['basis_prev']=left.basis.shift(1)
    # Contrarian dislocation + first hourly movement back toward zero; no future observations used.
    left['long_signal']=(left.funding_pct<=.10)&(left.basis_pct<=.10)&(left.basis>left.basis_prev)
    left['short_signal']=(left.funding_pct>=.90)&(left.basis_pct>=.90)&(left.basis<left.basis_prev)
    return left.reset_index().rename(columns={'bar_time':'open_time'})

@dataclass
class Trade:
    symbol:str; side:str; signal_time:str; entry_time:str; exit_time:str
    entry:float; exit:float; stop:float; target:float; gross_pnl:float; fees:float
    funding_pnl:float; net_pnl:float; r_multiple_net:float; exit_reason:str; bars_held:int; split:str

def split_name(t,start,end):
    f=(t-start).total_seconds()/max((end-start).total_seconds(),1)
    return 'train' if f<TRAIN_FRAC else ('validation' if f<TRAIN_FRAC+VAL_FRAC else 'oos')

def simulate(symbol,df,start,end):
    out=[]
    if df.empty or len(df)<1000:return out
    df=df.sort_values('open_time').reset_index(drop=True)
    fund=df[['funding_time','funding_rate']].dropna().drop_duplicates('funding_time').itertuples(index=False,name=None)
    events=list(fund); ei=0; pos=None; blocked_idx=-1
    for i in range(1,len(df)):
        r=df.iloc[i]; prev=df.iloc[i-1]; t=r.open_time
        while ei<len(events) and events[ei][0]<=t:
            ft,fr=events[ei]
            if pos is not None and pos['entry_time']<ft<=t:
                pos['funding']+=(-1 if pos['side']=='LONG' else 1)*NOTIONAL*float(fr)
            ei+=1
        if pos is not None:
            sl=(r.low<=pos['stop']) if pos['side']=='LONG' else (r.high>=pos['stop'])
            tp=(r.high>=pos['target']) if pos['side']=='LONG' else (r.low<=pos['target'])
            reason=raw=None
            if sl: reason,raw='stop',pos['stop']
            elif tp: reason,raw='target',pos['target']
            elif i-pos['entry_idx']>=TIMEOUT_BARS: reason,raw='timeout',r.close
            if reason:
                exitpx=raw*(1-SLIPPAGE if pos['side']=='LONG' else 1+SLIPPAGE)
                direction=1 if pos['side']=='LONG' else -1
                gross=direction*(exitpx-pos['entry'])/pos['entry']*NOTIONAL
                fees=NOTIONAL*FEE*2; net=gross-fees+pos['funding']
                out.append(Trade(symbol,pos['side'],str(pos['signal_time']),str(pos['entry_time']),str(t),float(pos['entry']),float(exitpx),float(pos['stop']),float(pos['target']),float(gross),float(fees),float(pos['funding']),float(net),float(net/max(pos['risk'],1e-9)),reason,i-pos['entry_idx'],split_name(t,start,end)))
                pos=None; blocked_idx=i
        # Only a previous fully closed candle can signal; no same-candle re-entry.
        if pos is None and i>blocked_idx and i<len(df)-1:
            side='LONG' if bool(prev.get('long_signal',False)) else ('SHORT' if bool(prev.get('short_signal',False)) else None)
            sp=prev.get('stop_pct',np.nan)
            if side and np.isfinite(sp):
                entry=float(r.open)*(1+SLIPPAGE if side=='LONG' else 1-SLIPPAGE); sp=float(np.clip(sp,MIN_STOP,MAX_STOP)); risk=NOTIONAL*sp
                if risk<=MAX_GROSS_RISK:
                    dist=entry*sp
                    stop,target=(entry-dist,entry+RR*dist) if side=='LONG' else (entry+dist,entry-RR*dist)
                    pos={'side':side,'entry':entry,'entry_time':t,'entry_idx':i,'signal_time':prev.open_time,'stop':stop,'target':target,'funding':0.0,'risk':risk}
    return out

def enforce_portfolio_capacity(trades, max_positions=10):
    """Conservatively cap simultaneous fixed-$100-margin trades at 10 (initial $1k equity)."""
    accepted=[]; active=[]
    for tr in sorted(trades, key=lambda z: (pd.Timestamp(z.entry_time), z.symbol)):
        entry=pd.Timestamp(tr.entry_time); exit_t=pd.Timestamp(tr.exit_time)
        while active and active[0] <= entry:
            heapq.heappop(active)
        if len(active) >= max_positions:
            continue
        accepted.append(tr); heapq.heappush(active, exit_t)
    return accepted

def metrics(d):
    if d.empty:return {'trades':0,'wins':0,'losses':0,'win_rate_pct':0.0,'net_pnl':0.0,'final_equity':INITIAL_EQUITY,'profit_factor':0.0,'max_loss_streak':0,'max_drawdown_pct':0.0,'fees_total':0.0,'funding_pnl_total':0.0}
    pnl=d.net_pnl.astype(float); wins=int((pnl>0).sum()); losses=int((pnl<=0).sum()); gp=float(pnl[pnl>0].sum()); gl=float(-pnl[pnl<0].sum()); streak=mx=0
    for v in pnl:
        streak=streak+1 if v<=0 else 0; mx=max(mx,streak)
    eq=INITIAL_EQUITY+d.sort_values('exit_time').net_pnl.cumsum(); peak=eq.cummax().clip(lower=INITIAL_EQUITY); dd=((peak-eq)/peak.replace(0,np.nan)).fillna(0)
    return {'trades':len(d),'wins':wins,'losses':losses,'win_rate_pct':round(100*wins/len(d),3),'net_pnl':round(float(pnl.sum()),2),'final_equity':round(float(INITIAL_EQUITY+pnl.sum()),2),'profit_factor':round(gp/gl,4) if gl else None,'max_loss_streak':mx,'max_drawdown_pct':round(100*float(dd.max()),3),'fees_total':round(float(d.fees.sum()),2),'funding_pnl_total':round(float(d.funding_pnl.sum()),2)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--days',type=int,default=1095); ap.add_argument('--out',default='fred_x1_results'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); now=pd.Timestamp.now(tz='UTC').floor('h'); end=now-pd.Timedelta(hours=1); start=end-pd.Timedelta(days=a.days)
    sm=int(start.timestamp()*1000); em=int((end+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)).timestamp()*1000)
    print(f'FRED-X1 {start} -> {end}; {len(SYMBOLS)} symbols; ${NOTIONAL:.0f} notional',flush=True)
    trades=[]; quality=[]
    for j,sym in enumerate(SYMBOLS,1):
        print(f'[{j}/{len(SYMBOLS)}] {sym}',flush=True)
        try:
            p=fetch_klines('/fapi/v1/klines','symbol',sym,sm,em)
            prem=fetch_klines('/fapi/v1/premiumIndexKlines','symbol',sym,sm,em)
            f=fetch_funding(sym,sm,em)
            if p.empty or prem.empty or f.empty:
                quality.append({'symbol':sym,'status':'FAIL_MISSING_CORE_DATA','price_rows':len(p),'premium_rows':len(prem),'funding_rows':len(f)}); continue
            p=p[p.close_time<=end]; prem=prem[prem.close_time<=end]; d=features(p,prem,f)
            if d.empty:
                quality.append({'symbol':sym,'status':'FAIL_ALIGNMENT'}); continue
            gaps=int((d.open_time.diff().dropna()>pd.Timedelta(hours=1)).sum())
            quality.append({'symbol':sym,'status':'OK','price_rows':len(p),'premium_rows':len(prem),'funding_rows':len(f),'aligned_rows':len(d),'hour_gaps':gaps,'first_bar':str(d.open_time.min()),'last_bar':str(d.open_time.max())})
            st=simulate(sym,d,start,end); trades.extend(st); print(f'  bars={len(d)} funding={len(f)} gaps={gaps} trades={len(st)}',flush=True)
        except Exception as e: quality.append({'symbol':sym,'status':'ERROR','error':repr(e)}); print(f'  ERROR {e}',flush=True)
        time.sleep(.12)
    trades=enforce_portfolio_capacity(trades, max_positions=max(1, int(INITIAL_EQUITY//MARGIN)))
    td=pd.DataFrame([asdict(t) for t in trades])
    if not td.empty:
        td['exit_time']=pd.to_datetime(td.exit_time,utc=True); td['entry_time']=pd.to_datetime(td.entry_time,utc=True); td=td.sort_values(['exit_time','symbol']).reset_index(drop=True); td['equity']=INITIAL_EQUITY+td.net_pnl.cumsum(); td.to_csv(out/'trades.csv',index=False)
        rows=[]
        for split in ['train','validation','oos','all']:
            sub=td if split=='all' else td[td.split==split]; m=metrics(sub); m.update({'split':split,'acceptance_wr_ge_50':m['win_rate_pct']>=50,'acceptance_streak_le_4':m['max_loss_streak']<=4,'fixed_rr_1_to_2':True,'oos_sample_sufficient':len(sub)>=MIN_OOS_TRADES if split=='oos' else None}); rows.append(m)
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
        pd.DataFrame([{'symbol':sym,**metrics(sub)} for sym,sub in td.groupby('symbol')]).to_csv(out/'by_symbol.csv',index=False)
        oos=td[td.split=='oos']; m=metrics(oos)
        with open(out/'REPORT.txt','w',encoding='utf-8') as f:
            f.write('FRED-X1 BACKTEST — RESEARCH ONLY\n'); f.write(f'Period: {start} to {end}\nInitial equity=${INITIAL_EQUITY}; margin=${MARGIN}; leverage={LEVERAGE}x; notional=${NOTIONAL}\nRR=1:2; fees each side={FEE}; slippage each side={SLIPPAGE}; max gross stop risk=${MAX_GROSS_RISK}\n\nOOS metrics:\n')
            for k,v in m.items(): f.write(f'{k}: {v}\n')
            f.write(f'\nAcceptance criteria, not guarantees: WR >=50%, max losing streak <=4, RR=1:2. Minimum OOS sample target={MIN_OOS_TRADES}.\n')
            f.write('If stop and target are both touched in one candle, stop is assumed first. Entry occurs next hour open. Funding charged at observed settlement times while position is open.\n')
            f.write('Portfolio capacity is capped at 10 simultaneous $100-margin positions based on initial $1,000 equity; capacity does not dynamically shrink after losses. Limitations: no liquidation engine, maintenance-margin tiers, order-book spread/impact, partial fills, or mark-to-market liquidation simulation. Do not deploy live based on this report alone.\n')
    else:
        pd.DataFrame([{'split':'all',**metrics(pd.DataFrame())}]).to_csv(out/'summary.csv',index=False)
        (out/'REPORT.txt').write_text('FRED-X1 BACKTEST — NO CLOSED TRADES\nNo trades generated; this is an insufficient/failed test, not evidence of profitability.\n',encoding='utf-8')
    pd.DataFrame(quality).to_csv(out/'data_quality.csv',index=False)
    print(f'Finished: {out.resolve()}')
if __name__=='__main__': main()
