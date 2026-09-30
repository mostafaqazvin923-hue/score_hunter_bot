#!/usr/bin/env python3
from __future__ import annotations
import json,time
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,requests

DAYS=455; WARMUP_DAYS=30; TOTAL_DAYS=DAYS+WARMUP_DAYS
TIMEFRAME='1h'; BAR_MS=3600000; WINDOW_BARS=900; MIN_COVERAGE=.985
ATR_PERIOD=20; SL_ATR=1.; TP_ATR=2.
RV_FAST=24; RV_SLOW=120; COMPRESSION_RATIO=.75
BREAKOUT_LOOKBACK=24; TR_MEDIAN_WINDOW=20; RANGE_EXPANSION_MIN=1.25
VOLUME_Z_WINDOW=48; VOLUME_Z_MIN=.50
LONG_CLOSE_LOCATION_MIN=.70; SHORT_CLOSE_LOCATION_MAX=.30
EMA_FAST=24; EMA_SLOW=96
FEE_RATE=.0007; SLIPPAGE=.0003
INITIAL_CAPITAL=1000.; MARGIN_PER_TRADE=100.; LEVERAGE=50.; NOTIONAL=5000.
SYMBOLS=['btc_usdt','eth_usdt','sol_usdt','sui_usdt','avax_usdt','near_usdt','ada_usdt','bnb_usdt','apt_usdt','crv_usdt','ondo_usdt','pendle_usdt','icp_usdt','wif_usdt']
URL='https://fapi.xt.com/future/market/v1/public/q/kline'
OUT=Path('reports/xt_v31_stage0_compression_breakout'); OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='V31-Research/1.0'

@dataclass
class Trade:
    symbol:str; signal_time:str; entry_time:str; direction:str; entry:float; sl:float; tp:float
    exit_time:str; exit_price:float; outcome:str; gross_R:float; net_R:float; split:str

def fetch_window(symbol,start_ms,end_ms):
    p={'symbol':symbol,'interval':TIMEFRAME,'startTime':int(start_ms),'endTime':int(end_ms),'limit':WINDOW_BARS}
    for a in range(3):
        try:
            r=S.get(URL,params=p,timeout=30); r.raise_for_status(); rows=r.json().get('result',[]); out=[]
            for z in rows:
                try: out.append({'timestamp':int(z['t']),'open':float(z['o']),'high':float(z['h']),'low':float(z['l']),'close':float(z['c']),'volume':float(z['v'])})
                except: pass
            if not out:return pd.DataFrame()
            d=pd.DataFrame(out).drop_duplicates('timestamp').sort_values('timestamp'); d['timestamp']=pd.to_datetime(d.timestamp,unit='ms',utc=True); return d.reset_index(drop=True)
        except Exception as e:
            if a==2: raise RuntimeError(f'XT fetch failed {symbol}: {e}')
            time.sleep(1+a)

def fetch_symbol(symbol,end_ms):
    start=end_ms-TOTAL_DAYS*24*60*60*1000; cur=start; parts=[]
    while cur<end_ms:
        nxt=min(end_ms,cur+WINDOW_BARS*BAR_MS); d=fetch_window(symbol,cur,nxt)
        if not d.empty: parts.append(d)
        cur=nxt; time.sleep(.08)
    if not parts: raise RuntimeError(f'No data for {symbol}')
    d=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    now=pd.Timestamp.now(tz='UTC')
    if not d.empty and d.timestamp.iloc[-1]+pd.Timedelta(hours=1)>now: d=d.iloc[:-1].copy()
    exp=TOTAL_DAYS*24; cov=len(d)/exp; gaps=int(((d.timestamp.diff().dropna().dt.total_seconds()/3600)>1.01).sum())
    if cov<MIN_COVERAGE or gaps: raise RuntimeError(f"Coverage failure {symbol}: rows={len(d)}, expected={exp}, coverage={cov:.4f}, gaps={gaps}, first={d.timestamp.min() if len(d) else None}, last={d.timestamp.max() if len(d) else None}")
    return d

def features(d):
    x=d.copy(); pc=x.close.shift(1)
    x['tr']=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x['atr']=x.tr.ewm(alpha=1/ATR_PERIOD,adjust=False,min_periods=ATR_PERIOD).mean()
    ret=np.log(x.close/x.close.shift(1)); rv=ret.rolling(RV_FAST,min_periods=RV_FAST).std(); med=rv.rolling(RV_SLOW,min_periods=RV_SLOW).median()
    x['compression']=rv.shift(1)<=COMPRESSION_RATIO*med.shift(1)
    x['prior_high']=x.high.shift(1).rolling(BREAKOUT_LOOKBACK,min_periods=BREAKOUT_LOOKBACK).max()
    x['prior_low']=x.low.shift(1).rolling(BREAKOUT_LOOKBACK,min_periods=BREAKOUT_LOOKBACK).min()
    base=x.tr.shift(1).rolling(TR_MEDIAN_WINDOW,min_periods=TR_MEDIAN_WINDOW).median(); x['range_expansion']=x.tr/base
    vm=x.volume.shift(1).rolling(VOLUME_Z_WINDOW,min_periods=VOLUME_Z_WINDOW).mean(); vs=x.volume.shift(1).rolling(VOLUME_Z_WINDOW,min_periods=VOLUME_Z_WINDOW).std(); x['volume_z']=(x.volume-vm)/vs.replace(0,np.nan)
    cr=(x.high-x.low).replace(0,np.nan); x['close_location']=(x.close-x.low)/cr
    ef=x.close.ewm(span=EMA_FAST,adjust=False,min_periods=EMA_FAST).mean().shift(1); es=x.close.ewm(span=EMA_SLOW,adjust=False,min_periods=EMA_SLOW).mean().shift(1)
    x['long_signal']=x.compression&(x.close>x.prior_high)&(x.range_expansion>=RANGE_EXPANSION_MIN)&(x.volume_z>=VOLUME_Z_MIN)&(x.close_location>=LONG_CLOSE_LOCATION_MIN)&(ef>=es)
    x['short_signal']=x.compression&(x.close<x.prior_low)&(x.range_expansion>=RANGE_EXPANSION_MIN)&(x.volume_z>=VOLUME_Z_MIN)&(x.close_location<=SHORT_CLOSE_LOCATION_MAX)&(ef<=es)
    return x

def split_for(ts,start,end):
    f=(ts-start).total_seconds()/(end-start).total_seconds()
    return 'DISCOVERY' if f<.5 else ('DEVELOPMENT' if f<.75 else 'VALIDATION')

def simulate(x,i,direction,split,symbol):
    ei=i+1
    if ei>=len(x): return None
    atr=float(x.atr.iloc[i]); raw=float(x.open.iloc[ei])
    if not np.isfinite(atr) or atr<=0 or not np.isfinite(raw) or raw<=0:return None
    entry=raw*(1+SLIPPAGE if direction=='LONG' else 1-SLIPPAGE)
    sl=entry-SL_ATR*atr if direction=='LONG' else entry+SL_ATR*atr
    tp=entry+TP_ATR*atr if direction=='LONG' else entry-TP_ATR*atr
    for j in range(ei,len(x)):
        hi=float(x.high.iloc[j]); lo=float(x.low.iloc[j])
        if direction=='LONG':
            hsl=lo<=sl; htp=hi>=tp
            if not(hsl or htp):continue
            win=htp and not hsl; exitp=tp*(1-SLIPPAGE) if win else sl*(1-SLIPPAGE)
        else:
            hsl=hi>=sl; htp=lo<=tp
            if not(hsl or htp):continue
            win=htp and not hsl; exitp=tp*(1+SLIPPAGE) if win else sl*(1+SLIPPAGE)
        gross=2. if win else -1.
        qty=NOTIONAL/entry; one_r=qty*abs(entry-sl); costs=(NOTIONAL*FEE_RATE*2+NOTIONAL*SLIPPAGE*2)/one_r
        return Trade(symbol,x.timestamp.iloc[i].isoformat(),x.timestamp.iloc[ei].isoformat(),direction,entry,sl,tp,x.timestamp.iloc[j].isoformat(),exitp,'WIN' if win else 'LOSS',gross,gross-costs,split)
    return None

def main():
    end_ms=int(datetime.now(timezone.utc).timestamp()*1000); raw={}; cov=[]
    for sym in SYMBOLS:
        print('[FETCH]',sym); d=fetch_symbol(sym,end_ms); raw[sym]=d
        gaps=int(((d.timestamp.diff().dropna().dt.total_seconds()/3600)>1.01).sum()); cov.append({'symbol':sym,'rows':len(d),'expected_rows':TOTAL_DAYS*24,'coverage':len(d)/(TOTAL_DAYS*24),'gaps_gt_1h':gaps,'first_timestamp':d.timestamp.min().isoformat(),'last_timestamp':d.timestamp.max().isoformat()})
    pd.DataFrame(cov).to_csv(OUT/'kline_coverage_audit.csv',index=False)
    first=max(d.timestamp.min() for d in raw.values()); end=min(d.timestamp.max() for d in raw.values()); rs=first+pd.Timedelta(days=WARMUP_DAYS)
    panel=[]; signals=[]; trades=[]
    for sym,d0 in raw.items():
        x=features(d0); x['symbol']=sym; x['split']=x.timestamp.map(lambda z: split_for(z,rs,end) if z>=rs else 'WARMUP'); panel.append(x)
        blocked_until=None
        for i in range(len(x)-1):
            ts=x.timestamp.iloc[i]
            if ts<rs or (blocked_until is not None and ts<=blocked_until): continue
            dirs=[]
            if bool(x.long_signal.iloc[i]):dirs.append('LONG')
            if bool(x.short_signal.iloc[i]):dirs.append('SHORT')
            # A valid bar cannot normally be both, but never allow two entries from one symbol candle.
            if len(dirs)>1: dirs=dirs[:1]
            for direction in dirs:
                sp=split_for(ts,rs,end); signals.append({'symbol':sym,'signal_time':ts.isoformat(),'direction':direction,'split':sp})
                t=simulate(x,i,direction,sp,sym)
                if t is not None:
                    trades.append(t); blocked_until=pd.Timestamp(t.exit_time)
                else:
                    # If no exit before data end, no later entry can exist anyway.
                    blocked_until=x.timestamp.iloc[-1]
                    break
    panel=pd.concat(panel,ignore_index=True).sort_values(['timestamp','symbol'])
    cols=['timestamp','symbol','open','high','low','close','volume','atr','compression','prior_high','prior_low','range_expansion','volume_z','close_location','long_signal','short_signal','split']
    panel[cols].to_csv(OUT/'research_panel.csv',index=False)
    pd.DataFrame(signals).to_csv(OUT/'signal_log.csv',index=False)
    tdf=pd.DataFrame([asdict(t) for t in trades]);
    if tdf.empty:tdf=pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    tdf.to_csv(OUT/'rr12_trade_log.csv',index=False)
    rows=[]
    for (sp,di),g in tdf.groupby(['split','direction']):
        wins=int((g.outcome=='WIN').sum()); losses=int((g.outcome=='LOSS').sum()); gw=g.loc[g.gross_R>0,'gross_R'].sum(); gl=-g.loc[g.gross_R<0,'gross_R'].sum()
        rows.append({'split':sp,'direction':di,'trades':len(g),'wins':wins,'losses':losses,'win_rate':wins/len(g),'pf_gross':gw/gl if gl else np.inf,'mean_gross_R':g.gross_R.mean(),'mean_net_R':g.net_R.mean(),'net_R':g.net_R.sum(),'same_candle_sl_tp':0})
    pd.DataFrame(rows).to_csv(OUT/'rr12_event_report.csv',index=False)
    # Diagnostic event study; no timeout strategy is created from it.
    ev=[]
    for sym,d0 in raw.items():
        x=features(d0)
        for i in range(len(x)-1):
            if x.timestamp.iloc[i]<rs:continue
            dirs=(['LONG'] if bool(x.long_signal.iloc[i]) else [])+(['SHORT'] if bool(x.short_signal.iloc[i]) else [])
            for di in dirs:
                entry=float(x.open.iloc[i+1])
                for h in [1,4,8,24,48]:
                    j=i+1+h-1
                    if j>=len(x):continue
                    r=float(x.close.iloc[j])/entry-1
                    ev.append({'split':split_for(x.timestamp.iloc[i],rs,end),'direction':di,'horizon_bars':h,'directional_return':r if di=='LONG' else -r})
    ev=pd.DataFrame(ev)
    if not ev.empty:
        er=ev.groupby(['split','direction','horizon_bars'],as_index=False).agg(n=('directional_return','size'),mean_directional_return=('directional_return','mean'),median_directional_return=('directional_return','median'),win_rate=('directional_return',lambda s:float((s>0).mean())))
    else: er=pd.DataFrame()
    er.to_csv(OUT/'event_study_report.csv',index=False)
    if not tdf.empty:
        ps=tdf.groupby(['split','symbol','direction'],as_index=False).agg(trades=('net_R','size'),wins=('outcome',lambda s:int((s=='WIN').sum())),losses=('outcome',lambda s:int((s=='LOSS').sum())),mean_net_R=('net_R','mean'),net_R=('net_R','sum')); ps['win_rate']=ps.wins/ps.trades
    else: ps=pd.DataFrame()
    ps.to_csv(OUT/'per_symbol_report.csv',index=False)
    meta={'strategy':'V31 Stage-0 Compression -> Expansion Breakout','days':DAYS,'warmup_days':WARMUP_DAYS,'timeframe':TIMEFRAME,'symbols':SYMBOLS,'sl_atr':SL_ATR,'tp_atr':TP_ATR,'compression_ratio':COMPRESSION_RATIO,'breakout_lookback':BREAKOUT_LOOKBACK,'range_expansion_min':RANGE_EXPANSION_MIN,'volume_z_min':VOLUME_Z_MIN,'entry':'next_bar_open','same_candle_sl_tp':'LOSS','timeout':False,'censored_open_trades':'EXCLUDED','overlap':'max_one_open_trade_per_symbol; different_symbols_may_overlap','same_candle_reentry':False,'parameter_optimization':False,'panel_rows':len(panel),'signals':len(signals),'closed_trades':len(tdf)}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'V31 COMPLETE: panel_rows={len(panel)} signals={len(signals)} closed_trades={len(tdf)}')

if __name__=='__main__':main()
                   
