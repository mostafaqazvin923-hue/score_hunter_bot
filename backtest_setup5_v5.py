#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP 5 V5 — DIAGNOSTIC STRUCTURE / ENTRY STUDY

Purpose
-------
This is NOT another filter-optimization version. V1-V4 all clustered near
~32% WR, so V5 tests four pre-declared mechanical entry/structure variants to
identify which part of the Setup-5 translation is responsible for the weak edge.

PDF-derived core (mechanical translation where the PDF is qualitative):
existing trend -> HTF zone reaction -> CHOCH -> OB -> aligned liquidity ->
liquidity pass/sweep -> return to OB -> entry. The PDF also emphasizes a sharp
move into/through CHOCH and meaningful structure between HTF reaction and CHOCH.
Numeric thresholds below are research translations, NOT claimed PDF numbers.

VARIANTS
--------
A_OB_LIMIT:
    After the required liquidity event, first return/touch of OB is the signal;
    entry is at the next candle open.

B_OB_CONFIRM:
    Same structure, but OB touch must close with directional reaction before
    next-candle entry.

C_SWEEP_THEN_OB:
    Liquidity sweep is explicitly required before OB return; sweep must reclaim
    the liquidity level on the same closed candle. This is the closest test of
    the PDF's "pass liquidity then reach OB" sequence.

D_STRONG_CHOCH:
    Same as C, plus a genuinely displacement-like CHOCH: body and close
    extension thresholds are stronger, and there must be at least one confirmed
    execution swing peak/valley between HTF reaction and CHOCH.

All four variants use the same data, symbols, 15m/4H architecture, fixed RR 1:2,
$1,000 capital, $100 margin, 50x leverage, real Binance Futures klines, causal
confirmed pivots, per-symbol overlap only, and chronological Train/Validation/OOS.
No OOS tuning. No timeout, trailing, BE, partial exits, or stop-after-loss.
"""
from __future__ import annotations
import io, os, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd
import requests

SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","SUIUSDT","AVAXUSDT","NEARUSDT","ADAUSDT","BNBUSDT","APTUSDT","CRVUSDT","ONDOUSDT","PENDLEUSDT","ICPUSDT","WIFUSDT"]
EXEC_TF="15m"; HTF_TF="4h"; TEST_DAYS=365; WARMUP_DAYS=90
INITIAL_CAPITAL=1000.0; MARGIN=100.0; LEVERAGE=50.0; NOTIONAL=MARGIN*LEVERAGE
RR=2.0; FEE_RATE=0.0007; SLIPPAGE=0.0003
PIVOT=2; ALIGN_TOL=0.004; SL_BUFFER=0.0005
MIN_RISK=0.0005; MAX_RISK=0.08
# Frozen research translations; not PDF literal values.
MAX_REACTION_WAIT=96; MAX_CHOCH_WAIT=160; MAX_LIQ_TO_OB_BARS=24; MAX_POST_SWEEP_WAIT=96
MIN_CHOCH_BODY_BASE=0.10; MIN_CHOCH_EXT_BASE=0.02
MIN_CHOCH_BODY_STRONG=0.60; MIN_CHOCH_EXT_STRONG=0.20
MIN_SWEEP_PEN_ATR=0.00; MIN_OB_REACTION_BODY_ATR=0.05
MAX_WORKERS=min(8,len(SYMBOLS)); REQUEST_TIMEOUT=30
ARCHIVE_BASE="https://data.binance.vision/data/futures/um"
BACKTEST_END=os.getenv("BACKTEST_END","").strip()
TRAIN_FRAC=.50; VALID_FRAC=.20; OOS_FRAC=.30
VARIANTS=["A_OB_LIMIT","B_OB_CONFIRM","C_SWEEP_THEN_OB","D_STRONG_CHOCH"]
SESSION=requests.Session(); SESSION.headers.update({"User-Agent":"setup5-v5-diagnostic/1.0"})

@dataclass
class Candidate:
    variant:str; symbol:str; direction:str; signal_idx:int; entry_idx:int
    htf_zone_idx:int; reaction_idx:int; choch_idx:int; ob_idx:int
    liq_idx1:int; liq_idx2:int; sweep_idx:int
    structural_target_idx:int; structural_target:float
    entry:float; sl:float; tp:float; risk:float
    choch_body_atr:float; choch_ext_atr:float; sweep_pen_atr:float
    structural_room_R:float; ifc_present:int; reaction_body_atr:float
    reaction_close_location:float

@dataclass
class Trade:
    variant:str; symbol:str; direction:str; signal_idx:int; entry_idx:int; exit_idx:int
    entry_time:str; exit_time:str; entry:float; sl:float; tp:float; exit:float
    pnl:float; result:str; R:float
    htf_zone_idx:int; reaction_idx:int; choch_idx:int; ob_idx:int; liq_idx1:int; liq_idx2:int; sweep_idx:int
    structural_target_idx:int; structural_target:float; choch_body_atr:float; choch_ext_atr:float
    sweep_pen_atr:float; structural_room_R:float; ifc_present:int; reaction_body_atr:float; reaction_close_location:float

def utc_now(): return pd.Timestamp.now(tz="UTC")
def get_end():
    if BACKTEST_END: return pd.Timestamp(BACKTEST_END,tz="UTC")+pd.Timedelta(hours=23,minutes=45)
    return utc_now().floor("D")-pd.Timedelta(minutes=15)
def months(start,end):
    x=pd.Timestamp(start.year,start.month,1,tz="UTC"); last=pd.Timestamp(end.year,end.month,1,tz="UTC")
    while x<=last: yield x; x+=pd.offsets.MonthBegin(1)
def parse_zip(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        n=next(x for x in z.namelist() if x.lower().endswith('.csv'))
        raw=pd.read_csv(z.open(n),header=None).iloc[:,:12]
    raw.columns=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_volume","taker_buy_quote","ignore"]
    raw["open_time"]=pd.to_numeric(raw.open_time,errors="coerce"); raw["close_time"]=pd.to_numeric(raw.close_time,errors="coerce")
    for c in ["open","high","low","close","volume"]: raw[c]=pd.to_numeric(raw[c],errors="coerce")
    raw=raw.dropna(subset=["open_time","open","high","low","close"])
    raw["open_time"]=pd.to_datetime(raw.open_time,unit="ms",utc=True); raw["close_time"]=pd.to_datetime(raw.close_time,unit="ms",utc=True)
    return raw.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
def fetch(symbol,interval,start,end):
    parts=[]
    for m in months(start,end):
        u=f"{ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{m:%Y-%m}.zip"
        try:
            r=SESSION.get(u,timeout=REQUEST_TIMEOUT)
            if r.status_code==200: parts.append(parse_zip(r.content))
        except Exception: pass
    # Daily fallback for the current/latest month or if monthly files were unavailable.
    d=pd.Timestamp(start.date(),tz="UTC")
    if not parts or (end-start).days<=45:
        while d<=end:
            u=f"{ARCHIVE_BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d:%Y-%m-%d}.zip"
            try:
                r=SESSION.get(u,timeout=REQUEST_TIMEOUT)
                if r.status_code==200: parts.append(parse_zip(r.content))
            except Exception: pass
            d+=pd.Timedelta(days=1)
    if not parts: raise RuntimeError(f"No Binance data: {symbol} {interval}")
    df=pd.concat(parts,ignore_index=True).sort_values("open_time").drop_duplicates("open_time")
    df=df[(df.open_time>=start)&(df.open_time<=end)&(df.close_time<=utc_now())].reset_index(drop=True)
    return df
def validate(df,mins,symbol,name):
    # 4H history contains fewer rows by design: ~6 candles/day.
    # 365d test + 90d warmup + HTF context does not guarantee 3000 rows.
    # Keep a meaningful integrity floor without rejecting valid history.
    min_rows = 30000 if name == "15m" else 1800
    if len(df) < min_rows: raise RuntimeError(f"{symbol} {name}: too few rows {len(df)} (minimum {min_rows})")
    if not df.open_time.is_monotonic_increasing or df.open_time.duplicated().any(): raise RuntimeError(f"{symbol} {name}: timestamps invalid")
    gaps=df.open_time.diff().dropna(); bad=gaps[gaps!=pd.Timedelta(minutes=mins)]
    if not bad.empty: raise RuntimeError(f"{symbol} {name}: gap {bad.max()}")
    if (df.high<df.low).any() or (df.open>df.high).any() or (df.open<df.low).any() or (df.close>df.high).any() or (df.close<df.low).any(): raise RuntimeError(f"{symbol} {name}: invalid OHLC")
def atr(df,n=14):
    p=df.close.shift(); return pd.concat([df.high-df.low,(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1).rolling(n,min_periods=n).mean()
def pivots(df,p=PIVOT):
    h=df.high.to_numpy(); l=df.low.to_numpy(); ph=np.zeros(len(df),bool); pl=np.zeros(len(df),bool)
    for i in range(p,len(df)-p):
        ph[i]=h[i]>max(h[i-p:i]) and h[i]>=max(h[i+1:i+p+1])
        pl[i]=l[i]<min(l[i-p:i]) and l[i]<=min(l[i+1:i+p+1])
    return ph,pl
def last_before(flag,idx):
    q=np.flatnonzero(flag[:max(0,idx-PIVOT+1)])
    return int(q[-1]) if len(q) else None
def htf_map(htf,times): return np.searchsorted(htf.close_time.astype('int64').to_numpy(),times.astype('int64').to_numpy(),side='right')-1
def trend(ex,ph,pl,i):
    hs=np.flatnonzero(ph[:i+1]); ls=np.flatnonzero(pl[:i+1]); hs=hs[hs+PIVOT<=i]; ls=ls[ls+PIVOT<=i]
    if len(hs)<2 or len(ls)<2:return False,False
    h1,h2=hs[-1],hs[-2]; l1,l2=ls[-1],ls[-2]
    return bool(ex.high.iloc[h1]<ex.high.iloc[h2] and ex.low.iloc[l1]<ex.low.iloc[l2]), bool(ex.high.iloc[h1]>ex.high.iloc[h2] and ex.low.iloc[l1]>ex.low.iloc[l2])
def htf_context(htf):
    ph,pl=pivots(htf); out=htf.copy(); out['ph']=ph; out['pl']=pl; return out

def find_choch(ex,ph,pl,reaction,zone_low,zone_high,direction,strong=False):
    body_thr=MIN_CHOCH_BODY_STRONG if strong else MIN_CHOCH_BODY_BASE; ext_thr=MIN_CHOCH_EXT_STRONG if strong else MIN_CHOCH_EXT_BASE
    for c in range(reaction+1,min(len(ex)-2,reaction+MAX_CHOCH_WAIT)):
        if direction=='LONG':
            q=np.flatnonzero(ph[:c+1]); q=q[q+PIVOT<=c]
            if not len(q): continue
            level=float(ex.high.iloc[q[-1]])
            if ex.low.iloc[c] < zone_low*(1-.0035): return None
            body=abs(float(ex.close.iloc[c])-float(ex.open.iloc[c]))/max(float(ex.atr.iloc[c]),1e-12)
            ext=(float(ex.close.iloc[c])-level)/max(float(ex.atr.iloc[c]),1e-12)
            if ex.close.iloc[c]>level and body>=body_thr and ext>=ext_thr:
                if strong:
                    mids=np.flatnonzero(pl[reaction+1:c]); mids=mids[mids+reaction+1+PIVOT<=c]
                    # At least one confirmed valley between reaction and CHOCH.
                    if not len(mids): continue
                return c,body,ext
        else:
            q=np.flatnonzero(pl[:c+1]); q=q[q+PIVOT<=c]
            if not len(q): continue
            level=float(ex.low.iloc[q[-1]])
            if ex.high.iloc[c] > zone_high*(1+.0035): return None
            body=abs(float(ex.close.iloc[c])-float(ex.open.iloc[c]))/max(float(ex.atr.iloc[c]),1e-12)
            ext=(level-float(ex.close.iloc[c]))/max(float(ex.atr.iloc[c]),1e-12)
            if ex.close.iloc[c]<level and body>=body_thr and ext>=ext_thr:
                if strong:
                    mids=np.flatnonzero(ph[reaction+1:c]); mids=mids[mids+reaction+1+PIVOT<=c]
                    if not len(mids): continue
                return c,body,ext
    return None

def ob_info(ex,pl,ph,reaction,choch,direction):
    if direction=='LONG':
        q=np.flatnonzero(pl[reaction:choch+1]); q=[reaction+int(x) for x in q if reaction+int(x)+PIVOT<=choch]
        if not q:return None
        ob=min(q,key=lambda x:ex.low.iloc[x]); lo=float(ex.low.iloc[ob]); hi=float(max(ex.open.iloc[ob],ex.close.iloc[ob])); sl=lo*(1-SL_BUFFER)
    else:
        q=np.flatnonzero(ph[reaction:choch+1]); q=[reaction+int(x) for x in q if reaction+int(x)+PIVOT<=choch]
        if not q:return None
        ob=max(q,key=lambda x:ex.high.iloc[x]); lo=float(min(ex.open.iloc[ob],ex.close.iloc[ob])); hi=float(ex.high.iloc[ob]); sl=hi*(1+SL_BUFFER)
    return ob,lo,hi,sl

def liquidity(ex,ph,pl,choch,ob,direction):
    flags=pl if direction=='LONG' else ph; q=np.flatnonzero(flags[choch+1:]); q=[choch+1+int(x) for x in q if choch+1+int(x)+PIVOT<len(ex)]
    best=None
    for a,b in zip(q,q[1:]):
        if b-ob>MAX_LIQ_TO_OB_BARS: break
        p1=float(ex.low.iloc[a] if direction=='LONG' else ex.high.iloc[a]); p2=float(ex.low.iloc[b] if direction=='LONG' else ex.high.iloc[b]); level=(p1+p2)/2
        if abs(p1-p2)/max(abs(level),1e-12)<=ALIGN_TOL: best=(a,b,level)
    return best

def sweep(ex,liq2,level,direction):
    for s in range(liq2+PIVOT,min(len(ex)-1,liq2+MAX_POST_SWEEP_WAIT)):
        a=max(float(ex.atr.iloc[s]),1e-12)
        if direction=='LONG': pen=(level-float(ex.low.iloc[s]))/a; ok=ex.low.iloc[s]<level and ex.close.iloc[s]>=level
        else: pen=(float(ex.high.iloc[s])-level)/a; ok=ex.high.iloc[s]>level and ex.close.iloc[s]<=level
        if ok and pen>=MIN_SWEEP_PEN_ATR:return s,pen
    return None

def structural(ex,ph,pl,choch,entry,direction):
    flags=ph if direction=='LONG' else pl; q=np.flatnonzero(flags[choch+1:entry]); q=[choch+1+int(x) for x in q if choch+1+int(x)+PIVOT<entry]
    if not q:return None
    idx=(max(q,key=lambda x:ex.high.iloc[x]) if direction=='LONG' else min(q,key=lambda x:ex.low.iloc[x]))
    return idx,float(ex.high.iloc[idx] if direction=='LONG' else ex.low.iloc[idx])

def ifc_after_ob(ex,ob,entry,direction):
    # Optional strengthening factor only: 3-candle fair-value-gap-like condition.
    # Logged, never required, and uses only candles after OB and before entry.
    for i in range(ob+1,max(ob+1,entry-1)):
        if direction=='LONG' and ex.low.iloc[i+1]>ex.high.iloc[i-1]: return 1
        if direction=='SHORT' and ex.high.iloc[i+1]<ex.low.iloc[i-1]: return 1
    return 0

def make_variant(symbol,ex,htf,variant):
    ex=ex.copy(); ex['atr']=atr(ex); ph,pl=pivots(ex); h=htf_context(htf); hm=htf_map(h,ex.open_time); out=[]
    strong=variant=='D_STRONG_CHOCH'
    for i in range(40,len(ex)-2):
        if not np.isfinite(ex.atr.iloc[i]) or ex.atr.iloc[i]<=0:continue
        down,up=trend(ex,ph,pl,i); hidx=int(hm[i])
        if hidx<0:continue
        for direction,active in [('LONG',down),('SHORT',up)]:
            if not active:continue
            flags=h.pl.to_numpy(bool) if direction=='LONG' else h.ph.to_numpy(bool)
            zq=np.flatnonzero(flags[:hidx+1]); zq=zq[zq+PIVOT<=hidx]
            if not len(zq):continue
            z=int(zq[-1])
            if direction=='LONG': zl=float(h.low.iloc[z]); zh=float(max(h.open.iloc[z],h.close.iloc[z]))
            else: zl=float(min(h.open.iloc[z],h.close.iloc[z])); zh=float(h.high.iloc[z])
            reaction=None
            for r in range(i,min(len(ex),i+MAX_REACTION_WAIT)):
                if ex.low.iloc[r]<=zh and ex.high.iloc[r]>=zl: reaction=r;break
            if reaction is None or reaction<=i:continue
            ch=find_choch(ex,ph,pl,reaction,zl,zh,direction,strong)
            if ch is None:continue
            choch,body,ext=ch
            obx=ob_info(ex,pl,ph,reaction,choch,direction)
            if not obx:continue
            ob,obl,obh,sl=obx
            liq=liquidity(ex,ph,pl,choch,ob,direction)
            if not liq:continue
            l1,l2,level=liq
            sw=sweep(ex,l2,level,direction)
            # Variant A/B allow the diagnostic to isolate whether explicit sweep
            # sequencing is the issue; C/D require it.
            if variant in ('C_SWEEP_THEN_OB','D_STRONG_CHOCH') and sw is None:continue
            sweep_idx,pen=(sw if sw else (l2,0.0))
            start=sweep_idx+1 if sw else l2+1
            sig=None
            for r in range(start,min(len(ex)-1,start+MAX_POST_SWEEP_WAIT)):
                touched=ex.low.iloc[r]<=obh and ex.high.iloc[r]>=obl
                if not touched:continue
                if variant=='B_OB_CONFIRM' or variant in ('C_SWEEP_THEN_OB','D_STRONG_CHOCH'):
                    rng=max(float(ex.high.iloc[r]-ex.low.iloc[r]),1e-12); cl=(float(ex.close.iloc[r])-float(ex.low.iloc[r]))/rng
                    rb=abs(float(ex.close.iloc[r])-float(ex.open.iloc[r]))/max(float(ex.atr.iloc[r]),1e-12)
                    ok=(float(ex.close.iloc[r])>float(ex.open.iloc[r]) and cl>=.55 and rb>=MIN_OB_REACTION_BODY_ATR) if direction=='LONG' else (float(ex.close.iloc[r])<float(ex.open.iloc[r]) and cl<=.45 and rb>=MIN_OB_REACTION_BODY_ATR)
                    if not ok:continue
                sig=r;break
            if sig is None:continue
            entry_idx=sig+1; entry=float(ex.open.iloc[entry_idx])*(1+SLIPPAGE if direction=='LONG' else 1-SLIPPAGE)
            risk=(entry-sl) if direction=='LONG' else (sl-entry)
            if risk<=0 or risk/entry<MIN_RISK or risk/entry>MAX_RISK:continue
            st=structural(ex,ph,pl,choch,entry_idx,direction)
            if not st:continue
            tid,target=st; room=(target-entry)/risk if direction=='LONG' else (entry-target)/risk
            if room<RR:continue
            tp=entry+RR*risk if direction=='LONG' else entry-RR*risk
            rng=max(float(ex.high.iloc[sig]-ex.low.iloc[sig]),1e-12); cl=(float(ex.close.iloc[sig])-float(ex.low.iloc[sig]))/rng; rb=abs(float(ex.close.iloc[sig])-float(ex.open.iloc[sig]))/max(float(ex.atr.iloc[sig]),1e-12)
            c=Candidate(variant,symbol,direction,sig,entry_idx,z,reaction,choch,ob,l1,l2,sweep_idx,tid,target,entry,sl,tp,risk,body,ext,pen,room,ifc_after_ob(ex,ob,entry_idx,direction),rb,cl)
            audit_candidate(c);out.append(c)
    uniq={(c.symbol,c.direction,c.entry_idx):c for c in out}; return sorted(uniq.values(),key=lambda c:c.entry_idx)

def audit_candidate(c):
    assert c.htf_zone_idx<c.choch_idx<c.entry_idx
    assert c.ob_idx<c.choch_idx
    assert c.liq_idx1<c.sweep_idx and c.liq_idx2<c.sweep_idx if c.sweep_idx>c.liq_idx2 else True
    assert c.sweep_idx<c.entry_idx
    assert c.structural_target_idx<c.entry_idx
    assert c.entry_idx==c.signal_idx+1
    assert (c.tp>c.entry>c.sl) if c.direction=='LONG' else (c.sl>c.entry>c.tp)

def simulate(ex,cands,symbol,variant):
    trades=[]; unresolved=0; occupied=-1
    for c in cands:
        if c.entry_idx<=occupied:continue
        exit_i=None; xp=None; res=None
        for j in range(c.entry_idx+1,len(ex)):
            hi=float(ex.high.iloc[j]); lo=float(ex.low.iloc[j])
            if c.direction=='LONG':
                if lo<=c.sl:exit_i,xp,res=j,c.sl,'LOSS';break
                if hi>=c.tp:exit_i,xp,res=j,c.tp,'WIN';break
            else:
                if hi>=c.sl:exit_i,xp,res=j,c.sl,'LOSS';break
                if lo<=c.tp:exit_i,xp,res=j,c.tp,'WIN';break
        if exit_i is None: unresolved+=1; occupied=len(ex)-1; continue
        gross=(xp-c.entry) if c.direction=='LONG' else (c.entry-xp); pnl_price=gross/c.entry*NOTIONAL; pnl=pnl_price-2*FEE_RATE*NOTIONAL
        rreal=pnl_price/(c.risk/c.entry*NOTIONAL)
        trades.append(Trade(variant,symbol,c.direction,c.signal_idx,c.entry_idx,exit_i,str(ex.open_time.iloc[c.entry_idx]),str(ex.open_time.iloc[exit_i]),c.entry,c.sl,c.tp,float(xp),float(pnl),res,float(rreal),c.htf_zone_idx,c.reaction_idx,c.choch_idx,c.ob_idx,c.liq_idx1,c.liq_idx2,c.sweep_idx,c.structural_target_idx,c.structural_target,c.choch_body_atr,c.choch_ext_atr,c.sweep_pen_atr,c.structural_room_R,c.ifc_present,c.reaction_body_atr,c.reaction_close_location))
        occupied=exit_i
    return trades,unresolved

def audit_trades(ts):
    by={}
    for t in ts:by.setdefault(t.symbol,[]).append(t)
    for sym,x in by.items():
        x.sort(key=lambda t:t.entry_idx)
        for a,b in zip(x,x[1:]):assert b.entry_idx>a.exit_idx,f"overlap/re-entry {sym}"
def metrics(ts):
    if not ts:return dict(trades=0,wins=0,losses=0,wr=0,pf=0,net=0,max_streak=0,avg_win=0,avg_loss=0,expectancy=0)
    x=sorted(ts,key=lambda t:(t.exit_time,t.symbol)); wins=[t.pnl for t in x if t.pnl>0]; losses=[t.pnl for t in x if t.pnl<=0]; gw=sum(wins); gl=abs(sum(losses)); cur=best=0
    for t in x:
        cur=cur+1 if t.result=='LOSS' else 0; best=max(best,cur)
    return dict(trades=len(x),wins=len(wins),losses=len(losses),wr=100*len(wins)/len(x),pf=gw/gl if gl else float('inf'),net=sum(t.pnl for t in x),max_streak=best,avg_win=float(np.mean(wins)) if wins else 0,avg_loss=float(np.mean(losses)) if losses else 0,expectancy=float(np.mean([t.pnl for t in x])))
def split(ts,start,end):
    te=start+(end-start)*TRAIN_FRAC; ve=te+(end-start)*VALID_FRAC
    f=lambda t:pd.Timestamp(t.entry_time,tz='UTC')
    return [t for t in ts if f(t)<te],[t for t in ts if te<=f(t)<ve],[t for t in ts if f(t)>=ve],te,ve
def process(symbol,start,end):
    ex=fetch(symbol,EXEC_TF,start-pd.Timedelta(days=WARMUP_DAYS),end); h=fetch(symbol,HTF_TF,start-pd.Timedelta(days=WARMUP_DAYS+10),end)
    validate(ex,15,symbol,'15m'); validate(h,240,symbol,'4h'); result={}
    for v in VARIANTS:
        c=make_variant(symbol,ex,h,v); t,u=simulate(ex,c,symbol,v); audit_trades(t); result[v]=(c,t,u)
    return symbol,result

def print_m(name,ts):
    m=metrics(ts);print(f"{name}: trades={m['trades']} W/L={m['wins']}/{m['losses']} WR={m['wr']:.2f}% PF={m['pf']:.3f} Net=${m['net']:.2f} MaxStreak={m['max_streak']}")
def main():
    end=get_end(); start=end-pd.Timedelta(days=TEST_DAYS)+pd.Timedelta(minutes=15)
    print('SETUP 5 V5 — DIAGNOSTIC STRUCTURE / ENTRY STUDY'); print(f'UTC: {start} -> {end}'); print(f'Symbols: {len(SYMBOLS)} TF: {EXEC_TF}/{HTF_TF} RR=1:{RR:.0f}'); print('Variants:',', '.join(VARIANTS))
    allc={v:[] for v in VARIANTS}; allt={v:[] for v in VARIANTS}; unresolved={v:0 for v in VARIANTS}; fails=[]; t0=time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fs={pool.submit(process,s,start,end):s for s in SYMBOLS}
        for f in as_completed(fs):
            s=fs[f]
            try:
                _,res=f.result()
                print('\n',s)
                for v in VARIANTS:
                    c,t,u=res[v]; allc[v]+=c; allt[v]+=t; unresolved[v]+=u; print(f'  {v:20s} candidates={len(c):4d} trades={len(t):4d} unresolved={u}')
            except Exception as e:fails.append((s,repr(e))); print(s,'FAILED',e)
    if fails:raise RuntimeError('; '.join(f'{s}: {e}' for s,e in fails))
    summary=[]; feature_rows=[]
    for v in VARIANTS:
        ts=sorted(allt[v],key=lambda x:(x.exit_time,x.symbol)); audit_trades(ts); train,valid,oos,te,ve=split(ts,start,end)
        print('\n'+'='*78); print(v); print_m('FULL',ts); print_m('TRAIN',train); print_m('VALID',valid); print_m('OOS LOCKED',oos)
        for d in ['LONG','SHORT']:
            x=[t for t in oos if t.direction==d]; print_m(f'OOS {d}',x)
        for ifc in [0,1]: print_m(f'OOS IFC={ifc}',[t for t in oos if t.ifc_present==ifc])
        for d in ['LONG','SHORT']:
            x=[t for t in oos if t.direction==d]
            if x: feature_rows.append({'variant':v,'slice':'OOS','direction':d,**metrics(x)})
        m=metrics(oos); summary.append({'variant':v,'full_trades':len(ts),'oos_trades':m['trades'],'oos_wr':m['wr'],'oos_pf':m['pf'],'oos_net':m['net'],'oos_max_streak':m['max_streak'],'unresolved':unresolved[v]})
        pd.DataFrame([asdict(t) for t in ts]).to_csv(f'setup5_v5_{v.lower()}_trades.csv',index=False)
        pd.DataFrame([asdict(c) for c in allc[v]]).to_csv(f'setup5_v5_{v.lower()}_candidates.csv',index=False)
    pd.DataFrame(summary).to_csv('setup5_v5_summary.csv',index=False); pd.DataFrame(feature_rows).to_csv('setup5_v5_oos_direction.csv',index=False)
    print('\nDIAGNOSTIC SUMMARY'); print(pd.DataFrame(summary).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nTARGET CHECK IS DESCRIPTIVE ONLY — NO OOS TUNING'); print('The winner is NOT selected by OOS metrics in this script.')
    print(f'Runtime: {time.time()-t0:.1f}s')
if __name__=='__main__':main()
