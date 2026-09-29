#!/usr/bin/env python3
"""HUNTER-V23-STAGE0: FUTURES BASIS DISLOCATION / REPRICING.

Locked hypothesis (no Stage-0 tuning): cross-sectional perpetual-vs-spot
basis contains information about short-horizon futures returns. We test the
high-basis/low-basis factor with causal basis z-score, 24h basis repricing,
and historical XT funding as context. RR=1:2; stop values are sensitivity
outputs only.

Integrity: independent XT spot + futures 1H OHLCV, historical XT funding,
completed candles only, next-bar entry, per-symbol overlap lock, simultaneous
different symbols allowed, same exit-candle re-entry blocked, same-candle
SL+TP=loss, no timeout/BE/trailing/pyramiding, fixed universe, no future
symbol selection. OI is intentionally omitted because a historical XT OI
series was not verified from the public API.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SYMBOLS=["btc_usdt","eth_usdt","sol_usdt","sui_usdt","avax_usdt","near_usdt","ada_usdt","bnb_usdt","apt_usdt","crv_usdt","ondo_usdt","pendle_usdt","icp_usdt","wif_usdt"]
FUT_URL="https://fapi.xt.com/future/market/v1/public/q/kline"
# XT spot kline endpoint used by the public API. If XT changes its spot path,
# fail loudly rather than substituting another data source.
SPOT_URL="https://sapi.xt.com/v4/public/kline"
FUND_URL="https://fapi.xt.com/future/market/v1/public/q/funding-rate-record"
LOOKBACK_DAYS=455; WARMUP_DAYS=90; H=3600000; LIMIT=1000
INITIAL=1000.; MARGIN=100.; NOTIONAL=5000.; RR=2.; FEE=.0007; SLIP=.0003; MAX_POS=10
ATR_N=14
BASIS_Z_N=120; BASIS_CHANGE_N=24; QUANT=.20; MIN_Z=1.0; MIN_CS=8; FUND_N=3
STOPS=(1.0,1.25,1.5)
CACHE=Path("data/xt_v23_stage0"); REPORT=Path("reports/xt_v23_stage0")


def rows(obj):
    if isinstance(obj,list): return obj
    if not isinstance(obj,dict): return None
    for k in ("result","data","rows","list"):
        v=obj.get(k)
        if isinstance(v,list): return v
    r=obj.get("result")
    if isinstance(r,dict):
        for k in ("items","list","rows","data"):
            v=r.get(k)
            if isinstance(v,list): return v
    return None


def krow(r):
    try:
        if isinstance(r,dict):
            a=[r.get("t",r.get("timestamp")),r.get("o",r.get("open")),r.get("h",r.get("high")),r.get("l",r.get("low")),r.get("c",r.get("close")),r.get("a",r.get("v",r.get("volume",r.get("amount"))))]
        elif isinstance(r,(list,tuple)) and len(r)>=6: a=r[:6]
        else: return None
        if any(v is None for v in a): return None
        return int(a[0]),*[float(v) for v in a[1:]]
    except Exception: return None


def validate(x,name,start,end,allow_spot_gaps=False):
    x=x.copy(); x.timestamp=pd.to_datetime(x.timestamp,utc=True)
    x=x.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
    x=x[(x.timestamp>=pd.to_datetime(start,unit="ms",utc=True))&(x.timestamp<=pd.to_datetime(end,unit="ms",utc=True))].copy()
    for c in ["open","high","low","close","volume"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    if x[["open","high","low","close","volume"]].isna().any().any(): raise RuntimeError(f"{name}: NaN")
    if (x[["open","high","low","close"]]<=0).any().any() or (x.volume<0).any(): raise RuntimeError(f"{name}: invalid OHLCV")
    gaps=x.timestamp.diff().dropna().dt.total_seconds()/3600
    bad=gaps>1+1e-9
    if bad.any():
        if not allow_spot_gaps:
            raise RuntimeError(f"{name}: 1H gap; no fabrication")
        # Spot is auxiliary context. We never fill or interpolate missing
        # candles. Instead, permit small historical holes and make every
        # feature that needs a 24h history explicitly require real 24h
        # spacing. Large holes remain fatal.
        max_gap=float(gaps.max())
        gap_count=int(bad.sum())
        if max_gap>6.0:
            raise RuntimeError(f"{name}: spot gap too large ({max_gap:.2f}h); no fabrication")
        print(f"[DATA] {name}: {gap_count} spot gap(s), max_gap={max_gap:.2f}h; gaps will NOT be filled")
    if len(x)<int((LOOKBACK_DAYS+WARMUP_DAYS)*24*.95): raise RuntimeError(f"{name}: insufficient history {len(x)}")
    return x.reset_index(drop=True)


def fetch_kline(url,symbol,prefix):
    CACHE.mkdir(parents=True,exist_ok=True); path=CACHE/f"{prefix}{symbol}.csv"
    now=int(time.time()*1000); start=((now-int((LOOKBACK_DAYS+WARMUP_DAYS)*86400e3)+H-1)//H)*H; end=(now//H)*H-H
    if path.exists(): return validate(pd.read_csv(path),prefix+symbol,start,end,allow_spot_gaps=(prefix=="spot_"))
    s=requests.Session(); cur=start; out=[]; page=0
    while cur<=end:
        page+=1
        if page>1000: raise RuntimeError(f"{symbol}: pagination guard")
        we=min(end,cur+LIMIT*H-1); p={"symbol":symbol,"interval":"1h","startTime":cur,"endTime":we,"limit":LIMIT}
        payload=None; err=None
        for a in range(4):
            try:
                q=s.get(url,params=p,timeout=30); q.raise_for_status(); payload=rows(q.json())
                if payload is None: raise RuntimeError("no kline list")
                break
            except Exception as e: err=e; time.sleep(1+a)
        if payload is None: raise RuntimeError(f"{symbol}: page {page}: {err}")
        inside=[v for v in (krow(r) for r in payload) if v and cur<=v[0]<=we]
        if not inside:
            if we>=end: break
            recovered=None
            for div in (2,4,8,16):
                lim=max(32,LIMIT//div); se=min(end,cur+lim*H-1); pp=dict(p,endTime=se,limit=lim)
                for a in range(3):
                    try:
                        q=s.get(url,params=pp,timeout=30); q.raise_for_status(); rr=rows(q.json()) or []
                        cand=[v for v in (krow(r) for r in rr) if v and cur<=v[0]<=se]
                        if cand: recovered=cand; break
                    except Exception: pass
                    time.sleep(1+a)
                if recovered: break
            if not recovered: raise RuntimeError(f"{symbol}: interior empty page; refusing skip/fabricate")
            inside=recovered
        out.extend(inside); mx=max(v[0] for v in inside); nxt=mx+H
        if nxt<=cur: raise RuntimeError(f"{symbol}: pagination stalled")
        cur=nxt
        if page%4==0: print(f"[FETCH] {prefix}{symbol} page={page} rows={len(out)}")
        if mx>=end: break
        time.sleep(.05)
    x=pd.DataFrame(out,columns=["ts","open","high","low","close","volume"]); x["timestamp"]=pd.to_datetime(x.pop("ts"),unit="ms",utc=True)
    x=x[x.timestamp<pd.Timestamp.now(tz="UTC").floor("1h")]
    x=validate(x,prefix+symbol,start,end,allow_spot_gaps=(prefix=="spot_")); x.to_csv(path,index=False); return x


def frow(r):
    try:
        if not isinstance(r,dict): return None
        t=r.get("createdTime",r.get("timestamp",r.get("t"))); rate=r.get("fundingRate",r.get("rate",r.get("r"))); i=r.get("id")
        if t is None or rate is None:return None
        return int(t),float(rate),None if i is None else int(i)
    except Exception:return None


def fetch_funding(symbol):
    CACHE.mkdir(parents=True,exist_ok=True); path=CACHE/f"funding_{symbol}.csv"; now=int(time.time()*1000); start=now-int((LOOKBACK_DAYS+WARMUP_DAYS)*86400e3); end=now
    if path.exists():
        f=pd.read_csv(path)
        f.timestamp=pd.to_datetime(f.timestamp,utc=True)
        f=f[(f.timestamp>=pd.to_datetime(start,unit="ms",utc=True))&(f.timestamp<=pd.to_datetime(end,unit="ms",utc=True))].drop_duplicates("timestamp").sort_values("timestamp")
        if len(f)>=int((LOOKBACK_DAYS+WARMUP_DAYS)*.9) and f.funding_rate.notna().all(): return f
    s=requests.Session(); next_id=None; seen=set(); out=[]
    for page in range(1,101):
        p={"symbol":symbol,"direction":"PREV","limit":100}
        if next_id is not None:p["id"]=next_id
        payload=None
        for a in range(4):
            try:q=s.get(FUND_URL,params=p,timeout=30); q.raise_for_status(); payload=q.json(); break
            except Exception:time.sleep(1+a)
        if payload is None:raise RuntimeError(f"{symbol}: funding request failed")
        rr=rows(payload) or []; items=[v for v in (frow(r) for r in rr) if v]
        if not items:break
        for v in items:
            if v[2] is None or v[2] not in seen:out.append(v); v[2] is not None and seen.add(v[2])
        oldest=min(v[0] for v in items)
        if oldest<=start:break
        ids=[v[2] for v in items if v[2] is not None]
        if not ids:raise RuntimeError(f"{symbol}: funding page has no ids for pagination")
        new=min(ids)-1
        if next_id is not None and new>=next_id:raise RuntimeError(f"{symbol}: funding pagination stalled")
        next_id=new; time.sleep(.05)
    if not out:raise RuntimeError(f"{symbol}: zero funding records")
    f=pd.DataFrame(out,columns=["timestamp_ms","funding_rate","id"]); f.timestamp=pd.to_datetime(f.timestamp_ms,unit="ms",utc=True); f=f.drop_duplicates("timestamp").sort_values("timestamp")
    f=f[(f.timestamp>=pd.to_datetime(start,unit="ms",utc=True))&(f.timestamp<=pd.to_datetime(end,unit="ms",utc=True))]
    if len(f)<int((LOOKBACK_DAYS+WARMUP_DAYS)*.9):raise RuntimeError(f"{symbol}: funding history incomplete: {len(f)}")
    f.to_csv(path,index=False); return f


def tr(x):
    p=x.close.shift(1); return pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1)


def build(fut,spot,fund):
    f=fut.set_index("timestamp").sort_index(); s=spot.set_index("timestamp").sort_index()
    x=f.join(s[["close","open","high","low","volume"]].add_suffix("_spot"),how="inner")
    x["basis"]=(x.close-x.close_spot)/x.close_spot
    x["atr"]=tr(x).rolling(ATR_N,min_periods=ATR_N).mean()
    # Never assume row-count == elapsed hours. A missing spot candle must
    # invalidate a 24h feature rather than silently stretching it across a gap.
    dt=x.index.to_series().diff()
    valid24=dt.eq(pd.Timedelta(hours=1))
    x["spot_ret24"]=np.nan
    x["basis_chg24"]=np.nan
    if len(x)>24:
        t24=x.index.to_series().diff(24)
        exact24=t24.eq(pd.Timedelta(hours=24))
        # Use timestamp-aligned lookup, not positional arithmetic.
        prior_idx=x.index-pd.Timedelta(hours=24)
        spot_now=x["close_spot"]
        spot_prev=pd.Series(spot_now.to_numpy(),index=x.index).reindex(prior_idx.to_numpy())
        spot_prev.index=x.index
        basis_prev=pd.Series(x["basis"].to_numpy(),index=x.index).reindex(prior_idx.to_numpy())
        basis_prev.index=x.index
        x.loc[exact24,"spot_ret24"]=(spot_now/spot_prev-1).loc[exact24]
        x.loc[exact24,"basis_chg24"]=(x["basis"]-basis_prev).loc[exact24]
    ff=fund.set_index("timestamp")["funding_rate"].sort_index(); x=pd.merge_asof(x.reset_index().sort_values("timestamp"),ff.rename("funding_rate").reset_index(),on="timestamp",direction="backward",allow_exact_matches=True).set_index("timestamp")
    x["funding_mean"]=x.funding_rate.rolling(FUND_N,min_periods=FUND_N).mean(); return x.replace([np.inf,-np.inf],np.nan)


def panel(streams):
    b=pd.concat({s:x.basis for s,x in streams.items()},axis=1,join="inner"); ch=pd.concat({s:x.basis_chg24 for s,x in streams.items()},axis=1,join="inner")
    med=b.median(axis=1,skipna=True); d=b.sub(med,axis=0); mu=d.rolling(BASIS_Z_N,min_periods=BASIS_Z_N).mean().shift(1); sd=d.rolling(BASIS_Z_N,min_periods=BASIS_Z_N).std(ddof=0).shift(1); z=(d-mu)/sd.replace(0,np.nan); return b,ch,z


def candidates(b,ch,z,streams,ts):
    r=b.loc[ts]; zr=z.loc[ts]; cr=ch.loc[ts]; valid=r.notna()&zr.notna()&cr.notna()
    if int(valid.sum())<MIN_CS:return []
    v=r[valid]; lo=v.quantile(QUANT); hi=v.quantile(1-QUANT); out=[]
    for sym in v.index:
        bz=float(zr[sym]); bc=float(cr[sym]); x=streams[sym]; fm=float(x.loc[ts,"funding_mean"]) if ts in x.index else np.nan; sr=float(x.loc[ts,"spot_ret24"]) if ts in x.index else np.nan
        if not all(np.isfinite(q) for q in (bz,bc,fm,sr)):continue
        if float(v[sym])>=hi and bz>=MIN_Z and bc<=0 and sr>=0: side=1
        elif float(v[sym])<=lo and bz<=-MIN_Z and bc>=0 and sr<=0: side=-1
        else:continue
        # Funding is recorded as causal context only; it never creates or
        # filters the direction. This avoids introducing an arbitrary funding
        # threshold after seeing results.
        out.append((sym,side,bz,fm))
    out.sort(key=lambda q:(abs(q[2]),q[0]),reverse=True); return out


def exit_trade(x,idx,side,entry,sl,tp):
    for j in range(idx,len(x)):
        b=x.iloc[j]; hs=float(b.low)<=sl if side==1 else float(b.high)>=sl; ht=float(b.high)>=tp if side==1 else float(b.low)<=tp
        if not(hs or ht):continue
        win=bool(ht and not hs); px=tp if win else sl; gross=NOTIONAL*(px-entry)/entry if side==1 else NOTIONAL*(entry-px)/entry; return x.index[j],gross-NOTIONAL*FEE*2,win
    return None


def simulate(streams,b,ch,z,stop):
    events=[]; raw=0
    for ts in b.index:
        for sym,side,bz,fm in candidates(b,ch,z,streams,ts):
            raw+=1; x=streams[sym]; i=x.index.get_loc(ts)
            if not isinstance(i,slice) and i+1<len(x):events.append((ts,sym,i,side,bz,fm))
    events.sort(key=lambda e:(e[0],e[1])); openp={}; last_exit={}; equity=INITIAL; trades=[]
    for ts,sym,i,side,bz,fm in events:
        x=streams[sym]; ei=i+1; ets=x.index[ei]
        for s in [s for s,p in openp.items() if p<ets]:del openp[s]
        if sym in openp or (sym in last_exit and ets<=last_exit[sym]) or len(openp)>=MAX_POS or equity<(len(openp)+1)*MARGIN:continue
        atr=float(x.iloc[i].atr)
        if not np.isfinite(atr) or atr<=0:continue
        raw_entry=float(x.iloc[ei].open); entry=raw_entry*(1+SLIP) if side==1 else raw_entry*(1-SLIP); d=stop*atr; sl,tp=(entry-d,entry+RR*d) if side==1 else (entry+d,entry-RR*d)
        res=exit_trade(x,ei,side,entry,sl,tp)
        if res is None:continue
        xt,pnl,win=res; openp[sym]=xt; last_exit[sym]=xt; equity+=pnl
        trades.append(dict(signal_ts=ts,entry_ts=ets,exit_ts=xt,symbol=sym,side="LONG" if side==1 else "SHORT",basis=float(b.loc[ts,sym]),basis_z=bz,basis_change_24h=float(ch.loc[ts,sym]),funding_mean=fm,entry=entry,sl=sl,tp=tp,win=int(win),pnl=float(pnl)))
    return pd.DataFrame(trades),raw


def report(stop,t,raw):
    print("\n"+"="*78+f"\nSTAGE-0 STOP = {stop:.2f} ATR | RR = 1:2\n"+"="*78); n=len(t)
    if n==0:print("Raw candidate signals   :",raw); print("Closed trades           : 0"); return
    p=t.pnl; w=int((p>0).sum()); l=n-w; gw=p[p>0].sum(); gl=-p[p<=0].sum(); pf=gw/gl if gl else float("inf"); eq=INITIAL+p.cumsum(); pk=eq.cummax(); dd=pk-eq; ddp=(dd/pk.replace(0,np.nan)).max()*100; st=cur=0
    for v in p:
        cur=cur+1 if v<=0 else 0; st=max(st,cur)
    print(f"Raw candidate signals   : {raw}\nClosed trades           : {n}\nWins                    : {w}\nLosses                  : {l}\nWin Rate                : {100*w/n:.2f}%\nProfit Factor           : {pf:.4f}\nNet PnL                 : ${p.sum():,.2f}\nMax Drawdown            : ${dd.max():,.2f}\nMax Drawdown %          : {ddp:.2f}%\nMax Loss Streak         : {st}\nExpectancy / Trade      : ${p.mean():.2f}\nFinal Equity            : ${eq.iloc[-1]:,.2f}")
    for side,g in t.groupby("side"):
        print(f"SIDE {side:5s} trades={len(g):4d} WR={100*g.win.mean():6.2f}% PnL=${g.pnl.sum():,.2f}")
    print("\nBY SYMBOL")
    for s,g in t.groupby("symbol"):
        print(f"{s:10s} trades={len(g):4d} WR={100*g.win.mean():6.2f}% PnL=${g.pnl.sum():,.2f}")


def main():
    print("HUNTER-V23.1-STAGE0 — FUTURES BASIS DISLOCATION / REPRICING"); print("XT 1H spot + perpetual | RR 1:2 | fixed universe | no optimization | no spot-gap filling")
    raw={}
    for n,sym in enumerate(SYMBOLS,1):
        print(f"\n[{n}/{len(SYMBOLS)}] {sym}: futures"); fut=fetch_kline(FUT_URL,sym,"fut_")
        print(f"[{n}/{len(SYMBOLS)}] {sym}: spot"); spot=fetch_kline(SPOT_URL,sym,"spot_")
        print(f"[{n}/{len(SYMBOLS)}] {sym}: funding"); fund=fetch_funding(sym); raw[sym]=build(fut,spot,fund)
        if len(raw[sym])<int((LOOKBACK_DAYS+WARMUP_DAYS)*24*.95):raise RuntimeError(f"{sym}: merged history too short")
    b,ch,z=panel(raw); print(f"\nData preparation complete.\nCommon completed timestamps: {len(b)}")
    for stop in STOPS:
        t,rawn=simulate(raw,b,ch,z,stop); report(stop,t,rawn); REPORT.mkdir(parents=True,exist_ok=True)
        if not t.empty:t.to_csv(REPORT/f"trades_stop_{stop:.2f}.csv",index=False)
    print("\nSTAGE-0 DECISION PROTOCOL\nNo stop is selected here. Apply the pre-registered activity/edge/DD gates externally.\nIf the family fails, close V23.1 without tuning symbols, sides, thresholds, funding bounds, or stops.\nIf it passes, proceed to clean walk-forward validation with the hypothesis locked.")

if __name__=="__main__":main()
