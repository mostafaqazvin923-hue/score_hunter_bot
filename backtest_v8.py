#!/usr/bin/env python3
"""
HUNTER-V151 — DERIVATIVES FUNDING DISLOCATION

Real derivatives-positioning research using XT's historical FUNDING records.
Historical XT public OI snapshots are not exposed as a historical series by the
current public API; therefore this version does NOT fabricate OI history.

Architecture
------------
15m raw -> 1H signal -> 4H context -> 1D regime

Alpha hypothesis
----------------
Crowded funding + price/funding disagreement + a causal 1H release candle.

LONG setup:
  - 1D bull regime and 4H bull context.
  - funding was abnormally negative / short-crowded before the trigger.
  - funding begins normalising while price holds/accelerates upward.
  - 1H closes through a prior 12h structure high with displacement.
  - volume confirms.

SHORT is the exact mirror.

Funding is historical exchange data, not a synthetic OHLCV proxy. Funding
records are merged causally and lagged one 1H bar so a record stamped at a
settlement time cannot leak into the candle that produced it.

Execution / integrity
---------------------
- Entry at next 1H open.
- Fixed RR 1:2.
- $100 margin, 50x leverage.
- No timeout, no BE, no trailing.
- No overlapping trades; max 3 portfolio positions.
- No entry on the timestamp a prior trade closes.
- Same-candle SL+TP => LOSS.
- Unresolved positions at dataset end remain OPEN.
- 365d XT 15m collector is the verified V144 collector.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SYMBOLS=["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
BASE="https://fapi.xt.com"
DATA_DIR=Path("data/xt_futures_v151")
LIMIT=1500; INTERVAL="15m"; DAYS=365; WARMUP_DAYS=35; MIN_ROWS=30000; MAX_RETRIES=6
INITIAL_EQUITY=1000.0; TRADE_MARGIN=100.0; LEVERAGE=50.0; FEE_RATE=0.0007; SLIPPAGE=0.0003; RR=2.0; MAX_OPEN_POSITIONS=3
S=requests.Session(); S.headers.update({"User-Agent":"HUNTER-backtest/151"})


def resolve_symbols():
    js=S.get(f"{BASE}/future/market/v1/public/symbol/list",timeout=20).json()
    data=js.get("result",js); rows=data.get("symbols",[]) if isinstance(data,dict) else data; out={}
    for x in rows:
        if not isinstance(x,dict): continue
        raw=str(x.get("symbol",x.get("name",""))).upper(); compact=raw.replace("_","").replace("-","").replace("/","").replace(":","")
        for a in SYMBOLS:
            aliases={a+"USDT",a+"_USDT",a+"/USDT",a+"/USDT:USDT"}; norms={z.replace("_","").replace("-","").replace("/","").replace(":","") for z in aliases}
            if compact in norms: out[a]=x.get("symbol") or x.get("name")
    return out


def get_json(url,params):
    last=None
    for i in range(MAX_RETRIES):
        try:
            r=S.get(url,params=params,timeout=20); r.raise_for_status(); js=r.json()
            if isinstance(js,dict):
                for k in ("code","returnCode"):
                    v=js.get(k)
                    if v not in (None,0,"0",200,"200"):
                        raise RuntimeError(f"XT {k}={v}: {js.get('msg',js.get('msgInfo',js.get('message','')))}")
            return js
        except Exception as e:
            last=e; time.sleep(min(2,0.3*(i+1)))
    raise RuntimeError(f"XT request failed: {last}")


def parse_kline(js):
    raw=js.get("result",js) if isinstance(js,dict) else js
    arr=raw.get("data",raw.get("rows",raw.get("list",[]))) if isinstance(raw,dict) else raw
    out=[]
    for x in arr if isinstance(arr,list) else []:
        try:
            if isinstance(x,dict): vals=[x.get("t",x.get("timestamp")),x.get("o",x.get("open")),x.get("h",x.get("high")),x.get("l",x.get("low")),x.get("c",x.get("close")),x.get("a",x.get("volume",x.get("q",0)))]
            else: vals=x[:6]
            out.append([int(vals[0]),*map(float,vals[1:6])])
        except Exception: pass
    return out


def fetch_symbol(asset,xt_symbol,days):
    now=int(time.time()*1000); start=now-int((days+WARMUP_DAYS)*86400*1000); step=15*60*1000; end=(now//step)*step-1
    best=[]; endpoint=f"{BASE}/future/market/v1/public/q/kline"
    for whole in range(5):
        rows=[]; cursor=start; guard=0
        while cursor<end and guard<2000:
            guard+=1; wend=min(end,cursor+LIMIT*step-1); params={"symbol":xt_symbol,"interval":INTERVAL,"startTime":cursor,"endTime":wend,"limit":LIMIT}
            batch=[]
            for _ in range(6):
                try: batch=parse_kline(get_json(endpoint,params));
                except Exception: batch=[]
                if batch: break
                time.sleep(.5)
            batch=[z for z in batch if start<=z[0]<=end]
            if not batch: break
            rows.extend(batch); mx=max(z[0] for z in batch); cursor=mx+1; time.sleep(.08)
        if rows:
            df=pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume"]).drop_duplicates("timestamp").sort_values("timestamp")
            if len(df)>len(best): best=df
            if len(df)>=MIN_ROWS and int(df.timestamp.max()-df.timestamp.min())>=int(days*86400*1000*.90):
                df.timestamp=pd.to_datetime(df.timestamp,unit="ms",utc=True); return df.set_index("timestamp").dropna()
        time.sleep(whole+1)
    raise RuntimeError(f"Insufficient XT data for {asset}: got {len(best)} rows")


def ensure_data(data_dir,days):
    data_dir.mkdir(parents=True,exist_ok=True); mp=resolve_symbols(); print(f"Verified XT symbols: {len(mp)} / {len(SYMBOLS)}")
    if len(mp)!=len(SYMBOLS): raise RuntimeError("XT symbol mapping incomplete")
    out={}
    for a in SYMBOLS:
        df=fetch_symbol(a,mp[a],days); df.to_csv(data_dir/f"{a}_USDT_15m.csv"); out[a]=df
        gaps=int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum()); print(f"{a:<8} rows={len(df)} gaps={gaps}")
    return out


def fetch_funding_history(xt_symbol,days):
    """Fetch backward through XT public funding-rate-record using id pagination."""
    endpoint=f"{BASE}/future/market/v1/public/q/funding-rate-record"
    target_ms=int(time.time()*1000)-int((days+WARMUP_DAYS)*86400*1000)
    rows=[]; last_id=None
    for page in range(200):
        params={"symbol":xt_symbol,"limit":1000,"direction":"PREV"}
        if last_id is not None: params["id"]=last_id
        js=get_json(endpoint,params); result=js.get("result",{}) if isinstance(js,dict) else {}
        items=result.get("items",[]) if isinstance(result,dict) else []
        if not items: break
        clean=[]
        for z in items:
            try: clean.append({"id":int(z["id"]),"timestamp":int(z["createdTime"]),"funding":float(z["fundingRate"])})
            except Exception: pass
        if not clean: break
        rows.extend(clean)
        min_ts=min(z["timestamp"] for z in clean); new_id=min(z["id"] for z in clean)
        if min_ts<=target_ms: break
        if last_id==new_id: break
        last_id=new_id; time.sleep(.12)
    if not rows: raise RuntimeError(f"No funding history returned for {xt_symbol}")
    df=pd.DataFrame(rows).drop_duplicates("id").sort_values("timestamp")
    df=df[df.timestamp>=target_ms]
    if len(df)<100: raise RuntimeError(f"Funding history too short for {xt_symbol}: {len(df)} records")
    df.timestamp=pd.to_datetime(df.timestamp,unit="ms",utc=True)
    return df.set_index("timestamp")


def resample(df,rule):
    return df.resample(rule,label="right",closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def atr14(x):
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()


def adx14(x):
    up=x.high.diff(); dn=-x.low.diff(); p=up.where((up>dn)&(up>0),0.0); m=dn.where((dn>up)&(dn>0),0.0)
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1); a=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    pdi=100*p.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/a; mdi=100*m.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); return dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()


def build_features(all15, funding):
    f={}; h1={a:resample(df,"1h") for a,df in all15.items()}; h4={a:resample(df,"4h") for a,df in all15.items()}; d1={a:resample(df,"1D") for a,df in all15.items()}
    for a in SYMBOLS:
        x=h1[a].copy(); q=h4[a].copy(); d=d1[a].copy(); fr=funding[a][["funding"]].copy()
        # Funding becomes usable only after its settlement timestamp; lag one full H1 bar for strict causality.
        left=pd.DataFrame({"ts":x.index}); right=fr.reset_index().rename(columns={fr.index.name or "index":"ts"}); merged=pd.merge_asof(left.sort_values("ts"),right.sort_values("ts"),on="ts",direction="backward"); merged.index=x.index; x["funding"]=merged["funding"].to_numpy()
        x["funding"]=x["funding"].ffill().shift(1)
        x["atr"]=atr14(x); x["ema20"]=x.close.ewm(span=20,adjust=False).mean(); x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
        x["ret12"]=x.close.pct_change(12); x["ret24"]=x.close.pct_change(24)
        x["vol_z"]=(x.volume-x.volume.rolling(48,min_periods=24).mean())/x.volume.rolling(48,min_periods=24).std()
        x["range_atr"]=(x.high-x.low)/x.atr; x["body_atr"]=(x.close-x.open).abs()/x.atr; x["close_pos"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
        x["fund_z"]=(x.funding-x.funding.rolling(120,min_periods=60).mean())/x.funding.rolling(120,min_periods=60).std()
        x["fund_chg"]=x.funding.diff(3); x["fund_abs_pct"]=x.funding.abs().rolling(120,min_periods=60).rank(pct=True)
        q["ema20"]=q.close.ewm(span=20,adjust=False).mean(); q["ema50"]=q.close.ewm(span=50,adjust=False).mean(); q["ema200"]=q.close.ewm(span=200,adjust=False).mean(); q["adx"]=adx14(q); q["slope"]=q.ema50.pct_change(3)
        q["trend"]=np.where((q.close>q.ema20)&(q.ema20>q.ema50)&(q.ema50>q.ema200),1,np.where((q.close<q.ema20)&(q.ema20<q.ema50)&(q.ema50<q.ema200),-1,0))
        d["ema50"]=d.close.ewm(span=50,adjust=False).mean(); d["ema200"]=d.close.ewm(span=200,adjust=False).mean(); d["slope"]=d.ema50.pct_change(5)
        d["regime"]=np.where((d.close>d.ema50)&(d.ema50>d.ema200)&(d.slope>0),1,np.where((d.close<d.ema50)&(d.ema50<d.ema200)&(d.slope<0),-1,0))
        x["h4_trend"]=q.trend.reindex(x.index,method="ffill"); x["h4_adx"]=q.adx.reindex(x.index,method="ffill"); x["h4_slope"]=q.slope.reindex(x.index,method="ffill"); x["d1_regime"]=d.regime.reindex(x.index,method="ffill")
        f[a]=x
    return f


def signals(f,p):
    out=[]; n=int(p["structure"])
    for a,x in f.items():
        hi=x.high.rolling(n,min_periods=n).max().shift(1); lo=x.low.rolling(n,min_periods=n).min().shift(1)
        for i in range(max(260,n+5),len(x)-1):
            r=x.iloc[i]
            vals=[r.atr,r.h4_adx,r.h4_slope,r.d1_regime,r.funding,r.fund_z,r.fund_chg,r.vol_z,r.range_atr,r.body_atr,r.close_pos,hi.iloc[i],lo.iloc[i]]
            if not all(np.isfinite(v) for v in vals): continue
            bull=r.d1_regime==1 and r.h4_trend==1 and r.h4_slope>0 and r.h4_adx>=p["adx"]
            bear=r.d1_regime==-1 and r.h4_trend==-1 and r.h4_slope<0 and r.h4_adx>=p["adx"]
            release=r.range_atr>=p["range_atr"] and r.body_atr>=p["body_atr"] and r.vol_z>=p["vol_z"]
            long_break=r.close>hi.iloc[i]+p["buffer"]*r.atr and r.close_pos>=p["close_pos"]
            short_break=r.close<lo.iloc[i]-p["buffer"]*r.atr and r.close_pos<=1-p["close_pos"]
            # Contrarian crowding: negative funding supports LONG, positive funding supports SHORT.
            long_funding=r.fund_z<=-p["fund_z"] and r.fund_chg>0
            short_funding=r.fund_z>=p["fund_z"] and r.fund_chg<0
            if bull and long_funding and release and long_break:
                out.append({"asset":a,"signal_ts":x.index[i],"side":"LONG","family":"FUNDING_DISLOCATION","atr":float(r.atr)})
            elif bear and short_funding and release and short_break:
                out.append({"asset":a,"signal_ts":x.index[i],"side":"SHORT","family":"FUNDING_DISLOCATION","atr":float(r.atr)})
    return out


def backtest(f,sig,p):
    byts={}
    for z in sig: byts.setdefault(z["signal_ts"],[]).append(z)
    all_ts=sorted(set().union(*(x.index for x in f.values()))); positions=[]; trades=[]; last_close=pd.Timestamp.min.tz_localize("UTC")
    for ts in all_ts:
        for pos in positions[:]:
            x=f[pos["asset"]]
            if ts not in x.index or ts<=pos["entry_ts"]: continue
            row=x.loc[ts]; sl_hit=row.low<=pos["sl"] if pos["side"]=="LONG" else row.high>=pos["sl"]; tp_hit=row.high>=pos["tp"] if pos["side"]=="LONG" else row.low<=pos["tp"]
            if sl_hit or tp_hit:
                outcome="LOSS" if sl_hit else "WIN"; ex=pos["sl"] if sl_hit else pos["tp"]; gross=((ex-pos["entry"])/pos["entry"] if pos["side"]=="LONG" else (pos["entry"]-ex)/pos["entry"])*TRADE_MARGIN*LEVERAGE; pnl=gross-(TRADE_MARGIN*LEVERAGE*FEE_RATE*2)
                trades.append({**pos,"exit_ts":ts,"outcome":outcome,"pnl":pnl}); positions.remove(pos); last_close=ts
        if ts<=last_close or len(positions)>=MAX_OPEN_POSITIONS: continue
        used=set()
        for z in byts.get(ts,[]):
            if len(positions)>=MAX_OPEN_POSITIONS or z["asset"] in used or any(p0["asset"]==z["asset"] for p0 in positions): continue
            x=f[z["asset"]]; fut=x.index[x.index>ts]
            if len(fut)==0: continue
            ets=fut[0]; entry=float(x.loc[ets,"open"])*(1+SLIPPAGE if z["side"]=="LONG" else 1-SLIPPAGE); risk=p["stop_atr"]*z["atr"]; sl=entry-risk if z["side"]=="LONG" else entry+risk; tp=entry+RR*risk if z["side"]=="LONG" else entry-RR*risk
            positions.append({"asset":z["asset"],"family":z["family"],"side":z["side"],"signal_ts":ts,"entry_ts":ets,"entry":entry,"sl":sl,"tp":tp}); used.add(z["asset"])
    return trades,positions


def metrics(trades):
    wins=sum(z["outcome"]=="WIN" for z in trades); gw=sum(z["pnl"] for z in trades if z["pnl"]>0); gl=-sum(z["pnl"] for z in trades if z["pnl"]<0); eq=INITIAL_EQUITY; peak=eq; dd=0; streak=mx=0
    for z in sorted(trades,key=lambda q:q["exit_ts"]):
        eq+=z["pnl"]; peak=max(peak,eq); dd=min(dd,eq-peak); streak=streak+1 if z["outcome"]=="LOSS" else 0; mx=max(mx,streak)
    if trades: days=max(1,(max(z["exit_ts"] for z in trades)-min(z["exit_ts"] for z in trades)).total_seconds()/86400)
    else: days=1
    return {"trades":len(trades),"wr":100*wins/len(trades) if trades else 0,"pf":gw/gl if gl else 0,"pnl":sum(z["pnl"] for z in trades),"dd":dd,"max_streak":mx,"trades_day":len(trades)/days}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=DAYS); ap.add_argument("--data-dir",default=str(DATA_DIR)); args=ap.parse_args()
    print("HUNTER-V151 — DERIVATIVES FUNDING DISLOCATION")
    print("Signal=1H | Context=4H | Regime=1D | Raw source=15m | Positioning=funding history")
    print("NOTE: XT public API exposes current OI, but not historical OI series; no synthetic OI is used.")
    all15=ensure_data(Path(args.data_dir),args.days); mp=resolve_symbols(); funding={}
    print("Fetching historical XT funding records...")
    for a in SYMBOLS:
        funding[a]=fetch_funding_history(mp[a],args.days); print(f"{a:<8} funding_records={len(funding[a])}")
    f=build_features(all15,funding)
    # Four deliberately different hypotheses, not a large optimization grid.
    configs=[
      {"name":"A_CROWDED_RELEASE","structure":12,"fund_z":1.0,"range_atr":1.15,"body_atr":0.35,"vol_z":0.25,"buffer":0.00,"close_pos":0.70,"adx":16,"stop_atr":1.00},
      {"name":"B_DEEP_CROWD","structure":12,"fund_z":1.5,"range_atr":1.15,"body_atr":0.35,"vol_z":0.25,"buffer":0.00,"close_pos":0.70,"adx":16,"stop_atr":1.00},
      {"name":"C_STRUCTURAL_RELEASE","structure":24,"fund_z":1.0,"range_atr":1.25,"body_atr":0.40,"vol_z":0.50,"buffer":0.03,"close_pos":0.72,"adx":20,"stop_atr":1.00},
      {"name":"D_DEEP_STRUCTURAL","structure":24,"fund_z":1.5,"range_atr":1.25,"body_atr":0.40,"vol_z":0.50,"buffer":0.03,"close_pos":0.72,"adx":20,"stop_atr":1.10},
    ]
    print(f"Testing {len(configs)} causal funding-dislocation hypotheses...")
    results=[]
    for i,p in enumerate(configs,1):
        sig=signals(f,p); trades,op=backtest(f,sig,p); m=metrics(trades); m.update({"open":len(op),"signals":len(sig)}); results.append((m,p)); print(f" tested {i}/{len(configs)}: {p['name']} | signals={len(sig)} trades={m['trades']}")
    print("\nBASE RESULTS:")
    for i,(m,p) in enumerate(results,1): print(f"{i}. {p['name']:<24} trades={m['trades']:>4} WR={m['wr']:>6.2f}% PF={m['pf']:.3f} PnL=${m['pnl']:,.2f} DD=${m['dd']:,.2f} streak={m['max_streak']:>2} t/day={m['trades_day']:.2f} open={m['open']}")
    viable=[z for z in results if z[0]["trades"]>=100 and z[0]["pf"]>1.0]
    print("\nMEASURABLE_EDGE_PRESENT:",bool(viable))
    if viable: print("NEXT: strongest candidate -> walk-forward/OOS; do not optimize in-sample.")
    else: print("No funding-dislocation hypothesis showed PF>1 with >=100 realized trades; reject this family rather than force it.")

if __name__=="__main__": main()
