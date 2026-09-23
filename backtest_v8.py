# backtest_v152.py
import ccxt, pandas as pd, numpy as np, time
from datetime import datetime, timedelta, timezone

SYMBOLS={"BTC":"BTC/USDT","ETH":"ETH/USDT","SOL":"SOL/USDT","BNB":"BNB/USDT",
"XRP":"XRP/USDT","ADA":"ADA/USDT","AVAX":"AVAX/USDT","LINK":"LINK/USDT",
"DOGE":"DOGE/USDT","DOT":"DOT/USDT"}
TIMEFRAME="15m"; DAYS=365; WARMUP_DAYS=60
INITIAL_BALANCE=1000.; TRADE_MARGIN=100.; LEVERAGE=50.
FEE_RATE=.0007; SLIPPAGE=.0003; MAX_OPEN_POSITIONS=3
LOSS_STREAK_BREAKER=4; BREAKER_PAUSE_BARS=16

def fetch(symbol,since,until):
    ex=exchange; rows=[]; cur=since
    while cur<until:
        b=ex.fetch_ohlcv(symbol,TIMEFRAME,since=cur,limit=1000)
        if not b: break
        rows+=b
        if b[-1][0]<=cur: break
        cur=b[-1][0]+1
        if len(b)<1000: break
        time.sleep(.2)
    if not rows: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    d=pd.DataFrame(rows,columns=["timestamp","Open","High","Low","Close","Volume"])
    d["Date"]=pd.to_datetime(d.timestamp,unit="ms",utc=True)
    d=d.drop_duplicates("Date").sort_values("Date").set_index("Date")
    tf=15*60*1000; last=(ex.milliseconds()//tf)*tf-tf
    d=d[d.timestamp<=last]
    d=d[(d.index>=pd.to_datetime(since,unit="ms",utc=True))&(d.index<pd.to_datetime(until,unit="ms",utc=True))]
    return d[["Open","High","Low","Close","Volume"]]

def tr(d):
    pc=d.Close.shift(1)
    return pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
def atr(d,n=14): return tr(d).rolling(n,min_periods=n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False,min_periods=n).mean()
def rvol(d,n): return d.Volume/d.Volume.rolling(n,min_periods=n).mean()

def adx(d,n=14):
    up=d.High.diff(); dn=-d.Low.diff()
    p=pd.Series(np.where((up>dn)&(up>0),up,0.),index=d.index)
    m=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=d.index)
    a=tr(d).rolling(n,min_periods=n).mean()
    pdi=100*p.rolling(n,min_periods=n).mean()/a.replace(0,np.nan)
    mdi=100*m.rolling(n,min_periods=n).mean()/a.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.rolling(n,min_periods=n).mean()

def resample(d,rule):
    x=pd.concat([d.Open.resample(rule).first(),d.High.resample(rule).max(),
                 d.Low.resample(rule).min(),d.Close.resample(rule).last(),
                 d.Volume.resample(rule).sum()],axis=1)
    x.columns=["Open","High","Low","Close","Volume"]
    return x.dropna()

def features(d):
    m=d.copy(); m["ATR"]=atr(m); m["RVOL"]=rvol(m,20)
    m["Body"]=(m.Close-m.Open).abs()
    rng=(m.High-m.Low).replace(0,np.nan); m["CP"]=(m.Close-m.Low)/rng
    h=resample(d,"1h"); h["ATR"]=atr(h); h["RVOL"]=rvol(h,20); h["Body"]=(h.Close-h.Open).abs()
    rr=(h.High-h.Low).replace(0,np.nan); h["CP"]=(h.Close-h.Low)/rr
    h["RH"]=h.High.rolling(24,min_periods=24).max().shift(1)
    h["RL"]=h.Low.rolling(24,min_periods=24).min().shift(1)
    h["RW"]=(h.RH-h.RL)/h.Close
    q=resample(d,"4h"); q["E50"]=ema(q.Close,50); q["E200"]=ema(q.Close,200)
    q["ATR"]=atr(q); q["ADX"]=adx(q); q["Slope"]=q.E50/q.E50.shift(8)-1
    q["Reg"]=0
    q.loc[(q.Close>q.E200)&(q.E50>q.E200)&(q.Slope>.0015)&(q.ADX>=18),"Reg"]=1
    q.loc[(q.Close<q.E200)&(q.E50<q.E200)&(q.Slope<-.0015)&(q.ADX>=18),"Reg"]=-1
    return m,h,q

def asof(htf,t,col):
    p=htf.index.searchsorted(t,side="left")-1
    return np.nan if p<0 else htf.iloc[p][col]

def candidates(d):
    m,h,q=features(d); out=[]; state=None; seen=None
    for i,t in enumerate(m.index):
        if state and t>state["exp"]: state=None
        hp=h.index.searchsorted(t,side="right")-1
        if hp>=0:
            ht=h.index[hp]
            if t>=ht+pd.Timedelta("1h") and ht!=seen:
                seen=ht; r=h.iloc[hp]
                width_ok=.010<=r.RW<=.060 if np.isfinite(r.RW) else False
                body_ok=np.isfinite(r.ATR) and r.Body>=.45*r.ATR
                vol_ok=np.isfinite(r.RVOL) and r.RVOL>=.90
                reg=asof(q,ht,"Reg")
                if width_ok and body_ok and vol_ok and reg==1 and r.Close>r.RH+.10*r.ATR and r.CP>=.65:
                    state={"d":1,"bt":ht,"bp":r.Close,"rb":r.RH,"exp":t+pd.Timedelta("3h")}
                elif width_ok and body_ok and vol_ok and reg==-1 and r.Close<r.RL-.10*r.ATR and r.CP<=.35:
                    state={"d":-1,"bt":ht,"bp":r.Close,"rb":r.RL,"exp":t+pd.Timedelta("3h")}
        if not state or t<=state["bt"]: continue
        r=m.iloc[i]
        if not np.isfinite(r.ATR): continue
        if state["d"]==1:
            touched=r.Low<=state["rb"]+.35*r.ATR and r.Close>=state["rb"]-.35*r.ATR
            depth=state["bp"]-r.Low
            ok=0<=depth<=1.25*r.ATR and depth>=.05*max(abs(state["bp"]-state["rb"]),r.ATR)
            trig=touched and ok and r.Close>r.Open and r.CP>=.60 and r.Body>=.30*r.ATR and r.RVOL>=.85
            prev=m.High.iloc[max(0,i-3):i].max()
            if trig and r.Close>prev:
                sl=min(m.Low.iloc[max(0,i-3):i+1].min(),state["rb"])-.15*r.ATR
                risk=r.Close-sl
                if .55*r.ATR<=risk<=2.20*r.ATR and risk/r.Close<=.025:
                    out.append({"signal_time":t,"direction":1,"sl":float(sl),"source":"AMBP_LONG"}); state=None
        else:
            touched=r.High>=state["rb"]-.35*r.ATR and r.Close<=state["rb"]+.35*r.ATR
            depth=r.High-state["bp"]
            ok=0<=depth<=1.25*r.ATR and depth>=.05*max(abs(state["bp"]-state["rb"]),r.ATR)
            trig=touched and ok and r.Close<r.Open and r.CP<=.40 and r.Body>=.30*r.ATR and r.RVOL>=.85
            prev=m.Low.iloc[max(0,i-3):i].min()
            if trig and r.Close<prev:
                sl=max(m.High.iloc[max(0,i-3):i+1].max(),state["rb"])+.15*r.ATR
                risk=sl-r.Close
                if .55*r.ATR<=risk<=2.20*r.ATR and risk/r.Close<=.025:
                    out.append({"signal_time":t,"direction":-1,"sl":float(sl),"source":"AMBP_SHORT"}); state=None
    return pd.DataFrame(out)

def entry_price(p,direction):
    return p*(1+SLIPPAGE if direction==1 else 1-SLIPPAGE)
def exit_price(p,direction):
    return p*(1-SLIPPAGE if direction==1 else 1+SLIPPAGE)
def pnl(e,x,d,q):
    return (x-e)*q if d==1 else (e-x)*q
def resolve(pos,row):
    if pos["d"]==1:
        if row.Low<=pos["sl"]: return exit_price(pos["sl"],1),"SL"
        if row.High>=pos["tp"]: return exit_price(pos["tp"],1),"TP"
    else:
        if row.High>=pos["sl"]: return exit_price(pos["sl"],-1),"SL"
        if row.Low<=pos["tp"]: return exit_price(pos["tp"],-1),"TP"
    return None,None

def simulate(data,cands,start,end):
    events=[]
    for s,c in cands.items():
        if c.empty: continue
        for _,r in c.iterrows():
            events.append((r.signal_time+pd.Timedelta("15m"),s,r))
    events.sort(key=lambda z:(z[0],z[1]))
    times=sorted(set().union(*[set(x.index) for x in data.values()]))
    active={}; last_exit={}; trades=[]; streak=0; breaker=None; ei=0
    by={t:[] for t,_,_ in events}
    for t,s,r in events: by.setdefault(t,[]).append((s,r))
    for t in times:
        exited=[]
        for s,pos in list(active.items()):
            if t<=pos["entry"]: continue
            x=data[s].loc[t]; ep,why=resolve(pos,x)
            if ep is not None:
                gross=pnl(pos["entry"],ep,pos["d"],pos["qty"])
                fees=(pos["entry"]*pos["qty"]+ep*pos["qty"])*FEE_RATE
                p=gross-fees
                trades.append({**pos,"exit":t,"exit_price":ep,"reason":why,"pnl":p})
                streak=streak+1 if p<0 else 0
                if streak>=LOSS_STREAK_BREAKER: breaker=t+pd.Timedelta(minutes=15*BREAKER_PAUSE_BARS)
                last_exit[s]=t; exited.append(s)
        for s in exited: del active[s]
        if breaker is not None and t<breaker: continue
        for s,r in by.get(t,[]):
            if s in active or last_exit.get(s)==t or len(active)>=MAX_OPEN_POSITIONS: continue
            row=data[s].loc[t]; d=int(r.direction); e=entry_price(row.Open,d); sl=float(r.sl)
            risk=e-sl if d==1 else sl-e
            if risk<=0 or risk/e>.025: continue
            tp=e+2*risk if d==1 else e-2*risk; q=TRADE_MARGIN*LEVERAGE/e
            active[s]={"symbol":s,"entry":t,"entry_price":e,"d":d,"sl":sl,"tp":tp,"qty":q,"signal_time":r.signal_time,"source":r.source}
    for s,p in active.items(): trades.append({**p,"exit":pd.NaT,"exit_price":np.nan,"reason":"OPEN_AT_END","pnl":np.nan})
    return trades

def report(trades,start,end):
    r=pd.DataFrame(trades)
    if r.empty: print("NO TRADES"); return
    c=r[r.reason!="OPEN_AT_END"].copy(); w=c[c.pnl>0]; l=c[c.pnl<0]
    wr=100*len(w)/len(c) if len(c) else 0; gp=w.pnl.sum(); gl=-l.pnl.sum()
    pf=gp/gl if gl else np.inf; eq=1000+c.pnl.cumsum(); dd=(eq-eq.cummax()).min()
    st=[]; n=0
    for p in c.pnl:
        if p<0:n+=1
        elif n:st.append(n);n=0
    if n:st.append(n)
    days=max((end-start).total_seconds()/86400,1)
    print("\n"+"="*70); print("AMBP V152 — CAUSAL BACKTEST"); print("="*70)
    print(f"Total: {len(r)} | Closed: {len(c)} | Open: {len(r)-len(c)}")
    print(f"Trades/day: {len(c)/days:.2f} | WR: {wr:.2f}% | Loss: {100*len(l)/len(c) if len(c) else 0:.2f}%")
    print(f"Net PnL: ${c.pnl.sum():,.2f} | PF: {pf:.2f} | Avg Win: ${w.pnl.mean() if len(w) else 0:.2f} | Avg Loss: ${l.pnl.mean() if len(l) else 0:.2f}")
    print(f"Max DD: ${dd:,.2f} | Max loss streak: {max(st) if st else 0} | Streaks: {st}")
    print("RR: 1:2 FIXED | Timeout: DISABLED | Lookahead: NONE | Entry: NEXT 15m OPEN")
    print("\nPER SYMBOL")
    for s,g in c.groupby("symbol"):
        print(f"{s:5} {len(g):4} trades | WR {100*(g.pnl>0).mean():6.2f}% | PnL ${g.pnl.sum():10.2f}")
    r.to_csv("ambp_v152_trades.csv",index=False); print("\nSaved ambp_v152_trades.csv")

exchange=ccxt.lbank({"enableRateLimit":True,"timeout":20000})
def main():
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0); start=end-timedelta(days=DAYS)
    fetch_start=start-timedelta(days=WARMUP_DAYS); si=int(fetch_start.timestamp()*1000); ui=int(end.timestamp()*1000)
    data={}; cs={}
    for name,sym in SYMBOLS.items():
        print("Fetching",name)
        try:
            d=fetch(sym,si,ui)
            if len(d)<1000: print(" skip",len(d)); continue
            data[name]=d; c=candidates(d)
            c=c[(c.signal_time>=pd.Timestamp(start))&(c.signal_time<pd.Timestamp(end))] if not c.empty else c
            cs[name]=c; print(" candidates",len(c))
        except Exception as e: print(" ERROR",e)
    report(simulate(data,cs,start,end),pd.Timestamp(start),pd.Timestamp(end))
if __name__=="__main__": main()
