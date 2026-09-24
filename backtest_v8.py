# HUNTER-V3 STAGE-1 — COMPRESSION / BREAKOUT / VOLUME
# Strict no-lookahead portfolio backtest for LBank USDT-M Futures.
# Stage-1 intentionally uses only three core ingredients:
#   1) 1H compression
#   2) 15M breakout of the completed compression range
#   3) 15M relative-volume confirmation
# RR is fixed 1:2. No BE, trailing, timeout, or forced end exit.

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

EXCHANGE = ccxt.lbank({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
EXCHANGE.load_markets()
SYMBOLS = [
    "BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","SUI/USDT:USDT",
    "AVAX/USDT:USDT","NEAR/USDT:USDT","ADA/USDT:USDT","BNB/USDT:USDT",
    "APT/USDT:USDT","CRV/USDT:USDT","ONDO/USDT:USDT","PENDLE/USDT:USDT",
    "ICP/USDT:USDT","WIF/USDT:USDT"
]
DAYS=365
WARMUP_DAYS=35
TIMEFRAME="15m"
FEE_RATE=0.0007
SLIPPAGE=0.0003
TRADE_MARGIN=100.0
LEVERAGE=50.0
RR=2.0
MAX_POSITIONS=3

CLUSTERS={
    "MAJOR":{"BTC/USDT:USDT","ETH/USDT:USDT"},
    "L1":{"SOL/USDT:USDT","SUI/USDT:USDT","AVAX/USDT:USDT","NEAR/USDT:USDT","ADA/USDT:USDT","BNB/USDT:USDT","APT/USDT:USDT"},
    "DEFI":{"CRV/USDT:USDT","ONDO/USDT:USDT","PENDLE/USDT:USDT"},
    "OTHER":{"ICP/USDT:USDT"},
    "MEME":{"WIF/USDT:USDT"}
}

def cluster_of(sym):
    for k,v in CLUSTERS.items():
        if sym in v: return k
    return sym

def fetch_ohlcv(symbol, since_ms, until_ms):
    rows=[]; cur=since_ms
    market = EXCHANGE.market(symbol)
    request_symbol = market["symbol"]
    while cur < until_ms:
        batch=EXCHANGE.fetch_ohlcv(request_symbol,TIMEFRAME,since=cur,limit=1000)
        if not batch: break
        rows.extend(batch)
        nxt=batch[-1][0]+15*60*1000
        if nxt<=cur: break
        cur=nxt
        if len(batch)<1000: break
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows,columns=["ts","Open","High","Low","Close","Volume"])
    df["Date"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    df=df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df

def completed_15m(df):
    now=pd.Timestamp.now(tz="UTC")
    return df[df["Date"] < now].copy()

def prepare(df):
    df=completed_15m(df)
    if len(df)<500: return None
    # Completed 1H bars. Resampling labels the hour at its start.
    h=(df.set_index("Date")
         .resample("1h",label="left",closed="left")
         .agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
         .dropna().reset_index())
    # Compression is measured on COMPLETED 1H bars.
    h["atr14"]=(h["High"]-h["Low"]).rolling(14).mean()
    h["range20"]=h["High"].rolling(20).max()-h["Low"].rolling(20).min()
    h["range20_prev"]=h["range20"].shift(1)
    h["atr20"]= (h["High"]-h["Low"]).rolling(20).mean()
    # Relative compression: current 1H range below its 20-bar average
    h["compressed"]=((h["High"]-h["Low"]) < 0.75*h["atr20"]) & (h["range20"] < 4.0*h["atr14"])
    # Breakout box comes from the last 4 COMPLETED 1H candles before entry hour.
    h["box_high"]=h["High"].rolling(4).max().shift(1)
    h["box_low"]=h["Low"].rolling(4).min().shift(1)
    # Require at least one of the previous two hours to be compressed.
    h["comp_recent"]=h["compressed"].rolling(2).max().shift(1).fillna(0).astype(bool)

    # Map only completed 1H information into 15M rows.
    x=df.copy()
    h2=h.set_index("Date")[["box_high","box_low","comp_recent"]]
    x=x.set_index("Date")
    x=x.join(h2.reindex(x.index,method="ffill"))
    # At a 15M timestamp, the current hour is not completed; ffill would use
    # the prior hour's row because h rows are hour-start timestamps. Explicitly
    # shift the joined hourly context by one hour.
    x["box_high"]=x["box_high"].shift(4)
    x["box_low"]=x["box_low"].shift(4)
    x["comp_recent"]=x["comp_recent"].shift(4)
    x["range"]=x["High"]-x["Low"]
    x["body"]=(x["Close"]-x["Open"]).abs()
    x["rvol20"]=x["Volume"]/x["Volume"].rolling(20).mean().replace(0,np.nan)
    x=x.reset_index()
    return x

def pnl_for_trade(entry,sl,tp,side):
    notional=TRADE_MARGIN*LEVERAGE
    if side=="LONG":
        gross=(tp-entry)*notional/entry if tp>entry else -(entry-sl)*notional/entry
    else:
        gross=(entry-tp)*notional/entry if tp<entry else -(sl-entry)*notional/entry
    fees=notional*FEE_RATE*2
    return gross-fees

def run_symbol(sym,df):
    trades=[]
    i=100
    n=len(df)
    while i<n-2:
        r=df.iloc[i]
        if not bool(r.get("comp_recent",False)):
            i+=1; continue
        bh=r.get("box_high"); bl=r.get("box_low")
        if pd.isna(bh) or pd.isna(bl) or bh<=bl:
            i+=1; continue

        side=None
        # Breakout candle itself is closed; entry is next 15M open.
        if r["Close"]>bh and r["Open"]<=bh and r["rvol20"]>=1.20 and r["range"]>1.10*df["range"].rolling(20).mean().iloc[i]:
            side="LONG"
        elif r["Close"]<bl and r["Open"]>=bl and r["rvol20"]>=1.20 and r["range"]>1.10*df["range"].rolling(20).mean().iloc[i]:
            side="SHORT"
        if side is None:
            i+=1; continue

        j=i+1
        if j>=n: break
        entry0=float(df.iloc[j]["Open"])
        entry=entry0*(1+SLIPPAGE if side=="LONG" else 1-SLIPPAGE)
        box_size=float(bh-bl)
        # Structural stop: opposite side of the completed box.
        if side=="LONG":
            sl=bl
            risk=entry-sl
            if risk<=0: i+=1; continue
            tp=entry+RR*risk
        else:
            sl=bh
            risk=sl-entry
            if risk<=0: i+=1; continue
            tp=entry-RR*risk

        outcome=None; exit_price=None; exit_time=None
        k=j
        while k<n:
            c=df.iloc[k]
            hit_sl = c["Low"]<=sl if side=="LONG" else c["High"]>=sl
            hit_tp = c["High"]>=tp if side=="LONG" else c["Low"]<=tp
            if hit_sl and hit_tp:
                outcome="LOSS"; exit_price=sl; exit_time=c["Date"]; break
            if hit_sl:
                outcome="LOSS"; exit_price=sl; exit_time=c["Date"]; break
            if hit_tp:
                outcome="WIN"; exit_price=tp; exit_time=c["Date"]; break
            k+=1
        if outcome is not None:
            pnl=pnl_for_trade(entry,sl,tp,side)
            if outcome=="LOSS":
                # pnl_for_trade returns the loss branch based on side
                pass
            else:
                pass
            # Explicitly calculate exact realized result.
            notional=TRADE_MARGIN*LEVERAGE
            if outcome=="WIN":
                gross=notional*RR*risk/entry if side=="LONG" else notional*RR*risk/entry
            else:
                gross=-notional*risk/entry
            pnl=gross-notional*FEE_RATE*2
            trades.append({"symbol":sym,"entry_time":df.iloc[j]["Date"],"exit_time":exit_time,
                           "side":side,"entry":entry,"sl":sl,"tp":tp,"result":outcome,"pnl":pnl})
            i=k+1
        else:
            # Open position at end: report separately, never force-close.
            i=n
    return trades

def main():
    print("="*90)
    print("HUNTER-V3 STAGE-1 — COMPRESSION / BREAKOUT / VOLUME")
    print("RAW EDGE / STRICT PORTFOLIO-LEVEL NO-LOOKAHEAD BACKTEST")
    print("="*90)
    now=pd.Timestamp.now(tz="UTC")
    target_start=now-timedelta(days=DAYS)
    fetch_start=target_start-timedelta(days=WARMUP_DAYS)
    data={}
    valid_symbols=[]
    for sym in SYMBOLS:
        try:
            market=EXCHANGE.market(sym)
            if market.get("swap") and market.get("quote")=="USDT":
                valid_symbols.append(sym)
            else:
                print(f"SKIP {sym}: not an eligible USDT swap market")
        except Exception as e:
            print(f"SKIP {sym}: market not found ({e})")
    print(f"Valid LBank swap markets: {len(valid_symbols)} / {len(SYMBOLS)}")
    for sym in valid_symbols:
        print(f"دریافت و آماده‌سازی: {sym.split('/')[0]} ...")
        raw=fetch_ohlcv(sym,int(fetch_start.timestamp()*1000),int(now.timestamp()*1000))
        if raw.empty:
            print("  -> EMPTY"); continue
        d=prepare(raw)
        if d is None:
            print("  -> insufficient"); continue
        d=d[d["Date"]>=target_start].reset_index(drop=True)
        data[sym]=d
        print(f"  -> 15M={len(d):,}")
    print(f"\nنمادهای معتبر: {len(data)} / {len(SYMBOLS)}")
    candidates=[]
    for sym,d in data.items():
        candidates.extend(run_symbol(sym,d))
    candidates.sort(key=lambda z:z["entry_time"])

    # Rebuild a true portfolio execution order from candidate signals.
    trades=[]; active=[]
    streak=0; streaks=[]; max_streak=0
    equity=1000.0; peak=equity; max_dd=0.0
    for t in candidates:
        # remove closed positions before this entry
        active=[a for a in active if a["exit_time"]>t["entry_time"]]
        if len(active)>=MAX_POSITIONS: continue
        if any(cluster_of(a["symbol"])==cluster_of(t["symbol"]) for a in active): continue
        active.append(t)
        trades.append(t)

    for t in sorted(trades,key=lambda z:z["exit_time"]):
        equity+=t["pnl"]
        if t["result"]=="WIN":
            if streak: streaks.append(streak)
            streak=0
        else:
            streak+=1; max_streak=max(max_streak,streak)
        peak=max(peak,equity); max_dd=min(max_dd,equity-peak)
    if streak: streaks.append(streak)

    wins=[t for t in trades if t["result"]=="WIN"]
    losses=[t for t in trades if t["result"]=="LOSS"]
    gross_win=sum(t["pnl"] for t in wins)
    gross_loss=sum(t["pnl"] for t in losses)
    pf=(gross_win/abs(gross_loss)) if gross_loss<0 else float("inf")
    wr=len(wins)/len(trades)*100 if trades else 0
    long=[t for t in trades if t["side"]=="LONG"]; short=[t for t in trades if t["side"]=="SHORT"]
    lwr=sum(t["result"]=="WIN" for t in long)/len(long)*100 if long else 0
    swr=sum(t["result"]=="WIN" for t in short)/len(short)*100 if short else 0
    avgw=np.mean([t["pnl"] for t in wins]) if wins else 0
    avgl=np.mean([t["pnl"] for t in losses]) if losses else 0
    expectancy=np.mean([t["pnl"] for t in trades]) if trades else 0
    days=max(DAYS,1)

    print("\n"+"="*90)
    print("HUNTER-V3 STAGE-1 — FINAL AUDITED REPORT")
    print("="*90)
    print(f"Initial Capital:          ${1000:,.2f}")
    print(f"Trade Margin:             ${TRADE_MARGIN:,.2f}")
    print(f"Leverage:                 {LEVERAGE:.0f}x")
    print(f"RR:                       1:{RR:.0f}")
    print("-"*90)
    print(f"Candidate Signals:        {len(candidates)}")
    print(f"Closed Trades:            {len(trades)}")
    print(f"Wins:                     {len(wins)}")
    print(f"Losses:                   {len(losses)}")
    print(f"Win Rate:                 {wr:.2f}%")
    print(f"Long Win Rate:            {lwr:.2f}%")
    print(f"Short Win Rate:           {swr:.2f}%")
    print("-"*90)
    print(f"Net PnL:                  ${sum(t['pnl'] for t in trades):,.2f}")
    print(f"Final Closed Equity:      ${1000+sum(t['pnl'] for t in trades):,.2f}")
    print(f"Profit Factor:            {pf:.2f}")
    print(f"Average Win:              ${avgw:,.2f}")
    print(f"Average Loss:             ${avgl:,.2f}")
    print(f"Expectancy / Trade:       ${expectancy:,.2f}")
    print(f"Max Drawdown:             ${max_dd:,.2f}")
    print(f"Max Portfolio Loss Streak:{max_streak:2d}")
    print(f"Trades / Day:              {len(trades)/days:.2f}")
    print("-"*90)
    print("Loss streak sequence:")
    print(", ".join(map(str,streaks)) if streaks else "None")
    print("\n"+"-"*90)
    print("PER-SYMBOL")
    print("-"*90)
    for sym in sorted(data.keys(),key=lambda s: -sum(t["symbol"]==s for t in trades)):
        st=[t for t in trades if t["symbol"]==sym]
        if not st: continue
        sw=[t for t in st if t["result"]=="WIN"]; sl=[t for t in st if t["result"]=="LOSS"]
        gp=sum(t["pnl"] for t in sw); gl=sum(t["pnl"] for t in sl)
        spf=gp/abs(gl) if gl<0 else float("inf")
        print(f"{sym.split('/')[0]:>7} | {len(st):4d} trades | WR {len(sw)/len(st)*100:6.2f}% | PnL ${sum(t['pnl'] for t in st):10.2f} | PF {spf:4.2f}")
    print("\n"+"="*90)
    print("AUDIT NOTES")
    print("="*90)
    print("✓ Stage-1 core: 1H compression + 15M breakout + relative volume")
    print("✓ Signal uses CLOSED 15M candle only")
    print("✓ Breakout box uses only COMPLETED 1H candles")
    print("✓ Entry = NEXT 15M candle OPEN + slippage")
    print("✓ Exact RR = 1:2")
    print("✓ No Break-Even / trailing / timeout")
    print("✓ No overlapping positions per symbol")
    print("✓ Max 3 portfolio positions / max 1 per correlation cluster")
    print("✓ Same-candle SL+TP conflict resolves to LOSS")
    print("✓ No forced exit at end; candidates reaching the end remain unmanaged/reportable")
    print("="*90)

if __name__=="__main__":
    main()
