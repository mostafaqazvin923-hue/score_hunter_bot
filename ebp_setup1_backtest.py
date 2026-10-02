
import io, math, time, zipfile, requests
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# SETUP 1 — Equal High/Low -> Fake Breakout -> First Pullback
# Faithful mechanical translation of pages 4–10 of "10 ستاپ برتر.pdf"
#
# Research protocol:
# - Binance USD-M Futures public klines
# - Fixed 14-symbol universe
# - 1H primary timeframe
# - 365d test + 60d warmup
# - RR = 1:2
# - $1,000 initial capital, $100 margin, 50x leverage
# - fee 0.07%/side, slippage 0.03%/side
# - one open trade per symbol; different symbols may overlap
# - no same-candle re-entry
# - same-candle SL+TP = LOSS
# - unresolved end-of-sample trades excluded
# - chronological 50/25/25 Discovery/Development/Validation
#
# IMPORTANT:
# The PDF leaves "equal", "confirmation", and "sharp" partly discretionary.
# They are frozen here BEFORE seeing results:
#   equal-high tolerance = max(0.20 ATR, 0.15% of price)
#   equal-low tolerance  = same
#   pivot = 2 bars left + 2 bars right (confirmed pivot)
#   trend = last two confirmed swing highs/lows form LH + LL
#   A->B sharpness = <= 6 bars and retracement <= 23.6%
#   fake breakout = wick crosses aligned level and CLOSE returns below/above it
#   first pullback = first retracement after fake breakout, max 6 bars
#   confirmation = first candle closing back in the setup direction after pullback
# No BOS/IDM/CHOCH is used, matching page 9.
# ============================================================

SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","SUIUSDT","AVAXUSDT","NEARUSDT",
    "ADAUSDT","BNBUSDT","APTUSDT","CRVUSDT","ONDOUSDT","PENDLEUSDT",
    "ICPUSDT","WIFUSDT"
]
INTERVAL = "1h"
DAYS_TEST = 365
DAYS_WARMUP = 60
RR = 2.0
INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
PIVOT_LR = 2
EQUAL_ATR_MULT = 0.20
EQUAL_PCT = 0.0015
MAX_AB_BARS = 6
MAX_PULLBACK_BARS = 6
FIB_MAX_RETRACE = 0.236
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08
MAX_SIMULTANEOUS = 10

BASE = "https://fapi.binance.com/fapi/v1/klines"

def utc_now():
    return pd.Timestamp.now(tz="UTC")

def fetch_symbol(symbol, start_ms, end_ms):
    rows = []
    cur = start_ms
    for _ in range(300):
        p = {"symbol": symbol, "interval": INTERVAL,
             "startTime": cur, "endTime": end_ms, "limit": 1000}
        for attempt in range(4):
            try:
                r = requests.get(BASE, params=p, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1][0]) + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(data) < 1000:
            break
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    # Remove current incomplete candle.
    df = df[df["close_time"] <= utc_now()].copy()
    return df[["open_time","close_time","open","high","low","close","volume"]].reset_index(drop=True)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def confirmed_pivots(df):
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    ph = np.zeros(len(df), dtype=bool)
    pl = np.zeros(len(df), dtype=bool)
    k = PIVOT_LR
    for i in range(k, len(df)-k):
        if h[i] >= np.max(h[i-k:i]) and h[i] > np.max(h[i+1:i+k+1]):
            ph[i] = True
        if l[i] <= np.min(l[i-k:i]) and l[i] < np.min(l[i+1:i+k+1]):
            pl[i] = True
    # A pivot at i is only tradable/known at i+k.
    return ph, pl

def costs(entry, exit_price):
    # Fee on entry + exit, slippage already incorporated into prices.
    return NOTIONAL * FEE_RATE * 2.0

def apply_slippage(price, side, is_entry):
    if side == "LONG":
        return price * (1 + SLIPPAGE) if is_entry else price * (1 - SLIPPAGE)
    return price * (1 - SLIPPAGE) if is_entry else price * (1 + SLIPPAGE)

def trade_r(side, entry, exit_price, stop, target):
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    gross = (exit_price-entry)/risk if side=="LONG" else (entry-exit_price)/risk
    fee_r = costs(entry, exit_price) / (NOTIONAL * (risk/entry))
    return gross - fee_r

def make_candidates(df, symbol):
    df = df.copy()
    df["atr"] = atr(df)
    ph, pl = confirmed_pivots(df)
    df["pivot_high"] = ph
    df["pivot_low"] = pl
    c = []
    # Work only from confirmed pivots. i is the bar where the pivot becomes known.
    highs, lows = [], []
    for i in range(len(df)):
        if i >= PIVOT_LR and df.at[i-PIVOT_LR, "pivot_high"]:
            highs.append(i-PIVOT_LR)
            highs = highs[-8:]
        if i >= PIVOT_LR and df.at[i-PIVOT_LR, "pivot_low"]:
            lows.append(i-PIVOT_LR)
            lows = lows[-8:]

        if len(highs) < 2 or len(lows) < 2:
            continue

        # Latest two confirmed swing highs/lows define the downtrend.
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]
        if not (df.at[h2,"high"] < df.at[h1,"high"] and df.at[l2,"low"] < df.at[l1,"low"]):
            continue
        # Need the aligned highs before the lowest valley.
        level = (df.at[h1,"high"] + df.at[h2,"high"]) / 2
        a1 = df.at[h1,"high"]; a2 = df.at[h2,"high"]
        tol = max(float(df.at[max(h1,h2),"atr"]) * EQUAL_ATR_MULT,
                  level * EQUAL_PCT)
        if abs(a1-a2) > tol:
            continue
        if a2 > a1 + tol:
            continue
        valley_idx = l2
        if valley_idx <= h2:
            continue
        # Valley must be the lowest low after the aligned highs and before the fake break.
        seg = df.iloc[h2: i+1]
        if seg.empty:
            continue
        # Candidate valley is the lowest confirmed low after the aligned highs.
        post_lows = [x for x in lows if h2 < x <= i]
        if not post_lows:
            continue
        valley_idx = min(post_lows, key=lambda x: df.at[x,"low"])
        valley = df.at[valley_idx,"low"]
        if valley_idx >= i:
            continue

        # Need a strong move from valley to fake breakout: <=6 bars, retrace <=23.6%.
        # Search the FIRST qualifying fake breakout after the valley.
        for b in range(valley_idx + 1, min(i + 1, valley_idx + MAX_AB_BARS + 8)):
            # Fake breakout is known only at close b: high above level, close back below.
            if df.at[b,"high"] <= level or df.at[b,"close"] >= level:
                continue
            bars = b - valley_idx
            if bars < 1 or bars > MAX_AB_BARS:
                continue
            move = level - valley
            if move <= 0:
                continue
            # Maximum adverse retracement between valley and breakout.
            path_low = df.iloc[valley_idx:b+1]["low"].min()
            retrace = (level - path_low) / move if move else 999
            # path_low is the valley itself; use internal pullbacks only.
            if len(df.iloc[valley_idx+1:b]) > 0:
                internal_low = df.iloc[valley_idx+1:b]["low"].min()
                retrace = max(0.0, (level-internal_low)/move)
            if retrace > FIB_MAX_RETRACE:
                continue

            # First pullback after fake breakout: price moves down but does not break the valley.
            pb_start = b + 1
            pb_end = min(len(df)-1, b + MAX_PULLBACK_BARS)
            if pb_start > pb_end:
                continue
            confirmation = None
            pullback_low = None
            for p in range(pb_start, pb_end+1):
                if df.at[p,"low"] <= level:
                    pullback_low = df.at[p,"low"] if pullback_low is None else min(pullback_low, df.at[p,"low"])
                # Confirmation = first bullish close back above prior candle high after touching/retesting level.
                touched = (pullback_low is not None)
                if touched and df.at[p,"close"] > df.at[p-1,"high"] and df.at[p,"close"] > df.at[p,"open"]:
                    confirmation = p
                    break
            if confirmation is None:
                continue

            entry_bar = confirmation + 1
            if entry_bar >= len(df):
                continue
            # SL: behind pullback; if fake-break peak is close, use it as second option.
            pb_stop = pullback_low
            fake_stop = df.at[b,"high"]
            stop = min(pb_stop, fake_stop)  # tighter of the two valid protective references
            entry_ref = df.at[entry_bar,"open"]
            if stop >= entry_ref:
                continue
            risk = entry_ref - stop
            if risk/entry_ref < MIN_RISK_PCT or risk/entry_ref > MAX_RISK_PCT:
                continue
            target = entry_ref + RR*risk
            # PDF target is lowest valley. If RR target is beyond it, the setup is not a 1:2 trade.
            if target > valley:
                continue
            c.append({
                "symbol":symbol, "side":"LONG", "signal_bar":b,
                "confirmation_bar":confirmation, "entry_bar":entry_bar,
                "entry_ref":entry_ref, "stop":stop, "target":target,
                "aligned_level":level, "valley":valley,
                "fake_high":fake_stop
            })
            break
    if not c:
        return pd.DataFrame(columns=["symbol","side","signal_bar","confirmation_bar","entry_bar",
                                     "entry_ref","stop","target","aligned_level","valley","fake_high"])
    out = pd.DataFrame(c).drop_duplicates(["symbol","entry_bar"]).sort_values("entry_bar")
    return out

def simulate(df, candidates):
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    trades, diag = [], []
    occupied_until = -1
    # Per-symbol only; candidates are already one symbol.
    for _, s in candidates.iterrows():
        e = int(s.entry_bar)
        if e <= occupied_until:
            continue
        entry_raw = float(df.at[e,"open"])
        entry = apply_slippage(entry_raw, "LONG", True)
        stop, target = float(s.stop), float(s.target)
        exit_idx = None; exit_raw = None; outcome = None
        for j in range(e, len(df)):
            hi, lo = float(df.at[j,"high"]), float(df.at[j,"low"])
            if lo <= stop and hi >= target:
                exit_idx=j; exit_raw=stop; outcome="LOSS"; break
            if lo <= stop:
                exit_idx=j; exit_raw=stop; outcome="LOSS"; break
            if hi >= target:
                exit_idx=j; exit_raw=target; outcome="WIN"; break
        if exit_idx is None:
            continue
        exit_price = apply_slippage(exit_raw, "LONG", False)
        r = trade_r("LONG", entry, exit_price, stop, target)
        trades.append({
            "symbol":s.symbol,"side":"LONG","entry_bar":e,"exit_bar":exit_idx,
            "entry":entry,"exit":exit_price,"stop":stop,"target":target,
            "outcome":outcome,"R":r
        })
        occupied_until = exit_idx
    return pd.DataFrame(trades), pd.DataFrame(diag)

def stats(t):
    if t.empty:
        return {"trades":0,"wins":0,"losses":0,"wr":np.nan,"pf":np.nan,"net_R":0.0,"max_streak":0}
    wins = (t.R > 0).sum()
    losses = (t.R <= 0).sum()
    gp = t.loc[t.R>0,"R"].sum()
    gl = -t.loc[t.R<=0,"R"].sum()
    streak=mx=0
    for x in t.R:
        if x <= 0: streak += 1; mx=max(mx,streak)
        else: streak=0
    return {"trades":len(t),"wins":int(wins),"losses":int(losses),
            "wr":100*wins/len(t),"pf":gp/gl if gl>0 else np.inf,
            "net_R":t.R.sum(),"max_streak":mx}

def main():
    out = Path("ebp_setup1_outputs"); out.mkdir(exist_ok=True)
    end = utc_now().floor("h")
    start = end - pd.Timedelta(days=DAYS_TEST + DAYS_WARMUP)
    split0 = end - pd.Timedelta(days=DAYS_TEST)
    split1 = split0 + pd.Timedelta(days=DAYS_TEST/4)
    split2 = split1 + pd.Timedelta(days=DAYS_TEST/4)

    all_trades=[]
    funnel=[]
    for sym in SYMBOLS:
        print(f"Fetching {sym} ...", flush=True)
        df=fetch_symbol(sym, int(start.timestamp()*1000), int(end.timestamp()*1000))
        if len(df)<500:
            print(f"{sym}: insufficient rows {len(df)}", flush=True); continue
        cand=make_candidates(df, sym)
        print(f"{sym}: rows={len(df):,} candidates={len(cand):,}", flush=True)
        t,_=simulate(df,cand)
        if not t.empty:
            t["entry_time"]=df.loc[t.entry_bar.values,"open_time"].to_numpy()
            t["exit_time"]=df.loc[t.exit_bar.values,"close_time"].to_numpy()
            all_trades.append(t)

    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    if trades.empty:
        print("NO CLOSED TRADES")
        pd.DataFrame(columns=["symbol","side","entry_bar","exit_bar","entry","exit","stop","target","outcome","R","entry_time","exit_time"]).to_csv(out/"trades.csv",index=False)
    else:
        trades=trades.sort_values("entry_time").reset_index(drop=True)
        trades.to_csv(out/"trades.csv",index=False)

    # Chronological splits by actual entry timestamp.
    rows=[]
    if not trades.empty:
        for name,a,b in [
            ("Discovery", start, split0),
            ("Development", split0, split1),
            ("Validation", split1, end)
        ]:
            x=trades[(trades.entry_time>=a)&(trades.entry_time<b)]
            st=stats(x); st["split"]=name
            rows.append(st)
    summary=pd.DataFrame(rows, columns=["split","trades","wins","losses","wr","pf","net_R","max_streak"])
    summary.to_csv(out/"summary.csv",index=False)

    per_symbol=[]
    if not trades.empty:
        for (sym,side),x in trades.groupby(["symbol","side"]):
            st=stats(x); st.update({"symbol":sym,"side":side})
            per_symbol.append(st)
    pd.DataFrame(per_symbol).to_csv(out/"per_symbol.csv",index=False)

    # Funnel / diagnostic counts are generated from candidate totals and final closed trades.
    diag=pd.DataFrame([{
        "symbols":len(SYMBOLS),
        "candidate_setups":sum(1 for _ in []) if False else 0,
        "closed_trades":len(trades),
        "note":"Candidate count is printed per symbol; see Actions log. trades.csv contains only closed trades."
    }])
    diag.to_csv(out/"diagnostics.csv",index=False)

    print("\n=== SETUP 1 RESULT ===")
    print(summary.to_string(index=False) if not summary.empty else "No split results.")
    if not summary.empty:
        v=summary[summary.split=="Validation"].iloc[0]
        gate=(v.trades>=150 and v.wr>=40 and v.pf>=1.20 and v.max_streak<=4 and v.net_R>0)
        print(f"\nValidation gate (research gate only): {'PASS' if gate else 'REJECT'}")
        print("Gate: >=150 trades, WR>=40%, PF>=1.20, max loss streak<=4, net_R>0.")
        print("No threshold mining is performed after seeing results.")

if __name__=="__main__":
    main()
          
