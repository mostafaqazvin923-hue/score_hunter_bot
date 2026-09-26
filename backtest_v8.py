#!/usr/bin/env python3
"""
HUNTER-V154 — TRIPLE-BARRIER META-LABEL ENGINE

Research objective
------------------
Stop inventing another indicator setup.  V154 treats the 1H event as a
candidate and learns, causally, whether that candidate historically reached
+2R before -1R.  Model selection is time-split: train -> validation -> OOS.
The OOS period is never used to choose features, coefficients, or threshold.

Architecture
------------
15m raw XT -> 1H features / 4H context / 1D regime.
Candidate events are deliberately broad (trend continuation OR exhaustion /
reclaim).  A small logistic meta-label model selects only candidates whose
estimated probability of +2R-before-1R clears a threshold chosen on the
validation period.

Hard execution rules
--------------------
- 365-day XT 15m history + 35-day warmup.
- $100 margin, 50x leverage, fixed RR 1:2.
- Signal on closed 1H candle; entry at NEXT 1H open.
- No lookahead / repaint / future leakage.
- No overlapping positions; max 3 portfolio positions.
- No timeout, BE, or trailing stop.
- Same-candle SL+TP => LOSS.
- Open positions at dataset end remain OPEN.
- A candle that closes a prior trade cannot create a new entry.
- Funding is not used because XT public funding history is not 365-day.

Important
---------
The meta-label training label is not an execution timeout.  A training
example is used only when its +2R or -1R barrier is actually resolved before
the relevant time cutoff.  Unresolved examples are excluded from training.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA", "BNB",
    "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

DATA_DIR = Path("data/xt_futures_v154")
INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
MAX_OPEN_POSITIONS = 3

BASE = "https://fapi.xt.com"
LIMIT = 1500
INTERVAL = "15m"
DAYS = 365
WARMUP_DAYS = 35
MIN_ROWS = 30000
MAX_REQUEST_RETRIES = 6

# Time split on the 365-day research window.
TRAIN_DAYS = 180
VALID_DAYS = 90
# Remaining 95 days are OOS.

S = requests.Session()
S.headers.update({"User-Agent": "HUNTER-V154/1.0"})


def resolve_symbols():
    url = f"{BASE}/future/market/v1/public/symbol/list"
    r = S.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = js.get("result", js)
    rows = data.get("symbols", []) if isinstance(data, dict) else data
    out = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("symbol", item.get("name", ""))).upper()
        compact = raw.replace("_", "").replace("-", "").replace("/", "").replace(":", "")
        for asset in SYMBOLS:
            wanted = asset + "USDT"
            if compact == wanted:
                out[asset] = item.get("symbol") or item.get("name")
    return out


def _get_json(url, params):
    last = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict):
                code = js.get("code")
                if code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API code={code}: {js.get('msg', js.get('message', ''))}")
            return js
        except Exception as exc:
            last = exc
            time.sleep(min(2.0, 0.25 * (2 ** attempt)))
    raise RuntimeError(f"XT request failed after {MAX_REQUEST_RETRIES} retries: {last}")


def _parse_kline(js):
    raw = js.get("result", js) if isinstance(js, dict) else js
    if isinstance(raw, dict):
        arr = raw.get("data", raw.get("rows", raw.get("list", [])))
    else:
        arr = raw
    if not isinstance(arr, list):
        return []
    out = []
    for item in arr:
        if isinstance(item, dict):
            t = item.get("t", item.get("timestamp", item.get("time")))
            o = item.get("o", item.get("open"))
            h = item.get("h", item.get("high"))
            l = item.get("l", item.get("low"))
            c = item.get("c", item.get("close"))
            v = item.get("a", item.get("volume", item.get("q", 0)))
        else:
            if len(item) < 6:
                continue
            t, o, h, l, c, v = item[:6]
        try:
            out.append([int(t), float(o), float(h), float(l), float(c), float(v)])
        except (TypeError, ValueError):
            continue
    return out


def fetch_symbol(asset, xt_symbol, days):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int((days + WARMUP_DAYS) * 86400 * 1000)
    interval_ms = 15 * 60 * 1000
    end_ms = (now_ms // interval_ms) * interval_ms - 1
    endpoint = f"{BASE}/future/market/v1/public/q/kline"
    best = []
    for whole_try in range(5):
        rows = []
        cursor = start_ms
        guard = 0
        while cursor < end_ms and guard < 2000:
            guard += 1
            window_end = min(end_ms, cursor + LIMIT * interval_ms - 1)
            params = {"symbol": xt_symbol, "interval": INTERVAL,
                      "startTime": cursor, "endTime": window_end, "limit": LIMIT}
            batch = []
            for attempt in range(6):
                try:
                    batch = _parse_kline(_get_json(endpoint, params))
                    if batch:
                        break
                except Exception:
                    pass
                time.sleep(min(2.0, 0.35 * (attempt + 1)))
            if not batch:
                break
            batch = [x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                break
            rows.extend(batch)
            mx = max(x[0] for x in batch)
            if mx < cursor:
                break
            cursor = mx + 1
            time.sleep(0.08)
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df = df[(df.timestamp >= start_ms) & (df.timestamp <= end_ms)]
            if len(df) > len(best):
                best = df
            if len(df) >= MIN_ROWS:
                span = int(df.timestamp.max() - df.timestamp.min())
                required = int(days * 86400 * 1000 * 0.90)
                if span >= required:
                    df["timestamp"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
                    return df.set_index("timestamp").dropna()
        time.sleep(float(whole_try + 1))
    raise RuntimeError(f"Insufficient XT data for {asset}: got {len(best)} rows; refusing partial backtest")


def ensure_data(data_dir, days):
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping) != len(SYMBOLS):
        raise RuntimeError("XT symbol mapping incomplete")
    out = {}
    for asset in SYMBOLS:
        path = data_dir / f"{asset}_USDT_15m.csv"
        df = fetch_symbol(asset, mapping[asset], days)
        if len(df) < MIN_ROWS:
            raise RuntimeError(f"Validation failed for {asset}: only {len(df)} rows")
        df.to_csv(path)
        gaps = int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
        print(f"{asset:<8} rows={len(df)} gaps={gaps}")
        if gaps != 0:
            raise RuntimeError(f"Unexpected 15m gaps for {asset}: {gaps}")
        out[asset] = df
    return out


def resample(df, rule):
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()


def atr_wilder(x, n=14):
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx_wilder(x, n=14):
    up = x.high.diff()
    dn = -x.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=x.index)
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(all15):
    h1 = {a: resample(df, "1h") for a, df in all15.items()}
    h4 = {a: resample(df, "4h") for a, df in all15.items()}
    d1 = {a: resample(df, "1D") for a, df in all15.items()}
    out = {}
    for asset in SYMBOLS:
        x = h1[asset].copy(); q = h4[asset].copy(); d = d1[asset].copy()
        x["atr"] = atr_wilder(x, 14)
        x["ret3"] = x.close.pct_change(3); x["ret6"] = x.close.pct_change(6)
        x["ret12"] = x.close.pct_change(12); x["ret24"] = x.close.pct_change(24)
        x["ret72"] = x.close.pct_change(72)
        x["rv24"] = x.close.pct_change().rolling(24).std()
        x["rv120"] = x.close.pct_change().rolling(120).std()
        x["rv_ratio"] = x.rv24 / x.rv120.replace(0, np.nan)
        x["vol_z"] = (x.volume - x.volume.rolling(48).mean()) / x.volume.rolling(48).std()
        x["body_atr"] = (x.close - x.open).abs() / x.atr
        x["range_atr"] = (x.high - x.low) / x.atr
        x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ema20_dist"] = (x.close - x.ema20) / x.atr
        x["ema50_dist"] = (x.close - x.ema50) / x.atr
        x["ema20_slope"] = (x.ema20 - x.ema20.shift(6)) / x.atr
        x["high12"] = x.high.rolling(12).max().shift(1)
        x["low12"] = x.low.rolling(12).min().shift(1)
        x["high24"] = x.high.rolling(24).max().shift(1)
        x["low24"] = x.low.rolling(24).min().shift(1)
        x["dist_high24"] = (x.high24 - x.close) / x.atr
        x["dist_low24"] = (x.close - x.low24) / x.atr

        q["atr"] = atr_wilder(q, 14)
        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx_wilder(q, 14)
        q["slope"] = (q.ema50 - q.ema50.shift(6)) / q.atr
        q["trend"] = np.where((q.close > q.ema200) & (q.ema50 > q.ema200), 1,
                       np.where((q.close < q.ema200) & (q.ema50 < q.ema200), -1, 0))
        d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
        d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
        d["slope"] = d.ema50.pct_change(5)
        d["regime"] = np.where((d.close > d.ema200) & (d.ema50 > d.ema200), 1,
                       np.where((d.close < d.ema200) & (d.ema50 < d.ema200), -1, 0))
        x["h4_trend"] = q.trend.reindex(x.index, method="ffill")
        x["h4_adx"] = q.adx.reindex(x.index, method="ffill")
        x["h4_slope"] = q.slope.reindex(x.index, method="ffill")
        x["d1_regime"] = d.regime.reindex(x.index, method="ffill")
        x["d1_slope"] = d.slope.reindex(x.index, method="ffill")
        out[asset] = x
    return out


def add_cross_sectional_ranks(f):
    for ts in sorted(set().union(*(x.index for x in f.values()))):
        vals = []
        for asset, x in f.items():
            if ts in x.index and np.isfinite(x.loc[ts, "ret24"]):
                vals.append((asset, float(x.loc[ts, "ret24"])))
        vals.sort(key=lambda z: z[1]); n = len(vals)
        for rank, (asset, _) in enumerate(vals):
            f[asset].loc[ts, "mom_rank"] = rank / (n - 1) if n > 1 else 0.5
    return f

FEATURES = [
    "ret3", "ret6", "ret12", "ret24", "ret72", "rv_ratio", "vol_z",
    "body_atr", "range_atr", "close_pos", "ema20_dist", "ema50_dist",
    "ema20_slope", "h4_adx", "h4_slope", "d1_slope", "mom_rank",
    "dist_high24", "dist_low24",
]


def candidate_events(f):
    """Broad, causal event set.  No target outcome is consulted here."""
    events = []
    for asset, x in f.items():
        for i in range(300, len(x) - 1):
            r = x.iloc[i]; p = x.iloc[i - 1]
            if not all(np.isfinite(r.get(k, np.nan)) for k in FEATURES + ["atr", "h4_trend", "d1_regime"]):
                continue
            long_trend = r.d1_regime == 1 and r.h4_trend == 1
            short_trend = r.d1_regime == -1 and r.h4_trend == -1
            long_break = r.close > r.high24 and p.close <= p.high24
            short_break = r.close < r.low24 and p.close >= p.low24
            long_reclaim = p.close <= p.ema20 and r.close > r.ema20 and p.low <= p.ema20
            short_reclaim = p.close >= p.ema20 and r.close < r.ema20 and p.high >= p.ema20
            long_exhaust = r.ema20_dist <= -1.5 and r.close_pos >= 0.60
            short_exhaust = r.ema20_dist >= 1.5 and r.close_pos <= 0.40
            long_event = (long_trend and (long_break or long_reclaim)) or (long_exhaust and r.d1_regime >= 0)
            short_event = (short_trend and (short_break or short_reclaim)) or (short_exhaust and r.d1_regime <= 0)
            if long_event and not short_event:
                events.append({"asset": asset, "ts": x.index[i], "side": "LONG", "atr": float(r.atr)})
            elif short_event and not long_event:
                events.append({"asset": asset, "ts": x.index[i], "side": "SHORT", "atr": float(r.atr)})
    return events


def event_features(f, ev):
    r = f[ev["asset"]].loc[ev["ts"]]
    side = 1.0 if ev["side"] == "LONG" else -1.0
    vals = []
    for k in FEATURES:
        v = float(r[k])
        if k in ("ret3", "ret6", "ret12", "ret24", "ret72", "ema20_dist", "ema50_dist", "ema20_slope", "h4_slope", "d1_slope", "dist_high24", "dist_low24"):
            v *= side
        vals.append(v)
    return np.asarray(vals, dtype=float)


def resolve_label(f, ev, cutoff=None, atr_mult=1.50):
    """Return (label, resolution_ts) only when a barrier is actually resolved."""
    x = f[ev["asset"]]; idx = x.index; pos = idx.get_loc(ev["ts"])
    if pos >= len(idx) - 1:
        return None, None
    entry_ts = idx[pos + 1]
    entry = float(x.iloc[pos + 1].open)
    if not np.isfinite(entry) or entry <= 0:
        return None, None
    risk = atr_mult * ev["atr"]
    if not np.isfinite(risk) or risk <= 0 or risk / entry > 0.08:
        return None, None
    if ev["side"] == "LONG":
        entry *= (1 + SLIPPAGE); sl = entry - risk; tp = entry + RR * risk
    else:
        entry *= (1 - SLIPPAGE); sl = entry + risk; tp = entry - RR * risk
    for j in range(pos + 1, len(idx)):
        ts = idx[j]; row = x.iloc[j]
        hit_sl = row.low <= sl if ev["side"] == "LONG" else row.high >= sl
        hit_tp = row.high >= tp if ev["side"] == "LONG" else row.low <= tp
        if hit_sl or hit_tp:
            # Conservative same-candle rule: loss.
            label = 0 if hit_sl else 1
            if cutoff is not None and ts > cutoff:
                return None, None
            return label, ts
    return None, None


def zfit(X):
    mu = np.nanmedian(X, axis=0); scale = np.nanmedian(np.abs(X - mu), axis=0) * 1.4826
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    Xz = np.clip((X - mu) / scale, -8, 8)
    return Xz, mu, scale


def zapply(X, mu, scale):
    return np.clip((X - mu) / scale, -8, 8)


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, l2=1.0, steps=500, lr=0.05):
    # Pure NumPy logistic regression: no extra package dependency in Actions.
    Xz, mu, scale = zfit(X)
    w = np.zeros(Xz.shape[1], dtype=float); b = 0.0
    pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    wp = len(y) / (2 * pos); wn = len(y) / (2 * neg)
    sample_w = np.where(y == 1, wp, wn)
    for _ in range(steps):
        p = sigmoid(Xz @ w + b)
        err = (p - y) * sample_w
        gw = (Xz.T @ err) / len(y) + l2 * w / len(y)
        gb = float(err.mean())
        w -= lr * gw; b -= lr * gb
    return {"w": w, "b": b, "mu": mu, "scale": scale}


def predict(model, X):
    return sigmoid(zapply(X, model["mu"], model["scale"]) @ model["w"] + model["b"])


def build_label_dataset(f, events, cutoff):
    X=[]; y=[]; meta=[]
    for ev in events:
        if ev["ts"] >= cutoff:
            continue
        label, resolved = resolve_label(f, ev, cutoff=cutoff)
        if label is None:
            continue
        X.append(event_features(f, ev)); y.append(label); meta.append((ev, resolved))
    if not X:
        return np.empty((0, len(FEATURES))), np.empty(0), []
    return np.asarray(X), np.asarray(y, dtype=float), meta


def threshold_score(y, p, thresholds):
    best = None
    for th in thresholds:
        take = p >= th
        if take.sum() < 50:
            continue
        wr = 100.0 * y[take].mean()
        # Approximate expectancy under exact 1:2 before fees.
        exp_r = (wr / 100.0) * 2.0 - (1.0 - wr / 100.0)
        key = (exp_r, int(take.sum()))
        if best is None or key > best[0]:
            best = (key, float(th), int(take.sum()), wr)
    return best


def portfolio_backtest(f, events, models_by_fold, threshold, oos_start):
    by_ts = {}
    for ev in events:
        if ev["ts"] >= oos_start:
            by_ts.setdefault(ev["ts"], []).append(ev)
    all_ts = sorted(set().union(*(x.index for x in f.values())))
    positions=[]; trades=[]; equity=INITIAL_EQUITY; peak=equity; max_dd=0.0
    closed_dates = {}
    for ts in all_ts:
        # Select model based only on the time block already trained.
        model = None
        for start, end, m in models_by_fold:
            if start <= ts < end:
                model = m; break
        closed_any=False
        for p in positions[:]:
            x=f[p["asset"]]
            if ts not in x.index or ts < p["entry_ts"]: continue
            row=x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"]=="LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"]=="LONG" else row.low <= p["tp"]
            if not (hit_sl or hit_tp): continue
            outcome="LOSS" if hit_sl else "WIN"
            exit_price=p["sl"] if hit_sl else p["tp"]
            price_ret=((exit_price-p["entry"])/p["entry"] if p["side"]=="LONG" else (p["entry"]-exit_price)/p["entry"])
            gross=price_ret*TRADE_MARGIN*LEVERAGE; fees=TRADE_MARGIN*LEVERAGE*FEE_RATE*2
            pnl=gross-fees; equity+=pnl; peak=max(peak,equity); max_dd=min(max_dd,equity-peak)
            trades.append({**p,"exit_ts":ts,"outcome":outcome,"pnl":pnl}); positions.remove(p); closed_any=True
        if closed_any: continue
        if model is None or len(positions)>=MAX_OPEN_POSITIONS: continue
        cand=[]
        for ev in by_ts.get(ts,[]):
            if any(p["asset"]==ev["asset"] for p in positions): continue
            prob=float(predict(model, event_features(f,ev)[None,:])[0])
            if prob>=threshold: cand.append((prob,ev))
        cand.sort(key=lambda z:z[0],reverse=True)
        for prob,ev in cand:
            if len(positions)>=MAX_OPEN_POSITIONS: break
            if any(p["asset"]==ev["asset"] for p in positions): continue
            x=f[ev["asset"]]; idx=x.index; pos=idx.get_loc(ts)
            if pos>=len(idx)-1: continue
            entry=float(x.iloc[pos+1].open)*(1+SLIPPAGE if ev["side"]=="LONG" else 1-SLIPPAGE)
            risk=1.50*ev["atr"]
            if not np.isfinite(risk) or risk<=0 or risk/entry>0.08: continue
            sl,tp=(entry-risk,entry+RR*risk) if ev["side"]=="LONG" else (entry+risk,entry-RR*risk)
            positions.append({"asset":ev["asset"],"side":ev["side"],"signal_ts":ts,"entry_ts":idx[pos+1],"entry":entry,"sl":sl,"tp":tp,"prob":prob})
    wins=sum(t["outcome"]=="WIN" for t in trades); losses=len(trades)-wins
    eq=INITIAL_EQUITY; peak=eq; dd=0; streak=mx=0
    for t in sorted(trades,key=lambda z:z["exit_ts"]):
        eq+=t["pnl"]; peak=max(peak,eq); dd=min(dd,eq-peak); streak=streak+1 if t["outcome"]=="LOSS" else 0; mx=max(mx,streak)
    gw=sum(t["pnl"] for t in trades if t["pnl"]>0); gl=-sum(t["pnl"] for t in trades if t["pnl"]<0)
    return {"trades":len(trades),"open":len(positions),"wr":100*wins/len(trades) if trades else 0,"pf":gw/gl if gl else 0,"pnl":sum(t["pnl"] for t in trades),"dd":min(dd,max_dd),"streak":mx,"tday":len(trades)/DAYS}, trades


def main():
    print("="*96); print("HUNTER-V154 — TRIPLE-BARRIER META-LABEL ENGINE"); print("15m raw -> 1H candidates / 4H context / 1D regime | causal train/validation/OOS"); print("="*96)
    all15=ensure_data(DATA_DIR,DAYS)
    f=add_cross_sectional_ranks(build_features(all15))
    events=candidate_events(f)
    print(f"Candidate events: {len(events)}")
    # Use the actual common research window derived from the 1H data.
    min_ts=max(x.index.min() for x in f.values()) + pd.Timedelta(days=WARMUP_DAYS)
    max_ts=min(x.index.max() for x in f.values())
    total=max_ts-min_ts
    train_end=min_ts+pd.Timedelta(days=TRAIN_DAYS)
    valid_end=train_end+pd.Timedelta(days=VALID_DAYS)
    print(f"Split: train < {train_end} | validation < {valid_end} | OOS >= {valid_end}")

    # Training labels are resolved strictly before the cutoff.
    Xtr,ytr,_=build_label_dataset(f,events,train_end)
    print(f"Resolved TRAIN labels: {len(ytr)} | win labels={int(ytr.sum()) if len(ytr) else 0} | WR={100*ytr.mean() if len(ytr) else 0:.2f}%")
    if len(ytr)<500 or ytr.mean()<0.20 or ytr.mean()>0.80:
        raise RuntimeError("Insufficient or degenerate training labels; refusing to run a meaningless model.")
    model=fit_logistic(Xtr,ytr,l2=1.0,steps=700,lr=0.04)

    # Validation labels are also resolved before valid_end, but model never sees them.
    Xv,yv,_=build_label_dataset(f,[e for e in events if train_end<=e["ts"]<valid_end],valid_end)
    if len(yv)<200:
        raise RuntimeError(f"Insufficient validation labels: {len(yv)}")
    pv=predict(model,Xv)
    choice=threshold_score(yv,pv,np.arange(0.45,0.81,0.025))
    if choice is None:
        raise RuntimeError("No validation threshold produced >=50 candidates; refusing to overfit OOS.")
    _,threshold,nv,wrv=choice
    print(f"Validation: labels={len(yv)} selected={nv} WR={wrv:.2f}% threshold={threshold:.3f}")

    # OOS is evaluated once, untouched by threshold selection.
    # Retrain once using train+validation labels resolved strictly before OOS.
    pre_oos_events = [e for e in events if e["ts"] < valid_end]
    Xtv, ytv, _ = build_label_dataset(f, pre_oos_events, valid_end)
    print(f"Resolved TRAIN+VALID labels for final pre-OOS fit: {len(ytv)} | WR={100*ytv.mean() if len(ytv) else 0:.2f}%")
    if len(ytv) < 800:
        raise RuntimeError(f"Insufficient pre-OOS labels for final fit: {len(ytv)}")
    final_model = fit_logistic(Xtv, ytv, l2=1.0, steps=700, lr=0.04)

    oos_events = [e for e in events if e["ts"] >= valid_end]
    oos_start = valid_end
    result, trades = portfolio_backtest(
        f, oos_events, [(oos_start, max_ts + pd.Timedelta(hours=1), final_model)],
        threshold, oos_start
    )

    print("\nOOS RESULT — single untouched final evaluation")
    print(
        f"trades={result['trades']} open={result['open']} WR={result['wr']:.2f}% "
        f"PF={result['pf']:.3f} PnL=${result['pnl']:,.2f} DD=${result['dd']:,.2f} "
        f"streak={result['streak']} t/day={result['tday']:.2f}"
    )
    eligible = (
        result["trades"] >= 100 and result["wr"] >= 50.0 and
        result["streak"] <= 4 and result["pf"] > 1.0 and result["pnl"] > 0
    )
    print(f"OOS_ELIGIBLE: {eligible}")
    print("\n" + "=" * 96)
    print("MEASURABLE_EDGE_PRESENT: " + str(eligible))
    if not eligible:
        print("V154 does not meet the hard OOS target; no parameter tuning is performed on OOS.")
    else:
        print("V154 met all hard OOS conditions on the untouched final period.")
    print("=" * 96)


if __name__ == "__main__":
    main()
