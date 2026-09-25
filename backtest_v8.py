#!/usr/bin/env python3
"""
HUNTER-V149 — FLOW / POSITIONING PROXY RESEARCH
1D regime + 4H trend context + 1H pullback/reclaim/momentum resumption.
Raw source: XT Futures 15m -> causal 1H/4H/1D.

Purpose
-------
V144-V148 did not produce the required edge. V149 therefore switches to a
different research family: FLOW / POSITIONING PROXY. Because the XT kline
history does not contain historical whale wallets or open-interest snapshots,
this version does NOT pretend to reconstruct true smart-money positions.
Instead it tests a causal market-participant-pressure proxy built from OHLCV:
volume-weighted candle pressure, cumulative flow, price/flow divergence,
absorption, and volatility expansion. Higher timeframes define regime; 1H
provides the trigger.

Signal concept
--------------
LONG (mirror for SHORT):
  1) 1D directional regime is established and EMA50 slope agrees.
  2) 4H trend agrees, with EMA structure and ADX strength.
  3) 1H price sweeps a prior rolling liquidity low and closes back above it.
  4) The same 1H candle shows bullish displacement/body strength.
  5) Volume confirms the event.
  6) Price is not excessively extended from 1H EMA20.

This is a causal liquidity-sweep -> reclaim -> displacement model, not a
future-pivot/retest model. Entry is always the NEXT 1H candle open.

Integrity rules
---------------
- No lookahead / future leak / repainting.
- 15m is only the raw source; signals are 1H.
- 4H and 1D values are right-labeled and forward-filled only after their
  completed candle timestamp.
- No overlapping positions; max 3 portfolio positions.
- No new entry on the same timestamp a prior trade closes.
- Fixed RR 1:2.
- No timeout / max-bars exit.
- Same-candle SL+TP => LOSS.
- Positions still open at dataset end remain OPEN.
- Minimum trade count is used for research ranking, not to manufacture wins.

V148 optimization
------------------
A compact causal grid varies the minimum score and a few structural
thresholds. Features are computed once. The score is never used to alter
outcomes; it only defines which causal setups are admitted.

IMPORTANT: an in-sample winner is NOT considered validated. A qualifying
candidate must be walk-forward/OOS tested before any live consideration.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = ["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
DATA_DIR = Path("data/xt_futures_v141")
OUT_DIR = DATA_DIR / "results"

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

S = requests.Session()
S.headers.update({"User-Agent": "HUNTER-backtest/148"})


def resolve_symbols():
    u = f"{BASE}/future/market/v1/public/symbol/list"
    r = S.get(u, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = js.get("result", js)
    rows = data.get("symbols", []) if isinstance(data, dict) else data
    out = {}
    for x in rows:
        if not isinstance(x, dict):
            continue
        raw = str(x.get("symbol", x.get("name", ""))).upper()
        compact = raw.replace("_", "").replace("-", "").replace("/", "").replace(":", "")
        for a in SYMBOLS:
            aliases = {a + "USDT", a + "_USDT", a + "/USDT", a + "/USDT:USDT"}
            norm = {z.replace("_", "").replace("-", "").replace("/", "").replace(":", "") for z in aliases}
            if compact in norm:
                out[a] = x.get("symbol") or x.get("name")
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
                return_code = js.get("returnCode")
                if code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API code={code}: {js.get('msg', js.get('message', ''))}")
                if return_code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API returnCode={return_code}: {js.get('msgInfo', js.get('msg', js.get('message', '')))}")
            return js
        except Exception as e:
            last = e
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
    for x in arr:
        if isinstance(x, dict):
            t = x.get("t", x.get("timestamp", x.get("time")))
            o = x.get("o", x.get("open")); h = x.get("h", x.get("high"))
            l = x.get("l", x.get("low")); c = x.get("c", x.get("close"))
            v = x.get("a", x.get("volume", x.get("q", 0)))
        else:
            if len(x) < 6:
                continue
            t, o, h, l, c, v = x[:6]
        try:
            out.append([int(t), float(o), float(h), float(l), float(c), float(v)])
        except (TypeError, ValueError):
            continue
    return out


def fetch_symbol(asset, xt_symbol, days):
    """
    Robust XT 15m collector.

    Important: XT documents both startTime and endTime for /q/kline.  We
    paginate FORWARD with startTime+endTime windows instead of relying on a
    single moving endTime.  This avoids the partial-page behaviour that caused
    the previous run to receive only 15 BTC candles.
    """
    now_ms=int(time.time()*1000)
    start_ms=now_ms-int((days+WARMUP_DAYS)*86400*1000)
    # Never use a still-forming candle.
    interval_ms=15*60*1000
    end_ms=(now_ms//interval_ms)*interval_ms-1

    best=[]
    endpoint=f"{BASE}/future/market/v1/public/q/kline"

    for whole_try in range(5):
        rows=[]
        cursor=start_ms
        guard=0
        failed=False

        while cursor < end_ms and guard < 2000:
            guard += 1
            # 1500 x 15m = 15.625 days. Keep a tiny overlap and dedupe later.
            window_end=min(end_ms, cursor + LIMIT*interval_ms - 1)
            params={
                "symbol":xt_symbol,
                "interval":INTERVAL,
                "startTime":cursor,
                "endTime":window_end,
                "limit":LIMIT,
            }

            batch=[]
            last_err=None
            for attempt in range(6):
                try:
                    batch=_parse_kline(_get_json(endpoint,params))
                    if batch:
                        break
                except Exception as e:
                    last_err=e
                time.sleep(min(2.0,0.35*(attempt+1)))

            if not batch:
                failed=True
                break

            # Keep only valid candles inside the requested window.
            batch=[x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                failed=True
                break

            rows.extend(batch)
            mn=min(x[0] for x in batch)
            mx=max(x[0] for x in batch)

            # Hard progress check. Never loop on the same XT page.
            if mx < cursor:
                failed=True
                break

            # Normal case: move just past the last candle received.
            next_cursor=mx+1
            if next_cursor <= cursor:
                failed=True
                break
            cursor=next_cursor

            # If XT returned fewer than LIMIT rows, the next forward request
            # is still valid; do not treat a short page as a fatal error.
            time.sleep(0.08)

        if rows:
            df=pd.DataFrame(
                rows,
                columns=["timestamp","open","high","low","close","volume"]
            )
            df=df.drop_duplicates("timestamp").sort_values("timestamp")
            df=df[(df.timestamp>=start_ms)&(df.timestamp<=end_ms)]

            if len(df)>len(best):
                best=df

            if len(df)>=MIN_ROWS:
                # Require broad time coverage, not merely row count.
                span=int(df.timestamp.max()-df.timestamp.min())
                required_span=int(days*86400*1000*0.90)
                if span >= required_span:
                    df["timestamp"]=pd.to_datetime(df.timestamp,unit="ms",utc=True)
                    return df.set_index("timestamp").dropna()

        time.sleep(1.0*(whole_try+1))

    got=len(best)
    if got:
        first=pd.to_datetime(int(best.timestamp.min()),unit="ms",utc=True)
        last=pd.to_datetime(int(best.timestamp.max()),unit="ms",utc=True)
        detail=f"; range={first} -> {last}"
    else:
        detail=""
    raise RuntimeError(
        f"Insufficient XT data for {asset}: got {got} rows{detail}; "
        f"refusing to run a partial backtest"
    )

def ensure_data(data_dir,days):
    data_dir.mkdir(parents=True,exist_ok=True); mapping=resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping)!=len(SYMBOLS): raise RuntimeError("XT symbol mapping incomplete")
    out={}
    for a in SYMBOLS:
        path=data_dir/f"{a}_USDT_15m.csv"
        try:
            df=fetch_symbol(a,mapping[a],days)
            if len(df)<MIN_ROWS:
                raise RuntimeError(f"Validation failed for {a}: only {len(df)} rows")
            df.to_csv(path)
            out[a]=df
            gaps=int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
            print(f"{a:<8} rows={len(df)} gaps={gaps}")
        except Exception as e:
            raise RuntimeError(f"XT download failed for {a}: {e}") from e
    return out

def resample(df, rule):
    return df.resample(rule, label="right", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()


def atr14(x):
    tr = pd.concat([(x.high-x.low),
                    (x.high-x.close.shift()).abs(),
                    (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def adx14(x):
    up = x.high.diff()
    dn = -x.low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([(x.high-x.low), (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def build_features(all15):
    """Build causal OHLCV flow/positioning proxies once."""
    f = {}
    h1 = {a: resample(df, "1h") for a, df in all15.items()}
    h4 = {a: resample(df, "4h") for a, df in all15.items()}
    d1 = {a: resample(df, "1D") for a, df in all15.items()}

    for a in SYMBOLS:
        x = h1[a].copy()
        q = h4[a].copy()
        d = d1[a].copy()

        # 1H price/volume pressure proxies. These are causal and use only the
        # completed candle at t; no true whale/OI information is fabricated.
        rng = (x.high - x.low).replace(0, np.nan)
        clv = ((x.close - x.low) - (x.high - x.close)) / rng
        body_frac = (x.close - x.open).abs() / rng
        x["atr"] = atr14(x)
        x["range_atr"] = (x.high - x.low) / x.atr
        x["body_atr"] = (x.close - x.open).abs() / x.atr
        x["close_pos"] = (x.close - x.low) / rng
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ret6"] = x.close.pct_change(6)
        x["ret24"] = x.close.pct_change(24)
        x["ret72"] = x.close.pct_change(72)
        x["vol_z"] = (x.volume - x.volume.rolling(48, min_periods=24).mean()) / x.volume.rolling(48, min_periods=24).std()

        # Signed-volume proxy: close location within the candle multiplied by
        # volume. Positive = demand pressure, negative = supply pressure.
        x["flow"] = clv * x.volume
        x["flow_z"] = (x.flow - x.flow.rolling(48, min_periods=24).mean()) / x.flow.rolling(48, min_periods=24).std()
        x["flow24"] = x.flow.rolling(24, min_periods=24).sum() / x.volume.rolling(24, min_periods=24).sum().replace(0, np.nan)
        x["flow72"] = x.flow.rolling(72, min_periods=72).sum() / x.volume.rolling(72, min_periods=72).sum().replace(0, np.nan)

        # Price/flow divergence: positive means price is rising with weaker
        # buying pressure; negative means price is falling with weaker selling.
        x["flow_div24"] = x.ret24 - x.flow24
        x["flow_slope"] = x.flow24 - x.flow24.shift(6)

        # Absorption proxy: unusually high volume but relatively small body,
        # followed by directional pressure/expansion.
        x["absorption"] = (x.vol_z >= 1.0) & (body_frac <= 0.35)
        x["vol_ratio"] = x.volume / x.volume.rolling(24, min_periods=12).mean()
        x["atr_pct"] = x.atr.rolling(240, min_periods=120).rank(pct=True)

        # 4H context.
        q["ema20"] = q.close.ewm(span=20, adjust=False).mean()
        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx14(q)
        q["ema50_slope"] = q.ema50.pct_change(3)
        q_rng = (q.high-q.low).replace(0,np.nan)
        q_clv = ((q.close-q.low)-(q.high-q.close))/q_rng
        q["flow"] = q_clv * q.volume
        q["flow24"] = q.flow.rolling(6, min_periods=6).sum() / q.volume.rolling(6, min_periods=6).sum().replace(0,np.nan)
        q["trend"] = np.where((q.close > q.ema20) & (q.ema20 > q.ema50) & (q.ema50 > q.ema200), 1,
                      np.where((q.close < q.ema20) & (q.ema20 < q.ema50) & (q.ema50 < q.ema200), -1, 0))

        # 1D primary regime.
        d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
        d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
        d["ema50_slope"] = d.ema50.pct_change(5)
        d["regime"] = np.where((d.close > d.ema50) & (d.ema50 > d.ema200) & (d.ema50_slope > 0), 1,
                       np.where((d.close < d.ema50) & (d.ema50 < d.ema200) & (d.ema50_slope < 0), -1, 0))

        x["h4_trend"] = q.trend.reindex(x.index, method="ffill")
        x["h4_adx"] = q.adx.reindex(x.index, method="ffill")
        x["h4_slope"] = q.ema50_slope.reindex(x.index, method="ffill")
        x["h4_flow"] = q.flow24.reindex(x.index, method="ffill")
        x["d1_regime"] = d.regime.reindex(x.index, method="ffill")
        x["d1_slope"] = d.ema50_slope.reindex(x.index, method="ffill")
        f[a] = x
    return f


def add_cross_sectional_rank(f):
    """Causal cross-sectional ranks from 24H return and 24H flow."""
    times = sorted(set().union(*(x.index for x in f.values())))
    for ts in times:
        vals_r = [(a, f[a].ret24.get(ts, np.nan)) for a in SYMBOLS]
        vals_f = [(a, f[a].flow24.get(ts, np.nan)) for a in SYMBOLS]
        for vals, col in [(vals_r, "mom_rank"), (vals_f, "flow_rank")]:
            vals = [(a, float(v)) for a, v in vals if np.isfinite(v)]
            vals.sort(key=lambda z: z[1])
            n = len(vals)
            if n < 2:
                continue
            for rank, (a, _) in enumerate(vals):
                f[a].loc[ts, col] = rank / (n - 1)
    return f


def signal_events(f, p):
    """Causal flow/positioning-proxy trigger; entry is next 1H open."""
    sig = []
    for asset, x in f.items():
        for i in range(260, len(x)-1):
            r = x.iloc[i]
            prev = x.iloc[i-1]
            keys = ["atr","range_atr","body_atr","close_pos","vol_z","flow_z","flow24",
                    "flow72","flow_slope","h4_adx","h4_slope","h4_flow","d1_regime",
                    "d1_slope","mom_rank","flow_rank"]
            if not all(np.isfinite(r.get(k, np.nan)) for k in keys):
                continue

            long_ctx = r.d1_regime == 1 and r.h4_trend == 1
            short_ctx = r.d1_regime == -1 and r.h4_trend == -1
            if not (long_ctx or short_ctx):
                continue

            # Pressure event: flow is positive/negative, improving, and
            # cross-sectional positioning agrees with the direction.
            long_flow = (r.flow24 >= p["flow24_min"] and r.flow72 >= p["flow72_min"]
                         and r.flow_slope >= p["flow_slope_min"] and r.flow_rank >= p["flow_rank_long"])
            short_flow = (r.flow24 <= -p["flow24_min"] and r.flow72 <= -p["flow72_min"]
                          and r.flow_slope <= -p["flow_slope_min"] and r.flow_rank <= p["flow_rank_short"])

            # Avoid chasing an already exhausted move.
            long_price = r.ret24 >= p["ret24_min"] and r.ret24 <= p["ret24_max"]
            short_price = r.ret24 <= -p["ret24_min"] and r.ret24 >= -p["ret24_max"]

            # Absorption then expansion: previous candle shows high-volume
            # indecision, current candle resolves directionally.
            prev_rng = max(float(prev.high-prev.low), 1e-12)
            prev_absorb = float(prev.volume) >= p["abs_vol_mult"] * float(x.volume.rolling(24, min_periods=12).mean().iloc[i-1]) and ((abs(float(prev.close-prev.open))/prev_rng) <= p["abs_body_frac"])
            long_expand = r.close > r.open and r.body_atr >= p["body_atr"] and r.close_pos >= p["close_pos"] and r.range_atr >= p["range_atr"]
            short_expand = r.close < r.open and r.body_atr >= p["body_atr"] and r.close_pos <= (1-p["close_pos"]) and r.range_atr >= p["range_atr"]

            ls = 0.0
            ss = 0.0
            if long_ctx:
                ls += 2.0
                if r.h4_slope > 0: ls += 0.75
                if r.h4_adx >= p["adx"]: ls += 0.75
                if long_flow: ls += 2.0
                if long_price: ls += 0.75
                if prev_absorb: ls += 1.0
                if long_expand: ls += 1.25
                if r.vol_z >= p["vol_z"]: ls += 0.50
                if r.h4_flow >= p["h4_flow_min"]: ls += 0.50
            if short_ctx:
                ss += 2.0
                if r.h4_slope < 0: ss += 0.75
                if r.h4_adx >= p["adx"]: ss += 0.75
                if short_flow: ss += 2.0
                if short_price: ss += 0.75
                if prev_absorb: ss += 1.0
                if short_expand: ss += 1.25
                if r.vol_z >= p["vol_z"]: ss += 0.50
                if r.h4_flow <= -p["h4_flow_min"]: ss += 0.50

            if ls >= p["min_score"] and long_expand and long_flow and ls > ss:
                sig.append({"asset":asset,"signal_ts":x.index[i],"side":"LONG",
                            "family":"FLOW_POSITIONING_PROXY","atr":float(r.atr),"score":ls})
            elif ss >= p["min_score"] and short_expand and short_flow and ss > ls:
                sig.append({"asset":asset,"signal_ts":x.index[i],"side":"SHORT",
                            "family":"FLOW_POSITIONING_PROXY","atr":float(r.atr),"score":ss})
    return sig

def backtest(f, sig):
    byts = {}
    for z in sig:
        byts.setdefault(z["signal_ts"], []).append(z)
    all_ts = sorted(set().union(*(x.index for x in f.values())))
    positions = []
    trades = []
    last_close = pd.Timestamp.min.tz_localize("UTC")

    for ts in all_ts:
        # Exit management is performed on EVERY 1H timestamp.
        for p in positions[:]:
            x = f[p["asset"]]
            if ts not in x.index or ts <= p["entry_ts"]:
                continue
            row = x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"] == "LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"] == "LONG" else row.low <= p["tp"]
            if hit_sl or hit_tp:
                # Conservative rule: if both are touched, SL wins.
                outcome = "LOSS" if hit_sl else "WIN"
                ex = p["sl"] if hit_sl else p["tp"]
                gross = ((ex-p["entry"])/p["entry"] if p["side"] == "LONG" else (p["entry"]-ex)/p["entry"])
                gross *= TRADE_MARGIN * LEVERAGE
                pnl = gross - (TRADE_MARGIN * LEVERAGE * FEE_RATE * 2)
                trades.append({**p, "exit_ts":ts, "outcome":outcome, "pnl":pnl})
                positions.remove(p)
                last_close = ts

        # Never enter on the same timestamp a result becomes known.
        if ts <= last_close or len(positions) >= MAX_OPEN_POSITIONS:
            continue

        used = set()
        for z in sorted(byts.get(ts, []), key=lambda q: abs(q["atr"]), reverse=True):
            if len(positions) >= MAX_OPEN_POSITIONS or z["asset"] in used:
                continue
            if any(p["asset"] == z["asset"] for p in positions):
                continue
            x = f[z["asset"]]
            future = x.index[x.index > ts]
            if len(future) == 0:
                continue
            ets = future[0]
            entry = float(x.loc[ets, "open"]) * (1 + SLIPPAGE if z["side"] == "LONG" else 1 - SLIPPAGE)
            risk = 1.5 * z["atr"]
            sl = entry - risk if z["side"] == "LONG" else entry + risk
            tp = entry + RR * risk if z["side"] == "LONG" else entry - RR * risk
            positions.append({"asset":z["asset"],"family":z["family"],"side":z["side"],
                              "signal_ts":ts,"entry_ts":ets,"entry":entry,"sl":sl,"tp":tp})
            used.add(z["asset"])

    return trades, positions


def metrics(trades):
    wins = sum(z["outcome"] == "WIN" for z in trades)
    gross_w = sum(z["pnl"] for z in trades if z["pnl"] > 0)
    gross_l = -sum(z["pnl"] for z in trades if z["pnl"] < 0)
    eq = INITIAL_EQUITY
    peak = eq
    dd = 0.0
    streak = 0
    mx = 0
    for z in sorted(trades, key=lambda q: q["exit_ts"]):
        eq += z["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        streak = streak + 1 if z["outcome"] == "LOSS" else 0
        mx = max(mx, streak)
    first = min((z["exit_ts"] for z in trades), default=None)
    last = max((z["exit_ts"] for z in trades), default=None)
    days = max(1.0, (last-first).total_seconds()/86400) if first is not None else 1.0
    return {
        "trades":len(trades),
        "wr":100*wins/len(trades) if trades else 0.0,
        "pf":gross_w/gross_l if gross_l else 0.0,
        "pnl":sum(z["pnl"] for z in trades),
        "dd":dd,
        "max_streak":mx,
        "trades_day":len(trades)/days,
    }


def score(m):
    # Research ranking only; never alters trade outcomes.
    if m["trades"] < 100:
        return -1e12
    wr_gap = max(0.0, 50.0 - m["wr"])
    streak_gap = max(0, m["max_streak"] - 4)
    dd_penalty = max(0.0, -m["dd"] - 1000.0) * 0.15
    return m["pnl"] + 700*m["pf"] - 180*wr_gap - 180*streak_gap - dd_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    print("=" * 96)
    print("HUNTER-V149 — FLOW / POSITIONING PROXY RESEARCH")
    print("Signal=1H | Context=4H | Regime=1D | Raw source=15m")
    print("OHLCV flow/positioning proxy: pressure + divergence + absorption + expansion")
    print("=" * 96)

    all15 = ensure_data(Path(args.data_dir), args.days)
    f = add_cross_sectional_rank(build_features(all15))

    # 3 x 2 x 2 x 2 = 24 causal configurations.
    grid = []
    for min_score in [5.75, 6.50, 7.25]:
        for flow24_min in [0.10, 0.20]:
            for flow_slope_min in [0.03, 0.06]:
                for adx in [16.0, 20.0]:
                    grid.append({
                        "min_score": min_score, "flow24_min": flow24_min,
                        "flow72_min": 0.05, "flow_slope_min": flow_slope_min,
                        "flow_rank_long": 0.60, "flow_rank_short": 0.40,
                        "ret24_min": -0.01, "ret24_max": 0.08,
                        "adx": adx, "vol_z": -0.25, "body_atr": 0.35,
                        "range_atr": 0.90, "close_pos": 0.60,
                        "abs_vol_mult": 1.50, "abs_body_frac": 0.35,
                        "h4_flow_min": 0.02,
                    })

    print(f"Testing {len(grid)} causal flow/positioning parameter sets...")
    results = []
    for n, p in enumerate(grid, 1):
        sig = signal_events(f, p)
        trades, openp = backtest(f, sig)
        m = metrics(trades)
        m.update({"open":len(openp), "signals":len(sig), "score":score(m)})
        results.append((m, p))
        if n % 12 == 0:
            print(f"  tested {n}/{len(grid)}")

    results.sort(key=lambda z: z[0]["score"], reverse=True)
    print("\nTop 15 candidates:")
    for i, (m, p) in enumerate(results[:15], 1):
        print(f"{i:>2} trades={m['trades']:>4} WR={m['wr']:>6.2f}% PF={m['pf']:.3f} "
              f"PnL=${m['pnl']:,.2f} DD=${m['dd']:,.2f} streak={m['max_streak']:>2} "
              f"t/day={m['trades_day']:.2f} | score={p['min_score']:.2f} "
              f"flow24={p['flow24_min']:.2f} slope={p['flow_slope_min']:.2f} adx={p['adx']:.0f}")

    eligible = [z for z in results if z[0]["wr"] >= 50.0 and z[0]["max_streak"] <= 4 and z[0]["trades"] >= 100]
    print("\nELIGIBLE:", bool(eligible))
    if eligible:
        m, p = eligible[0]
        print("Best eligible:", m)
        print("Parameters:", p)
        print("NEXT: strict walk-forward/OOS validation before any live consideration.")
    else:
        print("No tested configuration reached WR>=50% AND max loss streak<=4 with >=100 trades.")
        print("NEXT: do not force the target; reject/iterate only if the family shows measurable edge.")


if __name__ == "__main__":
    main()
