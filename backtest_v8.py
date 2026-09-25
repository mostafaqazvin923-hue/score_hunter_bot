#!/usr/bin/env python3
"""
HUNTER-V140-RESEARCH-XT
Fresh strategy research baseline. Previous V129/V2/V3/V4 logic is intentionally NOT used.

Families tested in one causal engine:
A) Volatility-compression -> Donchian breakout
B) 1D/4H trend -> 1H momentum pullback continuation
C) Regime-adaptive selector: A in expansion/breakout conditions, B in trend conditions

Data:
- XT USDT-M Futures
- 15m raw candles, backward-paginated from XT public API
- 365 days + 35 days warmup
- 14 symbols

Execution assumptions:
- Initial equity $1,000
- $100 margin, 50x leverage
- RR 1:2
- entry next 15m candle open after completed signal candle
- no timeout
- no overlapping positions
- same-candle SL+TP => LOSS
- unresolved position at dataset end => OPEN (not counted in WR/PF)

This is a research engine, not a live trading bot.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd

SYMBOLS = ["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
BASE = "https://fapi.xt.com"
LIST_URL = BASE + "/future/market/v1/public/symbol/list"
KLINE_URL = BASE + "/future/market/v1/public/q/kline"
TIMEFRAME = "15m"
LIMIT = 1500
DAYS = 365
WARMUP_DAYS = 35
REQUEST_TIMEOUT = 20
RETRIES = 5
SLEEP = 0.12

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
MAX_OPEN = 3

# Signal parameters are deliberately frozen for the first research run.
DONCHIAN_N = 96          # 24h on 15m
ATR_N = 14
ATR_EXP_N = 48
VOL_N = 20
BREAKOUT_ATR = 0.10
SL_ATR = 1.5

EMA_FAST_4H = 50
EMA_SLOW_4H = 200
EMA_FAST_1D = 50
EMA_SLOW_1D = 200
PULLBACK_EMA = 20
MOM_LOOKBACK = 12         # 12h
MOM_MIN_ATR = 0.75


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/xt_futures_v140")
    p.add_argument("--days", type=int, default=DAYS)
    return p.parse_args()


def _get_json(url, params):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"XT request failed: {last}")


def resolve_xt_symbols():
    data = _get_json(LIST_URL, {})
    rows = data.get("result", data)
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("list", []))
    mapping = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("symbol") or row.get("contractName") or row.get("pair") or "")
        norm = raw.upper().replace("-", "_").replace("/", "_")
        for asset in SYMBOLS:
            if asset in norm and "USDT" in norm:
                mapping[asset] = raw
    return mapping


def fetch_symbol_15m(asset, xt_symbol, days=DAYS):
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = now_ms - int((days + WARMUP_DAYS) * 86400 * 1000)
    end_ms = now_ms
    rows = []
    cursor = end_ms
    seen = set()
    while cursor > start_ms:
        params = {"symbol": xt_symbol, "interval": TIMEFRAME, "limit": LIMIT, "endTime": cursor}
        payload = _get_json(KLINE_URL, params)
        data = payload.get("result", payload)
        if isinstance(data, dict):
            data = data.get("data", data.get("list", []))
        if not data:
            break
        batch = []
        for x in data:
            if isinstance(x, dict):
                ts = x.get("timestamp", x.get("time", x.get("openTime")))
                vals = [ts, x.get("open"), x.get("high"), x.get("low"), x.get("close"), x.get("volume")]
            else:
                vals = list(x[:6])
            if vals[0] is None:
                continue
            try:
                ts = int(float(vals[0]))
                if ts < 10**12:
                    ts *= 1000
                if ts in seen:
                    continue
                seen.add(ts)
                batch.append([ts] + [float(v) for v in vals[1:]])
            except Exception:
                continue
        if not batch:
            break
        rows.extend(batch)
        oldest = min(x[0] for x in batch)
        cursor = oldest - 1
        if len(batch) < 2:
            break
        time.sleep(SLEEP)
    if not rows:
        raise RuntimeError(f"No XT data for {asset}")
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df = df.loc[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + WARMUP_DAYS + 2)]
    df = df.astype(float).dropna()
    return df


def ensure_data(data_dir: Path, days: int):
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_xt_symbols()
    if len(mapping) != len(SYMBOLS):
        missing = [s for s in SYMBOLS if s not in mapping]
        raise RuntimeError(f"XT symbol resolution incomplete: {missing}")
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    for asset in SYMBOLS:
        path = data_dir / f"{asset}_USDT_15m.csv"
        df = fetch_symbol_15m(asset, mapping[asset], days)
        df.reset_index().to_csv(path, index=False)
        gaps = int(((df.index.to_series().diff().dropna() / pd.Timedelta(minutes=15)) > 1).sum())
        print(f"{asset:<8} rows={len(df)} gaps={gaps}")


def load(path: Path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close","volume"])


def resample_ohlcv(df, rule):
    out = df.resample(rule, label="right", closed="right").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return out.dropna()


def atr(df, n=14):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low),(df.high-prev).abs(),(df.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def build_features(df15):
    h4 = resample_ohlcv(df15, "4h")
    d1 = resample_ohlcv(df15, "1D")
    h4["ema50"] = h4.close.ewm(span=EMA_FAST_4H, adjust=False, min_periods=EMA_FAST_4H).mean()
    h4["ema200"] = h4.close.ewm(span=EMA_SLOW_4H, adjust=False, min_periods=EMA_SLOW_4H).mean()
    h4["atr"] = atr(h4, ATR_N)
    h4["trend_long"] = (h4.close > h4.ema50) & (h4.ema50 > h4.ema200)
    h4["trend_short"] = (h4.close < h4.ema50) & (h4.ema50 < h4.ema200)

    d1["ema50"] = d1.close.ewm(span=EMA_FAST_1D, adjust=False, min_periods=EMA_FAST_1D).mean()
    d1["ema200"] = d1.close.ewm(span=EMA_SLOW_1D, adjust=False, min_periods=EMA_SLOW_1D).mean()
    d1["bull"] = (d1.close > d1.ema200) & (d1.ema50 > d1.ema200)
    d1["bear"] = (d1.close < d1.ema200) & (d1.ema50 < d1.ema200)

    f = df15.copy()
    f["atr"] = atr(f, ATR_N)
    f["atr_mean"] = f["atr"].rolling(ATR_EXP_N, min_periods=ATR_EXP_N).mean()
    f["vol_mean"] = f.volume.rolling(VOL_N, min_periods=VOL_N).mean()
    f["dc_high"] = f.high.shift(1).rolling(DONCHIAN_N, min_periods=DONCHIAN_N).max()
    f["dc_low"] = f.low.shift(1).rolling(DONCHIAN_N, min_periods=DONCHIAN_N).min()
    f["range_pct"] = (f.high - f.low) / f.close.replace(0, np.nan)
    f["body_pct"] = (f.close - f.open).abs() / f.close.replace(0, np.nan)
    f["compression"] = f.atr < f.atr_mean * 0.85
    f["expansion"] = f.atr > f.atr_mean * 1.10
    f["break_long"] = f.close > f.dc_high + BREAKOUT_ATR * f.atr
    f["break_short"] = f.close < f.dc_low - BREAKOUT_ATR * f.atr
    f["vol_ok"] = f.volume > 1.20 * f.vol_mean

    # 1H momentum pullback, all based only on completed 1H bars.
    h1 = resample_ohlcv(df15, "1h")
    h1["atr"] = atr(h1, ATR_N)
    h1["ema20"] = h1.close.ewm(span=PULLBACK_EMA, adjust=False, min_periods=PULLBACK_EMA).mean()
    h1["mom"] = h1.close.pct_change(MOM_LOOKBACK)
    h1["impulse_long"] = (h1.mom > 0) & ((h1.close - h1.close.shift(MOM_LOOKBACK)).abs() > MOM_MIN_ATR * h1.atr)
    h1["impulse_short"] = (h1.mom < 0) & ((h1.close - h1.close.shift(MOM_LOOKBACK)).abs() > MOM_MIN_ATR * h1.atr)
    h1["pull_long"] = (h1.low <= h1.ema20) & (h1.close > h1.ema20) & (h1.close > h1.open)
    h1["pull_short"] = (h1.high >= h1.ema20) & (h1.close < h1.ema20) & (h1.close < h1.open)
    h1["long_sig"] = h1.impulse_long.shift(1) & h1.pull_long
    h1["short_sig"] = h1.impulse_short.shift(1) & h1.pull_short

    # Align higher-timeframe information to 15m using only completed bars.
    f["d_bull"] = d1.bull.reindex(f.index, method="ffill")
    f["d_bear"] = d1.bear.reindex(f.index, method="ffill")
    f["h4_long"] = h4.trend_long.reindex(f.index, method="ffill")
    f["h4_short"] = h4.trend_short.reindex(f.index, method="ffill")
    f["h1_long"] = h1.long_sig.reindex(f.index, method="ffill").fillna(False)
    f["h1_short"] = h1.short_sig.reindex(f.index, method="ffill").fillna(False)

    # Family A: breakout after compression + expansion confirmation.
    f["A_long"] = f.d_bull & f.h4_long & f.compression.shift(1).fillna(False) & f.expansion & f.break_long & f.vol_ok
    f["A_short"] = f.d_bear & f.h4_short & f.compression.shift(1).fillna(False) & f.expansion & f.break_short & f.vol_ok

    # Family B: MTF momentum continuation.
    f["B_long"] = f.d_bull & f.h4_long & f.h1_long
    f["B_short"] = f.d_bear & f.h4_short & f.h1_short

    # Family C: regime adaptive union, but a signal keeps its family identity.
    f["C_long"] = f.A_long | f.B_long
    f["C_short"] = f.A_short | f.B_short
    return f


def simulate(all_data, signal_col_long, signal_col_short):
    times = sorted(set().union(*(df.index for df in all_data.values())))
    active = {}
    trades = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for ts in times:
        # Manage first; no new signal on the candle that resolves an old position.
        closed_this_bar = False
        for sym, pos in list(active.items()):
            df = all_data[sym]
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            hit_sl = bar.low <= pos.sl if pos.side == "LONG" else bar.high >= pos.sl
            hit_tp = bar.high >= pos.tp if pos.side == "LONG" else bar.low <= pos.tp
            if hit_sl or hit_tp:
                closed_this_bar = True
                outcome = "LOSS" if hit_sl else "WIN"
                px = pos.sl if hit_sl else pos.tp
                gross = ((px-pos.entry)/pos.entry if pos.side=="LONG" else (pos.entry-px)/pos.entry) * TRADE_MARGIN * LEVERAGE
                pnl = gross - TRADE_MARGIN*LEVERAGE*FEE_RATE*2
                equity += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity-peak)
                trades.append({"symbol":sym,"side":pos.side,"entry_ts":pos.entry_ts,"exit_ts":ts,"outcome":outcome,"pnl":pnl})
                del active[sym]
        if closed_this_bar or len(active) >= MAX_OPEN:
            continue
        candidates=[]
        for sym, df in all_data.items():
            if sym in active or ts not in df.index: continue
            row=df.loc[ts]
            if bool(row.get(signal_col_long,False)):
                candidates.append((sym,"LONG",float(row.atr)))
            elif bool(row.get(signal_col_short,False)):
                candidates.append((sym,"SHORT",float(row.atr)))
        for sym, side, a in candidates:
            if len(active)>=MAX_OPEN or not np.isfinite(a) or a<=0: break
            df=all_data[sym]
            nxt=df.index[df.index>ts]
            if len(nxt)==0: continue
            entry_ts=nxt[0]
            entry=float(df.loc[entry_ts,"open"])*(1+SLIPPAGE if side=="LONG" else 1-SLIPPAGE)
            sl=entry-SL_ATR*a if side=="LONG" else entry+SL_ATR*a
            risk=abs(entry-sl)
            tp=entry+RR*risk if side=="LONG" else entry-RR*risk
            active[sym]=type("P",(),{"side":side,"entry":entry,"sl":sl,"tp":tp,"entry_ts":entry_ts})()
    return trades, equity, max_dd, len(active)


def report(name,trades,equity,max_dd,open_n):
    closed=len(trades); wins=sum(t["outcome"]=="WIN" for t in trades); losses=closed-wins
    wr=100*wins/closed if closed else 0
    gross_win=sum(t["pnl"] for t in trades if t["pnl"]>0)
    gross_loss=-sum(t["pnl"] for t in trades if t["pnl"]<0)
    pf=gross_win/gross_loss if gross_loss else (float("inf") if gross_win else 0)
    ordered=sorted(trades,key=lambda x:pd.Timestamp(x["exit_ts"]))
    streak=maxst=0
    for t in ordered:
        streak=streak+1 if t["outcome"]=="LOSS" else 0; maxst=max(maxst,streak)
    days=max(1,(pd.Timestamp.now(tz="UTC")-min([pd.Timestamp(t["entry_ts"]) for t in trades],default=pd.Timestamp.now(tz="UTC"))).days)
    print(f"\n{name}")
    print(f"Trades={closed} Open={open_n} WR={wr:.2f}% PF={pf:.3f} NetPnL=${equity-INITIAL_EQUITY:,.2f} DD=${max_dd:,.2f} MaxLossStreak={maxst} Trades/day={closed/days:.2f}")


def main():
    args=parse_args(); data_dir=Path(args.data_dir)
    ensure_data(data_dir,args.days)
    all_data={}
    print("\nBuilding research features...")
    for s in SYMBOLS:
        all_data[s]=build_features(load(data_dir/f"{s}_USDT_15m.csv"))
    print("\nRunning frozen families...")
    for name,lcol,scol in [("A_BREAKOUT","A_long","A_short"),("B_MOMENTUM_PULLBACK","B_long","B_short"),("C_REGIME_ADAPTIVE","C_long","C_short")]:
        trades,eq,dd,op=simulate(all_data,lcol,scol)
        report(name,trades,eq,dd,op)

if __name__=="__main__": main()
