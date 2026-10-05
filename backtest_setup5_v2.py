#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP 5 V2 — Causal Quality-Filtered Backtest

Purpose
-------
A stricter, research-oriented translation of Setup 5 from the supplied PDF.
V1 produced many trades but only 34.09% WR, PF 0.882 and max loss streak 18.
V2 keeps the same 15m execution / 4H HTF architecture and the core sequence,
then adds ONLY causal quality filters whose information is known before entry.

Core sequence (long)
--------------------
downtrend -> HTF demand touch -> bullish CHOCH -> OB -> 2 aligned liquidity lows
-> liquidity sweep -> return to OB -> next-bar entry.
Short is the exact inverse.

IMPORTANT INTEGRITY RULES
-------------------------
- No lookahead / future leak.
- Confirmed pivots require PIVOT bars on both sides.
- HTF zones are mapped to execution bars only after the HTF candle is closed.
- CHOCH is confirmed by a CLOSED 15m candle; entry cannot occur on CHOCH candle.
- OB, liquidity, sweep and filters use only information available before entry.
- Structural target is the latest confirmed post-CHOCH structural extreme known before entry.
- No same-candle entry/exit ambiguity: the entry bar is not checked for SL/TP.
- If SL and TP are both hit on the same later candle, SL wins conservatively.
- Max one open trade PER SYMBOL. Different symbols may overlap.
- No same-symbol re-entry on the exit candle.
- No timeout, breakeven, trailing, partial exits or artificial stop-after-loss rules.
- Unresolved trades are reported separately.

RESEARCH SPLIT
--------------
The 365-day test window is split into:
  TRAIN = first 50%
  VALID = next 20%
  OOS   = final 30%

V2 parameters are fixed in this file rather than optimized on OOS. The split is
reported so that later research can tune on TRAIN, select on VALID, and lock
before OOS. This prevents using the OOS period as a tuning target.

Data
----
Binance USD-M Futures real 15m and 4h klines. No synthetic OHLCV/OI/CVD data.
"""

from __future__ import annotations

import io
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ------------------------- CONFIG -------------------------

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT", "NEARUSDT",
    "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT",
    "ICPUSDT", "WIFUSDT",
]

EXEC_TF = "15m"
HTF_TF = "4h"
TEST_DAYS = 365
WARMUP_DAYS = 90

# Account / execution
INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE
RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Source-mechanical structure
PIVOT = 2
ALIGN_TOL = 0.0025              # V2: stricter than V1 0.40%
MAX_LIQ_TO_OB_BARS = 18         # V2: stricter than V1 24
MAX_RISK = 0.08
MIN_RISK = 0.0005
SL_BUFFER = 0.0005

# V2 causal quality filters
MIN_CHOCH_BODY_ATR = 0.35       # closed CHOCH candle body vs ATR
MIN_CHOCH_CLOSE_EXT_ATR = 0.15  # close must extend beyond broken swing by ATR fraction
MIN_SWEEP_PEN_ATR = 0.10        # sweep penetration beyond liquidity level
MIN_STRUCTURAL_ROOM_R = 2.50     # structural target must be >= 2.5R away
MAX_ENTRY_DELAY_BARS = 48       # OB return must happen within 48 bars after sweep
MAX_OB_AGE_FROM_CHOCH = 36       # OB must not be stale when entry happens
MIN_OB_BODY_ATR = 0.05           # reject almost-flat OB candles

# Data / reproducibility
BACKTEST_END = os.getenv("BACKTEST_END", "").strip()  # YYYY-MM-DD, inclusive UTC date
REQUEST_TIMEOUT = 30
MAX_WORKERS = min(8, len(SYMBOLS))
MIN_EXEC_ROWS = 30000
MIN_HTF_ROWS = 1800

# Split inside the 365-day test window
TRAIN_FRAC = 0.50
VALID_FRAC = 0.20
OOS_FRAC = 0.30

ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "setup5-v2-research/1.0"})


@dataclass
class Trade:
    symbol: str
    direction: str
    signal_idx: int
    entry_idx: int
    exit_idx: int
    entry_time: str
    exit_time: str
    entry: float
    sl: float
    tp: float
    exit: float
    pnl: float
    result: str
    R: float
    htf_zone_idx: int
    reaction_idx: int
    choch_idx: int
    ob_idx: int
    liq_idx1: int
    liq_idx2: int
    sweep_idx: int
    structural_target_idx: int
    structural_target: float
    choch_body_atr: float
    sweep_pen_atr: float
    structural_room_R: float


# ------------------------- DATA -------------------------

def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def get_end_ts() -> pd.Timestamp:
    if BACKTEST_END:
        d = pd.Timestamp(BACKTEST_END, tz="UTC")
        return d + pd.Timedelta(hours=23, minutes=45)
    # Use previous completed UTC day. This avoids the current incomplete 15m candle.
    now = utc_now()
    return (now.floor("D") - pd.Timedelta(minutes=15))


def month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def parse_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        csv_name = next((n for n in names if n.lower().endswith(".csv")), None)
        if not csv_name:
            raise ValueError("ZIP contains no CSV")
        with z.open(csv_name) as f:
            raw = pd.read_csv(f, header=None)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.iloc[:, :12]
    raw.columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote", "ignore"
    ]
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw["close_time"] = pd.to_numeric(raw["close_time"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["open_time", "open", "high", "low", "close"])
    raw["open_time"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], unit="ms", utc=True)
    return raw.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def fetch_archive(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for m in month_iter(start, end):
        ym = m.strftime("%Y-%m")
        url = f"{ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and r.content:
                parts.append(parse_zip_csv(r.content))
        except Exception:
            pass

    # Daily fallback covers the latest month when monthly archive is not yet available.
    if not parts or (end - start).days <= 45:
        d = pd.Timestamp(start.date(), tz="UTC")
        while d <= end:
            url = f"{ARCHIVE_BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d.strftime('%Y-%m-%d')}.zip"
            try:
                r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200 and r.content:
                    parts.append(parse_zip_csv(r.content))
            except Exception:
                pass
            d += pd.Timedelta(days=1)

    if not parts:
        raise RuntimeError(f"No Binance archive data for {symbol} {interval}")

    df = pd.concat(parts, ignore_index=True)
    df = df[(df.open_time >= start) & (df.open_time <= end)].copy()
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    # Do not use an incomplete candle.
    df = df[df.close_time <= utc_now()].copy()
    return df.reset_index(drop=True)


def validate_ohlcv(df: pd.DataFrame, interval_minutes: int, symbol: str, name: str):
    if len(df) < (MIN_EXEC_ROWS if name == "15m" else MIN_HTF_ROWS):
        raise RuntimeError(f"{symbol} {name}: too few rows: {len(df)}")
    if not df.open_time.is_monotonic_increasing or df.open_time.duplicated().any():
        raise RuntimeError(f"{symbol} {name}: timestamps invalid/duplicated")
    delta = df.open_time.diff().dropna()
    expected = pd.Timedelta(minutes=interval_minutes)
    gaps = delta[delta != expected]
    if not gaps.empty:
        largest = gaps.max()
        raise RuntimeError(f"{symbol} {name}: data gap detected, largest={largest}")
    if (df.high < df.low).any() or (df.open > df.high).any() or (df.open < df.low).any() or (df.close > df.high).any() or (df.close < df.low).any():
        raise RuntimeError(f"{symbol} {name}: invalid OHLC relationship")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol} {name}: non-positive price")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ------------------------- CAUSAL STRUCTURE -------------------------

def confirmed_pivots(df: pd.DataFrame, p: int = PIVOT):
    hi = df.high.to_numpy()
    lo = df.low.to_numpy()
    ph = np.full(len(df), False)
    pl = np.full(len(df), False)
    for i in range(p, len(df) - p):
        if hi[i] > np.max(hi[i-p:i]) and hi[i] >= np.max(hi[i+1:i+p+1]):
            ph[i] = True
        if lo[i] < np.min(lo[i-p:i]) and lo[i] <= np.min(lo[i+1:i+p+1]):
            pl[i] = True
    return ph, pl


def last_confirmed_before(flag: np.ndarray, idx: int) -> Optional[int]:
    j = idx - PIVOT
    if j < 0:
        return None
    inds = np.flatnonzero(flag[:j+1])
    return int(inds[-1]) if len(inds) else None


def build_htf_context(htf: pd.DataFrame) -> pd.DataFrame:
    ph, pl = confirmed_pivots(htf, PIVOT)
    out = htf.copy()
    out["pivot_high"] = ph
    out["pivot_low"] = pl
    return out


def map_last_closed_htf(htf: pd.DataFrame, exec_times: pd.Series) -> np.ndarray:
    # searchsorted(..., side='right') - 1 gives the last HTF candle whose OPEN is
    # before/equal to the execution candle. Because execution signals occur after
    # a 15m candle closes and the HTF candle itself is only considered closed when
    # its close_time is before the signal time, we use close_time here.
    closes = htf.close_time.astype("int64").to_numpy()
    vals = exec_times.astype("int64").to_numpy()
    return np.searchsorted(closes, vals, side="right") - 1


def trend_state(df: pd.DataFrame, ph: np.ndarray, pl: np.ndarray, idx: int) -> Tuple[bool, bool]:
    highs = [int(x) for x in np.flatnonzero(ph[:idx+1]) if x + PIVOT <= idx]
    lows = [int(x) for x in np.flatnonzero(pl[:idx+1]) if x + PIVOT <= idx]
    if len(highs) < 2 or len(lows) < 2:
        return False, False
    h1, h2 = highs[-1], highs[-2]
    l1, l2 = lows[-1], lows[-2]
    down = df.high.iloc[h1] < df.high.iloc[h2] and df.low.iloc[l1] < df.low.iloc[l2]
    up = df.high.iloc[h1] > df.high.iloc[h2] and df.low.iloc[l1] > df.low.iloc[l2]
    return bool(down), bool(up)


@dataclass
class Candidate:
    symbol: str
    direction: str
    signal_idx: int
    entry_idx: int
    htf_zone_idx: int
    reaction_idx: int
    choch_idx: int
    ob_idx: int
    liq_idx1: int
    liq_idx2: int
    sweep_idx: int
    structural_target_idx: int
    structural_target: float
    entry: float
    sl: float
    tp: float
    risk_per_unit: float
    choch_body_atr: float
    sweep_pen_atr: float
    structural_room_R: float


# ------------------------- SETUP 5 V2 -------------------------

def make_candidates(symbol: str, ex: pd.DataFrame, htf: pd.DataFrame) -> List[Candidate]:
    n = len(ex)
    ex = ex.copy()
    ex["atr"] = atr(ex, 14)
    ph, pl = confirmed_pivots(ex, PIVOT)
    ex_ph, ex_pl = ph, pl

    htf = build_htf_context(htf)
    hph = htf.pivot_high.to_numpy(bool)
    hpl = htf.pivot_low.to_numpy(bool)
    htf_map = map_last_closed_htf(htf, ex.open_time)

    candidates: List[Candidate] = []
    occupied_signal_until = -1

    # We scan chronologically. Every event is only allowed to use data at indices
    # that are already closed before the eventual entry.
    for i in range(30, n - 2):
        if i <= occupied_signal_until:
            continue
        a = ex.atr.iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # We need a confirmed execution pivot history before the reaction/CHOCH.
        down, up = trend_state(ex, ex_ph, ex_pl, i)
        if not (down or up):
            continue

        hidx = int(htf_map[i])
        if hidx < 0:
            continue

        # ---------------- LONG ----------------
        if down:
            # Latest confirmed HTF demand pivot low. It must be known before i.
            zone_idxs = [j for j in np.flatnonzero(hpl[:hidx+1]) if j + PIVOT <= hidx]
            if zone_idxs:
                z = zone_idxs[-1]
                zone_low = float(htf.low.iloc[z])
                zone_high = float(max(htf.open.iloc[z], htf.close.iloc[z]))
                if zone_low <= 0 or zone_high < zone_low:
                    continue

                # Reaction = first execution candle touching the HTF zone after its
                # confirmed formation. No future bars are used.
                reaction = None
                for r in range(i, min(n, i + 96)):
                    if ex.low.iloc[r] <= zone_high and ex.high.iloc[r] >= zone_low:
                        reaction = r
                        break
                if reaction is None or reaction <= i:
                    continue

                # If zone is decisively invalidated before CHOCH, reject the setup.
                # This only checks candles before CHOCH as required.
                choch = None
                broken_level = None
                # Last confirmed execution swing high before each test bar.
                for c in range(reaction + 1, min(n - 2, reaction + 160)):
                    prev_highs = [x for x in np.flatnonzero(ex_ph[:c+1]) if x + PIVOT <= c]
                    if not prev_highs:
                        continue
                    lh_idx = prev_highs[-1]
                    lh = float(ex.high.iloc[lh_idx])
                    if ex.low.iloc[c] < zone_low * (1 - 0.0025):
                        # zone invalidated before CHOCH
                        break
                    body = abs(ex.close.iloc[c] - ex.open.iloc[c])
                    body_atr = body / max(float(ex.atr.iloc[c]), 1e-12)
                    ext_atr = (ex.close.iloc[c] - lh) / max(float(ex.atr.iloc[c]), 1e-12)
                    if ex.close.iloc[c] > lh and body_atr >= MIN_CHOCH_BODY_ATR and ext_atr >= MIN_CHOCH_CLOSE_EXT_ATR:
                        choch = c
                        broken_level = lh
                        choch_body_atr = body_atr
                        break
                if choch is None:
                    continue

                # Lowest confirmed swing low between reaction and CHOCH = OB.
                ob_idxs = [x for x in np.flatnonzero(ex_pl[reaction:choch+1])]
                ob_idxs = [reaction + int(x) for x in ob_idxs if reaction + int(x) + PIVOT <= choch]
                if not ob_idxs:
                    continue
                ob = min(ob_idxs, key=lambda x: ex.low.iloc[x])
                ob_low = float(ex.low.iloc[ob])
                ob_high = float(max(ex.open.iloc[ob], ex.close.iloc[ob]))
                if ob_high <= ob_low:
                    continue
                if abs(ex.close.iloc[ob] - ex.open.iloc[ob]) / max(float(ex.atr.iloc[ob]), 1e-12) < MIN_OB_BODY_ATR:
                    continue

                # Find two aligned confirmed lows after CHOCH and before the sweep.
                liqs = [x for x in np.flatnonzero(ex_pl[choch+1:])]
                liqs = [choch + 1 + int(x) for x in liqs if choch + 1 + int(x) + PIVOT < n]
                if len(liqs) < 2:
                    continue
                pair = None
                for k in range(1, len(liqs)):
                    a1, a2 = liqs[k-1], liqs[k]
                    if a2 - ob > MAX_LIQ_TO_OB_BARS:
                        break
                    p1, p2 = float(ex.low.iloc[a1]), float(ex.low.iloc[a2])
                    level = (p1 + p2) / 2.0
                    if abs(p1 - p2) / max(level, 1e-12) <= ALIGN_TOL:
                        pair = (a1, a2, level)
                if pair is None:
                    continue
                liq1, liq2, liq_level = pair

                # Sweep must occur after second liquidity pivot is confirmed.
                sweep = None
                sweep_pen_atr = None
                for s in range(liq2 + PIVOT, min(n - 2, liq2 + 72)):
                    pen = (liq_level - ex.low.iloc[s]) / max(float(ex.atr.iloc[s]), 1e-12)
                    if ex.low.iloc[s] < liq_level and ex.close.iloc[s] >= liq_level and pen >= MIN_SWEEP_PEN_ATR:
                        sweep = s
                        sweep_pen_atr = pen
                        break
                if sweep is None:
                    continue

                # Structural target: highest CONFIRMED execution pivot high after
                # CHOCH that is already confirmed before the eventual entry.
                # We update it as time advances, never using future bars.
                structural_candidates = [x for x in np.flatnonzero(ex_ph[choch+1:sweep+1]) if choch + 1 + int(x) + PIVOT <= sweep]
                structural_candidates = [choch + 1 + int(x) for x in structural_candidates]
                if not structural_candidates:
                    continue
                target_idx = max(structural_candidates, key=lambda x: ex.high.iloc[x])
                target = float(ex.high.iloc[target_idx])

                sl = ob_low * (1.0 - SL_BUFFER)
                entry = float(ex.open.iloc[sweep + 1]) * (1.0 + SLIPPAGE)
                risk = entry - sl
                if risk <= 0 or risk / entry < MIN_RISK or risk / entry > MAX_RISK:
                    continue
                tp = entry + RR * risk
                structural_room_R = (target - entry) / risk
                if structural_room_R < MIN_STRUCTURAL_ROOM_R:
                    continue

                # Return to OB after sweep, with no use of future candles beyond each
                # candidate entry. We select the first valid return.
                entry_idx = None
                for r in range(sweep + 1, min(n - 1, sweep + 1 + MAX_ENTRY_DELAY_BARS)):
                    if r - choch > MAX_OB_AGE_FROM_CHOCH:
                        break
                    if ex.low.iloc[r] <= ob_high and ex.high.iloc[r] >= ob_low:
                        entry_idx = r + 1
                        break
                if entry_idx is None or entry_idx >= n:
                    continue
                if entry_idx <= sweep:
                    continue

                # Recompute target using only pivots confirmed BEFORE the entry.
                valid_targets = [x for x in np.flatnonzero(ex_ph[choch+1:entry_idx]) if choch + 1 + int(x) + PIVOT < entry_idx]
                valid_targets = [choch + 1 + int(x) for x in valid_targets]
                if not valid_targets:
                    continue
                target_idx = max(valid_targets, key=lambda x: ex.high.iloc[x])
                target = float(ex.high.iloc[target_idx])
                structural_room_R = (target - entry) / risk
                if structural_room_R < MIN_STRUCTURAL_ROOM_R:
                    continue

                candidates.append(Candidate(
                    symbol=symbol, direction="LONG", signal_idx=entry_idx-1, entry_idx=entry_idx,
                    htf_zone_idx=z, reaction_idx=reaction, choch_idx=choch, ob_idx=ob,
                    liq_idx1=liq1, liq_idx2=liq2, sweep_idx=sweep,
                    structural_target_idx=target_idx, structural_target=target,
                    entry=entry, sl=sl, tp=tp, risk_per_unit=risk,
                    choch_body_atr=choch_body_atr, sweep_pen_atr=float(sweep_pen_atr),
                    structural_room_R=structural_room_R,
                ))
                occupied_signal_until = entry_idx
                continue

        # ---------------- SHORT ----------------
        if up:
            zone_idxs = [j for j in np.flatnonzero(hph[:hidx+1]) if j + PIVOT <= hidx]
            if zone_idxs:
                z = zone_idxs[-1]
                zone_low = float(min(htf.open.iloc[z], htf.close.iloc[z]))
                zone_high = float(htf.high.iloc[z])
                reaction = None
                for r in range(i, min(n, i + 96)):
                    if ex.high.iloc[r] >= zone_low and ex.low.iloc[r] <= zone_high:
                        reaction = r
                        break
                if reaction is None or reaction <= i:
                    continue

                choch = None
                for c in range(reaction + 1, min(n - 2, reaction + 160)):
                    prev_lows = [x for x in np.flatnonzero(ex_pl[:c+1]) if x + PIVOT <= c]
                    if not prev_lows:
                        continue
                    hl_idx = prev_lows[-1]
                    hl = float(ex.low.iloc[hl_idx])
                    if ex.high.iloc[c] > zone_high * (1 + 0.0025):
                        break
                    body = abs(ex.close.iloc[c] - ex.open.iloc[c])
                    body_atr = body / max(float(ex.atr.iloc[c]), 1e-12)
                    ext_atr = (hl - ex.close.iloc[c]) / max(float(ex.atr.iloc[c]), 1e-12)
                    if ex.close.iloc[c] < hl and body_atr >= MIN_CHOCH_BODY_ATR and ext_atr >= MIN_CHOCH_CLOSE_EXT_ATR:
                        choch = c
                        choch_body_atr = body_atr
                        break
                if choch is None:
                    continue

                ob_idxs = [x for x in np.flatnonzero(ex_ph[reaction:choch+1])]
                ob_idxs = [reaction + int(x) for x in ob_idxs if reaction + int(x) + PIVOT <= choch]
                if not ob_idxs:
                    continue
                ob = max(ob_idxs, key=lambda x: ex.high.iloc[x])
                ob_low = float(min(ex.open.iloc[ob], ex.close.iloc[ob]))
                ob_high = float(ex.high.iloc[ob])
                if ob_high <= ob_low:
                    continue
                if abs(ex.close.iloc[ob] - ex.open.iloc[ob]) / max(float(ex.atr.iloc[ob]), 1e-12) < MIN_OB_BODY_ATR:
                    continue

                liqs = [x for x in np.flatnonzero(ex_ph[choch+1:])]
                liqs = [choch + 1 + int(x) for x in liqs if choch + 1 + int(x) + PIVOT < n]
                if len(liqs) < 2:
                    continue
                pair = None
                for k in range(1, len(liqs)):
                    a1, a2 = liqs[k-1], liqs[k]
                    if a2 - ob > MAX_LIQ_TO_OB_BARS:
                        break
                    p1, p2 = float(ex.high.iloc[a1]), float(ex.high.iloc[a2])
                    level = (p1 + p2) / 2.0
                    if abs(p1 - p2) / max(level, 1e-12) <= ALIGN_TOL:
                        pair = (a1, a2, level)
                if pair is None:
                    continue
                liq1, liq2, liq_level = pair

                sweep = None
                sweep_pen_atr = None
                for s in range(liq2 + PIVOT, min(n - 2, liq2 + 72)):
                    pen = (ex.high.iloc[s] - liq_level) / max(float(ex.atr.iloc[s]), 1e-12)
                    if ex.high.iloc[s] > liq_level and ex.close.iloc[s] <= liq_level and pen >= MIN_SWEEP_PEN_ATR:
                        sweep = s
                        sweep_pen_atr = pen
                        break
                if sweep is None:
                    continue

                structural_candidates = [x for x in np.flatnonzero(ex_pl[choch+1:sweep+1]) if choch + 1 + int(x) + PIVOT <= sweep]
                structural_candidates = [choch + 1 + int(x) for x in structural_candidates]
                if not structural_candidates:
                    continue
                target_idx = min(structural_candidates, key=lambda x: ex.low.iloc[x])
                target = float(ex.low.iloc[target_idx])

                sl = ob_high * (1.0 + SL_BUFFER)
                entry = float(ex.open.iloc[sweep + 1]) * (1.0 - SLIPPAGE)
                risk = sl - entry
                if risk <= 0 or risk / entry < MIN_RISK or risk / entry > MAX_RISK:
                    continue
                tp = entry - RR * risk
                structural_room_R = (entry - target) / risk
                if structural_room_R < MIN_STRUCTURAL_ROOM_R:
                    continue

                entry_idx = None
                for r in range(sweep + 1, min(n - 1, sweep + 1 + MAX_ENTRY_DELAY_BARS)):
                    if r - choch > MAX_OB_AGE_FROM_CHOCH:
                        break
                    if ex.high.iloc[r] >= ob_low and ex.low.iloc[r] <= ob_high:
                        entry_idx = r + 1
                        break
                if entry_idx is None or entry_idx >= n:
                    continue
                if entry_idx <= sweep:
                    continue

                valid_targets = [x for x in np.flatnonzero(ex_pl[choch+1:entry_idx]) if choch + 1 + int(x) + PIVOT < entry_idx]
                valid_targets = [choch + 1 + int(x) for x in valid_targets]
                if not valid_targets:
                    continue
                target_idx = min(valid_targets, key=lambda x: ex.low.iloc[x])
                target = float(ex.low.iloc[target_idx])
                structural_room_R = (entry - target) / risk
                if structural_room_R < MIN_STRUCTURAL_ROOM_R:
                    continue

                candidates.append(Candidate(
                    symbol=symbol, direction="SHORT", signal_idx=entry_idx-1, entry_idx=entry_idx,
                    htf_zone_idx=z, reaction_idx=reaction, choch_idx=choch, ob_idx=ob,
                    liq_idx1=liq1, liq_idx2=liq2, sweep_idx=sweep,
                    structural_target_idx=target_idx, structural_target=target,
                    entry=entry, sl=sl, tp=tp, risk_per_unit=risk,
                    choch_body_atr=choch_body_atr, sweep_pen_atr=float(sweep_pen_atr),
                    structural_room_R=structural_room_R,
                ))
                occupied_signal_until = entry_idx

    # Deduplicate by symbol + direction + entry index.
    uniq = {}
    for c in candidates:
        uniq[(c.symbol, c.direction, c.entry_idx)] = c
    return sorted(uniq.values(), key=lambda x: (x.entry_idx, x.direction))


# ------------------------- EXECUTION -------------------------

def simulate_symbol(ex: pd.DataFrame, candidates: List[Candidate], symbol: str) -> Tuple[List[Trade], int]:
    trades: List[Trade] = []
    unresolved = 0
    occupied_until = -1

    for c in sorted(candidates, key=lambda x: x.entry_idx):
        if c.entry_idx <= occupied_until:
            continue
        if c.entry_idx >= len(ex):
            unresolved += 1
            continue

        # Entry happens at next candle open. The entry candle itself is not checked
        # for SL/TP, preventing intrabar ordering ambiguity.
        entry_idx = c.entry_idx
        entry = c.entry
        exit_idx = None
        exit_price = None
        result = None

        for j in range(entry_idx + 1, len(ex)):
            hi = float(ex.high.iloc[j])
            lo = float(ex.low.iloc[j])
            if c.direction == "LONG":
                hit_sl = lo <= c.sl
                hit_tp = hi >= c.tp
                if hit_sl:
                    exit_idx, exit_price, result = j, c.sl, "LOSS"
                    break
                if hit_tp:
                    exit_idx, exit_price, result = j, c.tp, "WIN"
                    break
            else:
                hit_sl = hi >= c.sl
                hit_tp = lo <= c.tp
                if hit_sl:
                    exit_idx, exit_price, result = j, c.sl, "LOSS"
                    break
                if hit_tp:
                    exit_idx, exit_price, result = j, c.tp, "WIN"
                    break

        if exit_idx is None:
            unresolved += 1
            occupied_until = len(ex) - 1
            continue

        gross = (exit_price - entry) if c.direction == "LONG" else (entry - exit_price)
        pnl_price = gross / entry * NOTIONAL
        # Entry and exit each pay fee on notional. Slippage is already incorporated
        # in entry; exit is conservatively charged without fabricating an exit price.
        fees = 2.0 * FEE_RATE * NOTIONAL
        pnl = pnl_price - fees
        if result == "LOSS" and pnl > 0:
            raise AssertionError("Loss trade produced positive net PnL")

        r_realized = pnl_price / (c.risk_per_unit / entry * NOTIONAL)
        trades.append(Trade(
            symbol=symbol, direction=c.direction, signal_idx=c.signal_idx,
            entry_idx=entry_idx, exit_idx=exit_idx,
            entry_time=str(ex.open_time.iloc[entry_idx]),
            exit_time=str(ex.open_time.iloc[exit_idx]),
            entry=float(entry), sl=float(c.sl), tp=float(c.tp), exit=float(exit_price),
            pnl=float(pnl), result=result, R=float(r_realized),
            htf_zone_idx=c.htf_zone_idx, reaction_idx=c.reaction_idx,
            choch_idx=c.choch_idx, ob_idx=c.ob_idx,
            liq_idx1=c.liq_idx1, liq_idx2=c.liq_idx2, sweep_idx=c.sweep_idx,
            structural_target_idx=c.structural_target_idx,
            structural_target=float(c.structural_target),
            choch_body_atr=float(c.choch_body_atr), sweep_pen_atr=float(c.sweep_pen_atr),
            structural_room_R=float(c.structural_room_R),
        ))
        occupied_until = exit_idx

    return trades, unresolved


# ------------------------- AUDIT -------------------------

def audit_candidate(c: Candidate):
    assert c.htf_zone_idx < c.choch_idx, "HTF zone must predate CHOCH"
    assert c.ob_idx < c.choch_idx, "OB must predate CHOCH"
    assert c.liq_idx1 < c.sweep_idx and c.liq_idx2 < c.sweep_idx, "Liquidity must predate sweep"
    assert c.sweep_idx < c.entry_idx, "Sweep must predate entry"
    assert c.structural_target_idx < c.entry_idx, "Structural target must be known before entry"
    assert c.entry_idx >= c.signal_idx + 1, "Entry must be next candle after signal"
    assert c.tp > c.entry > c.sl if c.direction == "LONG" else c.sl > c.entry > c.tp


def audit_trades(trades: List[Trade]):
    by_symbol: Dict[str, List[Trade]] = {}
    for t in trades:
        by_symbol.setdefault(t.symbol, []).append(t)
        assert t.entry_idx > t.signal_idx
        assert t.exit_idx > t.entry_idx
        assert t.exit_idx >= t.entry_idx + 1
        assert t.result in {"WIN", "LOSS"}
        if t.result == "WIN":
            assert t.exit >= t.entry if t.direction == "LONG" else t.exit <= t.entry
        if t.result == "LOSS":
            assert t.exit <= t.entry if t.direction == "LONG" else t.exit >= t.entry
    for sym, ts in by_symbol.items():
        ts.sort(key=lambda x: x.entry_idx)
        for a, b in zip(ts, ts[1:]):
            assert b.entry_idx > a.exit_idx, f"Same-symbol overlap/re-entry violation: {sym}"


def max_streak(trades: List[Trade]) -> int:
    best = cur = 0
    for t in sorted(trades, key=lambda x: x.exit_idx):
        if t.result == "LOSS":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def metrics(trades: List[Trade], initial_equity: float = INITIAL_CAPITAL) -> Dict[str, float]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0,
                "net_pnl": 0.0, "final_equity": initial_equity, "max_dd": 0.0,
                "max_streak": 0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    ts = sorted(trades, key=lambda x: (x.exit_time, x.symbol))
    wins = [t.pnl for t in ts if t.pnl > 0]
    losses = [t.pnl for t in ts if t.pnl <= 0]
    eq = initial_equity
    peak = eq
    max_dd = 0.0
    for t in ts:
        eq += t.pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(ts),
        "wins": len(wins),
        "losses": len(losses),
        "wr": 100.0 * len(wins) / len(ts),
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "net_pnl": sum(t.pnl for t in ts),
        "final_equity": eq,
        "max_dd": max_dd,
        "max_streak": max_streak(ts),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "expectancy": float(np.mean([t.pnl for t in ts])),
    }


def print_metrics(name: str, trades: List[Trade]):
    m = metrics(trades)
    print(f"\n{name}")
    print("=" * len(name))
    print(f"Trades       : {m['trades']}")
    print(f"W/L          : {m['wins']}/{m['losses']}")
    print(f"Win rate     : {m['wr']:.2f}%")
    print(f"Profit factor: {m['pf']:.3f}")
    print(f"Net PnL      : ${m['net_pnl']:.2f}")
    print(f"Final equity : ${m['final_equity']:.2f}")
    print(f"Max DD       : ${m['max_dd']:.2f}")
    print(f"Max streak   : {m['max_streak']}")
    print(f"Avg win      : ${m['avg_win']:.2f}")
    print(f"Avg loss     : ${m['avg_loss']:.2f}")
    print(f"Expectancy   : ${m['expectancy']:.2f}")


def split_by_time(trades: List[Trade], start: pd.Timestamp, end: pd.Timestamp):
    total = end - start
    train_end = start + total * TRAIN_FRAC
    valid_end = train_end + total * VALID_FRAC
    def ts(t): return pd.Timestamp(t.entry_time, tz="UTC")
    train = [t for t in trades if ts(t) < train_end]
    valid = [t for t in trades if train_end <= ts(t) < valid_end]
    oos = [t for t in trades if valid_end <= ts(t) <= end]
    return train, valid, oos, train_end, valid_end


# ------------------------- RUNNER -------------------------

def process_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    data_start = start - pd.Timedelta(days=WARMUP_DAYS)
    ex = fetch_archive(symbol, EXEC_TF, data_start, end)
    htf = fetch_archive(symbol, HTF_TF, data_start - pd.Timedelta(days=10), end)
    validate_ohlcv(ex, 15, symbol, "15m")
    validate_ohlcv(htf, 240, symbol, "4h")
    candidates = make_candidates(symbol, ex, htf)
    for c in candidates:
        audit_candidate(c)
    trades, unresolved = simulate_symbol(ex, candidates, symbol)
    audit_trades(trades)
    return symbol, ex, candidates, trades, unresolved


def main():
    end = get_end_ts()
    start = end - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(minutes=15)
    print("SETUP 5 V2 — CAUSAL QUALITY FILTERS")
    print(f"UTC       : {start} -> {end}")
    print(f"Symbols   : {len(SYMBOLS)}")
    print(f"TF        : {EXEC_TF} / HTF {HTF_TF}")
    print(f"Capital   : ${INITIAL_CAPITAL:.2f}")
    print(f"Margin    : ${MARGIN_PER_TRADE:.2f}")
    print(f"Leverage  : {LEVERAGE:.0f}x")
    print(f"RR        : 1:{RR:.0f}")
    print(f"Split     : train {TRAIN_FRAC:.0%} / valid {VALID_FRAC:.0%} / OOS {OOS_FRAC:.0%}")
    print("Data      : Binance USD-M Futures real klines")
    print("Filters   : CHOCH strength + sweep quality + structural room + freshness + tighter liquidity")

    all_candidates: List[Candidate] = []
    all_trades: List[Trade] = []
    unresolved_total = 0
    failures = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_symbol, s, start, end): s for s in SYMBOLS}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                symbol, ex, candidates, trades, unresolved = fut.result()
                all_candidates.extend(candidates)
                all_trades.extend(trades)
                unresolved_total += unresolved
                print(f"{symbol:10s} candidates={len(candidates):4d} trades={len(trades):4d} unresolved={unresolved}")
            except Exception as e:
                failures.append((s, repr(e)))
                print(f"{s:10s} FAILED: {e}")

    if failures:
        raise RuntimeError("Data/processing failures: " + "; ".join(f"{s}: {e}" for s, e in failures))

    # Global portfolio ordering is only for equity statistics. Cross-symbol overlap
    # is intentionally allowed and is NOT filtered here.
    all_trades.sort(key=lambda x: (x.exit_time, x.symbol))
    audit_trades(all_trades)

    print(f"\nCandidates : {len(all_candidates)}")
    print(f"Trades     : {len(all_trades)}")
    print(f"Unresolved : {unresolved_total}")
    print_metrics("FULL 365-DAY", all_trades)

    train, valid, oos, train_end, valid_end = split_by_time(all_trades, start, end)
    print_metrics("TRAIN", train)
    print_metrics("VALIDATION", valid)
    print_metrics("OOS (LOCKED)", oos)

    # Symbol breakdown on OOS.
    rows = []
    for s in SYMBOLS:
        st = [t for t in oos if t.symbol == s]
        m = metrics(st)
        rows.append({"symbol": s, **m})
    sym_df = pd.DataFrame(rows)
    print("\nOOS BY SYMBOL")
    print(sym_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save artifacts for later audit/research.
    pd.DataFrame([asdict(t) for t in all_trades]).to_csv("setup5_v2_trades.csv", index=False)
    pd.DataFrame([asdict(c) for c in all_candidates]).to_csv("setup5_v2_candidates.csv", index=False)
    sym_df.to_csv("setup5_v2_oos_by_symbol.csv", index=False)

    full = metrics(all_trades)
    oosm = metrics(oos)
    print("\nTARGET CHECK — NOT ENFORCED, ONLY MEASURED")
    print(f"OOS WR >= 50%       : {'PASS' if oosm['wr'] >= 50 else 'FAIL'}")
    print(f"OOS PF > 1.00       : {'PASS' if oosm['pf'] > 1 else 'FAIL'}")
    print(f"OOS max streak <= 4 : {'PASS' if oosm['max_streak'] <= 4 else 'FAIL'}")
    print(f"OOS Net PnL > $0    : {'PASS' if oosm['net_pnl'] > 0 else 'FAIL'}")
    print("\nIntegrity: no future data used for entry decisions; same-symbol overlap blocked;")
    print("          cross-symbol overlap allowed; same-candle re-entry blocked; SL wins ties.")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
