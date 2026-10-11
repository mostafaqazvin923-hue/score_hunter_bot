#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP 7 V1 — HTF Zone / Major CHOCH / OB Retest

Mechanical research translation of Setup 7 from `10 ستاپ برتر.pdf`, pages 41-46.

Rules: bullish HTF structure with >=2 bullish BOS -> HTF supply touch -> lower-high
sequence -> sharp bearish major CHOCH -> retest of OB from first pre-CHOCH peak.
Long is the inverse. Numeric definitions are frozen research translations.

IMPORTANT:
- This is a causal backtest. No future candle is used before it is closed/confirmed.
- 4H is the higher timeframe; 15m is the execution timeframe (two timeframe steps lower).
- A 4H candle becomes usable only after it closes.
- A 15m pivot becomes usable only after PIVOT confirmation candles have closed.
- Entry is never on the same candle that reveals the setup/confirmation.
- SL/TP checks begin on the candle AFTER entry; no same-candle entry/exit assumptions.
- Per-symbol overlap lock: max one open trade per symbol; different symbols may overlap.
- No timeout, breakeven, trailing stop, or same-candle re-entry.
- Fixed RR = 1:2.

SOURCE-TO-CODE TRANSLATION NOTES:
1) The PDF does not give numerical definitions for BOS/IFC/OB/liquidity tolerances.
   V1 therefore uses explicit mechanical definitions below; these are research translations,
   not claims that the PDF itself specifies these exact numbers.
2) IFC is translated as a bullish/bearish 3-candle fair-value-gap style imbalance.
3) The PDF says the minor CHOCH accepts a wick above the prior peak, so V1 uses wick break.
4) The PDF says entry needs strong confirmation. V1 uses a causal confirmation candle
   (bullish/bearish engulfing OR pin-bar style rejection) on the execution timeframe.
5) The PDF target says the highest peak after BOS / first peak ahead. V1 uses the latest
   confirmed execution-timeframe swing high after the BOS and before the entry structure,
   if available; otherwise it skips the setup because a structural target cannot be known.
6) Because the global project requirement is fixed RR 1:2, a trade is only accepted when
   the 2R target does not exceed the structural target. This prevents silently replacing
   the source target with an arbitrary 2R target.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT", "NEARUSDT",
    "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT",
    "ICPUSDT", "WIFUSDT",
]

BASE_ARCHIVE = "https://data.binance.vision/data/futures/um"
INTERVAL_EXEC = "15m"
INTERVAL_HTF = "4h"

TEST_DAYS = 365
WARMUP_DAYS = 90
PIVOT = 2

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE
RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

# Mechanical translation parameters — deliberately modest and not optimized.
MAX_SETUP_BARS = 120
MAX_LIQUIDITY_DISTANCE_BARS = 80
LIQUIDITY_TOLERANCE_ATR = 0.25
OB_ATR_TOLERANCE = 0.75

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
DOWNLOAD_WORKERS = 6
SYMBOL_WORKERS = 2
USER_AGENT = "setup7-v1-research/1.0"

OUTPUT_CSV = "setup7_v1_trades.csv"


# Setup 7 candidate logic is implemented causally in backtest_symbol().

@dataclass
class Trade:
    symbol: str
    side: str
    signal_idx: int
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    exit: float
    outcome: str
    pnl: float
    r_multiple: float
    setup_bos_idx: int
    minor_choch_idx: int
    entry_ob_idx: int


# =========================
# DATA
# =========================

def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def fetch_url(url: str) -> bytes:
    """Download one Binance archive with bounded retries.

    404 is not retried because it means the archive does not exist.
    Other request failures are retried with short exponential backoff.
    """
    last_err = None
    timeout = (10, 60)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(url)
            last_err = RuntimeError(f"HTTP {r.status_code}: {url}")
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < MAX_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 6))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def read_binance_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError("No CSV in Binance ZIP")
        with z.open(csvs[0]) as f:
            raw = pd.read_csv(f, header=None)

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    if raw.shape[1] < 6:
        raise ValueError(f"Unexpected Binance kline columns: {raw.shape[1]}")
    raw = raw.iloc[:, :len(cols)]
    raw.columns = cols[:raw.shape[1]]
    for c in ["open_time", "close_time"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["open_time", "open", "high", "low", "close"])
    raw["open_time"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], unit="ms", utc=True)
    return raw[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def _archive_jobs(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp):
    """Create only the archive requests actually needed.

    Completed calendar months use one monthly archive. The current calendar month
    uses daily archives only through yesterday. This avoids the expensive failed
    current-month monthly request that caused V1 to stall.
    """
    now = pd.Timestamp.now(tz="UTC")
    current_month = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    jobs = []

    for m in month_starts(start, end):
        month_end = (m + pd.offsets.MonthBegin(1)) - pd.Timedelta(milliseconds=1)
        if m < current_month:
            ym = m.strftime("%Y-%m")
            url = f"{BASE_ARCHIVE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
            jobs.append((m, url, False))
        else:
            day = max(m, start.normalize())
            last_day = min(month_end, end)
            yesterday = now.normalize() - pd.Timedelta(days=1)
            while day <= last_day and day.normalize() <= yesterday:
                ds = day.strftime("%Y-%m-%d")
                url = f"{BASE_ARCHIVE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
                jobs.append((day, url, True))
                day += pd.Timedelta(days=1)
    return jobs


def _download_one_job(job):
    period, url, is_daily = job
    try:
        return period, is_daily, read_binance_zip(fetch_url(url)), None
    except FileNotFoundError:
        return period, is_daily, None, "404"
    except Exception as e:
        return period, is_daily, None, str(e)


def download_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fast Binance UM downloader.

    V2 keeps the exact same market-data source and date rules as V1, but downloads
    independent archives concurrently. No OHLCV values are fabricated, filled, or
    interpolated. A missing archive contributes no rows.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = _archive_jobs(symbol, interval, start, end)
    if not jobs:
        raise RuntimeError(f"No archive jobs for {symbol} {interval}")

    print(f"[{symbol} {interval}] archives={len(jobs)} workers={DOWNLOAD_WORKERS}")
    frames = []
    failures = []
    with ThreadPoolExecutor(max_workers=min(DOWNLOAD_WORKERS, len(jobs))) as pool:
        futures = [pool.submit(_download_one_job, job) for job in jobs]
        for fut in as_completed(futures):
            period, is_daily, frame, err = fut.result()
            if frame is not None:
                frames.append(frame)
            elif err != "404":
                failures.append((period, err))

    if failures:
        preview = "; ".join(f"{p}: {e}" for p, e in failures[:3])
        raise RuntimeError(f"Download failures for {symbol} {interval}: {preview}")
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    df = df[(df["open_time"] >= start) & (df["open_time"] <= end)].copy()
    if df.empty:
        raise RuntimeError(f"Empty usable data for {symbol} {interval}")
    return df.reset_index(drop=True)

def validate_ohlcv(df: pd.DataFrame, interval_minutes: int, symbol: str) -> None:
    if df.empty:
        raise ValueError(f"{symbol}: empty OHLCV")
    if not df["open_time"].is_monotonic_increasing:
        raise ValueError(f"{symbol}: timestamps not sorted")
    if df["open_time"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate timestamps")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: invalid highs")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: invalid lows")
    diffs = df["open_time"].diff().dropna().dt.total_seconds() / 60.0
    # Gaps are not fabricated/filled. They are retained but setup state is reset across them.
    # This validator only rejects non-positive/duplicate chronology.


# =========================
# CAUSAL STRUCTURE HELPERS
# =========================

def confirmed_pivots(df: pd.DataFrame, p: int = PIVOT) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    for i in range(p, n - p):
        hwin = highs[i - p:i + p + 1]
        lwin = lows[i - p:i + p + 1]
        if highs[i] == np.max(hwin) and np.sum(hwin == highs[i]) == 1:
            ph[i] = highs[i]
        if lows[i] == np.min(lwin) and np.sum(lwin == lows[i]) == 1:
            pl[i] = lows[i]
    return ph, pl


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def is_bullish(c) -> bool:
    return c.close > c.open


def is_bearish(c) -> bool:
    return c.close < c.open


def bullish_confirmation(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    c = df.iloc[i]
    p = df.iloc[i - 1]
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-12)
    lower_wick = min(c.open, c.close) - c.low
    upper_wick = c.high - max(c.open, c.close)
    engulf = (
        is_bullish(c) and is_bearish(p) and
        c.open <= p.close and c.close >= p.open
    )
    pin = is_bullish(c) and lower_wick >= body * 2.0 and lower_wick >= upper_wick * 1.25 and body / rng <= 0.45
    return bool(engulf or pin)


def bearish_confirmation(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    c = df.iloc[i]
    p = df.iloc[i - 1]
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-12)
    upper_wick = c.high - max(c.open, c.close)
    lower_wick = min(c.open, c.close) - c.low
    engulf = (
        is_bearish(c) and is_bullish(p) and
        c.open >= p.close and c.close <= p.open
    )
    pin = is_bearish(c) and upper_wick >= body * 2.0 and upper_wick >= lower_wick * 1.25 and body / rng <= 0.45
    return bool(engulf or pin)


def fvg_bullish(df: pd.DataFrame, i: int) -> Optional[Tuple[float, float]]:
    """3-candle bullish imbalance ending at i; zone is [high(i-2), low(i)]."""
    if i < 2:
        return None
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if c.low > a.high and b.close > b.open:
        return float(a.high), float(c.low)
    return None


def fvg_bearish(df: pd.DataFrame, i: int) -> Optional[Tuple[float, float]]:
    if i < 2:
        return None
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if c.high < a.low and b.close < b.open:
        return float(c.high), float(a.low)
    return None


def find_latest_bullish_bos_events(df: pd.DataFrame, p: int = PIVOT) -> List[Tuple[int, int, float]]:
    """Return (break_idx, pivot_low_idx, broken_high) causally."""
    ph, pl = confirmed_pivots(df, p)
    events = []
    last_high = None
    last_high_idx = None
    last_low = None
    last_low_idx = None
    broken = False
    for i in range(len(df)):
        # Pivot at i becomes known only at i+p.
        confirm_idx = i + p
        if confirm_idx >= len(df):
            continue
        if not np.isnan(ph[i]):
            # It is now confirmed at confirm_idx; store only when we reach that bar.
            pass
        if not np.isnan(pl[i]):
            pass

        if i >= p:
            pivot_i = i - p
            if 0 <= pivot_i < len(df):
                if not np.isnan(ph[pivot_i]):
                    last_high = ph[pivot_i]
                    last_high_idx = pivot_i
                if not np.isnan(pl[pivot_i]):
                    last_low = pl[pivot_i]
                    last_low_idx = pivot_i

        if last_high is not None and last_low_idx is not None and df.iloc[i].close > last_high:
            if not broken:
                events.append((i, last_low_idx, float(last_high)))
                broken = True
        # Once price has made a new confirmed swing high, permit a future BOS.
        if last_high_idx is not None and i > last_high_idx and not np.isnan(ph[last_high_idx]):
            # Do not reset here; the new pivot high will replace last_high when confirmed.
            pass
        if last_high is not None and df.iloc[i].close <= last_high:
            # Keep broken state until a new confirmed high replaces the broken level.
            pass
        if i >= p and not np.isnan(ph[i - p]):
            broken = False
    return events


def find_latest_bearish_bos_events(df: pd.DataFrame, p: int = PIVOT) -> List[Tuple[int, int, float]]:
    ph, pl = confirmed_pivots(df, p)
    events = []
    last_high = None
    last_low = None
    last_high_idx = None
    broken = False
    for i in range(len(df)):
        if i >= p:
            pivot_i = i - p
            if not np.isnan(ph[pivot_i]):
                last_high = ph[pivot_i]
                last_high_idx = pivot_i
            if not np.isnan(pl[pivot_i]):
                last_low = pl[pivot_i]
        if last_low is not None and last_high_idx is not None and df.iloc[i].close < last_low:
            if not broken:
                events.append((i, last_high_idx, float(last_low)))
                broken = True
        if i >= p and not np.isnan(pl[i - p]):
            broken = False
    return events


def htf_bullish_regime(htf: pd.DataFrame) -> pd.Series:
    """Causal HTF regime: the latest two confirmed HTF BOS events must be bullish.

    This is a strict mechanical translation of the PDF's 'at least two bullish BOS'
    direction rule. The value is only available after the corresponding 4H candle closes.
    """
    ph, pl = confirmed_pivots(htf, PIVOT)
    regime = np.zeros(len(htf), dtype=bool)
    last_high = None
    last_low = None
    bos_dirs: List[int] = []
    last_bos_idx = -10**9

    for i in range(len(htf)):
        if i >= PIVOT:
            pi = i - PIVOT
            if not np.isnan(ph[pi]):
                last_high = ph[pi]
            if not np.isnan(pl[pi]):
                last_low = pl[pi]

        # A bullish BOS is confirmed on the close of i.
        if last_high is not None and htf.iloc[i].close > last_high:
            if i != last_bos_idx:
                bos_dirs.append(1)
                bos_dirs = bos_dirs[-4:]
                last_bos_idx = i
        elif last_low is not None and htf.iloc[i].close < last_low:
            if i != last_bos_idx:
                bos_dirs.append(-1)
                bos_dirs = bos_dirs[-4:]
                last_bos_idx = i

        if sum(d == 1 for d in bos_dirs[-2:]) >= 2 and len(bos_dirs) >= 2:
            regime[i] = True

    # Only closed HTF candles can be used; caller maps by close_time.
    return pd.Series(regime, index=htf.index)


# =========================
# BINANCE DATA + CAUSAL STRUCTURE HELPERS
# =========================

def find_ob_after_bos_in_valley(df: pd.DataFrame, bos_idx: int, side: str, max_bars: int = 50) -> Optional[Tuple[int, int, Tuple[float, float]]]:
    """Find the post-BOS valley, its OB, and the immediate IFC.

    PDF sequence is BOS -> decline into the BOS valley -> OB -> immediately after it IFC.
    V1 translates this causally as the first confirmed swing valley after BOS followed by
    an OB candle whose next two candles form the corresponding 3-candle imbalance.
    """
    ph, pl = confirmed_pivots(df, PIVOT)
    end = min(len(df) - PIVOT - 1, bos_idx + max_bars)
    for valley_idx in range(bos_idx + PIVOT, end):
        pivot = pl[valley_idx] if side == "LONG" else ph[valley_idx]
        if np.isnan(pivot):
            continue
        # The valley/peak must be formed after the BOS.
        search_start = max(bos_idx + 1, valley_idx - 8)
        search_end = min(valley_idx + 4, len(df) - 3)
        for k in range(search_start, search_end + 1):
            if side == "LONG":
                if not is_bearish(df.iloc[k]):
                    continue
                zone = fvg_bullish(df, k + 2)
            else:
                if not is_bullish(df.iloc[k]):
                    continue
                zone = fvg_bearish(df, k + 2)
            if zone is None:
                continue
            # OB must be in/very near the structural valley.
            if side == "LONG" and df.iloc[k].low > float(pivot) + max(abs(float(pivot)) * 0.002, 1e-12):
                continue
            if side == "SHORT" and df.iloc[k].high < float(pivot) - max(abs(float(pivot)) * 0.002, 1e-12):
                continue
            return valley_idx, k, zone
    return None

def candle_zone(df: pd.DataFrame, idx: int, side: str) -> Tuple[float, float]:
    c = df.iloc[idx]
    # OB zone is the full candle range. This is explicit and avoids inventing a body-only rule.
    return float(c.low), float(c.high)


def find_aligned_liquidity(df: pd.DataFrame, correction_low_idx: int, start_idx: int, end_idx: int,
                           atr_s: pd.Series) -> Optional[Tuple[int, int]]:
    """Find two swing lows aligned within ATR tolerance, first around correction then later.

    The first valley is the correction valley; the second is formed during the subsequent decline.
    """
    ph, pl = confirmed_pivots(df, PIVOT)
    first_candidates = [i for i in range(max(PIVOT, correction_low_idx - 4), min(end_idx, correction_low_idx + 5)) if not np.isnan(pl[i])]
    if not first_candidates:
        first = correction_low_idx
    else:
        first = min(first_candidates, key=lambda i: abs(i - correction_low_idx))

    for j in range(max(first + PIVOT + 1, start_idx), end_idx):
        if np.isnan(pl[j]):
            continue
        a = atr_s.iloc[j]
        if not np.isfinite(a):
            continue
        if abs(float(pl[j] - pl[first])) <= LIQUIDITY_TOLERANCE_ATR * float(a):
            return first, j
    return None


def detect_setup_long(df: pd.DataFrame, bos_idx: int, bos_valley_idx: int, atr_s: pd.Series) -> Optional[SetupCandidate]:
    if bos_idx <= bos_valley_idx + 4:
        return None

    initial = find_ob_after_bos_in_valley(df, bos_idx, "LONG")
    if initial is None:
        return None
    bos_valley_idx, ob_idx, ifc = initial
    ifc_low, ifc_high = ifc
    ifc_idx = ob_idx + 2
    if ifc_idx >= len(df) - 5:
        return None

    # Small correction: at least two consecutive bearish candles, and must not touch IFC.
    correction_end = None
    for i in range(ifc_idx + 1, min(len(df), ifc_idx + 20)):
        if i + 1 >= len(df):
            break
        if is_bearish(df.iloc[i]) and is_bearish(df.iloc[i + 1]):
            # Correction must remain strictly above IFC's upper edge for long.
            if df.iloc[i].low > ifc_high and df.iloc[i + 1].low > ifc_high:
                correction_end = i + 1
                break
    if correction_end is None:
        return None

    # Price then breaks/BOS and starts declining. For the bullish source pattern this means
    # a local high followed by a decline into the liquidity sweep.
    ph, pl = confirmed_pivots(df, PIVOT)
    correction_low = int(np.argmin(df["low"].to_numpy()[ifc_idx:correction_end + 1])) + ifc_idx
    search_end = min(len(df) - 1, correction_end + MAX_SETUP_BARS)
    liq = find_aligned_liquidity(df, correction_low, correction_end + 1, search_end, atr_s)
    if liq is None:
        return None
    liq1, liq2 = liq

    # Liquidity sweep: price trades below the aligned zone after liq2.
    zone_low = min(float(df.iloc[liq1].low), float(df.iloc[liq2].low))
    sweep_idx = None
    for i in range(liq2 + 1, min(search_end, liq2 + MAX_LIQUIDITY_DISTANCE_BARS)):
        if df.iloc[i].low < zone_low:
            sweep_idx = i
            break
    if sweep_idx is None:
        return None

    # Before reaching IFC, form a valley and then a minor CHOCH.
    # Here "before reaching IFC" is interpreted in source geometry: the return leg's valley/OB
    # is above the IFC zone, then price breaks the prior local peak by wick.
    pre_ifc_end = sweep_idx + MAX_SETUP_BARS
    valley_candidates = []
    for i in range(sweep_idx + PIVOT, min(len(df) - PIVOT, pre_ifc_end)):
        if not np.isnan(pl[i]) and df.iloc[i].low > ifc_high:
            valley_candidates.append(i)
    if not valley_candidates:
        return None
    minor_valley = valley_candidates[-1]

    # Minor CHOCH: wick break of the most recent confirmed swing high before the valley.
    prior_highs = [i for i in range(max(PIVOT, minor_valley - 40), minor_valley) if not np.isnan(ph[i])]
    if not prior_highs:
        return None
    prior_peak_idx = prior_highs[-1]
    prior_peak = float(ph[prior_peak_idx])
    minor_choch = None
    for i in range(minor_valley + PIVOT, min(len(df), minor_valley + 30)):
        if df.iloc[i].high > prior_peak:
            minor_choch = i
            break
    if minor_choch is None:
        return None

    # Entry OB is the valley that crossed liquidity, formed before IFC, and caused minor CHOCH.
    # Mechanical translation: last bearish candle inside the minor-valley base before CHOCH,
    # while staying above IFC.
    entry_ob_idx = None
    for i in range(minor_choch - 1, minor_valley - 1, -1):
        c = df.iloc[i]
        if is_bearish(c) and c.low > ifc_high:
            entry_ob_idx = i
            break
    if entry_ob_idx is None:
        return None
    ob_low, ob_high = candle_zone(df, entry_ob_idx, "LONG")
    if ob_low <= ifc_high:
        return None

    # Structural target = highest confirmed peak after the original BOS and before entry.
    target_peaks = [float(ph[i]) for i in range(bos_idx + PIVOT, minor_choch) if not np.isnan(ph[i])]
    if not target_peaks:
        return None
    structural_target = max(target_peaks)

    # Confirmation is evaluated when price touches the OB. We require a confirmation candle
    # after the first touch, and entry occurs at the next candle open.
    confirmation_idx = None
    for i in range(minor_choch + 1, min(len(df) - 1, minor_choch + 40)):
        if df.iloc[i].low <= ob_high and df.iloc[i].high >= ob_low:
            if bullish_confirmation(df, i):
                confirmation_idx = i
                break
    if confirmation_idx is None:
        return None

    return SetupCandidate(
        side="LONG", bos_idx=bos_idx, ob_idx=ob_idx, ifc_idx=ifc_idx,
        ifc_low=ifc_low, ifc_high=ifc_high, correction_end_idx=correction_end,
        liquidity_1_idx=liq1, liquidity_2_idx=liq2, sweep_idx=sweep_idx,
        minor_choch_idx=minor_choch, entry_ob_idx=entry_ob_idx,
        entry_ob_low=ob_low, entry_ob_high=ob_high,
        structural_target=structural_target, confirmation_idx=confirmation_idx,
    )


def detect_setup_short(df: pd.DataFrame, bos_idx: int, bos_peak_idx: int, atr_s: pd.Series) -> Optional[SetupCandidate]:
    # Exact inverse of the long sequence.
    if bos_idx <= bos_peak_idx + 4:
        return None

    initial = find_ob_after_bos_in_valley(df, bos_idx, "SHORT")
    if initial is None:
        return None
    bos_peak_idx, ob_idx, ifc = initial
    ifc_low, ifc_high = ifc
    ifc_idx = ob_idx + 2
    if ifc_idx >= len(df) - 5:
        return None

    correction_end = None
    for i in range(ifc_idx + 1, min(len(df), ifc_idx + 20)):
        if i + 1 >= len(df):
            break
        if is_bullish(df.iloc[i]) and is_bullish(df.iloc[i + 1]):
            if df.iloc[i].high < ifc_low and df.iloc[i + 1].high < ifc_low:
                correction_end = i + 1
                break
    if correction_end is None:
        return None

    ph, pl = confirmed_pivots(df, PIVOT)
    correction_peak = int(np.argmax(df["high"].to_numpy()[ifc_idx:correction_end + 1])) + ifc_idx
    search_end = min(len(df) - 1, correction_end + MAX_SETUP_BARS)
    # Mirror liquidity using swing highs.
    first_candidates = [i for i in range(max(PIVOT, correction_peak - 4), min(search_end, correction_peak + 5)) if not np.isnan(ph[i])]
    first = min(first_candidates, key=lambda i: abs(i - correction_peak)) if first_candidates else correction_peak
    liq = None
    for j in range(max(first + PIVOT + 1, correction_end + 1), search_end):
        if np.isnan(ph[j]) or not np.isfinite(atr_s.iloc[j]):
            continue
        if abs(float(ph[j] - ph[first])) <= LIQUIDITY_TOLERANCE_ATR * float(atr_s.iloc[j]):
            liq = (first, j)
            break
    if liq is None:
        return None
    liq1, liq2 = liq

    zone_high = max(float(df.iloc[liq1].high), float(df.iloc[liq2].high))
    sweep_idx = None
    for i in range(liq2 + 1, min(search_end, liq2 + MAX_LIQUIDITY_DISTANCE_BARS)):
        if df.iloc[i].high > zone_high:
            sweep_idx = i
            break
    if sweep_idx is None:
        return None

    valley_candidates = []
    for i in range(sweep_idx + PIVOT, min(len(df) - PIVOT, sweep_idx + MAX_SETUP_BARS)):
        if not np.isnan(ph[i]) and df.iloc[i].high < ifc_low:
            valley_candidates.append(i)
    if not valley_candidates:
        return None
    minor_peak = valley_candidates[-1]

    prior_lows = [i for i in range(max(PIVOT, minor_peak - 40), minor_peak) if not np.isnan(pl[i])]
    if not prior_lows:
        return None
    prior_valley_idx = prior_lows[-1]
    prior_valley = float(pl[prior_valley_idx])
    minor_choch = None
    for i in range(minor_peak + PIVOT, min(len(df), minor_peak + 30)):
        if df.iloc[i].low < prior_valley:
            minor_choch = i
            break
    if minor_choch is None:
        return None

    entry_ob_idx = None
    for i in range(minor_choch - 1, minor_peak - 1, -1):
        c = df.iloc[i]
        if is_bullish(c) and c.high < ifc_low:
            entry_ob_idx = i
            break
    if entry_ob_idx is None:
        return None
    ob_low, ob_high = candle_zone(df, entry_ob_idx, "SHORT")
    if ob_high >= ifc_low:
        return None

    target_lows = [float(pl[i]) for i in range(bos_idx + PIVOT, minor_choch) if not np.isnan(pl[i])]
    if not target_lows:
        return None
    structural_target = min(target_lows)

    confirmation_idx = None
    for i in range(minor_choch + 1, min(len(df) - 1, minor_choch + 40)):
        if df.iloc[i].high >= ob_low and df.iloc[i].low <= ob_high:
            if bearish_confirmation(df, i):
                confirmation_idx = i
                break
    if confirmation_idx is None:
        return None

    return SetupCandidate(
        side="SHORT", bos_idx=bos_idx, ob_idx=ob_idx, ifc_idx=ifc_idx,
        ifc_low=ifc_low, ifc_high=ifc_high, correction_end_idx=correction_end,
        liquidity_1_idx=liq1, liquidity_2_idx=liq2, sweep_idx=sweep_idx,
        minor_choch_idx=minor_choch, entry_ob_idx=entry_ob_idx,
        entry_ob_low=ob_low, entry_ob_high=ob_high,
        structural_target=structural_target, confirmation_idx=confirmation_idx,
    )


# =========================
# TRADE ENGINE
# =========================

def apply_entry_slippage(price: float, side: str) -> float:
    return price * (1.0 + SLIPPAGE) if side == "LONG" else price * (1.0 - SLIPPAGE)


def apply_exit_slippage(price: float, side: str) -> float:
    # Adverse slippage at exit.
    return price * (1.0 - SLIPPAGE) if side == "LONG" else price * (1.0 + SLIPPAGE)


def trade_pnl(entry: float, exit_price: float, side: str) -> float:
    raw = (exit_price - entry) if side == "LONG" else (entry - exit_price)
    gross = NOTIONAL * (raw / entry)
    entry_fee = NOTIONAL * FEE_RATE
    exit_notional = NOTIONAL * (exit_price / entry)
    exit_fee = exit_notional * FEE_RATE
    return gross - entry_fee - exit_fee


def backtest_symbol(symbol: str, exec_df: pd.DataFrame, htf_df: pd.DataFrame) -> List[Trade]:
    """Causal Setup 7 implementation.

    SHORT: established bullish HTF structure (>=2 recent bullish BOS) touches a
    confirmed 4H supply zone; execution TF forms a lower high, then a sharp bearish
    close breaks the last confirmed swing low (major CHOCH). The last bullish candle
    before that impulse is the OB. Entry is a resting-limit touch after CHOCH.
    LONG is the exact inverse using HTF demand and bullish CHOCH.

    All zones become usable only after their source candle/pivot is confirmed.
    Numerical thresholds are explicitly a frozen mechanical translation, not literal
    numeric rules supplied by the PDF.
    """
    ex = exec_df.copy().reset_index(drop=True)
    ht = htf_df.copy().reset_index(drop=True)
    n = len(ex)
    if n < 100 or len(ht) < 20:
        return []
    ex_atr = atr(ex).to_numpy(float)
    ph, pl = confirmed_pivots(ex, PIVOT)
    hph, hpl = confirmed_pivots(ht, PIVOT)
    htf_bull = htf_bullish_regime(ht).to_numpy(bool)

    # Map the latest *confirmed* HTF pivot high/low zone and regime to each execution bar.
    # A pivot at HTF index k is not available until k+PIVOT has closed.
    zone_rows = []
    last_hi_zone = None
    last_lo_zone = None
    for i in range(len(ht)):
        if i >= PIVOT:
            k = i - PIVOT
            if not np.isnan(hph[k]):
                last_hi_zone = (ht.iloc[k].low, ht.iloc[k].high)
            if not np.isnan(hpl[k]):
                last_lo_zone = (ht.iloc[k].low, ht.iloc[k].high)
        zone_rows.append({
            "available_at": ht.iloc[i].close_time,
            "htf_bull": bool(htf_bull[i]),
            "supply_low": last_hi_zone[0] if last_hi_zone else np.nan,
            "supply_high": last_hi_zone[1] if last_hi_zone else np.nan,
            "demand_low": last_lo_zone[0] if last_lo_zone else np.nan,
            "demand_high": last_lo_zone[1] if last_lo_zone else np.nan,
        })
    mapped = pd.merge_asof(
        ex[["open_time"]].sort_values("open_time"),
        pd.DataFrame(zone_rows).sort_values("available_at"),
        left_on="open_time", right_on="available_at", direction="backward",
    ).reset_index(drop=True)
    ex["htf_bull"] = mapped["htf_bull"].fillna(False).to_numpy(bool)
    for c in ["supply_low", "supply_high", "demand_low", "demand_high"]:
        ex[c] = mapped[c].to_numpy(float)

    trades: List[Trade] = []
    occupied_until = -1
    # confirmed swing points are only acted on at i, where pivot center i-PIVOT is known.
    last_pivot_high = None
    last_pivot_high_idx = None
    last_pivot_low = None
    last_pivot_low_idx = None
    recent_highs = []
    recent_lows = []
    pending = None
    last_htf_touch_side = None
    touch_idx = -1

    for i in range(PIVOT, n):
        center = i - PIVOT
        if not np.isnan(ph[center]):
            last_pivot_high = float(ph[center]); last_pivot_high_idx = center
            recent_highs.append((center, float(ph[center])))
            recent_highs = recent_highs[-8:]
        if not np.isnan(pl[center]):
            last_pivot_low = float(pl[center]); last_pivot_low_idx = center
            recent_lows.append((center, float(pl[center])))
            recent_lows = recent_lows[-8:]

        row = ex.iloc[i]
        atr_i = ex_atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # Arm only after a confirmed higher-timeframe zone is actually touched.
        if bool(row.htf_bull) and np.isfinite(row.supply_low) and row.high >= row.supply_low and row.low <= row.supply_high:
            if last_htf_touch_side != "SHORT":
                last_htf_touch_side = "SHORT"; touch_idx = i; pending = None
        if (not bool(row.htf_bull)) and np.isfinite(row.demand_low) and row.low <= row.demand_high and row.high >= row.demand_low:
            if last_htf_touch_side != "LONG":
                last_htf_touch_side = "LONG"; touch_idx = i; pending = None

        # Expire an untriggered setup after a fixed, pre-registered 120 execution bars.
        if last_htf_touch_side and i - touch_idx > MAX_SETUP_BARS:
            last_htf_touch_side = None; pending = None

        # After HTF supply touch, a lower high followed by a sharp close below the last
        # confirmed swing low is the bearish major CHOCH. Inverse for LONG.
        if pending is None and last_htf_touch_side == "SHORT" and i > touch_idx + PIVOT and last_pivot_low is not None:
            has_lh = len(recent_highs) >= 2 and recent_highs[-1][1] < recent_highs[-2][1]
            sharp = (row.close < last_pivot_low and row.close < row.open and
                     (row.high - row.low) >= 1.0 * atr_i and
                     (row.open - row.close) >= 0.45 * (row.high - row.low))
            if has_lh and sharp:
                ob_idx = None
                for k in range(i - 1, max(touch_idx, i - 13) - 1, -1):
                    if ex.iloc[k].close > ex.iloc[k].open:
                        ob_idx = k; break
                if ob_idx is not None:
                    pending = {"side":"SHORT", "signal_idx":i, "ob_idx":ob_idx,
                               "ob_low":float(ex.iloc[ob_idx].low), "ob_high":float(ex.iloc[ob_idx].high),
                               "target":None, "expires":min(n-1, i+MAX_SETUP_BARS)}
                    last_htf_touch_side = None

        elif pending is None and last_htf_touch_side == "LONG" and i > touch_idx + PIVOT and last_pivot_high is not None:
            has_hl = len(recent_lows) >= 2 and recent_lows[-1][1] > recent_lows[-2][1]
            sharp = (row.close > last_pivot_high and row.close > row.open and
                     (row.high - row.low) >= 1.0 * atr_i and
                     (row.close - row.open) >= 0.45 * (row.high - row.low))
            if has_hl and sharp:
                ob_idx = None
                for k in range(i - 1, max(touch_idx, i - 13) - 1, -1):
                    if ex.iloc[k].close < ex.iloc[k].open:
                        ob_idx = k; break
                if ob_idx is not None:
                    pending = {"side":"LONG", "signal_idx":i, "ob_idx":ob_idx,
                               "ob_low":float(ex.iloc[ob_idx].low), "ob_high":float(ex.iloc[ob_idx].high),
                               "target":None, "expires":min(n-1, i+MAX_SETUP_BARS)}
                    last_htf_touch_side = None

        # Target OB is identified only when the first post-CHOCH swing is actually
        # confirmed in real time. Never inspect future candles to precompute the target.
        if pending is not None and pending["target"] is None and center > pending["signal_idx"]:
            if pending["side"] == "SHORT" and not np.isnan(pl[center]):
                for k in range(center - 1, max(pending["signal_idx"], center - 13) - 1, -1):
                    if ex.iloc[k].close < ex.iloc[k].open:
                        pending["target"] = float(ex.iloc[k].high); pending["target_known_idx"] = i; break
            elif pending["side"] == "LONG" and not np.isnan(ph[center]):
                for k in range(center - 1, max(pending["signal_idx"], center - 13) - 1, -1):
                    if ex.iloc[k].close > ex.iloc[k].open:
                        pending["target"] = float(ex.iloc[k].low); pending["target_known_idx"] = i; break

        if pending is None or pending["target"] is None or i <= pending.get("target_known_idx", i) or i <= pending["signal_idx"] or i > pending["expires"]:
            continue
        if i <= occupied_until:
            continue

        side = pending["side"]
        ob_low, ob_high = pending["ob_low"], pending["ob_high"]
        # Resting limit order is active only after CHOCH/target confirmation.
        # Fill at the OB edge when this later candle touches it. Because OHLC cannot
        # establish intrabar order, NEVER evaluate SL/TP on the entry candle; begin
        # exit checks on the next candle (project integrity rule).
        if side == "SHORT":
            # A valid retest requires actual OHLC intersection with the OB zone.
            if row.high < ob_low or row.low > ob_high:
                continue
            entry = apply_entry_slippage(ob_low, side)
            stop = ob_high
            risk = stop - entry
            target_2r = entry - RR * risk
            structural = pending["target"]
            if structural >= entry or target_2r < structural:
                pending = None; continue
            target = target_2r
        else:
            # A valid retest requires actual OHLC intersection with the OB zone.
            if row.low > ob_high or row.high < ob_low:
                continue
            entry = apply_entry_slippage(ob_high, side)
            stop = ob_low
            risk = entry - stop
            target_2r = entry + RR * risk
            structural = pending["target"]
            if structural <= entry or target_2r > structural:
                pending = None; continue
            target = target_2r

        risk_pct = risk / entry if entry > 0 else math.inf
        if risk <= 0 or risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:
            pending = None; continue

        exit_idx = None; exit_raw = None; outcome = None
        for j in range(i + 1, n):
            hi = float(ex.iloc[j].high); lo = float(ex.iloc[j].low)
            hit_sl = hi >= stop if side == "SHORT" else lo <= stop
            hit_tp = lo <= target if side == "SHORT" else hi >= target
            if hit_sl and hit_tp:
                exit_idx=j; exit_raw=stop; outcome="SL"; break
            if hit_sl:
                exit_idx=j; exit_raw=stop; outcome="SL"; break
            if hit_tp:
                exit_idx=j; exit_raw=target; outcome="TP"; break
        if exit_idx is None:
            pending = None; continue
        exit_price = apply_exit_slippage(float(exit_raw), side)
        pnl = trade_pnl(entry, exit_price, side)
        r = pnl / (NOTIONAL * risk_pct) if risk_pct > 0 else 0.0
        trades.append(Trade(
            symbol=symbol, side=side, signal_idx=pending["signal_idx"], entry_idx=i, exit_idx=exit_idx,
            entry_time=ex.iloc[i].open_time, exit_time=ex.iloc[exit_idx].open_time,
            entry=entry, stop=stop, target=target, exit=exit_price, outcome=outcome,
            pnl=pnl, r_multiple=r, setup_bos_idx=pending["signal_idx"],
            minor_choch_idx=pending["signal_idx"], entry_ob_idx=pending["ob_idx"],
        ))
        occupied_until = exit_idx
        pending = None
    return trades

def max_loss_streak(trades: List[Trade]) -> int:
    best = cur = 0
    for t in sorted(trades, key=lambda x: x.exit_time):
        if t.outcome == "SL":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0,
                "final_equity": INITIAL_CAPITAL, "profit_factor": 0.0, "max_loss_streak": 0}
    wins = [t for t in trades if t.outcome == "TP"]
    losses = [t for t in trades if t.outcome == "SL"]
    gross_win = sum(max(t.pnl, 0) for t in trades)
    gross_loss = -sum(min(t.pnl, 0) for t in trades)
    net = sum(t.pnl for t in trades)
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": 100.0 * len(wins) / len(trades), "net_pnl": net,
        "final_equity": INITIAL_CAPITAL + net,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "max_loss_streak": max_loss_streak(trades),
    }


def enforce_portfolio_cap(trades: List[Trade], max_positions: int = 10) -> List[Trade]:
    """Chronologically enforce max 10 simultaneous portfolio positions."""
    accepted: List[Trade] = []
    open_trades: List[Trade] = []
    for t in sorted(trades, key=lambda x: (x.entry_time, x.exit_time, x.symbol)):
        open_trades = [o for o in open_trades if o.exit_time > t.entry_time]
        if len(open_trades) >= max_positions:
            continue
        # Defensive portfolio-level symbol lock.
        if any(o.symbol == t.symbol for o in open_trades):
            continue
        accepted.append(t)
        open_trades.append(t)
    return accepted


def save_trades(trades: List[Trade], path: str) -> None:
    rows = []
    for t in trades:
        rows.append(t.__dict__)
    pd.DataFrame(rows).to_csv(path, index=False)


def run() -> int:
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
    test_start = end - pd.Timedelta(days=TEST_DAYS)
    data_start = test_start - pd.Timedelta(days=WARMUP_DAYS)

    print("=" * 78)
    print("SETUP 7 V1 — HTF SUPPLY/DEMAND / MAJOR CHOCH / OB RETEST")
    print(f"Period: {test_start} -> {end}")
    print(f"Execution: {INTERVAL_EXEC} | HTF: {INTERVAL_HTF}")
    print(f"Capital=${INITIAL_CAPITAL:.2f} | margin=${MARGIN_PER_TRADE:.2f} | leverage={LEVERAGE:.0f}x | RR=1:{RR:g}")
    print("Integrity: causal confirmation, per-symbol lock, max 10 portfolio positions, RR 1:2")
    print("=" * 78)

    all_trades: List[Trade] = []
    failures = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_symbol(symbol: str):
        print(f"\n[{symbol}] downloading 15m + 4h in parallel...")
        ex = download_klines(symbol, INTERVAL_EXEC, data_start, end)
        ht = download_klines(symbol, INTERVAL_HTF, data_start - pd.Timedelta(days=7), end)
        validate_ohlcv(ex, 15, symbol)
        validate_ohlcv(ht, 240, symbol)
        trades = backtest_symbol(symbol, ex, ht)
        trades = [t for t in trades if t.entry_time >= test_start and t.entry_time <= end]
        return symbol, ex, ht, trades

    with ThreadPoolExecutor(max_workers=SYMBOL_WORKERS) as pool:
        futures = {pool.submit(process_symbol, symbol): symbol for symbol in SYMBOLS}
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                symbol, ex, ht, trades = fut.result()
                all_trades.extend(trades)
                print(f"[{symbol}] rows15={len(ex):,} rows4h={len(ht):,} trades={len(trades)}")
            except Exception as e:
                failures.append((symbol, str(e)))
                print(f"[{symbol}] ERROR: {e}")

    all_trades = enforce_portfolio_cap(all_trades, max_positions=10)
    all_trades.sort(key=lambda x: x.exit_time)
    stats = summarize(all_trades)
    save_trades(all_trades, OUTPUT_CSV)

    print("\n" + "=" * 78)
    print("RESULT")
    print(f"Trades        : {stats['trades']}")
    print(f"Wins / Losses : {stats['wins']} / {stats['losses']}")
    print(f"Win rate      : {stats['win_rate']:.2f}%")
    print(f"Net PnL       : ${stats['net_pnl']:.2f}")
    print(f"Final equity  : ${stats['final_equity']:.2f}")
    print(f"Profit factor : {stats['profit_factor']:.3f}")
    print(f"Max loss streak: {stats['max_loss_streak']}")
    print(f"Trades CSV    : {OUTPUT_CSV}")
    if failures:
        print("\nDATA FAILURES:")
        for s, e in failures:
            print(f"- {s}: {e}")
    print("=" * 78)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
