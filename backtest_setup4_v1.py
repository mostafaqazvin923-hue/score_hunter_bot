# SETUP 4 V1
# Higher-TF Zone -> Reaction -> OB -> Aligned Liquidity
# -> Liquidity Break -> Return to OB
#
# Source: 10 ستاپ برتر.pdf — Setup 4, pages 23-27
#
# Mechanical translations (NOT literal numerical rules from PDF):
# 4H higher-timeframe zone
# 15m execution / structure
# PIVOT=2
# HTF zone = confirmed 4H swing +/- 0.5 ATR
# short-term reaction <= 24 x 15m candles
# aligned liquidity tolerance = 0.4%
# liquidity must form within 96 x 15m candles from OB
#
# Research rules:
# - Fixed RR 1:2
# - $100 margin
# - 50x leverage
# - $5,000 notional
# - max 1 open trade per symbol
# - different symbols may overlap
# - no same-candle re-entry
# - no timeout exit
# - no synthetic data
# - no forward fill
# - causal pivots
# - OOS untouched by tuning

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "APTUSDT",
    "CRVUSDT",
    "ONDOUSDT",
    "PENDLEUSDT",
    "ICPUSDT",
    "WIFUSDT",
]

BASE_INTERVAL = "15m"
HTF_INTERVAL = "4h"

# Fresh OOS range.
# Not used by the previously tested Setup 2 / Setup 3 OOS windows.
OOS_START = pd.Timestamp("2023-10-04 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2024-10-03 23:00:00", tz="UTC")

WARMUP_DAYS = 90

RESEARCH_START = OOS_START - pd.Timedelta(days=365)
RESEARCH_END = OOS_START - pd.Timedelta(minutes=15)

RESEARCH_SPAN = RESEARCH_END - RESEARCH_START
DISCOVERY_END = RESEARCH_START + RESEARCH_SPAN * 0.60
DEVELOPMENT_END = RESEARCH_START + RESEARCH_SPAN * 0.80

WARMUP_START = RESEARCH_START - pd.Timedelta(days=WARMUP_DAYS)

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

PIVOT = 2

HTF_ATR_PERIOD = 14
HTF_ZONE_ATR_MULT = 0.50

# Mechanical translation of "short-term movement".
MAX_REACTION_BARS = 24

# Mechanical translation of "aligned highs/lows".
ALIGN_TOL = 0.004

# Mechanical translation of:
# "liquidity area should not be far from the OB."
MAX_LIQUIDITY_DISTANCE_BARS = 96

SL_BUFFER_PCT = 0.0005


SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "setup4-v1-research/1.0"
})


# ============================================================
# DATA
# ============================================================

def fetch_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=60)

    if r.status_code != 200:
        raise requests.HTTPError(
            f"{r.status_code}: {url}",
            response=r
        )

    return r.content


def month_strings(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    a = (
        start
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    b = (
        end
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    result = []

    while a <= b:
        result.append(str(a))
        a += 1

    return result


def read_archive(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        csv_files = [
            name for name in z.namelist()
            if name.endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError("ZIP contains no CSV")

        with z.open(csv_files[0]) as f:
            raw = pd.read_csv(f, header=None)

    if raw.shape[1] < 6:
        raise RuntimeError("Unexpected Binance kline schema")

    raw = raw.iloc[:, :6].copy()

    raw.columns = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    raw["ts"] = pd.to_datetime(
        raw["ts"],
        unit="ms",
        utc=True
    )

    raw = raw.set_index("ts")

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        raw[col] = pd.to_numeric(
            raw[col],
            errors="coerce"
        )

    raw = raw.dropna()

    return raw


def fetch_symbol(symbol: str) -> pd.DataFrame:
    frames = []

    first_available_seen = False

    for month in month_strings(
        WARMUP_START,
        OOS_END
    ):
        url = (
            "https://data.binance.vision/data/futures/um/monthly/"
            f"klines/{symbol}/{BASE_INTERVAL}/"
            f"{symbol}-{BASE_INTERVAL}-{month}.zip"
        )

        try:
            frames.append(
                read_archive(
                    fetch_bytes(url)
                )
            )

            first_available_seen = True

        except requests.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            # Before listing: allowed.
            if status == 404 and not first_available_seen:
                continue

            # After first available archive:
            # another missing month is a fatal data gap.
            if status == 404:
                raise RuntimeError(
                    f"Historical archive gap after first "
                    f"available month: {symbol} {month}"
                ) from exc

            raise

    if not frames:
        raise RuntimeError(
            f"No Binance Futures history available for {symbol}"
        )

    df = pd.concat(frames)

    df = (
        df
        .sort_index()
        .loc[~df.index.duplicated(keep="first")]
    )

    expected = pd.Timedelta(minutes=15)

    gaps = df.index.to_series().diff().dropna()

    bad = gaps[gaps != expected]

    if not bad.empty:
        raise RuntimeError(
            f"Fatal futures gap for {symbol}: "
            f"{bad.iloc[0]} at {bad.index[0]}"
        )

    df = df.loc[
        (df.index >= WARMUP_START)
        & (df.index <= OOS_END)
    ]

    if df.empty:
        raise RuntimeError(
            f"Empty requested range for {symbol}"
        )

    if df.index.min() > WARMUP_START:
        raise RuntimeError(
            f"Insufficient warmup coverage for {symbol}"
        )

    if df.index.max() < OOS_END:
        raise RuntimeError(
            f"Insufficient OOS coverage for {symbol}"
        )

    return df


def resample_htf(df: pd.DataFrame) -> pd.DataFrame:
    htf = (
        df
        .resample(
            HTF_INTERVAL,
            label="right",
            closed="right"
        )
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )

    tr = pd.concat(
        [
            htf["high"] - htf["low"],
            (
                htf["high"]
                - htf["close"].shift()
            ).abs(),
            (
                htf["low"]
                - htf["close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    htf["atr"] = (
        tr
        .rolling(HTF_ATR_PERIOD)
        .mean()
    )

    return htf


# ============================================================
# CAUSAL PIVOTS
# ============================================================

def confirmed_pivots(
    df: pd.DataFrame,
    p: int = PIVOT
):
    n = len(df)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    pivot_high = np.zeros(
        n,
        dtype=bool
    )

    pivot_low = np.zeros(
        n,
        dtype=bool
    )

    for i in range(p, n - p):

        left_h = highs[i - p:i]
        right_h = highs[i + 1:i + p + 1]

        left_l = lows[i - p:i]
        right_l = lows[i + 1:i + p + 1]

        pivot_high[i] = (
            highs[i] >= left_h.max()
            and highs[i] > right_h.max()
        )

        pivot_low[i] = (
            lows[i] <= left_l.min()
            and lows[i] < right_l.min()
        )

    return pivot_high, pivot_low


# ============================================================
# CANDIDATE
# ============================================================

@dataclass
class Candidate:

    symbol: str
    side: str

    ob_idx: int
    trigger_idx: int
    entry_idx: int

    ob_high: float
    ob_low: float

    entry: float
    stop: float
    target: float

    structural_target: float
    liquidity_level: float

    zone_time: pd.Timestamp


# ============================================================
# HTF ZONES
# ============================================================

def build_htf_zones(
    df: pd.DataFrame
):
    htf = resample_htf(df)

    pivot_high, pivot_low = confirmed_pivots(htf)

    zones = []

    for i in range(
        PIVOT,
        len(htf) - PIVOT
    ):

        atr = htf["atr"].iloc[i]

        if not np.isfinite(atr):
            continue

        timestamp = htf.index[i]

        half_width = (
            HTF_ZONE_ATR_MULT * atr
        )

        # Confirmed 4H swing high
        # -> resistance zone -> bearish setup.
        if pivot_high[i]:

            center = htf["high"].iloc[i]

            zones.append({
                "side": "short",
                "time": timestamp,
                "lo": center - half_width,
                "hi": center + half_width,
            })

        # Confirmed 4H swing low
        # -> support zone -> bullish setup.
        if pivot_low[i]:

            center = htf["
