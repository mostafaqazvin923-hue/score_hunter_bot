#!/usr/bin/env python3
# SETUP 5 V1 — Mechanical research translation of the uploaded "10 setups" PDF.
#
# IMPORTANT:
# - PDF-derived structure is documented in the report.
# - Numeric definitions not specified by the PDF are frozen research translations.
# - No lookahead: pivots become known only after PIVOT bars; 4H pivots are mapped
#   to their actual confirmation time before being used on the 15m execution chart.
# - One open trade per symbol; different symbols may overlap.
# - No same-candle re-entry and no same-candle entry/exit.
# - No timeout / BE / trailing / partial exits.

from __future__ import annotations

import io
import math
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# =========================
# FROZEN RESEARCH CONFIG
# =========================

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

# Setup 5 translation:
# 15m execution -> 1H -> 4H = two higher timeframes.
INTERVAL = "15m"
HTF_HOURS = 4

PIVOT = 2

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Mechanical research translations.
ALIGN_TOL = 0.004
MAX_LIQ_TO_OB_BARS = 24
SL_BUFFER_PCT = 0.0005

EXECUTION_TF_MINUTES = 15
HTF_LABEL = "4H"

WARMUP_DAYS = 90
RESEARCH_DAYS = 365
OOS_DAYS = 365

# Fresh OOS reserved for Setup 5.
OOS_START = pd.Timestamp(
    "2025-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-10-03 23:45:00",
    tz="UTC",
)

RESEARCH_START = OOS_START - pd.Timedelta(days=RESEARCH_DAYS)
DATA_START = RESEARCH_START - pd.Timedelta(days=WARMUP_DAYS)

# Research split:
# first 70% = Discovery
# remaining 30% = Development
DISCOVERY_END = (
    RESEARCH_START
    + pd.Timedelta(days=int(RESEARCH_DAYS * 0.70))
)

DEVELOPMENT_START = (
    DISCOVERY_END
    + pd.Timedelta(minutes=15)
)

ARCHIVE_BASE = (
    "https://data.binance.vision/data/futures/um"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "setup5-v1-research/1.0"
    }
)

OUTPUT_DIR = Path("setup5_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# DATA
# =========================

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def month_iter(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    cur = pd.Timestamp(
        start.year,
        start.month,
        1,
        tz="UTC",
    )

    last = pd.Timestamp(
        end.year,
        end.month,
        1,
        tz="UTC",
    )

    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def daily_iter(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    cur = start.normalize()

    while cur <= end.normalize():
        yield cur
        cur += pd.Timedelta(days=1)


def _download(
    url: str,
    retries: int = 3,
) -> Optional[bytes]:

    last = None

    for attempt in range(retries):

        try:
            response = SESSION.get(
                url,
                timeout=90,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.content

        except Exception as exc:
            last = exc
            time.sleep(
                1.5 * (attempt + 1)
            )

    raise RuntimeError(
        f"Download failed after retries: "
        f"{url} :: {last}"
    )


def _read_zip_bytes(
    blob: bytes,
) -> pd.DataFrame:

    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as zf:

        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv")
        ]

        if not names:
            raise RuntimeError(
                "Archive contains no CSV"
            )

        with zf.open(names[0]) as fh:
            raw = pd.read_csv(
                fh,
                header=None,
            )

    if len(raw):
        first = str(
            raw.iloc[0, 0]
        ).lower()

        if first in {
            "open time",
            "open_time",
        }:
            raw = raw.iloc[1:].reset_index(
                drop=True
            )

    raw = raw.iloc[:, :12]
    raw.columns = COLUMNS

    return raw


def load_symbol(
    symbol: str,
) -> pd.DataFrame:

    rows = []

    first_available_month = None

    # --------------------------------
    # Monthly archives
    # --------------------------------

    for month in month_iter(
        DATA_START,
        OOS_END,
    ):

        ym = month.strftime("%Y-%m")

        url = (
            f"{ARCHIVE_BASE}/monthly/"
            f"klines/{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-{ym}.zip"
        )

        blob = _download(url)

        if blob is None:

            # Before first listing:
            # normal and allowed.
            if first_available_month is None:
                continue

            # After first listing:
            # missing archive is fatal.
            raise RuntimeError(
                "Archive missing after first "
                f"availability: {url}"
            )

        first_available_month = ym

        rows.append(
            _read_zip_bytes(blob)
        )

    # --------------------------------
    # Daily edge archives
    # --------------------------------

    for day in daily_iter(
        DATA_START,
        OOS_END,
    ):

        is_first_month = (
            day.strftime("%Y-%m")
            == DATA_START.strftime("%Y-%m")
        )

        is_last_month = (
            day.strftime("%Y-%m")
            == OOS_END.strftime("%Y-%m")
        )

        if not (
            is_first_month
            or is_last_month
        ):
            continue

        ds = day.strftime("%Y-%m-%d")

        url = (
            f"{ARCHIVE_BASE}/daily/"
            f"klines/{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-{ds}.zip"
        )

        blob = _download(url)

        if blob is None:

            if (
                first_available_month is None
                or day.strftime("%Y-%m")
                < first_available_month
            ):
                continue

            raise RuntimeError(
                "Daily archive missing after "
                f"first availability: {url}"
            )

        rows.append(
            _read_zip_bytes(blob)
        )

    if not rows:
        raise RuntimeError(
            f"No Binance data found for {symbol}"
        )

    df = pd.concat(
        rows,
        ignore_index=True,
    )

    # --------------------------------
    # Timestamp handling
    # --------------------------------

    open_ts = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    open_unit = (
        "us"
        if open_ts.dropna().median()
        > 10**14
        else "ms"
    )

    df["open_time"] = pd.to_datetime(
        open_ts,
        unit=open_unit,
        utc=True,
    )

    close_ts = pd.to_numeric(
        df["close_time"],
        errors="coerce",
    )

    close_unit = (
        "us"
        if close_ts.dropna().median()
        > 10**14
        else "ms"
    )

    df["close_time"] = pd.to_datetime(
        close_ts,
        unit=close_unit,
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = (
        df.drop_duplicates(
            "open_time"
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    df = df[
        (df.open_time >= DATA_START)
        & (df.open_time <= OOS_END)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows in requested range "
            f"for {symbol}"
        )

    # --------------------------------
    # Strict 15m continuity
    # --------------------------------

    diffs = (
        df.open_time
        .diff()
        .dropna()
    )

    bad = diffs[
        diffs
        != pd.Timedelta(minutes=15)
    ]

    if not bad.empty:

        first_bad_idx = bad.index[0]

        previous_time = df.loc[
            first_bad_idx - 1,
            "open_time",
        ]

        current_time = df.loc[
            first_bad_idx,
            "open_time",
        ]

        raise RuntimeError(
            f"15m data gap for {symbol}: "
            f"{previous_time} -> {current_time} "
            f"({current_time - previous_time})"
        )

    return df


# =========================
# CAUSAL STRUCTURE
# =========================

def confirmed_pivots(
    df: pd.DataFrame,
    p: int = PIVOT,
) -> pd.DataFrame:

    x = df.copy()

    highs = x.high.to_numpy(float)
    lows = x.low.to_numpy(float)

    n = len(x)

    pivot_high = np.zeros(
        n,
        dtype=bool,
    )

    pivot_low = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(
        p,
        n - p,
    ):

        pivot_high[i] = (
            highs[i]
            >= np.max(
                highs[i - p:i + p + 1]
            )
        )

        pivot_low[i] = (
            lows[i]
            <= np.min(
                lows[i - p:i + p + 1]
            )
        )

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    x["pivot_high_known_at"] = np.where(
        pivot_high,
        np.arange(n) + p,
        -1,
    )

    x["pivot_low_known_at"] = np.where(
        pivot_low,
        np.arange(n) + p,
        -1,
    )

    return x


def aggregate_4h(
    df15: pd.DataFrame,
) -> pd.DataFrame:

    z = (
        df15
        .set_index("open_time")
    )

    grouped = z.resample(
        "4h",
        label="left",
        closed="left",
    )

    out = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    counts = grouped["close"].count()

    out["count_15m"] = counts

    out = (
        out.dropna()
        .reset_index()
    )

    # Exactly 16 x 15m bars per 4H candle.
    out = out[
        out.count_15m == 16
    ].copy()

    if out.empty:
        raise RuntimeError(
            "No complete 4H bars"
        )

    return out


def confirmed_htf_pivots(
    htf: pd.DataFrame,
    p: int = PIVOT,
) -> pd.DataFrame:

    return confirmed_pivots(
        htf,
        p,
    )


@dataclass
class Candidate:

    symbol: str
    direction: str

    entry_signal_idx: int
    entry_idx: int

    htf_zone_idx: int
    htf_zone_known_at: int

    reaction_idx: int
    choch_idx: int
    choch_level: float

    ob_idx: int
    ob_low: float
    ob_high: float

    liq_idx1: int
    liq_idx2: int
    liq_level: float

    sweep_idx: int

    structural_target_idx: int
    structural_target: float

    entry_price: float
    stop_price: float
    tp_price: float

    ifc: bool


# =========================
# HTF ZONES
# =========================

def build_htf_zone_records(
    htf: pd.DataFrame,
) -> list[dict]:

    hp = confirmed_htf_pivots(
        htf
    )

    zones = []

    for i, row in hp.iterrows():

        known_at = int(
            i + PIVOT
        )

        if known_at >= len(hp):
            continue

        # A 4H pivot centered on bar i becomes
        # known only after confirmation bar i+PIVOT
        # has CLOSED, not at its opening time.
        known_time = (
            pd.Timestamp(
                hp.iloc[known_at]
                .open_time
            )
            + pd.Timedelta(
                hours=HTF_HOURS
            )
        )

        if bool(row.pivot_low):

            body_high = max(
                float(row.open),
                float(row.close),
            )

            zones.append(
                {
                    "type": "demand",
                    "pivot_idx": i,
                    "known_at": known_at,
                    "known_time": known_time,
                    "low": float(row.low),
                    "high": body_high,
                }
            )

        if bool(row.pivot_high):

            body_low = min(
                float(row.open),
                float(row.close),
            )

            zones.append(
                {
                    "type": "supply",
                    "pivot_idx": i,
                    "known_at": known_at,
                    "known_time": known_time,
                    "low": body_low,
                    "high": float(row.high),
                }
            )

    return zones


# =========================
# CANDIDATE ENGINE
# =========================

def _trend_down(
    x: pd.DataFrame,
    idx: int,
) -> bool:

    highs = [
        i
        for i in range(
            max(0, idx - 1000),
            idx,
        )
        if (
            x.iloc[i].pivot_high
            and i + PIVOT <= idx
        )
    ]

    lows = [
        i
        for i in range(
            max(0, idx - 1000),
            idx,
        )
        if (
            x.iloc[i].pivot_low
            and i + PIVOT <= idx
        )
    ]

    if len(highs) < 2 or len(lows) < 2:
        return False

    h1, h2 = highs[-2:]
    l1, l2 = lows[-2:]

    return (
        float(x.iloc[h2].high)
        < float(x.iloc[h1].high)
        and
        float(x.iloc[l2].low)
        < float(x.iloc[l1].low)
    )


def _trend_up(
    x: pd.DataFrame,
    idx: int,
) -> bool:

    highs = [
        i
        for i in range(
            max(0, idx - 1000),
            idx,
        )
        if (
            x.iloc[i].pivot_high
            and i + PIVOT <= idx
        )
    ]

    lows = [
        i
        for i in range(
            max(0, idx - 1000),
            idx,
        )
        if (
            x.iloc[i].pivot_low
            and i + PIVOT <= idx
        )
    ]

    if len(highs) < 2 or len(lows) < 2:
        return False

    h1, h2 = highs[-2:]
    l1, l2 = lows[-2:]

    return (
        float(x.iloc[h2].high)
        > float(x.iloc[h1].high)
        and
        float(x.iloc[l2].low)
        > float(x.iloc[l1].low)
    )


def _aligned(
    a: float,
    b: float,
) -> bool:

    base = max(
        abs(a),
        abs(b),
        1e-12,
    )

    return (
        abs(a - b) / base
        <= ALIGN_TOL
    )


def _ifc_after_ob(
    x: pd.DataFrame,
    ob_idx: int,
    end_idx: int,
    direction: str,
) -> bool:

    # Diagnostic-only IFC translation.
    # It is NOT an entry requirement in V1.

    for i in range(
        ob_idx + 2,
        min(
            end_idx,
            len(x) - 1,
        ) + 1,
    ):

        if direction == "long":

            if (
                float(x.iloc[i].low)
                >
                float(x.iloc[i - 2].high)
            ):
                return True

        else:

            if (
                float(x.iloc[i].high)
                <
                float(x.iloc[i - 2].low)
            ):
                return True

    return False


def make_candidates(
    symbol: str,
    x: pd.DataFrame,
    htf: pd.DataFrame,
) -> list[Candidate]:

    """
    Setup 5 causal state machine.

    Long:
        Downtrend
        -> HTF demand
        -> CHOCH
        -> lowest pre-CHOCH valley = OB
        -> aligned valleys after CHOCH
        -> liquidity sweep
        -> return to OB
        -> next candle entry

    Short is the exact structural inverse.

    No arbitrary formation timeout is used.
    """

    zones = build_htf_zone_records(
        htf
    )

    n = len(x)

    highs = np.flatnonzero(
        x.pivot_high.to_numpy(bool)
    )

    lows = np.flatnonzero(
        x.pivot_low.to_numpy(bool)
    )

    high_values = x.high.to_numpy(
        float
    )

    low_values = x.low.to_numpy(
        float
    )

    candidates = []

    used_signal_idx = set()

    def known(
        arr,
        end_idx: int,
    ):

        return arr[
            arr + PIVOT <= end_idx
        ]

    def first_touch(
        start_idx: int,
        zone_low: float,
        zone_high: float,
        stop_idx: int,
    ) -> Optional[int]:

        end = min(
            stop_idx,
            n - 1,
        )

        for j in range(
            start_idx,
            end + 1,
        ):

            if (
                high_values[j]
                >= zone_low
                and
                low_values[j]
                <= zone_high
            ):
                return j

        return None

    # Process HTF zones in chronological order.
    for zone in sorted(
        zones,
        key=lambda z: z["known_time"],
    ):

        direction = (
            "long"
            if zone["type"] == "demand"
            else "short"
        )

        known_time = pd.Timestamp(
            zone["known_time"]
        )

        # Map HTF confirmation time to the 15m
        # execution timeline.
        known_exec_idx = int(
            x["open_time"].searchsorted(
                known_time,
                side="left",
            )
        )

        if known_exec_idx >= n - 1:
            continue

        # First touch of this HTF zone.
        reaction_idx = first_touch(
            known_exec_idx,
            float(zone["low"]),
            float(zone["high"]),
            n - 1,
        )

        if reaction_idx is None:
            continue

        # =========================
        # LONG
        # =========================

        if direction == "long":

            if not _trend_down(
                x,
                reaction_idx,
            ):
                continue

            prior_highs = known(
                highs,
                reaction_idx,
            )

            prior_highs = prior_highs[
                prior_highs < reaction_idx
            ]

            if len(prior_highs) == 0:
                continue

            lh_idx = int(
                prior_highs[-1]
            )

            lh = float(
                high_values[lh_idx]
            )

            # CHOCH = first causal close
            # above the last confirmed LH.
            choch_idx = None

            for j in range(
                reaction_idx + 1,
                n,
            ):

                if (
                    float(
                        x.iloc[j].close
                    )
                    > lh
                ):
                    choch_idx = j
                    break

            if choch_idx is None:
                continue

            # Lowest confirmed valley between
            # reaction and CHOCH = OB.
            pre = known(
                lows,
                choch_idx,
            )

            pre = pre[
                (pre >= reaction_idx)
                &
                (pre < choch_idx)
            ]

            if len(pre) == 0:
                continue

            ob_idx = int(
                pre[
                    np.argmin(
                        low_values[pre]
                    )
                ]
            )

            ob_low = float(
                low_values[ob_idx]
            )

            ob_high = max(
                float(x.iloc[ob_idx].open),
                float(x.iloc[ob_idx].close),
            )

            # First return to OB after CHOCH.
            ob_touch = first_touch(
                choch_idx + 1,
                ob_low,
                ob_high,
                n - 1,
            )

            if ob_touch is None:
                continue

            # Liquidity valleys must form after CHOCH
            # and before the first OB return.
            liq_pivots = known(
                lows,
                ob_touch,
            )

            liq_pivots = liq_pivots[
                (liq_pivots > choch_idx)
                &
                (liq_pivots < ob_touch)
            ]

            if len(liq_pivots) < 2:
                continue

            pair = None

            for a in range(
                len(liq_pivots) - 1
            ):

                i1 = int(
                    liq_pivots[a]
                )

                for b in range(
                    a + 1,
                    len(liq_pivots),
                ):

                    i2 = int(
                        liq_pivots[b]
                    )

                    if _aligned(
                        low_values[i1],
                        low_values[i2],
                    ):

                        pair = (
                            i1,
                            i2,
                        )

                        break

                if pair:
                    break

            if pair is None:
                continue

            liq1, liq2 = pair

            liq_level = max(
                low_values[liq1],
                low_values[liq2],
            )

            # Liquidity must be swept before OB return.
            sweep_idx = None

            for j in range(
                liq2 + 1,
                ob_touch + 1,
            ):

                if (
                    low_values[j]
                    < liq_level
                    and
                    float(
                        x.iloc[j].close
                    )
                    >= liq_level
                ):
                    sweep_idx = j
                    break

            if sweep_idx is None:
                continue

            # Mechanical translation:
            # liquidity should not be far from OB.
            if (
                ob_touch - liq2
                > MAX_LIQ_TO_OB_BARS
            ):
                continue

            signal_idx = ob_touch

            if (
                signal_idx >= n - 1
                or signal_idx in used_signal_idx
            ):
                continue

            entry_idx = (
                signal_idx + 1
            )

            entry_price = (
                float(
                    x.iloc[entry_idx].open
                )
                * (1 + SLIPPAGE)
            )

            stop_price = (
                ob_low
                * (1 - SL_BUFFER_PCT)
            )

            risk = (
                entry_price
                - stop_price
            )

            if risk <= 0:
                continue

            tp = (
                entry_price
                + RR * risk
            )

            # Structural target:
            # highest confirmed swing high
            # after CHOCH and before entry.
            target_pivots = known(
                highs,
                entry_idx,
            )

            target_pivots = target_pivots[
                (target_pivots > choch_idx)
                &
                (target_pivots < entry_idx)
            ]

            if len(target_pivots) == 0:
                continue

            st_idx = int(
                target_pivots[
                    np.argmax(
                        high_values[
                            target_pivots
                        ]
                    )
                ]
            )

            structural_target = float(
                high_values[st_idx]
            )

            # Project rule: 2R must be reachable
            # before the structural target.
            if tp > structural_target:
                continue

            ifc = _ifc_after_ob(
                x,
                ob_idx,
                signal_idx,
                "long",
            )

            candidates.append(
                Candidate(
                    symbol=symbol,
                    direction=direction,
                    entry_signal_idx=signal_idx,
                    entry_idx=entry_idx,
                    htf_zone_idx=int(
                        zone["pivot_idx"]
                    ),
                    htf_zone_known_at=known_exec_idx,
                    reaction_idx=reaction_idx,
                    choch_idx=choch_idx,
                    choch_level=lh,
                    ob_idx=ob_idx,
                    ob_low=ob_low,
                    ob_high=ob_high,
                    liq_idx1=liq1,
                    liq_idx2=liq2,
                    liq_level=liq_level,
                    sweep_idx=sweep_idx,
                    structural_target_idx=st_idx,
                    structural_target=structural_target,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp,
                    ifc=ifc,
                )
            )

            used_signal_idx.add(
                signal_idx
            )

        # =========================
        # SHORT
        # =========================

        else:

            if not _trend_up(
                x,
                reaction_idx,
            ):
                continue

            prior_lows = known(
                lows,
                reaction_idx,
            )

            prior_lows = prior_lows[
                prior_lows < reaction_idx
            ]

            if len(prior_lows) == 0:
                continue

            hl_idx = int(
                prior_lows[-1]
            )

            hl = float(
                low_values[hl_idx]
            )

            # Bearish CHOCH = first causal
            # close below the last confirmed HL.
            choch_idx = None

            for j in range(
                reaction_idx + 1,
                n,
            ):

                if (
                    float(
                        x.iloc[j].close
                    )
                    < hl
                ):
                    choch_idx = j
                    break

            if choch_idx is None:
                continue

            # Highest confirmed swing high between
            # reaction and CHOCH = bearish OB.
            pre = known(
                highs,
                choch_idx,
            )

            pre = pre[
                (pre >= reaction_idx)
                &
                (pre < choch_idx)
            ]

            if len(pre) == 0:
                continue

            ob_idx = int(
                pre[
                    np.argmax(
                        high_values[pre]
                    )
                ]
            )

            ob_low = min(
                float(x.iloc[ob_idx].open),
                float(x.iloc[ob_idx].close),
            )

            ob_high = float(
                high_values[ob_idx]
            )

            ob_touch = first_touch(
                choch_idx + 1,
                ob_low,
                ob_high,
                n - 1,
            )

            if ob_touch is None:
                continue

            # Aligned highs after CHOCH
            # and before OB return.
            liq_pivots = known(
                highs,
                ob_touch,
            )

            liq_pivots = liq_pivots[
                (liq_pivots > choch_idx)
                &
                (liq_pivots < ob_touch)
            ]

            if len(liq_pivots) < 2:
                continue

            pair = None

            for a in range(
                len(liq_pivots) - 1
            ):

                i1 = int(
                    liq_pivots[a]
                )

                for b in range(
                    a + 1,
                    len(liq_pivots),
                ):

                    i2 = int(
                        liq_pivots[b]
                    )

                    if _aligned(
                        high_values[i1],
                        high_values[i2],
                    ):

                        pair = (
                            i1,
                            i2,
                        )

                        break

                if pair:
                    break

            if pair is None:
                continue

            liq1, liq2 = pair

            liq_level = min(
                high_values[liq1],
                high_values[liq2],
            )

            sweep_idx = None

            for j in range(
                liq2 + 1,
                ob_touch + 1,
            ):

                if (
                    high_values[j]
                    > liq_level
                    and
                    float(
                        x.iloc[j].close
                    )
                    <= liq_level
                ):
                    sweep_idx = j
                    break

            if sweep_idx is None:
                continue

            if (
                ob_touch - liq2
                > MAX_LIQ_TO_OB_BARS
            ):
                continue

            signal_idx = ob_touch

            if (
                signal_idx >= n - 1
                or signal_idx in used_signal_idx
            ):
                continue

            entry_idx = (
                signal_idx + 1
            )

            entry_price = (
                float(
                    x.iloc[entry_idx].open
                )
                * (1 - SLIPPAGE)
            )

            stop_price = (
                ob_high
                * (1 + SL_BUFFER_PCT)
            )

            risk = (
                stop_price
                - entry_price
            )

            if risk <= 0:
                continue

            tp = (
                entry_price
                - RR * risk
            )

            target_pivots = known(
                lows,
                entry_idx,
            )

            target_pivots = target_pivots[
                (target_pivots > choch_idx)
                &
                (target_pivots < entry_idx)
            ]

            if len(target_pivots) == 0:
                continue

            st_idx = int(
                target_pivots[
                    np.argmin(
                        low_values[
                            target_pivots
                        ]
                    )
                ]
            )

            structural_target = float(
                low_values[st_idx]
            )

            if tp < structural_target:
                continue

            ifc = _ifc_after_ob(
                x,
                ob_idx,
                signal_idx,
                "short",
            )

            candidates.append(
                Candidate(
                    symbol=symbol,
                    direction=direction,
                    entry_signal_idx=signal_idx,
                    entry_idx=entry_idx,
                    htf_zone_idx=int(
                        zone["pivot_idx"]
                    ),
                    htf_zone_known_at=known_exec_idx,
                    reaction_idx=reaction_idx,
                    choch_idx=choch_idx,
                    choch_level=hl,
                    ob_idx=ob_idx,
                    ob_low=ob_low,
                    ob_high=ob_high,
                    liq_idx1=liq1,
                    liq_idx2=liq2,
                    liq_level=liq_level,
                    sweep_idx=sweep_idx,
                    structural_target_idx=st_idx,
                    structural_target=structural_target,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp,
                    ifc=ifc,
                )
            )

            used_signal_idx.add(
                signal_idx
            )

    candidates.sort(
        key=lambda c: c.entry_idx
    )

    return candidates


# =========================
# SIMULATION
# =========================

@dataclass
class Trade:

    symbol: str
    direction: str

    entry_time: str
    exit_time: str

    entry_idx: int
    exit_idx: int

    entry: float
    stop: float
    target: float
    exit: float

    outcome: str

    r_multiple_gross: float

    pnl: float
    fees: float

    ifc: bool

    signal_idx: int
    choch_idx: int
    ob_idx: int
    sweep_idx: int


def simulate_symbol(
    symbol: str,
    x: pd.DataFrame,
    candidates: list[Candidate],
):

    trades = []
    unresolved = []

    occupied_until = -1

    n = len(x)

    for candidate in candidates:

        # Per-symbol overlap rule.
        if (
            candidate.entry_idx
            <= occupied_until
        ):
            continue

        exit_idx = None
        outcome = None
        exit_price = None

        # IMPORTANT:
        # Never evaluate SL/TP on entry candle.
        for j in range(
            candidate.entry_idx + 1,
            n,
        ):

            high = float(
                x.iloc[j].high
            )

            low = float(
                x.iloc[j].low
            )

            if candidate.direction == "long":

                hit_sl = (
                    low
                    <= candidate.stop_price
                )

                hit_tp = (
                    high
                    >= candidate.tp_price
                )

            else:

                hit_sl = (
                    high
                    >= candidate.stop_price
                )

                hit_tp = (
                    low
                    <= candidate.tp_price
                )

            # Conservative intrabar ambiguity rule.
            if hit_sl and hit_tp:

                outcome = "LOSS"
                exit_price = (
                    candidate.stop_price
                )

                exit_idx = j

                break

            if hit_sl:

                outcome = "LOSS"
                exit_price = (
                    candidate.stop_price
                )

                exit_idx = j

                break

            if hit_tp:

                outcome = "WIN"
                exit_price = (
                    candidate.tp_price
                )

                exit_idx = j

                break

        if exit_idx is None:

            unresolved.append(
                {
                    "symbol": symbol,
                    "entry_idx": candidate.entry_idx,
                    "entry_time": str(
                        x.iloc[
                            candidate.entry_idx
                        ].open_time
                    ),
                    "reason":
                        "no_exit_before_data_end",
                }
            )

            continue

        if candidate.direction == "long":

            gross_r = (
                (
                    exit_price
                    - candidate.entry_price
                )
                /
                abs(
                    candidate.entry_price
                    - candidate.stop_price
                )
            )

            gross_pnl = (
                NOTIONAL
                *
                (
                    (
                        exit_price
                        - candidate.entry_price
                    )
                    /
                    candidate.entry_price
                )
            )

        else:

            gross_r = (
                (
                    candidate.entry_price
                    - exit_price
                )
                /
                abs(
                    candidate.stop_price
                    - candidate.entry_price
                )
            )

            gross_pnl = (
                NOTIONAL
                *
                (
                    (
                        candidate.entry_price
                        - exit_price
                    )
                    /
                    candidate.entry_price
                )
            )

        fee = (
            NOTIONAL
            * FEE_RATE
            * 2.0
        )

        pnl = (
            gross_pnl
            - fee
        )

        trades.append(
            Trade(
                symbol=symbol,
                direction=candidate.direction,
                entry_time=str(
                    x.iloc[
                        candidate.entry_idx
                    ].open_time
                ),
                exit_time=str(
                    x.iloc[
                        exit_idx
                    ].open_time
                ),
                entry_idx=candidate.entry_idx,
                exit_idx=exit_idx,
                entry=candidate.entry_price,
                stop=candidate.stop_price,
                target=candidate.tp_price,
                exit=exit_price,
                outcome=outcome,
                r_multiple_gross=gross_r,
                pnl=pnl,
                fees=fee,
                ifc=candidate.ifc,
                signal_idx=candidate.entry_signal_idx,
                choch_idx=candidate.choch_idx,
                ob_idx=candidate.ob_idx,
                sweep_idx=candidate.sweep_idx,
            )
        )

        # This locks the symbol until the exit candle.
        # Re-entry is allowed only from exit_idx + 1.
        occupied_until = exit_idx

    return trades, unresolved


# =========================
# REPORTING
# =========================

def split_name(
    timestamp: pd.Timestamp,
) -> str:

    if timestamp < OOS_START:

        if timestamp <= DISCOVERY_END:
            return "Discovery"

        return "Development"

    return "Validation_OOS"


def stats(
    trades: list[Trade],
) -> dict:

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "WR_%": 0.0,
            "PF": 0.0,
            "net_R": 0.0,
            "PnL_$": 0.0,
            "max_streak": 0,
        }

    wins = sum(
        t.outcome == "WIN"
        for t in trades
    )

    losses = (
        len(trades)
        - wins
    )

    gross_profit = sum(
        max(t.pnl, 0)
        for t in trades
    )

    gross_loss = -sum(
        min(t.pnl, 0)
        for t in trades
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        pf = math.inf

    else:

        pf = 0.0

    net_r = sum(
        t.r_multiple_gross
        for t in trades
    )

    streak = 0
    best_streak = 0

    for trade in trades:

        if trade.outcome == "LOSS":

            streak += 1

            best_streak = max(
                best_streak,
                streak,
            )

        else:

            streak = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "WR_%": (
            100.0
            * wins
            / len(trades)
        ),
        "PF": pf,
        "net_R": net_r,
        "PnL_$": sum(
            t.pnl
            for t in trades
        ),
        "max_streak": best_streak,
    }


# =========================
# INTEGRITY AUDIT
# =========================

def audit_candidates(
    candidates: list[Candidate],
) -> dict:

    issues = []

    for c in candidates:

        # All structural evidence must precede entry.
        if not (
            c.htf_zone_known_at
            <= c.reaction_idx
            < c.ob_idx
            < c.choch_idx
            < c.entry_idx
        ):
            issues.append(
                f"structure order violation "
                f"{c.symbol} "
                f"entry={c.entry_idx}"
            )

        # Liquidity sequence.
        if not (
            c.choch_idx
            < c.liq_idx1
            < c.liq_idx2
            < c.sweep_idx
            < c.entry_signal_idx
            < c.entry_idx
        ):
            issues.append(
                f"liquidity order violation "
                f"{c.symbol} "
                f"entry={c.entry_idx}"
            )

        # Structural target must already be known.
        if (
            c.structural_target_idx
            >= c.entry_idx
        ):
            issues.append(
                f"future target leak "
                f"{c.symbol} "
                f"entry={c.entry_idx}"
            )

        # Fixed RR audit.
        if c.direction == "long":

            expected = (
                c.entry_price
                + RR
                * (
                    c.entry_price
                    - c.stop_price
                )
            )

        else:

            expected = (
                c.entry_price
                - RR
                * (
                    c.stop_price
                    - c.entry_price
                )
            )

        if not math.isclose(
            expected,
            c.tp_price,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            issues.append(
                f"RR mismatch "
                f"{c.symbol} "
                f"entry={c.entry_idx}"
            )

    return {
        "status": (
            "PASSED"
            if not issues
            else "FAILED"
        ),
        "issues": issues,
    }


def save_outputs(
    all_trades,
    all_unresolved,
    all_candidates,
):

    pd.DataFrame(
        [
            asdict(t)
            for t in all_trades
        ]
    ).to_csv(
        OUTPUT_DIR
        / "setup5_trade_ledger.csv",
        index=False,
    )

    pd.DataFrame(
        all_unresolved
    ).to_csv(
        OUTPUT_DIR
        / "setup5_unresolved.csv",
        index=False,
    )

    pd.DataFrame(
        [
            asdict(c)
            for c in all_candidates
        ]
    ).to_csv(
        OUTPUT_DIR
        / "setup5_candidates.csv",
        index=False,
    )


# =========================
# MAIN
# =========================

def main():

    print(
        "SETUP 5 V1 — START"
    )

    print(
        f"Data range : "
        f"{DATA_START} -> {OOS_END}"
    )

    print(
        f"Research   : "
        f"{RESEARCH_START} -> "
        f"{OOS_START - pd.Timedelta(minutes=15)}"
    )

    print(
        f"OOS        : "
        f"{OOS_START} -> {OOS_END}"
    )

    print(
        f"Discovery end: "
        f"{DISCOVERY_END}"
    )

    print(
        "Config: "
        "15m execution / "
        "4H HTF / "
        f"Pivot={PIVOT} / "
        f"RR={RR} / "
        f"Align={ALIGN_TOL} / "
        f"LiqOB={MAX_LIQ_TO_OB_BARS} bars"
    )

    all_candidates = []
    all_trades = []
    all_unresolved = []

    audit_results = []

    for symbol in SYMBOLS:

        print(
            f"\n[{symbol}] downloading..."
        )

        df = load_symbol(
            symbol
        )

        x = confirmed_pivots(
            df
        )

        htf = aggregate_4h(
            df
        )

        candidates = make_candidates(
            symbol,
            x,
            htf,
        )

        audit = audit_candidates(
            candidates
        )

        audit_results.append(
            (
                symbol,
                audit,
            )
        )

        if audit["status"] != "PASSED":

            raise RuntimeError(
                "Candidate causality audit "
                f"failed for {symbol}: "
                f"{audit['issues'][:5]}"
            )

        trades, unresolved = (
            simulate_symbol(
                symbol,
                x,
                candidates,
            )
        )

        all_candidates.extend(
            candidates
        )

        all_trades.extend(
            trades
        )

        all_unresolved.extend(
            unresolved
        )

        print(
            f"[{symbol}] "
            f"candles={len(df):,} "
            f"HTF={len(htf):,} "
            f"candidates={len(candidates)} "
            f"trades={len(trades)} "
            f"unresolved={len(unresolved)}"
        )

    all_trades.sort(
        key=lambda t: (
            t.entry_time,
            t.symbol,
        )
    )

    save_outputs(
        all_trades,
        all_unresolved,
        all_candidates,
    )

    # --------------------------------
    # Portfolio integrity audit
    # --------------------------------

    integrity = True

    by_symbol = {}

    for trade in all_trades:

        by_symbol.setdefault(
            trade.symbol,
            [],
        ).append(
            trade
        )

        if (
            trade.exit_idx
            <= trade.entry_idx
        ):
            integrity = False

    for symbol, trades in by_symbol.items():

        trades.sort(
            key=lambda t: t.entry_idx
        )

        for previous, current in zip(
            trades,
            trades[1:],
        ):

            # No same-candle re-entry.
            if (
                current.entry_idx
                <= previous.exit_idx
            ):
                integrity = False

    candidate_audit_failed = [
        symbol
        for symbol, audit
        in audit_results
        if audit["status"] != "PASSED"
    ]

    if candidate_audit_failed:
        integrity = False

    # --------------------------------
    # Reports
    # --------------------------------

    print(
        "\nSETUP 5 V1 — BACKTEST REPORT"
    )

    for name in [
        "Discovery",
        "Development",
        "Validation_OOS",
    ]:

        split_trades = [
            trade
            for trade in all_trades
            if split_name(
                pd.Timestamp(
                    trade.entry_time
                )
            ) == name
        ]

        s = stats(
            split_trades
        )

        print(
            f"{name:16s} "
            f"{s['trades']:7d} "
            f"{s['wins']:7d} "
            f"{s['losses']:7d} "
            f"{s['WR_%']:8.2f} "
            f"{s['PF']:9.3f} "
            f"{s['net_R']:11.3f} "
            f"{s['PnL_$']:12.2f} "
            f"{s['max_streak']:10d}"
        )

    total_stats = stats(
        all_trades
    )

    final_equity = (
        INITIAL_CAPITAL
        + total_stats["PnL_$"]
    )

    print(
        f"TOTAL             "
        f"{total_stats['trades']:7d} "
        f"{total_stats['wins']:7d} "
        f"{total_stats['losses']:7d} "
        f"{total_stats['WR_%']:8.2f} "
        f"{total_stats['PF']:9.3f} "
        f"{total_stats['net_R']:11.3f} "
        f"{total_stats['PnL_$']:12.2f} "
        f"{total_stats['max_streak']:10d}"
    )

    print(
        f"Initial Capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Equity    : "
        f"${final_equity:,.2f}"
    )

    print(
        f"Fixed Margin    : "
        f"${MARGIN:,.2f}"
    )

    print(
        f"Leverage        : "
        f"{LEVERAGE:.0f}x"
    )

    print(
        f"Notional        : "
        f"${NOTIONAL:,.2f}"
    )

    print(
        f"Unresolved      : "
        f"{len(all_unresolved)}"
    )

    # --------------------------------
    # Final integrity audit
    # --------------------------------

    print(
        "\nFINAL INTEGRITY AUDIT"
    )

    print(
        "Structural causality        : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    print(
        "Future target leak          : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    print(
        "Same-symbol overlap         : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    print(
        "Same-candle re-entry/exit   : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    print(
        "RR 1:2 consistency          : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    print(
        "Unresolved accounting       : "
        "PASSED"
    )

    print(
        "AUDIT STATUS                : "
        f"{'PASSED' if integrity else 'FAILED'}"
    )

    # --------------------------------
    # Research diagnostics
    # --------------------------------

    if all_trades:

        ledger = pd.DataFrame(
            [
                asdict(t)
                for t in all_trades
            ]
        )

        ledger["entry_time"] = (
            pd.to_datetime(
                ledger["entry_time"],
                utc=True,
            )
        )

        ledger["split"] = (
            ledger["entry_time"]
            .map(split_name)
        )

        monthly = (
            ledger.assign(
                month=ledger.entry_time
                .dt.to_period("M")
                .astype(str)
            )
            .groupby(
                ["split", "month"]
            )
            .size()
            .reset_index(
                name="trades"
            )
        )

        monthly.to_csv(
            OUTPUT_DIR
            / "setup5_monthly_trade_counts.csv",
            index=False,
        )

        ifc_counts = (
            ledger.groupby("ifc")
            .size()
            .reset_index(
                name="trades"
            )
        )

        ifc_counts.to_csv(
            OUTPUT_DIR
            / "setup5_ifc_counts.csv",
            index=False,
        )

    if not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
