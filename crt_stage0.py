import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CRT STAGE-0
# 4H RANGE -> SWEEP -> RECLAIM -> 15M MSS -> RETEST
#
# Research venue: Binance USD-M Futures public historical data
#
# Locked research rules:
# - Fixed universe
# - 365 days + 30 days warmup
# - 4H parent range
# - 15M execution
# - Signal only after completed candles
# - Entry on next 15M candle OPEN after retest confirmation
# - RR = 1:2
# - No timeout
# - No BE
# - No trailing
# - No pyramiding
# - One open trade per symbol
# - Different symbols may overlap
# - No same-candle re-entry
# - Same-candle SL + TP = LOSS
# - Final unresolved trades are censored
# - Chronological 50/25/25 split
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


RESEARCH_DAYS = 365
WARMUP_DAYS = 30

HTF_INTERVAL = "4h"
LTF_INTERVAL = "15m"

RR = 2.0

# Research cost assumptions
FEE_RATE = 0.0007
SLIPPAGE_RATE = 0.0003

MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

# Portfolio audit only.
# Per-symbol overlap is enforced by the simulator.
MAX_SIMULTANEOUS_POSITIONS = 10

# CRT sweep filter
MAX_SWEEP_DEPTH = 0.35

# 15M MSS
MSS_LOOKBACK = 4
MSS_MAX_BARS = 8

# Retest after MSS
RETEST_MAX_BARS = 6

# Avoid absurdly tiny/huge stop distances.
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

DATA_DIR = Path("crt_data")


session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 CRT-Stage0-Research"
    }
)


# ============================================================
# BINANCE DATA
# ============================================================

def month_range(start, end):
    current = pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")

    months = []

    while current <= last:
        months.append(str(current))
        current += 1

    return months


def download_archive(url):
    try:
        response = session.get(
            url,
            timeout=60,
        )

        if response.status_code == 200:
            return response.content

        return None

    except requests.RequestException:
        return None


def read_binance_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError("No CSV file inside Binance archive.")

        with z.open(csv_files[0]) as f:
            df = pd.read_csv(
                f,
                header=None,
            )

    # Binance archives can contain a header row.
    if len(df) > 0:
        first = str(df.iloc[0, 0]).strip().lower()

        if first in ("open_time", "open time"):
            df = df.iloc[1:].copy()

    if df.shape[1] < 12:
        raise RuntimeError(
            f"Unexpected Binance CSV columns: {df.shape[1]}"
        )

    df = df.iloc[:, :12].copy()

    df.columns = [
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

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["open_time"]
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    ].copy()


def fetch_binance_klines(
    symbol,
    interval,
    start,
    end,
):
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        DATA_DIR
        / f"{symbol}_{interval}.csv"
    )

    # Use cache only when it covers the requested period.
    if cache_file.exists():

        try:
            cached = pd.read_csv(
                cache_file
            )

            cached["open_time"] = pd.to_datetime(
                cached["open_time"],
                utc=True,
            )

            cached = cached.sort_values(
                "open_time"
            )

            if (
                len(cached) > 0
                and cached["open_time"].min() <= pd.Timestamp(start)
                and cached["open_time"].max() >= pd.Timestamp(end)
            ):
                return cached

        except Exception:
            pass

    frames = []

    # --------------------------------------------------------
    # Monthly archives
    # --------------------------------------------------------

    for ym in month_range(
        start,
        end,
    ):

        url = (
            "https://data.binance.vision/data/futures/um/"
            f"monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{ym}.zip"
        )

        blob = download_archive(url)

        if blob is not None:

            try:
                frame = read_binance_zip(
                    blob
                )

                frames.append(frame)

            except Exception as exc:
                print(
                    f"  monthly parse failed "
                    f"{symbol} {ym}: {exc}"
                )

        time.sleep(0.05)

    # --------------------------------------------------------
    # Daily fallback.
    #
    # Used for recent/current months that may not yet have
    # monthly archive files.
    # --------------------------------------------------------

    if frames:
        existing = pd.concat(
            frames,
            ignore_index=True,
        )

        existing_days = set(
            existing["open_time"]
            .dt.strftime("%Y-%m-%d")
        )

    else:
        existing = pd.DataFrame()
        existing_days = set()

    daily_start = pd.Timestamp(
        start,
        tz="UTC",
    ).floor("D")

    daily_end = pd.Timestamp(
        end,
        tz="UTC",
    ).floor("D")

    for day in pd.date_range(
        daily_start,
        daily_end,
        freq="D",
    ):

        date_string = day.strftime(
            "%Y-%m-%d"
        )

        if date_string in existing_days:
            continue

        url = (
            "https://data.binance.vision/data/futures/um/"
            f"daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_string}.zip"
        )

        blob = download_archive(url)

        if blob is not None:

            try:
                frame = read_binance_zip(
                    blob
                )

                frames.append(frame)

            except Exception as exc:
                print(
                    f"  daily parse failed "
                    f"{symbol} {date_string}: {exc}"
                )

        time.sleep(0.02)

    if not frames:
        raise RuntimeError(
            f"{symbol} {interval}: "
            "no Binance historical data downloaded."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["open_time"]
        )
        .sort_values(
            "open_time"
        )
        .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(
        start,
        tz="UTC",
    )

    end_ts = pd.Timestamp(
        end,
        tz="UTC",
    )

    df = df[
        (df["open_time"] >= start_ts)
        & (df["open_time"] < end_ts)
    ].copy()

    if len(df) == 0:
        raise RuntimeError(
            f"{symbol} {interval}: "
            "empty dataframe after date filtering."
        )

    df.to_csv(
        cache_file,
        index=False,
    )

    return df


# ============================================================
# DATA QUALITY
# ============================================================

def audit_continuity(
    df,
    interval_minutes,
    symbol,
    interval_name,
):
    if len(df) < 10:
        raise RuntimeError(
            f"{symbol} {interval_name}: "
            "too few rows."
        )

    timestamps = (
        df["open_time"]
        .sort_values()
        .drop_duplicates()
    )

    deltas = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    bad = deltas[
        deltas
        > interval_minutes * 1.10
    ]

    if len(bad) > 0:

        maximum_gap = float(
            bad.max()
        )

        raise RuntimeError(
            f"{symbol} {interval_name}: "
            f"data gap detected. "
            f"Max gap={maximum_gap:.2f} minutes."
        )


# ============================================================
# 4H REFERENCE FEATURES
# ============================================================

def prepare_4h(df):
    x = df.copy()

    x["range"] = (
        x["high"]
        - x["low"]
    )

    previous_close = (
        x["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous_close
            ).abs(),
            (
                x["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = (
        true_range
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    x["range_pct"] = (
        x["range"]
        / x["close"]
    )

    x["ema20"] = (
        x["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    x["ema50"] = (
        x["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # IMPORTANT:
    #
    # We use the CLOSED parent candle's relationship.
    #
    # +1 = bullish context
    # -1 = bearish context
    #
    # No future candle is used.

    x["bias"] = np.where(
        x["ema20"] > x["ema50"],
        1,
        -1,
    )

    return x


# ============================================================
# BUILD LTF DATA WITH PREVIOUS CLOSED 4H RANGE
# ============================================================

def attach_parent_range(
    h4,
    m15,
):
    reference = h4.copy()

    reference["htf_close_time"] = (
        reference["open_time"]
        + pd.Timedelta(
            hours=4
        )
    )

    reference = reference[
        [
            "htf_close_time",
            "open",
            "high",
            "low",
            "close",
            "range",
            "atr14",
            "range_pct",
            "bias",
        ]
    ].copy()

    reference.columns = [
        "htf_close_time",
        "parent_open",
        "parent_high",
        "parent_low",
        "parent_close",
        "parent_range",
        "parent_atr",
        "parent_range_pct",
        "parent_bias",
    ]

    ltf = m15.copy()

    # Only a COMPLETED 4H candle can become the parent.
    #
    # Example:
    # 4H candle 08:00 -> 12:00
    # It becomes available to 15M data starting at 12:00.

    ltf = pd.merge_asof(
        ltf.sort_values(
            "open_time"
        ),
        reference.sort_values(
            "htf_close_time"
        ),
        left_on="open_time",
        right_on="htf_close_time",
        direction="backward",
    )

    ltf = ltf.dropna(
        subset=[
            "parent_high",
            "parent_low",
            "parent_bias",
        ]
    ).reset_index(
        drop=True
    )

    return ltf


# ============================================================
# CRT DETECTION
# ============================================================

def find_crt_setups(
    df,
    symbol,
):
    bars = (
        df
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    trades = []

    i = (
        MSS_LOOKBACK
        + 2
    )

    total = len(bars)

    while i < total - 20:

        current = bars.iloc[i]

        parent_high = float(
            current.parent_high
        )

        parent_low = float(
            current.parent_low
        )

        parent_range = (
            parent_high
            - parent_low
        )

        if (
            not np.isfinite(
                parent_high
            )
            or not np.isfinite(
                parent_low
            )
            or parent_range <= 0
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # Parent candle identity
        # ----------------------------------------------------

        parent_time = current.htf_close_time

        # ----------------------------------------------------
        # CRT SWEEP
        #
        # LONG:
        # 15M candle sweeps below parent low
        # and closes back inside range.
        #
        # SHORT:
        # 15M candle sweeps above parent high
        # and closes back inside range.
        # ----------------------------------------------------

        side = None

        if (
            current.low < parent_low
            and current.close > parent_low
            and current.close < parent_high
        ):
            side = "LONG"
            sweep_price = float(
                current.low
            )

        elif (
            current.high > parent_high
            and current.close < parent_high
            and current.close > parent_low
        ):
            side = "SHORT"
            sweep_price = float(
                current.high
            )

        else:
            i += 1
            continue

        # ----------------------------------------------------
        # Make sure the parent did not change on the sweep.
        # ----------------------------------------------------

        if (
            current.htf_close_time
            != parent_time
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # Sweep depth.
        #
        # Too-deep penetration is treated as breakout/
        # price discovery rather than clean liquidity sweep.
        # ----------------------------------------------------

        if side == "LONG":

            sweep_depth = (
                parent_low
                - sweep_price
            ) / parent_range

        else:

            sweep_depth = (
                sweep_price
                - parent_high
            ) / parent_range

        if (
            sweep_depth < 0
            or sweep_depth
            > MAX_SWEEP_DEPTH
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # HTF directional context.
        #
        # Long sweep needs bullish parent context.
        # Short sweep needs bearish parent context.
        #
        # This is intentionally simple and PRE-REGISTERED.
        # ----------------------------------------------------

        parent_bias = int(
            current.parent_bias
        )

        if (
            side == "LONG"
            and parent_bias != 1
        ):
            i += 1
            continue

        if (
            side == "SHORT"
            and parent_bias != -1
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # 15M MSS
        #
        # After the sweep:
        #
        # LONG:
        # close above recent local high
        #
        # SHORT:
        # close below recent local low
        # ----------------------------------------------------

        mss_index = None

        search_end = min(
            i + MSS_MAX_BARS + 1,
            total,
        )

        for j in range(
            i + 1,
            search_end,
        ):

            if side == "LONG":

                local_high = (
                    bars["high"]
                    .iloc[
                        max(
                            0,
                            j - MSS_LOOKBACK,
                        ):j
                    ]
                    .max()
                )

                if (
                    bars["close"]
                    .iloc[j]
                    > local_high
                ):
                    mss_index = j
                    break

            else:

                local_low = (
                    bars["low"]
                    .iloc[
                        max(
                            0,
                            j - MSS_LOOKBACK,
                        ):j
                    ]
                    .min()
                )

                if (
                    bars["close"]
                    .iloc[j]
                    < local_low
                ):
                    mss_index = j
                    break

        if mss_index is None:
            i += 1
            continue

        # ----------------------------------------------------
        # RETEST
        #
        # We use the body of the MSS candle as the retest zone.
        #
        # The retest itself is confirmed by price trading into
        # the zone.
        #
        # Entry happens at the NEXT candle OPEN.
        # ----------------------------------------------------

        mss_open = float(
            bars["open"]
            .iloc[mss_index]
        )

        mss_close = float(
            bars["close"]
            .iloc[mss_index]
        )

        zone_high = max(
            mss_open,
            mss_close,
        )

        zone_low = min(
            mss_open,
            mss_close,
        )

        entry_index = None

        retest_end = min(
            mss_index
            + RETEST_MAX_BARS
            + 1,
            total,
        )

        for j in range(
            mss_index + 1,
            retest_end,
        ):

            touches_zone = (
                bars["low"].iloc[j]
                <= zone_high
                and
                bars["high"].iloc[j]
                >= zone_low
            )

            if touches_zone:

                # IMPORTANT:
                # The retest candle is confirmation only.
                # Entry is NEXT candle open.
                entry_index = j + 1
                break

        if (
            entry_index is None
            or entry_index >= total
        ):
            i = mss_index + 1
            continue

        entry_price = float(
            bars["open"]
            .iloc[entry_index]
        )

        # Entry must remain inside the original CRT range.
        if not (
            parent_low
            < entry_price
            < parent_high
        ):
            i = entry_index
            continue

        # ----------------------------------------------------
        # STOP = SWEEP EXTREME
        #
        # TARGET = 2R
        # ----------------------------------------------------

        stop_price = sweep_price

        if side == "LONG":

            risk = (
                entry_price
                - stop_price
            )

            target_price = (
                entry_price
                + RR * risk
            )

        else:

            risk = (
                stop_price
                - entry_price
            )

            target_price = (
                entry_price
                - RR * risk
            )

        if risk <= 0:
            i = entry_index
            continue

        risk_pct = (
            risk
            / entry_price
        )

        if (
            risk_pct < MIN_RISK_PCT
            or risk_pct > MAX_RISK_PCT
        ):
            i = entry_index
            continue

        # ----------------------------------------------------
        # RR FEASIBILITY CHECK
        #
        # If the opposite CRT side is closer than 2R,
        # the trade is not a valid CRT 1:2 setup.
        # ----------------------------------------------------

        if side == "LONG":

            available_reward = (
                parent_high
                - entry_price
            )

        else:

            available_reward = (
                entry_price
                - parent_low
            )

        if (
            available_reward
            < RR * risk
        ):
            i = entry_index
            continue

        # ----------------------------------------------------
        # FORWARD TRADE SIMULATION
        #
        # No timeout.
        # No trailing.
        # No BE.
        #
        # Same candle SL + TP = LOSS.
        # ----------------------------------------------------

        exit_index = None
        gross_r = None
        exit_reason = None

        for k in range(
            entry_index,
            total,
        ):

            bar_high = float(
                bars["high"]
                .iloc[k]
            )

            bar_low = float(
                bars["low"]
                .iloc[k]
            )

            if side == "LONG":

                hit_sl = (
                    bar_low
                    <= stop_price
                )

                hit_tp = (
                    bar_high
                    >= target_price
                )

            else:

                hit_sl = (
                    bar_high
                    >= stop_price
                )

                hit_tp = (
                    bar_low
                    <= target_price
                )

            # Ambiguous candle:
            # conservative assumption = LOSS.
            if hit_sl and hit_tp:

                exit_index = k
                gross_r = -1.0
                exit_reason = (
                    "BOTH_SL_TP_SAME_CANDLE"
                )
                break

            if hit_sl:

                exit_index = k
                gross_r = -1.0
                exit_reason = "SL"
                break

            if hit_tp:

                exit_index = k
                gross_r = RR
                exit_reason = "TP"
                break

        # ----------------------------------------------------
        # CENSORED TRADE
        #
        # If neither SL nor TP occurs before dataset end,
        # exclude it from performance.
        # ----------------------------------------------------

        if exit_index is None:
            break

        # ----------------------------------------------------
        # COST MODEL
        #
        # Round trip:
        # entry fee + exit fee
        # entry slippage + exit slippage
        #
        # Convert absolute trading cost to R.
        # ----------------------------------------------------

        trading_cost = (
            2.0
            * NOTIONAL
            * (
                FEE_RATE
                + SLIPPAGE_RATE
            )
        )

        dollar_risk = (
            NOTIONAL
            * risk_pct
        )

        cost_r = (
            trading_cost
            / dollar_risk
        )

        net_r = (
            gross_r
            - cost_r
        )

        trades.append(
            {
                "symbol": symbol,
                "parent_time": parent_time,
                "sweep_time": current.open_time,
                "mss_time": bars[
                    "open_time"
                ].iloc[mss_index],
                "retest_time": bars[
                    "open_time"
                ].iloc[
                    entry_index - 1
                ],
                "entry_time": bars[
                    "open_time"
                ].iloc[entry_index],
                "exit_time": bars[
                    "open_time"
                ].iloc[exit_index],

                "side": side,

                "parent_high": parent_high,
                "parent_low": parent_low,
                "parent_range": parent_range,

                "sweep_price": sweep_price,
                "sweep_depth": sweep_depth,

                "entry": entry_price,
                "sl": stop_price,
                "tp": target_price,

                "risk_pct": risk_pct,

                "gross_r": gross_r,
                "cost_r": cost_r,
                "net_r": net_r,

                "win": int(
                    gross_r > 0
                ),

                "exit_reason": exit_reason,
            }
        )

        # ----------------------------------------------------
        # PER-SYMBOL OVERLAP LOCK
        #
        # Nothing else on this symbol can be opened until
        # this trade closes.
        #
        # +1 means no same-candle re-entry.
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(
        trades
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_summary(
    trades,
):
    if len(trades) == 0:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr_pct": np.nan,
            "gross_r": 0.0,
            "net_r": 0.0,
            "pf": np.nan,
            "max_loss_streak": 0,
            "avg_net_r": np.nan,
        }

    wins = int(
        (
            trades["gross_r"]
            > 0
        ).sum()
    )

    losses = int(
        (
            trades["gross_r"]
            < 0
        ).sum()
    )

    gross_profit = (
        trades.loc[
            trades["net_r"] > 0,
            "net_r",
        ].sum()
    )

    gross_loss = -(
        trades.loc[
            trades["net_r"] < 0,
            "net_r",
        ].sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit
            / gross_loss
        )
    else:
        pf = np.inf

    streak = 0
    max_streak = 0

    for win in trades[
        "win"
    ].tolist():

        if win == 0:

            streak += 1

            max_streak = max(
                max_streak,
                streak,
            )

        else:
            streak = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr_pct": (
            100.0
            * wins
            / len(trades)
        ),
        "gross_r": trades[
            "gross_r"
        ].sum(),
        "net_r": trades[
            "net_r"
        ].sum(),
        "pf": pf,
        "max_loss_streak": max_streak,
        "avg_net_r": trades[
            "net_r"
        ].mean(),
    }


# ============================================================
# PORTFOLIO AUDIT
# ============================================================

def audit_overlap(
    trades,
):
    if len(trades) == 0:

        return {
            "symbol_overlap_errors": 0,
            "max_simultaneous": 0,
            "portfolio_limit": (
                MAX_SIMULTANEOUS_POSITIONS
            ),
        }

    ordered = trades.sort_values(
        "entry_time"
    ).reset_index(
        drop=True
    )

    active = []

    symbol_overlap_errors = 0
    max_simultaneous = 0

    for _, row in ordered.iterrows():

        entry_time = row[
            "entry_time"
        ]

        # Remove closed positions.
        active = [
            item
            for item in active
            if item["exit_time"]
            > entry_time
        ]

        # Check same-symbol overlap.
        for item in active:

            if (
                item["symbol"]
                == row["symbol"]
            ):

                symbol_overlap_errors += 1

        active.append(
            {
                "symbol": row[
                    "symbol"
                ],
                "exit_time": row[
                    "exit_time"
                ],
            }
        )

        max_simultaneous = max(
            max_simultaneous,
            len(active),
        )

    return {
        "symbol_overlap_errors": (
            symbol_overlap_errors
        ),
        "max_simultaneous": (
            max_simultaneous
        ),
        "portfolio_limit": (
            MAX_SIMULTANEOUS_POSITIONS
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    end = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .floor("15min")
    )

    start = (
        end
        - pd.Timedelta(
            days=(
                RESEARCH_DAYS
                + WARMUP_DAYS
            )
        )
    )

    print("=" * 70)
    print(
        "CRT STAGE-0"
    )
    print(
        "4H RANGE -> SWEEP -> RECLAIM -> "
        "15M MSS -> RETEST"
    )
    print("=" * 70)

    print(
        "Window:",
        start,
        "->",
        end,
    )

    print(
        "Symbols:",
        len(SYMBOLS),
    )

    print(
        "RR:",
        RR,
    )

    print(
        "Fee:",
        FEE_RATE,
    )

    print(
        "Slippage:",
        SLIPPAGE_RATE,
    )

    print()

    all_trades = []

    for index, symbol in enumerate(
        SYMBOLS,
        start=1,
    ):

        print(
            f"[{index}/{len(SYMBOLS)}] "
            f"{symbol}",
            flush=True,
        )

        # ----------------------------------------------------
        # Download 4H
        # ----------------------------------------------------

        h4 = fetch_binance_klines(
            symbol,
            HTF_INTERVAL,
            start,
            end,
        )

        # ----------------------------------------------------
        # Download 15M
        # ----------------------------------------------------

        m15 = fetch_binance_klines(
            symbol,
            LTF_INTERVAL,
            start,
            end,
        )

        # ----------------------------------------------------
        # Remove incomplete final candles.
        # ----------------------------------------------------

        h4 = h4[
            h4["open_time"]
            + pd.Timedelta(
                hours=4
            )
            <= end
        ].copy()

        m15 = m15[
            m15["open_time"]
            + pd.Timedelta(
                minutes=15
            )
            <= end
        ].copy()

        # ----------------------------------------------------
        # Data integrity audit.
        # ----------------------------------------------------

        audit_continuity(
            h4,
            240,
            symbol,
            "4H",
        )

        audit_continuity(
            m15,
            15,
            symbol,
            "15M",
        )

        # ----------------------------------------------------
        # Prepare parent candles.
        # ----------------------------------------------------

        h4 = prepare_4h(
            h4
        )

        # ----------------------------------------------------
        # Attach previous CLOSED 4H candle to 15M.
        # ----------------------------------------------------

        data = attach_parent_range(
            h4,
            m15,
        )

        # ----------------------------------------------------
        # Detect CRT trades.
        # ----------------------------------------------------

        symbol_trades = find_crt_setups(
            data,
            symbol,
        )

        print(
            "  Trades:",
            len(symbol_trades),
        )

        if len(symbol_trades) > 0:

            all_trades.append(
                symbol_trades
            )

    if not all_trades:

        raise RuntimeError(
            "CRT produced zero trades."
        )

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    trades = trades.sort_values(
        "entry_time"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    #
    # 50% Discovery
    # 25% Development
    # 25% Validation
    #
    # IMPORTANT:
    # Validation is untouched.
    # ========================================================

    n = len(trades)

    split_1 = int(
        n * 0.50
    )

    split_2 = int(
        n * 0.75
    )

    trades["split"] = np.where(
        np.arange(n) < split_1,
        "DISCOVERY",
        np.where(
            np.arange(n) < split_2,
            "DEVELOPMENT",
            "VALIDATION",
        ),
    )

    # ========================================================
    # OVERALL SPLIT SUMMARY
    # ========================================================

    split_rows = []

    for split_name in [
        "DISCOVERY",
        "DEVELOPMENT",
        "VALIDATION",
    ]:

        subset = trades[
            trades["split"]
            == split_name
        ].copy()

        stats = calculate_summary(
            subset
        )

        stats["split"] = split_name

        split_rows.append(
            stats
        )

    summary = pd.DataFrame(
        split_rows
    )

    summary = summary[
        [
            "split",
            "trades",
            "wins",
            "losses",
            "wr_pct",
            "gross_r",
            "net_r",
            "pf",
            "max_loss_streak",
            "avg_net_r",
        ]
    ]

    # ========================================================
    # SYMBOL / SIDE SUMMARY
    # ========================================================

    symbol_rows = []

    for (
        split_name,
        symbol,
        side,
    ), group in trades.groupby(
        [
            "split",
            "symbol",
            "side",
        ]
    ):

        stats = calculate_summary(
            group
        )

        stats.update(
            {
                "split": split_name,
                "symbol": symbol,
                "side": side,
            }
        )

        symbol_rows.append(
            stats
        )

    symbol_summary = pd.DataFrame(
        symbol_rows
    )

    symbol_summary = symbol_summary[
        [
            "split",
            "symbol",
            "side",
            "trades",
            "wins",
            "losses",
            "wr_pct",
            "gross_r",
            "net_r",
            "pf",
            "max_loss_streak",
            "avg_net_r",
        ]
    ]

    # ========================================================
    # OVERLAP AUDIT
    # ========================================================

    overlap = audit_overlap(
        trades
    )

    audit_df = pd.DataFrame(
        [
            {
                "total_trades": len(
                    trades
                ),
                "symbol_overlap_errors": (
                    overlap[
                        "symbol_overlap_errors"
                    ]
                ),
                "max_simultaneous": (
                    overlap[
                        "max_simultaneous"
                    ]
                ),
                "portfolio_limit": (
                    overlap[
                        "portfolio_limit"
                    ]
                ),
                "portfolio_limit_exceeded": (
                    overlap[
                        "max_simultaneous"
                    ]
                    > MAX_SIMULTANEOUS_POSITIONS
                ),
            }
        ]
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    trades.to_csv(
        "crt_stage0_trades.csv",
        index=False,
    )

    summary.to_csv(
        "crt_stage0_summary.csv",
        index=False,
    )

    symbol_summary.to_csv(
        "crt_stage0_validation_symbols.csv",
        index=False,
    )

    audit_df.to_csv(
        "crt_stage0_audit.csv",
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CRT STAGE-0 SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "OVERLAP AUDIT"
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Files saved:"
    )

    print(
        "  crt_stage0_trades.csv"
    )

    print(
        "  crt_stage0_summary.csv"
    )

    print(
        "  crt_stage0_validation_symbols.csv"
    )

    print(
        "  crt_stage0_audit.csv"
    )

    print()
    print(
        "CRT STAGE-0 COMPLETE"
    )


if __name__ == "__main__":
    main()
