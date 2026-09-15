# ============================================================
# HUNTER-V77
# Research Version
#
# Changes vs V76:
# 1) BTC momentum regime
# 2) Breadth: LONG >= 55%, SHORT <= 45%
# 3) Relative momentum vs BTC
# 4) ATR volatility filter
# 5) Extreme momentum filter
# 6) Max 3 positions per direction
# 7) Fixed $100 margin
# 8) Conservative trailing-stop management
# 9) Detailed loss-streak analysis
# 10) Per-symbol complete statistics
# 11) Filter rejection diagnostics
# ============================================================

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

# ------------------------------------------------------------
# Install/import dependencies
# ------------------------------------------------------------

try:
    import ccxt
except ImportError:
    print("Installing ccxt...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt"]
    )
    import ccxt

try:
    import pandas as pd
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pandas"]
    )
    import pandas as pd

import numpy as np


# ============================================================
# CONFIG
# ============================================================

VERSION = "HUNTER-V77"

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

# ------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------

INITIAL_CAPITAL = 1000.0

TRADE_MARGIN = 100.0
LEVERAGE = 80.0

MAX_POSITIONS = 5

# Maximum simultaneous positions in same direction
MAX_LONG_POSITIONS = 3
MAX_SHORT_POSITIONS = 3

# ------------------------------------------------------------
# Trading costs
# ------------------------------------------------------------

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------

ATR_PERIOD = 14

INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45

EMA_WARMUP = 200

# ------------------------------------------------------------
# V77 filters
# ------------------------------------------------------------

# Breadth
LONG_BREADTH_MIN = 0.55
SHORT_BREADTH_MAX = 0.45

# Relative momentum vs BTC
REQUIRE_RELATIVE_MOMENTUM = True

# ATR / Close
#
# 0.03 = 3%
#
# Candidate is rejected if ATR percentage is above this.
ATR_MAX_PCT = 0.03

# Extreme momentum filter
#
# Long:
#     0.035 < Mom_Long < 0.25
#
# Short:
#    -0.25 < Mom_Long < -0.035
#
MAX_MOM = 0.25

# Original momentum thresholds
MOM_SHORT_LONG_MIN = 0.012
MOM_LONG_LONG_MIN = 0.035

MOM_SHORT_SHORT_MAX = -0.012
MOM_LONG_SHORT_MAX = -0.035


# ============================================================
# 20 COIN UNIVERSE
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "BNB": "BNB/USDT",
    "XRP": "XRP/USDT",
    "SOL": "SOL/USDT",
    "DOGE": "DOGE/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
    "TRX": "TRX/USDT",
    "HYPE": "HYPE/USDT",
    "ZEC": "ZEC/USDT",
    "AVAX": "AVAX/USDT",
    "SUI": "SUI/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "ATOM": "ATOM/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "RENDER": "RENDER/USDT",
    "ONDO": "ONDO/USDT",
}


# ============================================================
# LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
})


# ============================================================
# DATA FETCH
# ============================================================

def fetch_ohlcv_full(symbol, timeframe, since_ms, until_ms):
    """
    Fetch complete OHLCV history using LBank pagination.
    """

    all_rows = []

    limit = 1000

    current_since = since_ms

    for attempt in range(3):

        try:

            while current_since < until_ms:

                rows = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )

                if not rows:
                    break

                all_rows.extend(rows)

                last_ts = rows[-1][0]

                # Protection against repeated pagination
                if last_ts <= current_since:
                    break

                current_since = last_ts + 1

                # Small delay
                time.sleep(exchange.rateLimit / 1000)

                # If fewer than limit returned,
                # probably reached available history
                if len(rows) < limit:
                    break

            break

        except Exception as e:

            print(
                f"  Fetch error {symbol}, attempt "
                f"{attempt + 1}/3: {e}"
            )

            if attempt < 2:
                time.sleep(3)
                current_since = since_ms
                all_rows = []

    if not all_rows:
        return None

    df = pd.DataFrame(
        all_rows,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    df = df.sort_values("Timestamp")

    # Remove incomplete last candle
    now_ms = int(time.time() * 1000)

    if len(df) > 0:

        last_ms = int(
            df["Timestamp"].iloc[-1].timestamp() * 1000
        )

        candle_duration_ms = 4 * 60 * 60 * 1000

        if last_ms + candle_duration_ms > now_ms:
            df = df.iloc[:-1]

    # --------------------------------------------------------
    # Gap check
    # --------------------------------------------------------

    if len(df) > 1:

        diffs = df["Timestamp"].diff().dropna()

        max_gap = diffs.max()

        expected = pd.Timedelta(hours=4)

        tolerance = pd.Timedelta(minutes=10)

        if max_gap > expected + tolerance:

            print(
                f"  WARNING: {symbol} has gap "
                f"{max_gap}"
            )

    df = df.set_index("Timestamp")

    df = df.dropna()

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = (
        df["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["EMA200"] = (
        df["Close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (df["High"] - prev_close).abs()

    tr3 = (df["Low"] - prev_close).abs()

    df["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        df["TR"]
        .rolling(ATR_PERIOD)
        .mean()
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["Mom_Short"] = (
        df["Close"] /
        df["Close"].shift(10)
        - 1.0
    )

    df["Mom_Long"] = (
        df["Close"] /
        df["Close"].shift(30)
        - 1.0
    )

    # --------------------------------------------------------
    # ATR percentage
    # --------------------------------------------------------

    df["ATR_Pct"] = (
        df["ATR"] /
        df["Close"]
    )

    # --------------------------------------------------------
    # Previous candle values
    # --------------------------------------------------------

    df["Prev_Low"] = df["Low"].shift(1)

    df["Prev_High"] = df["High"].shift(1)

    df["Prev_EMA20"] = df["EMA20"].shift(1)

    return df


# ============================================================
# LOAD ALL DATA
# ============================================================

print("=" * 70)
print(VERSION)
print("=" * 70)

print(
    f"Fetching {LOOKBACK_DAYS} days of "
    f"{TIMEFRAME} LBank data..."
)

end_time = datetime.utcnow()

start_time = end_time - timedelta(
    days=LOOKBACK_DAYS
)

since_ms = int(
    start_time.timestamp() * 1000
)

until_ms = int(
    end_time.timestamp() * 1000
)


DATA = {}

valid_symbols = []

for name, symbol in SYMBOLS.items():

    print(f"Fetching {name} ...")

    try:

        df = fetch_ohlcv_full(
            symbol,
            TIMEFRAME,
            since_ms,
            until_ms
        )

        if df is None or len(df) < EMA_WARMUP + 50:

            print(
                f"  INVALID: insufficient data"
            )

            continue

        df = add_indicators(df)

        df = df.dropna()

        if len(df) < EMA_WARMUP:

            print(
                f"  INVALID after indicators"
            )

            continue

        DATA[name] = df

        valid_symbols.append(name)

        print(
            f"  OK: {len(df)} candles"
        )

    except Exception as e:

        print(
            f"  FAILED: {e}"
        )


print()
print(
    f"{len(valid_symbols)}/{len(SYMBOLS)} "
    f"valid symbols"
)

print(valid_symbols)


if "BTC" not in DATA:

    raise RuntimeError(
        "BTC data is required."
    )


# ============================================================
# COMMON TIMESTAMPS
# ============================================================

ALL_TIMESTAMPS = sorted(
    set().union(
        *[
            set(df.index)
            for df in DATA.values()
        ]
    )
)


# ============================================================
# BTC DATA
# ============================================================

BTC = DATA["BTC"]


# ============================================================
# PORTFOLIO STATE
# ============================================================

capital = INITIAL_CAPITAL

open_positions = []

closed_trades = []

equity_curve = []

peak_equity = INITIAL_CAPITAL

max_drawdown_dollar = 0.0

max_drawdown_pct = 0.0


# ============================================================
# FILTER DIAGNOSTICS
# ============================================================

filter_stats = {

    "total_symbol_checks": 0,

    "btc_regime_rejected": 0,

    "breadth_rejected": 0,

    "relative_momentum_rejected": 0,

    "atr_rejected": 0,

    "extreme_momentum_rejected": 0,

    "trend_rejected": 0,

    "momentum_rejected": 0,

    "pullback_rejected": 0,

    "already_open_rejected": 0,

    "direction_limit_rejected": 0,

    "max_positions_rejected": 0,

    "accepted_candidates": 0,
}


# ============================================================
# HELPERS
# ============================================================

def count_direction(direction):

    return sum(
        1
        for p in open_positions
        if p["side"] == direction
    )


def calculate_unrealized_equity():

    equity = capital

    for p in open_positions:

        name = p["name"]

        side = p["side"]

        entry = p["entry"]

        current = p["last_price"]

        qty = p["qty"]

        if side == "LONG":

            raw_pnl = (
                current - entry
            ) * qty

        else:

            raw_pnl = (
                entry - current
            ) * qty

        equity += raw_pnl

    return equity


def close_position(
    position,
    exit_price,
    timestamp,
    reason
):

    global capital

    side = position["side"]

    entry = position["entry"]

    qty = position["qty"]

    margin = position["margin"]

    notional = position["notional"]

    if side == "LONG":

        raw_pnl = (
            exit_price - entry
        ) * qty

    else:

        raw_pnl = (
            entry - exit_price
        ) * qty

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    entry_fee = (
        notional *
        FEE_RATE
    )

    exit_notional = (
        exit_price *
        qty
    )

    exit_fee = (
        exit_notional *
        FEE_RATE
    )

    total_fee = (
        entry_fee +
        exit_fee
    )

    pnl = raw_pnl - total_fee

    capital += pnl

    # --------------------------------------------------------
    # R calculation
    # --------------------------------------------------------

    sl_dist_pct = (
        position["initial_sl_distance_pct"]
    )

    if sl_dist_pct > 0:

        raw_r = (
            raw_pnl /
            margin /
            sl_dist_pct
        )

        fee_r = (
            (FEE_RATE * 2) /
            sl_dist_pct
        )

        r_real = raw_r - fee_r

    else:

        r_real = 0.0

    trade = {

        "Timestamp": timestamp,

        "Symbol": position["name"],

        "Side": side,

        "Entry": entry,

        "Exit": exit_price,

        "Qty": qty,

        "Margin": margin,

        "Notional": notional,

        "RawPnL": raw_pnl,

        "Fees": total_fee,

        "PnL": pnl,

        "R": r_real,

        "Reason": reason,

        "EntryATR": position["entry_atr"],

        "EntryATR_Pct": position["entry_atr_pct"],

        "EntryMomShort": position["entry_mom_short"],

        "EntryMomLong": position["entry_mom_long"],

        "EntryRelativeMom": position[
            "entry_relative_mom"
        ],

        "EntryBreadth": position[
            "entry_breadth"
        ],

        "EntryBTCMomLong": position[
            "entry_btc_mom_long"
        ],
    }

    closed_trades.append(trade)


# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

for timestamp in ALL_TIMESTAMPS:

    # ========================================================
    # 1. MANAGE OPEN POSITIONS
    # ========================================================

    positions_to_close = []

    for p in list(open_positions):

        name = p["name"]

        df = DATA[name]

        if timestamp not in df.index:
            continue

        candle = df.loc[timestamp]

        p["last_price"] = float(
            candle["Close"]
        )

        p["bars"] += 1

        side = p["side"]

        # ----------------------------------------------------
        # FIRST:
        # check existing stop BEFORE modifying trailing stop
        #
        # This avoids same-candle trail/recheck bias.
        # ----------------------------------------------------

        if side == "LONG":

            old_stop = p["stop"]

            if candle["Low"] <= old_stop:

                positions_to_close.append(
                    (
                        p,
                        old_stop * (
                            1 - SLIPPAGE
                        ),
                        "STOP"
                    )
                )

                continue

        else:

            old_stop = p["stop"]

            if candle["High"] >= old_stop:

                positions_to_close.append(
                    (
                        p,
                        old_stop * (
                            1 + SLIPPAGE
                        ),
                        "STOP"
                    )
                )

                continue

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        if p["bars"] >= TIMEOUT_CANDLES:

            exit_price = float(
                candle["Close"]
            )

            positions_to_close.append(
                (
                    p,
                    exit_price,
                    "TIMEOUT"
                )
            )

            continue

        # ----------------------------------------------------
        # UPDATE TRAILING STOP
        #
        # Do NOT recheck this updated stop on the same candle.
        # ----------------------------------------------------

        atr = float(candle["ATR"])

        if side == "LONG":

            new_stop = (
                candle["Close"]
                - atr *
                TRAILING_ATR_MULTIPLIER
            )

            if new_stop > p["stop"]:

                p["stop"] = new_stop

        else:

            new_stop = (
                candle["Close"]
                + atr *
                TRAILING_ATR_MULTIPLIER
            )

            if new_stop < p["stop"]:

                p["stop"] = new_stop

    # --------------------------------------------------------
    # Execute closes
    # --------------------------------------------------------

    for p, exit_price, reason in positions_to_close:

        if p not in open_positions:
            continue

        close_position(
            p,
            exit_price,
            timestamp,
            reason
        )

        open_positions.remove(p)


    # ========================================================
    # 2. GET BTC REGIME
    # ========================================================

    if timestamp not in BTC.index:

        continue

    btc = BTC.loc[timestamp]

    btc_close = float(
        btc["Close"]
    )

    btc_ema50 = float(
        btc["EMA50"]
    )

    btc_ema200 = float(
        btc["EMA200"]
    )

    btc_mom_long = float(
        btc["Mom_Long"]
    )

    # --------------------------------------------------------
    # V77 BTC regime
    # --------------------------------------------------------

    btc_bull = (
        btc_close > btc_ema200
        and
        btc_ema50 > btc_ema200
        and
        btc_mom_long > 0
    )

    btc_bear = (
        btc_close < btc_ema200
        and
        btc_ema50 < btc_ema200
        and
        btc_mom_long < 0
    )


    # ========================================================
    # 3. MARKET BREADTH
    # ========================================================

    active_symbols = 0

    bullish_symbols = 0

    for name in valid_symbols:

        df = DATA[name]

        if timestamp not in df.index:
            continue

        row = df.loc[timestamp]

        if pd.isna(row["EMA200"]):
            continue

        active_symbols += 1

        if row["Close"] > row["EMA200"]:

            bullish_symbols += 1


    if active_symbols > 0:

        breadth_ratio = (
            bullish_symbols /
            active_symbols
        )

    else:

        breadth_ratio = 0.5


    # ========================================================
    # 4. MARKET PERMISSIONS
    # ========================================================

    allow_longs = (
        btc_bull
        and
        breadth_ratio >=
        LONG_BREADTH_MIN
    )

    allow_shorts = (
        btc_bear
        and
        breadth_ratio <=
        SHORT_BREADTH_MAX
    )


    # ========================================================
    # 5. BUILD CANDIDATES
    # ========================================================

    long_candidates = []

    short_candidates = []


    for name in valid_symbols:

        if name == "BTC":

            # BTC itself can trade,
            # but still follows all filters.
            pass


        df = DATA[name]

        if timestamp not in df.index:

            continue

        row = df.loc[timestamp]

        filter_stats[
            "total_symbol_checks"
        ] += 1


        # ----------------------------------------------------
        # Already open?
        # ----------------------------------------------------

        if any(
            p["name"] == name
            for p in open_positions
        ):

            filter_stats[
                "already_open_rejected"
            ] += 1

            continue


        # ----------------------------------------------------
        # Basic values
        # ----------------------------------------------------

        close = float(row["Close"])

        ema20 = float(row["EMA20"])

        ema50 = float(row["EMA50"])

        ema200 = float(row["EMA200"])

        atr = float(row["ATR"])

        atr_pct = float(row["ATR_Pct"])

        mom_short = float(
            row["Mom_Short"]
        )

        mom_long = float(
            row["Mom_Long"]
        )

        prev_low = float(
            row["Prev_Low"]
        )

        prev_high = float(
            row["Prev_High"]
        )

        prev_ema20 = float(
            row["Prev_EMA20"]
        )


        # ----------------------------------------------------
        # Relative Momentum
        # ----------------------------------------------------

        relative_momentum = (
            mom_long -
            btc_mom_long
        )


        # ====================================================
        # LONG
        # ====================================================

        long_ok = True

        # ----------------------------------------------------
        # BTC regime
        # ----------------------------------------------------

        if not btc_bull:

            long_ok = False

            filter_stats[
                "btc_regime_rejected"
            ] += 1

        # ----------------------------------------------------
        # Breadth
        # ----------------------------------------------------

        if long_ok:

            if breadth_ratio < LONG_BREADTH_MIN:

                long_ok = False

                filter_stats[
                    "breadth_rejected"
                ] += 1

        # ----------------------------------------------------
        # Relative momentum
        # ----------------------------------------------------

        if long_ok and REQUIRE_RELATIVE_MOMENTUM:

            if relative_momentum <= 0:

                long_ok = False

                filter_stats[
                    "relative_momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # ATR filter
        # ----------------------------------------------------

        if long_ok:

            if atr_pct > ATR_MAX_PCT:

                long_ok = False

                filter_stats[
                    "atr_rejected"
                ] += 1

        # ----------------------------------------------------
        # Extreme momentum
        # ----------------------------------------------------

        if long_ok:

            if not (
                mom_long >
                MOM_LONG_LONG_MIN
                and
                mom_long <
                MAX_MOM
            ):

                long_ok = False

                filter_stats[
                    "extreme_momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        if long_ok:

            if not (
                close > ema20
                and
                ema20 > ema50
                and
                close > ema200
            ):

                long_ok = False

                filter_stats[
                    "trend_rejected"
                ] += 1

        # ----------------------------------------------------
        # Short momentum
        # ----------------------------------------------------

        if long_ok:

            if mom_short <= MOM_SHORT_LONG_MIN:

                long_ok = False

                filter_stats[
                    "momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # Pullback
        # ----------------------------------------------------

        if long_ok:

            if not (
                prev_low <=
                prev_ema20 * 1.015
            ):

                long_ok = False

                filter_stats[
                    "pullback_rejected"
                ] += 1


        if long_ok:

            # ------------------------------------------------
            # Stop distance
            # ------------------------------------------------

            stop = (
                close -
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_dist_pct = (
                close - stop
            ) / close

            if not (
                0.01 <=
                sl_dist_pct <=
                0.04
            ):

                long_ok = False

            else:

                long_candidates.append({

                    "name": name,

                    "side": "LONG",

                    "score": float(
                        mom_long
                    ),

                    "row": row,

                    "atr": atr,

                    "atr_pct": atr_pct,

                    "mom_short": mom_short,

                    "mom_long": mom_long,

                    "relative_momentum":
                        relative_momentum,

                    "breadth":
                        breadth_ratio,

                    "btc_mom_long":
                        btc_mom_long,

                    "stop":
                        stop,

                    "sl_dist_pct":
                        sl_dist_pct,
                })


        # ====================================================
        # SHORT
        # ====================================================

        short_ok = True

        # ----------------------------------------------------
        # BTC regime
        # ----------------------------------------------------

        if not btc_bear:

            short_ok = False

            filter_stats[
                "btc_regime_rejected"
            ] += 1

        # ----------------------------------------------------
        # Breadth
        # ----------------------------------------------------

        if short_ok:

            if breadth_ratio > SHORT_BREADTH_MAX:

                short_ok = False

                filter_stats[
                    "breadth_rejected"
                ] += 1

        # ----------------------------------------------------
        # Relative momentum
        # ----------------------------------------------------

        if short_ok and REQUIRE_RELATIVE_MOMENTUM:

            if relative_momentum >= 0:

                short_ok = False

                filter_stats[
                    "relative_momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        if short_ok:

            if atr_pct > ATR_MAX_PCT:

                short_ok = False

                filter_stats[
                    "atr_rejected"
                ] += 1

        # ----------------------------------------------------
        # Extreme momentum
        # ----------------------------------------------------

        if short_ok:

            if not (
                mom_long <
                MOM_LONG_SHORT_MAX
                and
                mom_long >
                -MAX_MOM
            ):

                short_ok = False

                filter_stats[
                    "extreme_momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        if short_ok:

            if not (
                close < ema20
                and
                ema20 < ema50
                and
                close < ema200
            ):

                short_ok = False

                filter_stats[
                    "trend_rejected"
                ] += 1

        # ----------------------------------------------------
        # Momentum
        # ----------------------------------------------------

        if short_ok:

            if mom_short >= MOM_SHORT_SHORT_MAX:

                short_ok = False

                filter_stats[
                    "momentum_rejected"
                ] += 1

        # ----------------------------------------------------
        # Pullback
        # ----------------------------------------------------

        if short_ok:

            if not (
                prev_high >=
                prev_ema20 * 0.985
            ):

                short_ok = False

                filter_stats[
                    "pullback_rejected"
                ] += 1


        if short_ok:

            stop = (
                close +
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_dist_pct = (
                stop - close
            ) / close

            if not (
                0.01 <=
                sl_dist_pct <=
                0.04
            ):

                short_ok = False

            else:

                short_candidates.append({

                    "name": name,

                    "side": "SHORT",

                    "score": float(
                        -mom_long
                    ),

                    "row": row,

                    "atr": atr,

                    "atr_pct": atr_pct,

                    "mom_short": mom_short,

                    "mom_long": mom_long,

                    "relative_momentum":
                        relative_momentum,

                    "breadth":
                        breadth_ratio,

                    "btc_mom_long":
                        btc_mom_long,

                    "stop":
                        stop,

                    "sl_dist_pct":
                        sl_dist_pct,
                })


    # ========================================================
    # 6. RANK
    # ========================================================

    long_candidates.sort(
        key=lambda x: float(x["score"]),
        reverse=True
    )

    short_candidates.sort(
        key=lambda x: float(x["score"]),
        reverse=True
    )


    # ========================================================
    # 7. ENTRY
    # ========================================================

    slots_available = (
        MAX_POSITIONS -
        len(open_positions)
    )


    # --------------------------------------------------------
    # LONG entries
    # --------------------------------------------------------

    if allow_longs and slots_available > 0:

        current_longs = count_direction(
            "LONG"
        )

        for candidate in long_candidates:

            if slots_available <= 0:
                break

            if current_longs >= MAX_LONG_POSITIONS:

                filter_stats[
                    "direction_limit_rejected"
                ] += 1

                break

            # ------------------------------------------------
            # SAME-CANDLE OPEN
            #
            # Kept intentionally for V74/V75/V76
            # comparability.
            #
            # IMPORTANT:
            # This remains a look-ahead bias because the
            # signal uses current candle Close.
            # A clean no-lookahead benchmark should be run
            # separately.
            # ------------------------------------------------

            row = candidate["row"]

            entry = (
                float(row["Open"]) *
                (1 + SLIPPAGE)
            )

            atr = candidate["atr"]

            stop = (
                entry -
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_dist_pct = (
                entry - stop
            ) / entry

            if not (
                0.01 <=
                sl_dist_pct <=
                0.04
            ):
                continue

            margin = TRADE_MARGIN

            notional = (
                margin *
                LEVERAGE
            )

            qty = (
                notional /
                entry
            )

            position = {

                "name":
                    candidate["name"],

                "side":
                    "LONG",

                "entry":
                    entry,

                "qty":
                    qty,

                "margin":
                    margin,

                "notional":
                    notional,

                "stop":
                    stop,

                "initial_stop":
                    stop,

                "bars":
                    0,

                "last_price":
                    entry,

                "entry_atr":
                    atr,

                "entry_atr_pct":
                    candidate["atr_pct"],

                "entry_mom_short":
                    candidate["mom_short"],

                "entry_mom_long":
                    candidate["mom_long"],

                "entry_relative_mom":
                    candidate[
                        "relative_momentum"
                    ],

                "entry_breadth":
                    candidate[
                        "breadth"
                    ],

                "entry_btc_mom_long":
                    candidate[
                        "btc_mom_long"
                    ],

                "initial_sl_distance_pct":
                    sl_dist_pct,
            }

            open_positions.append(
                position
            )

            current_longs += 1

            slots_available -= 1

            filter_stats[
                "accepted_candidates"
            ] += 1


    # --------------------------------------------------------
    # SHORT entries
    # --------------------------------------------------------

    if allow_shorts and slots_available > 0:

        current_shorts = count_direction(
            "SHORT"
        )

        for candidate in short_candidates:

            if slots_available <= 0:
                break

            if current_shorts >= MAX_SHORT_POSITIONS:

                filter_stats[
                    "direction_limit_rejected"
                ] += 1

                break

            row = candidate["row"]

            entry = (
                float(row["Open"]) *
                (1 - SLIPPAGE)
            )

            atr = candidate["atr"]

            stop = (
                entry +
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_dist_pct = (
                stop - entry
            ) / entry

            if not (
                0.01 <=
                sl_dist_pct <=
                0.04
            ):
                continue

            margin = TRADE_MARGIN

            notional = (
                margin *
                LEVERAGE
            )

            qty = (
                notional /
                entry
            )

            position = {

                "name":
                    candidate["name"],

                "side":
                    "SHORT",

                "entry":
                    entry,

                "qty":
                    qty,

                "margin":
                    margin,

                "notional":
                    notional,

                "stop":
                    stop,

                "initial_stop":
                    stop,

                "bars":
                    0,

                "last_price":
                    entry,

                "entry_atr":
                    atr,

                "entry_atr_pct":
                    candidate["atr_pct"],

                "entry_mom_short":
                    candidate["mom_short"],

                "entry_mom_long":
                    candidate["mom_long"],

                "entry_relative_mom":
                    candidate[
                        "relative_momentum"
                    ],

                "entry_breadth":
                    candidate[
                        "breadth"
                    ],

                "entry_btc_mom_long":
                    candidate[
                        "btc_mom_long"
                    ],

                "initial_sl_distance_pct":
                    sl_dist_pct,
            }

            open_positions.append(
                position
            )

            current_shorts += 1

            slots_available -= 1

            filter_stats[
                "accepted_candidates"
            ] += 1


    # ========================================================
    # 8. EQUITY
    # ========================================================

    equity = calculate_unrealized_equity()

    equity_curve.append({

        "Timestamp":
            timestamp,

        "Capital":
            capital,

        "Equity":
            equity,

        "OpenPositions":
            len(open_positions),

        "LongPositions":
            count_direction("LONG"),

        "ShortPositions":
            count_direction("SHORT"),

        "Breadth":
            breadth_ratio,

        "BTC_MomLong":
            btc_mom_long,

        "BTC_Bull":
            btc_bull,

        "BTC_Bear":
            btc_bear,
    })


    # ========================================================
    # 9. DRAWDOWN
    # ========================================================

    if equity > peak_equity:

        peak_equity = equity

    dd_dollar = (
        equity -
        peak_equity
    )

    dd_pct = (
        dd_dollar /
        peak_equity
        if peak_equity > 0
        else 0
    )

    if dd_dollar < max_drawdown_dollar:

        max_drawdown_dollar = (
            dd_dollar
        )

    if dd_pct < max_drawdown_pct:

        max_drawdown_pct = (
            dd_pct
        )


# ============================================================
# FORCE CLOSE REMAINING POSITIONS
# ============================================================

if open_positions:

    final_timestamp = ALL_TIMESTAMPS[-1]

    for p in list(open_positions):

        name = p["name"]

        df = DATA[name]

        if final_timestamp in df.index:

            final_price = float(
                df.loc[
                    final_timestamp,
                    "Close"
                ]
            )

        else:

            final_price = p["last_price"]

        close_position(
            p,
            final_price,
            final_timestamp,
            "END"
        )

        open_positions.remove(p)


# ============================================================
# TRADES DATAFRAME
# ============================================================

trades_df = pd.DataFrame(
    closed_trades
)


if trades_df.empty:

    print()
    print("NO TRADES GENERATED.")
    sys.exit(0)


# ============================================================
# BASIC STATS
# ============================================================

total_trades = len(
    trades_df
)

wins_df = trades_df[
    trades_df["PnL"] > 0
]

losses_df = trades_df[
    trades_df["PnL"] <= 0
]

wins = len(wins_df)

losses = len(losses_df)

win_rate = (
    wins /
    total_trades *
    100
)

net_pnl = trades_df[
    "PnL"
].sum()

net_r = trades_df[
    "R"
].sum()

final_capital = (
    INITIAL_CAPITAL +
    net_pnl
)


# ============================================================
# LOSS STREAK ANALYSIS
# ============================================================

results = [
    1 if x > 0 else 0
    for x in trades_df["PnL"]
]

streaks = []

current_streak = 0

for result in results:

    if result == 0:

        current_streak += 1

    else:

        if current_streak > 0:

            streaks.append(
                current_streak
            )

        current_streak = 0


if current_streak > 0:

    streaks.append(
        current_streak
    )


if streaks:

    max_loss_streak = max(
        streaks
    )

    avg_loss_streak = np.mean(
        streaks
    )

else:

    max_loss_streak = 0

    avg_loss_streak = 0


# ------------------------------------------------------------
# Exact distribution
# ------------------------------------------------------------

streak_distribution = {}

for s in streaks:

    streak_distribution[s] = (
        streak_distribution.get(s, 0)
        + 1
    )


streak_rows = []

for length, count in sorted(
    streak_distribution.items()
):

    streak_rows.append({

        "LossStreak":
            length,

        "Occurrences":
            count,
    })


streak_df = pd.DataFrame(
    streak_rows
)


# ============================================================
# PORTFOLIO DRAW DOWN FROM EQUITY CURVE
# ============================================================

equity_df = pd.DataFrame(
    equity_curve
)

if not equity_df.empty:

    equity_df["Peak"] = (
        equity_df["Equity"]
        .cummax()
    )

    equity_df["DrawdownDollar"] = (
        equity_df["Equity"] -
        equity_df["Peak"]
    )

    equity_df["DrawdownPct"] = np.where(
        equity_df["Peak"] != 0,
        equity_df["DrawdownDollar"] /
        equity_df["Peak"] * 100,
        0
    )

    portfolio_max_dd_dollar = (
        equity_df[
            "DrawdownDollar"
        ].min()
    )

    portfolio_max_dd_pct = (
        equity_df[
            "DrawdownPct"
        ].min()
    )

else:

    portfolio_max_dd_dollar = 0

    portfolio_max_dd_pct = 0


# ============================================================
# LONG / SHORT STATS
# ============================================================

def direction_stats(df, side):

    x = df[
        df["Side"] == side
    ]

    if x.empty:

        return {

            "Trades": 0,

            "Wins": 0,

            "Losses": 0,

           
