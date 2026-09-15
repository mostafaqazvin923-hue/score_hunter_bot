# ============================================================
# HUNTER-V77 FIXED
# ============================================================
# 20 coins
# BTC regime + BTC momentum
# Breadth filter
# Relative momentum vs BTC
# ATR volatility filter
# Extreme momentum filter
# Max 3 positions per direction
# Fixed $100 margin x 80 leverage
# Conservative trailing stop
# Detailed loss streak
# Detailed per-symbol statistics
# Filter diagnostics
# CSV exports
# ============================================================

import sys
import time
import subprocess
from datetime import datetime, timedelta

# ============================================================
# DEPENDENCIES
# ============================================================

try:
    import ccxt
except ImportError:
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

try:
    import numpy as np
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "numpy"]
    )
    import numpy as np


# ============================================================
# CONFIG
# ============================================================

VERSION = "HUNTER-V77-FIXED"

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

INITIAL_CAPITAL = 1000.0

TRADE_MARGIN = 100.0
LEVERAGE = 80.0

MAX_POSITIONS = 5
MAX_LONG_POSITIONS = 3
MAX_SHORT_POSITIONS = 3

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14

INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

# ============================================================
# V77 FILTERS
# ============================================================

LONG_BREADTH_MIN = 0.55
SHORT_BREADTH_MAX = 0.45

REQUIRE_RELATIVE_MOMENTUM = True

# Maximum ATR / Close
ATR_MAX_PCT = 0.03

# Maximum 30-candle momentum
MAX_MOM = 0.25

# Original momentum thresholds
MOM_SHORT_LONG_MIN = 0.012
MOM_LONG_LONG_MIN = 0.035

MOM_SHORT_SHORT_MAX = -0.012
MOM_LONG_SHORT_MAX = -0.035


# ============================================================
# UNIVERSE
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
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
})


# ============================================================
# FETCH DATA
# ============================================================

def fetch_ohlcv_full(symbol, timeframe, since_ms, until_ms):

    rows_all = []

    limit = 1000
    current_since = since_ms

    for attempt in range(3):

        try:

            rows_all = []
            current_since = since_ms

            while current_since < until_ms:

                rows = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )

                if not rows:
                    break

                rows_all.extend(rows)

                last_ts = rows[-1][0]

                if last_ts <= current_since:
                    break

                current_since = last_ts + 1

                if len(rows) < limit:
                    break

                time.sleep(
                    max(exchange.rateLimit, 100) / 1000
                )

            break

        except Exception as e:

            print(
                f"  Fetch error {symbol} "
                f"attempt {attempt + 1}/3: {e}"
            )

            if attempt < 2:
                time.sleep(3)

    if not rows_all:
        return None

    df = pd.DataFrame(
        rows_all,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True,
    )

    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    df = df.sort_values(
        "Timestamp"
    )

    # --------------------------------------------------------
    # Remove incomplete last candle
    # --------------------------------------------------------

    if len(df) > 0:

        now_ms = int(
            time.time() * 1000
        )

        last_ms = int(
            df["Timestamp"].iloc[-1].timestamp()
            * 1000
        )

        candle_ms = (
            4 * 60 * 60 * 1000
        )

        if (
            last_ms + candle_ms
            > now_ms
        ):

            df = df.iloc[:-1]

    # --------------------------------------------------------
    # Gap warning
    # --------------------------------------------------------

    if len(df) > 1:

        diffs = (
            df["Timestamp"]
            .diff()
            .dropna()
        )

        if not diffs.empty:

            max_gap = diffs.max()

            allowed_gap = (
                pd.Timedelta(hours=4)
                + pd.Timedelta(minutes=10)
            )

            if max_gap > allowed_gap:

                print(
                    f"  WARNING {symbol}: "
                    f"largest gap = {max_gap}"
                )

    df = df.set_index(
        "Timestamp"
    )

    df = df.dropna()

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # ATR
    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"] -
        df["Low"]
    )

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    df["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        df["TR"]
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )

    # Momentum
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

    # ATR percentage
    df["ATR_Pct"] = (
        df["ATR"] /
        df["Close"]
    )

    # Previous values
    df["Prev_Low"] = (
        df["Low"].shift(1)
    )

    df["Prev_High"] = (
        df["High"].shift(1)
    )

    df["Prev_EMA20"] = (
        df["EMA20"].shift(1)
    )

    df = df.dropna()

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print(VERSION)
print("=" * 70)

end_time = datetime.utcnow()

start_time = (
    end_time -
    timedelta(days=LOOKBACK_DAYS)
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

    print(
        f"Fetching {name} "
        f"({symbol}) ..."
    )

    try:

        df = fetch_ohlcv_full(
            symbol,
            TIMEFRAME,
            since_ms,
            until_ms
        )

        if (
            df is None
            or len(df) <
            EMA_WARMUP + 50
        ):

            print(
                f"  INVALID: "
                f"insufficient candles"
            )

            continue

        df = add_indicators(df)

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
    f"{len(valid_symbols)}/"
    f"{len(SYMBOLS)} valid symbols"
)

print(
    ", ".join(valid_symbols)
)

if "BTC" not in DATA:

    raise RuntimeError(
        "BTC data unavailable."
    )


# ============================================================
# TIMESTAMPS
# ============================================================

ALL_TIMESTAMPS = sorted(
    set().union(
        *[
            set(df.index)
            for df in DATA.values()
        ]
    )
)


BTC = DATA["BTC"]


# ============================================================
# PORTFOLIO
# ============================================================

capital = INITIAL_CAPITAL

open_positions = []

closed_trades = []

equity_curve = []

peak_equity = INITIAL_CAPITAL


# ============================================================
# FILTER DIAGNOSTICS
# ============================================================

filter_stats = {
    "symbol_checks": 0,
    "already_open": 0,
    "btc_regime": 0,
    "breadth": 0,
    "relative_momentum": 0,
    "atr": 0,
    "extreme_momentum": 0,
    "trend": 0,
    "momentum": 0,
    "pullback": 0,
    "invalid_stop_distance": 0,
    "accepted": 0,
    "direction_limit": 0,
    "position_limit": 0,
}


# ============================================================
# HELPERS
# ============================================================

def count_direction(direction):

    return sum(
        1
        for position in open_positions
        if position["side"] == direction
    )


def calculate_equity():

    equity = capital

    for position in open_positions:

        current = position[
            "last_price"
        ]

        entry = position[
            "entry"
        ]

        qty = position[
            "qty"
        ]

        if position["side"] == "LONG":

            unrealized = (
                current - entry
            ) * qty

        else:

            unrealized = (
                entry - current
            ) * qty

        equity += unrealized

    return equity


def close_position(
    position,
    exit_price,
    timestamp,
    reason,
):

    global capital

    entry = position["entry"]
    qty = position["qty"]
    side = position["side"]
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

    pnl = (
        raw_pnl -
        total_fee
    )

    capital += pnl

    sl_dist_pct = (
        position[
            "initial_sl_distance_pct"
        ]
    )

    if sl_dist_pct > 0:

        raw_r = (
            raw_pnl /
            margin /
            sl_dist_pct
        )

        fee_r = (
            FEE_RATE * 2
        ) / sl_dist_pct

        real_r = (
            raw_r -
            fee_r
        )

    else:

        real_r = 0.0

    closed_trades.append({

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
        "R": real_r,
        "Reason": reason,

        "EntryATR":
            position["entry_atr"],

        "EntryATR_Pct":
            position["entry_atr_pct"],

        "EntryMomShort":
            position["entry_mom_short"],

        "EntryMomLong":
            position["entry_mom_long"],

        "EntryRelativeMom":
            position["entry_relative_mom"],

        "EntryBreadth":
            position["entry_breadth"],

        "EntryBTCMomLong":
            position["entry_btc_mom_long"],
    })


# ============================================================
# MAIN BACKTEST
# ============================================================

for timestamp in ALL_TIMESTAMPS:

    # ========================================================
    # MANAGE OPEN POSITIONS
    # ========================================================

    to_close = []

    for position in list(open_positions):

        name = position["name"]

        if timestamp not in DATA[name].index:
            continue

        candle = DATA[name].loc[
            timestamp
        ]

        position["last_price"] = float(
            candle["Close"]
        )

        position["bars"] += 1

        side = position["side"]

        # ----------------------------------------------------
        # Check OLD stop first
        # ----------------------------------------------------

        old_stop = position["stop"]

        if side == "LONG":

            if (
                candle["Low"]
                <= old_stop
            ):

                exit_price = (
                    old_stop *
                    (1 - SLIPPAGE)
                )

                to_close.append(
                    (
                        position,
                        exit_price,
                        "STOP"
                    )
                )

                continue

        else:

            if (
                candle["High"]
                >= old_stop
            ):

                exit_price = (
                    old_stop *
                    (1 + SLIPPAGE)
                )

                to_close.append(
                    (
                        position,
                        exit_price,
                        "STOP"
                    )
                )

                continue

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        if (
            position["bars"]
            >= TIMEOUT_CANDLES
        ):

            to_close.append(
                (
                    position,
                    float(candle["Close"]),
                    "TIMEOUT"
                )
            )

            continue

        # ----------------------------------------------------
        # Trailing update
        # ----------------------------------------------------

        atr = float(
            candle["ATR"]
        )

        if side == "LONG":

            new_stop = (
                candle["Close"]
                -
                atr *
                TRAILING_ATR_MULTIPLIER
            )

            if new_stop > position["stop"]:

                position["stop"] = (
                    new_stop
                )

        else:

            new_stop = (
                candle["Close"]
                +
                atr *
                TRAILING_ATR_MULTIPLIER
            )

            if new_stop < position["stop"]:

                position["stop"] = (
                    new_stop
                )

    # --------------------------------------------------------
    # Close positions
    # --------------------------------------------------------

    for (
        position,
        exit_price,
        reason,
    ) in to_close:

        if position not in open_positions:
            continue

        close_position(
            position,
            exit_price,
            timestamp,
            reason
        )

        open_positions.remove(
            position
        )


    # ========================================================
    # BTC
    # ========================================================

    if timestamp not in BTC.index:
        continue

    btc = BTC.loc[
        timestamp
    ]

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
    # BREADTH
    # ========================================================

    active = 0
    bullish = 0

    for name in valid_symbols:

        if timestamp not in DATA[name].index:
            continue

        row = DATA[name].loc[
            timestamp
        ]

        active += 1

        if (
            row["Close"] >
            row["EMA200"]
        ):

            bullish += 1

    if active > 0:

        breadth = (
            bullish /
            active
        )

    else:

        breadth = 0.5


    allow_longs = (
        btc_bull
        and
        breadth >=
        LONG_BREADTH_MIN
    )

    allow_shorts = (
        btc_bear
        and
        breadth <=
        SHORT_BREADTH_MAX
    )


    # ========================================================
    # CANDIDATES
    # ========================================================

    long_candidates = []
    short_candidates = []

    for name in valid_symbols:

        filter_stats[
            "symbol_checks"
        ] += 1

        if any(
            position["name"] == name
            for position in open_positions
        ):

            filter_stats[
                "already_open"
            ] += 1

            continue

        if timestamp not in DATA[name].index:
            continue

        row = DATA[name].loc[
            timestamp
        ]

        close = float(
            row["Close"]
        )

        ema20 = float(
            row["EMA20"]
        )

        ema50 = float(
            row["EMA50"]
        )

        ema200 = float(
            row["EMA200"]
        )

        atr = float(
            row["ATR"]
        )

        atr_pct = float(
            row["ATR_Pct"]
        )

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

        relative_momentum = (
            mom_long -
            btc_mom_long
        )


        # ====================================================
        # LONG
        # ====================================================

        long_ok = True

        if not btc_bull:

            long_ok = False

            filter_stats[
                "btc_regime"
            ] += 1

        elif breadth < LONG_BREADTH_MIN:

            long_ok = False

            filter_stats[
                "breadth"
            ] += 1

        elif (
            REQUIRE_RELATIVE_MOMENTUM
            and
            relative_momentum <= 0
        ):

            long_ok = False

            filter_stats[
                "relative_momentum"
            ] += 1

        elif atr_pct > ATR_MAX_PCT:

            long_ok = False

            filter_stats[
                "atr"
            ] += 1

        elif not (
            mom_long >
            MOM_LONG_LONG_MIN
            and
            mom_long <
            MAX_MOM
        ):

            long_ok = False

            filter_stats[
                "extreme_momentum"
            ] += 1

        elif not (
            close > ema20
            and
            ema20 > ema50
            and
            close > ema200
        ):

            long_ok = False

            filter_stats[
                "trend"
            ] += 1

        elif (
            mom_short <=
            MOM_SHORT_LONG_MIN
        ):

            long_ok = False

            filter_stats[
                "momentum"
            ] += 1

        elif not (
            prev_low <=
            prev_ema20 * 1.015
        ):

            long_ok = False

            filter_stats[
                "pullback"
            ] += 1

        if long_ok:

            stop = (
                close -
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_distance_pct = (
                close - stop
            ) / close

            if not (
                0.01 <=
                sl_distance_pct <=
                0.04
            ):

                filter_stats[
                    "invalid_stop_distance"
                ] += 1

            else:

                long_candidates.append({

                    "name": name,
                    "side": "LONG",
                    "score": float(mom_long),
                    "atr": atr,
                    "atr_pct": atr_pct,
                    "mom_short": mom_short,
                    "mom_long": mom_long,
                    "relative_momentum":
                        relative_momentum,
                    "breadth": breadth,
                    "btc_mom_long":
                        btc_mom_long,
                    "stop": stop,
                    "sl_distance_pct":
                        sl_distance_pct,
                })


        # ====================================================
        # SHORT
        # ====================================================

        short_ok = True

        if not btc_bear:

            short_ok = False

            filter_stats[
                "btc_regime"
            ] += 1

        elif breadth > SHORT_BREADTH_MAX:

            short_ok = False

            filter_stats[
                "breadth"
            ] += 1

        elif (
            REQUIRE_RELATIVE_MOMENTUM
            and
            relative_momentum >= 0
        ):

            short_ok = False

            filter_stats[
                "relative_momentum"
            ] += 1

        elif atr_pct > ATR_MAX_PCT:

            short_ok = False

            filter_stats[
                "atr"
            ] += 1

        elif not (
            mom_long <
            MOM_LONG_SHORT_MAX
            and
            mom_long >
            -MAX_MOM
        ):

            short_ok = False

            filter_stats[
                "extreme_momentum"
            ] += 1

        elif not (
            close < ema20
            and
            ema20 < ema50
            and
            close < ema200
        ):

            short_ok = False

            filter_stats[
                "trend"
            ] += 1

        elif (
            mom_short >=
            MOM_SHORT_SHORT_MAX
        ):

            short_ok = False

            filter_stats[
                "momentum"
            ] += 1

        elif not (
            prev_high >=
            prev_ema20 * 0.985
        ):

            short_ok = False

            filter_stats[
                "pullback"
            ] += 1

        if short_ok:

            stop = (
                close +
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_distance_pct = (
                stop - close
            ) / close

            if not (
                0.01 <=
                sl_distance_pct <=
                0.04
            ):

                filter_stats[
                    "invalid_stop_distance"
                ] += 1

            else:

                short_candidates.append({

                    "name": name,
                    "side": "SHORT",
                    "score": float(-mom_long),
                    "atr": atr,
                    "atr_pct": atr_pct,
                    "mom_short": mom_short,
                    "mom_long": mom_long,
                    "relative_momentum":
                        relative_momentum,
                    "breadth": breadth,
                    "btc_mom_long":
                        btc_mom_long,
                    "stop": stop,
                    "sl_distance_pct":
                        sl_distance_pct,
                })


    # ========================================================
    # RANK
    # ========================================================

    long_candidates.sort(
        key=lambda item: float(
            item["score"]
        ),
        reverse=True
    )

    short_candidates.sort(
        key=lambda item: float(
            item["score"]
        ),
        reverse=True
    )


    # ========================================================
    # ENTRY
    # ========================================================

    slots = (
        MAX_POSITIONS -
        len(open_positions)
    )


    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if allow_longs and slots > 0:

        current_longs = count_direction(
            "LONG"
        )

        for candidate in long_candidates:

            if slots <= 0:
                break

            if (
                current_longs
                >=
                MAX_LONG_POSITIONS
            ):

                filter_stats[
                    "direction_limit"
                ] += 1

                break

            entry = float(
                DATA[
                    candidate["name"]
                ].loc[
                    timestamp,
                    "Open"
                ]
            )

            entry *= (
                1 + SLIPPAGE
            )

            atr = candidate["atr"]

            stop = (
                entry -
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_distance_pct = (
                entry - stop
            ) / entry

            if not (
                0.01 <=
                sl_distance_pct <=
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

            open_positions.append({

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
                    sl_distance_pct,
            })

            current_longs += 1
            slots -= 1

            filter_stats[
                "accepted"
            ] += 1


    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if allow_shorts and slots > 0:

        current_shorts = count_direction(
            "SHORT"
        )

        for candidate in short_candidates:

            if slots <= 0:
                break

            if (
                current_shorts
                >=
                MAX_SHORT_POSITIONS
            ):

                filter_stats[
                    "direction_limit"
                ] += 1

                break

            entry = float(
                DATA[
                    candidate["name"]
                ].loc[
                    timestamp,
                    "Open"
                ]
            )

            entry *= (
                1 - SLIPPAGE
            )

            atr = candidate["atr"]

            stop = (
                entry +
                atr *
                INITIAL_ATR_MULTIPLIER
            )

            sl_distance_pct = (
                stop - entry
            ) / entry

            if not (
                0.01 <=
                sl_distance_pct <=
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

            open_positions.append({

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
                    sl_distance_pct,
            })

            current_shorts += 1
            slots -= 1

            filter_stats[
                "accepted"
            ] += 1


    # ========================================================
    # EQUITY
    # ========================================================

    equity = calculate_equity()

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
            breadth,

        "BTC_MomLong":
            btc_mom_long,

        "BTC_Bull":
            btc_bull,

        "BTC_Bear":
            btc_bear,
    })

    if equity > peak_equity:

        peak_equity = equity


# ============================================================
# FORCE CLOSE
# ============================================================

if open_positions:

    final_timestamp = (
        ALL_TIMESTAMPS[-1]
    )

    for position in list(
        open_positions
    ):

        name = position["name"]

        if (
            final_timestamp
            in DATA[name].index
        ):

            final_price = float(
                DATA[name]
                .loc[
                    final_timestamp,
                    "Close"
                ]
            )

        else:

            final_price = position[
                "last_price"
            ]

        close_position(
            position,
            final_price,
            final_timestamp,
            "END"
        )

        open_positions.remove(
            position
        )


# ============================================================
# TRADES DATAFRAME
# ============================================================

trades_df = pd.DataFrame(
    closed_trades
)

if trades_df.empty:

    print()
    print("=" * 70)
    print("NO TRADES")
    print("=" * 70)
    sys.exit(0)


# ============================================================
# OVERALL STATS
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

net_pnl = float(
    trades_df["PnL"].sum()
)

net_r = float(
    trades_df["R"].sum()
)

final_capital = (
    INITIAL_CAPITAL +
    net_pnl
)


# ============================================================
# LOSS STREAK
# ============================================================

results = [
    1 if pnl > 0 else 0
    for pnl in trades_df["PnL"]
]

loss_streaks = []

current_streak = 0

for result in results:

    if result == 0:

        current_streak += 1

    else:

        if current_streak > 0:

            loss_streaks.append(
                current_streak
            )

        current_streak = 0

if current_streak > 0:

    loss_streaks.append(
        current_streak
    )

if loss_streaks:

    max_loss_streak = max(
        loss_streaks
    )

    avg_loss_streak = float(
        np.mean(loss_streaks)
    )

else:

    max_loss_streak = 0
    avg_loss_streak = 0.0


streak_counts = {}

for streak in loss_streaks:

    streak_counts[streak] = (
        streak_counts.get(
            streak,
            0
        ) + 1
    )


streak_rows = []

for streak, count in sorted(
    streak_counts.items()
):

    streak_rows.append({

        "LossStreak":
            streak,

        "Occurrences":
            count,
    })

streak_df = pd.DataFrame(
    streak_rows
)


# ============================================================
# EQUITY / DRAWDOWN
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
        equity_df["Equity"]
        -
        equity_df["Peak"]
    )

    equity_df["DrawdownPct"] = (
        np.where(
            equity_df["Peak"] != 0,
            equity_df[
                "DrawdownDollar"
            ]
            /
            equity_df["Peak"]
            * 100,
            0
        )
    )

    max_dd_dollar = float(
        equity_df[
            "DrawdownDollar"
        ].min()
    )

    max_dd_pct = float(
        equity_df[
            "DrawdownPct"
        ].min()
    )

else:

    max_dd_dollar = 0.0
    max_dd_pct = 0.0


# ============================================================
# DIRECTION STATS
# ============================================================

def get_direction_stats(
    df,
    side
):

    subset = df[
        df["Side"] == side
    ]

    if subset.empty:

        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "WinRate": 0.0,
            "PnL": 0.0,
            "R": 0.0,
            "AvgR": 0.0,
        }

    side_wins = subset[
        subset["PnL"] > 0
    ]

    side_losses = subset[
        subset["PnL"] <= 0
    ]

    return {
        "Trades":
            len(subset),

        "Wins":
            len(side_wins),

        "Losses":
            len(side_losses),

        "WinRate":
            len(side_wins)
            /
            len(subset)
            *
            100,

        "PnL":
            float(
                subset["PnL"].sum()
            ),

        "R":
            float(
                subset["R"].sum()
            ),

        "AvgR":
            float(
                subset["R"].mean()
            ),
    }


long_stats = get_direction_stats(
    trades_df,
    "LONG"
)

short_stats = get_direction_stats(
    trades_df,
    "SHORT"
)


# ============================================================
# PER SYMBOL
# ============================================================

symbol_rows = []

for name in valid_symbols:

    subset = trades_df[
        trades_df["Symbol"] == name
    ]

    if subset.empty:

        symbol_rows.append({

            "Symbol": name,
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "WinRate": 0.0,
            "PnL": 0.0,
            "NetR": 0.0,
            "AvgR": 0.0,
            "AvgPnL": 0.0,
            "BestTrade": 0.0,
            "WorstTrade": 0.0,
            "LongTrades": 0,
            "LongWR": 0.0,
            "LongPnL": 0.0,
            "ShortTrades": 0,
            "ShortWR": 0.0,
            "ShortPnL": 0.0,
            "MaxLossStreak": 0,
        })

        continue


    symbol_wins = subset[
        subset["PnL"] > 0
    ]

    symbol_losses = subset[
        subset["PnL"] <= 0
    ]


    # --------------------------------------------------------
    # Symbol loss streak
    # --------------------------------------------------------

    max_symbol_streak = 0
    current_symbol_streak = 0

    for pnl in subset["PnL"]:

        if pnl <= 0:

            current_symbol_streak += 1

            max_symbol_streak = max(
                max_symbol_streak,
                current_symbol_streak
            )

        else:

            current_symbol_streak = 0


    # --------------------------------------------------------
    # Long
    # --------------------------------------------------------

    long_subset = subset[
        subset["Side"] == "LONG"
    ]

    if not long_subset.empty:

        long_wins = long_subset[
            long_subset["PnL"] > 0
        ]

        long_wr = (
            len(long_wins)
            /
            len(long_subset)
            *
            100
        )

        long_pnl = float(
            long_subset["PnL"].sum()
        )

    else:

        long_wr = 0.0
        long_pnl = 0.0


    # --------------------------------------------------------
    # Short
    # --------------------------------------------------------

    short_subset = subset[
        subset["Side"] == "SHORT"
    ]

    if not short_subset.empty:

        short_wins = short_subset[
            short_subset["PnL"] > 0
        ]

        short_wr = (
            len(short_wins)
            /
            len(short_subset)
            *
            100
        )

        short_pnl = float(
            short_subset["PnL"].sum()
        )

    else:

        short_wr = 0.0
        short_pnl = 0.0


    symbol_rows.append({

        "Symbol":
            name,

        "Trades":
            len(subset),

        "Wins":
            len(symbol_wins),

        "Losses":
            len(symbol_losses),

        "WinRate":
            len(symbol_wins)
            /
            len(subset)
            *
            100,

        "PnL":
            float(
                subset["PnL"].sum()
            ),

        "NetR":
            float(
                subset["R"].sum()
            ),

        "AvgR":
            float(
                subset["R"].mean()
            ),

        "AvgPnL":
            float(
                subset["PnL"].mean()
            ),

        "BestTrade":
            float(
                subset["PnL"].max()
            ),

        "WorstTrade":
            float(
                subset["PnL"].min()
            ),

        "LongTrades":
            len(long_subset),

        "LongWR":
            long_wr,

        "LongPnL":
            long_pnl,

        "ShortTrades":
            len(short_subset),

        "ShortWR":
            short_wr,

        "ShortPnL":
            short_pnl,

        "MaxLossStreak":
            max_symbol_streak,
    })


symbol_df = pd.DataFrame(
    symbol_rows
)

symbol_df = symbol_df.sort_values(
    "PnL",
    ascending=False
)


# ============================================================
# EXIT STATS
# ============================================================

exit_df = (
    trades_df
    .groupby("Reason")
    .agg(
        Trades=("PnL", "count"),
        Wins=(
            "PnL",
            lambda x: int(
                (x > 0).sum()
            )
        ),
        PnL=("PnL", "sum"),
        NetR=("R", "sum"),
        AvgR=("R", "mean"),
    )
    .reset_index()
)

exit_df["WinRate"] = np.where(
    exit_df["Trades"] > 0,
    exit_df["Wins"]
    /
    exit_df["Trades"]
    * 100,
    0.0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print(f"{VERSION} RESULTS")
print("=" * 70)

print(
    f"Valid Symbols : "
    f"{len(valid_symbols)}/{len(SYMBOLS)}"
)

print(
    f"Total Trades  : "
    f"{total_trades}"
)

print(
    f"Wins          : "
    f"{wins}"
)

print(
    f"Losses        : "
    f"{losses}"
)

print(
    f"Win Rate      : "
    f"{win_rate:.2f}%"
)

print(
    f"Net R         : "
    f"{net_r:.2f}R"
)

print(
    f"Net PnL       : "
    f"${net_pnl:,.2f}"
)

print(
    f"Final Capital : "
    f"${final_capital:,.2f}"
)

print(
    f"Return        : "
    f"{(
        final_capital /
        INITIAL_CAPITAL
        - 1
    ) * 100:.2f}%"
)

print(
    f"Max Drawdown  : "
    f"${max_dd_dollar:,.2f} "
    f"({max_dd_pct:.2f}%)"
)

print(
    f"Max Loss Streak: "
    f"{max_loss_streak}"
)

print(
    f"Avg Loss Streak: "
    f"{avg_loss_streak:.2f}"
)


# ============================================================
# LONG / SHORT
# ============================================================

print()
print("-" * 70)
print("LONG / SHORT")
print("-" * 70)

print(
    f"LONG  | "
    f"Trades={long_stats['Trades']} | "
    f"WR={long_stats['WinRate']:.2f}% | "
    f"PnL=${long_stats['PnL']:,.2f} | "
    f"R={long_stats['R']:.2f}"
)

print(
    f"SHORT | "
    f"Trades={short_stats['Trades']} | "
    f"WR={short_stats['WinRate']:.2f}% | "
    f"PnL=${short_stats['PnL']:,.2f} | "
    f"R={short_stats['R']:.2f}"
)


# ============================================================
# LOSS STREAK DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("LOSS STREAK DISTRIBUTION")
print("=" * 70)

if streak_counts:

    for length in sorted(
        streak_counts
    ):

        print(
            f"{length:>2} consecutive losses "
            f": {streak_counts[length]} times"
        )

else:

    print("No loss streaks.")


# ============================================================
# PER SYMBOL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PER SYMBOL RESULTS")
print("=" * 70)

print(
    f"{'SYMBOL':<10}"
    f"{'TRADES':>8}"
    f"{'WR':>9}"
    f"{'PNL':>14}"
    f"{'R':>10}"
    f"{'AVG R':>10}"
    f"{'MAX LS':>10}"
)

print("-" * 70)

for _, row in symbol_df.iterrows():

    print(
        f"{row['Symbol']:<10}"
        f"{int(row['Trades']):>8}"
        f"{row['WinRate']:>8.2f}%"
        f"{row['PnL']:>14,.2f}"
        f"{row['NetR']:>10.2f}"
        f"{row['AvgR']:>10.3f}"
        f"{int(row['MaxLossStreak']):>10}"
    )


# ============================================================
# DETAILED EACH SYMBOL
# ============================================================

print()
print("=" * 70)
print("DETAILED SYMBOL REPORT")
print("=" * 70)

for _, row in symbol_df.iterrows():

    print()
    print(
        f"### {row['Symbol']}"
    )

    print(
        f"Trades       : "
        f"{int(row['Trades'])}"
    )

    print(
        f"Wins         : "
        f"{int(row['Wins'])}"
    )

    print(
        f"Losses       : "
        f"{int(row['Losses'])}"
    )

    print(
        f"Win Rate     : "
        f"{row['WinRate']:.2f}%"
    )

    print(
        f"PnL          : "
        f"${row['PnL']:,.2f}"
    )

    print(
        f"Net R        : "
        f"{row['NetR']:.2f}R"
    )

    print(
        f"Average R    : "
        f"{row['AvgR']:.3f}"
    )

    print(
        f"Average PnL  : "
        f"${row['AvgPnL']:,.2f}"
    )

    print(
        f"Best Trade   : "
        f"${row['BestTrade']:,.2f}"
    )

    print(
        f"Worst Trade  : "
        f"${row['WorstTrade']:,.2f}"
    )

    print(
        f"Max Loss Streak: "
        f"{int(row['MaxLossStreak'])}"
    )

    print(
        f"LONG         : "
        f"{int(row['LongTrades'])} trades | "
        f"WR={row['LongWR']:.2f}% | "
        f"PnL=${row['LongPnL']:,.2f}"
    )

    print(
        f"SHORT        : "
        f"{int(row['ShortTrades'])} trades | "
        f"WR={row['ShortWR']:.2f}% | "
        f"PnL=${row['ShortPnL']:,.2f}"
    )


# ============================================================
# EXIT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("EXIT REASON ANALYSIS")
print("=" * 70)

for _, row in exit_df.iterrows():

    print(
        f"{row['Reason']:<10}"
        f" Trades={int(row['Trades']):>5}"
        f" WR={row['WinRate']:>8.2f}%"
        f" PnL=${row['PnL']:>12,.2f}"
        f" R={row['NetR']:>9.2f}"
    )


# ============================================================
# FILTER DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("FILTER DIAGNOSTICS")
print("=" * 70)

for key, value in filter_stats.items():

    print(
        f"{key:<25}: {value}"
    )


# ============================================================
# WEAK COINS
# ============================================================

print()
print("=" * 70)
print("WEAK COIN DIAGNOSTICS")
print("=" * 70)

for threshold in [
    50,
    55,
    60,
]:

    weak = symbol_df[
        (
            symbol_df["Trades"] >= 5
        )
        &
        (
            symbol_df["WinRate"]
            <
            threshold
        )
    ]

    print()
    print(
        f"Below {threshold}% WR "
        f"(minimum 5 trades):"
    )

    if weak.empty:

        print("  None")

    else:

        print(
            "  "
            +
            ", ".join(
                weak["Symbol"].tolist()
            )
        )


# ============================================================
# TOP COINS
# ============================================================

print()
print("=" * 70)
print("TOP 10 BY PNL")
print("=" * 70)

for _, row in (
    symbol_df
    .sort_values(
        "PnL",
        ascending=False
    )
    .head(10)
    .iterrows()
):

    print(
        f"{row['Symbol']:<10}"
        f" PnL=${row['PnL']:>12,.2f}"
        f" WR={row['WinRate']:>7.2f}%"
        f" Trades={int(row['Trades'])}"
    )


# ============================================================
# WORST COINS
# ============================================================

print()
print("=" * 70)
print("BOTTOM 10 BY PNL")
print("=" * 70)

for _, row in (
    symbol_df
    .sort_values(
        "PnL",
        ascending=True
    )
    .head(10)
    .iterrows()
):

    print(
        f"{row['Symbol']:<10}"
        f" PnL=${row['PnL']:>12,.2f}"
        f" WR={row['WinRate']:>7.2f}%"
        f" Trades={int(row['Trades'])}"
    )


# ============================================================
# SAVE CSV
# ============================================================

trades_df.to_csv(
    "hunter_v77_trades.csv",
    index=False
)

symbol_df.to_csv(
    "hunter_v77_symbols.csv",
    index=False
)

streak_df.to_csv(
    "hunter_v77_loss_streaks.csv",
    index=False
)

exit_df.to_csv(
    "hunter_v77_exit_stats.csv",
    index=False
)

equity_df.to_csv(
    "hunter_v77_equity.csv",
    index=False
)

filter_df = pd.DataFrame(
    [
        {
            "Filter": key,
            "Count": value,
        }
        for key, value
        in filter_stats.items()
    ]
)

filter_df.to_csv(
    "hunter_v77_filter_diagnostics.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("CSV FILES")
print("=" * 70)

print(
    "hunter_v77_trades.csv"
)

print(
    "hunter_v77_symbols.csv"
)

print(
    "hunter_v77_loss_streaks.csv"
)

print(
    "hunter_v77_exit_stats.csv"
)

print(
    "hunter_v77_equity.csv"
)

print(
    "hunter_v77_filter_diagnostics.csv"
)

print()
print("=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)
