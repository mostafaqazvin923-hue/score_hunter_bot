# ============================================================
# HUNTER-X CLEAN V5
# LBank 1H | Closed 4H Regime | NO LOOKAHEAD
# Breakout -> Retest -> Confirmation State Machine
# ============================================================

import os
import sys
import subprocess
import time

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

try:
    import pandas as pd
    import numpy as np
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pandas", "numpy"
    ])
    import pandas as pd
    import numpy as np


# ============================================================
# SETTINGS
# ============================================================

EXCHANGE_ID = "lbank"

DAYS = 365
TIMEFRAME = "1h"
LIMIT = 500

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "NEAR/USDT",
    "SUI/USDT",
    "DOT/USDT",
    "SHIB/USDT",
    "PEPE/USDT",
    "ARB/USDT",
    "OP/USDT",
    "POL/USDT",
    "ATOM/USDT",
    "RENDER/USDT",
    "INJ/USDT",
    "FET/USDT",
    "APT/USDT",
    "TIA/USDT",
    "ICP/USDT",
    "BCH/USDT",
    "LTC/USDT",
    "CRV/USDT",
    "PENDLE/USDT",
    "AAVE/USDT",
    "GRT/USDT",
    "XLM/USDT"
]


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

STRUCTURE_LOOKBACK = 15

# Breakout quality
BREAKOUT_VOLUME_MULTIPLIER = 1.25
BREAKOUT_MIN_BODY_RATIO = 0.55
BREAKOUT_MIN_CLOSE_LOCATION = 0.70

# Retest
MAX_RETEST_CANDLES = 8
RETEST_TOLERANCE = 0.0020
RETEST_MAX_DEPTH = 0.0030

# Confirmation
CONFIRM_MIN_BODY_RATIO = 0.45
CONFIRM_MIN_CLOSE_LOCATION = 0.65

LONG_RSI_CONFIRM = 52
SHORT_RSI_CONFIRM = 48

# 4H trend filter
ADX_4H_MIN = 22

# Risk
ATR_STOP_MULTIPLIER = 0.25
MAX_RISK_PERCENT = 0.045

# Target
TARGET_RR = 2.0

# Maximum holding
MAX_HOLDING_CANDLES = 40

# EMA200 warmup
MIN_4H_WARMUP = 230


# ============================================================
# OUTPUT
# ============================================================

TRADES_CSV = "HUNTER_X_CLEAN_V5_TRADES.csv"
SYMBOL_REPORT_CSV = "HUNTER_X_CLEAN_V5_SYMBOL_REPORT.csv"


# ============================================================
# INDICATORS
# ============================================================

def wilder_rma(series, length):
    return series.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()


def rsi_wilder(close, length=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder_rma(gain, length)
    avg_loss = wilder_rma(loss, length)

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)

    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss == 0)),
        50
    )

    return rsi


def atr_wilder(df, length=14):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return wilder_rma(tr, length)


def adx_wilder(df, length=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = wilder_rma(tr, length)

    plus_di = (
        100
        * wilder_rma(plus_dm, length)
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * wilder_rma(minus_dm, length)
        / atr.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = wilder_rma(dx, length)

    return adx, plus_di, minus_di


def add_indicators(df):

    df = df.copy()

    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["RSI"] = rsi_wilder(
        df["close"],
        14
    )

    df["ATR"] = atr_wilder(
        df,
        14
    )

    (
        df["ADX"],
        df["DI_PLUS"],
        df["DI_MINUS"]
    ) = adx_wilder(
        df,
        14
    )

    df["VolumeAvg20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    return df


# ============================================================
# 4H CONTEXT
# ============================================================

def add_4h_context(df):

    h4 = (
        df.set_index("Date")
        .resample(
            "4h",
            label="left",
            closed="left"
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }
        )
        .dropna()
        .reset_index()
    )

    h4 = add_indicators(h4)

    # Exact time when 4H candle becomes closed/known
    h4["CloseTime"] = (
        h4["Date"]
        + pd.Timedelta(hours=4)
    )

    # Previous EMA200 from the SAME 4H table
    h4["EMA200_PREV"] = (
        h4["EMA200"].shift(1)
    )

    # --------------------------------------------------------
    # LONG 4H REGIME
    # --------------------------------------------------------

    h4["LONG_REGIME"] = (
        (h4["close"] > h4["EMA200"])
        &
        (h4["EMA20"] > h4["EMA50"])
        &
        (h4["EMA50"] > h4["EMA200"])
        &
        (h4["EMA200"] > h4["EMA200_PREV"])
        &
        (h4["ADX"] >= ADX_4H_MIN)
        &
        (h4["DI_PLUS"] > h4["DI_MINUS"])
        &
        (h4["RSI"] > 55)
    )

    # --------------------------------------------------------
    # SHORT 4H REGIME
    # --------------------------------------------------------

    h4["SHORT_REGIME"] = (
        (h4["close"] < h4["EMA200"])
        &
        (h4["EMA20"] < h4["EMA50"])
        &
        (h4["EMA50"] < h4["EMA200"])
        &
        (h4["EMA200"] < h4["EMA200_PREV"])
        &
        (h4["ADX"] >= ADX_4H_MIN)
        &
        (h4["DI_MINUS"] > h4["DI_PLUS"])
        &
        (h4["RSI"] < 45)
    )

    # --------------------------------------------------------
    # EMA200 WARMUP
    # --------------------------------------------------------

    h4 = h4.iloc[
        MIN_4H_WARMUP:
    ].copy()

    # --------------------------------------------------------
    # 1H DECISION TIME
    # --------------------------------------------------------

    decision = df.copy()

    # 1H candle at Date covers Date -> Date+1h.
    # Decision occurs AFTER candle closes.
    decision["DecisionTime"] = (
        decision["Date"]
        + pd.Timedelta(hours=1)
    )

    # --------------------------------------------------------
    # MERGE ONLY CLOSED 4H CANDLES
    # --------------------------------------------------------

    h4_merge = h4[
        [
            "CloseTime",
            "close",
            "EMA20",
            "EMA50",
            "EMA200",
            "EMA200_PREV",
            "RSI",
            "ATR",
            "ADX",
            "DI_PLUS",
            "DI_MINUS",
            "LONG_REGIME",
            "SHORT_REGIME"
        ]
    ].copy()

    h4_merge.rename(
        columns={
            "close": "H4_Close",
            "EMA20": "H4_EMA20",
            "EMA50": "H4_EMA50",
            "EMA200": "H4_EMA200",
            "EMA200_PREV": "H4_EMA200_PREV",
            "RSI": "H4_RSI",
            "ATR": "H4_ATR",
            "ADX": "H4_ADX",
            "DI_PLUS": "H4_DI_PLUS",
            "DI_MINUS": "H4_DI_MINUS"
        },
        inplace=True
    )

    h4_merge = h4_merge.sort_values(
        "CloseTime"
    )

    decision = pd.merge_asof(
        decision.sort_values("DecisionTime"),
        h4_merge,
        left_on="DecisionTime",
        right_on="CloseTime",
        direction="backward",
        allow_exact_matches=True
    )

    return decision.reset_index(
        drop=True
    )


# ============================================================
# LBank DATA
# ============================================================

def make_exchange():

    exchange_class = getattr(
        ccxt,
        EXCHANGE_ID
    )

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap"
            }
        }
    )

    return exchange


def fetch_1h(exchange, symbol, days=365):

    print(
        f"\n📥 Downloading {symbol} ..."
    )

    now_ms = exchange.milliseconds()

    since_ms = (
        now_ms
        - days
        * 24
        * 60
        * 60
        * 1000
    )

    all_rows = []

    cursor = since_ms

    while cursor < now_ms:

        try:

            rows = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=cursor,
                limit=LIMIT
            )

            if not rows:
                break

            all_rows.extend(rows)

            last_ts = rows[-1][0]

            if last_ts <= cursor:
                break

            cursor = (
                last_ts
                + 60 * 60 * 1000
            )

            print(
                f"   {len(all_rows)} candles | "
                f"{pd.to_datetime(last_ts, unit='ms')}"
            )

            time.sleep(
                exchange.rateLimit / 1000
            )

            if len(rows) < LIMIT:
                break

        except Exception as e:

            print(
                f"⚠️ Data error {symbol}: {e}"
            )

            time.sleep(3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["Date"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    ).dt.tz_convert(None)

    df = (
        df
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove incomplete current candle
    current_floor = (
        pd.Timestamp.utcnow()
        .floor("h")
        .tz_localize(None)
    )

    df = df[
        df["Date"]
        + pd.Timedelta(hours=1)
        <= current_floor
    ].copy()

    return df[
        [
            "Date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].reset_index(drop=True)


# ============================================================
# CANDLE QUALITY
# ============================================================

def candle_body_ratio(row):

    candle_range = (
        row["high"]
        - row["low"]
    )

    if candle_range <= 0:
        return 0.0

    return (
        abs(row["close"] - row["open"])
        / candle_range
    )


def close_location(row, direction):

    candle_range = (
        row["high"]
        - row["low"]
    )

    if candle_range <= 0:
        return 0.5

    if direction == "LONG":

        return (
            row["close"]
            - row["low"]
        ) / candle_range

    return (
        row["high"]
        - row["close"]
    ) / candle_range


# ============================================================
# BREAKOUT FILTER
# ============================================================

def breakout_valid(row, direction):

    if (
        not np.isfinite(
            row["VolumeAvg20"]
        )
        or row["VolumeAvg20"] <= 0
    ):
        return False

    # Volume confirmation
    if (
        row["volume"]
        < row["VolumeAvg20"]
        * BREAKOUT_VOLUME_MULTIPLIER
    ):
        return False

    # Strong candle body
    if (
        candle_body_ratio(row)
        < BREAKOUT_MIN_BODY_RATIO
    ):
        return False

    # Close near candle extreme
    if (
        close_location(
            row,
            direction
        )
        < BREAKOUT_MIN_CLOSE_LOCATION
    ):
        return False

    return True


# ============================================================
# CONFIRMATION FILTER
# ============================================================

def confirmation_valid(row, direction):

    # Strong confirmation body
    if (
        candle_body_ratio(row)
        < CONFIRM_MIN_BODY_RATIO
    ):
        return False

    # Close near extreme
    if (
        close_location(
            row,
            direction
        )
        < CONFIRM_MIN_CLOSE_LOCATION
    ):
        return False

    if direction == "LONG":

        if row["close"] <= row["open"]:
            return False

        if row["RSI"] <= LONG_RSI_CONFIRM:
            return False

    else:

        if row["close"] >= row["open"]:
            return False

        if row["RSI"] >= SHORT_RSI_CONFIRM:
            return False

    return True


# ============================================================
# CREATE TRADE
# ============================================================

def create_trade(
    df,
    entry_i,
    direction,
    breakout_level,
    breakout_i
):

    entry = float(
        df.loc[
            entry_i,
            "close"
        ]
    )

    atr = float(
        df.loc[
            entry_i,
            "ATR"
        ]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    # Only candles from breakout to entry.
    window = df.iloc[
        breakout_i:
        entry_i + 1
    ]

    if direction == "LONG":

        swing_low = float(
            window["low"].min()
        )

        sl = (
            swing_low
            - ATR_STOP_MULTIPLIER * atr
        )

        if sl >= entry:
            return None

        risk = entry - sl

        tp = (
            entry
            + TARGET_RR * risk
        )

    else:

        swing_high = float(
            window["high"].max()
        )

        sl = (
            swing_high
            + ATR_STOP_MULTIPLIER * atr
        )

        if sl <= entry:
            return None

        risk = sl - entry

        tp = (
            entry
            - TARGET_RR * risk
        )

    risk_pct = (
        risk / entry
    )

    if (
        risk_pct <= 0
        or risk_pct > MAX_RISK_PERCENT
    ):
        return None

    return {
        "EntryIndex": entry_i,
        "BreakoutIndex": breakout_i,
        "EntryTime": df.loc[
            entry_i,
            "DecisionTime"
        ],
        "Direction": direction,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "RiskPct": risk_pct * 100,
        "RR": TARGET_RR,
        "BreakoutLevel": breakout_level
    }


# ============================================================
# FORWARD TRADE SIMULATION
# ============================================================

def simulate_trade(df, trade):

    entry_i = trade[
        "EntryIndex"
    ]

    max_i = min(
        len(df) - 1,
        entry_i
        + MAX_HOLDING_CANDLES
    )

    for j in range(
        entry_i + 1,
        max_i + 1
    ):

        row = df.loc[j]

        if trade["Direction"] == "LONG":

            hit_sl = (
                row["low"]
                <= trade["SL"]
            )

            hit_tp = (
                row["high"]
                >= trade["TP"]
            )

        else:

            hit_sl = (
                row["high"]
                >= trade["SL"]
            )

            hit_tp = (
                row["low"]
                <= trade["TP"]
            )

        # ----------------------------------------------------
        # If both happen in same candle:
        # conservative assumption = SL first.
        # ----------------------------------------------------

        if hit_sl and hit_tp:

            trade["Result"] = "LOSS"
            trade["ExitIndex"] = j
            trade["ExitTime"] = row[
                "DecisionTime"
            ]
            trade["Exit"] = trade["SL"]
            trade["R"] = -1.0

            return trade

        if hit_sl:

            trade["Result"] = "LOSS"
            trade["ExitIndex"] = j
            trade["ExitTime"] = row[
                "DecisionTime"
            ]
            trade["Exit"] = trade["SL"]
            trade["R"] = -1.0

            return trade

        if hit_tp:

            trade["Result"] = "WIN"
            trade["ExitIndex"] = j
            trade["ExitTime"] = row[
                "DecisionTime"
            ]
            trade["Exit"] = trade["TP"]
            trade["R"] = TARGET_RR

            return trade

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    j = max_i

    row = df.loc[j]

    exit_price = float(
        row["close"]
    )

    if trade["Direction"] == "LONG":

        r = (
            exit_price
            - trade["Entry"]
        ) / (
            trade["Entry"]
            - trade["SL"]
        )

    else:

        r = (
            trade["Entry"]
            - exit_price
        ) / (
            trade["SL"]
            - trade["Entry"]
        )

    trade["Result"] = "TIMEOUT"
    trade["ExitIndex"] = j
    trade["ExitTime"] = row[
        "DecisionTime"
    ]
    trade["Exit"] = exit_price
    trade["R"] = float(r)

    return trade


# ============================================================
# MAIN BACKTEST — TRUE STATE MACHINE
# ============================================================

def backtest_symbol(df, symbol):

    if df.empty:
        return []

    # 1H indicators
    df = add_indicators(df)

    # Closed 4H context
    df = add_4h_context(df)

    if len(df) < 500:
        return []

    trades = []

    # --------------------------------------------------------
    # STATE
    #
    # pending = None
    #
    # OR:
    # {
    #   direction,
    #   breakout_i,
    #   breakout_time,
    #   level
    # }
    # --------------------------------------------------------

    pending = None

    active_trade = None

    i = max(
        250,
        STRUCTURE_LOOKBACK + 20
    )

    while i < len(df):

        row = df.loc[i]

        # ====================================================
        # ACTIVE TRADE
        # ====================================================

        if active_trade is not None:

            completed = simulate_trade(
                df,
                active_trade
            )

            completed["Symbol"] = symbol

            trades.append(
                completed
            )

            exit_i = completed[
                "ExitIndex"
            ]

            active_trade = None
            pending = None

            # No new signal while trade is open
            i = exit_i + 1

            continue

        # ====================================================
        # PENDING BREAKOUT
        # ====================================================

        if pending is not None:

            age = (
                i
                - pending["breakout_i"]
            )

            # Breakout candle itself is not retest candle
            if age <= 0:

                i += 1
                continue

            # Retest window expired
            if age > MAX_RETEST_CANDLES:

                pending = None

                i += 1
                continue

            # ------------------------------------------------
            # 4H regime must STILL agree.
            # This uses current closed 4H context only.
            # ------------------------------------------------

            if pending["direction"] == "LONG":

                if not bool(
                    row.get(
                        "LONG_REGIME",
                        False
                    )
                ):

                    pending = None

                    i += 1
                    continue

            else:

                if not bool(
                    row.get(
                        "SHORT_REGIME",
                        False
                    )
                ):

                    pending = None

                    i += 1
                    continue

            level = pending[
                "level"
            ]

            # =================================================
            # LONG RETEST
            # =================================================

            if pending["direction"] == "LONG":

                # Price reaches broken resistance
                touched = (
                    row["low"]
                    <= level
                    * (
                        1
                        + RETEST_TOLERANCE
                    )
                )

                # Reject if price penetrates too deeply
                too_deep = (
                    row["low"]
                    <
                    level
                    * (
                        1
                        - RETEST_MAX_DEPTH
                    )
                )

                if too_deep:

                    pending = None

                    i += 1
                    continue

                # Candle must close back above level
                held = (
                    row["close"]
                    > level
                )

                if (
                    touched
                    and held
                    and confirmation_valid(
                        row,
                        "LONG"
                    )
                ):

                    active_trade = create_trade(
                        df=df,
                        entry_i=i,
                        direction="LONG",
                        breakout_level=level,
                        breakout_i=pending[
                            "breakout_i"
                        ]
                    )

                    if active_trade is not None:

                        continue

                    pending = None

                    i += 1
                    continue

            # =================================================
            # SHORT RETEST
            # =================================================

            else:

                touched = (
                    row["high"]
                    >= level
                    * (
                        1
                        - RETEST_TOLERANCE
                    )
                )

                too_deep = (
                    row["high"]
                    >
                    level
                    * (
                        1
                        + RETEST_MAX_DEPTH
                    )
                )

                if too_deep:

                    pending = None

                    i += 1
                    continue

                # Candle must close back below level
                held = (
                    row["close"]
                    < level
                )

                if (
                    touched
                    and held
                    and confirmation_valid(
                        row,
                        "SHORT"
                    )
                ):

                    active_trade = create_trade(
                        df=df,
                        entry_i=i,
                        direction="SHORT",
                        breakout_level=level,
                        breakout_i=pending[
                            "breakout_i"
                        ]
                    )

                    if active_trade is not None:

                        continue

                    pending = None

                    i += 1
                    continue

            i += 1

            continue

        # ====================================================
        # NO PENDING BREAKOUT
        # SEARCH CURRENT CANDLE ONLY
        # ====================================================

        if i < STRUCTURE_LOOKBACK:

            i += 1
            continue

        # IMPORTANT:
        # Current candle is NOT part of structure.
        structure = df.iloc[
            i - STRUCTURE_LOOKBACK:
            i
        ]

        struct_high = float(
            structure["high"].max()
        )

        struct_low = float(
            structure["low"].min()
        )

        long_regime = bool(
            row.get(
                "LONG_REGIME",
                False
            )
        )

        short_regime = bool(
            row.get(
                "SHORT_REGIME",
                False
            )
        )

        # ----------------------------------------------------
        # LONG BREAKOUT
        # ----------------------------------------------------

        long_breakout = (
            long_regime
            and row["close"] > struct_high
            and breakout_valid(
                row,
                "LONG"
            )
        )

        # ----------------------------------------------------
        # SHORT BREAKOUT
        # ----------------------------------------------------

        short_breakout = (
            short_regime
            and row["close"] < struct_low
            and breakout_valid(
                row,
                "SHORT"
            )
        )

        if long_breakout:

            pending = {
                "direction": "LONG",
                "breakout_i": i,
                "breakout_time": row[
                    "DecisionTime"
                ],
                "level": struct_high
            }

        elif short_breakout:

            pending = {
                "direction": "SHORT",
                "breakout_i": i,
                "breakout_time": row[
                    "DecisionTime"
                ],
                "level": struct_low
            }

        i += 1

    return trades


# ============================================================
# REPORT
# ============================================================

def build_report(all_trades):

    if not all_trades:

        print(
            "\n❌ No trades found."
        )

        return

    trades = pd.DataFrame(
        all_trades
    )

    trades["EntryTime"] = pd.to_datetime(
        trades["EntryTime"]
    )

    trades["ExitTime"] = pd.to_datetime(
        trades["ExitTime"]
    )

    # ========================================================
    # IMPORTANT:
    # Sort chronologically BEFORE portfolio equity/DD.
    # ========================================================

    trades = (
        trades
        .sort_values(
            [
                "EntryTime",
                "ExitTime",
                "Symbol"
            ]
        )
        .reset_index(drop=True)
    )

    total = len(trades)

    wins = int(
        (
            trades["Result"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            trades["Result"]
            == "LOSS"
        ).sum()
    )

    timeouts = int(
        (
            trades["Result"]
            == "TIMEOUT"
        ).sum()
    )

    decisive = (
        wins
        + losses
    )

    # --------------------------------------------------------
    # WIN RATE EXCLUDING TIMEOUT
    # --------------------------------------------------------

    decisive_wr = (
        100 * wins / decisive
        if decisive > 0
        else 0
    )

    # --------------------------------------------------------
    # WIN RATE INCLUDING TIMEOUTS
    # --------------------------------------------------------

    all_trade_positive = (
        100 * wins / total
        if total > 0
        else 0
    )

    # --------------------------------------------------------
    # NET R
    # --------------------------------------------------------

    net_r = float(
        trades["R"].sum()
    )

    # --------------------------------------------------------
    # PROFIT FACTOR
    # --------------------------------------------------------

    gross_profit = float(
        trades.loc[
            trades["R"] > 0,
            "R"
        ].sum()
    )

    gross_loss = abs(
        float(
            trades.loc[
                trades["R"] < 0,
                "R"
            ].sum()
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = np.inf

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    trades["EquityR"] = (
        trades["R"].cumsum()
    )

    trades["PeakR"] = (
        trades["EquityR"].cummax()
    )

    trades["DrawdownR"] = (
        trades["EquityR"]
        - trades["PeakR"]
    )

    max_dd = float(
        trades["DrawdownR"].min()
    )

    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    avg_win = (
        float(
            trades.loc[
                trades["Result"] == "WIN",
                "R"
            ].mean()
        )
        if wins
        else 0
    )

    avg_loss = (
        float(
            trades.loc[
                trades["Result"] == "LOSS",
                "R"
            ].mean()
        )
        if losses
        else 0
    )

    avg_timeout = (
        float(
            trades.loc[
                trades["Result"] == "TIMEOUT",
                "R"
            ].mean()
        )
        if timeouts
        else 0
    )

    expectancy = (
        net_r / total
        if total
        else 0
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "🏹 HUNTER-X CLEAN V5"
    )
    print(
        "CAUSAL STATE-MACHINE BACKTEST"
    )
    print("=" * 70)

    print(
        f"Total Trades             : {total}"
    )

    print(
        f"Wins                     : {wins}"
    )

    print(
        f"Losses                   : {losses}"
    )

    print(
        f"Timeouts                 : {timeouts}"
    )

    print(
        f"Win Rate (W/L only)      : "
        f"{decisive_wr:.2f}%"
    )

    print(
        f"Win Rate (all trades)    : "
        f"{all_trade_positive:.2f}%"
    )

    print(
        f"Net Profit               : "
        f"{net_r:.2f}R"
    )

    print(
        f"Profit Factor            : "
        f"{profit_factor:.2f}"
    )

    print(
        f"Expectancy / trade       : "
        f"{expectancy:.4f}R"
    )

    print(
        f"Average Win              : "
        f"{avg_win:.2f}R"
    )

    print(
        f"Average Loss             : "
        f"{avg_loss:.2f}R"
    )

    print(
        f"Average Timeout          : "
        f"{avg_timeout:.2f}R"
    )

    print(
        f"Max Drawdown             : "
        f"{max_dd:.2f}R"
    )

    print("=" * 70)

    # ========================================================
    # SYMBOL REPORT
    # ========================================================

    rows = []

    for symbol, g in trades.groupby(
        "Symbol",
        sort=False
    ):

        w = int(
            (
                g["Result"]
                == "WIN"
            ).sum()
        )

        l = int(
            (
                g["Result"]
                == "LOSS"
            ).sum()
        )

        t = int(
            (
                g["Result"]
                == "TIMEOUT"
            ).sum()
        )

        d = w + l

        rows.append(
            {
                "Symbol": symbol,
                "Trades": len(g),
                "Wins": w,
                "Losses": l,
                "Timeouts": t,

                "WinRate_WL":
                    100 * w / d
                    if d
                    else 0,

                "WinRate_All":
                    100 * w / len(g)
                    if len(g)
                    else 0,

                "NetR":
                    g["R"].sum(),

                "AvgR":
                    g["R"].mean(),

                "AvgTimeoutR":
                    (
                        g.loc[
                            g["Result"]
                            == "TIMEOUT",
                            "R"
                        ].mean()
                        if t
                        else 0
                    )
            }
        )

    symbol_report = pd.DataFrame(
        rows
    ).sort_values(
        "NetR",
        ascending=False
    )

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        TRADES_CSV,
        index=False
    )

    symbol_report.to_csv(
        SYMBOL_REPORT_CSV,
        index=False
    )

    print(
        "\n📊 SYMBOL REPORT"
    )

    print(
        symbol_report.to_string(
            index=False
        )
    )

    print(
        f"\n💾 Saved: {TRADES_CSV}"
    )

    print(
        f"💾 Saved: {SYMBOL_REPORT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "🏹 HUNTER-X CLEAN V5"
    )
    print(
        "LBank | 1H EXECUTION | "
        "CLOSED 4H REGIME | NO LOOKAHEAD"
    )
    print("=" * 70)

    exchange = make_exchange()

    all_trades = []

    for symbol in SYMBOLS:

        try:

            df = fetch_1h(
                exchange,
                symbol,
                DAYS
            )

            if df.empty:

                print(
                    f"⚠️ {symbol}: No data"
                )

                continue

            print(
                f"📈 {symbol}: "
                f"{len(df)} candles | "
                f"{df['Date'].min()} -> "
                f"{df['Date'].max()}"
            )

            symbol_trades = (
                backtest_symbol(
                    df,
                    symbol
                )
            )

            print(
                f"   ✅ {symbol}: "
                f"{len(symbol_trades)} trades"
            )

            all_trades.extend(
                symbol_trades
            )

        except Exception as e:

            print(
                f"❌ {symbol}: "
                f"{type(e).__name__}: {e}"
            )

    build_report(
        all_trades
    )


if __name__ == "__main__":
    main()
