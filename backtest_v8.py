import os
import subprocess
import sys
from datetime import datetime, timedelta

# ============================================================
# نصب خودکار کتابخانه‌ها
# ============================================================

try:
    import ccxt
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "ccxt"
    ])
    import ccxt

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pandas"
    ])
    import pandas as pd

try:
    import numpy as np
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "numpy"
    ])
    import numpy as np


# ============================================================
# LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000
})


SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT"
}


BACKTEST_DAYS = int(
    os.getenv("BACKTEST_DAYS", "365")
)

FETCH_LIMIT = 1000


# ============================================================
# V10 SETTINGS
# ============================================================

# RR = 1:2
RISK_REWARD = 2.0

# حداقل فاصله SL
MIN_SL_PCT = 0.0035

# حداکثر فاصله SL
MAX_SL_PCT = 0.0150

# ATR multiplier برای SL
ATR_SL_MULT = 0.80

# حداکثر زمان نگهداری
MAX_HOLD_BARS = 24

# کارمزد هر طرف
FEE_RATE = 0.0006

# اسلیپیج واقعی
SLIPPAGE_RATE = 0.0002

# فیلترهای استراتژی
H4_ADX_MIN = 18
H4_RSI_LONG_MIN = 52
H4_RSI_SHORT_MAX = 48

# شیب EMA
H4_EMA20_SLOPE_MIN = 0.0005

# 1H RSI
RSI_LONG_MIN = 48
RSI_LONG_MAX = 68

RSI_SHORT_MIN = 32
RSI_SHORT_MAX = 52

# حجم
VOLUME_MIN = 0.80

# کیفیت کندل
MIN_BODY_RATIO = 0.35

# خروج timeout
TIMEOUT_IS_LOSS = True

# اگر True باشد فقط یک معامله در کل پرتفوی
# اگر False باشد هر نماد فقط یک معامله همزمان دارد
PORTFOLIO_LOCK = False

OUTPUT_CSV = "hunter_x_v10_lbank_trades.csv"


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = df["Close"].shift(1)

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = (
        df["High"]
        - prev_close
    ).abs()

    tr3 = (
        df["Low"]
        - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(
        14
    ).mean()

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["VOL_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOL_RATIO"] = (
        df["Volume"]
        / df["VOL_MA20"]
    )

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    df["RANGE"] = (
        df["High"]
        - df["Low"]
    )

    df["BODY"] = (
        df["Close"]
        - df["Open"]
    ).abs()

    df["BODY_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0
    )

    # --------------------------------------------------------
    # Upper / Lower wick
    # --------------------------------------------------------

    df["UPPER_WICK"] = (
        df["High"]
        - df[
            ["Open", "Close"]
        ].max(axis=1)
    )

    df["LOWER_WICK"] = (
        df[
            ["Open", "Close"]
        ].min(axis=1)
        - df["Low"]
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where(
        (up_move > down_move)
        & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move)
        & (down_move > 0),
        down_move,
        0
    )

    atr14 = df["ATR"]

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index
        ).rolling(14).mean()
        / atr14
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index
        ).rolling(14).mean()
        / atr14
    )

    dx = (
        100
        * (
            plus_di
            - minus_di
        ).abs()
        /
        (
            plus_di
            + minus_di
        ).replace(
            0,
            np.nan
        )
    )

    df["ADX"] = dx.rolling(
        14
    ).mean()

    # --------------------------------------------------------
    # EMA slope
    # --------------------------------------------------------

    df["EMA20_SLOPE"] = (
        (
            df["EMA20"]
            / df["EMA20"].shift(3)
        )
        - 1
    )

    # --------------------------------------------------------
    # Recent swing levels
    # --------------------------------------------------------

    df["SWING_LOW_6"] = (
        df["Low"]
        .shift(1)
        .rolling(6)
        .min()
    )

    df["SWING_HIGH_6"] = (
        df["High"]
        .shift(1)
        .rolling(6)
        .max()
    )

    return df


# ============================================================
# FETCH LBank 1H
# ============================================================

def fetch_lbank_1h(symbol, days=365):

    print("")
    print("=" * 60)
    print(
        f"📥 دریافت {symbol} از LBank"
    )
    print("=" * 60)

    start_date = (
        datetime.utcnow()
        - timedelta(days=days + 10)
    )

    since_timestamp = int(
        start_date.timestamp()
        * 1000
    )

    now_timestamp = (
        exchange.milliseconds()
    )

    all_ohlcv = []

    current_since = (
        since_timestamp
    )

    request_number = 0

    while (
        current_since
        < now_timestamp
    ):

        request_number += 1

        try:

            print(
                f"🔄 درخواست #{request_number} | "
                f"تعداد فعلی: "
                f"{len(all_ohlcv)}"
            )

            ohlcv = (
                exchange.fetch_ohlcv(
                    symbol,
                    timeframe="1h",
                    since=current_since,
                    limit=FETCH_LIMIT
                )
            )

            if not ohlcv:

                print(
                    "⚠️ دیتای بیشتری دریافت نشد."
                )

                break

            all_ohlcv.extend(
                ohlcv
            )

            current_since = (
                ohlcv[-1][0] + 1
            )

            print(
                f"✔️ +{len(ohlcv)} | "
                f"مجموع {len(all_ohlcv)}"
            )

            if len(ohlcv) < FETCH_LIMIT:
                break

        except Exception as e:

            print(
                f"❌ خطا: {e}"
            )

            break

    if not all_ohlcv:

        return None

    # --------------------------------------------------------
    # حذف duplicate
    # --------------------------------------------------------

    unique = {}

    for row in all_ohlcv:

        unique[row[0]] = row

    all_ohlcv = list(
        unique.values()
    )

    all_ohlcv.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    df["Date"] = (
        pd.to_datetime(
            df["Timestamp"],
            unit="ms",
            utc=True
        )
        .dt
        .tz_localize(None)
    )

    df.set_index(
        "Date",
        inplace=True
    )

    # --------------------------------------------------------
    # فقط 365 روز
    # --------------------------------------------------------

    now_naive = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .tz_localize(None)
    )

    cutoff = (
        now_naive
        - pd.Timedelta(
            days=days
        )
    )

    df = df[
        df.index >= cutoff
    ].copy()

    # --------------------------------------------------------
    # حذف کندل جاری
    # --------------------------------------------------------

    current_hour = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .tz_localize(None)
        .replace(
            minute=0,
            second=0,
            microsecond=0
        )
    )

    df = df[
        df.index < current_hour
    ].copy()

    print(
        f"✅ {symbol}: "
        f"{len(df)} کندل"
    )

    return df


# ============================================================
# 4H
# ============================================================

def make_4h(df1h):

    df4h = (
        df1h
        .resample("4h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna()
    )

    return df4h


# ============================================================
# MTF ALIGNMENT
#
# هر 1H فقط از آخرین 4H کاملاً بسته‌شده استفاده می‌کند.
# ============================================================

def attach_previous_4h(
    df1h,
    df4h
):

    h4 = calculate_indicators(
        df4h
    )

    h4 = h4.copy()

    # زمان بسته‌شدن واقعی 4H
    h4["H4_CLOSE_TIME"] = (
        h4.index
        + pd.Timedelta(hours=4)
    )

    h4_features = h4[
        [
            "H4_CLOSE_TIME",
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "ADX",
            "EMA20_SLOPE",
            "ATR"
        ]
    ].copy()

    h4_features = (
        h4_features
        .rename(
            columns={
                "EMA20":
                    "H4_EMA20",
                "EMA50":
                    "H4_EMA50",
                "EMA200":
                    "H4_EMA200",
                "RSI":
                    "H4_RSI",
                "ADX":
                    "H4_ADX",
                "EMA20_SLOPE":
                    "H4_EMA20_SLOPE",
                "ATR":
                    "H4_ATR"
            }
        )
    )

    h4_features = (
        h4_features
        .sort_values(
            "H4_CLOSE_TIME"
        )
    )

    result = (
        df1h
        .reset_index()
        .rename(
            columns={
                "Date": "Time"
            }
        )
        .sort_values(
            "Time"
        )
    )

    # فقط 4Hهایی که قبل از شروع
    # کندل 1H کاملاً بسته شده‌اند
    result = pd.merge_asof(
        result,
        h4_features,
        left_on="Time",
        right_on="H4_CLOSE_TIME",
        direction="backward",
        allow_exact_matches=True
    )

    result.set_index(
        "Time",
        inplace=True
    )

    result.drop(
        columns=[
            "H4_CLOSE_TIME"
        ],
        inplace=True,
        errors="ignore"
    )

    return result


# ============================================================
# ساخت SL/TP
# ============================================================

def calculate_levels(
    df,
    signal_idx,
    side,
    entry_price
):

    row = df.iloc[
        signal_idx
    ]

    atr = row["ATR"]

    if pd.isna(atr) or atr <= 0:
        return None

    # --------------------------------------------------------
    # ساختار
    # --------------------------------------------------------

    if side == "LONG":

        structure_sl = (
            row["SWING_LOW_6"]
        )

        if pd.isna(structure_sl):
            return None

        structure_distance = (
            entry_price
            - structure_sl
        )

        atr_distance = (
            atr
            * ATR_SL_MULT
        )

        risk_distance = max(
            structure_distance,
            atr_distance
        )

        sl_price = (
            entry_price
            - risk_distance
        )

        risk_pct = (
            entry_price
            - sl_price
        ) / entry_price

        if (
            risk_pct < MIN_SL_PCT
            or risk_pct > MAX_SL_PCT
        ):
            return None

        tp_price = (
            entry_price
            + (
                risk_distance
                * RISK_REWARD
            )
        )

    else:

        structure_sl = (
            row["SWING_HIGH_6"]
        )

        if pd.isna(structure_sl):
            return None

        structure_distance = (
            structure_sl
            - entry_price
        )

        atr_distance = (
            atr
            * ATR_SL_MULT
        )

        risk_distance = max(
            structure_distance,
            atr_distance
        )

        sl_price = (
            entry_price
            + risk_distance
        )

        risk_pct = (
            sl_price
            - entry_price
        ) / entry_price

        if (
            risk_pct < MIN_SL_PCT
            or risk_pct > MAX_SL_PCT
        ):
            return None

        tp_price = (
            entry_price
            - (
                risk_distance
                * RISK_REWARD
            )
        )

    return {
        "SL": sl_price,
        "TP": tp_price,
        "RiskPct": risk_pct
    }


# ============================================================
# SIMULATE
# ============================================================

def simulate_trade(
    df,
    entry_idx,
    side,
    entry_price,
    sl_price,
    tp_price,
    risk_pct
):

    max_exit = min(
        entry_idx
        + MAX_HOLD_BARS,
        len(df) - 1
    )

    outcome = None
    exit_idx = None
    raw_exit_price = None

    for j in range(
        entry_idx,
        max_exit + 1
    ):

        candle = df.iloc[j]

        high = candle["High"]
        low = candle["Low"]

        if side == "LONG":

            hit_sl = (
                low <= sl_price
            )

            hit_tp = (
                high >= tp_price
            )

        else:

            hit_sl = (
                high >= sl_price
            )

            hit_tp = (
                low <= tp_price
            )

        # ----------------------------------------------------
        # Same candle = LOSS
        # ----------------------------------------------------

        if hit_sl and hit_tp:

            outcome = "LOSS"

            raw_exit_price = (
                sl_price
            )

            exit_idx = j

            break

        if hit_sl:

            outcome = "LOSS"

            raw_exit_price = (
                sl_price
            )

            exit_idx = j

            break

        if hit_tp:

            outcome = "WIN"

            raw_exit_price = (
                tp_price
            )

            exit_idx = j

            break

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    if outcome is None:

        exit_idx = max_exit

        raw_exit_price = (
            df.iloc[
                exit_idx
            ]["Close"]
        )

        outcome = "TIMEOUT"

    # --------------------------------------------------------
    # Slippage
    #
    # اسلیپیج فقط یک بار اعمال می‌شود.
    # --------------------------------------------------------

    if side == "LONG":

        if outcome == "WIN":

            exit_price = (
                raw_exit_price
                * (
                    1
                    - SLIPPAGE_RATE
                )
            )

        elif outcome == "LOSS":

            exit_price = (
                raw_exit_price
                * (
                    1
                    - SLIPPAGE_RATE
                )
            )

        else:

            exit_price = (
                raw_exit_price
                * (
                    1
                    - SLIPPAGE_RATE
                )
            )

        entry_execution = (
            entry_price
            * (
                1
                + SLIPPAGE_RATE
            )
        )

        gross_return = (
            exit_price
            / entry_execution
        ) - 1

    else:

        if outcome == "WIN":

            exit_price = (
                raw_exit_price
                * (
                    1
                    + SLIPPAGE_RATE
                )
            )

        elif outcome == "LOSS":

            exit_price = (
                raw_exit_price
                * (
                    1
                    + SLIPPAGE_RATE
                )
            )

        else:

            exit_price = (
                raw_exit_price
                * (
                    1
                    + SLIPPAGE_RATE
                )
            )

        entry_execution = (
            entry_price
            * (
                1
                - SLIPPAGE_RATE
            )
        )

        gross_return = (
            entry_execution
            / exit_price
        ) - 1

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    net_return = (
        gross_return
        - (
            FEE_RATE * 2
        )
    )

    # --------------------------------------------------------
    # R
    # --------------------------------------------------------

    net_r = (
        net_return
        / risk_pct
    )

    return {
        "EntryIndex":
            entry_idx,

        "ExitIndex":
            exit_idx,

        "Side":
            side,

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "SL":
            sl_price,

        "TP":
            tp_price,

        "RiskPct":
            risk_pct * 100,

        "Outcome":
            outcome,

        "GrossReturn":
            gross_return * 100,

        "NetReturn":
            net_return * 100,

        "R":
            net_r
    }


# ============================================================
# BACKTEST SYMBOL
# ============================================================

def backtest_symbol(
    symbol,
    df
):

    print("")
    print(
        f"🚀 V10 BACKTEST: {symbol}"
    )

    trades = []

    start_idx = 300

    locked_until = -1

    for i in range(
        start_idx,
        len(df)
        - MAX_HOLD_BARS
        - 2
    ):

        if i < locked_until:
            continue

        row = df.iloc[i]

        previous = df.iloc[
            i - 1
        ]

        # ====================================================
        # H4 REGIME
        # ====================================================

        h4_ema20 = row[
            "H4_EMA20"
        ]

        h4_ema50 = row[
            "H4_EMA50"
        ]

        h4_ema200 = row[
            "H4_EMA200"
        ]

        h4_rsi = row[
            "H4_RSI"
        ]

        h4_adx = row[
            "H4_ADX"
        ]

        h4_slope = row[
            "H4_EMA20_SLOPE"
        ]

        if any(
            pd.isna(x)
            for x in [
                h4_ema20,
                h4_ema50,
                h4_ema200,
                h4_rsi,
                h4_adx,
                h4_slope
            ]
        ):
            continue

        # ====================================================
        # LONG REGIME
        # ====================================================

        long_regime = (
            h4_ema20
            > h4_ema50
            > h4_ema200
            and
            h4_rsi
            >= H4_RSI_LONG_MIN
            and
            h4_adx
            >= H4_ADX_MIN
            and
            h4_slope
            >= H4_EMA20_SLOPE_MIN
        )

        # ====================================================
        # SHORT REGIME
        # ====================================================

        short_regime = (
            h4_ema20
            < h4_ema50
            < h4_ema200
            and
            h4_rsi
            <= H4_RSI_SHORT_MAX
            and
            h4_adx
            >= H4_ADX_MIN
            and
            h4_slope
            <= -H4_EMA20_SLOPE_MIN
        )

        # ====================================================
        # 1H INDICATORS
        # ====================================================

        atr = row["ATR"]

        if (
            pd.isna(atr)
            or atr <= 0
        ):
            continue

        if pd.isna(
            row["VOL_RATIO"]
        ):
            continue

        if (
            row["BODY_RATIO"]
            < MIN_BODY_RATIO
        ):
            continue

        # ====================================================
        # Volume
        # ====================================================

        volume_ok = (
            row["VOL_RATIO"]
            >= VOLUME_MIN
        )

        if not volume_ok:
            continue

        # ====================================================
        # LONG SETUP
        #
        # Pullback to EMA20/EMA50
        # + rejection
        # + recovery
        # ====================================================

        ema_zone_low = min(
            row["EMA20"],
            row["EMA50"]
        )

        ema_zone_high = max(
            row["EMA20"],
            row["EMA50"]
        )

        long_touch = (
            row["Low"]
            <= ema_zone_high
            and
            row["Low"]
            >= ema_zone_low
            * 0.985
        )

        long_reclaim = (
            row["Close"]
            > row["EMA20"]
        )

        long_candle = (
            row["Close"]
            > row["Open"]
        )

        long_momentum = (
            row["Close"]
            > previous["Close"]
        )

        long_rsi = (
            RSI_LONG_MIN
            <= row["RSI"]
            <= RSI_LONG_MAX
        )

        long_rejection = (
            row["LOWER_WICK"]
            >= row["BODY"] * 0.50
        )

        long_setup = (
            long_regime
            and
            long_touch
            and
            long_reclaim
            and
            long_candle
            and
            long_momentum
            and
            long_rsi
            and
            long_rejection
        )

        # ====================================================
        # SHORT SETUP
        # ====================================================

        short_touch = (
            row["High"]
            >= ema_zone_low
            and
            row["High"]
            <= ema_zone_high
            * 1.015
        )

        short_reclaim = (
            row["Close"]
            < row["EMA20"]
        )

        short_candle = (
            row["Close"]
            < row["Open"]
        )

        short_momentum = (
            row["Close"]
            < previous["Close"]
        )

        short_rsi = (
            RSI_SHORT_MIN
            <= row["RSI"]
            <= RSI_SHORT_MAX
        )

        short_rejection = (
            row["UPPER_WICK"]
            >= row["BODY"] * 0.50
        )

        short_setup = (
            short_regime
            and
            short_touch
            and
            short_reclaim
            and
            short_candle
            and
            short_momentum
            and
            short_rsi
            and
            short_rejection
        )

        # ====================================================
        # ENTRY NEXT OPEN
        # ====================================================

        entry_idx = i + 1

        if (
            entry_idx
            >= len(df)
        ):
            break

        entry_price = (
            df.iloc[
                entry_idx
            ]["Open"]
        )

        # ====================================================
        # LONG
        # ====================================================

        if long_setup:

            levels = calculate_levels(
                df,
                i,
                "LONG",
                entry_price
            )

            if levels is None:
                continue

            trade = simulate_trade(
                df,
                entry_idx,
                "LONG",
                entry_price,
                levels["SL"],
                levels["TP"],
                levels["RiskPct"]
            )

            if trade:

                trade["Symbol"] = symbol

                trade["Setup"] = (
                    "EMA_PULLBACK_LONG"
                )

                trade["EntryTime"] = (
                    df.index[
                        entry_idx
                    ]
                )

                trade["ExitTime"] = (
                    df.index[
                        trade["ExitIndex"]
                    ]
                )

                trades.append(
                    trade
                )

                locked_until = (
                    trade["ExitIndex"]
                    + 1
                )

        # ====================================================
        # SHORT
        # ====================================================

        elif short_setup:

            levels = calculate_levels(
                df,
                i,
                "SHORT",
                entry_price
            )

            if levels is None:
                continue

            trade = simulate_trade(
                df,
                entry_idx,
                "SHORT",
                entry_price,
                levels["SL"],
                levels["TP"],
                levels["RiskPct"]
            )

            if trade:

                trade["Symbol"] = symbol

                trade["Setup"] = (
                    "EMA_PULLBACK_SHORT"
                )

                trade["EntryTime"] = (
                    df.index[
                        entry_idx
                    ]
                )

                trade["ExitTime"] = (
                    df.index[
                        trade["ExitIndex"]
                    ]
                )

                trades.append(
                    trade
                )

                locked_until = (
                    trade["ExitIndex"]
                    + 1
                )

    print(
        f"✔️ {symbol}: "
        f"{len(trades)} معامله"
    )

    return trades


# ============================================================
# REPORT
# ============================================================

def calculate_report(
    portfolio_df
):

    total = len(
        portfolio_df
    )

    wins = len(
        portfolio_df[
            portfolio_df[
                "Outcome"
            ] == "WIN"
        ]
    )

    losses = len(
        portfolio_df[
            portfolio_df[
                "Outcome"
            ] == "LOSS"
        ]
    )

    timeouts = len(
        portfolio_df[
            portfolio_df[
                "Outcome"
            ] == "TIMEOUT"
        ]
    )

    decisive = (
        wins + losses
    )

    win_rate = (
        wins
        / decisive
        * 100
        if decisive > 0
        else 0
    )

    total_r = (
        portfolio_df[
            "R"
        ].sum()
    )

    avg_r = (
        portfolio_df[
            "R"
        ].mean()
    )

    gross_profit = (
        portfolio_df.loc[
            portfolio_df["R"] > 0,
            "R"
        ].sum()
    )

    gross_loss = abs(
        portfolio_df.loc[
            portfolio_df["R"] < 0,
            "R"
        ].sum()
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    equity = (
        portfolio_df[
            "R"
        ].cumsum()
    )

    peak = (
        equity.cummax()
    )

    drawdown = (
        equity - peak
    )

    max_dd = (
        drawdown.min()
    )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_dd": max_dd
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 75)
    print(
        "HUNTER-X V10"
    )
    print(
        "LBank 1H → 4H MTF EMA PULLBACK"
    )
    print("=" * 75)

    print(
        f"📅 Backtest: "
        f"{BACKTEST_DAYS} days"
    )

    print(
        "🎯 RR: 1:2"
    )

    print(
        "📡 Data: LBank / CCXT"
    )

    print(
        "⏱️ Timeframes: 1H + 4H"
    )

    print(
        f"🔒 Portfolio Lock: "
        f"{PORTFOLIO_LOCK}"
    )

    # ========================================================
    # DATA
    # ========================================================

    data_1h = {}
    data_4h = {}

    for symbol, lbank_symbol in (
        SYMBOLS.items()
    ):

        df1h = fetch_lbank_1h(
            lbank_symbol,
            BACKTEST_DAYS
        )

        if df1h is None:
            continue

        if len(df1h) < 500:

            print(
                f"⚠️ {symbol}: "
                f"داده کافی نیست"
            )

            continue

        df4h = make_4h(
            df1h
        )

        data_1h[
            symbol
        ] = df1h

        data_4h[
            symbol
        ] = df4h

    # ========================================================
    # BACKTEST
    # ========================================================

    all_trades = []

    for symbol in SYMBOLS:

        if symbol not in data_1h:
            continue

        print("")
        print("-" * 75)

        df1h = calculate_indicators(
            data_1h[symbol]
        )

        df = attach_previous_4h(
            df1h,
            data_4h[symbol]
        )

        trades = backtest_symbol(
            symbol,
            df
        )

        all_trades.extend(
            trades
        )

    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print(
            "⚠️ هیچ معامله‌ای پیدا نشد."
        )

        return

    trades_df = pd.DataFrame(
        all_trades
    )

    trades_df.sort_values(
        "EntryTime",
        inplace=True
    )

    # ========================================================
    # PORTFOLIO LOCK OPTIONAL
    # ========================================================

    if PORTFOLIO_LOCK:

        accepted = []

        free_time = None

        for _, trade in (
            trades_df.iterrows()
        ):

            if (
                free_time is None
                or
                trade["EntryTime"]
                >= free_time
            ):

                accepted.append(
                    trade
                )

                free_time = (
                    trade["ExitTime"]
                )

        portfolio_df = (
            pd.DataFrame(
                accepted
            )
        )

    else:

        portfolio_df = (
            trades_df.copy()
        )

    # ========================================================
    # REPORT
    # ========================================================

    stats = calculate_report(
        portfolio_df
    )

    print("")
    print("=" * 75)
    print(
        "📊 FINAL V10 RESULT"
    )
    print("=" * 75)

    print(
        f"Candidate Trades: "
        f"{len(trades_df)}"
    )

    print(
        f"Accepted Trades: "
        f"{stats['total']}"
    )

    print(
        f"WIN: "
        f"{stats['wins']}"
    )

    print(
        f"LOSS: "
        f"{stats['losses']}"
    )

    print(
        f"TIMEOUT: "
        f"{stats['timeouts']}"
    )

    print(
        f"🎯 Win Rate: "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"📈 Profit Factor: "
        f"{stats['profit_factor']:.3f}"
    )

    print(
        f"💰 Net R: "
        f"{stats['total_r']:.2f}R"
    )

    print(
        f"📊 Avg R: "
        f"{stats['avg_r']:.3f}R"
    )

    print(
        f"📉 Max DD: "
        f"{stats['max_dd']:.2f}R"
    )

    # ========================================================
    # TRADES / YEAR
    # ========================================================

    if len(
        portfolio_df
    ) > 1:

        first_date = (
            portfolio_df[
                "EntryTime"
            ].min()
        )

        last_date = (
            portfolio_df[
                "EntryTime"
            ].max()
        )

        days = max(
            1,
            (
                last_date
                - first_date
            ).total_seconds()
            / 86400
        )

        trades_per_year = (
            len(portfolio_df)
            / days
            * 365
        )

    else:

        trades_per_year = (
            len(portfolio_df)
        )

    print(
        f"📅 Trades/Year: "
        f"{trades_per_year:.1f}"
    )

    # ========================================================
    # SYMBOL
    # ========================================================

    print("")
    print("=" * 75)
    print(
        "📌 SYMBOL"
    )
    print("=" * 75)

    symbol_rows = []

    for symbol, group in (
        portfolio_df.groupby(
            "Symbol"
        )
    ):

        w = len(
            group[
                group["Outcome"]
                == "WIN"
            ]
        )

        l = len(
            group[
                group["Outcome"]
                == "LOSS"
            ]
        )

        t = len(
            group[
                group["Outcome"]
                == "TIMEOUT"
            ]
        )

        dec = w + l

        wr = (
            w / dec * 100
            if dec > 0
            else 0
        )

        symbol_rows.append({
            "Symbol": symbol,
            "Trades": len(group),
            "Wins": w,
            "Losses": l,
            "Timeout": t,
            "WinRate": wr,
            "NetR":
                group["R"].sum()
        })

    print(
        pd.DataFrame(
            symbol_rows
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # SIDE
    # ========================================================

    print("")
    print("=" * 75)
    print(
        "📌 LONG / SHORT"
    )
    print("=" * 75)

    side_rows = []

    for side, group in (
        portfolio_df.groupby(
            "Side"
        )
    ):

        w = len(
            group[
                group["Outcome"]
                == "WIN"
            ]
        )

        l = len(
            group[
                group["Outcome"]
                == "LOSS"
            ]
        )

        dec = w + l

        wr = (
            w / dec * 100
            if dec > 0
            else 0
        )

        side_rows.append({
            "Side": side,
            "Trades": len(group),
            "Wins": w,
            "Losses": l,
            "WinRate": wr,
            "NetR":
                group["R"].sum()
        })

    print(
        pd.DataFrame(
            side_rows
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # MONTH
    # ========================================================

    portfolio_df[
        "Month"
    ] = (
        portfolio_df[
            "EntryTime"
        ].dt.to_period("M")
    )

    monthly = (
        portfolio_df
        .groupby("Month")
        .agg(
            Trades=(
                "R",
                "count"
            ),
            NetR=(
                "R",
                "sum"
            ),
            AvgR=(
                "R",
                "mean"
            )
        )
    )

    print("")
    print("=" * 75)
    print(
        "📅 MONTHLY"
    )
    print("=" * 75)

    print(
        monthly.to_string()
    )

    # ========================================================
    # CSV
    # ========================================================

    portfolio_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("")
    print(
        f"💾 CSV: "
        f"{OUTPUT_CSV}"
    )

    # ========================================================
    # AUDIT
    # ========================================================

    print("")
    print("=" * 75)
    print(
        "🔍 ANTI-LOOKAHEAD AUDIT"
    )
    print("=" * 75)

    print(
        "✔️ 1H data from LBank"
    )

    print(
        "✔️ 4H constructed from 1H"
    )

    print(
        "✔️ Only completed 4H candles used"
    )

    print(
        "✔️ Signal candle closed first"
    )

    print(
        "✔️ Entry = next 1H OPEN"
    )

    print(
        "✔️ One active trade per symbol"
    )

    print(
        "✔️ No same-symbol overlap"
    )

    print(
        "✔️ Same-candle TP/SL = LOSS"
    )

    print(
        "✔️ Fee included"
    )

    print(
        "✔️ Slippage included once"
    )

    print(
        "✔️ RR = 1:2"
    )

    print("")
    print("=" * 75)
    print(
        "✅ V10 BACKTEST FINISHED"
    )
    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
