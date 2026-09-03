import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# WHALE PULLBACK 2R v2
# REAL BINANCE FUTURES BACKTEST
#
# 1H  = Trend Filter
# 15M = Entry
# RR  = 1:2
#
# OUTPUT:
# فقط تعداد کل معاملات + Win Rate کلی
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT"
]

INTERVAL = "15m"

# تعداد روزهای بک تست
DAYS = 365

# تنظیمات اندیکاتورها
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

# تنظیمات ستاپ
BREAKOUT_LOOKBACK = 10
PULLBACK_MAX_BARS = 8

ADX_1H_MIN = 20
ADX_15M_MIN = 18

TARGET_RR = 2.0

# حداکثر فاصله SL نسبت به Entry
MAX_SL_PERCENT = 0.03


# ============================================================
# دریافت دیتای Binance Futures
# ============================================================

def get_binance_klines(symbol, interval, start_time, end_time):

    url = "https://fapi.binance.com/fapi/v1/klines"

    all_data = []

    current_start = start_time

    while current_start < end_time:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1500
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_data.extend(data)

        last_open_time = data[-1][0]

        current_start = last_open_time + 1

        if len(data) < 1500:
            break

        time.sleep(0.15)

    columns = [
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
        "ignore"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["open_time"])

    df["timestamp"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = abs(df["high"] - previous_close)

    tr3 = abs(df["low"] - previous_close)

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr


# ============================================================
# ADX
# ============================================================

def calculate_adx(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=df.index
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100 *
        abs(plus_di - minus_di)
        / denominator.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return adx


# ============================================================
# ساخت اندیکاتورها
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD
    )

    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD
    )

    df["adx"] = calculate_adx(
        df,
        ADX_PERIOD
    )

    return df


# ============================================================
# ساخت تایم فریم 1H واقعی از 15M
# ============================================================

def build_1h(df):

    df = df.copy()

    df = df.set_index("timestamp")

    h1 = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    h1 = h1.dropna()

    h1 = h1.reset_index()

    h1 = add_indicators(h1)

    return h1


# ============================================================
# اتصال وضعیت 1H به 15M
# ============================================================

def merge_1h_filter(df15, df1h):

    df15 = df15.copy()
    df1h = df1h.copy()

    h1_columns = [
        "timestamp",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "adx"
    ]

    df1h = df1h[h1_columns].copy()

    df1h = df1h.rename(
        columns={
            "ema20": "h1_ema20",
            "ema50": "h1_ema50",
            "ema200": "h1_ema200",
            "rsi": "h1_rsi",
            "adx": "h1_adx"
        }
    )

    merged = pd.merge_asof(
        df15.sort_values("timestamp"),
        df1h.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    return merged


# ============================================================
# بک تست یک نماد
# ============================================================

def backtest_symbol(df):

    wins = 0
    losses = 0

    total_trades = 0

    i = 250

    while i < len(df) - 30:

        row = df.iloc[i]

        # ----------------------------------------------------
        # اطلاعات 1H
        # ----------------------------------------------------

        h1_close = row["close"]

        h1_ema20 = row["h1_ema20"]
        h1_ema50 = row["h1_ema50"]
        h1_ema200 = row["h1_ema200"]

        h1_rsi = row["h1_rsi"]
        h1_adx = row["h1_adx"]

        if pd.isna(h1_ema200) or pd.isna(h1_adx):
            i += 1
            continue

        # ----------------------------------------------------
        # LONG TREND
        # ----------------------------------------------------

        long_trend = (
            h1_close > h1_ema200
            and h1_ema20 > h1_ema50
            and h1_ema50 > h1_ema200
            and h1_rsi > 50
            and h1_adx >= ADX_1H_MIN
        )

        # ----------------------------------------------------
        # SHORT TREND
        # ----------------------------------------------------

        short_trend = (
            h1_close < h1_ema200
            and h1_ema20 < h1_ema50
            and h1_ema50 < h1_ema200
            and h1_rsi < 50
            and h1_adx >= ADX_1H_MIN
        )

        if not long_trend and not short_trend:
            i += 1
            continue

        # ====================================================
        # بررسی BREAKOUT
        # ====================================================

        lookback_start = i - BREAKOUT_LOOKBACK

        recent_high = df.iloc[
            lookback_start:i
        ]["high"].max()

        recent_low = df.iloc[
            lookback_start:i
        ]["low"].min()

        current = df.iloc[i]

        # ====================================================
        # LONG BREAKOUT
        # ====================================================

        if (
            long_trend
            and current["close"] > recent_high
            and current["close"] > current["open"]
            and current["adx"] >= ADX_15M_MIN
            and current["rsi"] > 50
        ):

            breakout_level = recent_high

            # ----------------------------------------------
            # منتظر Pullback / Retest می‌شویم
            # ----------------------------------------------

            found_entry = False

            for k in range(
                i + 1,
                min(i + 1 + PULLBACK_MAX_BARS, len(df))
            ):

                candle = df.iloc[k]

                # Retest سطح شکست
                touched_level = (
                    candle["low"] <= breakout_level
                    and candle["high"] >= breakout_level
                )

                if not touched_level:
                    continue

                # Confirmation صعودی
                confirmation = (
                    candle["close"] > candle["open"]
                    and candle["close"] > breakout_level
                    and candle["rsi"] > 50
                    and candle["adx"] >= ADX_15M_MIN
                )

                if not confirmation:
                    continue

                entry = candle["close"]

                # Swing Low پولبک
                swing_start = max(i + 1, k - 3)

                swing_low = df.iloc[
                    swing_start:k + 1
                ]["low"].min()

                atr = candle["atr"]

                if pd.isna(atr) or atr <= 0:
                    continue

                sl = swing_low - (atr * 0.5)

                risk = entry - sl

                if risk <= 0:
                    continue

                if risk > entry * MAX_SL_PERCENT:
                    continue

                tp = entry + (
                    risk * TARGET_RR
                )

                found_entry = True

                # ------------------------------------------
                # بررسی نتیجه معامله
                # ------------------------------------------

                result = None

                for j in range(
                    k + 1,
                    min(k + 25, len(df))
                ):

                    future = df.iloc[j]

                    hit_sl = future["low"] <= sl
                    hit_tp = future["high"] >= tp

                    # اگر هر دو در یک کندل لمس شوند:
                    # محافظه‌کارانه SL محسوب می‌کنیم
                    if hit_sl and hit_tp:
                        result = "LOSS"
                        break

                    if hit_sl:
                        result = "LOSS"
                        break

                    if hit_tp:
                        result = "WIN"
                        break

                if result == "WIN":
                    wins += 1
                    total_trades += 1

                elif result == "LOSS":
                    losses += 1
                    total_trades += 1

                if result is not None:
                    i = j
                else:
                    i = k

                break

            if found_entry:
                continue

        # ====================================================
        # SHORT BREAKOUT
        # ====================================================

        if (
            short_trend
            and current["close"] < recent_low
            and current["close"] < current["open"]
            and current["adx"] >= ADX_15M_MIN
            and current["rsi"] < 50
        ):

            breakout_level = recent_low

            found_entry = False

            for k in range(
                i + 1,
                min(i + 1 + PULLBACK_MAX_BARS, len(df))
            ):

                candle = df.iloc[k]

                # Retest
                touched_level = (
                    candle["high"] >= breakout_level
                    and candle["low"] <= breakout_level
                )

                if not touched_level:
                    continue

                # Confirmation نزولی
                confirmation = (
                    candle["close"] < candle["open"]
                    and candle["close"] < breakout_level
                    and candle["rsi"] < 50
                    and candle["adx"] >= ADX_15M_MIN
                )

                if not confirmation:
                    continue

                entry = candle["close"]

                swing_start = max(i + 1, k - 3)

                swing_high = df.iloc[
                    swing_start:k + 1
                ]["high"].max()

                atr = candle["atr"]

                if pd.isna(atr) or atr <= 0:
                    continue

                sl = swing_high + (atr * 0.5)

                risk = sl - entry

                if risk <= 0:
                    continue

                if risk > entry * MAX_SL_PERCENT:
                    continue

                tp = entry - (
                    risk * TARGET_RR
                )

                found_entry = True

                result = None

                for j in range(
                    k + 1,
                    min(k + 25, len(df))
                ):

                    future = df.iloc[j]

                    hit_sl = future["high"] >= sl
                    hit_tp = future["low"] <= tp

                    # هر دو در یک کندل = LOSS
                    if hit_sl and hit_tp:
                        result = "LOSS"
                        break

                    if hit_sl:
                        result = "LOSS"
                        break

                    if hit_tp:
                        result = "WIN"
                        break

                if result == "WIN":
                    wins += 1
                    total_trades += 1

                elif result == "LOSS":
                    losses += 1
                    total_trades += 1

                if result is not None:
                    i = j
                else:
                    i = k

                break

        i += 1

    return total_trades, wins, losses


# ============================================================
# اجرای کل بک تست
# ============================================================

def main():

    end_time = int(
        time.time() * 1000
    )

    start_time = end_time - (
        DAYS * 24 * 60 * 60 * 1000
    )

    total_trades = 0
    total_wins = 0
    total_losses = 0

    print("=" * 60)
    print("WHALE PULLBACK 2R v2")
    print("REAL BINANCE FUTURES BACKTEST")
    print("=" * 60)

    for symbol in SYMBOLS:

        print(f"\nDownloading {symbol} ...")

        try:

            df15 = get_binance_klines(
                symbol,
                INTERVAL,
                start_time,
                end_time
            )

            if df15.empty:

                print(
                    f"{symbol}: NO DATA"
                )

                continue

            print(
                f"{symbol}: "
                f"{len(df15):,} candles"
            )

            # اندیکاتورهای 15M
            df15 = add_indicators(df15)

            # ساخت 1H واقعی
            df1h = build_1h(df15)

            # اتصال 1H به 15M
            df = merge_1h_filter(
                df15,
                df1h
            )

            trades, wins, losses = backtest_symbol(
                df
            )

            total_trades += trades
            total_wins += wins
            total_losses += losses

        except Exception as e:

            print(
                f"{symbol}: ERROR -> {e}"
            )

    # ========================================================
    # نتیجه نهایی
    # ========================================================

    if total_trades > 0:

        win_rate = (
            total_wins /
            total_trades
        ) * 100

    else:

        win_rate = 0

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"TOTAL TRADES : {total_trades}"
    )

    print(
        f"WIN RATE     : {win_rate:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
