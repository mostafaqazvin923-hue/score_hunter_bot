# ============================================================
# HUNTER-X V11
# FIXED 2% TP / 1% SL
# 4H REGIME + 1H BREAKOUT/PULLBACK/CONFIRMATION
# LBank Perpetual Futures
# NO LOOK-AHEAD
# ============================================================

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
]

TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"

# تعداد کندل دریافت اولیه
FETCH_LIMIT = 1000

# -----------------------------
# FIXED TP / SL
# -----------------------------

TAKE_PROFIT_PCT = 0.02     # +2%
STOP_LOSS_PCT = 0.01       # -1%

# -----------------------------
# COSTS
# -----------------------------

# 0.04% per side
FEE_PER_SIDE = 0.0004

# 0.04% per side
SLIPPAGE_PER_SIDE = 0.0004

# -----------------------------
# BACKTEST
# -----------------------------

COOLDOWN_BARS = 6

# حداکثر مدت نگهداری معامله
MAX_HOLD_BARS = 80

# تقسیم IS / OOS
OOS_RATIO = 0.20


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap"
    }
})

exchange.load_markets()


# ============================================================
# DATA
# ============================================================

def fetch_ohlcv(symbol, timeframe, limit=1000):

    print(f"Downloading {symbol} {timeframe} ...")

    data = exchange.fetch_ohlcv(
        symbol,
        timeframe,
        limit=limit
    )

    if not data:
        raise RuntimeError(
            f"No OHLCV data returned for {symbol} {timeframe}"
        )

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # حذف آخرین کندل چون ممکن است هنوز بسته نشده باشد
    # --------------------------------------------------------

    now = pd.Timestamp.now(tz="UTC")

    if len(df) > 0:

        last_open = df.iloc[-1]["timestamp"]

        if timeframe == "1h":
            candle_end = last_open + pd.Timedelta(hours=1)
        elif timeframe == "4h":
            candle_end = last_open + pd.Timedelta(hours=4)
        else:
            candle_end = last_open

        if candle_end > now:
            df = df.iloc[:-1].copy()

    return df.reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

def add_4h_indicators(df):

    df = df.copy()

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["ema200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    df["ema200_prev"] = df["ema200"].shift(1)

    return df


def add_1h_indicators(df):

    df = df.copy()

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr14"] = (
        df["tr"]
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # Volume average
    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Candle body
    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    # Previous high / low
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)

    return df


# ============================================================
# 4H REGIME
# ============================================================

def get_4h_regime(row):

    if pd.isna(row["ema200"]):
        return 0

    # LONG
    if (
        row["close"] > row["ema200"]
        and row["ema50"] > row["ema200"]
        and row["ema200"] >= row["ema200_prev"]
    ):
        return 1

    # SHORT
    if (
        row["close"] < row["ema200"]
        and row["ema50"] < row["ema200"]
        and row["ema200"] <= row["ema200_prev"]
    ):
        return -1

    return 0


# ============================================================
# MAP LAST CLOSED 4H CANDLE TO 1H
# ============================================================

def align_4h_to_1h(df1h, df4h):

    regime_data = df4h.copy()

    regime_data["regime"] = regime_data.apply(
        get_4h_regime,
        axis=1
    )

    regime_data = regime_data[
        [
            "timestamp",
            "regime"
        ]
    ]

    # فقط 4H candleهایی که تا زمان 1H بسته شده‌اند
    df1h = pd.merge_asof(
        df1h.sort_values("timestamp"),
        regime_data.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    df1h["regime"] = (
        df1h["regime"]
        .fillna(0)
        .astype(int)
    )

    return df1h


# ============================================================
# BREAKOUT / PULLBACK / CONFIRMATION
# ============================================================

def long_signal(df, i):

    if i < 30:
        return False

    row = df.iloc[i]

    # -----------------------------------------
    # باید 4H روند صعودی باشد
    # -----------------------------------------

    if row["regime"] != 1:
        return False

    # -----------------------------------------
    # داده کافی
    # -----------------------------------------

    if pd.isna(row["atr14"]):
        return False

    if pd.isna(row["volume_ma20"]):
        return False

    # ========================================================
    # STEP 1 — IMPULSE
    # ========================================================

    impulse_start = i - 3

    impulse_move = (
        df.iloc[i - 1]["close"]
        / df.iloc[impulse_start]["close"]
        - 1
    )

    # حرکت صعودی حداقل حدود 0.8%
    if impulse_move < 0.008:
        return False

    # ========================================================
    # STEP 2 — PULLBACK
    # ========================================================

    pullback = df.iloc[i - 2:i]

    if len(pullback) < 2:
        return False

    # حداقل یکی از دو کندل پولبک باید نزولی باشد
    if not (
        (pullback["close"] < pullback["open"]).any()
    ):
        return False

    # پولبک نباید بیشتر از 1% افت کند
    pullback_low = pullback["low"].min()

    impulse_high = df.iloc[i - 1]["high"]

    pullback_depth = (
        impulse_high - pullback_low
    ) / impulse_high

    if pullback_depth > 0.01:
        return False

    # ========================================================
    # STEP 3 — CONFIRMATION
    # ========================================================

    # کندل فعلی باید صعودی باشد
    if row["close"] <= row["open"]:
        return False

    # Close باید در نیمه بالایی کندل باشد
    candle_range = row["high"] - row["low"]

    if candle_range <= 0:
        return False

    close_location = (
        row["close"] - row["low"]
    ) / candle_range

    if close_location < 0.60:
        return False

    # بدنه حداقل 0.25 ATR
    if row["body"] < 0.25 * row["atr14"]:
        return False

    # حجم حداقل برابر میانگین 20
    if row["volume"] < row["volume_ma20"]:
        return False

    # close باید بالاتر از high کندل قبلی باشد
    if row["close"] <= df.iloc[i - 1]["high"]:
        return False

    return True


def short_signal(df, i):

    if i < 30:
        return False

    row = df.iloc[i]

    # -----------------------------------------
    # 4H bearish regime
    # -----------------------------------------

    if row["regime"] != -1:
        return False

    if pd.isna(row["atr14"]):
        return False

    if pd.isna(row["volume_ma20"]):
        return False

    # ========================================================
    # STEP 1 — IMPULSE
    # ========================================================

    impulse_start = i - 3

    impulse_move = (
        df.iloc[i - 1]["close"]
        / df.iloc[impulse_start]["close"]
        - 1
    )

    # حرکت نزولی حداقل 0.8%
    if impulse_move > -0.008:
        return False

    # ========================================================
    # STEP 2 — PULLBACK
    # ========================================================

    pullback = df.iloc[i - 2:i]

    if len(pullback) < 2:
        return False

    # حداقل یکی از دو کندل پولبک باید صعودی باشد
    if not (
        (pullback["close"] > pullback["open"]).any()
    ):
        return False

    pullback_high = pullback["high"].max()

    impulse_low = df.iloc[i - 1]["low"]

    pullback_depth = (
        pullback_high - impulse_low
    ) / impulse_low

    if pullback_depth > 0.01:
        return False

    # ========================================================
    # STEP 3 — CONFIRMATION
    # ========================================================

    if row["close"] >= row["open"]:
        return False

    candle_range = row["high"] - row["low"]

    if candle_range <= 0:
        return False

    close_location = (
        row["close"] - row["low"]
    ) / candle_range

    if close_location > 0.40:
        return False

    if row["body"] < 0.25 * row["atr14"]:
        return False

    if row["volume"] < row["volume_ma20"]:
        return False

    if row["close"] >= df.iloc[i - 1]["low"]:
        return False

    return True


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction
):

    if entry_index >= len(df):
        return None

    entry_candle = df.iloc[entry_index]

    raw_entry = entry_candle["open"]

    # --------------------------------------------------------
    # Slippage on entry
    # --------------------------------------------------------

    if direction == 1:
        entry = raw_entry * (
            1 + SLIPPAGE_PER_SIDE
        )
    else:
        entry = raw_entry * (
            1 - SLIPPAGE_PER_SIDE
        )

    # --------------------------------------------------------
    # TP / SL
    # --------------------------------------------------------

    if direction == 1:

        tp = entry * (
            1 + TAKE_PROFIT_PCT
        )

        sl = entry * (
            1 - STOP_LOSS_PCT
        )

    else:

        tp = entry * (
            1 - TAKE_PROFIT_PCT
        )

        sl = entry * (
            1 + STOP_LOSS_PCT
        )

    # --------------------------------------------------------
    # Search future candles
    # --------------------------------------------------------

    end_index = min(
        len(df),
        entry_index + MAX_HOLD_BARS + 1
    )

    exit_price = None
    exit_index = None
    exit_reason = None

    for j in range(
        entry_index,
        end_index
    ):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if direction == 1:

            hit_sl = low <= sl
            hit_tp = high >= tp

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

        # ----------------------------------------------------
        # اگر هر دو در یک کندل لمس شوند:
        # محافظه‌کارانه SL اول
        # ----------------------------------------------------

        if hit_sl and hit_tp:

            exit_price = sl
            exit_index = j
            exit_reason = "SL_AND_TP_SAME_CANDLE"
            break

        elif hit_sl:

            exit_price = sl
            exit_index = j
            exit_reason = "SL"
            break

        elif hit_tp:

            exit_price = tp
            exit_index = j
            exit_reason = "TP"
            break

    # --------------------------------------------------------
    # Time exit
    # --------------------------------------------------------

    if exit_price is None:

        exit_index = end_index - 1

        if exit_index < entry_index:
            return None

        exit_price = df.iloc[
            exit_index
        ]["close"]

        exit_reason = "TIME"

    # --------------------------------------------------------
    # Slippage on exit
    # --------------------------------------------------------

    if direction == 1:

        exit_price = exit_price * (
            1 - SLIPPAGE_PER_SIDE
        )

    else:

        exit_price = exit_price * (
            1 + SLIPPAGE_PER_SIDE
        )

    # --------------------------------------------------------
    # Gross return
    # --------------------------------------------------------

    if direction == 1:

        gross_return = (
            exit_price - entry
        ) / entry

    else:

        gross_return = (
            entry - exit_price
        ) / entry

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    net_return = (
        gross_return
        - 2 * FEE_PER_SIDE
    )

    # --------------------------------------------------------
    # Convert to R
    #
    # Risk = 1%
    # --------------------------------------------------------

    R = net_return / STOP_LOSS_PCT

    # --------------------------------------------------------
    # Win
    # --------------------------------------------------------

    win = R > 0

    return {
        "entry_index": entry_index,
        "exit_index": exit_index,
        "entry": entry,
        "exit": exit_price,
        "direction": (
            "LONG"
            if direction == 1
            else "SHORT"
        ),
        "reason": exit_reason,
        "R": R,
        "return_pct": net_return * 100,
        "win": win
    }


# ============================================================
# BACKTEST SYMBOL
# ============================================================

def backtest_symbol(symbol):

    print("\n" + "=" * 70)
    print(f"BACKTEST: {symbol}")
    print("=" * 70)

    df1h = fetch_ohlcv(
        symbol,
        TIMEFRAME_1H,
        FETCH_LIMIT
    )

    time.sleep(0.3)

    df4h = fetch_ohlcv(
        symbol,
        TIMEFRAME_4H,
        FETCH_LIMIT
    )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df4h = add_4h_indicators(df4h)
    df1h = add_1h_indicators(df1h)

    # --------------------------------------------------------
    # Align 4H regime
    # --------------------------------------------------------

    df1h = align_4h_to_1h(
        df1h,
        df4h
    )

    # --------------------------------------------------------
    # Need enough data
    # --------------------------------------------------------

    if len(df1h) < 300:
        print(
            f"Not enough 1H data: {len(df1h)}"
        )
        return []

    # --------------------------------------------------------
    # IS / OOS boundary
    # --------------------------------------------------------

    split_index = int(
        len(df1h) * (1 - OOS_RATIO)
    )

    trades = []

    locked_until = -1

    i = 30

    last_signal_index = (
        len(df1h)
        - MAX_HOLD_BARS
        - 2
    )

    while i < last_signal_index:

        # ----------------------------------------------------
        # No overlapping trade
        # ----------------------------------------------------

        if i <= locked_until:
            i += 1
            continue

        direction = 0

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        if long_signal(df1h, i):

            direction = 1

        elif short_signal(df1h, i):

            direction = -1

        if direction == 0:

            i += 1
            continue

        # ----------------------------------------------------
        # Entry = NEXT candle OPEN
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(df1h):
            break

        # ----------------------------------------------------
        # IMPORTANT:
        # classification by ENTRY candle
        # ----------------------------------------------------

        sample = (
            "IS"
            if entry_index < split_index
            else "OOS"
        )

        # ----------------------------------------------------
        # Don't allow trades crossing boundary
        # ----------------------------------------------------

        trade = simulate_trade(
            df1h,
            entry_index,
            direction
        )

        if trade is None:
            i += 1
            continue

        # ----------------------------------------------------
        # If trade crosses IS/OOS boundary,
        # don't include it.
        # ----------------------------------------------------

        if (
            trade["entry_index"] < split_index
            and trade["exit_index"] >= split_index
        ):

            i += 1
            continue

        trade["symbol"] = symbol
        trade["sample"] = sample

        trade["entry_time"] = df1h.iloc[
            trade["entry_index"]
        ]["timestamp"]

        trade["exit_time"] = df1h.iloc[
            trade["exit_index"]
        ]["timestamp"]

        trades.append(trade)

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        locked_until = (
            trade["exit_index"]
            + COOLDOWN_BARS
        )

        i = locked_until + 1

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if not trades:

        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "WinRate": 0,
            "NetR": 0,
            "PF": 0,
            "Expectancy": 0,
            "MaxDD": 0
        }

    df = pd.DataFrame(trades)

    total = len(df)

    wins = int(
        (df["R"] > 0).sum()
    )

    losses = total - wins

    win_rate = (
        wins / total * 100
    )

    net_r = df["R"].sum()

    gross_profit = df.loc[
        df["R"] > 0,
        "R"
    ].sum()

    gross_loss = abs(
        df.loc[
            df["R"] < 0,
            "R"
        ].sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit
            / gross_loss
        )
    else:
        pf = np.inf

    expectancy = (
        df["R"].mean()
    )

    equity = (
        df["R"]
        .cumsum()
    )

    peak = equity.cummax()

    drawdown = (
        peak - equity
    )

    max_dd = drawdown.max()

    return {
        "Trades": total,
        "Wins": wins,
        "Losses": losses,
        "WinRate": win_rate,
        "NetR": net_r,
        "PF": pf,
        "Expectancy": expectancy,
        "MaxDD": max_dd
    }


# ============================================================
# MAIN
# ============================================================

all_trades = []

for symbol in SYMBOLS:

    try:

        symbol_trades = backtest_symbol(
            symbol
        )

        all_trades.extend(
            symbol_trades
        )

    except Exception as e:

        print(
            f"\nERROR {symbol}: {e}"
        )

# ============================================================
# FINAL RESULTS
# ============================================================

if not all_trades:

    print("\nNO TRADES FOUND.")
    raise SystemExit


trades_df = pd.DataFrame(
    all_trades
)

# ============================================================
# OVERALL
# ============================================================

overall = calculate_metrics(
    all_trades
)

# ============================================================
# IS / OOS
# ============================================================

is_trades = [
    x for x in all_trades
    if x["sample"] == "IS"
]

oos_trades = [
    x for x in all_trades
    if x["sample"] == "OOS"
]

is_metrics = calculate_metrics(
    is_trades
)

oos_metrics = calculate_metrics(
    oos_trades
)


# ============================================================
# PRINT FINAL
# ============================================================

print("\n")
print("=" * 90)
print("              HUNTER-X V11 FINAL BACKTEST")
print("=" * 90)

print("\n📌 SETUP")
print("-" * 90)
print("TP                 : 2.00%")
print("SL                 : 1.00%")
print("Risk / Reward      : 1 : 2")
print("Execution          : 1H")
print("Regime             : 4H")
print("Entry              : Next 1H candle OPEN")
print("Fee / side         : 0.04%")
print("Slippage / side    : 0.04%")

print("\n")
print("🔥 OVERALL RESULT")
print("-" * 90)

print(
    f"TOTAL TRADES      : {overall['Trades']}"
)

print(
    f"TOTAL WINS        : {overall['Wins']}"
)

print(
    f"TOTAL LOSSES      : {overall['Losses']}"
)

print(
    f"TOTAL WIN RATE    : {overall['WinRate']:.2f}%"
)

print(
    f"TOTAL NET R       : {overall['NetR']:.2f}R"
)

print(
    f"TOTAL PF          : {overall['PF']:.3f}"
)

print(
    f"TOTAL EXPECTANCY  : {overall['Expectancy']:.3f}R"
)

print(
    f"TOTAL MAX DD      : {overall['MaxDD']:.2f}R"
)


# ============================================================
# IS
# ============================================================

print("\n")
print("📊 IN-SAMPLE")
print("-" * 90)

print(
    f"Trades            : {is_metrics['Trades']}"
)

print(
    f"Wins              : {is_metrics['Wins']}"
)

print(
    f"Losses            : {is_metrics['Losses']}"
)

print(
    f"Win Rate          : {is_metrics['WinRate']:.2f}%"
)

print(
    f"Net R             : {is_metrics['NetR']:.2f}R"
)

print(
    f"PF                : {is_metrics['PF']:.3f}"
)

print(
    f"Expectancy        : {is_metrics['Expectancy']:.3f}R"
)

print(
    f"Max DD            : {is_metrics['MaxDD']:.2f}R"
)


# ============================================================
# OOS
# ============================================================

print("\n")
print("🧪 OUT-OF-SAMPLE")
print("-" * 90)

print(
    f"Trades            : {oos_metrics['Trades']}"
)

print(
    f"Wins              : {oos_metrics['Wins']}"
)

print(
    f"Losses            : {oos_metrics['Losses']}"
)

print(
    f"Win Rate          : {oos_metrics['WinRate']:.2f}%"
)

print(
    f"Net R             : {oos_metrics['NetR']:.2f}R"
)

print(
    f"PF                : {oos_metrics['PF']:.3f}"
)

print(
    f"Expectancy        : {oos_metrics['Expectancy']:.3f}R"
)

print(
    f"Max DD            : {oos_metrics['MaxDD']:.2f}R"
)


# ============================================================
# SYMBOL RESULTS
# ============================================================

print("\n")
print("📈 RESULTS BY SYMBOL")
print("-" * 90)

symbol_rows = []

for symbol in SYMBOLS:

    subset = [
        x for x in all_trades
        if x["symbol"] == symbol
    ]

    if not subset:
        continue

    m = calculate_metrics(
        subset
    )

    symbol_rows.append({
        "Symbol": symbol,
        "Trades": m["Trades"],
        "Wins": m["Wins"],
        "Losses": m["Losses"],
        "WinRate": m["WinRate"],
        "NetR": m["NetR"],
        "PF": m["PF"],
        "Expectancy": m["Expectancy"]
    })

symbol_df = pd.DataFrame(
    symbol_rows
)

print(
    symbol_df.to_string(
        index=False,
        formatters={
            "WinRate": "{:.2f}%".format,
            "NetR": "{:.2f}".format,
            "PF": "{:.3f}".format,
            "Expectancy": "{:.3f}".format
        }
    )
)


# ============================================================
# LONG / SHORT
# ============================================================

print("\n")
print("↕️ LONG / SHORT")
print("-" * 90)

for direction in ["LONG", "SHORT"]:

    subset = [
        x for x in all_trades
        if x["direction"] == direction
    ]

    m = calculate_metrics(
        subset
    )

    print(
        f"{direction:<8} "
        f"Trades={m['Trades']:<5} "
        f"Wins={m['Wins']:<5} "
        f"Losses={m['Losses']:<5} "
        f"WR={m['WinRate']:.2f}% "
        f"NetR={m['NetR']:.2f}R "
        f"PF={m['PF']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

trades_df.to_csv(
    "hunter_x_v11_trades.csv",
    index=False
)

symbol_df.to_csv(
    "hunter_x_v11_symbol_results.csv",
    index=False
)

print("\n")
print("=" * 90)
print("✅ BACKTEST COMPLETE")
print("=" * 90)

print(
    "\n📁 hunter_x_v11_trades.csv"
)

print(
    "📁 hunter_x_v11_symbol_results.csv"
)

print("\n")
