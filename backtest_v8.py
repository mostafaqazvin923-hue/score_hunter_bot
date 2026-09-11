import os
import sys
import subprocess
from datetime import datetime, timedelta

# ============================================================
# AUTO INSTALL
# ============================================================

try:
    import ccxt
except ImportError:
    print("📦 Installing ccxt...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt"]
    )
    import ccxt

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

TIMEFRAME_1H = "1h"

LOOKBACK_DAYS = 365

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

STRUCTURE_LOOKBACK = 15

BREAKOUT_VOLUME_MULTIPLIER = 1.10

# Retest tolerance
RETEST_TOLERANCE = 0.003

# Maximum candles allowed for retest after breakout
MAX_RETEST_CANDLES = 13

# Stop distance
ATR_STOP_MULTIPLIER = 0.25

# Maximum allowed risk distance
MAX_RISK_PERCENT = 0.045

# Risk / Reward
TARGET_RR = 2.0

# Maximum holding time
MAX_HOLDING_CANDLES = 40

# Conservative handling:
# if SL and TP are both touched in same candle => LOSS
SAME_CANDLE_PRIORITY = "SL"

# ============================================================
# SYMBOLS
# ============================================================

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
    "DOT": "DOT/USDT",
    "SHIB": "SHIB/USDT",
    "PEPE": "PEPE/USDT",
    "ARB": "ARB/USDT",
    "OP": "OP/USDT",
    "POL": "POL/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "INJ": "INJ/USDT",
    "FET": "FET/USDT",
    "APT": "APT/USDT",
    "TIA": "TIA/USDT",
    "ICP": "ICP/USDT",
    "BCH": "BCH/USDT",
    "LTC": "LTC/USDT",
    "CRV": "CRV/USDT",
    "PENDLE": "PENDLE/USDT",
    "AAVE": "AAVE/USDT",
    "GRT": "GRT/USDT",
    "XLM": "XLM/USDT"
}


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap"
    }
})


# ============================================================
# DOWNLOAD 1H DATA
# ============================================================

start_date = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 70)
print(
    f"📥 Downloading {LOOKBACK_DAYS} days of 1H LBank data "
    f"for {len(SYMBOLS)} symbols"
)
print("=" * 70)

data_1h = {}


for symbol, lbank_symbol in SYMBOLS.items():

    print(f"\n🔹 {symbol} -> {lbank_symbol}")

    all_ohlcv = []

    current_since = since_timestamp

    max_retries = 80
    retries = 0

    while retries < max_retries:

        try:

            now_timestamp = exchange.milliseconds()

            if current_since >= now_timestamp:
                break

            ohlcv = exchange.fetch_ohlcv(
                lbank_symbol,
                timeframe=TIMEFRAME_1H,
                since=current_since,
                limit=500
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            last_timestamp = ohlcv[-1][0]

            next_since = last_timestamp + 3600000

            if next_since <= current_since:
                current_since += 3600000
            else:
                current_since = next_since

            print(
                f"   received: {len(all_ohlcv)} candles",
                end="\r"
            )

            if len(ohlcv) < 500:
                break

            exchange.sleep(exchange.rateLimit / 1000)

            retries += 1

        except Exception as e:

            print(
                f"\n   ❌ Error {symbol}: {e}"
            )

            retries += 1

            exchange.sleep(3)

    if not all_ohlcv:

        print(
            f"\n   ❌ No data for {symbol}"
        )

        continue

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

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

    df.dropna(inplace=True)

    df.drop_duplicates(
        subset=["Date"],
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    filename = f"{symbol}_1h_lbank_data.csv"

    df.to_csv(
        filename,
        index=False
    )

    data_1h[symbol] = df

    print(
        f"\n   ✔️ {symbol}: {len(df)} candles"
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=EMA_MID,
            adjust=False
        )
        .mean()
    )

    df["EMA_200"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI - Wilder style
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / (
        avg_loss + 1e-10
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # ATR - Wilder
    # --------------------------------------------------------

    high_low = (
        df["High"] -
        df["Low"]
    )

    high_close = (
        df["High"] -
        df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] -
        df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["TR"] = tr

    df["ATR"] = (
        tr
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    up_move = df["High"].diff()

    down_move = -df["Low"].diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    )

    atr_wilder = (
        tr
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False
        )
        .mean()
    )

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False
        ).mean()
        /
        (atr_wilder + 1e-10)
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False
        ).mean()
        /
        (atr_wilder + 1e-10)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (
            plus_di +
            minus_di +
            1e-10
        )
    )

    df["ADX"] = (
        dx
        .ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False
        )
        .mean()
    )

    return df


# ============================================================
# BUILD 4H DATA
# ============================================================

def build_4h(df1h):

    temp = (
        df1h
        .set_index("Date")
        .resample("4h", label="left", closed="left")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna()
        .reset_index()
    )

    temp = calculate_indicators(temp)

    return temp


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(symbol, df1h):

    if len(df1h) < 500:

        print(
            f"⚠️ {symbol}: insufficient data "
            f"({len(df1h)})"
        )

        return []

    df1h = calculate_indicators(
        df1h.copy()
    )

    df4h = build_4h(
        df1h
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # A 4H candle is only usable AFTER it has completely closed.
    #
    # For each 1H candle at time t, we use the latest 4H candle
    # whose CLOSE time is <= t.
    #
    # This eliminates 4H look-ahead.
    # --------------------------------------------------------

    df4h["CloseTime"] = (
        df4h["Date"] +
        pd.Timedelta(hours=4)
    )

    df4h = df4h.sort_values(
        "CloseTime"
    )

    df1h = df1h.sort_values(
        "Date"
    )

    df1h = pd.merge_asof(
        df1h,
        df4h[
            [
                "CloseTime",
                "EMA_20",
                "EMA_50",
                "EMA_200",
                "RSI",
                "ATR",
                "ADX",
                "Close"
            ]
        ].rename(
            columns={
                "EMA_20": "EMA20_4H",
                "EMA_50": "EMA50_4H",
                "EMA_200": "EMA200_4H",
                "RSI": "RSI_4H",
                "ATR": "ATR_4H",
                "ADX": "ADX_4H",
                "Close": "Close_4H"
            }
        ),
        left_on="Date",
        right_on="CloseTime",
        direction="backward"
    )

    # --------------------------------------------------------
    # Previous 4H EMA200
    # --------------------------------------------------------

    df4h["EMA200_PREV"] = (
        df4h["EMA_200"].shift(1)
    )

    df4h_prev = df4h[
        [
            "CloseTime",
            "EMA200_PREV"
        ]
    ]

    df1h = pd.merge_asof(
        df1h.sort_values("Date"),
        df4h_prev.sort_values("CloseTime"),
        left_on="Date",
        right_on="CloseTime",
        direction="backward"
    )

    trades = []

    locked_until = -1

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for i in range(
        EMA_SLOW + 5,
        len(df1h) - MAX_HOLDING_CANDLES - 5
    ):

        if i <= locked_until:
            continue

        c = df1h.iloc[i]

        # ----------------------------------------------------
        # Need valid closed 4H data
        # ----------------------------------------------------

        if pd.isna(c["EMA20_4H"]):
            continue

        if pd.isna(c["EMA50_4H"]):
            continue

        if pd.isna(c["EMA200_4H"]):
            continue

        if pd.isna(c["RSI_4H"]):
            continue

        if pd.isna(c["ADX_4H"]):
            continue

        if pd.isna(c["EMA200_PREV"]):
            continue

        # ----------------------------------------------------
        # 4H REGIME
        # ----------------------------------------------------

        slope_positive = (
            c["EMA200_4H"] >
            c["EMA200_PREV"]
        )

        slope_negative = (
            c["EMA200_4H"] <
            c["EMA200_PREV"]
        )

        is_long_regime = (
            c["Close_4H"] >
            c["EMA200_4H"]
            and
            c["EMA20_4H"] >
            c["EMA50_4H"]
            and
            c["EMA50_4H"] >
            c["EMA200_4H"]
            and
            slope_positive
            and
            c["ADX_4H"] >= 20
            and
            c["RSI_4H"] > 55
        )

        is_short_regime = (
            c["Close_4H"] <
            c["EMA200_4H"]
            and
            c["EMA20_4H"] <
            c["EMA50_4H"]
            and
            c["EMA50_4H"] <
            c["EMA200_4H"]
            and
            slope_negative
            and
            c["ADX_4H"] >= 20
            and
            c["RSI_4H"] < 45
        )

        if (
            not is_long_regime
            and
            not is_short_regime
        ):
            continue

        # ----------------------------------------------------
        # 1H STRUCTURE
        #
        # IMPORTANT:
        # Structure excludes current candle.
        # ----------------------------------------------------

        if i < STRUCTURE_LOOKBACK:
            continue

        lookback = df1h.iloc[
            i - STRUCTURE_LOOKBACK:i
        ]

        struct_high = (
            lookback["High"].max()
        )

        struct_low = (
            lookback["Low"].min()
        )

        avg_volume = (
            lookback["Volume"].mean()
        )

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

        breakout_long = (
            c["Close"] > struct_high
            and
            c["Volume"] >=
            avg_volume *
            BREAKOUT_VOLUME_MULTIPLIER
        )

        breakout_short = (
            c["Close"] < struct_low
            and
            c["Volume"] >=
            avg_volume *
            BREAKOUT_VOLUME_MULTIPLIER
        )

        # ====================================================
        # LONG
        # ====================================================

        if (
            is_long_regime
            and
            breakout_long
        ):

            breakout_idx = i

            entered = False

            for p in range(
                1,
                MAX_RETEST_CANDLES + 1
            ):

                entry_idx = (
                    breakout_idx + p
                )

                if entry_idx >= len(df1h):
                    break

                pc = df1h.iloc[
                    entry_idx
                ]

                # --------------------------------------------
                # Retest
                # --------------------------------------------

                retest = (
                    pc["Low"] <=
                    struct_high *
                    (1 + RETEST_TOLERANCE)
                )

                if not retest:
                    continue

                # --------------------------------------------
                # Confirmation
                # --------------------------------------------

                bullish_confirmation = (
                    pc["Close"] >
                    pc["Open"]
                    and
                    pc["RSI"] > 50
                )

                if not bullish_confirmation:
                    continue

                entry_price = pc["Close"]

                swing_low = (
                    df1h.iloc[
                        breakout_idx:
                        entry_idx + 1
                    ]["Low"].min()
                )

                atr = pc["ATR"]

                if pd.isna(atr) or atr <= 0:
                    continue

                sl = (
                    swing_low -
                    ATR_STOP_MULTIPLIER *
                    atr
                )

                risk = (
                    entry_price -
                    sl
                )

                risk_percent = (
                    risk /
                    entry_price
                )

                if risk <= 0:
                    continue

                if (
                    risk_percent >
                    MAX_RISK_PERCENT
                ):
                    continue

                tp = (
                    entry_price +
                    TARGET_RR *
                    risk
                )

                # --------------------------------------------
                # Forward simulation
                #
                # This is the ONLY place where future candles
                # are examined, AFTER entry has been fixed.
                # --------------------------------------------

                outcome = "OPEN"

                exit_price = np.nan

                exit_idx = None

                max_exit = min(
                    entry_idx +
                    MAX_HOLDING_CANDLES,
                    len(df1h) - 1
                )

                for j in range(
                    entry_idx + 1,
                    max_exit + 1
                ):

                    fc = df1h.iloc[j]

                    hit_sl = (
                        fc["Low"] <= sl
                    )

                    hit_tp = (
                        fc["High"] >= tp
                    )

                    if hit_sl and hit_tp:

                        if SAME_CANDLE_PRIORITY == "SL":

                            outcome = "LOSS"
                            exit_price = sl

                        else:

                            outcome = "WIN"
                            exit_price = tp

                        exit_idx = j

                        break

                    elif hit_sl:

                        outcome = "LOSS"
                        exit_price = sl
                        exit_idx = j

                        break

                    elif hit_tp:

                        outcome = "WIN"
                        exit_price = tp
                        exit_idx = j

                        break

                # --------------------------------------------
                # Max holding exit
                # --------------------------------------------

                if outcome == "OPEN":

                    exit_idx = max_exit

                    exit_price = (
                        df1h.iloc[
                            exit_idx
                        ]["Close"]
                    )

                    # Mark-to-market R
                    pnl_price = (
                        exit_price -
                        entry_price
                    )

                    r_multiple = (
                        pnl_price /
                        risk
                    )

                    outcome = "TIMEOUT"

                else:

                    if outcome == "WIN":
                        r_multiple = TARGET_RR
                    else:
                        r_multiple = -1.0

                trades.append({
                    "Symbol": symbol,
                    "Side": "LONG",
                    "BreakoutTime": c["Date"],
                    "EntryTime": pc["Date"],
                    "ExitTime": df1h.iloc[
                        exit_idx
                    ]["Date"],
                    "Entry": entry_price,
                    "SL": sl,
                    "TP": tp,
                    "RiskPercent": risk_percent * 100,
                    "Outcome": outcome,
                    "R": r_multiple
                })

                locked_until = exit_idx

                entered = True

                break

            if entered:
                continue

        # ====================================================
        # SHORT
        # ====================================================

        if (
            is_short_regime
            and
            breakout_short
        ):

            breakout_idx = i

            entered = False

            for p in range(
                1,
                MAX_RETEST_CANDLES + 1
            ):

                entry_idx = (
                    breakout_idx + p
                )

                if entry_idx >= len(df1h):
                    break

                pc = df1h.iloc[
                    entry_idx
                ]

                # --------------------------------------------
                # Retest
                # --------------------------------------------

                retest = (
                    pc["High"] >=
                    struct_low *
                    (1 - RETEST_TOLERANCE)
                )

                if not retest:
                    continue

                # --------------------------------------------
                # Confirmation
                # --------------------------------------------

                bearish_confirmation = (
                    pc["Close"] <
                    pc["Open"]
                    and
                    pc["RSI"] < 50
                )

                if not bearish_confirmation:
                    continue

                entry_price = pc["Close"]

                swing_high = (
                    df1h.iloc[
                        breakout_idx:
                        entry_idx + 1
                    ]["High"].max()
                )

                atr = pc["ATR"]

                if pd.isna(atr) or atr <= 0:
                    continue

                sl = (
                    swing_high +
                    ATR_STOP_MULTIPLIER *
                    atr
                )

                risk = (
                    sl -
                    entry_price
                )

                risk_percent = (
                    risk /
                    entry_price
                )

                if risk <= 0:
                    continue

                if (
                    risk_percent >
                    MAX_RISK_PERCENT
                ):
                    continue

                tp = (
                    entry_price -
                    TARGET_RR *
                    risk
                )

                # --------------------------------------------
                # Forward simulation
                # --------------------------------------------

                outcome = "OPEN"

                exit_price = np.nan

                exit_idx = None

                max_exit = min(
                    entry_idx +
                    MAX_HOLDING_CANDLES,
                    len(df1h) - 1
                )

                for j in range(
                    entry_idx + 1,
                    max_exit + 1
                ):

                    fc = df1h.iloc[j]

                    hit_sl = (
                        fc["High"] >= sl
                    )

                    hit_tp = (
                        fc["Low"] <= tp
                    )

                    if hit_sl and hit_tp:

                        if SAME_CANDLE_PRIORITY == "SL":

                            outcome = "LOSS"
                            exit_price = sl

                        else:

                            outcome = "WIN"
                            exit_price = tp

                        exit_idx = j

                        break

                    elif hit_sl:

                        outcome = "LOSS"
                        exit_price = sl
                        exit_idx = j

                        break

                    elif hit_tp:

                        outcome = "WIN"
                        exit_price = tp
                        exit_idx = j

                        break

                # --------------------------------------------
                # Max holding exit
                # --------------------------------------------

                if outcome == "OPEN":

                    exit_idx = max_exit

                    exit_price = (
                        df1h.iloc[
                            exit_idx
                        ]["Close"]
                    )

                    pnl_price = (
                        entry_price -
                        exit_price
                    )

                    r_multiple = (
                        pnl_price /
                        risk
                    )

                    outcome = "TIMEOUT"

                else:

                    if outcome == "WIN":
                        r_multiple = TARGET_RR
                    else:
                        r_multiple = -1.0

                trades.append({
                    "Symbol": symbol,
                    "Side": "SHORT",
                    "BreakoutTime": c["Date"],
                    "EntryTime": pc["Date"],
                    "ExitTime": df1h.iloc[
                        exit_idx
                    ]["Date"],
                    "Entry": entry_price,
                    "SL": sl,
                    "TP": tp,
                    "RiskPercent": risk_percent * 100,
                    "Outcome": outcome,
                    "R": r_multiple
                })

                locked_until = exit_idx

                entered = True

                break

    return trades


# ============================================================
# RUN BACKTEST
# ============================================================

print("\n")
print("=" * 70)
print("🚀 HUNTER-X CLEAN V4 BACKTEST")
print("=" * 70)

all_trades = []


for symbol, df in data_1h.items():

    print(
        f"\n🔍 Backtesting {symbol}..."
    )

    symbol_trades = backtest_symbol(
        symbol,
        df
    )

    all_trades.extend(
        symbol_trades
    )

    print(
        f"   ✔️ Trades: {len(symbol_trades)}"
    )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("📊 FINAL PORTFOLIO REPORT")
print("=" * 70)


if not all_trades:

    print(
        "⚠️ No valid trades found."
    )

    sys.exit()


pf_df = pd.DataFrame(
    all_trades
)

# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

total_trades = len(pf_df)

wins = (
    pf_df["Outcome"] == "WIN"
).sum()

losses = (
    pf_df["Outcome"] == "LOSS"
).sum()

timeouts = (
    pf_df["Outcome"] == "TIMEOUT"
).sum()

decisive_trades = (
    wins + losses
)

win_rate = (
    wins /
    decisive_trades *
    100
    if decisive_trades > 0
    else 0
)

# ------------------------------------------------------------
# R statistics
# ------------------------------------------------------------

total_R = (
    pf_df["R"].sum()
)

gross_profit_R = (
    pf_df.loc[
        pf_df["R"] > 0,
        "R"
    ].sum()
)

gross_loss_R = abs(
    pf_df.loc[
        pf_df["R"] < 0,
        "R"
    ].sum()
)

profit_factor = (
    gross_profit_R /
    gross_loss_R
    if gross_loss_R > 0
    else np.inf
)

expectancy_R = (
    pf_df["R"].mean()
)

# ------------------------------------------------------------
# Equity / Drawdown
# ------------------------------------------------------------

equity = (
    pf_df["R"]
    .cumsum()
)

peak = (
    equity
    .cummax()
)

drawdown = (
    equity -
    peak
)

max_drawdown_R = (
    drawdown.min()
)

# ------------------------------------------------------------
# Average win/loss
# ------------------------------------------------------------

average_win = (
    pf_df.loc[
        pf_df["R"] > 0,
        "R"
    ].mean()
)

average_loss = (
    pf_df.loc[
        pf_df["R"] < 0,
        "R"
    ].mean()
)

# ============================================================
# PRINT
# ============================================================

print(
    f"🔸 Total Trades: {total_trades}"
)

print(
    f"🟢 Wins: {wins}"
)

print(
    f"🔴 Losses: {losses}"
)

print(
    f"⏳ Timeouts: {timeouts}"
)

print(
    f"🎯 Win Rate: {win_rate:.2f}%"
)

print(
    f"💰 Net Profit: {total_R:.2f}R"
)

print(
    f"📈 Profit Factor: {profit_factor:.2f}"
)

print(
    f"📊 Expectancy: {expectancy_R:.3f}R/trade"
)

print(
    f"📉 Max Drawdown: {max_drawdown_R:.2f}R"
)

print(
    f"🏆 Average Win: {average_win:.2f}R"
)

print(
    f"💔 Average Loss: {average_loss:.2f}R"
)


# ============================================================
# SYMBOL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("📌 PERFORMANCE BY SYMBOL")
print("=" * 70)

symbol_rows = []

for symbol, group in pf_df.groupby("Symbol"):

    symbol_wins = (
        group["Outcome"] == "WIN"
    ).sum()

    symbol_losses = (
        group["Outcome"] == "LOSS"
    ).sum()

    symbol_decisive = (
        symbol_wins +
        symbol_losses
    )

    symbol_wr = (
        symbol_wins /
        symbol_decisive *
        100
        if symbol_decisive > 0
        else 0
    )

    symbol_R = (
        group["R"].sum()
    )

    symbol_rows.append({
        "Symbol": symbol,
        "Trades": len(group),
        "Wins": symbol_wins,
        "Losses": symbol_losses,
        "Timeouts": (
            group["Outcome"] ==
            "TIMEOUT"
        ).sum(),
        "WinRate": symbol_wr,
        "NetR": symbol_R
    })

symbol_report = pd.DataFrame(
    symbol_rows
)

symbol_report = symbol_report.sort_values(
    "NetR",
    ascending=False
)

print(
    symbol_report.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

pf_df.to_csv(
    "HUNTER_X_CLEAN_V4_ALL_TRADES.csv",
    index=False
)

symbol_report.to_csv(
    "HUNTER_X_CLEAN_V4_SYMBOL_REPORT.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("✅ CLEAN BACKTEST COMPLETED")
print("=" * 70)

print(
    "📁 HUNTER_X_CLEAN_V4_ALL_TRADES.csv"
)

print(
    "📁 HUNTER_X_CLEAN_V4_SYMBOL_REPORT.csv"
)

print(
    "\n⚠️ مهم:"
)

print(
    "این نسخه قبل از ورود هیچ اطلاعاتی از آینده "
    "برای تصمیم‌گیری استفاده نمی‌کند."
)

print(
    "درصد Win Rate فقط بر اساس معاملات WIN/LOSS "
    "محاسبه شده و TIMEOUT جداگانه گزارش می‌شود."
)
