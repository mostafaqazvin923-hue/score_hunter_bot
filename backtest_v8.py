# ============================================================
# HUNTER-X 1% 2R — CLEAN LBank 1 YEAR BACKTEST
# ============================================================
#
# هدف:
#   TP = +1%
#   SL = -0.50%
#   RR  = 1:2
#
# تایم‌فریم:
#   1H Entry
#   4H Trend Filter
#
# ویژگی‌های مهم:
#   - NO LOOKAHEAD
#   - فقط کندل‌های کاملاً بسته
#   - Entry در Open کندل بعدی
#   - بدون Break-Even
#   - بدون Trailing
#   - بدون معامله همزمان در Portfolio
#   - Same Candle TP/SL => SL first (محافظه‌کارانه)
#   - Timeout جداگانه
# ============================================================

import os
import sys
import subprocess
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------
try:
    import ccxt
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "ccxt"
    ])
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
}

TIMEFRAME = "1h"

TEST_DAYS = 365
WARMUP_DAYS = 100

TP_PCT = 0.0100
SL_PCT = 0.0050

RR = TP_PCT / SL_PCT

MAX_HOLD_BARS = 24

# ------------------------------------------------------------
# Fees
# ------------------------------------------------------------
# فقط Fee را در P/L کم می‌کنیم.
# Slippage مستقیماً در قیمت Entry/Exit اعمال می‌شود.
# بنابراین Double Count نداریم.
# ------------------------------------------------------------

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002


# ============================================================
# LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
})

exchange.load_markets()


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result


def atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def adx(df, period=14):

    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_value = atr(df, period)

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = (
        100 *
        (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# DOWNLOAD LBank 1H DATA
# ============================================================

def download_lbank(symbol, days=465):

    print(f"\n📥 Downloading {symbol} ...")

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    since = int(start_time.timestamp() * 1000)

    all_ohlcv = []

    while True:

        try:

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe="1h",
                since=since,
                limit=1000
            )

        except Exception as e:

            print(f"❌ Error downloading {symbol}: {e}")
            break

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        print(
            f"\r   Candles downloaded: {len(all_ohlcv)}",
            end=""
        )

        last_timestamp = ohlcv[-1][0]

        new_since = last_timestamp + 1

        if new_since <= since:
            break

        since = new_since

        if len(ohlcv) < 1000:
            break

    print()

    if not all_ohlcv:
        return None

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

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    df = df.sort_values("Timestamp")

    df = df.set_index("Timestamp")

    # Remove current unfinished 1H candle
    now = pd.Timestamp.now(tz="UTC")

    current_hour = now.floor("h")

    df = df[df.index < current_hour]

    return df


# ============================================================
# 4H DATA
# ============================================================

def prepare_4h(df):

    df4 = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    df4 = df4.dropna()

    # Indicators
    df4["EMA20"] = ema(df4["Close"], 20)
    df4["EMA50"] = ema(df4["Close"], 50)
    df4["EMA200"] = ema(df4["Close"], 200)

    df4["RSI"] = rsi(df4["Close"], 14)
    df4["ATR"] = atr(df4, 14)
    df4["ADX"] = adx(df4, 14)

    return df4


# ============================================================
# MAP PREVIOUS CLOSED 4H TO EACH 1H CANDLE
# ============================================================

def attach_4h(df1, df4):

    htf = df4.copy()

    # IMPORTANT:
    # At 1H candle X we use the PREVIOUS completed 4H candle.
    htf["HTF_TIME"] = htf.index

    htf = htf.rename(columns={
        "Close": "HTF_Close",
        "EMA20": "HTF_EMA20",
        "EMA50": "HTF_EMA50",
        "EMA200": "HTF_EMA200",
        "RSI": "HTF_RSI",
        "ADX": "HTF_ADX",
        "ATR": "HTF_ATR"
    })

    htf = htf[
        [
            "HTF_Close",
            "HTF_EMA20",
            "HTF_EMA50",
            "HTF_EMA200",
            "HTF_RSI",
            "HTF_ADX",
            "HTF_ATR"
        ]
    ]

    # Shift by one completed 4H candle
    htf = htf.shift(1)

    # Map each 1H candle to its 4H bucket
    df = df1.copy()

    df["HTF_BUCKET"] = df.index.floor("4h")

    htf.index.name = "HTF_BUCKET"

    df = df.join(
        htf,
        on="HTF_BUCKET"
    )

    return df


# ============================================================
# PREPARE 1H INDICATORS
# ============================================================

def prepare_1h(df):

    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)

    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)
    df["ADX"] = adx(df, 14)

    df["VOL_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Previous 20 candles ONLY
    df["PREV20_HIGH"] = (
        df["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["PREV20_LOW"] = (
        df["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    df["RANGE"] = (
        df["High"] - df["Low"]
    )

    df["BODY"] = (
        df["Close"] - df["Open"]
    ).abs()

    df["BODY_RATIO"] = (
        df["BODY"] /
        df["RANGE"].replace(0, np.nan)
    )

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def get_signal(df, i):

    row = df.iloc[i]

    required = [
        "HTF_Close",
        "HTF_EMA20",
        "HTF_EMA50",
        "HTF_EMA200",
        "HTF_RSI",
        "HTF_ADX",
        "EMA20",
        "EMA50",
        "RSI",
        "ADX",
        "ATR",
        "VOL_MA20",
        "PREV20_HIGH",
        "PREV20_LOW",
        "BODY_RATIO"
    ]

    for col in required:

        if pd.isna(row[col]):
            return None

    close = row["Close"]

    # --------------------------------------------------------
    # HTF TREND
    # --------------------------------------------------------

    long_htf = (
        row["HTF_EMA20"] >
        row["HTF_EMA50"] >
        row["HTF_EMA200"]
        and
        row["HTF_Close"] >
        row["HTF_EMA20"]
        and
        row["HTF_RSI"] >= 50
        and
        row["HTF_ADX"] >= 15
    )

    short_htf = (
        row["HTF_EMA20"] <
        row["HTF_EMA50"] <
        row["HTF_EMA200"]
        and
        row["HTF_Close"] <
        row["HTF_EMA20"]
        and
        row["HTF_RSI"] <= 50
        and
        row["HTF_ADX"] >= 15
    )

    # --------------------------------------------------------
    # 1H MOMENTUM
    # --------------------------------------------------------

    long_momentum = (
        row["EMA20"] > row["EMA50"]
        and
        row["RSI"] >= 50
        and
        row["ADX"] >= 15
    )

    short_momentum = (
        row["EMA20"] < row["EMA50"]
        and
        row["RSI"] <= 50
        and
        row["ADX"] >= 15
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    long_breakout = (
        close > row["PREV20_HIGH"]
        and
        row["BODY_RATIO"] >= 0.45
    )

    short_breakout = (
        close < row["PREV20_LOW"]
        and
        row["BODY_RATIO"] >= 0.45
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ok = (
        row["Volume"] >=
        row["VOL_MA20"] * 0.90
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    atr_pct = (
        row["ATR"] / close
    )

    volatility_ok = (
        atr_pct >= 0.0015
        and
        atr_pct <= 0.015
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if (
        long_htf
        and long_momentum
        and long_breakout
        and volume_ok
        and volatility_ok
    ):
        return "LONG"

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if (
        short_htf
        and short_momentum
        and short_breakout
        and volume_ok
        and volatility_ok
    ):
        return "SHORT"

    return None


# ============================================================
# EXECUTE TRADE
# ============================================================

def execute_trade(df, signal_index, direction, symbol):

    # Entry happens on NEXT candle OPEN
    if signal_index + 1 >= len(df):
        return None

    entry_row = df.iloc[signal_index + 1]

    entry_time = df.index[signal_index + 1]

    raw_entry = float(entry_row["Open"])

    # --------------------------------------------------------
    # Slippage
    # --------------------------------------------------------

    if direction == "LONG":
        entry_price = raw_entry * (1 + SLIPPAGE)
    else:
        entry_price = raw_entry * (1 - SLIPPAGE)

    # --------------------------------------------------------
    # TP / SL
    # --------------------------------------------------------

    if direction == "LONG":

        tp = entry_price * (1 + TP_PCT)
        sl = entry_price * (1 - SL_PCT)

    else:

        tp = entry_price * (1 - TP_PCT)
        sl = entry_price * (1 + SL_PCT)

    max_exit_index = min(
        signal_index + 1 + MAX_HOLD_BARS,
        len(df) - 1
    )

    exit_price = None
    exit_time = None
    result = None

    bars_held = 0

    # --------------------------------------------------------
    # Walk forward ONLY
    # --------------------------------------------------------

    for j in range(
        signal_index + 1,
        max_exit_index + 1
    ):

        candle = df.iloc[j]

        high = float(candle["High"])
        low = float(candle["Low"])

        bars_held += 1

        if direction == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Same candle => SL FIRST
            if hit_sl and hit_tp:

                exit_price = sl * (
                    1 - SLIPPAGE
                )

                result = "LOSS"

                exit_time = df.index[j]

                break

            elif hit_sl:

                exit_price = sl * (
                    1 - SLIPPAGE
                )

                result = "LOSS"

                exit_time = df.index[j]

                break

            elif hit_tp:

                exit_price = tp * (
                    1 - SLIPPAGE
                )

                result = "WIN"

                exit_time = df.index[j]

                break

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Same candle => SL FIRST
            if hit_sl and hit_tp:

                exit_price = sl * (
                    1 + SLIPPAGE
                )

                result = "LOSS"

                exit_time = df.index[j]

                break

            elif hit_sl:

                exit_price = sl * (
                    1 + SLIPPAGE
                )

                result = "LOSS"

                exit_time = df.index[j]

                break

            elif hit_tp:

                exit_price = tp * (
                    1 + SLIPPAGE
                )

                result = "WIN"

                exit_time = df.index[j]

                break

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if result is None:

        j = max_exit_index

        exit_time = df.index[j]

        raw_exit = float(
            df.iloc[j]["Close"]
        )

        if direction == "LONG":

            exit_price = raw_exit * (
                1 - SLIPPAGE
            )

        else:

            exit_price = raw_exit * (
                1 + SLIPPAGE
            )

        result = "TIMEOUT"

    # --------------------------------------------------------
    # Gross return
    # --------------------------------------------------------

    if direction == "LONG":

        gross_return = (
            exit_price / entry_price - 1
        )

    else:

        gross_return = (
            entry_price / exit_price - 1
        )

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    total_fee = TAKER_FEE * 2

    net_return = (
        gross_return - total_fee
    )

    return {
        "Symbol": symbol,
        "Direction": direction,
        "SignalTime": df.index[signal_index],
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "Entry": entry_price,
        "Exit": exit_price,
        "TP": tp,
        "SL": sl,
        "Result": result,
        "BarsHeld": bars_held,
        "GrossPct": gross_return * 100,
        "NetPct": net_return * 100,
    }


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def backtest_symbol(df, symbol):

    trades = []

    i = 250

    while i < len(df) - 2:

        signal = get_signal(
            df,
            i
        )

        if signal is None:

            i += 1
            continue

        trade = execute_trade(
            df,
            i,
            signal,
            symbol
        )

        if trade is not None:

            trades.append(trade)

            # ------------------------------------------------
            # IMPORTANT:
            # After entering a trade, skip until trade exits.
            # Prevent overlapping trades per symbol.
            # ------------------------------------------------

            exit_time = trade["ExitTime"]

            future_indices = np.where(
                df.index >= exit_time
            )[0]

            if len(future_indices) > 0:

                exit_idx = future_indices[0]

                i = exit_idx + 1

            else:

                i += 1

        else:

            i += 1

    return trades


# ============================================================
# PORTFOLIO LOCK
# ============================================================

def apply_portfolio_lock(all_trades):

    if not all_trades:
        return []

    trades = pd.DataFrame(all_trades)

    trades["EntryTime"] = pd.to_datetime(
        trades["EntryTime"]
    )

    trades["ExitTime"] = pd.to_datetime(
        trades["ExitTime"]
    )

    trades = trades.sort_values(
        ["EntryTime", "ExitTime"]
    )

    accepted = []

    portfolio_free_time = None

    for _, trade in trades.iterrows():

        entry_time = trade["EntryTime"]

        if (
            portfolio_free_time is None
            or
            entry_time > portfolio_free_time
        ):

            accepted.append(
                trade.to_dict()
            )

            portfolio_free_time = (
                trade["ExitTime"]
            )

    return accepted


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_metrics(trades):

    if not trades:

        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Timeouts": 0,
            "WinRate": 0,
            "NetPct": 0,
            "AvgTradePct": 0,
            "ProfitFactor": 0,
            "MaxDD": 0,
        }

    df = pd.DataFrame(trades)

    wins = int(
        (df["Result"] == "WIN").sum()
    )

    losses = int(
        (df["Result"] == "LOSS").sum()
    )

    timeouts = int(
        (df["Result"] == "TIMEOUT").sum()
    )

    resolved = wins + losses

    if resolved > 0:

        win_rate = (
            wins / resolved
        ) * 100

    else:

        win_rate = 0

    net_pct = df["NetPct"].sum()

    avg_trade = df["NetPct"].mean()

    positive = df.loc[
        df["NetPct"] > 0,
        "NetPct"
    ].sum()

    negative = abs(
        df.loc[
            df["NetPct"] < 0,
            "NetPct"
        ].sum()
    )

    if negative > 0:
        profit_factor = (
            positive / negative
        )
    else:
        profit_factor = float("inf")

    equity = df["NetPct"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    max_dd = drawdown.min()

    return {
        "Trades": len(df),
        "Wins": wins,
        "Losses": losses,
        "Timeouts": timeouts,
        "WinRate": win_rate,
        "NetPct": net_pct,
        "AvgTradePct": avg_trade,
        "ProfitFactor": profit_factor,
        "MaxDD": max_dd,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("🚀 HUNTER-X 1% 2R — CLEAN LBank BACKTEST")
    print("=" * 70)

    print(f"🎯 TP       : {TP_PCT * 100:.2f}%")
    print(f"🛑 SL       : {SL_PCT * 100:.2f}%")
    print(f"📐 RR       : 1:{RR:.1f}")
    print(f"⏱ Timeframe: 1H + 4H")
    print(f"📅 Test     : {TEST_DAYS} days")
    print(f"⏳ Max Hold : {MAX_HOLD_BARS} candles")
    print()

    all_candidates = []

    symbol_results = []

    for name, symbol in SYMBOLS.items():

        try:

            df = download_lbank(
                symbol,
                TEST_DAYS + WARMUP_DAYS
            )

            if df is None or len(df) < 500:

                print(
                    f"⚠️ {name}: insufficient data"
                )

                continue

            # ------------------------------------------------
            # 4H
            # ------------------------------------------------

            df4 = prepare_4h(df)

            # ------------------------------------------------
            # 1H
            # ------------------------------------------------

            df1 = prepare_1h(df)

            # ------------------------------------------------
            # Previous closed 4H
            # ------------------------------------------------

            df1 = attach_4h(
                df1,
                df4
            )

            # ------------------------------------------------
            # Only test last 365 days
            # ------------------------------------------------

            cutoff = (
                df1.index.max()
                - pd.Timedelta(days=TEST_DAYS)
            )

            df1 = df1[
                df1.index >= cutoff
            ].copy()

            print(
                f"🔎 Testing {name}: "
                f"{len(df1)} candles"
            )

            trades = backtest_symbol(
                df1,
                name
            )

            print(
                f"   Candidate trades: "
                f"{len(trades)}"
            )

            all_candidates.extend(
                trades
            )

            # Individual symbol result
            if trades:

                metrics = calculate_metrics(
                    trades
                )

                symbol_results.append({
                    "Symbol": name,
                    **metrics
                })

        except Exception as e:

            print(
                f"\n❌ ERROR {name}: {e}"
            )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    final_trades = apply_portfolio_lock(
        all_candidates
    )

    print()
    print("=" * 70)
    print("📊 FINAL PORTFOLIO RESULT")
    print("=" * 70)

    metrics = calculate_metrics(
        final_trades
    )

    print(
        f"Total Trades : {metrics['Trades']}"
    )

    print(
        f"Wins         : {metrics['Wins']}"
    )

    print(
        f"Losses       : {metrics['Losses']}"
    )

    print(
        f"Timeouts     : {metrics['Timeouts']}"
    )

    print(
        f"Win Rate     : {metrics['WinRate']:.2f}%"
    )

    print(
        f"Net Return   : {metrics['NetPct']:.2f}%"
    )

    print(
        f"Avg Trade    : {metrics['AvgTradePct']:.3f}%"
    )

    print(
        f"Profit Factor: {metrics['ProfitFactor']:.2f}"
    )

    print(
        f"Max Drawdown : {metrics['MaxDD']:.2f}%"
    )

    # ========================================================
    # TRADES / DAY / YEAR
    # ========================================================

    if final_trades:

        trade_df = pd.DataFrame(
            final_trades
        )

        first_date = pd.to_datetime(
            trade_df["EntryTime"]
        ).min()

        last_date = pd.to_datetime(
            trade_df["EntryTime"]
        ).max()

        days = (
            last_date - first_date
        ).total_seconds() / 86400

        if days > 0:

            trades_per_day = (
                len(trade_df) / days
            )

        else:

            trades_per_day = 0

        trades_per_year = (
            trades_per_day * 365
        )

        print()
        print(
            f"Trades / Day  : "
            f"{trades_per_day:.2f}"
        )

        print(
            f"Trades / Year : "
            f"{trades_per_year:.0f}"
        )

    # ========================================================
    # SYMBOL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("📌 PER SYMBOL")
    print("=" * 70)

    if symbol_results:

        symbol_df = pd.DataFrame(
            symbol_results
        )

        print(
            symbol_df[
                [
                    "Symbol",
                    "Trades",
                    "Wins",
                    "Losses",
                    "Timeouts",
                    "WinRate",
                    "NetPct",
                    "ProfitFactor"
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # LONG / SHORT
    # ========================================================

    if final_trades:

        trade_df = pd.DataFrame(
            final_trades
        )

        print()
        print("=" * 70)
        print("📈 LONG / SHORT")
        print("=" * 70)

        side_rows = []

        for side in [
            "LONG",
            "SHORT"
        ]:

            x = trade_df[
                trade_df["Direction"] == side
            ]

            if len(x) == 0:
                continue

            wins = (
                x["Result"] == "WIN"
            ).sum()

            losses = (
                x["Result"] == "LOSS"
            ).sum()

            resolved = wins + losses

            wr = (
                wins / resolved * 100
                if resolved > 0
                else 0
            )

            side_rows.append({
                "Side": side,
                "Trades": len(x),
                "Wins": wins,
                "Losses": losses,
                "Timeouts": (
                    x["Result"] == "TIMEOUT"
                ).sum(),
                "NetPct": x["NetPct"].sum(),
                "WinRate": wr
            })

        if side_rows:

            print(
                pd.DataFrame(
                    side_rows
                ).to_string(
                    index=False
                )
            )

    # ========================================================
    # RESULT DISTRIBUTION
    # ========================================================

    if final_trades:

        print()
        print("=" * 70)
        print("🎯 RESULT DISTRIBUTION")
        print("=" * 70)

        result_counts = (
            pd.DataFrame(final_trades)
            ["Result"]
            .value_counts()
        )

        print(
            result_counts.to_string()
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if final_trades:

        result_df = pd.DataFrame(
            final_trades
        )

        result_df.to_csv(
            "hunter_x_1pct_2r_lbank_trades.csv",
            index=False
        )

        print()
        print(
            "💾 Saved:"
            " hunter_x_1pct_2r_lbank_trades.csv"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    print()
    print("=" * 70)
    print("🔐 ANTI LOOKAHEAD AUDIT")
    print("=" * 70)

    print("✅ Entry = next candle OPEN")
    print("✅ Previous 20 candles used for breakout")
    print("✅ Previous CLOSED 4H used")
    print("✅ Current unfinished 1H removed")
    print("✅ No future candle used for signal")
    print("✅ No overlapping portfolio trades")
    print("✅ Same candle TP+SL = SL first")
    print("✅ Timeout counted separately")
    print("✅ RR = 1:2")
    print("✅ Slippage not double-counted")

    print()
    print("=" * 70)
    print("🏁 BACKTEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
