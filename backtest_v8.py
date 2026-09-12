# HUNTER-X CLEAN V6
# 1H Execution + Closed 4H Regime
# FIXED RR = 1:2
# NO LOOKAHEAD

import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LBANK_URL = "https://api.lbank.info/v2/kline.do"

SYMBOLS = [
    "btc_usdt",
    "eth_usdt",
    "sol_usdt",
    "xrp_usdt",
    "ada_usdt",
    "avax_usdt",
    "link_usdt",
    "near_usdt",
    "sui_usdt",
    "dot_usdt",
    "aave_usdt",
    "atom_usdt",
    "apt_usdt",
    "crv_usdt",
    "fet_usdt",
    "icp_usdt",
    "arb_usdt",
    "op_usdt",
    "pol_usdt",
    "ltc_usdt",
    "bch_usdt",
    "shib_usdt",
    "pepe_usdt",
    "grt_usdt",
    "inj_usdt",
    "pendle_usdt",
    "render_usdt",
    "tia_usdt",
    "xlm_usdt",
    "fil_usdt",
]

DAYS = 365
WARMUP_DAYS = 35

# ============================================================
# RISK / RR
# ============================================================

# هر باخت = -1R
# هر برد = +2R
RR = 2.0

# فقط برای محاسبه سایز پوزیشن
RISK_PER_TRADE = 0.005

FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

MAX_HOLD_BARS = 40


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def true_range(df):

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    return pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)


def atr(df, length=14):

    return true_range(df).ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


def adx(df, length=14):

    high = df["high"]
    low = df["low"]

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0
        ),
        index=df.index
    )

    tr = true_range(df)

    atr_value = tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        / atr_value
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


# ============================================================
# DOWNLOAD LBank
# ============================================================

def download_lbank(symbol):

    print(f"\n📥 Downloading {symbol} ...")

    now = datetime.now(timezone.utc)

    start = now - timedelta(
        days=DAYS + WARMUP_DAYS
    )

    start_ts = int(start.timestamp())
    end_ts = int(now.timestamp())

    all_rows = []

    cursor = end_ts

    for _ in range(30):

        params = {
            "symbol": symbol,
            "size": 2000,
            "type": "hour1",
            "time": cursor
        }

        response = requests.get(
            LBANK_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if str(
            data.get("result", "")
        ).lower() != "true":

            raise RuntimeError(
                f"LBank error: {data}"
            )

        rows = data.get("data", [])

        if not rows:
            break

        all_rows.extend(rows)

        oldest = min(
            int(row[0])
            for row in rows
        )

        if oldest <= start_ts:
            break

        cursor = oldest - 3600

        time.sleep(0.15)

    if not all_rows:

        raise RuntimeError(
            f"No data for {symbol}"
        )

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df
        .dropna()
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    df = df[
        (
            df["timestamp"]
            >= pd.Timestamp(
                start,
                tz="UTC"
            )
        )
        &
        (
            df["timestamp"]
            <= pd.Timestamp(
                now,
                tz="UTC"
            )
        )
    ]

    df = df.set_index(
        "timestamp"
    )

    print(
        f"📈 {symbol}: {len(df)} candles | "
        f"{df.index.min()} -> {df.index.max()}"
    )

    return df


# ============================================================
# 1H INDICATORS
# ============================================================

def prepare_1h(df):

    x = df.copy()

    x["ema20"] = x["close"].ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    x["ema50"] = x["close"].ewm(
        span=50,
        adjust=False,
        min_periods=50
    ).mean()

    x["ema200"] = x["close"].ewm(
        span=200,
        adjust=False,
        min_periods=200
    ).mean()

    x["rsi"] = rsi(
        x["close"],
        14
    )

    x["atr"] = atr(
        x,
        14
    )

    x["adx"] = adx(
        x,
        14
    )

    x["volume_ma"] = (
        x["volume"]
        .rolling(20)
        .mean()
    )

    x["range"] = (
        x["high"] - x["low"]
    )

    x["body"] = (
        x["close"] - x["open"]
    ).abs()

    x["body_ratio"] = (
        x["body"]
        /
        x["range"].replace(
            0,
            np.nan
        )
    )

    x["close_location"] = (
        (x["close"] - x["low"])
        /
        x["range"].replace(
            0,
            np.nan
        )
    )

    x["candle_atr"] = (
        x["range"]
        /
        x["atr"].replace(
            0,
            np.nan
        )
    )

    # IMPORTANT:
    # فقط کندل‌های قبل از کندل فعلی
    x["prior_low"] = (
        x["low"]
        .shift(1)
        .rolling(12)
        .min()
    )

    x["prior_high"] = (
        x["high"]
        .shift(1)
        .rolling(12)
        .max()
    )

    x["structure_low"] = (
        x["low"]
        .shift(1)
        .rolling(48)
        .min()
    )

    x["structure_high"] = (
        x["high"]
        .shift(1)
        .rolling(48)
        .max()
    )

    return x


# ============================================================
# 4H CLOSED CONTEXT
# ============================================================

def add_4h_context(df):

    four = pd.DataFrame({

        "open":
            df["open"].resample(
                "4h",
                label="right",
                closed="right"
            ).first(),

        "high":
            df["high"].resample(
                "4h",
                label="right",
                closed="right"
            ).max(),

        "low":
            df["low"].resample(
                "4h",
                label="right",
                closed="right"
            ).min(),

        "close":
            df["close"].resample(
                "4h",
                label="right",
                closed="right"
            ).last(),

        "volume":
            df["volume"].resample(
                "4h",
                label="right",
                closed="right"
            ).sum()
    }).dropna()

    four["ema20_4h"] = (
        four["close"]
        .ewm(
            span=20,
            adjust=False,
            min_periods=20
        )
        .mean()
    )

    four["ema50_4h"] = (
        four["close"]
        .ewm(
            span=50,
            adjust=False,
            min_periods=50
        )
        .mean()
    )

    four["ema200_4h"] = (
        four["close"]
        .ewm(
            span=200,
            adjust=False,
            min_periods=200
        )
        .mean()
    )

    four["rsi_4h"] = rsi(
        four["close"],
        14
    )

    four["atr_4h"] = atr(
        four,
        14
    )

    four["adx_4h"] = adx(
        four,
        14
    )

    # ========================================================
    # VERY IMPORTANT
    #
    # shift(1) means the 1H candle can ONLY see the previous
    # CLOSED 4H candle.
    # ========================================================

    context = four[
        [
            "ema20_4h",
            "ema50_4h",
            "ema200_4h",
            "rsi_4h",
            "atr_4h",
            "adx_4h"
        ]
    ].shift(1)

    context = context.reindex(
        df.index,
        method="ffill"
    )

    return df.join(
        context
    )


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(df, i):

    if i < 250:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    required = [
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "atr",
        "adx",
        "volume_ma",
        "prior_low",
        "prior_high",
        "structure_low",
        "structure_high",
        "ema20_4h",
        "ema50_4h",
        "ema200_4h",
        "rsi_4h",
        "adx_4h"
    ]

    for col in required:

        if pd.isna(row[col]):
            return None

    # ========================================================
    # 4H TREND
    # ========================================================

    long_4h = (
        row["ema20_4h"]
        >
        row["ema50_4h"]
        >
        row["ema200_4h"]
        and
        row["rsi_4h"] >= 50
        and
        row["adx_4h"] >= 18
    )

    short_4h = (
        row["ema20_4h"]
        <
        row["ema50_4h"]
        <
        row["ema200_4h"]
        and
        row["rsi_4h"] <= 50
        and
        row["adx_4h"] >= 18
    )

    if not long_4h and not short_4h:
        return None

    # ========================================================
    # 1H TREND
    # ========================================================

    long_trend = (
        row["ema20"]
        >
        row["ema50"]
        >
        row["ema200"]
        and
        row["close"]
        >
        row["ema50"]
        and
        row["ema20"]
        >
        prev["ema20"]
    )

    short_trend = (
        row["ema20"]
        <
        row["ema50"]
        <
        row["ema200"]
        and
        row["close"]
        <
        row["ema50"]
        and
        row["ema20"]
        <
        prev["ema20"]
    )

    if not long_trend and not short_trend:
        return None

    # ========================================================
    # ADX
    # ========================================================

    if row["adx"] < 16:
        return None

    # ========================================================
    # DON'T TRADE HUGE EXHAUSTION CANDLES
    # ========================================================

    if row["candle_atr"] > 2.8:
        return None

    # ========================================================
    # VOLUME
    # ========================================================

    volume_ok = (
        row["volume"]
        >=
        row["volume_ma"] * 1.05
    )

    # ========================================================
    # LIQUIDITY SWEEP
    # ========================================================

    long_sweep = (
        row["low"]
        <
        row["prior_low"]
        and
        row["close"]
        >
        row["prior_low"]
    )

    short_sweep = (
        row["high"]
        >
        row["prior_high"]
        and
        row["close"]
        <
        row["prior_high"]
    )

    # ========================================================
    # DISPLACEMENT
    # ========================================================

    bullish_displacement = (
        row["close"] > row["open"]
        and
        row["body"]
        >=
        0.45 * row["atr"]
        and
        row["body_ratio"]
        >=
        0.55
        and
        row["close_location"]
        >=
        0.65
    )

    bearish_displacement = (
        row["close"] < row["open"]
        and
        row["body"]
        >=
        0.45 * row["atr"]
        and
        row["body_ratio"]
        >=
        0.55
        and
        row["close_location"]
        <=
        0.35
    )

    # ========================================================
    # RSI MOMENTUM
    # ========================================================

    long_rsi = (
        row["rsi"] > 48
        and
        row["rsi"] > prev["rsi"]
    )

    short_rsi = (
        row["rsi"] < 52
        and
        row["rsi"] < prev["rsi"]
    )

    # ========================================================
    # LONG
    # ========================================================

    if (
        long_4h
        and
        long_trend
        and
        long_sweep
        and
        bullish_displacement
        and
        long_rsi
        and
        volume_ok
    ):

        stop = (
            min(
                row["low"],
                row["prior_low"]
            )
            -
            0.20 * row["atr"]
        )

        return {
            "side": "LONG",
            "stop": float(stop),
            "score": 10
        }

    # ========================================================
    # SHORT
    # ========================================================

    if (
        short_4h
        and
        short_trend
        and
        short_sweep
        and
        bearish_displacement
        and
        short_rsi
        and
        volume_ok
    ):

        stop = (
            max(
                row["high"],
                row["prior_high"]
            )
            +
            0.20 * row["atr"]
        )

        return {
            "side": "SHORT",
            "stop": float(stop),
            "score": 10
        }

    return None


# ============================================================
# SIMULATION
# ============================================================

def simulate_symbol(
    symbol,
    df,
    initial_equity=10000
):

    trades = []

    equity = initial_equity

    i = 250

    cooldown = 0

    while i < len(df) - 1:

        if cooldown > 0:

            cooldown -= 1
            i += 1
            continue

        signal = generate_signal(
            df,
            i
        )

        if signal is None:

            i += 1
            continue

        # ====================================================
        # ENTRY = NEXT CANDLE OPEN
        # ====================================================

        entry_index = i + 1

        entry_candle = df.iloc[
            entry_index
        ]

        raw_entry = float(
            entry_candle["open"]
        )

        side = signal["side"]

        # Entry slippage
        if side == "LONG":

            entry = (
                raw_entry
                *
                (1 + SLIPPAGE_RATE)
            )

        else:

            entry = (
                raw_entry
                *
                (1 - SLIPPAGE_RATE)
            )

        stop = signal["stop"]

        if side == "LONG":

            risk_distance = (
                entry - stop
            )

        else:

            risk_distance = (
                stop - entry
            )

        if risk_distance <= 0:

            i += 1
            continue

        # Don't allow ridiculous stops.
        if risk_distance > (
            1.6 * float(
                df.iloc[i]["atr"]
            )
        ):

            i += 1
            continue

        # ====================================================
        # FIXED RR 1:2
        # ====================================================

        if side == "LONG":

            target = (
                entry
                +
                RR * risk_distance
            )

        else:

            target = (
                entry
                -
                RR * risk_distance
            )

        # ====================================================
        # SIMULATE AFTER ENTRY
        #
        # This is NOT lookahead for signal generation.
        # The signal was already created at candle i.
        # We are only resolving the trade after entry.
        # ====================================================

        risk_cash = (
            equity
            *
            RISK_PER_TRADE
        )

        qty = (
            risk_cash
            /
            risk_distance
        )

        result = None
        exit_price = None
        exit_index = None

        max_end = min(
            len(df),
            entry_index + MAX_HOLD_BARS
        )

        for j in range(
            entry_index,
            max_end
        ):

            candle = df.iloc[j]

            high = float(
                candle["high"]
            )

            low = float(
                candle["low"]
            )

            if side == "LONG":

                hit_sl = (
                    low <= stop
                )

                hit_tp = (
                    high >= target
                )

            else:

                hit_sl = (
                    high >= stop
                )

                hit_tp = (
                    low <= target
                )

            # Conservative:
            # If both happen in same candle,
            # assume SL happened first.
            if hit_sl and hit_tp:

                result = "SL"
                exit_price = stop
                exit_index = j

                break

            if hit_sl:

                result = "SL"
                exit_price = stop
                exit_index = j

                break

            if hit_tp:

                result = "TP"
                exit_price = target
                exit_index = j

                break

        # ====================================================
        # TIMEOUT
        # ====================================================

        if result is None:

            exit_index = (
                max_end - 1
            )

            exit_price = float(
                df.iloc[
                    exit_index
                ]["close"]
            )

            result = "TIME"

        # ====================================================
        # EXIT SLIPPAGE
        # ====================================================

        if side == "LONG":

            exit_price *= (
                1 - SLIPPAGE_RATE
            )

        else:

            exit_price *= (
                1 + SLIPPAGE_RATE
            )

        # ====================================================
        # PNL
        # ====================================================

        if side == "LONG":

            gross_pnl = (
                exit_price - entry
            ) * qty

        else:

            gross_pnl = (
                entry - exit_price
            ) * qty

        entry_fee = (
            entry
            *
            qty
            *
            FEE_RATE
        )

        exit_fee = (
            abs(exit_price)
            *
            qty
            *
            FEE_RATE
        )

        net_pnl = (
            gross_pnl
            -
            entry_fee
            -
            exit_fee
        )

        # R based on actual net result.
        r_multiple = (
            net_pnl
            /
            risk_cash
        )

        equity += net_pnl

        trades.append({
            "symbol": symbol,
            "side": side,
            "signal_time": df.index[i],
            "entry_time": df.index[entry_index],
            "exit_time": df.index[exit_index],
            "entry": entry,
            "stop": stop,
            "target": target,
            "exit": exit_price,
            "risk_cash": risk_cash,
            "result": result,
            "R": r_multiple,
            "PnL": net_pnl,
            "bars_held": (
                exit_index
                -
                entry_index
                +
                1
            ),
            "score": signal["score"],
            "equity": equity
        })

        # Never immediately enter on the same exit.
        cooldown = 2

        # Jump to after the trade.
        i = exit_index + 1

    return trades


# ============================================================
# REPORT
# ============================================================

def report(trades):

    if not trades:

        print(
            "\n❌ No trades generated."
        )

        return

    df = pd.DataFrame(
        trades
    )

    wins = df[
        df["result"] == "TP"
    ]

    losses = df[
        df["result"] == "SL"
    ]

    timeouts = df[
        df["result"] == "TIME"
    ]

    wl = (
        len(wins)
        +
        len(losses)
    )

    win_rate_wl = (
        len(wins)
        /
        wl
        *
        100
        if wl
        else 0
    )

    win_rate_all = (
        len(wins)
        /
        len(df)
        *
        100
    )

    net_r = df["R"].sum()

    gross_win = (
        wins["R"].sum()
        if len(wins)
        else 0
    )

    gross_loss = abs(
        losses["R"].sum()
    ) if len(losses) else 0

    profit_factor = (
        gross_win
        /
        gross_loss
        if gross_loss > 0
        else float("inf")
    )

    expectancy = (
        df["R"].mean()
    )

    equity = df[
        "equity"
    ].astype(float)

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    max_dd = (
        abs(drawdown.min())
        *
        100
    )

    print("\n")
    print("=" * 75)
    print("🏹 HUNTER-X CLEAN V6")
    print("CAUSAL STATE-MACHINE")
    print("=" * 75)

    print(
        f"Total Trades       : {len(df)}"
    )

    print(
        f"Wins               : {len(wins)}"
    )

    print(
        f"Losses             : {len(losses)}"
    )

    print(
        f"Timeouts           : {len(timeouts)}"
    )

    print(
        f"Win Rate (W/L)     : {win_rate_wl:.2f}%"
    )

    print(
        f"Win Rate (All)     : {win_rate_all:.2f}%"
    )

    print(
        f"Net Profit         : {net_r:.2f}R"
    )

    print(
        f"Profit Factor      : {profit_factor:.2f}"
    )

    print(
        f"Expectancy         : {expectancy:.4f}R"
    )

    print(
        f"Average Win        : "
        f"{wins['R'].mean() if len(wins) else 0:.2f}R"
    )

    print(
        f"Average Loss       : "
        f"{losses['R'].mean() if len(losses) else 0:.2f}R"
    )

    print(
        f"Max Drawdown       : {max_dd:.2f}%"
    )

    print(
        f"RR                 : 1:{RR:.1f}"
    )

    print("=" * 75)

    # ========================================================
    # SYMBOL REPORT
    # ========================================================

    print("\n📊 SYMBOL REPORT")

    rows = []

    for symbol, group in df.groupby(
        "symbol"
    ):

        w = group[
            group["result"] == "TP"
        ]

        l = group[
            group["result"] == "SL"
        ]

        t = group[
            group["result"] == "TIME"
        ]

        wl_count = (
            len(w)
            +
            len(l)
        )

        rows.append({
            "Symbol": symbol,
            "Trades": len(group),
            "Wins": len(w),
            "Losses": len(l),
            "Timeouts": len(t),
            "WinRate_WL": (
                len(w)
                /
                wl_count
                *
                100
                if wl_count
                else 0
            ),
            "WinRate_All": (
                len(w)
                /
                len(group)
                *
                100
            ),
            "NetR": group["R"].sum(),
            "AvgR": group["R"].mean()
        })

    symbol_report = pd.DataFrame(
        rows
    )

    symbol_report = (
        symbol_report
        .sort_values(
            "NetR",
            ascending=False
        )
    )

    print(
        symbol_report.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        "HUNTER_X_CLEAN_V6_TRADES.csv",
        index=False
    )

    symbol_report.to_csv(
        "HUNTER_X_CLEAN_V6_SYMBOL_REPORT.csv",
        index=False
    )

    print(
        "\n💾 Saved:"
    )

    print(
        "HUNTER_X_CLEAN_V6_TRADES.csv"
    )

    print(
        "HUNTER_X_CLEAN_V6_SYMBOL_REPORT.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--days",
        type=int,
        default=DAYS
    )

    args = parser.parse_args()

    global DAYS

    DAYS = args.days

    print("=" * 75)
    print("🏹 HUNTER-X CLEAN V6")
    print("=" * 75)

    print(
        f"Period       : {DAYS} days"
    )

    print(
        "Timeframes   : 1H + CLOSED 4H"
    )

    print(
        "Entry        : NEXT 1H OPEN"
    )

    print(
        "Risk/Reward  : 1:2 FIXED"
    )

    print(
        "Lookahead    : DISABLED"
    )

    print("=" * 75)

    all_trades = []

    cache_dir = Path(
        "lbank_cache_v6"
    )

    cache_dir.mkdir(
        exist_ok=True
    )

    for index, symbol in enumerate(
        SYMBOLS,
        1
    ):

        print(
            f"\n[{index}/{len(SYMBOLS)}] {symbol}"
        )

        try:

            cache_file = (
                cache_dir
                /
                f"{symbol}_{DAYS}.csv"
            )

            if cache_file.exists():

                df = pd.read_csv(
                    cache_file,
                    parse_dates=[
                        "timestamp"
                    ]
                )

                df["timestamp"] = (
                    pd.to_datetime(
                        df["timestamp"],
                        utc=True
                    )
                )

                df = df.set_index(
                    "timestamp"
                )

                print(
                    f"💾 Cache loaded: "
                    f"{len(df)} candles"
                )

            else:

                df = download_lbank(
                    symbol
                )

                df.reset_index().rename(
                    columns={
                        "timestamp":
                        "timestamp"
                    }
                ).to_csv(
                    cache_file,
                    index=False
                )

            df = prepare_1h(
                df
            )

            df = add_4h_context(
                df
            )

            trades = simulate_symbol(
                symbol,
                df
            )

            print(
                f"✅ {symbol}: "
                f"{len(trades)} trades"
            )

            all_trades.extend(
                trades
            )

        except Exception as e:

            print(
                f"❌ ERROR {symbol}: {e}"
            )

    report(
        all_trades
    )


if __name__ == "__main__":
    main()
