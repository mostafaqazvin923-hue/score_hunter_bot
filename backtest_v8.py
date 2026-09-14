# backtest_v9.py
# HUNTER-ICHIMOKU V9
# 4H = regime/trend | 1H = Kijun Pullback + Reclaim entry
# RR 1:2 | No lookahead | LBank 1H data
#
# Required packages:
#   pip install ccxt pandas numpy requests

import os
import time
import ccxt
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT",
    "LTC/USDT", "UNI/USDT", "RENDER/USDT", "LINK/USDT",
    "ATOM/USDT"
]

EXCHANGE_ID = "lbank"
TIMEFRAME = "1h"

# 1-year window ending at current UTC time.
DAYS = 365
LIMIT = 1000

# Ichimoku standard settings
TENKAN = 9
KIJUN = 26
SENKOU_B = 52
DISPLACEMENT = 26

# Entry / risk
RR = 2.0
SL_BUFFER_ATR = 0.20
MIN_SL_PCT = 0.003       # 0.30%
MAX_SL_PCT = 0.040       # 4.00%
PULLBACK_LOOKBACK = 8
SWING_LOOKBACK = 8

# ATR
ATR_PERIOD = 14

# Execution costs
FEE_RATE = 0.0007        # per side
SLIPPAGE = 0.0003        # 0.03%

# Maximum concurrent positions
MAX_POSITIONS = 3

# Maximum holding period in 1H candles
MAX_HOLD = 24


# =========================
# INDICATORS
# =========================
def ichimoku(df):
    high = df["high"]
    low = df["low"]

    tenkan = (
        high.rolling(TENKAN).max() +
        low.rolling(TENKAN).min()
    ) / 2

    kijun = (
        high.rolling(KIJUN).max() +
        low.rolling(KIJUN).min()
    ) / 2

    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (
        high.rolling(SENKOU_B).max() +
        low.rolling(SENKOU_B).min()
    ) / 2

    # IMPORTANT:
    # For a decision at candle t, the cloud value that is
    # visible at t is the value calculated 26 candles earlier.
    span_a_visible = span_a_raw.shift(DISPLACEMENT)
    span_b_visible = span_b_raw.shift(DISPLACEMENT)

    out = df.copy()
    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["span_a"] = span_a_visible
    out["span_b"] = span_b_visible
    out["cloud_top"] = out[["span_a", "span_b"]].max(axis=1)
    out["cloud_bottom"] = out[["span_a", "span_b"]].min(axis=1)

    # Chikou confirmation without looking into the future:
    # current close versus close 26 candles ago.
    out["chikou_long"] = out["close"] > out["close"].shift(DISPLACEMENT)
    out["chikou_short"] = out["close"] < out["close"].shift(DISPLACEMENT)

    return out


def add_atr(df):
    out = df.copy()
    prev_close = out["close"].shift(1)

    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    out["atr"] = tr.rolling(ATR_PERIOD).mean()
    return out


# =========================
# DATA
# =========================
def fetch_ohlcv(exchange, symbol, since_ms, until_ms):
    rows = []
    cursor = since_ms

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=TIMEFRAME,
            since=cursor,
            limit=LIMIT
        )

        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]

        if last_ts <= cursor:
            break

        cursor = last_ts + 60 * 60 * 1000

        if len(batch) < LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[
        (df["timestamp"] >= since_ms) &
        (df["timestamp"] <= until_ms)
    ].copy()

    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    return df.reset_index(drop=True)


# =========================
# 4H REGIME
# =========================
def build_4h(df1h):
    x = df1h.set_index("datetime")[["open", "high", "low", "close", "volume"]]

    df4 = x.resample("4h", label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    df4 = ichimoku(df4.reset_index())
    df4 = add_atr(df4)

    return df4


def map_4h_to_1h(df1h, df4):
    a = df1h.copy()
    b = df4.copy()

    # Only CLOSED 4H candles can influence the following 1H candle.
    b["regime_time"] = b["datetime"]

    a = pd.merge_asof(
        a.sort_values("datetime"),
        b[[
            "regime_time",
            "close",
            "tenkan",
            "kijun",
            "cloud_top",
            "cloud_bottom",
            "chikou_long",
            "chikou_short"
        ]].rename(columns={
            "close": "h4_close",
            "tenkan": "h4_tenkan",
            "kijun": "h4_kijun",
            "cloud_top": "h4_cloud_top",
            "cloud_bottom": "h4_cloud_bottom",
            "chikou_long": "h4_chikou_long",
            "chikou_short": "h4_chikou_short"
        }),
        left_on="datetime",
        right_on="regime_time",
        direction="backward"
    )

    return a


# =========================
# SIGNAL
# =========================
def add_signals(df):
    x = df.copy()

    # 4H bullish regime
    x["regime_long"] = (
        (x["h4_close"] > x["h4_cloud_top"]) &
        (x["h4_tenkan"] > x["h4_kijun"]) &
        (x["h4_kijun"] >= x["h4_kijun"].shift(1)) &
        (x["h4_chikou_long"])
    )

    # 4H bearish regime
    x["regime_short"] = (
        (x["h4_close"] < x["h4_cloud_bottom"]) &
        (x["h4_tenkan"] < x["h4_kijun"]) &
        (x["h4_kijun"] <= x["h4_kijun"].shift(1)) &
        (x["h4_chikou_short"])
    )

    # 1H trend position
    x["above_cloud"] = x["close"] > x["cloud_top"]
    x["below_cloud"] = x["close"] < x["cloud_bottom"]

    # Price has recently touched / approached Kijun.
    # ATR-normalized distance avoids a fixed percentage threshold.
    x["near_kijun"] = (
        ((x["low"] <= x["kijun"] + 0.25 * x["atr"]) &
         (x["high"] >= x["kijun"] - 0.25 * x["atr"]))
        .rolling(PULLBACK_LOOKBACK)
        .max()
        .astype(bool)
    )

    # Reclaim candle:
    # previous candle was at/below Kijun, current candle closes above it.
    x["long_reclaim"] = (
        (x["close"] > x["kijun"]) &
        (x["close"].shift(1) <= x["kijun"].shift(1)) &
        (x["close"] > x["open"])
    )

    x["short_reclaim"] = (
        (x["close"] < x["kijun"]) &
        (x["close"].shift(1) >= x["kijun"].shift(1)) &
        (x["close"] < x["open"])
    )

    # Tenkan confirmation on the actual entry candle.
    x["tenkan_long"] = x["tenkan"] > x["kijun"]
    x["tenkan_short"] = x["tenkan"] < x["kijun"]

    # Displacement/strength: body must be meaningful relative to ATR.
    body = (x["close"] - x["open"]).abs()
    x["strong_long"] = (
        (x["close"] > x["open"]) &
        (body >= 0.30 * x["atr"])
    )
    x["strong_short"] = (
        (x["close"] < x["open"]) &
        (body >= 0.30 * x["atr"])
    )

    # Final signals are evaluated ONLY after the signal candle closes.
    x["long_signal"] = (
        x["regime_long"] &
        x["above_cloud"] &
        x["near_kijun"] &
        x["long_reclaim"] &
        x["tenkan_long"] &
        x["strong_long"]
    )

    x["short_signal"] = (
        x["regime_short"] &
        x["below_cloud"] &
        x["near_kijun"] &
        x["short_reclaim"] &
        x["tenkan_short"] &
        x["strong_short"]
    )

    return x


# =========================
# BACKTEST
# =========================
def simulate_trade(df, i, side):
    # Entry at NEXT candle open: no lookahead.
    if i + 1 >= len(df):
        return None

    signal = df.iloc[i]
    entry_candle = df.iloc[i + 1]

    entry = float(entry_candle["open"])

    if side == "LONG":
        entry *= (1 + SLIPPAGE)

        recent_low = df.iloc[max(0, i - SWING_LOOKBACK + 1):i + 1]["low"].min()
        sl = min(
            recent_low,
            float(signal["kijun"]) - SL_BUFFER_ATR * float(signal["atr"])
        )

        risk = entry - sl

        if risk <= 0:
            return None

        sl_pct = risk / entry
        if not (MIN_SL_PCT <= sl_pct <= MAX_SL_PCT):
            return None

        tp = entry + RR * risk

    else:
        entry *= (1 - SLIPPAGE)

        recent_high = df.iloc[max(0, i - SWING_LOOKBACK + 1):i + 1]["high"].max()
        sl = max(
            recent_high,
            float(signal["kijun"]) + SL_BUFFER_ATR * float(signal["atr"])
        )

        risk = sl - entry

        if risk <= 0:
            return None

        sl_pct = risk / entry
        if not (MIN_SL_PCT <= sl_pct <= MAX_SL_PCT):
            return None

        tp = entry - RR * risk

    fee_r = (2 * FEE_RATE) / sl_pct

    end = min(len(df), i + 1 + MAX_HOLD)

    result = "TIMEOUT"
    exit_price = float(df.iloc[end - 1]["close"])
    exit_index = end - 1

    for j in range(i + 1, end):
        c = df.iloc[j]

        if side == "LONG":
            hit_sl = c["low"] <= sl
            hit_tp = c["high"] >= tp
        else:
            hit_sl = c["high"] >= sl
            hit_tp = c["low"] <= tp

        # Conservative: if both are hit in one candle, count LOSS.
        if hit_sl and hit_tp:
            result = "LOSS"
            exit_price = sl
            exit_index = j
            break
        elif hit_sl:
            result = "LOSS"
            exit_price = sl
            exit_index = j
            break
        elif hit_tp:
            result = "WIN"
            exit_price = tp
            exit_index = j
            break

    if result == "WIN":
        gross_r = RR
    elif result == "LOSS":
        gross_r = -1.0
    else:
        if side == "LONG":
            gross_r = (exit_price - entry) / risk
        else:
            gross_r = (entry - exit_price) / risk

    net_r = gross_r - fee_r

    return {
        "side": side,
        "signal_time": signal["datetime"],
        "entry_time": entry_candle["datetime"],
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "result": result,
        "gross_r": gross_r,
        "net_r": net_r,
        "hold": exit_index - i,
        "exit_time": df.iloc[exit_index]["datetime"],
        "exit_price": exit_price
    }


def backtest_symbol(df):
    df = df.copy()
    df = ichimoku(df)
    df = add_atr(df)

    # Need enough history for 4H + Ichimoku.
    df4 = build_4h(df)
    df = map_4h_to_1h(df, df4)
    df = add_signals(df)

    trades = []
    active_until = -1

    for i in range(len(df) - MAX_HOLD - 1):
        if i <= active_until:
            continue

        side = None

        if bool(df.iloc[i]["long_signal"]):
            side = "LONG"
        elif bool(df.iloc[i]["short_signal"]):
            side = "SHORT"

        if side is None:
            continue

        trade = simulate_trade(df, i, side)

        if trade is None:
            continue

        trades.append(trade)
        active_until = i + trade["hold"]

    return trades


# =========================
# REPORT
# =========================
def print_summary(all_trades):
    if not all_trades:
        print("\nNo trades.")
        return

    t = pd.DataFrame(all_trades)

    total = len(t)
    wins = int((t["result"] == "WIN").sum())
    losses = int((t["result"] == "LOSS").sum())
    timeouts = int((t["result"] == "TIMEOUT").sum())

    wr_ex_timeout = (
        wins / (wins + losses) * 100
        if wins + losses else 0
    )

    net_r = t["net_r"].sum()
    avg_r = t["net_r"].mean()

    gross_profit = t.loc[t["net_r"] > 0, "net_r"].sum()
    gross_loss = -t.loc[t["net_r"] < 0, "net_r"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = t["net_r"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()

    print("\n" + "=" * 55)
    print("HUNTER-ICHIMOKU V9 — CLEAN BACKTEST")
    print("=" * 55)
    print(f"Trades              : {total}")
    print(f"Wins                : {wins}")
    print(f"Losses              : {losses}")
    print(f"Timeouts            : {timeouts}")
    print(f"WR (W/L only)       : {wr_ex_timeout:.2f}%")
    print(f"Net R               : {net_r:.2f}R")
    print(f"Avg R / trade       : {avg_r:.4f}R")
    print(f"Profit Factor       : {pf:.3f}")
    print(f"Max Drawdown        : {max_dd:.2f}R")
    print("=" * 55)

    print("\nExit reasons:")
    print(t["result"].value_counts().to_string())

    print("\nBy side:")
    for side in ["LONG", "SHORT"]:
        s = t[t["side"] == side]
        if len(s):
            w = (s["result"] == "WIN").sum()
            l = (s["result"] == "LOSS").sum()
            wr = w / (w + l) * 100 if w + l else 0
            print(
                f"{side:5s} | trades={len(s):4d} | "
                f"W={w:3d} L={l:3d} | WR={wr:6.2f}% | "
                f"Net={s['net_r'].sum():8.2f}R"
            )

    print("\nBy symbol:")
    if "symbol" in t.columns:
        for symbol, s in t.groupby("symbol"):
            w = (s["result"] == "WIN").sum()
            l = (s["result"] == "LOSS").sum()
            wr = w / (w + l) * 100 if w + l else 0
            print(
                f"{symbol:12s} | {len(s):4d} | "
                f"WR={wr:6.2f}% | Net={s['net_r'].sum():8.2f}R"
            )


# =========================
# MAIN
# =========================
def main():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    now_ms = exchange.milliseconds()
    since_ms = now_ms - DAYS * 24 * 60 * 60 * 1000

    all_trades = []

    for symbol in SYMBOLS:
        print(f"\nDownloading {symbol} ...")

        try:
            df = fetch_ohlcv(
                exchange,
                symbol,
                since_ms,
                now_ms
            )

            print(f"  Candles: {len(df)}")

            if len(df) < 3000:
                print("  Skipped: insufficient data.")
                continue

            trades = backtest_symbol(df)

            for tr in trades:
                tr["symbol"] = symbol

            all_trades.extend(trades)

            print(f"  Trades: {len(trades)}")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    if not all_trades:
        print("\nNo trades generated.")
        return

    # Sort chronologically so portfolio statistics are valid.
    all_trades = sorted(
        all_trades,
        key=lambda x: x["entry_time"]
    )

    # Global max concurrent position cap.
    accepted = []
    active = []

    for tr in all_trades:
        entry_time = tr["entry_time"]

        active = [
            x for x in active
            if x["exit_time"] > entry_time
        ]

        if len(active) < MAX_POSITIONS:
            accepted.append(tr)
            active.append(tr)

    print_summary(accepted)

    out = pd.DataFrame(accepted)
    out.to_csv("v9_trades.csv", index=False)
    print("\nSaved: v9_trades.csv")


if __name__ == "__main__":
    main()
