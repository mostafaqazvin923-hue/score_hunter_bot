import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt"]
    )
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V74.2
# Golden Core + Market Breadth Shield
# Fixed Fee / PnL / Position Handling
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

REMOVED_COINS = {
    "NEAR", "OP", "HYPE", "HBAR", "AVAX",
    "SUI", "PENDLE", "TIA", "FET", "SEI",
    "ARB", "DOT", "ETC", "SHIB", "STX",
    "RUNE", "MKR", "APT", "LTC", "AR",
    "IMX", "PEPE", "BONK",
}

SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


# ============================================================
# STRATEGY SETTINGS
# ============================================================

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45
EMA_WARMUP = 200


# ============================================================
# MONEY MANAGEMENT (V74 CORE)
# ============================================================

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0


start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)


print("=" * 65)
print("📥 دریافت داده‌ها - HUNTER-V74.2")
print("=" * 65)

processed_data = {}


# ============================================================
# FETCH DATA
# ============================================================

def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000
                )
                break
            except Exception:
                if attempt == 2:
                    return None

        if not batch:
            break

        last_ts = batch[-1][0]
        if last_seen is not None and last_ts <= last_seen:
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # حذف کندل ناقص آخر
    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    # کنترل گپ دیتا
    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(hours=4, minutes=10):
        return None

    # ====================================================
    # INDICATORS
    # ====================================================

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))

    df["ATR"] = (
        pd.concat([tr1, tr2, tr3], axis=1)
        .max(axis=1)
        .rolling(ATR_PERIOD)
        .mean()
    )

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Mom_Short"] = (
        df["Close"] - df["Close"].shift(10)
    ) / df["Close"].shift(10)

    df["Mom_Long"] = (
        df["Close"] - df["Close"].shift(30)
    ) / df["Close"].shift(30)

    df.set_index("Date", inplace=True)
    return df


# ============================================================
# LOAD ALL SYMBOL DATA
# ============================================================

for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}")
print("⚙️ شروع اجرای بک‌تست HUNTER-V74.2...")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(processed_data):
    all_timestamps = sorted({
        ts
        for df in processed_data.values()
        for ts in df.index
    })

    active_positions = {}
    all_trades = []

    for ts in all_timestamps:
        symbols_to_close = []

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            # -----------------------------
            # LONG TRAILING
            # -----------------------------
            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_sl = (
                        pos["highest_price"]
                        - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    )
                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_sl

                hit_sl = c4h["Low"] <= pos["stop_loss"]

            # -----------------------------
            # SHORT TRAILING
            # -----------------------------
            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_sl = (
                        pos["lowest_price"]
                        + TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    )
                    if new_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_sl

                hit_sl = c4h["High"] >= pos["stop_loss"]

            current_index = df.index.get_loc(ts)
            candles_held = current_index - pos["entry_index"]
            timeout = candles_held >= TIMEOUT_CANDLES

            if hit_sl or timeout:
                entry = pos["entry_price"]
                risk = pos["initial_risk"]

                if pos["side"] == "LONG":
                    if hit_sl:
                        exit_price = min(pos["stop_loss"], c4h["Open"])
                    else:
                        exit_price = c4h["Close"]

                    raw_r = (exit_price - entry) / risk
                    price_return = (exit_price - entry) / entry
                else:
                    if hit_sl:
                        exit_price = max(pos["stop_loss"], c4h["Open"])
                    else:
                        exit_price = c4h["Close"]

                    raw_r = (entry - exit_price) / risk
                    price_return = (entry - exit_price) / entry

                # اصلاح کارمزد بر اساس R واقعی
                fee_r = (FEE_RATE * 2) / pos["sl_dist_pct"]
                real_r = raw_r - fee_r

                outcome = "WIN" if real_r > 0 else "LOSS"
                notional = TRADE_MARGIN * LEVERAGE

                dollar_pnl = (
                    notional * price_return
                ) - (
                    notional * FEE_RATE * 2
                )

                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": real_r,
                    "Dollar_PnL": dollar_pnl,
                    "ExitOrder": len(all_trades)
                })

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        # ====================================================
        # MARKET REGIME
        # ====================================================

        market_bull = True
        btc_c = None

        if "BTC" in processed_data and ts in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = btc_c["Close"] > btc_c["EMA200"]

        # ====================================================
        # MARKET BREADTH SHIELD
        # ====================================================

        bullish_count = 0
        total_symbols = 0

        for symbol, df in processed_data.items():
            if ts in df.index:
                total_symbols += 1
                if df.loc[ts, "Close"] > df.loc[ts, "EMA200"]:
                    bullish_count += 1

        market_breadth = (
            bullish_count / total_symbols
            if total_symbols > 0
            else 0.5
        )

        allow_longs = market_breadth >= 0.35
        allow_shorts = market_breadth <= 0.65

        # ====================================================
        # RANKING SYMBOLS
        # ====================================================

        scores = {}
        for symbol, df in processed_data.items():
            if ts in df.index:
                mom = df.loc[ts, "Mom_Long"]
                if not np.isnan(mom):
                    scores[symbol] = mom

        if not scores:
            continue

        ranked_symbols = sorted(
            scores.keys(),
            key=lambda x: scores[x],
            reverse=market_bull
        )

        # ====================================================
        # ENTRY ENGINE
        # ====================================================

        for symbol in ranked_symbols:
            if len(active_positions) >= MAX_POSITIONS:
                break

            if symbol in active_positions:
                continue

            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            c4h = df.iloc[i]
            prev = df.iloc[i - 1]

            # -----------------------------
            # LONG SETUP
            # -----------------------------
            if market_bull:
                if not allow_longs:
                    continue

                regime_ok = (
                    c4h["Close"] > c4h["EMA20"]
                    and c4h["EMA20"] > c4h["EMA50"]
                    and c4h["Close"] > c4h["EMA200"]
                )

                pullback_ok = prev["Low"] <= prev["EMA20"] * 1.015

                momentum_ok = (
                    c4h["Mom_Short"] > 0.012
                    and c4h["Mom_Long"] > 0.035
                )

                valid_signal = regime_ok and pullback_ok and momentum_ok
                side = "LONG"

            # -----------------------------
            # SHORT SETUP
            # -----------------------------
            else:
                if not allow_shorts:
                    continue

                regime_ok = (
                    c4h["Close"] < c4h["EMA20"]
                    and c4h["EMA20"] < c4h["EMA50"]
                    and c4h["Close"] < c4h["EMA200"]
                )

                pullback_ok = prev["High"] >= prev["EMA20"] * 0.985

                momentum_ok = (
                    c4h["Mom_Short"] < -0.012
                    and c4h["Mom_Long"] < -0.035
                )

                valid_signal = regime_ok and pullback_ok and momentum_ok
                side = "SHORT"

            if not valid_signal:
                continue

            # ====================================================
            # CREATE POSITION
            # ====================================================

            if side == "LONG":
                entry_price = c4h["Open"] * (1 + SLIPPAGE)
                stop_loss = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"]
            else:
                entry_price = c4h["Open"] * (1 - SLIPPAGE)
                stop_loss = entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]

            initial_risk = abs(entry_price - stop_loss)
            sl_dist_pct = initial_risk / entry_price

            # جلوگیری از استاپ خیلی کوچک یا خیلی بزرگ
            if not (0.01 <= sl_dist_pct <= 0.04):
                continue

            active_positions[symbol] = {
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "initial_risk": initial_risk,
                "sl_dist_pct": sl_dist_pct,
                "entry_index": i,
            }

    return pd.DataFrame(all_trades)


# ============================================================
# SUMMARY REPORT
# ============================================================

def summarize_result(trades_df):
    print("\n" + "=" * 68)
    print("📊 گزارش نهایی HUNTER-V74.2")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        return

    trades_df = trades_df.sort_values(
        ["Timestamp", "ExitOrder"],
        kind="stable"
    ).reset_index(drop=True)

    total_trades = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())

    win_rate = (
        wins / total_trades * 100
        if total_trades > 0
        else 0
    )

    net_r = float(trades_df["Return"].sum())
    total_pnl = float(trades_df["Dollar_PnL"].sum())
    final_capital = INITIAL_CAPITAL + total_pnl

    longs = trades_df[trades_df["Side"] == "LONG"]
    shorts = trades_df[trades_df["Side"] == "SHORT"]

    long_wr = (
        (longs["Outcome"] == "WIN").sum() / len(longs) * 100
        if len(longs)
        else 0
    )

    short_wr = (
        (shorts["Outcome"] == "WIN").sum() / len(shorts) * 100
        if len(shorts)
        else 0
    )

    max_loss_streak = 0
    current_loss = 0

    for outcome in trades_df["Outcome"]:
        if outcome == "LOSS":
            current_loss += 1
            max_loss_streak = max(max_loss_streak, current_loss)
        else:
            current_loss = 0

    equity = INITIAL_CAPITAL
    curve = []

    for pnl in trades_df["Dollar_PnL"]:
        equity += pnl
        curve.append(equity)

    curve = pd.Series(curve)
    peak = curve.cummax()
    dd = curve - peak
    max_dd = float(dd.min())

    print(f"🔸 سرمایه اولیه: ${INITIAL_CAPITAL:,.2f}")
    print(f"🔸 Margin هر معامله: ${TRADE_MARGIN:,.2f} | Leverage: {LEVERAGE}x")
    print(f"🔸 تعداد معاملات: {total_trades}\n")
    print(f"📈 LONG: تعداد={len(longs)} | WinRate={long_wr:.2f}%")
    print(f"📉 SHORT: تعداد={len(shorts)} | WinRate={short_wr:.2f}%\n")
    print(f"🎯 Win Rate کلی: {win_rate:.2f}%")
    print(f"💰 مجموع R: {net_r:.2f}R")
    print(f"💵 سود/زیان: ${total_pnl:,.2f}")
    print(f"🏦 سرمایه نهایی: ${final_capital:,.2f}")
    print(f"📉 Max Drawdown: ${max_dd:,.2f}")
    print(f"❄️ Max Loss Streak: {max_loss_streak}")


if __name__ == "__main__":
    df_trades = run_backtest(processed_data)
    summarize_result(df_trades)
    print("\n✨ بک‌تست HUNTER-V74.2 تمام شد.")
