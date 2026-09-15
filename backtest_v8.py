import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V83.6
#
# Fixed:
# - Per-symbol next candle
# - Correct R fee calculation
# - Execution-time margin sizing
# - Portfolio mark-to-market equity
# - Portfolio Max Drawdown
# - Closed-trade Max Drawdown
# - Open positions at test end
# - Final MTM equity
# - Incremental realized PnL
# - Equity <= 0 protection
#
# Strategy parameters are unchanged from V83.5
# ============================================================


# ============================================================
# LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True
})


# ============================================================
# UNIVERSE
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
    "NEAR",
    "OP",
    "HYPE",
    "HBAR",
    "AVAX",
    "SUI",
    "PENDLE",
    "TIA",
    "FET",
    "SEI",
    "ARB",
    "DOT",
    "ETC",
    "SHIB",
    "STX",
    "RUNE",
    "MKR",
    "APT",
    "LTC",
    "AR",
    "IMX",
    "PEPE",
    "BONK",
}


SYMBOLS = {
    k: v
    for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14

INITIAL_ATR_MULTIPLIER = 2.0
TRAILING_ATR_MULTIPLIER = 2.0

EMA_WARMUP = 200


# ============================================================
# FINANCIAL PARAMETERS
# ============================================================

INITIAL_CAPITAL = 1000.0

BASE_MARGIN_PCT = 0.02
MAX_TOTAL_MARGIN_PCT = 0.12
LEVERAGE = 30.0


# ============================================================
# DATA RANGE
# ============================================================

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)


print("=" * 68)
print("📥 دریافت داده‌ها - HUNTER-V83.6")
print("=" * 68)


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
                    limit=1000,
                )
                break

            except Exception as e:
                if attempt == 2:
                    print(f"⚠️ خطا در دریافت {lbank_symbol}: {e}")
                    return None

        if not batch:
            break

        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            print(f"⚠️ pagination تکراری برای {lbank_symbol}")
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
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        candle_ms = 4 * 60 * 60 * 1000

        if last_ms + candle_ms > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()
    if not deltas.empty:
        if deltas.max() > pd.Timedelta(hours=4, minutes=10):
            print(f"⚠️ {lbank_symbol}: gap بزرگ‌تر از 4h10m")
            return None

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = true_range.rolling(ATR_PERIOD).mean()
    df["ATR_Avg50"] = df["ATR"].rolling(50).mean()
    df["ATR_Ratio"] = df["ATR"] / df["ATR_Avg50"]

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Mom_Short"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    df["Mom_Long"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30)

    df.set_index("Date", inplace=True)
    return df


for symbol, lbank_symbol in SYMBOLS.items():
    print(f"   📡 {symbol:<8} {lbank_symbol}")
    df4h = fetch_symbol_data(lbank_symbol)

    if df4h is not None:
        processed_data[symbol] = df4h
        print(f"      ✅ {len(df4h):,} candles")
    else:
        print(f"      ❌ حذف شد")

print()
print(f"✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}")
print("⚙️ شروع اجرای بک‌تست HUNTER-V83.6...")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_unrealized_pnl(ts, active_positions, processed_data):
    unrealized_pnl = 0.0

    for symbol, pos in active_positions.items():
        df = processed_data[symbol]
        if ts not in df.index:
            continue

        c_price = float(df.loc[ts, "Close"])
        notional = pos["used_margin"] * LEVERAGE

        if pos["side"] == "LONG":
            ret_pct = (c_price - pos["entry_price"]) / pos["entry_price"]
        else:
            ret_pct = (pos["entry_price"] - c_price) / pos["entry_price"]

        unrealized_pnl += notional * ret_pct

    return float(unrealized_pnl)


def calculate_current_equity(realized_pnl, unrealized_pnl):
    return INITIAL_CAPITAL + realized_pnl + unrealized_pnl


def calculate_market_state(ts, processed_data):
    btc_c = None
    regime_multiplier = 1.0

    if "BTC" in processed_data and ts in processed_data["BTC"].index:
        btc_c = processed_data["BTC"].loc[ts]
        atr_ratio = btc_c["ATR_Ratio"]

        if not np.isnan(atr_ratio) and atr_ratio > 1.5:
            regime_multiplier = 0.6
        elif btc_c["Close"] > btc_c["EMA20"] and btc_c["EMA20"] > btc_c["EMA50"]:
            regime_multiplier = 1.0
        else:
            regime_multiplier = 0.5

    market_bull = True
    if btc_c is not None:
        market_bull = btc_c["Close"] > btc_c["EMA200"]

    bullish_count = 0
    total_active_symbols = 0

    for symbol, df in processed_data.items():
        if ts not in df.index:
            continue
        total_active_symbols += 1
        row = df.loc[ts]
        if row["Close"] > row["EMA200"]:
            bullish_count += 1

    if total_active_symbols > 0:
        market_breadth_ratio = bullish_count / total_active_symbols
    else:
        market_breadth_ratio = 0.5

    allow_longs = True
    allow_shorts = True

    if btc_c is not None:
        btc_below_200 = btc_c["Close"] < btc_c["EMA200"]
        if btc_below_200 and market_breadth_ratio < 0.35:
            allow_longs = False
        elif not btc_below_200 and market_breadth_ratio > 0.65:
            allow_shorts = False

    return {
        "btc_c": btc_c,
        "regime_multiplier": regime_multiplier,
        "market_bull": market_bull,
        "market_breadth_ratio": market_breadth_ratio,
        "allow_longs": allow_longs,
        "allow_shorts": allow_shorts,
    }


def get_cluster_multiplier(loss_pressure):
    if loss_pressure == 0:
        return 1.0
    elif loss_pressure == 1:
        return 0.5
    elif loss_pressure == 2:
        return 0.3
    else:
        return 0.15


def update_loss_pressure(loss_pressure, closed_losses, closed_wins):
    if closed_losses > 0:
        if closed_losses >= 2:
            loss_pressure += closed_losses
        else:
            loss_pressure += 1
    elif closed_wins > 0:
        if loss_pressure > 0:
            loss_pressure -= 1

    return max(0, loss_pressure)


def calculate_trade_result(pos, candle):
    side = pos["side"]
    if side == "LONG":
        hit_sl = candle["Low"] <= pos["stop_loss"]
        if not hit_sl and candle["High"] > pos["highest_price"]:
            pos["highest_price"] = candle["High"]
            new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * candle["ATR"]
            if new_trailing_sl > pos["stop_loss"]:
                pos["stop_loss"] = new_trailing_sl
    else:
        hit_sl = candle["High"] >= pos["stop_loss"]
        if not hit_sl and candle["Low"] < pos["lowest_price"]:
            pos["lowest_price"] = candle["Low"]
            new_trailing_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * candle["ATR"]
            if new_trailing_sl < pos["stop_loss"]:
                pos["stop_loss"] = new_trailing_sl

    return hit_sl


def build_exit_result(pos, candle, hit_sl, is_timeout, early_exit):
    side = pos["side"]
    entry_price = pos["entry_price"]
    initial_risk = pos["initial_risk"]
    sl_dist_pct = pos["sl_dist_pct"]

    if side == "LONG":
        if hit_sl and not early_exit:
            exit_price = min(pos["stop_loss"], candle["Open"])
        else:
            exit_price = candle["Close"]

        raw_r = (exit_price - entry_price) / initial_risk
        price_return_pct = (exit_price - entry_price) / entry_price
    else:
        if hit_sl and not early_exit:
            exit_price = max(pos["stop_loss"], candle["Open"])
        else:
            exit_price = candle["Close"]

        raw_r = (entry_price - exit_price) / initial_risk
        price_return_pct = (entry_price - exit_price) / entry_price

    fee_pct_of_price = FEE_RATE * 2
    fee_in_r = fee_pct_of_price / sl_dist_pct
    net_r = raw_r - fee_in_r

    notional = pos["used_margin"] * LEVERAGE
    dollar_pnl = notional * price_return_pct - (notional * FEE_RATE * 2)

    if net_r > 0:
        outcome = "WIN"
    else:
        outcome = "LOSS"

    return {
        "exit_price": float(exit_price),
        "raw_r": float(raw_r),
        "fee_r": float(fee_in_r),
        "Return": float(net_r),
        "Dollar_PnL": float(dollar_pnl),
        "Outcome": outcome,
    }


# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================

def run_backtest(processed_data):
    if not processed_data:
        print("❌ هیچ داده معتبری برای بک‌تست وجود ندارد.")
        return pd.DataFrame(), pd.DataFrame(), {}

    all_timestamps = sorted({ts for df in processed_data.values() for ts in df.index})
    timestamp_map = {ts: i for i, ts in enumerate(all_timestamps)}

    active_positions = {}
    all_trades = []
    pending_signals = []
    equity_records = []
    realized_pnl = 0.0
    loss_pressure = 0

    for ts in all_timestamps:
        symbols_to_close = []
        closed_losses_this_candle = 0
        closed_wins_this_candle = 0

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                hit_sl = c4h["Low"] <= pos["stop_loss"]
                if not hit_sl and c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
            else:
                hit_sl = c4h["High"] >= pos["stop_loss"]
                if not hit_sl and c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_trailing_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl

            curr_i = df.index.get_loc(ts)
            candles_held = curr_i - pos["entry_index"]
            max_timeout = 24 if symbol in ["BTC", "ETH"] else 18
            is_timeout = candles_held >= max_timeout

            early_exit = False
            if candles_held >= 12 and not hit_sl:
                if pos["side"] == "LONG":
                    current_r = (c4h["Close"] - pos["entry_price"]) / pos["initial_risk"]
                else:
                    current_r = (pos["entry_price"] - c4h["Close"]) / pos["initial_risk"]
                if current_r < 0:
                    early_exit = True

            if hit_sl or is_timeout or early_exit:
                result = build_exit_result(
                    pos=pos,
                    candle=c4h,
                    hit_sl=hit_sl,
                    is_timeout=is_timeout,
                    early_exit=early_exit,
                )

                realized_pnl += result["Dollar_PnL"]

                if hit_sl and not early_exit:
                    exit_order = "STOP"
                elif is_timeout:
                    exit_order = "TIMEOUT"
                elif early_exit:
                    exit_order = "EARLY_EXIT"
                else:
                    exit_order = "OTHER"

                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": result["Outcome"],
                    "Return": result["Return"],
                    "Dollar_PnL": result["Dollar_PnL"],
                    "EntryPrice": pos["entry_price"],
                    "ExitPrice": result["exit_price"],
                    "UsedMargin": pos["used_margin"],
                    "Notional": pos["used_margin"] * LEVERAGE,
                    "SL_Distance_Pct": pos["sl_dist_pct"] * 100,
                    "Raw_R": result["raw_r"],
                    "Fee_R": result["fee_r"],
                    "ExitOrder": len(all_trades),
                    "ExitReason": exit_order,
                })

                if result["Outcome"] == "LOSS":
                    closed_losses_this_candle += 1
                else:
                    closed_wins_this_candle += 1

                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            del active_positions[symbol]

        loss_pressure = update_loss_pressure(
            loss_pressure=loss_pressure,
            closed_losses=closed_losses_this_candle,
            closed_wins=closed_wins_this_candle,
        )

        cluster_multiplier = get_cluster_multiplier(loss_pressure)

        unrealized_pnl = calculate_unrealized_pnl(
            ts=ts,
            active_positions=active_positions,
            processed_data=processed_data,
        )

        current_equity = calculate_current_equity(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
        )

        if current_equity <= 0:
            print()
            print("🛑 Equity به صفر یا منفی رسید.")
            print(f"   Timestamp: {ts}")
            print(f"   Equity: ${current_equity:,.2f}")
            break

        market_state = calculate_market_state(ts=ts, processed_data=processed_data)
        regime_multiplier = market_state["regime_multiplier"]
        market_bull = market_state["market_bull"]
        market_breadth_ratio = market_state["market_breadth_ratio"]
        allow_longs = market_state["allow_longs"]
        allow_shorts = market_state["allow_shorts"]

        total_active_margin = sum(p["used_margin"] for p in active_positions.values())
        max_allowed_margin = current_equity * MAX_TOTAL_MARGIN_PCT

        valid_pending = [sig for sig in pending_signals if sig["signal_ts"] == ts]
        pending_signals = [sig for sig in pending_signals if sig["signal_ts"] > ts]

        if valid_pending:
            base_margin_amount = current_equity * BASE_MARGIN_PCT
            current_trade_margin = base_margin_amount * regime_multiplier * cluster_multiplier

            for sig in valid_pending:
                if len(active_positions) >= MAX_POSITIONS:
                    break

                symbol = sig["symbol"]
                if symbol in active_positions:
                    continue

                df = processed_data[symbol]
                if ts not in df.index:
                    continue

                i = df.index.get_loc(ts)
                c4h = df.iloc[i]
                side = sig["side"]

                current_side_count = sum(1 for p in active_positions.values() if p["side"] == side)
                if current_side_count >= 2:
                    continue

                if current_trade_margin <= 0:
                    continue

                if total_active_margin + current_trade_margin > max_allowed_margin:
                    break

                if pd.isna(c4h["ATR"]):
                    continue

                if side == "LONG":
                    entry_price = c4h["Open"] * (1 + SLIPPAGE)
                    initial_sl = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"]
                else:
                    entry_price = c4h["Open"] * (1 - SLIPPAGE)
                    initial_sl = entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]

                initial_risk = abs(entry_price - initial_sl)
                if entry_price <= 0 or initial_risk <= 0:
                    continue

                sl_dist_pct = initial_risk / entry_price
                if not (0.008 <= sl_dist_pct <= 0.025):
                    continue

                active_positions[symbol] = {
                    "side": side,
                    "entry_price": float(entry_price),
                    "stop_loss": float(initial_sl),
                    "highest_price": float(entry_price),
                    "lowest_price": float(entry_price),
                    "initial_risk": float(initial_risk),
                    "sl_dist_pct": float(sl_dist_pct),
                    "entry_index": int(i),
                    "used_margin": float(current_trade_margin),
                }

                total_active_margin += current_trade_margin

        total_active_margin = sum(p["used_margin"] for p in active_positions.values())

        equity_records.append({
            "Timestamp": ts,
            "Realized_PnL": float(realized_pnl),
            "Unrealized_PnL": float(unrealized_pnl),
            "Equity": float(current_equity),
            "Active_Positions": int(len(active_positions)),
            "Active_Margin": float(total_active_margin),
            "Margin_Utilization_Pct": (total_active_margin / current_equity) * 100 if current_equity > 0 else 0,
            "Loss_Pressure": int(loss_pressure),
            "Cluster_Multiplier": float(cluster_multiplier),
            "BTC_Bull": bool(market_bull),
            "Breadth": float(market_breadth_ratio),
        })

        current_scores = {}
        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue
            val = df.loc[ts, "Mom_Long"]
            if not np.isnan(val):
                current_scores[symbol] = val

        if not current_scores:
            continue

        ranked_symbols = sorted(
            current_scores.keys(),
            key=lambda x: current_scores[x],
            reverse=market_bull,
        )

        for symbol in ranked_symbols:
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            if i + 1 >= len(df):
                continue

            next_symbol_ts = df.index[i + 1]
            c4h = df.iloc[i]
            prev_c = df.iloc[i - 1]

            if market_bull:
                side = "LONG"
            else:
                side = "SHORT"

            if side == "LONG" and not allow_longs:
                continue
            if side == "SHORT" and not allow_shorts:
                continue

            if market_bull:
                regime_ok = c4h["Close"] > c4h["EMA20"] and c4h["EMA20"] > c4h["EMA50"]
                pullback_ok = prev_c["Low"] <= prev_c["EMA20"] * 1.015
                no_pump_chase = c4h["Close"] < prev_c["Close"] * 1.02
                valid_signal = (
                    regime_ok
                    and pullback_ok
                    and no_pump_chase
                    and c4h["Mom_Short"] > 0.005
                    and c4h["Mom_Long"] > 0.015
                )
                side = "LONG"
            else:
                regime_ok = c4h["Close"] < c4h["EMA20"] and c4h["EMA20"] < c4h["EMA50"]
                pullback_ok = prev_c["High"] >= prev_c["EMA20"] * 0.985
                mom_short_ok = c4h["Mom_Short"] < -0.012 and c4h["Mom_Long"] < -0.03
                bear_confirm = (c4h["EMA50"] < c4h["EMA200"]) or (c4h["Close"] < c4h["EMA200"])
                valid_signal = regime_ok and pullback_ok and mom_short_ok and bear_confirm
                side = "SHORT"

            if valid_signal:
                pending_signals.append({
                    "symbol": symbol,
                    "side": side,
                    "signal_ts": next_symbol_ts,
                })

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(equity_records)

    final_state = {
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(equity_df.iloc[-1]["Unrealized_PnL"]) if not equity_df.empty else 0.0,
        "final_realized_capital": INITIAL_CAPITAL + realized_pnl,
        "final_mtm_equity": float(equity_df.iloc[-1]["Equity"]) if not equity_df.empty else INITIAL_CAPITAL,
        "active_positions": active_positions.copy(),
        "loss_pressure": int(loss_pressure),
    }

    return trades_df, equity_df, final_state


# ============================================================
# REPORT & SUMMARY FUNCTIONS
# ============================================================

def calculate_max_drawdown(equity_series):
    if equity_series.empty:
        return {"max_dd_dollar": 0.0, "max_dd_pct": 0.0}

    equity = equity_series.astype(float)
    peak = equity.cummax()
    drawdown = equity - peak
    drawdown_pct = (drawdown / peak.replace(0, np.nan)) * 100

    return {
        "max_dd_dollar": float(drawdown.min()),
        "max_dd_pct": float(drawdown_pct.min()),
    }


def calculate_max_loss_streak(trades_df):
    if trades_df.empty:
        return 0

    max_losses = 0
    current_losses = 0

    for outcome in trades_df["Outcome"]:
        if outcome == "LOSS":
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0

    return max_losses


def calculate_loss_cluster_stats(trades_df):
    if trades_df.empty:
        return {"max_losses_same_timestamp": 0, "losses_in_clusters": 0, "loss_cluster_pct": 0.0}

    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    if losses.empty:
        return {"max_losses_same_timestamp": 0, "losses_in_clusters": 0, "loss_cluster_pct": 0.0}

    grouped = losses.groupby("Timestamp").size()
    max_same_timestamp = int(grouped.max())
    clustered_losses = int(grouped[grouped >= 2].sum())
    total_losses = len(losses)
    cluster_pct = (clustered_losses / total_losses) * 100

    return {
        "max_losses_same_timestamp": max_same_timestamp,
        "losses_in_clusters": clustered_losses,
        "loss_cluster_pct": float(cluster_pct),
    }


def print_open_positions(active_positions):
    print()
    print("-" * 68)
    print("📂 پوزیشن‌های باز در پایان بک‌تست")
    print("-" * 68)

    if not active_positions:
        print("   هیچ پوزیشن بازی وجود ندارد.")
        return

    for symbol, pos in sorted(active_positions.items()):
        notional = pos["used_margin"] * LEVERAGE
        print(
            f"   {symbol:<8} {pos['side']:<5} | "
            f"Entry=${pos['entry_price']:,.6f} | "
            f"SL=${pos['stop_loss']:,.6f} | "
            f"Margin=${pos['used_margin']:,.2f} | "
            f"Notional=${notional:,.2f}"
        )


def summarize_result(trades_df, equity_df, final_state):
    print()
    print("=" * 68)
    print("📊 گزارش جامع عملکرد - HUNTER-V83.6")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        if not equity_df.empty:
            final_equity = float(equity_df.iloc[-1]["Equity"])
            print(f"💵 Equity نهایی: ${final_equity:,.2f}")
        print_open_positions(final_state.get("active_positions", {}))
        return

    trades_df = trades_df.sort_values(["Timestamp", "ExitOrder"], kind="stable").reset_index(drop=True)

    total_trades = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    wr = (wins / total_trades) * 100

    net_r = float(trades_df["Return"].sum())
    total_dollar_pnl = float(trades_df["Dollar_PnL"].sum())
    final_realized_capital = INITIAL_CAPITAL + total_dollar_pnl

    closed_equity = INITIAL_CAPITAL + trades_df["Dollar_PnL"].cumsum()
    closed_dd = calculate_max_drawdown(closed_equity)

    if not equity_df.empty:
        portfolio_dd = calculate_max_drawdown(equity_df["Equity"])
    else:
        portfolio_dd = {"max_dd_dollar": 0.0, "max_dd_pct": 0.0}

    longs_df = trades_df[trades_df["Side"] == "LONG"]
    long_total = len(longs_df)
    long_wins = int((longs_df["Outcome"] == "WIN").sum())
    long_losses = int((longs_df["Outcome"] == "LOSS").sum())
    long_wr = (long_wins / long_total) * 100 if long_total > 0 else 0
    long_pnl = float(longs_df["Dollar_PnL"].sum()) if long_total > 0 else 0.0

    shorts_df = trades_df[trades_df["Side"] == "SHORT"]
    short_total = len(shorts_df)
    short_wins = int((shorts_df["Outcome"] == "WIN").sum())
    short_losses = int((shorts_df["Outcome"] == "LOSS").sum())
    short_wr = (short_wins / short_total) * 100 if short_total > 0 else 0
    short_pnl = float(shorts_df["Dollar_PnL"].sum()) if short_total > 0 else 0.0

    max_losses = calculate_max_loss_streak(trades_df)
    cluster_stats = calculate_loss_cluster_stats(trades_df)

    final_unrealized = float(final_state.get("unrealized_pnl", 0.0))
    final_mtm_equity = float(final_state.get("final_mtm_equity", final_realized_capital))

    avg_pnl = total_dollar_pnl / total_trades
    avg_r = net_r / total_trades

    print(f"🔸 سرمایه اولیه: ${INITIAL_CAPITAL:,.2f}")
    print(f"🔸 Margin پایه هر معامله: {BASE_MARGIN_PCT * 100:.2f}% Equity")
    print(f"🔸 سقف کل Margin: {MAX_TOTAL_MARGIN_PCT * 100:.2f}% Equity")
    print(f"🔸 Leverage: {LEVERAGE:.1f}x")
    print(f"🔸 تعداد کل معاملات: {total_trades}")
    print()
    print("📈 LONG")
    print(f"   کل = {long_total} | برد = {long_wins} | باخت = {long_losses} | Win Rate = {long_wr:.2f}% | PnL = ${long_pnl:,.2f}")
    print()
    print("📉 SHORT")
    print(f"   کل = {short_total} | برد = {short_wins} | باخت = {short_losses} | Win Rate = {short_wr:.2f}% | PnL = ${short_pnl:,.2f}")
    print()
    print(f"🎯 Win Rate کلی: {wr:.2f}%")
    print(f"💰 مجموع بازدهی خالص: {net_r:.2f}R")
    print(f"📊 میانگین بازده هر معامله: {avg_r:.4f}R")
    print(f"💵 مجموع سود/زیان Realized: ${total_dollar_pnl:,.2f}")
    print(f"💵 میانگین PnL هر معامله: ${avg_pnl:,.2f}")
    print()
    print(f"🏦 Final Realized Capital: ${final_realized_capital:,.2f}")
    print(f"📌 Unrealized PnL پایان تست: ${final_unrealized:,.2f}")
    print(f"🏦 Final Mark-to-Market Equity: ${final_mtm_equity:,.2f}")
    print()
    print("📉 DRAWDOWN")
    print(f"   Closed-Trade Max DD: {closed_dd['max_dd_pct']:.2f}% (${closed_dd['max_dd_dollar']:,.2f})")
    print(f"   Portfolio MTM Max DD: {portfolio_dd['max_dd_pct']:.2f}% (${portfolio_dd['max_dd_dollar']:,.2f})")
    print()
    print("❄️ LOSS STATISTICS")
    print(f"   Max Loss Streak: {max_losses}")
    print(f"   Max Losses Same Timestamp: {cluster_stats['max_losses_same_timestamp']}")
    print(f"   Losses Inside Clusters: {cluster_stats['losses_in_clusters']}")
    print(f"   Clustered Loss Percentage: {cluster_stats['loss_cluster_pct']:.2f}%")

    print_open_positions(final_state.get("active_positions", {}))

    if not equity_df.empty:
        last_row = equity_df.iloc[-1]
        print()
        print("-" * 68)
        print("📋 وضعیت پورتفولیو در آخرین کندل")
        print("-" * 68)
        print(f"   Active Positions: {int(last_row['Active_Positions'])}")
        print(f"   Active Margin: ${last_row['Active_Margin']:,.2f}")
        print(f"   Margin Utilization: {last_row['Margin_Utilization_Pct']:.2f}%")
        print(f"   BTC Bull Regime: {last_row['BTC_Bull']}")
        print(f"   Market Breadth: {last_row['Breadth'] * 100:.2f}%")

    print()
    print("=" * 68)
    print("✅ گزارش HUNTER-V83.6 پایان یافت.")
    print("=" * 68)


if __name__ == "__main__":
    df_trades, df_equity, final_state = run_backtest(processed_data)
    summarize_result(trades_df=df_trades, equity_df=df_equity, final_state=final_state)
    print()
    print("✨ اجرای HUNTER-V83.6 به اتمام رسید.")
