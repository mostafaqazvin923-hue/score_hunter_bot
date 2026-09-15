# ============================================================
# HUNTER-V76 — UNIFIED BACKTEST SCRIPT
# ============================================================

import os
import sys
import time
import math
import warnings
import subprocess
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Auto install dependencies
# ------------------------------------------------------------

try:
    import ccxt
except ImportError:
    print("📦 Installing ccxt...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "ccxt", "--quiet"
    ])
    import ccxt

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("📦 Installing pandas/numpy...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pandas", "numpy", "--quiet"
    ])
    import pandas as pd
    import numpy as np


# ============================================================
# CONFIG
# ============================================================

VERSION = "HUNTER-V76"

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

# ------------------------------------------------------------
# Trading costs
# ------------------------------------------------------------

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

# ------------------------------------------------------------
# ATR / exits
# ------------------------------------------------------------

ATR_PERIOD = 14

TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8

TIMEOUT_CANDLES = 45

# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------

EMA_WARMUP = 200

# ------------------------------------------------------------
# MONEY MANAGEMENT
# ------------------------------------------------------------

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0


# ============================================================
# UNIVERSE — 20 COINS
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "BNB": "BNB/USDT",
    "XRP": "XRP/USDT",
    "SOL": "SOL/USDT",
    "DOGE": "DOGE/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
    "TRX": "TRX/USDT",
    "HYPE": "HYPE/USDT",
    "ZEC": "ZEC/USDT",
    "AVAX": "AVAX/USDT",
    "SUI": "SUI/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "ATOM": "ATOM/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "RENDER": "RENDER/USDT",
    "ONDO": "ONDO/USDT",
}


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000,
})


# ============================================================
# GLOBAL DATA CONTAINERS
# ============================================================

DATA = {}
FAILED_SYMBOLS = []


# ============================================================
# FETCH OHLCV
# ============================================================

def fetch_ohlcv_full(exchange, symbol, timeframe, since_ms):
    all_rows = []
    current_since = since_ms
    max_retries = 3
    page_limit = 1000

    while True:
        success = False
        for attempt in range(max_retries):
            try:
                rows = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=current_since, limit=page_limit
                )
                success = True
                break
            except Exception as e:
                print(f"⚠️ {symbol} fetch attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)

        if not success:
            raise RuntimeError(f"Failed to fetch {symbol}")

        if not rows:
            break

        all_rows.extend(rows)
        last_timestamp = rows[-1][0]

        if last_timestamp <= current_since:
            break

        current_since = last_timestamp + 1
        now_ms = exchange.milliseconds()

        if last_timestamp >= now_ms - (4 * 60 * 60 * 1000):
            break

        if len(all_rows) > 200000:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        raise RuntimeError(f"No OHLCV data returned for {symbol}")

    df = pd.DataFrame(
        all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")

    now = pd.Timestamp.now(tz="UTC")
    candle_duration = pd.Timedelta(hours=4)
    if len(df) > 0:
        last_candle_start = df.index[-1]
        if now < last_candle_start + candle_duration:
            df = df.iloc[:-1]

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()

    df["Mom_Short"] = df["Close"].pct_change(10)
    df["Mom_Long"] = df["Close"].pct_change(30)

    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_EMA20"] = df["EMA20"].shift(1)
    return df


# ============================================================
# LOAD ALL SYMBOLS
# ============================================================

def load_market_data():
    print("=" * 70)
    print(f"🚀 {VERSION}")
    print("=" * 70)
    print(f"Exchange : LBank")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Lookback : {LOOKBACK_DAYS} days")
    print(f"Universe : {len(SYMBOLS)} symbols")
    print(f"Margin   : ${TRADE_MARGIN:.2f}")
    print(f"Leverage : {LEVERAGE:.0f}x")
    print("=" * 70)

    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(since.timestamp() * 1000)

    for name, symbol in SYMBOLS.items():
        print(f"\n📥 [{name}] {symbol}")
        try:
            df = fetch_ohlcv_full(exchange, symbol, TIMEFRAME, since_ms)
            if len(df) < EMA_WARMUP + 50:
                print(f"❌ {name}: not enough candles ({len(df)})")
                FAILED_SYMBOLS.append(name)
                continue
            df = add_indicators(df)
            DATA[name] = df
            print(f"✅ {name}: {len(df)} candles | {df.index[0]} → {df.index[-1]}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            FAILED_SYMBOLS.append(name)

    print("\n" + "=" * 70)
    print(f"VALID SYMBOLS: {len(DATA)}/{len(SYMBOLS)}")
    if FAILED_SYMBOLS:
        print("FAILED:", ", ".join(FAILED_SYMBOLS))
    print("=" * 70)

    if "BTC" not in DATA:
        raise RuntimeError("BTC data is required for HUNTER-V76")

load_market_data()

ALL_TIMESTAMPS = sorted(set().union(*(set(df.index) for df in DATA.values())))


# ============================================================
# BACKTEST STATE
# ============================================================

capital = INITIAL_CAPITAL
open_positions = {}
closed_trades = []
equity_curve = []
peak_equity = INITIAL_CAPITAL
max_drawdown_dollar = 0.0
max_drawdown_pct = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default

def get_row(symbol, timestamp):
    df = DATA.get(symbol)
    if df is None:
        return None
    try:
        row = df.loc[timestamp]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        if len(row) == 0:
            return None
        row = row.iloc[-1]
    return row

def get_btc_regime(timestamp):
    row = get_row("BTC", timestamp)
    if row is None:
        return {"bull": False, "bear": False, "valid": False}
    close = safe_float(row["Close"])
    ema50 = safe_float(row["EMA50"])
    ema200 = safe_float(row["EMA200"])
    if not all(np.isfinite(x) for x in [close, ema50, ema200]):
        return {"bull": False, "bear": False, "valid": False}
    bull = (close > ema200 and ema50 > ema200)
    bear = (close < ema200 and ema50 < ema200)
    return {"bull": bull, "bear": bear, "valid": True, "close": close, "ema50": ema50, "ema200": ema200}

def get_market_breadth(timestamp):
    bullish_count = 0
    active_count = 0
    for symbol, df in DATA.items():
        row = get_row(symbol, timestamp)
        if row is None:
            continue
        close = safe_float(row["Close"])
        ema200 = safe_float(row["EMA200"])
        if not np.isfinite(close) or not np.isfinite(ema200):
            continue
        active_count += 1
        if close > ema200:
            bullish_count += 1
    if active_count == 0:
        return {"bullish_count": 0, "active_count": 0, "ratio": 0.0}
    return {"bullish_count": bullish_count, "active_count": active_count, "ratio": bullish_count / active_count}

def get_direction_permissions(timestamp):
    btc = get_btc_regime(timestamp)
    breadth = get_market_breadth(timestamp)
    if not btc["valid"]:
        return {"allow_longs": False, "allow_shorts": False, "btc_bull": False, "btc_bear": False, "breadth_ratio": breadth["ratio"]}
    breadth_ratio = breadth["ratio"]
    return {
        "allow_longs": (btc["bull"] and breadth_ratio >= 0.35),
        "allow_shorts": (btc["bear"] and breadth_ratio <= 0.65),
        "btc_bull": btc["bull"],
        "btc_bear": btc["bear"],
        "breadth_ratio": breadth_ratio
    }

def get_position_size(entry_price):
    if entry_price <= 0:
        return None
    notional = TRADE_MARGIN * LEVERAGE
    return {"margin": TRADE_MARGIN, "notional": notional, "quantity": notional / entry_price}

def calculate_initial_stop(side, entry_price, atr):
    if entry_price is None or atr is None:
        return None
    entry_price, atr = safe_float(entry_price), safe_float(atr)
    if not np.isfinite(entry_price) or not np.isfinite(atr) or entry_price <= 0 or atr <= 0:
        return None
    distance = atr * INITIAL_ATR_MULTIPLIER
    distance_pct = distance / entry_price
    if not (0.01 <= distance_pct <= 0.04):
        return None
    stop_price = (entry_price - distance) if side == "LONG" else (entry_price + distance)
    if stop_price <= 0:
        return None
    return {"stop_price": stop_price, "stop_distance": distance, "stop_distance_pct": distance_pct}

def update_trailing_stop(position, candle):
    side = position["side"]
    current_stop = position["stop_price"]
    atr = safe_float(candle["ATR"])
    if not np.isfinite(atr) or atr <= 0:
        return current_stop
    if side == "LONG":
        high = safe_float(candle["High"])
        if not np.isfinite(high):
            return current_stop
        return max(current_stop, high - atr * TRAILING_ATR_MULTIPLIER)
    else:
        low = safe_float(candle["Low"])
        if not np.isfinite(low):
            return current_stop
        return min(current_stop, low + atr * TRAILING_ATR_MULTIPLIER)

def determine_stop_exit(position, candle):
    side, stop_price = position["side"], position["stop_price"]
    high, low = safe_float(candle["High"]), safe_float(candle["Low"])
    if not np.isfinite(high) or not np.isfinite(low):
        return None, None
    if side == "LONG" and low <= stop_price:
        return stop_price, "STOP"
    if side == "SHORT" and high >= stop_price:
        return stop_price, "STOP"
    return None, None

def calculate_trade_pnl(position, exit_price):
    entry_price, notional, side = position["entry_price"], position["notional"], position["side"]
    if entry_price <= 0 or exit_price <= 0 or notional <= 0:
        return 0.0, 0.0
    raw_return = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
    gross_pnl = raw_return * notional
    total_fee = (notional * FEE_RATE) + (abs(notional * (exit_price / entry_price)) * FEE_RATE)
    net_pnl = gross_pnl - total_fee
    stop_pct = position["initial_stop_distance_pct"]
    net_r = (raw_return / stop_pct - (total_fee / notional) / stop_pct) if (stop_pct and stop_pct > 0) else np.nan
    return net_pnl, net_r

def close_position(symbol, timestamp, exit_price, reason):
    global capital
    if symbol not in open_positions:
        return None
    position = open_positions[symbol]
    exit_price = safe_float(exit_price)
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None
    pnl, net_r = calculate_trade_pnl(position, exit_price)
    capital_before = capital
    capital += pnl
    trade = {
        "symbol": symbol, "side": position["side"],
        "entry_timestamp": position["entry_timestamp"], "exit_timestamp": timestamp,
        "entry_price": position["entry_price"], "exit_price": exit_price,
        "margin": position["margin"], "notional": position["notional"], "quantity": position["quantity"],
        "initial_stop": position["initial_stop"], "initial_stop_distance_pct": position["initial_stop_distance_pct"],
        "pnl": pnl, "R": net_r, "result": "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE"),
        "exit_reason": reason, "capital_before": capital_before, "capital_after": capital,
        "bars_held": position.get("bars_held", 0)
    }
    closed_trades.append(trade)
    del open_positions[symbol]
    return trade

def manage_open_positions(timestamp):
    for symbol in list(open_positions.keys()):
        position = open_positions[symbol]
        candle = get_row(symbol, timestamp)
        if candle is None:
            continue
        position["bars_held"] = position.get("bars_held", 0) + 1
        exit_price, reason = determine_stop_exit(position, candle)
        if exit_price is not None:
            close_position(symbol, timestamp, exit_price, reason)
            continue
        if (timestamp - position["entry_timestamp"]).total_seconds() / (4 * 3600) >= TIMEOUT_CANDLES:
            exit_price = safe_float(candle["Close"])
            if exit_price is not None:
                close_position(symbol, timestamp, exit_price, "TIMEOUT")
                continue
        position["stop_price"] = update_trailing_stop(position, candle)

def update_equity(timestamp):
    global peak_equity, max_drawdown_dollar, max_drawdown_pct
    equity = capital
    if equity > peak_equity:
        peak_equity = equity
    drawdown = equity - peak_equity
    if drawdown < max_drawdown_dollar:
        max_drawdown_dollar = drawdown
    dd_pct = (drawdown / peak_equity * 100) if peak_equity > 0 else 0.0
    if dd_pct < max_drawdown_pct:
        max_drawdown_pct = dd_pct
    equity_curve.append({"timestamp": timestamp, "capital": capital, "equity": equity, "drawdown": drawdown, "drawdown_pct": dd_pct})

def has_open_position(symbol):
    return symbol in open_positions

def create_position(symbol, side, timestamp, entry_price, atr):
    if has_open_position(symbol):
        return None
    sizing = get_position_size(entry_price)
    if sizing is None:
        return None
    stop_data = calculate_initial_stop(side, entry_price, atr)
    if stop_data is None:
        return None
    position = {
        "symbol": symbol, "side": side, "entry_timestamp": timestamp,
        "entry_price": float(entry_price), "margin": float(sizing["margin"]),
        "notional": float(sizing["notional"]), "quantity": float(sizing["quantity"]),
        "initial_stop": float(stop_data["stop_price"]), "stop_price": float(stop_data["stop_price"]),
        "initial_stop_distance": float(stop_data["stop_distance"]),
        "initial_stop_distance_pct": float(stop_data["stop_distance_pct"]), "bars_held": 0
    }
    open_positions[symbol] = position
    return position


# ============================================================
# SIGNAL ENGINE
# ============================================================

LONG_MOM_SHORT_MIN = 0.012
LONG_MOM_LONG_MIN = 0.035
SHORT_MOM_SHORT_MAX = -0.012
SHORT_MOM_LONG_MAX = -0.035
LONG_PULLBACK_MAX = 1.015
SHORT_PULLBACK_MIN = 0.985

def get_signal(symbol, timestamp):
    row = get_row(symbol, timestamp)
    if row is None:
        return None
    req = ["Close", "EMA20", "EMA50", "EMA200", "Prev_Low", "Prev_High", "Prev_EMA20", "Mom_Short", "Mom_Long", "ATR"]
    val = {c: safe_float(row[c]) for c in req}
    if not all(np.isfinite(v) for v in val.values()):
        return None

    if val["Close"] > val["EMA20"] > val["EMA50"] and val["Close"] > val["EMA200"] and val["Prev_Low"] <= val["Prev_EMA20"] * LONG_PULLBACK_MAX and val["Mom_Short"] > LONG_MOM_SHORT_MIN and val["Mom_Long"] > LONG_MOM_LONG_MIN:
        return "LONG"
    if val["Close"] < val["EMA20"] < val["EMA50"] and val["Close"] < val["EMA200"] and val["Prev_High"] >= val["Prev_EMA20"] * SHORT_PULLBACK_MIN and val["Mom_Short"] < SHORT_MOM_SHORT_MAX and val["Mom_Long"] < SHORT_MOM_LONG_MAX:
        return "SHORT"
    return None

def get_signal_score(symbol, timestamp, side):
    row = get_row(symbol, timestamp)
    if row is None:
        return None
    mom_long = safe_float(row["Mom_Long"])
    if not np.isfinite(mom_long):
        return None
    return float(mom_long if side == "LONG" else -mom_long)

def collect_candidates(timestamp, permissions):
    long_c, short_c = [], []
    if permissions["allow_longs"]:
        for symbol in DATA.keys():
            if not has_open_position(symbol) and get_signal(symbol, timestamp) == "LONG":
                score = get_signal_score(symbol, timestamp, "LONG")
                if score is not None:
                    long_c.append({"symbol": symbol, "side": "LONG", "score": score})
    if permissions["allow_shorts"]:
        for symbol in DATA.keys():
            if not has_open_position(symbol) and get_signal(symbol, timestamp) == "SHORT":
                score = get_signal_score(symbol, timestamp, "SHORT")
                if score is not None:
                    short_c.append({"symbol": symbol, "side": "SHORT", "score": score})
    long_c.sort(key=lambda x: x["score"], reverse=True)
    short_c.sort(key=lambda x: x["score"], reverse=True)
    return long_c, short_c

def get_entry_price(row, side):
    op = safe_float(row["Open"])
    if not np.isfinite(op) or op <= 0:
        return None
    return float(op * (1.0 + SLIPPAGE) if side == "LONG" else op * (1.0 - SLIPPAGE))

def execute_entry(candidate, timestamp):
    if has_open_position(candidate["symbol"]) or len(open_positions) >= MAX_POSITIONS:
        return None
    row = get_row(candidate["symbol"], timestamp)
    if row is None:
        return None
    entry_price = get_entry_price(row, candidate["side"])
    atr = safe_float(row["ATR"])
    if entry_price is None or not np.isfinite(atr) or atr <= 0:
        return None
    return create_position(candidate["symbol"], candidate["side"], timestamp, entry_price, atr)


# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

def run_backtest():
    global capital
    print("\n" + "=" * 70 + "\n🚀 STARTING HUNTER-V76 BACKTEST\n" + "=" * 70)
    total_timestamps = len(ALL_TIMESTAMPS)
    processed = 0

    for timestamp in ALL_TIMESTAMPS:
        processed += 1
        btc_row = get_row("BTC", timestamp)
        if btc_row is None or not np.isfinite(safe_float(btc_row["EMA200"])):
            continue

        manage_open_positions(timestamp)
        permissions = get_direction_permissions(timestamp)
        long_c, short_c = collect_candidates(timestamp, permissions)

        available_slots = MAX_POSITIONS - len(open_positions)
        if available_slots > 0:
            candidates = []
            if permissions["allow_longs"]: candidates.extend(long_c)
            if permissions["allow_shorts"]: candidates.extend(short_c)
            candidates.sort(key=lambda x: x["score"], reverse=True)

            for candidate in candidates:
                if len(open_positions) >= MAX_POSITIONS:
                    break
                execute_entry(candidate, timestamp)

        update_equity(timestamp)

        progress = int(processed / total_timestamps * 100)
        if progress % 10 == 0 and processed == int(total_timestamps * (progress / 100)):
            print(f"📈 Progress: {progress}% | Trades={len(closed_trades)} | Open={len(open_positions)} | Capital=${capital:,.2f}")

    if len(open_positions) > 0:
        for symbol in list(open_positions.keys()):
            row = get_row(symbol, ALL_TIMESTAMPS[-1])
            if row is not None and np.isfinite(safe_float(row["Close"])):
                close_position(symbol, ALL_TIMESTAMPS[-1], safe_float(row["Close"]), "END_OF_TEST")

    if len(ALL_TIMESTAMPS) > 0:
        update_equity(ALL_TIMESTAMPS[-1])

    return {"capital": capital, "trades": closed_trades, "equity_curve": equity_curve}

RESULT = run_backtest()
trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()


# ============================================================
# FINAL REPORTING & METRICS
# ============================================================

def calculate_basic_stats(trades):
    if trades is None or len(trades) == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "win_rate": 0.0, "net_pnl": 0.0, "net_r": 0.0, "avg_pnl": 0.0, "avg_r": 0.0}
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(trades["R"], errors="coerce")
    wins, losses, total = int((pnl > 0).sum()), int((pnl < 0).sum()), len(pnl)
    valid_r = r[np.isfinite(r)]
    return {
        "trades": total, "wins": wins, "losses": losses, "breakeven": int((pnl == 0).sum()),
        "win_rate": (wins / total * 100) if total > 0 else 0.0, "net_pnl": float(pnl.sum()),
        "net_r": float(valid_r.sum()) if len(valid_r) > 0 else 0.0, "avg_pnl": float(pnl.mean()),
        "avg_r": float(valid_r.mean()) if len(valid_r) > 0 else 0.0
    }

def calculate_loss_streaks(trades):
    if trades is None or len(trades) == 0:
        return {"max_streak": 0, "average_streak": 0.0, "total_streaks": 0, "streak_counts": {}, "losses_inside_streaks": 0}
    current, streaks = 0, []
    for _, trade in trades.iterrows():
        pnl = safe_float(trade["pnl"])
        if np.isfinite(pnl) and pnl < 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
                current = 0
    if current > 0:
        streaks.append(current)
    if not streaks:
        return {"max_streak": 0, "average_streak": 0.0, "total_streaks": 0, "streak_counts": {}, "losses_inside_streaks": 0}
    counts = {}
    for length in streaks:
        counts[length] = counts.get(length, 0) + 1
    return {"max_streak": max(streaks), "average_streak": sum(streaks) / len(streaks), "total_streaks": len(streaks), "streak_counts": counts, "losses_inside_streaks": sum(streaks)}

def calculate_drawdown_stats(trades):
    if trades is None or len(trades) == 0:
        return {"max_dd_dollar": 0.0, "max_dd_pct": 0.0, "peak_capital": INITIAL_CAPITAL, "trough_capital": INITIAL_CAPITAL}
    cap = pd.to_numeric(trades["capital_after"], errors="coerce").dropna()
    if len(cap) == 0:
        return {"max_dd_dollar": 0.0, "max_dd_pct": 0.0, "peak_capital": INITIAL_CAPITAL, "trough_capital": INITIAL_CAPITAL}
    peak = cap.cummax()
    dd_dl = cap - peak
    dd_pct = dd_dl / peak * 100
    min_idx = dd_dl.idxmin()
    return {"max_dd_dollar": float(dd_dl.min()), "max_dd_pct": float(dd_pct.min()), "peak_capital": float(peak.loc[min_idx]), "trough_capital": float(cap.loc[min_idx])}

def print_final_report():
    stats = calculate_basic_stats(trades_df)
    dd = calculate_drawdown_stats(trades_df)
    streaks = calculate_loss_streaks(trades_df)
    
    print("\n\n" + "=" * 80 + "\n🔥🔥🔥 HUNTER-V76 FINAL REPORT 🔥🔥🔥\n" + "=" * 80)
    print(f"Initial Capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital   : ${capital:,.2f}")
    print(f"Net PnL         : ${stats['net_pnl']:,.2f}")
    print(f"Return          : {((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):.2f}%")
    print(f"Win Rate        : {stats['win_rate']:.2f}% | Trades: {stats['trades']} (Wins: {stats['wins']}, Losses: {stats['losses']})")
    print(f"Max Drawdown    : ${dd['max_dd_dollar']:,.2f} ({dd['max_dd_pct']:.2f}%)")
    print(f"Max Loss Streak : {streaks['max_streak']} | Avg Loss Streak: {streaks['average_streak']:.2f}")
    print("=" * 80)

print_final_report()
