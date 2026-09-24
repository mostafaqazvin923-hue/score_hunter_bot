# HUNTER-V3 STAGE-1 — COMPRESSION / BREAKOUT / VOLUME
# LBank USDT-M FUTURES / SWAP DATA
#
# DATA SOURCE CONTRACT:
#   LBank SwapU markets via CCXT's LBank contract market type.
#   Symbols use CCXT unified futures form: BTC/USDT:USDT.
#   CCXT is pinned to a known current release in requirements.txt.
#
# The strategy engine below is unchanged in concept from V3 Stage-1:
#   1) 1H compression
#   2) 15M breakout of completed compression box
#   3) 15M relative-volume confirmation
# RR = 1:2, no BE/trailing/timeout, no overlap, max 3 portfolio positions.

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
# CONFIG
# ============================================================

EXCHANGE = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {"defaultType": "swap"},
})

SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "SUI": "SUI/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "NEAR": "NEAR/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "APT": "APT/USDT:USDT",
    "CRV": "CRV/USDT:USDT",
    "ONDO": "ONDO/USDT:USDT",
    "PENDLE": "PENDLE/USDT:USDT",
    "ICP": "ICP/USDT:USDT",
    "WIF": "WIF/USDT:USDT",
}

DAYS = 365
WARMUP_DAYS = 35
TIMEFRAME = "15m"
TIMEFRAME_MS = 15 * 60 * 1000
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
RR = 2.0
MAX_POSITIONS = 3

CLUSTERS = {
    "MAJOR": {"BTC", "ETH"},
    "L1": {"SOL", "SUI", "AVAX", "NEAR", "ADA", "BNB", "APT"},
    "DEFI": {"CRV", "ONDO", "PENDLE"},
    "OTHER": {"ICP"},
    "MEME": {"WIF"},
}

# ============================================================
# DATA LOADER — copied in structure from the known-working V46 path
# ============================================================

START_DATE = datetime.now() - timedelta(days=DAYS + WARMUP_DAYS)
SINCE_TIMESTAMP = int(START_DATE.timestamp() * 1000)


def fetch_symbol_data(lbank_symbol):
    """Fetch paginated LBank OHLCV using the same CCXT path as V46."""
    all_ohlcv = []
    current_since = SINCE_TIMESTAMP
    last_seen = None

    while current_since < EXCHANGE.milliseconds():
        batch = None

        for attempt in range(3):
            try:
                batch = EXCHANGE.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000,
                    params={"type": "swap"},
                )
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"⚠️ دریافت ناقص {lbank_symbol}: {exc}")
                    return None

        if not batch:
            break

        first_ts = int(batch[0][0])
        last_ts = int(batch[-1][0])

        if last_seen is not None and last_ts <= last_seen:
            print(f"⚠️ pagination متوقف شد: {lbank_symbol}")
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts

        # Advance by exactly one candle, matching the V46 pagination idea.
        current_since = last_ts + TIMEFRAME_MS

        if len(batch) < 1000:
            break

        # Defensive protection against malformed API timestamps.
        if last_ts < first_ts:
            print(f"⚠️ timestamp نامعتبر در {lbank_symbol}")
            return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df[["Timestamp", "Date", "Open", "High", "Low", "Close", "Volume"]]

    df.dropna(subset=["Date", *numeric_cols], inplace=True)
    df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0) & (df["Close"] > 0)]
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove the currently forming 15m candle.
    now_ms = EXCHANGE.milliseconds()
    if len(df) >= 1:
        last_ms = int(df.iloc[-1]["Timestamp"])
        if last_ms + TIMEFRAME_MS > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < 600:
        return None

    # Gap audit. Small timestamp irregularities are tolerated; a real missing
    # candle is rejected because it can distort compression and breakout logic.
    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(minutes=15, seconds=30):
        print(f"⚠️ gap بزرگ در {lbank_symbol}; نماد حذف شد.")
        return None

    return df.reset_index(drop=True)


def validate_market_universe():
    """Confirm every requested symbol exists before downloading history."""
    EXCHANGE.load_markets()
    valid = []
    invalid = []

    for name, symbol in SYMBOLS.items():
        try:
            market = EXCHANGE.market(symbol)
            if market.get("swap") is not True or market.get("contract") is not True:
                invalid.append((name, symbol, "not a LBank USDT-M swap market"))
            else:
                valid.append((name, symbol))
        except Exception as exc:
            invalid.append((name, symbol, str(exc)))

    print(f"CCXT version: {ccxt.__version__}")
    print(f"Valid LBank USDT-M swap markets: {len(valid)} / {len(SYMBOLS)}")
    if invalid:
        for item in invalid:
            print(f"⚠️ invalid market: {item}")

    return valid


def load_all_data():
    valid = validate_market_universe()
    processed = {}

    print("=" * 82)
    print("📥 LBank USDT-M FUTURES OHLCV DATA LOAD — V46 PAGINATION / 15m")
    print("=" * 82)

    for name, symbol in valid:
        print(f"دریافت و آماده‌سازی: {name} ...", flush=True)
        df = fetch_symbol_data(symbol)
        if df is None:
            print(f"❌ {name}: حذف شد")
            continue

        processed[name] = prepare(df)
        if processed[name] is None:
            print(f"❌ {name}: prepare failed")
            processed.pop(name, None)
        else:
            print(f"✅ {name}: {len(processed[name])} کندل 15m")

    print(f"\n✅ تعداد نمادهای معتبر: {len(processed)} از {len(SYMBOLS)}")
    return processed


# ============================================================
# STRATEGY PREPARATION
# ============================================================


def prepare(df):
    df = df.copy()

    # At this point fetch_symbol_data already removed the current candle.
    if len(df) < 600:
        return None

    h = (
        df.set_index("Date")
        .resample("1h", label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    # Compression is calculated only from completed 1H bars.
    h["atr14"] = (h["High"] - h["Low"]).rolling(14).mean()
    h["range20"] = h["High"].rolling(20).max() - h["Low"].rolling(20).min()
    h["atr20"] = (h["High"] - h["Low"]).rolling(20).mean()

    h["compressed"] = (
        ((h["High"] - h["Low"]) < 0.75 * h["atr20"])
        & (h["range20"] < 4.0 * h["atr14"])
    )

    h["box_high"] = h["High"].rolling(4).max().shift(1)
    h["box_low"] = h["Low"].rolling(4).min().shift(1)
    h["comp_recent"] = (
        h["compressed"].rolling(2).max().shift(1).fillna(0).astype(bool)
    )

    # Map only completed hourly context to 15m candles.
    x = df.copy().set_index("Date")
    h2 = h.set_index("Date")[["box_high", "box_low", "comp_recent"]]
    x = x.join(h2.reindex(x.index, method="ffill"))

    # The hourly row at HH:00 is not complete until HH:59:59. Shift four
    # 15m rows so the strategy can only use the previous completed hour.
    x["box_high"] = x["box_high"].shift(4)
    x["box_low"] = x["box_low"].shift(4)
    x["comp_recent"] = x["comp_recent"].shift(4)

    x["range"] = x["High"] - x["Low"]
    x["body"] = (x["Close"] - x["Open"]).abs()
    x["range20_mean"] = x["range"].rolling(20).mean()
    vol_mean = x["Volume"].rolling(20).mean().replace(0, np.nan)
    x["rvol20"] = x["Volume"] / vol_mean

    return x.reset_index()


def cluster_of(symbol):
    for cluster, members in CLUSTERS.items():
        if symbol in members:
            return cluster
    return symbol


# ============================================================
# TRADE ENGINE
# ============================================================


def pnl_for_trade(entry, risk, outcome):
    notional = TRADE_MARGIN * LEVERAGE
    gross = notional * RR * risk / entry if outcome == "WIN" else -notional * risk / entry
    fees = notional * FEE_RATE * 2
    return gross - fees


def run_symbol(symbol, df):
    trades = []
    i = 100
    n = len(df)

    while i < n - 2:
        r = df.iloc[i]

        if not bool(r.get("comp_recent", False)):
            i += 1
            continue

        box_high = r.get("box_high")
        box_low = r.get("box_low")
        if pd.isna(box_high) or pd.isna(box_low) or box_high <= box_low:
            i += 1
            continue

        range20_mean = r.get("range20_mean")
        rvol = r.get("rvol20")
        if pd.isna(range20_mean) or pd.isna(rvol):
            i += 1
            continue

        side = None
        expansion = r["range"] > 1.10 * range20_mean
        volume_ok = rvol >= 1.20

        if (
            r["Close"] > box_high
            and r["Open"] <= box_high
            and volume_ok
            and expansion
        ):
            side = "LONG"
        elif (
            r["Close"] < box_low
            and r["Open"] >= box_low
            and volume_ok
            and expansion
        ):
            side = "SHORT"

        if side is None:
            i += 1
            continue

        # Signal candle is closed. Entry is next 15m candle open.
        j = i + 1
        if j >= n:
            break

        entry0 = float(df.iloc[j]["Open"])
        entry = entry0 * (1 + SLIPPAGE if side == "LONG" else 1 - SLIPPAGE)

        if side == "LONG":
            sl = float(box_low)
            risk = entry - sl
            if risk <= 0:
                i += 1
                continue
            tp = entry + RR * risk
        else:
            sl = float(box_high)
            risk = sl - entry
            if risk <= 0:
                i += 1
                continue
            tp = entry - RR * risk

        outcome = None
        exit_price = None
        exit_time = None
        k = j

        while k < n:
            c = df.iloc[k]
            hit_sl = c["Low"] <= sl if side == "LONG" else c["High"] >= sl
            hit_tp = c["High"] >= tp if side == "LONG" else c["Low"] <= tp

            # Conservative intrabar rule: if both are touched in one candle,
            # SL is assumed to happen first.
            if hit_sl and hit_tp:
                outcome = "LOSS"
                exit_price = sl
                exit_time = c["Date"]
                break
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_time = c["Date"]
                break
            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_time = c["Date"]
                break
            k += 1

        if outcome is not None:
            pnl = pnl_for_trade(entry, risk, outcome)
            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "entry_time": df.iloc[j]["Date"],
                    "exit_time": exit_time,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "exit": exit_price,
                    "risk": risk,
                    "outcome": outcome,
                    "pnl": pnl,
                    "cluster": cluster_of(symbol),
                }
            )

            # No overlapping trades on the same symbol. The next possible
            # signal is strictly after the candle where the result is known.
            i = k + 1
        else:
            # Open position reaches end of available data: do not force-close.
            break

    return trades


def run_portfolio(processed_data):
    candidates = []
    for symbol, df in processed_data.items():
        candidates.extend(run_symbol(symbol, df))

    candidates.sort(key=lambda t: (t["entry_time"], t["symbol"]))

    accepted = []
    active = []

    for trade in candidates:
        entry_time = trade["entry_time"]

        active = [p for p in active if p["exit_time"] > entry_time]

        if len(active) >= MAX_POSITIONS:
            continue

        if any(p["cluster"] == trade["cluster"] for p in active):
            continue

        accepted.append(trade)
        active.append(trade)

    return pd.DataFrame(accepted), len(candidates)


def summarize(trades):
    print("\n" + "=" * 76)
    print("📊 HUNTER-V3 STAGE-1 RESULT")
    print("=" * 76)

    if trades.empty:
        print("Closed Trades: 0")
        print("Win Rate: 0.00%")
        print("Net PnL: $0.00")
        print("Max Consecutive Losses: 0")
        print("⚠️ No closed trades.")
        return

    trades = trades.sort_values(["exit_time", "entry_time"]).reset_index(drop=True)
    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())
    total = len(trades)
    win_rate = 100.0 * wins / total
    net = float(trades["pnl"].sum())

    streak = 0
    max_streak = 0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for outcome, pnl in zip(trades["outcome"], trades["pnl"]):
        if outcome == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    print(f"Candidate Signals: {len(trades)} accepted from portfolio engine")
    print(f"Closed Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Net PnL: ${net:,.2f}")
    print(f"Max Consecutive Losses: {max_streak}")
    print(f"Max Drawdown: ${max_dd:,.2f}")

    print("\nPer-symbol:")
    for symbol, g in trades.groupby("symbol"):
        w = int((g["outcome"] == "WIN").sum())
        wr = 100.0 * w / len(g)
        pnl = float(g["pnl"].sum())
        print(f"  {symbol:<7} {len(g):>4} trades | WR {wr:>6.2f}% | PnL ${pnl:>10.2f}")


def main():
    print("=" * 82)
    print("HUNTER-V3 STAGE-1 — V46 DATA LOADER / COMPRESSION BREAKOUT VOLUME")
    print("=" * 82)
    print("DATA MODE: LBank USDT-M FUTURES / SwapU via CCXT")
    print(f"CCXT version: {ccxt.__version__}")
    print("SYMBOL MODE: BTC/USDT:USDT and equivalent linear swaps")

    processed_data = load_all_data()
    if not processed_data:
        raise RuntimeError("No valid LBank OHLCV datasets were loaded.")

    trades, candidate_count = run_portfolio(processed_data)
    print(f"\nRaw symbol-level closed candidates: {candidate_count}")
    summarize(trades)


if __name__ == "__main__":
    main()
