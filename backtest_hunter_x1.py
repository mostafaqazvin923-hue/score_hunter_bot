
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ==================== CONFIG ====================
BASE = "https://data.binance.vision/data/futures/um"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT",
    "AVAXUSDT", "NEARUSDT", "ADAUSDT", "BNBUSDT",
    "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT",
    "ICPUSDT", "WIFUSDT",
]
BASE_TF = "15m"
TIMEFRAMES = {"15m": "15min", "1h": "1h", "4h": "4h"}

TEST_DAYS = 365
WARMUP_DAYS = 60
INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE
RR = 2.0
FEE = 0.0007
SLIPPAGE = 0.0003

# Fixed hypothesis parameters; no optimization on test results.
ATR_N = 14
EMA_FAST = 50
EMA_SLOW = 200
BREAKOUT_N = 20
SQUEEZE_N = 100
SQUEEZE_Q = 0.25
VOLUME_N = 20
VOLUME_MULT = 1.15
SWING_N = 10
ATR_STOP_MULT = 1.5
MIN_STOP_PCT = 0.001
MAX_STOP_PCT = 0.08

OUT = Path("hunter_x1_results")
OUT.mkdir(exist_ok=True)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HUNTER-X1-backtest/1.0"


def utc_now():
    return datetime.now(timezone.utc)


def fetch_zip(url, retries=3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1 + attempt)
    return None


def parse_zip(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with z.open(names[0]) as f:
            d = pd.read_csv(f, header=None)

    # Binance UM futures kline schema begins with open_time and OHLCV.
    if d.empty or d.shape[1] < 6:
        return pd.DataFrame()

    # Some archive exports include a header row; remove it safely.
    d = d[pd.to_numeric(d.iloc[:, 0], errors="coerce").notna()].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.iloc[:, :6]
    d.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    ts = pd.to_numeric(d["timestamp"], errors="coerce")
    # UM archive timestamps are normally milliseconds; tolerate microseconds.
    if ts.dropna().median() > 1e14:
        ts = ts / 1000
    d["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().set_index("timestamp").sort_index()
    return d[~d.index.duplicated(keep="last")]


def download_symbol(symbol, start, end):
    """Monthly archives plus daily fallback when a monthly archive is absent."""
    chunks = []
    month = datetime(start.year, start.month, 1, tzinfo=timezone.utc)

    while month < end:
        next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        tag = month.strftime("%Y-%m")
        url = (
            f"{BASE}/monthly/klines/{symbol}/{BASE_TF}/"
            f"{symbol}-{BASE_TF}-{tag}.zip"
        )
        payload = fetch_zip(url)

        if payload:
            frame = parse_zip(payload)
            if not frame.empty:
                chunks.append(frame)
        else:
            day = month
            while day < next_month and day < end:
                if day.date() >= start.date():
                    date_tag = day.strftime("%Y-%m-%d")
                    daily_url = (
                        f"{BASE}/daily/klines/{symbol}/{BASE_TF}/"
                        f"{symbol}-{BASE_TF}-{date_tag}.zip"
                    )
                    dp = fetch_zip(daily_url)
                    if dp:
                        frame = parse_zip(dp)
                        if not frame.empty:
                            chunks.append(frame)
                day += timedelta(days=1)
        month = next_month

    if not chunks:
        return pd.DataFrame()

    d = pd.concat(chunks).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d[(d.index >= pd.Timestamp(start)) & (d.index < pd.Timestamp(end))]
    # Exclude any candle that has not fully closed yet.
    cutoff = pd.Timestamp(utc_now()) - pd.Timedelta(minutes=15)
    return d[d.index <= cutoff]


def resample_ohlcv(d, rule):
    expected = {"15min": 1, "1h": 4, "4h": 16}[rule]
    x = d.resample(rule, origin="epoch", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n=("close", "count"),
    )
    # Missing source bars invalidate the aggregated candle.
    x = x[x["n"] == expected].drop(columns="n")
    return x.dropna()


def indicators(d):
    x = d.copy()
    prev = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1 / ATR_N, adjust=False, min_periods=ATR_N
    ).mean()
    x["ema_fast"] = x["close"].ewm(
        span=EMA_FAST, adjust=False, min_periods=EMA_FAST
    ).mean()
    x["ema_slow"] = x["close"].ewm(
        span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW
    ).mean()

    mid = x["close"].rolling(20, min_periods=20).mean()
    sd = x["close"].rolling(20, min_periods=20).std(ddof=0)
    x["bb_width"] = (4 * sd / mid).replace([np.inf, -np.inf], np.nan)

    # Threshold is based only on earlier candle widths.
    pw = x["bb_width"].shift(1)
    threshold = pw.rolling(SQUEEZE_N, min_periods=SQUEEZE_N).quantile(SQUEEZE_Q)
    x["squeeze"] = (pw <= threshold).rolling(5, min_periods=5).max().eq(1)

    x["prior_high"] = x["high"].shift(1).rolling(
        BREAKOUT_N, min_periods=BREAKOUT_N
    ).max()
    x["prior_low"] = x["low"].shift(1).rolling(
        BREAKOUT_N, min_periods=BREAKOUT_N
    ).min()
    x["vol_median"] = x["volume"].shift(1).rolling(
        VOLUME_N, min_periods=VOLUME_N
    ).median()
    x["swing_low"] = x["low"].shift(1).rolling(
        SWING_N, min_periods=SWING_N
    ).min()
    x["swing_high"] = x["high"].shift(1).rolling(
        SWING_N, min_periods=SWING_N
    ).max()

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["close_location"] = (x["close"] - x["low"]) / candle_range
    return x


def signal(r):
    required = [
        "atr", "ema_fast", "ema_slow", "prior_high", "prior_low",
        "vol_median", "swing_low", "swing_high", "close_location",
    ]
    if any(pd.isna(r[k]) for k in required):
        return None
    if not bool(r["squeeze"]):
        return None
    if r["volume"] < r["vol_median"] * VOLUME_MULT:
        return None

    if r["close"] > r["prior_high"] and r["close_location"] >= 0.70:
        if r["ema_fast"] > r["ema_slow"]:
            return "LONG"

    if r["close"] < r["prior_low"] and r["close_location"] <= 0.30:
        if r["ema_fast"] < r["ema_slow"]:
            return "SHORT"
    return None


def simulate(tf, frames, test_start, test_end):
    results = []

    for symbol, raw in frames.items():
        if raw.empty:
            continue
        d = indicators(raw)
        ix = d.index
        i = 0

        while i < len(d) - 1:
            ts = ix[i]
            if ts < test_start or ts >= test_end:
                i += 1
                continue

            side = signal(d.iloc[i])
            if side is None:
                i += 1
                continue

            # Signal candle is closed; entry is at the next candle open.
            entry_i = i + 1
            entry_open = float(d["open"].iloc[entry_i])
            entry = entry_open * (1 + SLIPPAGE if side == "LONG"
                                  else 1 - SLIPPAGE)
            r = d.iloc[i]
            atr = float(r["atr"])

            if side == "LONG":
                stop = min(float(r["swing_low"]), entry - ATR_STOP_MULT * atr)
                risk = entry - stop
                target = entry + RR * risk
            else:
                stop = max(float(r["swing_high"]), entry + ATR_STOP_MULT * atr)
                risk = stop - entry
                target = entry - RR * risk

            if risk <= 0 or not MIN_STOP_PCT <= risk / entry <= MAX_STOP_PCT:
                i += 1
                continue

            qty = NOTIONAL / entry
            exit_i, exit_raw, reason = None, None, None

            # No timeout. Scan forward until SL/TP or end of available data.
            for j in range(entry_i, len(d)):
                bar = d.iloc[j]
                if side == "LONG":
                    hit_sl = bar["low"] <= stop
                    hit_tp = bar["high"] >= target
                else:
                    hit_sl = bar["high"] >= stop
                    hit_tp = bar["low"] <= target

                if hit_sl:  # conservative if both levels touch same candle
                    exit_i, exit_raw = j, stop
                    reason = "SL_BOTH" if hit_tp else "SL"
                    break
                if hit_tp:
                    exit_i, exit_raw, reason = j, target, "TP"
                    break

            if exit_i is None:
                # Mark at last available close, explicitly reported as data-end.
                exit_i = len(d) - 1
                exit_raw = float(d["close"].iloc[exit_i])
                reason = "DATA_END_MARK"

            # Don't allow entries outside the test period.
            if ix[entry_i] >= test_end:
                break

            exit_price = float(exit_raw) * (
                1 - SLIPPAGE if side == "LONG" else 1 + SLIPPAGE
            )
            direction = 1 if side == "LONG" else -1
            gross = direction * (exit_price - entry) * qty
            fees = FEE * qty * (entry + exit_price)
            net = gross - fees
            results.append({
                "timeframe": tf, "symbol": symbol, "side": side,
                "signal_time": ts.isoformat(),
                "entry_time": ix[entry_i].isoformat(),
                "exit_time": ix[exit_i].isoformat(),
                "entry": entry, "stop": stop, "target": target,
                "exit_price": exit_price, "exit_reason": reason,
                "qty": qty, "gross_pnl": gross, "fees": fees,
                "net_pnl": net, "bars_held": exit_i - entry_i + 1,
            })

            # The symbol remains locked until the exit bar; next scan starts
            # after that bar, preventing same-candle re-entry.
            i = exit_i + 1

    return results


def summarize(tf, trades):
    if not trades:
        return {
            "timeframe": tf, "trades": 0, "wins": 0, "losses": 0,
            "win_rate_pct": 0.0, "net_pnl": 0.0,
            "final_equity": INITIAL_CAPITAL, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "max_loss_streak": 0,
            "fees_total": 0.0, "data_end_marks": 0,
        }

    t = pd.DataFrame(trades).sort_values(["exit_time", "symbol"])
    pnl = t["net_pnl"]
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = gp / gl if gl else (float("inf") if gp else 0.0)

    equity = INITIAL_CAPITAL + pnl.cumsum()
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    dd = (peak - equity) / peak.replace(0, np.nan)

    streak = max_streak = 0
    for value in pnl:
        if value <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "timeframe": tf,
        "trades": len(t),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": 100 * wins / len(t),
        "net_pnl": float(pnl.sum()),
        "final_equity": float(INITIAL_CAPITAL + pnl.sum()),
        "profit_factor": pf,
        "max_drawdown_pct": float(dd.max() * 100),
        "max_loss_streak": max_streak,
        "fees_total": float(t["fees"].sum()),
        "data_end_marks": int((t["exit_reason"] == "DATA_END_MARK").sum()),
    }


def main():
    end = utc_now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=TEST_DAYS + WARMUP_DAYS)
    test_start = pd.Timestamp(end - timedelta(days=TEST_DAYS))
    test_end = pd.Timestamp(end)

    print(f"Test period: {test_start} -> {test_end}")
    print(f"Capital=${INITIAL_CAPITAL:.2f}; margin=${MARGIN:.2f}; "
          f"leverage={LEVERAGE:g}x; notional=${NOTIONAL:.2f}; RR=1:{RR:g}")

    base_frames = {}
    for n, symbol in enumerate(SYMBOLS, 1):
        try:
            d = download_symbol(symbol, start, end)
            if d.empty:
                print(f"[{n}/{len(SYMBOLS)}] {symbol}: NO DATA")
                continue
            base_frames[symbol] = d
            print(f"[{n}/{len(SYMBOLS)}] {symbol}: {len(d):,} candles")
        except Exception as exc:
            print(f"[{n}/{len(SYMBOLS)}] {symbol}: ERROR {exc}")

    if not base_frames:
        raise RuntimeError("No real market data downloaded; no synthetic fallback.")

    summaries = []
    for tf, rule in TIMEFRAMES.items():
        frames = {s: resample_ohlcv(d, rule) for s, d in base_frames.items()}
        frames = {
            s: d for s, d in frames.items()
            if len(d) >= EMA_SLOW + SQUEEZE_N + 50
        }
        trades = simulate(tf, frames, test_start, test_end)
        tdf = pd.DataFrame(trades)
        tdf.to_csv(OUT / f"trades_{tf}.csv", index=False)
        summary = summarize(tf, trades)
        summaries.append(summary)
        print(f"\n=== {tf} ===")
        print(pd.Series(summary).to_string())

    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved results in: {OUT.resolve()}")
    print("This is a hypothesis test, not evidence of future profitability.")


if __name__ == "__main__":
    main()
