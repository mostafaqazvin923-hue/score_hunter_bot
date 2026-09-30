
#!/usr/bin/env python3
"""
V32 — Futures Aggressive Flow / Absorption — Stage 0
Research venue: Binance USD-M Futures
Execution venue: NOT changed; this is research only.

Protocol:
- Fixed 14-symbol universe.
- 1H Binance USD-M Futures public klines.
- Uses ONLY fields that are actually published in the historical kline archive:
  OHLCV + taker-buy base/quote volume.
- No OI, liquidation, CVD reconstruction, funding, or fabricated data.
- Signal is formed only from a fully closed bar t.
- Entry, if tested, is at next bar open (t+1).
- RR = 1:2, SL = 1 ATR, TP = 2 ATR.
- Same-candle SL+TP = LOSS.
- No timeout / BE / trailing / pyramiding.
- Max one open trade per symbol; different symbols may overlap.
- No same-candle re-entry.
- Censored trades at the end of the sample are excluded.
- Discovery / Development / Validation = 50% / 25% / 25% chronological.
- Thresholds are pre-registered; there is NO threshold mining after results.

Research question:
Does aggressive futures flow alter the forward return distribution, and if so,
does the effect survive a realistic fixed 1:2 execution model?

Pre-registered event families:
1) FLOW_CONTINUATION:
   extreme aggressive flow + price response in the same direction.
2) FLOW_ABSORPTION:
   extreme aggressive flow + weak price response; trade against the flow.
3) FLOW_EXHAUSTION:
   extreme prior flow followed by a material flow reversal; trade against the
   previous extreme.

The script writes:
reports/binance_v32_order_flow/
  data_quality.csv
  event_study.csv
  rr12_trades.csv
  rr12_summary.csv
  protocol_audit.txt
  research_panel.csv
"""

from __future__ import annotations
import calendar
import io
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ----------------------------
# LOCKED RESEARCH CONFIG
# ----------------------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT",
    "NEARUSDT", "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT",
    "ONDOUSDT", "PENDLEUSDT", "ICPUSDT", "WIFUSDT",
]

INTERVAL = "1h"
DAYS = 455
WARMUP_DAYS = 30

FLOW_Z_WINDOW = 48          # 2 days of 1H bars
FLOW_Z_TRIGGER = 1.50
RESPONSE_Z_WINDOW = 48
RESPONSE_Z_TRIGGER = 0.50
ABSORPTION_RESPONSE_ATR = 0.25
EXHAUSTION_LOOKBACK = 3
EXHAUSTION_CURRENT_ABS_Z = 0.25

ATR_WINDOW = 24
RR_SL_ATR = 1.0
RR_TP_ATR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003
INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE

REPORT_DIR = Path("reports/binance_v32_order_flow")
CACHE_DIR = Path(".cache/binance_v32")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


# ----------------------------
# Utilities
# ----------------------------
def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = pd.Timestamp(cur.year + 1, 1, 1, tz="UTC")
        else:
            cur = pd.Timestamp(cur.year, cur.month + 1, 1, tz="UTC")


def safe_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean().shift(1)
    sd = s.rolling(window, min_periods=window).std(ddof=0).shift(1)
    return (s - mu) / sd.replace(0, np.nan)


def max_loss_streak(outcomes) -> int:
    best = cur = 0
    for x in outcomes:
        if x == "LOSS":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def profit_factor(r):
    r = np.asarray(r, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return np.nan if losses <= 0 else gains / losses


def wr(r):
    r = np.asarray(r, dtype=float)
    return np.nan if len(r) == 0 else float((r > 0).mean())


def net_r_for_exit(gross_r: float) -> float:
    # Approximate round-trip cost on fixed notional, expressed in R where
    # 1R = margin_per_trade. Fees are charged on notional both ways;
    # slippage is applied once at entry and once at exit.
    round_trip_cost_usd = NOTIONAL * (2.0 * FEE_RATE + 2.0 * SLIPPAGE)
    return gross_r - round_trip_cost_usd / MARGIN_PER_TRADE


# ----------------------------
# Binance public archive
# ----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "V32-OrderFlow-Research/1.0"})


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    ym = f"{year:04d}-{month:02d}"
    fname = f"{symbol}-{INTERVAL}-{ym}.zip"
    url = f"{BASE_URL}/{symbol}/{INTERVAL}/{fname}"
    local = CACHE_DIR / symbol / fname
    local.parent.mkdir(parents=True, exist_ok=True)

    if not local.exists():
        for attempt in range(4):
            try:
                r = SESSION.get(url, timeout=45)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                local.write_bytes(r.content)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))

    try:
        with zipfile.ZipFile(local) as z:
            names = z.namelist()
            csv_name = next(n for n in names if n.lower().endswith(".csv"))
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=COLS)
        return df
    except Exception:
        # Corrupt cache: remove and retry once.
        try:
            local.unlink()
        except FileNotFoundError:
            pass
        r = SESSION.get(url, timeout=45)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        local.write_bytes(r.content)
        with zipfile.ZipFile(local) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(csv_name) as f:
                return pd.read_csv(f, header=None, names=COLS)


def fetch_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for y, m in month_iter(start, end):
        df = download_month(symbol, y, m)
        if df is not None and len(df):
            parts.append(df)

    if not parts:
        raise RuntimeError(f"No Binance archive data for {symbol}")

    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    # Remove incomplete current candle and any bars outside requested range.
    now = utc_now().floor("h")
    df = df[df["timestamp"] < now]
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    expected = pd.date_range(start.floor("h"), end.floor("h"), freq="1h", tz="UTC")
    idx = pd.DatetimeIndex(df["timestamp"])
    missing = expected.difference(idx)
    gap_count = len(missing)

    if len(df) < int(len(expected) * 0.97):
        raise RuntimeError(
            f"{symbol}: coverage too low rows={len(df)} expected={len(expected)} "
            f"coverage={len(df)/len(expected):.4f}"
        )

    if gap_count:
        # Missing bars are never forward-filled. We retain the data but flag gaps;
        # event generation will not bridge gaps.
        print(f"[WARN] {symbol}: missing hourly bars={gap_count}")

    df["symbol"] = symbol
    df["gap_from_prev"] = (
        df["timestamp"].diff().dt.total_seconds().div(3600).fillna(1) > 1.01
    )
    return df.reset_index(drop=True)


# ----------------------------
# Feature engineering
# ----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["atr"] = (
        pd.concat([
            x["high"] - x["low"],
            (x["high"] - x["close"].shift(1)).abs(),
            (x["low"] - x["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
    ).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    x["ret1"] = x["close"].pct_change()
    x["range_atr"] = (x["high"] - x["low"]) / x["atr"].replace(0, np.nan)

    # Published Binance field: taker buy quote asset volume.
    # Sell-side aggressive quote volume is total quote volume minus taker buy quote.
    x["taker_sell_quote"] = (x["quote_volume"] - x["taker_buy_quote"]).clip(lower=0)
    x["flow_imbalance"] = (
        (x["taker_buy_quote"] - x["taker_sell_quote"])
        / x["quote_volume"].replace(0, np.nan)
    )

    x["flow_z"] = safe_zscore(x["flow_imbalance"], FLOW_Z_WINDOW)
    x["ret_z"] = safe_zscore(x["ret1"], RESPONSE_Z_WINDOW)

    # Flow persistence is descriptive only; it uses current/past closed bars.
    x["flow_sign"] = np.sign(x["flow_imbalance"])
    x["flow_persist3"] = (
        x["flow_sign"].rolling(3, min_periods=3).sum().abs()
    )

    # Signed price response in ATR units.
    x["response_atr"] = x["ret1"] * x["close"] / x["atr"].replace(0, np.nan)

    # Previous extreme flow is explicitly shifted so exhaustion cannot use
    # the future/current outcome.
    x["prev_flow_z"] = x["flow_z"].shift(EXHAUSTION_LOOKBACK)
    x["prev_flow_sign"] = np.sign(x["prev_flow_z"])

    # A bar is considered a contiguous candidate only if neither it nor the
    # preceding lookback crosses a data gap.
    gap = x["gap_from_prev"].astype(bool)
    gap_recent = gap.rolling(max(FLOW_Z_WINDOW, EXHAUSTION_LOOKBACK) + 1,
                             min_periods=1).max()
    x["gap_safe"] = ~gap_recent.astype(bool)

    # ----------------------------
    # PRE-REGISTERED EVENT FAMILIES
    # ----------------------------
    extreme = x["flow_z"].abs() >= FLOW_Z_TRIGGER

    # 1) Continuation: extreme flow + same-direction price response.
    x["continuation_dir"] = np.where(
        (x["flow_z"] >= FLOW_Z_TRIGGER) & (x["ret_z"] >= RESPONSE_Z_TRIGGER), 1,
        np.where(
            (x["flow_z"] <= -FLOW_Z_TRIGGER) & (x["ret_z"] <= -RESPONSE_Z_TRIGGER),
            -1, 0
        )
    )

    # 2) Absorption: extreme flow but the bar's move is weak relative to ATR.
    # Trade against the aggressive side.
    x["absorption_dir"] = np.where(
        (x["flow_z"] >= FLOW_Z_TRIGGER)
        & (x["response_atr"].abs() <= ABSORPTION_RESPONSE_ATR), -1,
        np.where(
            (x["flow_z"] <= -FLOW_Z_TRIGGER)
            & (x["response_atr"].abs() <= ABSORPTION_RESPONSE_ATR),
            1, 0
        )
    )

    # 3) Exhaustion: previous extreme flow, then current flow reverses toward
    # neutral/opposite. Direction is opposite the prior extreme.
    x["exhaustion_dir"] = np.where(
        (x["prev_flow_z"] >= FLOW_Z_TRIGGER)
        & (x["flow_z"] <= EXHAUSTION_CURRENT_ABS_Z), -1,
        np.where(
            (x["prev_flow_z"] <= -FLOW_Z_TRIGGER)
            & (x["flow_z"] >= -EXHAUSTION_CURRENT_ABS_Z),
            1, 0
        )
    )

    x["signal_dir_continuation"] = x["continuation_dir"]
    x["signal_dir_absorption"] = x["absorption_dir"]
    x["signal_dir_exhaustion"] = x["exhaustion_dir"]

    return x


# ----------------------------
# Event study
# ----------------------------
def event_study(df: pd.DataFrame, direction_col: str, family: str) -> list[dict]:
    out = []
    for i in range(len(df) - 24):
        row = df.iloc[i]
        d = int(row[direction_col])
        if d == 0 or not bool(row["gap_safe"]):
            continue
        # Signal is known at close of i. Forward returns start from next open.
        entry = float(df.iloc[i + 1]["open"])
        rec = {
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "family": family,
            "direction": "LONG" if d > 0 else "SHORT",
            "flow_z": float(row["flow_z"]),
            "response_atr": float(row["response_atr"]),
        }
        for h in [1, 4, 8, 24]:
            j = i + h
            if j >= len(df):
                rec[f"fwd_{h}h_pct"] = np.nan
            else:
                rec[f"fwd_{h}h_pct"] = d * (float(df.iloc[j]["close"]) / entry - 1.0)
        out.append(rec)
    return out


# ----------------------------
# RR 1:2 engine
# ----------------------------
def simulate_family(df: pd.DataFrame, direction_col: str, family: str) -> list[dict]:
    trades = []
    open_until = -1
    last_exit_idx = -1

    for i in range(len(df) - 1):
        if i >= open_until:
            d = int(df.iloc[i][direction_col])
            if d == 0 or not bool(df.iloc[i]["gap_safe"]):
                continue

            # No same-candle re-entry: a signal on the exit bar cannot open.
            if i <= last_exit_idx:
                continue

            entry_bar = i + 1
            entry = float(df.iloc[entry_bar]["open"])
            atr = float(df.iloc[i]["atr"])
            if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
                continue

            # Slippage: adverse on entry.
            exec_entry = entry * (1.0 + SLIPPAGE if d > 0 else 1.0 - SLIPPAGE)
            stop = exec_entry - d * RR_SL_ATR * atr
            target = exec_entry + d * RR_TP_ATR * atr

            exit_idx = None
            gross_r = None
            reason = None

            for j in range(entry_bar, len(df)):
                # Never carry through a data gap.
                if j > entry_bar and bool(df.iloc[j]["gap_from_prev"]):
                    break

                hi = float(df.iloc[j]["high"])
                lo = float(df.iloc[j]["low"])

                if d > 0:
                    hit_sl = lo <= stop
                    hit_tp = hi >= target
                else:
                    hit_sl = hi >= stop
                    hit_tp = lo <= target

                if hit_sl and hit_tp:
                    reason = "SL_TP_SAME_CANDLE_LOSS"
                    gross_r = -1.0
                    exit_idx = j
                    break
                if hit_sl:
                    reason = "SL"
                    gross_r = -1.0
                    exit_idx = j
                    break
                if hit_tp:
                    reason = "TP"
                    gross_r = 2.0
                    exit_idx = j
                    break

            if exit_idx is None:
                # Censored open trade: excluded, not marked as loss.
                continue

            net_r = net_r_for_exit(float(gross_r))
            trades.append({
                "symbol": df.iloc[i]["symbol"],
                "signal_timestamp": df.iloc[i]["timestamp"],
                "entry_timestamp": df.iloc[entry_bar]["timestamp"],
                "exit_timestamp": df.iloc[exit_idx]["timestamp"],
                "family": family,
                "direction": "LONG" if d > 0 else "SHORT",
                "entry": exec_entry,
                "stop": stop,
                "target": target,
                "exit_reason": reason,
                "gross_r": gross_r,
                "net_r": net_r,
            })

            last_exit_idx = exit_idx
            open_until = exit_idx + 1

    return trades


def summarize(trades: pd.DataFrame, split_name: str, family: str, direction: str | None = None):
    t = trades[trades["split"] == split_name]
    if direction:
        t = t[t["direction"] == direction]
    if family:
        t = t[t["family"] == family]
    r = t["net_r"].to_numpy(float) if len(t) else np.array([])
    gross = t["gross_r"].to_numpy(float) if len(t) else np.array([])
    return {
        "split": split_name,
        "family": family,
        "direction": direction or "ALL",
        "trades": len(t),
        "wins": int((r > 0).sum()) if len(r) else 0,
        "losses": int((r <= 0).sum()) if len(r) else 0,
        "win_rate": wr(r),
        "profit_factor": profit_factor(r),
        "gross_R": float(gross.sum()) if len(gross) else 0.0,
        "net_R": float(r.sum()) if len(r) else 0.0,
        "mean_net_R": float(r.mean()) if len(r) else np.nan,
        "max_loss_streak": max_loss_streak(["WIN" if x > 0 else "LOSS" for x in r]),
    }


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    # Chronological split by timestamp globally, not by symbol.
    d = df.sort_values(["timestamp", "symbol"]).copy()
    ts = d["timestamp"]
    q1 = ts.quantile(0.50)
    q2 = ts.quantile(0.75)
    d["split"] = np.where(ts <= q1, "DISCOVERY",
                   np.where(ts <= q2, "DEVELOPMENT", "VALIDATION"))
    return d


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end = (utc_now() - pd.Timedelta(hours=1)).floor("h")
    start = (end - pd.Timedelta(days=DAYS + WARMUP_DAYS)).floor("h")
    research_start = (end - pd.Timedelta(days=DAYS)).floor("h")

    print("=== V32 ORDER FLOW STAGE-0 ===")
    print(f"Research window: {research_start} -> {end}")
    print(f"Warmup start:    {start}")
    print(f"Symbols: {len(SYMBOLS)}")

    panels = []
    quality = []

    for n, symbol in enumerate(SYMBOLS, 1):
        print(f"[{n}/{len(SYMBOLS)}] downloading {symbol}")
        df = fetch_symbol(symbol, start, end)
        f = build_features(df)
        f = f[f["timestamp"] >= research_start].copy()
        panels.append(f)

        expected = int((end - research_start).total_seconds() / 3600) + 1
        quality.append({
            "symbol": symbol,
            "rows": len(f),
            "expected_rows": expected,
            "coverage": len(f) / expected,
            "duplicate_timestamps": int(f["timestamp"].duplicated().sum()),
            "gap_bars": int(f["gap_from_prev"].sum()),
            "first_timestamp": f["timestamp"].min(),
            "last_timestamp": f["timestamp"].max(),
            "finite_flow_imbalance": int(np.isfinite(f["flow_imbalance"]).sum()),
        })

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Research-panel audit: all rolling features are shifted; future returns are
    # only constructed in event-study / execution loops after signal timestamp.
    panel.to_csv(REPORT_DIR / "research_panel.csv", index=False)
    pd.DataFrame(quality).to_csv(REPORT_DIR / "data_quality.csv", index=False)

    all_events = []
    all_trades = []

    for symbol, g in panel.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for col, family in [
            ("signal_dir_continuation", "FLOW_CONTINUATION"),
            ("signal_dir_absorption", "FLOW_ABSORPTION"),
            ("signal_dir_exhaustion", "FLOW_EXHAUSTION"),
        ]:
            all_events.extend(event_study(g, col, family))
            all_trades.extend(simulate_family(g, col, family))

    events = pd.DataFrame(all_events)
    trades = pd.DataFrame(all_trades)

    if len(events):
        events = add_split(events.rename(columns={"timestamp": "timestamp"}))
    else:
        events = pd.DataFrame()

    if len(trades):
        # Split by signal timestamp, not entry/exit timestamp.
        trades = add_split(trades.rename(columns={"signal_timestamp": "timestamp"}))
        trades.to_csv(REPORT_DIR / "rr12_trades.csv", index=False)
    else:
        trades = pd.DataFrame()
        trades.to_csv(REPORT_DIR / "rr12_trades.csv", index=False)

    if len(events):
        events.to_csv(REPORT_DIR / "event_study.csv", index=False)
    else:
        pd.DataFrame().to_csv(REPORT_DIR / "event_study.csv", index=False)

    # Event-study summary
    es_rows = []
    if len(events):
        for (split, family, direction), g in events.groupby(["split", "family", "direction"]):
            row = {
                "split": split,
                "family": family,
                "direction": direction,
                "events": len(g),
                "mean_fwd_1h": g["fwd_1h_pct"].mean(),
                "mean_fwd_4h": g["fwd_4h_pct"].mean(),
                "mean_fwd_8h": g["fwd_8h_pct"].mean(),
                "mean_fwd_24h": g["fwd_24h_pct"].mean(),
                "median_fwd_24h": g["fwd_24h_pct"].median(),
                "positive_fwd_24h_pct": (g["fwd_24h_pct"] > 0).mean(),
            }
            es_rows.append(row)
    pd.DataFrame(es_rows).to_csv(REPORT_DIR / "event_study_summary.csv", index=False)

    # RR summaries
    rows = []
    if len(trades):
        for family in ["FLOW_CONTINUATION", "FLOW_ABSORPTION", "FLOW_EXHAUSTION"]:
            for split in ["DISCOVERY", "DEVELOPMENT", "VALIDATION"]:
                for direction in ["LONG", "SHORT"]:
                    rows.append(summarize(trades, split, family, direction))
                rows.append(summarize(trades, split, family, None))
    rr_summary = pd.DataFrame(rows)
    rr_summary.to_csv(REPORT_DIR / "rr12_summary.csv", index=False)

    # Protocol audit
    audit = []
    audit.append("V32 PROTOCOL AUDIT")
    audit.append("==================")
    audit.append(f"UTC generated: {utc_now()}")
    audit.append("Data: Binance USD-M Futures public historical 1H klines")
    audit.append("Taker buy quote volume is taken directly from Binance kline field.")
    audit.append("Aggressive sell quote volume = total quote volume - taker buy quote volume.")
    audit.append("No OI, liquidation, funding, or fabricated CVD used.")
    audit.append("Signal uses closed bar t only.")
    audit.append("Execution uses next bar open t+1.")
    audit.append("Rolling z-scores are shifted by one bar.")
    audit.append("Same-candle SL+TP = LOSS.")
    audit.append("No timeout, BE, trailing, pyramiding.")
    audit.append("Max one open trade per symbol.")
    audit.append("No same-candle re-entry.")
    audit.append("End-of-sample censored trades are excluded.")
    audit.append("Discovery/Development/Validation = chronological 50/25/25.")
    audit.append("Thresholds were fixed before reading this run's results.")
    audit.append("")
    audit.append("COST MODEL")
    audit.append(f"Fee rate per side: {FEE_RATE}")
    audit.append(f"Slippage per side: {SLIPPAGE}")
    audit.append(f"Margin/trade: ${MARGIN_PER_TRADE}")
    audit.append(f"Leverage: {LEVERAGE}x")
    audit.append(f"Notional: ${NOTIONAL}")
    audit.append("")
    audit.append("IMPORTANT: Binance is research venue only. XT remains the intended execution venue.")
    (REPORT_DIR / "protocol_audit.txt").write_text("\n".join(audit), encoding="utf-8")

    # Console verdict: deliberately conservative.
    print("\n=== V32 COMPLETE ===")
    print(f"panel_rows={len(panel)} events={len(events)} trades={len(trades)}")
    if len(rr_summary):
        print(rr_summary.to_string(index=False))
    print(f"Reports: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
