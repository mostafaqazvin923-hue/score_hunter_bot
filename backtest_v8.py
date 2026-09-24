
#!/usr/bin/env python3
"""
XT USDT-M Futures 15m historical collector.

Uses XT's public Futures Kline endpoint directly (not CCXT):
GET https://fapi.xt.com/future/market/v1/public/q/kline

Pagination is timestamp-based and capped at 1500 rows/request.
The collector downloads one year of REAL XT Futures OHLCV per symbol,
deduplicates candles, validates OHLCV, checks 15m gaps, and writes CSVs.

Run:
    python xt_futures_15m_collector.py --days 365 --symbols BTC_USDT

For the full 14-symbol set:
    python xt_futures_15m_collector.py --days 365
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"
INTERVAL = "15m"
LIMIT = 1500
REQUEST_TIMEOUT = 20
RETRIES = 5
SLEEP_SECONDS = 0.12
DAYS = 365

SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "SUI_USDT", "AVAX_USDT",
    "NEAR_USDT", "ADA_USDT", "BNB_USDT", "APT_USDT", "CRV_USDT",
    "ONDO_USDT", "PENDLE_USDT", "ICP_USDT", "WIF_USDT",
]

OUTPUT_DIR = Path("data/xt_futures_15m")


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_result(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected XT response type: {type(payload).__name__}")

    rc = payload.get("returnCode")
    if rc not in (0, "0", None):
        err = payload.get("error") or payload.get("msgInfo") or payload
        raise RuntimeError(f"XT API error: {err}")

    result = payload.get("result")
    if result is None:
        # Fail loudly rather than silently accepting an empty/changed schema.
        raise RuntimeError(f"XT response has no result: {payload}")

    if not isinstance(result, list):
        raise RuntimeError(f"XT result is not a list: {type(result).__name__}")

    return result


def fetch_batch(session: requests.Session, symbol: str, start_ms: int, end_ms: int):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return parse_result(r.json())
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(
        f"Failed {symbol} start={start_ms} end={end_ms}: {last_error}"
    )


def normalize_batch(rows, symbol: str) -> pd.DataFrame:
    out = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{symbol}: unexpected kline row: {row!r}")

        # XT documented fields: a=volume, c=close, h=high, l=low,
        # o=open, s=symbol, t=time, v=turnover.
        required = ("t", "o", "h", "l", "c", "a")
        if any(k not in row for k in required):
            raise RuntimeError(f"{symbol}: malformed kline row: {row!r}")

        out.append({
            "Timestamp": int(row["t"]),
            "Open": float(row["o"]),
            "High": float(row["h"]),
            "Low": float(row["l"]),
            "Close": float(row["c"]),
            "Volume": float(row["a"]),
            "Turnover": float(row["v"]) if row.get("v") is not None else float("nan"),
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df


def validate_ohlcv(df: pd.DataFrame, symbol: str):
    if df.empty:
        raise RuntimeError(f"{symbol}: no candles downloaded")

    bad = (
        (df["Open"] <= 0) | (df["High"] <= 0) |
        (df["Low"] <= 0) | (df["Close"] <= 0) |
        (df["Volume"] < 0) |
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) |
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    if bad.any():
        raise RuntimeError(f"{symbol}: {int(bad.sum())} invalid OHLCV rows")

    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=15)]
    wrong = diffs[diffs < pd.Timedelta(minutes=15)]

    return {
        "rows": int(len(df)),
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "duplicates": int(df["Date"].duplicated().sum()),
        "gaps": int(len(gaps)),
        "max_gap_minutes": (
            int(gaps.max().total_seconds() / 60) if len(gaps) else 15
        ),
        "wrong_intervals": int(len(wrong)),
    }


def download_symbol(
    session: requests.Session,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
):
    start_ms = ms(start_dt)
    end_ms = ms(end_dt)
    cursor = start_ms
    rows = []
    requests_count = 0

    while cursor <= end_ms:
        batch = fetch_batch(session, symbol, cursor, end_ms)
        requests_count += 1

        if not batch:
            break

        df = normalize_batch(batch, symbol)
        if df.empty:
            break

        rows.extend(df.to_dict("records"))

        last_ts = int(df["Timestamp"].max())

        # Strict forward progress. This prevents infinite loops and overlap.
        if last_ts < cursor:
            raise RuntimeError(
                f"{symbol}: API returned data before cursor "
                f"({last_ts} < {cursor})"
            )

        next_cursor = last_ts + 1

        print(
            f"  {symbol}: request={requests_count:02d} "
            f"batch={len(df):4d} "
            f"through={pd.to_datetime(last_ts, unit='ms', utc=True)}"
        )

        if last_ts >= end_ms:
            break

        if len(df) < LIMIT:
            # With an explicit endTime this normally means the requested
            # interval has been exhausted.
            break

        cursor = next_cursor
        time.sleep(SLEEP_SECONDS)

        if requests_count > 1000:
            raise RuntimeError(f"{symbol}: pagination safety limit exceeded")

    if not rows:
        raise RuntimeError(f"{symbol}: zero rows returned")

    all_df = pd.DataFrame(rows)
    all_df["Date"] = pd.to_datetime(all_df["Timestamp"], unit="ms", utc=True)

    all_df = (
        all_df
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Exact requested range.
    all_df = all_df[
        (all_df["Date"] >= pd.Timestamp(start_dt)) &
        (all_df["Date"] <= pd.Timestamp(end_dt))
    ].copy()

    # Never include a still-forming 15m candle.
    now = pd.Timestamp.now(tz="UTC")
    if not all_df.empty:
        last = all_df["Date"].iloc[-1]
        if last + pd.Timedelta(minutes=15) > now:
            all_df = all_df.iloc[:-1].copy()

    report = validate_ohlcv(all_df, symbol)
    report["api_requests"] = requests_count
    return all_df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--output", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be positive")

    symbols = args.symbols
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=args.days)

    session = requests.Session()
    session.headers.update({"User-Agent": "HUNTER-XT-Futures-Backtest/1.0"})

    reports = []
    failures = []

    print("=== XT USDT-M FUTURES 15m HISTORICAL COLLECTOR ===")
    print(f"Endpoint : {BASE_URL}")
    print(f"Interval : {INTERVAL}")
    print(f"Range    : {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"Symbols  : {len(symbols)}")
    print()

    for symbol in symbols:
        try:
            print(f"Downloading {symbol} ...")
            df, report = download_symbol(session, symbol, start_dt, end_dt)

            out = out_dir / f"{symbol}_15m.csv"
            df.to_csv(out, index=False)

            report["symbol"] = symbol
            report["output"] = str(out)
            reports.append(report)

            print(
                f"  OK rows={report['rows']:,} "
                f"requests={report['api_requests']} "
                f"gaps={report['gaps']} "
                f"max_gap={report['max_gap_minutes']}m"
            )
            print()

        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(f"  FAIL {symbol}: {exc}")
            print()

    report_path = out_dir / "AUDIT_REPORT.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now().isoformat(),
                "days": args.days,
                "interval": INTERVAL,
                "reports": reports,
                "failures": [
                    {"symbol": s, "error": e} for s, e in failures
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== FINAL AUDIT ===")
    print(f"Loaded : {len(reports)}/{len(symbols)}")
    print(f"Failed : {len(failures)}")
    print(f"Report : {report_path}")

    # A failed symbol must fail the workflow. No partial backtest.
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
