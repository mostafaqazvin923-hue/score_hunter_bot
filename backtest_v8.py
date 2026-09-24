
#!/usr/bin/env python3
"""
XT USDT-M Futures 15m historical collector - V2.

Key fix vs V1:
1) Discover active Futures symbols from XT first:
   GET /future/market/v1/public/symbol/list
2) Never assume a symbol exists just because it appears in our desired list.
3) Resolve desired assets case-insensitively and with several safe aliases.
4) Test BTC first, then continue to all requested symbols.
5) Pagination remains timestamp-based, max 1500 candles/request.
6) A partial/invalid dataset causes the run to fail; no partial backtest.

Official XT docs expose the Futures symbol/list and Futures Kline endpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fapi.xt.com"
SYMBOL_LIST_URL = f"{BASE}/future/market/v1/public/symbol/list"
KLINE_URL = f"{BASE}/future/market/v1/public/q/kline"

INTERVAL = "15m"
LIMIT = 1500
TIMEOUT = 20
RETRIES = 5
SLEEP = 0.12
DEFAULT_DAYS = 365

TARGET_ASSETS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

OUTPUT_DIR = Path("data/xt_futures_15m")


def now_utc():
    return datetime.now(timezone.utc)


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def api_json(session, url, params=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected response type: {type(payload).__name__}")

            rc = payload.get("returnCode")
            if rc not in (None, 0, "0"):
                err = payload.get("error") or payload.get("msgInfo") or payload
                raise RuntimeError(f"XT API error: {err}")

            return payload
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2 * attempt, 5))

    raise RuntimeError(f"API request failed after {RETRIES} attempts: {last}")


def extract_list(payload):
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "list", "data", "symbols"):
            if isinstance(result.get(key), list):
                return result[key]
    raise RuntimeError(f"Could not find symbol list in response: {payload}")


def discover_symbols(session):
    payload = api_json(session, SYMBOL_LIST_URL)
    items = extract_list(payload)

    found = {}
    for item in items:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = (
                item.get("symbol")
                or item.get("s")
                or item.get("name")
                or item.get("pair")
            )
        else:
            continue

        if raw:
            found[str(raw).strip().upper()] = item

    if not found:
        raise RuntimeError("XT symbol/list returned zero usable symbols")

    return found


def resolve_symbol(asset, discovered):
    wanted = asset.upper()

    candidates = [
        f"{wanted}_USDT",
        f"{wanted}/USDT",
        f"{wanted}-USDT",
        wanted,
    ]

    # First exact/canonical matching.
    for candidate in candidates:
        if candidate.upper() in discovered:
            return candidate.upper()

    # Then normalize separators for safety.
    def norm(x):
        return str(x).upper().replace("/", "_").replace("-", "_")

    matches = [
        actual for actual in discovered
        if norm(actual) == f"{wanted}_USDT"
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def kline_rows(payload):
    result = payload.get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"XT Kline result is not a list: {payload}")
    return result


def normalize_rows(rows, symbol):
    records = []

    for row in rows:
        if isinstance(row, dict):
            # Official XT fields.
            keys = ("t", "o", "h", "l", "c", "a")
            if not all(k in row for k in keys):
                raise RuntimeError(f"{symbol}: malformed dict kline: {row}")
            records.append({
                "Timestamp": int(row["t"]),
                "Open": float(row["o"]),
                "High": float(row["h"]),
                "Low": float(row["l"]),
                "Close": float(row["c"]),
                "Volume": float(row["a"]),
                "Turnover": float(row["v"]) if row.get("v") is not None else float("nan"),
            })

        elif isinstance(row, (list, tuple)):
            # Defensive support if XT returns array rows.
            if len(row) < 6:
                raise RuntimeError(f"{symbol}: malformed array kline: {row}")
            records.append({
                "Timestamp": int(row[0]),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
                "Turnover": float(row[6]) if len(row) > 6 else float("nan"),
            })
        else:
            raise RuntimeError(f"{symbol}: unsupported kline row: {row!r}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df


def fetch_batch(session, symbol, start_ms, end_ms):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    payload = api_json(session, KLINE_URL, params)
    return normalize_rows(kline_rows(payload), symbol)


def audit(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty dataset")

    bad = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    if bad.any():
        raise RuntimeError(f"{symbol}: {int(bad.sum())} invalid OHLCV rows")

    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=15)]
    wrong = diffs[diffs != pd.Timedelta(minutes=15)]

    return {
        "rows": int(len(df)),
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "duplicates": int(df["Date"].duplicated().sum()),
        "gaps": int(len(gaps)),
        "max_gap_minutes": int(gaps.max().total_seconds() / 60) if len(gaps) else 15,
        "wrong_intervals": int(len(wrong)),
    }


def download_symbol(session, symbol, start_dt, end_dt):
    cursor = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    all_batches = []
    calls = 0
    previous_last = None

    while cursor <= end_ms:
        batch = fetch_batch(session, symbol, cursor, end_ms)
        calls += 1

        if batch.empty:
            break

        batch = batch.sort_values("Date").reset_index(drop=True)
        first = int(batch["Timestamp"].iloc[0])
        last = int(batch["Timestamp"].iloc[-1])

        if previous_last is not None and first <= previous_last:
            raise RuntimeError(
                f"{symbol}: pagination overlap/non-progress: "
                f"first={first}, previous_last={previous_last}"
            )

        if last < cursor:
            raise RuntimeError(f"{symbol}: API moved backwards")

        all_batches.append(batch)

        print(
            f"  {symbol}: request={calls:02d} "
            f"rows={len(batch):4d} "
            f"{batch['Date'].iloc[0]} -> {batch['Date'].iloc[-1]}"
        )

        if last >= end_ms:
            break

        previous_last = last
        cursor = last + 1

        if calls > 1000:
            raise RuntimeError(f"{symbol}: pagination safety stop")

        time.sleep(SLEEP)

    if not all_batches:
        raise RuntimeError(f"{symbol}: zero historical candles")

    df = pd.concat(all_batches, ignore_index=True)
    df = (
        df.drop_duplicates(subset=["Date"], keep="last")
          .sort_values("Date")
          .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

    # Remove current, still-forming 15m candle.
    current = pd.Timestamp.now(tz="UTC")
    if not df.empty and df["Date"].iloc[-1] + pd.Timedelta(minutes=15) > current:
        df = df.iloc[:-1].copy()

    report = audit(df, symbol)
    report["api_requests"] = calls
    return df, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--symbols", nargs="*", default=TARGET_ASSETS)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be > 0")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_dt = now_utc()
    start_dt = end_dt - timedelta(days=args.days)

    session = requests.Session()
    session.headers.update({"User-Agent": "HUNTER-XT-Futures-Collector/2.0"})

    print("=== XT USDT-M FUTURES SYMBOL DISCOVERY ===")
    discovered = discover_symbols(session)
    print(f"XT Futures symbols discovered: {len(discovered)}")

    resolved = {}
    unresolved = []

    for asset in args.symbols:
        actual = resolve_symbol(asset, discovered)
        if actual is None:
            unresolved.append(asset)
        else:
            resolved[asset] = actual

    print("\nRequested -> XT symbol")
    for asset in args.symbols:
        print(f"  {asset:8s} -> {resolved.get(asset, 'NOT FOUND')}")

    if unresolved:
        # Print useful candidates for debugging instead of guessing symbols.
        print("\nUNRESOLVED symbols:", ", ".join(unresolved))
        print("\nAvailable XT symbols containing USDT and requested asset:")
        for asset in unresolved:
            matches = [
                s for s in discovered
                if asset.upper() in s and "USDT" in s
            ]
            print(f"  {asset}: {matches[:20]}")

        raise SystemExit(2)

    # Mandatory first-stage BTC validation.
    btc_symbol = resolved.get("BTC")
    if btc_symbol is None:
        raise SystemExit("BTC Futures symbol was not resolved")

    print("\n=== STAGE 1: BTC 15m 365-DAY VALIDATION ===")
    btc_df, btc_report = download_symbol(session, btc_symbol, start_dt, end_dt)

    expected = args.days * 24 * 4
    print(
        f"BTC RESULT: rows={btc_report['rows']:,} "
        f"expected≈{expected:,} "
        f"coverage={btc_report['start']} -> {btc_report['end']} "
        f"gaps={btc_report['gaps']}"
    )

    if btc_report["rows"] < int(expected * 0.995):
        raise RuntimeError(
            f"BTC coverage is materially incomplete: "
            f"{btc_report['rows']:,}/{expected:,} rows"
        )

    btc_df.to_csv(out_dir / f"{btc_symbol}_15m.csv", index=False)

    # Stage 2: all symbols.
    print("\n=== STAGE 2: ALL REQUESTED SYMBOLS ===")
    reports = [dict(btc_report, requested_asset="BTC", xt_symbol=btc_symbol)]
    failures = []

    for asset in args.symbols:
        if asset == "BTC":
            continue

        symbol = resolved[asset]
        try:
            df, report = download_symbol(session, symbol, start_dt, end_dt)
            report["requested_asset"] = asset
            report["xt_symbol"] = symbol
            df.to_csv(out_dir / f"{symbol}_15m.csv", index=False)
            reports.append(report)

            print(
                f"  OK {asset}: rows={report['rows']:,} "
                f"gaps={report['gaps']} "
                f"requests={report['api_requests']}"
            )
        except Exception as exc:
            failures.append({"asset": asset, "symbol": symbol, "error": str(exc)})
            print(f"  FAIL {asset}/{symbol}: {exc}")

    report = {
        "generated_at": now_utc().isoformat(),
        "days": args.days,
        "interval": INTERVAL,
        "discovered_symbols": len(discovered),
        "resolved": resolved,
        "reports": reports,
        "failures": failures,
    }

    report_path = out_dir / "AUDIT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== FINAL AUDIT ===")
    print(f"Resolved : {len(resolved)}/{len(args.symbols)}")
    print(f"Loaded   : {len(reports)}/{len(args.symbols)}")
    print(f"Failed   : {len(failures)}")
    print(f"Report   : {report_path}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
