
import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V79
# Quality-score allocation + loss-cluster penalty
# ============================================================

TIMEFRAME = "4h"
LOOKBACK_DAYS = 365

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

MAX_POSITIONS = 5
MAX_SAME_SYMBOL = 2

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
EMA_WARMUP = 200
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0
TIMEOUT_CANDLES = 45

# V76 signal core
MIN_LONG_MOM_SHORT = 0.012
MIN_LONG_MOM_LONG = 0.035
MIN_SHORT_MOM_SHORT = -0.012
MIN_SHORT_MOM_LONG = -0.035

# V79: ranking penalty for recent same-symbol/same-direction losses.
LOSS_PENALTY_1 = 2.0
LOSS_PENALTY_2 = 4.0
LOSS_PENALTY_3 = 6.0

OUTPUT_DIR = "hunter_v79_output"

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000,
})


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def utc_now():
    return datetime.now(timezone.utc)


def fetch_symbol_ohlcv(symbol, since_ms, until_ms):
    rows = []
    cursor = int(since_ms)
    last_cursor = None

    for _ in range(100):
        batch = None
        last_error = None

        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=cursor,
                    limit=1000,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    import time
                    time.sleep(2.0 * (attempt + 1))

        if batch is None:
            raise RuntimeError(
                f"fetch_ohlcv failed for {symbol}: {last_error}"
            )

        if not batch:
            break

        rows.extend(batch)
        newest = int(batch[-1][0])

        if last_cursor is not None and newest <= last_cursor:
            break

        last_cursor = newest

        if newest >= until_ms:
            break

        cursor = newest + 1

        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    )

    df = (
        df.drop_duplicates(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    # Remove the still-forming candle.
    current_ms = int(utc_now().timestamp() * 1000)
    candle_ms = 4 * 60 * 60 * 1000
    df = df[df["Timestamp"] + candle_ms <= current_ms].copy()

    df = df[
        (df["Timestamp"] >= since_ms)
        & (df["Timestamp"] <= until_ms)
    ].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .drop_duplicates(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        return df

    diffs = df["Timestamp"].diff().dropna()
    max_allowed_gap = candle_ms + 10 * 60 * 1000

    if not diffs.empty and diffs.max() > max_allowed_gap:
        raise RuntimeError(
            f"Large OHLCV gap detected for {symbol}"
        )

    return df


def add_indicators(df):
    d = df.copy()

    prev_close = d["Close"].shift(1)

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - prev_close).abs()
    tr3 = (d["Low"] - prev_close).abs()

    d["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    d["ATR"] = d["TR"].rolling(ATR_PERIOD).mean()

    d["EMA20"] = d["Close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    d["EMA50"] = d["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    d["EMA200"] = d["Close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    d["Mom_Short"] = d["Close"].pct_change(10)
    d["Mom_Long"] = d["Close"].pct_change(30)

    d["ATR_PCT"] = (
        d["ATR"] / d["Close"].replace(0, np.nan)
    )

    d["EMA20_SLOPE"] = d["EMA20"].pct_change(5)
    d["DIST_EMA20"] = (
        d["Close"] / d["EMA20"] - 1.0
    )

    d["EMA20_prev"] = d["EMA20"].shift(1)
    d["Low_prev"] = d["Low"].shift(1)
    d["High_prev"] = d["High"].shift(1)

    d["TimestampDT"] = pd.to_datetime(
        d["Timestamp"],
        unit="ms",
        utc=True,
    )

    return d


def load_data():
    now_ms = int(utc_now().timestamp() * 1000)
    since_ms = int(
        (
            utc_now()
            - timedelta(days=LOOKBACK_DAYS)
        ).timestamp() * 1000
    )

    data = {}

    print("=" * 72)
    print("HUNTER-V79 DATA LOAD")
    print("=" * 72)

    for name, symbol in SYMBOLS.items():
        try:
            raw = fetch_symbol_ohlcv(
                symbol,
                since_ms,
                now_ms,
            )

            if len(raw) < EMA_WARMUP + 50:
                print(
                    f"{name:8s} INVALID - "
                    f"only {len(raw)} candles"
                )
                continue

            data[name] = add_indicators(raw)

            print(
                f"{name:8s} OK - "
                f"{len(data[name])} candles"
            )

        except Exception as exc:
            print(
                f"{name:8s} ERROR - {exc}"
            )

    print(
        f"\nValid Symbols: "
        f"{len(data)}/{len(SYMBOLS)}"
    )

    return data


def row_at_or_before(df, ts):
    values = df["TimestampDT"].values
    pos = (
        values.searchsorted(
            np.datetime64(ts),
            side="right",
        )
        - 1
    )

    if pos < 0:
        return None

    return df.iloc[int(pos)]


def build_time_index(data):
    all_ts = set()

    for df in data.values():
        all_ts.update(
            df["TimestampDT"].tolist()
        )

    return sorted(all_ts)


def get_market_context(data, ts):
    btc = data.get("BTC")

    btc_row = (
        row_at_or_before(btc, ts)
        if btc is not None
        else None
    )

    if btc_row is None:
        return {
            "btc_bull": False,
            "btc_bear": False,
            "btc_mom_long": 0.0,
            "btc_mom_short": 0.0,
            "breadth": 0.5,
        }

    btc_bull = (
        btc_row["Close"] > btc_row["EMA200"]
        and btc_row["EMA50"] > btc_row["EMA200"]
    )

    btc_bear = (
        btc_row["Close"] < btc_row["EMA200"]
        and btc_row["EMA50"] < btc_row["EMA200"]
    )

    bull_count = 0
    active = 0

    for df in data.values():
        row = row_at_or_before(df, ts)

        if row is None:
            continue

        if pd.isna(row["EMA200"]):
            continue

        active += 1

        if row["Close"] > row["EMA200"]:
            bull_count += 1

    breadth = (
        bull_count / active
        if active
        else 0.5
    )

    return {
        "btc_bull": bool(btc_bull),
        "btc_bear": bool(btc_bear),
        "btc_mom_long": safe_float(
            btc_row["Mom_Long"]
        ),
        "btc_mom_short": safe_float(
            btc_row["Mom_Short"]
        ),
        "breadth": float(breadth),
    }


def raw_signal(row):
    if (
        pd.isna(row["EMA200"])
        or pd.isna(row["ATR"])
        or pd.isna(row["EMA20_prev"])
    ):
        return None

    long_ok = (
        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]
        and row["Close"] > row["EMA200"]
        and row["Low_prev"]
        <= row["EMA20_prev"] * 1.015
        and row["Mom_Short"]
        > MIN_LONG_MOM_SHORT
        and row["Mom_Long"]
        > MIN_LONG_MOM_LONG
    )

    short_ok = (
        row["Close"] < row["EMA20"]
        and row["EMA20"] < row["EMA50"]
        and row["Close"] < row["EMA200"]
        and row["High_prev"]
        >= row["EMA20_prev"] * 0.985
        and row["Mom_Short"]
        < MIN_SHORT_MOM_SHORT
        and row["Mom_Long"]
        < MIN_SHORT_MOM_LONG
    )

    if not long_ok and not short_ok:
        return None

    if long_ok and short_ok:
        return (
            "long"
            if row["Mom_Long"] >= 0
            else "short"
        )

    return (
        "long"
        if long_ok
        else "short"
    )


def quality_score(
    row,
    side,
    market,
    recent_losses,
):
    """
    Continuous ranking score.

    IMPORTANT:
    These features rank valid V76-style signals.
    They do NOT hard-filter the signal stream.

    That is intentional: V78 lost too many trades through
    direction_limit. V79 uses quality to decide which candidates
    receive the five available portfolio slots.
    """

    mom_long = safe_float(
        row["Mom_Long"]
    )
    mom_short = safe_float(
        row["Mom_Short"]
    )
    atr_pct = safe_float(
        row["ATR_PCT"]
    )
    distance = safe_float(
        row["DIST_EMA20"]
    )
    slope = safe_float(
        row["EMA20_SLOPE"]
    )

    btc_mom_long = safe_float(
        market["btc_mom_long"]
    )
    breadth = safe_float(
        market["breadth"]
    )

    score = 0.0

    if side == "long":
        # BTC trend agreement.
        if market["btc_bull"]:
            score += 3.0
        elif market["btc_bear"]:
            score -= 2.5
        else:
            score += 0.5

        # Breadth agreement.
        score += 3.0 * (
            (breadth - 0.50) / 0.50
        )

        # Coin momentum.
        score += 3.0 * max(
            -1.0,
            min(1.0, mom_long / 0.12),
        )

        score += 1.5 * max(
            -1.0,
            min(1.0, mom_short / 0.05),
        )

        # Relative momentum versus BTC.
        relative_momentum = (
            mom_long - btc_mom_long
        )

        score += 2.5 * max(
            -1.0,
            min(
                1.0,
                relative_momentum / 0.10,
            ),
        )

        # EMA20 slope.
        score += 1.5 * max(
            -1.0,
            min(1.0, slope / 0.03),
        )

        # Avoid very stretched continuation entries.
        if -0.015 <= distance <= 0.035:
            score += 1.5
        elif distance > 0.060:
            score -= 2.0

        # Volatility regime preference.
        if 0.010 <= atr_pct <= 0.045:
            score += 1.0
        elif atr_pct > 0.070:
            score -= 1.5

    else:
        # BTC trend agreement.
        if market["btc_bear"]:
            score += 3.0
        elif market["btc_bull"]:
            score -= 2.5
        else:
            score += 0.5

        # Breadth agreement.
        score += 3.0 * (
            (0.50 - breadth) / 0.50
        )

        # Coin momentum.
        score += 3.0 * max(
            -1.0,
            min(1.0, (-mom_long) / 0.12),
        )

        score += 1.5 * max(
            -1.0,
            min(1.0, (-mom_short) / 0.05),
        )

        # Relative momentum versus BTC.
        relative_momentum = (
            mom_long - btc_mom_long
        )

        score += 2.5 * max(
            -1.0,
            min(
                1.0,
                (-relative_momentum) / 0.10,
            ),
        )

        # EMA20 slope.
        score += 1.5 * max(
            -1.0,
            min(1.0, (-slope) / 0.03),
        )

        # Avoid very stretched continuation entries.
        if -0.035 <= distance <= 0.015:
            score += 1.5
        elif distance < -0.060:
            score -= 2.0

        # Volatility regime preference.
        if 0.010 <= atr_pct <= 0.045:
            score += 1.0
        elif atr_pct > 0.070:
            score -= 1.5

    # Recent losses: ranking penalty only.
    if recent_losses == 1:
        score -= LOSS_PENALTY_1
    elif recent_losses == 2:
        score -= LOSS_PENALTY_2
    elif recent_losses >= 3:
        score -= LOSS_PENALTY_3

    return float(score)


def run_backtest(data):
    timestamps = build_time_index(data)

    positions = {}
    trades = []
    equity_curve = []

    # Consecutive losses for the exact symbol + direction.
    loss_cluster = defaultdict(int)

    diagnostics = Counter()

    for ts in timestamps:
        # --------------------------------------------------------
        # 1. Manage existing positions.
        # Old stop is checked BEFORE trailing-stop update.
        # This avoids same-candle trailing look-ahead.
        # --------------------------------------------------------
        to_close = []

        for key, pos in list(
            positions.items()
        ):
            df = data[pos["symbol"]]
            row = row_at_or_before(
                df,
                ts,
            )

            if row is None:
                continue

            if ts < pos["entry_ts"]:
                continue

            pos["candles"] += 1

            side = pos["side"]
            exit_price = None
            exit_reason = None

            if side == "long":
                if row["Low"] <= pos["stop"]:
                    exit_price = (
                        pos["stop"]
                        * (1.0 - SLIPPAGE)
                    )
                    exit_reason = "STOP"

                elif (
                    pos["candles"]
                    >= TIMEOUT_CANDLES
                ):
                    exit_price = (
                        row["Close"]
                        * (1.0 - SLIPPAGE)
                    )
                    exit_reason = "TIMEOUT"

                if exit_reason is None:
                    new_stop = (
                        row["Close"]
                        - row["ATR"]
                        * TRAILING_ATR_MULTIPLIER
                    )

                    if np.isfinite(new_stop):
                        pos["stop"] = max(
                            pos["stop"],
                            new_stop,
                        )

            else:
                if row["High"] >= pos["stop"]:
                    exit_price = (
                        pos["stop"]
                        * (1.0 + SLIPPAGE)
                    )
                    exit_reason = "STOP"

                elif (
                    pos["candles"]
                    >= TIMEOUT_CANDLES
                ):
                    exit_price = (
                        row["Close"]
                        * (1.0 + SLIPPAGE)
                    )
                    exit_reason = "TIMEOUT"

                if exit_reason is None:
                    new_stop = (
                        row["Close"]
                        + row["ATR"]
                        * TRAILING_ATR_MULTIPLIER
                    )

                    if np.isfinite(new_stop):
                        pos["stop"] = min(
                            pos["stop"],
                            new_stop,
                        )

            if exit_price is None:
                continue

            qty = pos["qty"]

            if side == "long":
                raw_pnl = (
                    exit_price
                    - pos["entry"]
                ) * qty
            else:
                raw_pnl = (
                    pos["entry"]
                    - exit_price
                ) * qty

            total_fee = (
                (
                    pos["entry"] * qty
                )
                + (
                    exit_price * qty
                )
            ) * FEE_RATE

            pnl = raw_pnl - total_fee

            # Correct R definition:
            # 1R = initial dollar risk.
            risk_dollar = pos["risk_dollar"]

            real_r = (
                pnl / risk_dollar
                if risk_dollar > 0
                else 0.0
            )

            won = pnl > 0
            cluster_key = (
                pos["symbol"],
                side,
            )

            if won:
                loss_cluster[
                    cluster_key
                ] = 0
            else:
                loss_cluster[
                    cluster_key
                ] += 1

            trades.append({
                "symbol": pos["symbol"],
                "side": side.upper(),
                "entry_ts": pos["entry_ts"],
                "exit_ts": ts,
                "entry": pos["entry"],
                "exit": exit_price,
                "initial_stop": pos["initial_stop"],
                "final_stop": pos["stop"],
                "qty": qty,
                "margin": pos["margin"],
                "leverage": LEVERAGE,
                "risk_dollar": risk_dollar,
                "pnl": pnl,
                "r": real_r,
                "win": int(won),
                "exit_reason": exit_reason,
                "quality_score": pos["quality_score"],
                "loss_cluster_before_entry": (
                    pos[
                        "loss_cluster_before_entry"
                    ]
                ),
            })

            to_close.append(key)

        for key in to_close:
            positions.pop(key, None)

        # --------------------------------------------------------
        # 2. Build all V76-valid candidates.
        # --------------------------------------------------------
        market = get_market_context(
            data,
            ts,
        )

        candidates = []

        for name, df in data.items():
            diagnostics[
                "symbol_checks"
            ] += 1

            row = row_at_or_before(
                df,
                ts,
            )

            if row is None:
                continue

            if (
                pd.isna(row["EMA200"])
                or pd.isna(row["ATR"])
            ):
                continue

            side = raw_signal(row)

            if side is None:
                continue

            diagnostics[
                f"{side}_signal"
            ] += 1

            key = (name, side)

            if key in positions:
                diagnostics[
                    "already_open"
                ] += 1
                continue

            # Keep the V76 execution model for direct comparison:
            # signal is evaluated on this bar and entry uses this bar
            # Open. This intentionally matches the historical baseline.
            if side == "long":
                entry = (
                    row["Open"]
                    * (1.0 + SLIPPAGE)
                )
                initial_stop = (
                    entry
                    - row["ATR"]
                    * INITIAL_ATR_MULTIPLIER
                )
            else:
                entry = (
                    row["Open"]
                    * (1.0 - SLIPPAGE)
                )
                initial_stop = (
                    entry
                    + row["ATR"]
                    * INITIAL_ATR_MULTIPLIER
                )

            stop_distance_pct = (
                abs(
                    entry
                    - initial_stop
                )
                / entry
            )

            if not np.isfinite(
                stop_distance_pct
            ):
                continue

            # V76 stop sanity range.
            if not (
                0.01
                <= stop_distance_pct
                <= 0.04
            ):
                diagnostics[
                    "invalid_stop_distance"
                ] += 1
                continue

            recent_losses = loss_cluster.get(
                key,
                0,
            )

            score = quality_score(
                row,
                side,
                market,
                recent_losses,
            )

            candidates.append({
                "symbol": name,
                "side": side,
                "ts": ts,
                "entry": float(entry),
                "initial_stop": float(
                    initial_stop
                ),
                "stop_distance_pct": float(
                    stop_distance_pct
                ),
                "quality_score": float(
                    score
                ),
                "loss_cluster": int(
                    recent_losses
                ),
                "mom_long": safe_float(
                    row["Mom_Long"]
                ),
            })

        diagnostics[
            "candidates_before_ranking"
        ] += len(candidates)

        # --------------------------------------------------------
        # 3. Rank all valid candidates.
        #
        # IMPORTANT:
        # No MAX_LONGS=3 / MAX_SHORTS=3.
        # V78 discarded 1007 signals through direction_limit.
        # V79 has one portfolio limit: MAX_POSITIONS=5.
        # --------------------------------------------------------
        candidates.sort(
            key=lambda item: (
                safe_float(
                    item["quality_score"]
                ),
                safe_float(
                    item["mom_long"]
                ),
            ),
            reverse=True,
        )

        slots = (
            MAX_POSITIONS
            - len(positions)
        )

        if slots > 0:
            chosen = []
            same_symbol_count = Counter()

            for candidate in candidates:
                if len(chosen) >= slots:
                    break

                name = candidate[
                    "symbol"
                ]

                side = candidate[
                    "side"
                ]

                key = (name, side)

                if key in positions:
                    continue

                if (
                    same_symbol_count[name]
                    >= MAX_SAME_SYMBOL
                ):
                    diagnostics[
                        "symbol_allocation_limit"
                    ] += 1
                    continue

                chosen.append(candidate)
                same_symbol_count[
                    name
                ] += 1

            diagnostics[
                "accepted"
            ] += len(chosen)

            diagnostics[
                "ranked_out"
            ] += max(
                0,
                len(candidates)
                - len(chosen),
            )

            for candidate in chosen:
                name = candidate[
                    "symbol"
                ]

                side = candidate[
                    "side"
                ]

                key = (
                    name,
                    side,
                )

                margin = TRADE_MARGIN
                notional = (
                    margin * LEVERAGE
                )

                qty = (
                    notional
                    / candidate["entry"]
                )

                risk_dollar = (
                    abs(
                        candidate["entry"]
                        - candidate[
                            "initial_stop"
                        ]
                    )
                    * qty
                )

                if risk_dollar <= 0:
                    continue

                positions[key] = {
                    "symbol": name,
                    "side": side,
                    "entry_ts": ts,
                    "entry": candidate[
                        "entry"
                    ],
                    "initial_stop": candidate[
                        "initial_stop"
                    ],
                    "stop": candidate[
                        "initial_stop"
                    ],
                    "qty": qty,
                    "margin": margin,
                    "risk_dollar": risk_dollar,
                    "candles": 0,
                    "quality_score": candidate[
                        "quality_score"
                    ],
                    "loss_cluster_before_entry": candidate[
                        "loss_cluster"
                    ],
                }

        # --------------------------------------------------------
        # 4. Mark-to-market equity.
        # Used only for DD reporting.
        # --------------------------------------------------------
        mtm = 0.0

        for pos in positions.values():
            df = data[pos["symbol"]]

            row = row_at_or_before(
                df,
                ts,
            )

            if row is None:
                continue

            if pos["side"] == "long":
                unreal = (
                    row["Close"]
                    - pos["entry"]
                ) * pos["qty"]
            else:
                unreal = (
                    pos["entry"]
                    - row["Close"]
                ) * pos["qty"]

            unreal -= (
                row["Close"]
                * pos["qty"]
                * FEE_RATE
            )

            mtm += unreal

        # Closed PnL is accumulated directly into capital.
        closed_capital = (
            INITIAL_CAPITAL
            + sum(
                t["pnl"]
                for t in trades
            )
        )

        equity = (
            closed_capital
            + mtm
        )

        equity_curve.append({
            "timestamp": ts,
            "closed_capital": closed_capital,
            "equity": equity,
            "open_positions": len(
                positions
            ),
        })

    return (
        trades,
        equity_curve,
        diagnostics,
    )


def max_loss_streak(trades):
    current = 0
    maximum = 0
    distribution = Counter()

    for trade in trades:
        if trade["pnl"] < 0:
            current += 1
        else:
            if current > 0:
                distribution[
                    current
                ] += 1

            maximum = max(
                maximum,
                current,
            )

            current = 0

    if current > 0:
        distribution[
            current
        ] += 1

        maximum = max(
            maximum,
            current,
        )

    return (
        maximum,
        distribution,
    )


def calculate_drawdown(equity_curve):
    if not equity_curve:
        return (
            0.0,
            0.0,
        )

    equity = pd.Series(
        [
            item["equity"]
            for item in equity_curve
        ],
        dtype=float,
    )

    peak = equity.cummax()
    drawdown = equity - peak

    max_dd_dollar = float(
        drawdown.min()
    )

    if max_dd_dollar >= 0:
        return (
            0.0,
            0.0,
        )

    trough_idx = drawdown.idxmin()
    peak_value = float(
        peak.loc[trough_idx]
    )

    max_dd_pct = (
        max_dd_dollar
        / peak_value
        * 100.0
        if peak_value > 0
        else 0.0
    )

    return (
        max_dd_dollar,
        max_dd_pct,
    )


def summarize(
    data,
    trades,
    equity_curve,
    diagnostics,
):
    total = len(trades)

    wins = sum(
        t["pnl"] > 0
        for t in trades
    )

    losses = sum(
        t["pnl"] <= 0
        for t in trades
    )

    pnl = sum(
        t["pnl"]
        for t in trades
    )

    net_r = sum(
        t["r"]
        for t in trades
    )

    final_capital = (
        INITIAL_CAPITAL
        + pnl
    )

    return_pct = (
        (
            final_capital
            / INITIAL_CAPITAL
        )
        - 1.0
    ) * 100.0

    win_rate = (
        wins / total * 100.0
        if total
        else 0.0
    )

    max_dd_dollar, max_dd_pct = (
        calculate_drawdown(
            equity_curve
        )
    )

    max_streak, streak_distribution = (
        max_loss_streak(trades)
    )

    print("\n" + "=" * 72)
    print("HUNTER-V79 RESULTS")
    print("=" * 72)

    print(
        f"Valid Symbols : "
        f"{len(data)}/{len(SYMBOLS)}"
    )
    print(
        f"Total Trades  : {total}"
    )
    print(
        f"Wins          : {wins}"
    )
    print(
        f"Losses        : {losses}"
    )
    print(
        f"Win Rate      : "
        f"{win_rate:.2f}%"
    )
    print(
        f"Net R         : "
        f"{net_r:.2f}R"
    )
    print(
        f"Net PnL       : "
        f"${pnl:,.2f}"
    )
    print(
        f"Final Capital : "
        f"${final_capital:,.2f}"
    )
    print(
        f"Return        : "
        f"{return_pct:.2f}%"
    )
    print(
        f"Max Drawdown  : "
        f"${max_dd_dollar:,.2f} "
        f"({max_dd_pct:.2f}%)"
    )
    print(
        f"Max Loss Streak: "
        f"{max_streak}"
    )

    print("\n" + "-" * 72)
    print("DIRECTION")
    print("-" * 72)

    for side in [
        "LONG",
        "SHORT",
    ]:
        side_trades = [
            t
            for t in trades
            if t["side"] == side
        ]

        side_wins = sum(
            t["pnl"] > 0
            for t in side_trades
        )

        side_pnl = sum(
            t["pnl"]
            for t in side_trades
        )

        side_r = sum(
            t["r"]
            for t in side_trades
        )

        side_wr = (
            side_wins
            / len(side_trades)
            * 100.0
            if side_trades
            else 0.0
        )

        print(
            f"{side:5s} | "
            f"Trades={len(side_trades):4d} | "
            f"WR={side_wr:6.2f}% | "
            f"PnL=${side_pnl:10,.2f} | "
            f"R={side_r:8.2f}"
        )

    print("\n" + "-" * 72)
    print("EXIT REASONS")
    print("-" * 72)

    for reason in [
        "STOP",
        "TIMEOUT",
    ]:
        reason_trades = [
            t
            for t in trades
            if t["exit_reason"]
            == reason
        ]

        reason_wins = sum(
            t["pnl"] > 0
            for t in reason_trades
        )

        reason_pnl = sum(
            t["pnl"]
            for t in reason_trades
        )

        reason_r = sum(
            t["r"]
            for t in reason_trades
        )

        reason_wr = (
            reason_wins
            / len(reason_trades)
            * 100.0
            if reason_trades
            else 0.0
        )

        print(
            f"{reason:8s} | "
            f"Trades={len(reason_trades):4d} | "
            f"WR={reason_wr:6.2f}% | "
            f"PnL=${reason_pnl:10,.2f} | "
            f"R={reason_r:8.2f}"
        )

    print("\n" + "-" * 72)
    print("PER SYMBOL")
    print("-" * 72)

    for name in SYMBOLS:
        symbol_trades = [
            t
            for t in trades
            if t["symbol"] == name
        ]

        if not symbol_trades:
            print(
                f"{name:8s} | "
                f"Trades={0:3d} | "
                f"WR={0.00:6.2f}% | "
                f"PnL=${0.00:10,.2f} | "
                f"R={0.00:8.2f} | "
                f"MaxLS=0"
            )
            continue

        symbol_wins = sum(
            t["pnl"] > 0
            for t in symbol_trades
        )

        symbol_pnl = sum(
            t["pnl"]
            for t in symbol_trades
        )

        symbol_r = sum(
            t["r"]
            for t in symbol_trades
        )

        symbol_wr = (
            symbol_wins
            / len(symbol_trades)
            * 100.0
        )

        local_streak, _ = (
            max_loss_streak(
                symbol_trades
            )
        )

        print(
            f"{name:8s} | "
            f"Trades={len(symbol_trades):3d} | "
            f"WR={symbol_wr:6.2f}% | "
            f"PnL=${symbol_pnl:10,.2f} | "
            f"R={symbol_r:8.2f} | "
            f"MaxLS={local_streak}"
        )

    print("\n" + "-" * 72)
    print("LOSS STREAK DISTRIBUTION")
    print("-" * 72)

    if streak_distribution:
        for length in sorted(
            streak_distribution
        ):
            print(
                f"{length:2d} loss: "
                f"{streak_distribution[length]} times"
            )
    else:
        print("No losing streaks.")

    print("\n" + "-" * 72)
    print("DIAGNOSTICS")
    print("-" * 72)

    for key in sorted(diagnostics):
        print(
            f"{key:32s}: "
            f"{diagnostics[key]}"
        )

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_r": net_r,
        "pnl": pnl,
        "final_capital": final_capital,
        "return_pct": return_pct,
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "max_loss_streak": max_streak,
    }


def save_outputs(
    trades,
    equity_curve,
    diagnostics,
    summary,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    trades_df = pd.DataFrame(
        trades
    )

    if not trades_df.empty:
        trades_df[
            "entry_ts"
        ] = pd.to_datetime(
            trades_df["entry_ts"],
            utc=True,
        )

        trades_df[
            "exit_ts"
        ] = pd.to_datetime(
            trades_df["exit_ts"],
            utc=True,
        )

        trades_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "trades.csv",
            ),
            index=False,
        )

    equity_df = pd.DataFrame(
        equity_curve
    )

    if not equity_df.empty:
        equity_df[
            "timestamp"
        ] = pd.to_datetime(
            equity_df["timestamp"],
            utc=True,
        )

        equity_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "equity_curve.csv",
            ),
            index=False,
        )

    diagnostics_df = pd.DataFrame(
        sorted(
            diagnostics.items()
        ),
        columns=[
            "metric",
            "value",
        ],
    )

    diagnostics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostics.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "summary.csv",
        ),
        index=False,
    )

    print(
        f"\nCSV: {OUTPUT_DIR}/"
    )


def main():
    data = load_data()

    if not data:
        raise RuntimeError(
            "No valid symbols loaded."
        )

    trades, equity_curve, diagnostics = (
        run_backtest(data)
    )

    summary = summarize(
        data,
        trades,
        equity_curve,
        diagnostics,
    )

    save_outputs(
        trades,
        equity_curve,
        diagnostics,
        summary,
    )


if __name__ == "__main__":
    main()
