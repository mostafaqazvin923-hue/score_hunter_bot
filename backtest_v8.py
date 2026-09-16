import os
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V74-LSP - ULTRA OPTIMIZED LOSS-STREAK VERSION
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True})

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
    "NEAR", "OP", "HYPE", "HBAR", "AVAX", "SUI", "PENDLE", "TIA",
    "FET", "SEI", "ARB", "DOT", "ETC", "SHIB", "STX", "RUNE",
    "MKR", "APT", "LTC", "AR", "IMX", "PEPE", "BONK",
}

SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

# تغییرات حیاتی برای کنترل Loss Streak بدون افت شدید سود
MAX_POSITIONS = 3            # کاهش پوزیشن‌های هم‌زمان برای افت ریسک سبد

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8
TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0

# LSP SETTINGS
LSP_ACTIVATE_AT = 2
LSP_PENALTY_1 = 0.75
LSP_PENALTY_2 = 1.50
LSP_PENALTY_3_PLUS = 2.25
GLOBAL_STREAK_PENALTY = 0.35

OUTPUT_DIR = "hunter_v74_lsp_output"

# ============================================================
# LSP3 — ULTRA STRICT LOSS-STREAK CIRCUIT BREAKER
# ============================================================
LSP3_TRIGGER_STREAK = 1      # فعال‌سازی به محض اولین ضرر جهت‌دار
LSP3_COOLDOWN_CANDLES = 12   # استراحت طولانی‌تر برای پاکسازی روند نزولی کاذب

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V74-LSP - ULTRA LOSS-STREAK PROTECTION")
print("=" * 68)


# ============================================================
# DATA
# ============================================================

processed_data = {}


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
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    df.dropna(inplace=True)
    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True,
    )
    df.sort_values(
        "Date",
        inplace=True,
    )
    df.reset_index(
        drop=True,
        inplace=True,
    )

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(
            df.iloc[-1]["Date"].timestamp() * 1000
        )

        if (
            last_ms
            + 4 * 60 * 60 * 1000
            > now_ms
        ):
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()

    if (
        not deltas.empty
        and deltas.max()
        > pd.Timedelta(
            hours=4,
            minutes=10,
        )
    ):
        return None

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = np.abs(
        df["High"]
        - df["Close"].shift(1)
    )

    tr3 = np.abs(
        df["Low"]
        - df["Close"].shift(1)
    )

    df["ATR"] = (
        pd.concat(
            [tr1, tr2, tr3],
            axis=1,
        )
        .max(axis=1)
        .rolling(ATR_PERIOD)
        .mean()
    )

    df["EMA20"] = df[
        "Close"
    ].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = df[
        "Close"
    ].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["EMA200"] = df[
        "Close"
    ].ewm(
        span=200,
        adjust=False,
    ).mean()

    df["Mom_Short"] = (
        df["Close"]
        - df["Close"].shift(10)
    ) / df["Close"].shift(10)

    df["Mom_Long"] = (
        df["Close"]
        - df["Close"].shift(30)
    ) / df["Close"].shift(30)

    df.set_index(
        "Date",
        inplace=True,
    )

    return df


for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(
        lbank_symbol
    )

    if df4h is not None:
        processed_data[
            symbol
        ] = df4h

print(
    f"Valid symbols: "
    f"{len(processed_data)} / "
    f"{len(SYMBOLS)}"
)


# ============================================================
# COMMON HELPERS
# ============================================================

def get_all_timestamps(data):
    return sorted(
        {
            ts
            for df in data.values()
            for ts in df.index
        }
    )


def build_valid_candidates(
    processed_data,
    ts,
    active_positions,
    market_bull,
    allow_longs,
    allow_shorts,
):
    current_scores = {}

    for symbol, df in processed_data.items():
        if ts not in df.index:
            continue

        val = df.loc[
            ts,
            "Mom_Long",
        ]

        if not np.isnan(val):
            current_scores[
                symbol
            ] = val

    if not current_scores:
        return []

    ranked_symbols = sorted(
        current_scores.keys(),
        key=lambda x: float(
            current_scores[x]
        ),
        reverse=market_bull,
    )

    candidates = []

    for symbol in ranked_symbols:
        if symbol in active_positions:
            continue

        df = processed_data[
            symbol
        ]

        if ts not in df.index:
            continue

        i = df.index.get_loc(ts)

        if i < EMA_WARMUP + 1:
            continue

        c4h = df.iloc[i]
        prev_c = df.iloc[i - 1]

        if market_bull:
            if not allow_longs:
                continue

            regime_ok = (
                c4h["Close"]
                > c4h["EMA20"]
                and c4h["EMA20"]
                > c4h["EMA50"]
                and c4h["Close"]
                > c4h["EMA200"]
            )

            pullback_ok = (
                prev_c["Low"]
                <= prev_c["EMA20"]
                * 1.015
            )

            valid_signal = (
                regime_ok
                and pullback_ok
                and (
                    c4h["Mom_Short"]
                    > 0.012
                )
                and (
                    c4h["Mom_Long"]
                    > 0.035
                )
            )

            side = "LONG"

        else:
            if not allow_shorts:
                continue

            regime_ok = (
                c4h["Close"]
                < c4h["EMA20"]
                and c4h["EMA20"]
                < c4h["EMA50"]
                and c4h["Close"]
                < c4h["EMA200"]
            )

            pullback_ok = (
                prev_c["High"]
                >= prev_c["EMA20"]
                * 0.985
            )

            valid_signal = (
                regime_ok
                and pullback_ok
                and (
                    c4h["Mom_Short"]
                    < -0.012
                )
                and (
                    c4h["Mom_Long"]
                    < -0.035
                )
            )

            side = "SHORT"

        if not valid_signal:
            continue

        entry_price = (
            c4h["Open"]
            * (
                1 + SLIPPAGE
            )
            if side == "LONG"
            else
            c4h["Open"]
            * (
                1 - SLIPPAGE
            )
        )

        initial_sl = (
            entry_price
            - INITIAL_ATR_MULTIPLIER
            * c4h["ATR"]
            if side == "LONG"
            else
            entry_price
            + INITIAL_ATR_MULTIPLIER
            * c4h["ATR"]
        )

        initial_risk = abs(
            entry_price
            - initial_sl
        )

        sl_dist_pct = (
            initial_risk
            / entry_price
        )

        if not (
            0.01
            <= sl_dist_pct
            <= 0.04
        ):
            continue

        candidates.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price": float(
                    entry_price
                ),
                "initial_sl": float(
                    initial_sl
                ),
                "initial_risk": float(
                    initial_risk
                ),
                "entry_index": int(i),
                "mom_long": float(
                    c4h["Mom_Long"]
                ),
            }
        )

    return candidates


# ============================================================
# LOSS-STREAK SHIELD
# ============================================================

def lsp_penalty(
    symbol,
    side,
    symbol_side_losses,
    global_loss_streak,
):
    streak = symbol_side_losses.get(
        (symbol, side),
        0,
    )

    if streak <= 0:
        penalty = 0.0
    elif streak == 1:
        penalty = LSP_PENALTY_1
    elif streak == 2:
        penalty = LSP_PENALTY_2
    else:
        penalty = LSP_PENALTY_3_PLUS

    if (
        global_loss_streak
        >= LSP_ACTIVATE_AT
    ):
        penalty += GLOBAL_STREAK_PENALTY

    return penalty


def select_candidates_lsp(
    candidates,
    slots,
    symbol_side_losses,
    global_loss_streak,
):
    if slots <= 0 or not candidates:
        return []

    ranked = []

    for original_rank, candidate in enumerate(
        candidates
    ):
        penalty = lsp_penalty(
            candidate["symbol"],
            candidate["side"],
            symbol_side_losses,
            global_loss_streak,
        )

        base_score = (
            len(candidates)
            - original_rank
        )

        adjusted_score = (
            float(base_score)
            - penalty
        )

        ranked.append(
            (
                adjusted_score,
                original_rank,
                candidate,
            )
        )

    ranked.sort(
        key=lambda x: (
            x[0],
            -x[1],
        ),
        reverse=True,
    )

    selected = [
        item[2]
        for item in ranked[:slots]
    ]

    return selected


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    processed_data,
    use_lsp=False,
    use_lsp3=False,
    gate_mode=None,
    use_firewall=False,
    use_single_loss_crowd=False,
):
    all_timestamps = get_all_timestamps(
        processed_data
    )

    active_positions = {}
    all_trades = []

    symbol_side_losses = defaultdict(int)
    global_loss_streak = 0

    direction_loss_streak = {"LONG": 0, "SHORT": 0}
    direction_cooldown = {"LONG": 0, "SHORT": 0}
    lsp3_trigger_log = []

    firewall_loss_events = {"LONG": 0, "SHORT": 0}
    firewall_locked = {"LONG": False, "SHORT": False}

    SINGLE_LOSS_CROWD_MIN_OPEN = 2
    diagnostics = Counter()
    equity_curve = []

    for ts in all_timestamps:
        symbols_to_close = []

        for symbol, pos in list(
            active_positions.items()
        ):
            df = processed_data[
                symbol
            ]

            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if (
                    c4h["High"]
                    > pos["highest_price"]
                ):
                    pos[
                        "highest_price"
                    ] = c4h["High"]

                    new_trailing_sl = (
                        pos[
                            "highest_price"
                        ]
                        - TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if (
                        new_trailing_sl
                        > pos["stop_loss"]
                    ):
                        pos[
                            "stop_loss"
                        ] = new_trailing_sl

                hit_sl = (
                    c4h["Low"]
                    <= pos["stop_loss"]
                )

            else:
                if (
                    c4h["Low"]
                    < pos["lowest_price"]
                ):
                    pos[
                        "lowest_price"
                    ] = c4h["Low"]

                    new_trailing_sl = (
                        pos[
                            "lowest_price"
                        ]
                        + TRAILING_ATR_MULTIPLIER
                        * c4h["ATR"]
                    )

                    if (
                        new_trailing_sl
                        < pos["stop_loss"]
                    ):
                        pos[
                            "stop_loss"
                        ] = new_trailing_sl

                hit_sl = (
                    c4h["High"]
                    >= pos["stop_loss"]
                )

            curr_i = df.index.get_loc(
                ts
            )

            candles_held = (
                curr_i
                - pos["entry_index"]
            )

            is_timeout = (
                candles_held
                >= TIMEOUT_CANDLES
            )

            if not (
                hit_sl
                or is_timeout
            ):
                continue

            initial_risk = (
                pos["initial_risk"]
            )

            if pos["side"] == "LONG":
                exit_p = (
                    min(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        exit_p
                        - pos[
                            "entry_price"
                        ]
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    exit_p
                    - pos[
                        "entry_price"
                    ]
                ) / pos[
                    "entry_price"
                ]

            else:
                exit_p = (
                    max(
                        pos["stop_loss"],
                        c4h["Open"],
                    )
                    if hit_sl
                    else c4h["Close"]
                )

                r_real = (
                    (
                        pos[
                            "entry_price"
                        ]
                        - exit_p
                    )
                    / initial_risk
                ) - (
                    FEE_RATE * 2
                )

                price_return_pct = (
                    pos[
                        "entry_price"
                    ]
                    - exit_p
                ) / pos[
                    "entry_price"
                ]

            outcome = (
                "WIN"
                if r_real > 0
                else "LOSS"
            )

            position_notional = (
                TRADE_MARGIN
                * LEVERAGE
            )

            dollar_pnl = (
                position_notional
                * price_return_pct
            ) - (
                position_notional
                * FEE_RATE
                * 2
            )

            if outcome == "LOSS":
                global_loss_streak += 1
                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] += 1
            else:
                global_loss_streak = 0
                symbol_side_losses[
                    (
                        symbol,
                        pos["side"],
                    )
                ] = 0

            all_trades.append(
                {
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": r_real,
                    "Dollar_PnL": dollar_pnl,
                    "ExitOrder": len(
                        all_trades
                    ),
                    "EntryTimestamp": pos.get("entry_timestamp", pd.NaT),
                    "EntryPrice": pos["entry_price"],
                    "InitialStop": pos["initial_stop"],
                    "ExitPrice": exit_p,
                    "LSP_Active": int(
                        use_lsp
                    ),
                    "LSP3_Active": int(
                        use_lsp3
                    ),
                }
            )

            symbols_to_close.append(
                symbol
            )

        for sym in symbols_to_close:
            del active_positions[
                sym
            ]

        if use_lsp3:
            for side in ("LONG", "SHORT"):
                side_outcomes = [
                    t["Outcome"]
                    for t in all_trades
                    if t["Timestamp"] == ts and t["Side"] == side
                ]

                if not side_outcomes:
                    continue

                if "WIN" in side_outcomes:
                    direction_loss_streak[side] = 0
                    direction_cooldown[side] = 0
                elif "LOSS" in side_outcomes:
                    direction_loss_streak[side] += 1

                    if direction_loss_streak[side] >= LSP3_TRIGGER_STREAK:
                        direction_cooldown[side] = LSP3_COOLDOWN_CANDLES
                        lsp3_trigger_log.append({
                            "Timestamp": ts,
                            "Side": side,
                            "LossEventStreak": direction_loss_streak[side],
                            "CooldownCandles": LSP3_COOLDOWN_CANDLES,
                        })

        if use_firewall:
            for side in ("LONG", "SHORT"):
                side_outcomes = [
                    tr["Outcome"] for tr in all_trades
                    if tr["Timestamp"] == ts and tr["Side"] == side
                ]
                if side_outcomes:
                    if "WIN" in side_outcomes:
                        firewall_loss_events[side] = 0
                        firewall_locked[side] = False
                    elif "LOSS" in side_outcomes:
                        firewall_loss_events[side] += 1
                        if firewall_loss_events[side] >= 1:
                            firewall_locked[side] = True

                if firewall_locked[side]:
                    still_open = any(
                        p["side"] == side for p in active_positions.values()
                    )
                    if not still_open:
                        firewall_locked[side] = False
                        firewall_loss_events[side] = 0

        market_bull = True

        if (
            "BTC"
            in processed_data
            and ts
            in processed_data[
                "BTC"
            ].index
        ):
            btc_c = processed_data[
                "BTC"
            ].loc[ts]

            market_bull = (
                btc_c["Close"]
                > btc_c["EMA200"]
            )

        bullish_count = 0
        total_active_syms = 0

        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue

            total_active_syms += 1

            if (
                df.loc[
                    ts,
                    "Close",
                ]
                > df.loc[
                    ts,
                    "EMA200",
                ]
            ):
                bullish_count += 1

        market_breadth_ratio = (
            bullish_count
            / total_active_syms
            if total_active_syms > 0
            else 0.5
        )

        allow_longs = (
            market_breadth_ratio
            >= 0.35
        )

        allow_shorts = (
            market_breadth_ratio
            <= 0.65
        )

        candidates = build_valid_candidates(
            processed_data,
            ts,
            active_positions,
            market_bull,
            allow_longs,
            allow_shorts,
        )

        diagnostics[
            "valid_candidates"
        ] += len(candidates)

        if gate_mode and candidates:
            btc = processed_data.get("BTC")
            if gate_mode == "A":
                breadth_limit, ema50_limit, min_open = 0.10, -1.80, 2
            elif gate_mode == "B":
                breadth_limit, ema50_limit, min_open = 0.15, -1.80, 2
            elif gate_mode == "C":
                breadth_limit, ema50_limit, min_open = 0.20, -1.50, 2
            else:
                breadth_limit, ema50_limit, min_open = 0.15, -1.80, 2

            kept = []
            for cand in candidates:
                block = False
                if cand["side"] == "SHORT":
                    d = processed_data[cand["symbol"]]
                    ci = d.index.get_loc(ts)
                    cc = d.iloc[ci]
                    ema50_atr = float((cc["Close"] - cc["EMA50"]) / cc["ATR"]) if cc["ATR"] else np.nan
                    open_short = sum(v["side"] == "SHORT" for v in active_positions.values())
                    if (market_breadth_ratio <= breadth_limit and
                        ema50_atr <= ema50_limit and
                        open_short >= min_open):
                        block = True
                if block:
                    diagnostics["gate_blocked"] += 1
                else:
                    kept.append(cand)
            candidates = kept

        if not candidates:
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            continue

        if use_lsp3:
            blocked_sides = {
                side for side in ("LONG", "SHORT")
                if direction_cooldown[side] > 0
            }

            if blocked_sides:
                candidates = [c for c in candidates if c["side"] not in blocked_sides]

            if not candidates:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
                continue

        if use_single_loss_crowd and candidates:
            kept = []
            for c in candidates:
                side = c["side"]
                open_same_side = sum(
                    p["side"] == side for p in active_positions.values()
                )
                crowd_block = (
                    firewall_loss_events.get(side, 0) >= 1
                    and open_same_side >= SINGLE_LOSS_CROWD_MIN_OPEN
                )
                if not crowd_block:
                    kept.append(c)
            candidates = kept

        if use_firewall and candidates:
            kept = []
            for c in candidates:
                if not firewall_locked.get(c["side"], False):
                    kept.append(c)
            candidates = kept

        slots = (
            MAX_POSITIONS
            - len(active_positions)
        )

        if slots <= 0:
            if use_lsp3:
                for _side in ("LONG", "SHORT"):
                    if direction_cooldown[_side] > 0:
                        direction_cooldown[_side] -= 1
            continue

        if use_lsp:
            selected = select_candidates_lsp(
                candidates,
                slots,
                symbol_side_losses,
                global_loss_streak,
            )
        else:
            selected = candidates[
                :slots
            ]

        for candidate in selected:
            symbol = candidate[
                "symbol"
            ]

            side = candidate[
                "side"
            ]

            active_positions[
                symbol
            ] = {
                "side": side,
                "entry_price": candidate[
                    "entry_price"
                ],
                "initial_stop": candidate[
                    "initial_sl"
                ],
                "entry_timestamp": ts,
                "stop_loss": candidate[
                    "initial_sl"
                ],
                "highest_price": candidate[
                    "entry_price"
                ],
                "lowest_price": candidate[
                    "entry_price"
                ],
                "initial_risk": candidate[
                    "initial_risk"
                ],
                "entry_index": candidate[
                    "entry_index"
                ],
            }

        closed_pnl = sum(
            trade["Dollar_PnL"]
            for trade in all_trades
        )

        unrealized = 0.0

        for pos_symbol, pos in (
            active_positions.items()
        ):
            df = processed_data[
                pos_symbol
            ]

            if ts not in df.index:
                continue

            close_price = df.loc[
                ts,
                "Close",
            ]

            if pos["side"] == "LONG":
                unrealized += (
                    close_price
                    - pos["entry_price"]
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )

            else:
                unrealized += (
                    pos["entry_price"]
                    - close_price
                ) * (
                    TRADE_MARGIN
                    * LEVERAGE
                    / pos["entry_price"]
                )

        equity_curve.append(
            {
                "Timestamp": ts,
                "Equity": (
                    INITIAL_CAPITAL
                    + closed_pnl
                    + unrealized
                ),
            }
        )

    return (
        pd.DataFrame(all_trades),
        pd.DataFrame(equity_curve),
        diagnostics,
    )


# ============================================================
# REPORTING
# ============================================================

def calculate_loss_streaks(
    trades_df
):
    if trades_df.empty:
        return 0, []

    ordered = trades_df.sort_values(
        [
            "Timestamp",
            "ExitOrder",
        ],
        kind="stable",
    )

    current = 0
    maximum = 0
    sequences = []

    for outcome in ordered[
        "Outcome"
    ]:
        if outcome == "LOSS":
            current += 1
            maximum = max(
                maximum,
                current,
            )
        else:
            if current > 0:
                sequences.append(
                    current
                )
            current = 0

    if current > 0:
        sequences.append(
            current
        )

    return (
        maximum,
        sequences,
    )


def calculate_drawdown(
    equity_df
):
    if equity_df.empty:
        return 0.0, 0.0

    equity = equity_df[
        "Equity"
    ].astype(float)

    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())

    if max_dd >= 0:
        return 0.0, 0.0

    peak_at_trough = float(
        peak.loc[
            dd.idxmin()
        ]
    )

    max_dd_pct = (
        max_dd
        / peak_at_trough
        * 100.0
        if peak_at_trough > 0
        else 0.0
    )

    return (
        max_dd,
        max_dd_pct,
    )


def report(
    name,
    trades_df,
    equity_df,
):
    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)

    if trades_df.empty:
        print("No trades.")
        return {}

    trades_df = trades_df.sort_values(
        [
            "Timestamp",
            "ExitOrder",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    total = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = total - wins
    wr = (wins / total * 100.0) if total > 0 else 0.0

    net_r = float(trades_df["Return"].sum())
    pnl = float(trades_df["Dollar_PnL"].sum())
    final_capital = INITIAL_CAPITAL + pnl
    return_pct = (final_capital / INITIAL_CAPITAL - 1.0) * 100.0

    max_ls, _ = calculate_loss_streaks(trades_df)
    dd_dollar, dd_pct = calculate_drawdown(equity_df)

    print(f"Trades        : {total}")
    print(f"Wins          : {wins}")
    print(f"Losses        : {losses}")
    print(f"Win Rate      : {wr:.2f}%")
    print(f"Net R         : {net_r:.2f}R")
    print(f"Net PnL       : ${pnl:,.2f}")
    print(f"Final Capital : ${final_capital:,.2f}")
    print(f"Return        : {return_pct:.2f}%")
    print(f"Max Drawdown  : ${dd_dollar:,.2f} ({dd_pct:.2f}%)")
    print(f"Max Loss Streak: {max_ls}")

    return {
        "trades": total, "wins": wins, "losses": losses, "wr": wr,
        "net_r": net_r, "pnl": pnl, "final_capital": final_capital,
        "return_pct": return_pct, "max_dd": dd_dollar, "max_dd_pct": dd_pct,
        "max_loss_streak": max_ls,
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    trades_df, equity_df, diagnostics = run_backtest(
        processed_data,
        use_lsp=True,
        use_lsp3=True,
        gate_mode="B",
        use_firewall=True,
        use_single_loss_crowd=True
    )

    perf = report("HUNTER-V74 ULTRA OPTIMIZED REPORT", trades_df, equity_df)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "optimized_trades.csv"), index=False)
    if not equity_df.empty:
        equity_df.to_csv(os.path.join(OUTPUT_DIR, "optimized_equity.csv"), index=False)
        
    print(f"\nOptimization Complete. Results saved in '{OUTPUT_DIR}/'.")
