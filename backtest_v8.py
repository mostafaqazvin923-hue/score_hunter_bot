
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
# HUNTER-V74-LSP
#
# Golden Core = EXACT V74 signal / execution logic
# ONLY ADDITION = Loss-Streak Protection in candidate priority
#
# IMPORTANT:
# - No new indicator filter
# - No new coin
# - No change to leverage / margin
# - No change to SL / trailing / timeout
# - No hard "stop trading" after losses
# - Valid signal count is preserved; LSP only changes priority
#   when the 5-position portfolio is full.
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

MAX_POSITIONS = 5

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

# ------------------------------------------------------------
# LSP SETTINGS
#
# The shield is deliberately SOFT.
# It never rejects an otherwise valid V74 signal.
# It only changes priority while portfolio slots are limited.
# ------------------------------------------------------------

# Global loss streak before the shield becomes active.
LSP_ACTIVATE_AT = 2

# Penalty by consecutive losses of the same symbol + side.
# This is used only for ranking candidates.
LSP_PENALTY_1 = 0.75
LSP_PENALTY_2 = 1.50
LSP_PENALTY_3_PLUS = 2.25

# Extra penalty when the WHOLE portfolio is in a loss streak.
# Kept deliberately small so V74's signal stream is preserved.
GLOBAL_STREAK_PENALTY = 0.35

OUTPUT_DIR = "hunter_v74_lsp_output"

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V74-LSP")
print("Golden Core + Soft Loss-Streak Protection")
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

    # Remove incomplete final candle.
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
    """
    EXACT V74 signal construction.

    This function intentionally keeps the original V74 rules.
    """

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

    # EXACT V74 ranking.
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
    """
    Soft penalty only.

    The signal remains valid.
    No signal is rejected here.
    """

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
    """
    LSP selection.

    Crucial behavior:
    - Does NOT remove valid candidates.
    - Does NOT impose a direction cap.
    - Does NOT reduce MAX_POSITIONS.
    - Only changes which valid candidate gets a scarce slot.
    """

    if slots <= 0:
        return []

    if not candidates:
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

        # Primary criterion remains the ORIGINAL V74
        # Momentum ranking. The penalty is only a small
        # secondary adjustment when candidates compete.
        #
        # Convert original rank to a stable quality number:
        # earlier V74 candidates receive a higher base score.
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
):
    all_timestamps = get_all_timestamps(
        processed_data
    )

    active_positions = {}
    all_trades = []

    # Consecutive losses for exact symbol + side.
    symbol_side_losses = defaultdict(int)

    # Consecutive losses across portfolio exit order.
    global_loss_streak = 0

    diagnostics = Counter()

    equity_curve = []

    for ts in all_timestamps:
        symbols_to_close = []

        # --------------------------------------------------------
        # 1. EXACT V74 POSITION MANAGEMENT
        # --------------------------------------------------------
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
                    "LSP_Active": int(
                        use_lsp
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

        # --------------------------------------------------------
        # 2. EXACT V74 MARKET REGIME
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # 3. EXACT V74 VALID CANDIDATES
        # --------------------------------------------------------
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

        if not candidates:
            continue

        slots = (
            MAX_POSITIONS
            - len(active_positions)
        )

        if slots <= 0:
            diagnostics[
                "portfolio_full"
            ] += 1
            continue

        # --------------------------------------------------------
        # 4. SELECT
        #
        # BASELINE:
        #   first candidates exactly as V74 ranking gives them.
        #
        # LSP:
        #   same candidates, but soft penalty can reorder them.
        # --------------------------------------------------------
        if use_lsp:
            selected = select_candidates_lsp(
                candidates,
                slots,
                symbol_side_losses,
                global_loss_streak,
            )

            diagnostics[
                "lsp_selection_events"
            ] += 1

            if (
                global_loss_streak
                >= LSP_ACTIVATE_AT
            ):
                diagnostics[
                    "lsp_active_events"
                ] += 1

        else:
            selected = candidates[
                :slots
            ]

        diagnostics[
            "selected_entries"
        ] += len(selected)

        # --------------------------------------------------------
        # 5. EXACT V74 ENTRY
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # 6. Simple mark-to-market curve.
        # Does not affect trading logic.
        # --------------------------------------------------------
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

    max_dd = float(
        dd.min()
    )

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
    diagnostics,
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

    total = len(
        trades_df
    )

    wins = int(
        (
            trades_df["Outcome"]
            == "WIN"
        ).sum()
    )

    losses = total - wins

    wr = (
        wins
        / total
        * 100.0
    )

    net_r = float(
        trades_df[
            "Return"
        ].sum()
    )

    pnl = float(
        trades_df[
            "Dollar_PnL"
        ].sum()
    )

    final_capital = (
        INITIAL_CAPITAL
        + pnl
    )

    return_pct = (
        final_capital
        / INITIAL_CAPITAL
        - 1.0
    ) * 100.0

    max_ls, sequences = (
        calculate_loss_streaks(
            trades_df
        )
    )

    dd_dollar, dd_pct = (
        calculate_drawdown(
            equity_df
        )
    )

    print(
        f"Trades        : {total}"
    )
    print(
        f"Wins          : {wins}"
    )
    print(
        f"Losses        : {losses}"
    )
    print(
        f"Win Rate      : {wr:.2f}%"
    )
    print(
        f"Net R         : {net_r:.2f}R"
    )
    print(
        f"Net PnL       : ${pnl:,.2f}"
    )
    print(
        f"Final Capital : ${final_capital:,.2f}"
    )
    print(
        f"Return        : {return_pct:.2f}%"
    )
    print(
        f"Max Drawdown  : "
        f"${dd_dollar:,.2f} "
        f"({dd_pct:.2f}%)"
    )
    print(
        f"Max Loss Streak: {max_ls}"
    )

    print("\nDIRECTION")

    for side in [
        "LONG",
        "SHORT",
    ]:
        sdf = trades_df[
            trades_df["Side"]
            == side
        ]

        side_total = len(sdf)

        side_wins = int(
            (
                sdf["Outcome"]
                == "WIN"
            ).sum()
        )

        side_wr = (
            side_wins
            / side_total
            * 100.0
            if side_total
            else 0.0
        )

        side_pnl = float(
            sdf[
                "Dollar_PnL"
            ].sum()
        )

        print(
            f"{side:5s} | "
            f"Trades={side_total:4d} | "
            f"WR={side_wr:6.2f}% | "
            f"PnL=${side_pnl:10,.2f}"
        )

    print("\nLOSS STREAK DISTRIBUTION")

    distribution = Counter(
        sequences
    )

    if distribution:
        for length in sorted(
            distribution
        ):
            print(
                f"{length:2d} loss: "
                f"{distribution[length]} times"
            )
    else:
        print(
            "No loss streaks."
        )

    print("\nPER SYMBOL")

    for symbol in SYMBOLS:
        sdf = trades_df[
            trades_df["Symbol"]
            == symbol
        ]

        if sdf.empty:
            continue

        symbol_total = len(sdf)

        symbol_wins = int(
            (
                sdf["Outcome"]
                == "WIN"
            ).sum()
        )

        symbol_wr = (
            symbol_wins
            / symbol_total
            * 100.0
        )

        symbol_pnl = float(
            sdf[
                "Dollar_PnL"
            ].sum()
        )

        symbol_max_ls, _ = (
            calculate_loss_streaks(
                sdf
            )
        )

        print(
            f"{symbol:8s} | "
            f"Trades={symbol_total:3d} | "
            f"WR={symbol_wr:6.2f}% | "
            f"PnL=${symbol_pnl:10,.2f} | "
            f"MaxLS={symbol_max_ls}"
        )

    print("\nDIAGNOSTICS")

    for key in sorted(
        diagnostics
    ):
        print(
            f"{key:28s}: "
            f"{diagnostics[key]}"
        )

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "net_r": net_r,
        "pnl": pnl,
        "final_capital": final_capital,
        "return_pct": return_pct,
        "max_dd": dd_dollar,
        "max_dd_pct": dd_pct,
        "max_loss_streak": max_ls,
    }


def save_csvs(
    baseline_trades,
    baseline_equity,
    lsp_trades,
    lsp_equity,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if not baseline_trades.empty:
        baseline_trades.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "baseline_v74_trades.csv",
            ),
            index=False,
        )

    if not lsp_trades.empty:
        lsp_trades.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "v74_lsp_trades.csv",
            ),
            index=False,
        )

    if not baseline_equity.empty:
        baseline_equity.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "baseline_v74_equity.csv",
            ),
            index=False,
        )

    if not lsp_equity.empty:
        lsp_equity.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "v74_lsp_equity.csv",
            ),
            index=False,
        )

    print(
        f"\nCSV output: "
        f"{OUTPUT_DIR}/"
    )


def print_comparison(
    baseline,
    lsp,
):
    print("\n" + "=" * 68)
    print("V74 BASELINE vs V74-LSP")
    print("=" * 68)

    if not baseline or not lsp:
        return

    print(
        f"{'Metric':24s}"
        f"{'V74':>16s}"
        f"{'V74-LSP':>16s}"
        f"{'Delta':>16s}"
    )

    print("-" * 72)

    rows = [
        (
            "Trades",
            baseline["trades"],
            lsp["trades"],
        ),
        (
            "Win Rate %",
            baseline["wr"],
            lsp["wr"],
        ),
        (
            "Net PnL $",
            baseline["pnl"],
            lsp["pnl"],
        ),
        (
            "Net R",
            baseline["net_r"],
            lsp["net_r"],
        ),
        (
            "Max DD $",
            baseline["max_dd"],
            lsp["max_dd"],
        ),
        (
            "Max DD %",
            baseline["max_dd_pct"],
            lsp["max_dd_pct"],
        ),
        (
            "Max Loss Streak",
            baseline["max_loss_streak"],
            lsp["max_loss_streak"],
        ),
    ]

    for metric, b, v in rows:
        delta = v - b

        if (
            "Rate" in metric
            or "DD %" in metric
        ):
            print(
                f"{metric:24s}"
                f"{b:16.2f}"
                f"{v:16.2f}"
                f"{delta:16.2f}"
            )

        elif (
            "PnL" in metric
            or "DD $" in metric
        ):
            print(
                f"{metric:24s}"
                f"${b:15,.2f}"
                f"${v:15,.2f}"
                f"${delta:15,.2f}"
            )

        else:
            print(
                f"{metric:24s}"
                f"{b:16.2f}"
                f"{v:16.2f}"
                f"{delta:16.2f}"
            )

    print("\nIMPORTANT:")
    print(
        "V74-LSP never rejects a valid V74 signal."
    )
    print(
        "It only changes priority when MAX_POSITIONS slots are scarce."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(
        "\nRunning EXACT V74 baseline..."
    )

    baseline_trades, baseline_equity, baseline_diag = (
        run_backtest(
            processed_data,
            use_lsp=False,
        )
    )

    baseline_summary = report(
        "V74 BASELINE (UNCHANGED CORE)",
        baseline_trades,
        baseline_equity,
        baseline_diag,
    )

    print(
        "\nRunning V74-LSP..."
    )

    lsp_trades, lsp_equity, lsp_diag = (
        run_backtest(
            processed_data,
            use_lsp=True,
        )
    )

    lsp_summary = report(
        "V74-LSP (ONLY LOSS-STREAK PRIORITY SHIELD)",
        lsp_trades,
        lsp_equity,
        lsp_diag,
    )

    print_comparison(
        baseline_summary,
        lsp_summary,
    )

    save_csvs(
        baseline_trades,
        baseline_equity,
        lsp_trades,
        lsp_equity,
    )

    print(
        "\nHUNTER-V74-LSP BACKTEST COMPLETE."
    )
