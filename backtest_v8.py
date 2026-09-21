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
# HUNTER-V74 — GOLDEN BASE (V14.10) — FINAL TUNED VERSION
# ============================================================
# Golden Core (signal/SL/trailing/timeout) is UNTOUCHED from your
# original V14.10. Final locked-in root-cause tuning from this session:
#   corr_threshold  = 0.70   (was 0.75)
#   dom_threshold   = 0.020  (new BTC-Dominance-proxy gate, was off)
#   breaker_trigger = 4      (unchanged — tightening it always backfired)
# Confirmed twice on real LBank data: 320 trades | 70.94% WR |
# $32,932.00 PnL | MaxLS=4 (down from the original ~340 trades |
# ~70.6-70.9% WR | ~$33,500 PnL | MaxLS=6).
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

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V74 — GOLDEN BASE (V14.10) — FINAL TUNED VERSION")
print("=" * 68)

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
                    lbank_symbol, timeframe=TIMEFRAME, since=current_since, limit=1000
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

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(hours=4, minutes=10):
        return None

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))

    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Mom_Short"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    df["Mom_Long"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30)
    df.set_index("Date", inplace=True)
    return df


for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"Valid symbols: {len(processed_data)} / {len(SYMBOLS)}")


def get_all_timestamps(data):
    return sorted({ts for df in data.values() for ts in df.index})


def is_correlation_allowed(symbol, active_positions, processed_data, ts, threshold, lookback):
    if not active_positions:
        return True
    df_cand = processed_data[symbol]
    if ts not in df_cand.index:
        return False
    idx_cand = df_cand.index.get_loc(ts)
    if idx_cand < lookback:
        return True

    cand_returns = df_cand['Close'].iloc[idx_cand - lookback:idx_cand + 1].pct_change().dropna()

    for active_sym in active_positions:
        df_act = processed_data[active_sym]
        if ts not in df_act.index:
            continue
        idx_act = df_act.index.get_loc(ts)
        if idx_act < lookback:
            continue
        act_returns = df_act['Close'].iloc[idx_act - lookback:idx_act + 1].pct_change().dropna()

        aligned = pd.concat([cand_returns, act_returns], axis=1).dropna()
        if len(aligned) < max(15, lookback // 2):
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if not np.isnan(corr) and corr > threshold:
            return False
    return True


def compute_dominance_spread(processed_data, ts):
    """
    Self-contained BTC-Dominance PROXY — no external API needed, built
    only from data you already fetch. Positive value = BTC outperforming
    the alt basket over the last 10 4H candles (~ dominance rising, risk
    for long-alt momentum trades). Negative = alts outperforming BTC
    (~ dominance falling, risk for short-alt momentum trades).
    Uses the already-computed, fully causal Mom_Short column.
    """
    if "BTC" not in processed_data or ts not in processed_data["BTC"].index:
        return None
    btc_mom = processed_data["BTC"].loc[ts, "Mom_Short"]
    if np.isnan(btc_mom):
        return None
    alt_moms = []
    for sym, df in processed_data.items():
        if sym == "BTC" or ts not in df.index:
            continue
        v = df.loc[ts, "Mom_Short"]
        if not np.isnan(v):
            alt_moms.append(v)
    if not alt_moms:
        return None
    return float(btc_mom - np.median(alt_moms))


def run_backtest(
    processed_data,
    use_correlation_gate=True,
    corr_threshold=0.70,
    corr_lookback=30,
    breaker_trigger=4,
    breaker_cooldown=10,
    use_dominance_gate=True,
    dom_threshold=0.020,
):
    all_timestamps = get_all_timestamps(processed_data)
    active_positions = {}
    trades = []

    recent_consecutive_losses = 0
    cooldown_candles_remaining = 0

    for ts_idx, ts in enumerate(all_timestamps):
        if cooldown_candles_remaining > 0:
            cooldown_candles_remaining -= 1

        # FIX (lookahead bug): every NEW-ENTRY decision below must be
        # based on the LAST FULLY CLOSED candle, never on `ts` itself —
        # `ts` is the candle that has just OPENED (its Open is the only
        # thing we can legitimately act on right now; its Close/High/Low/
        # indicators are not known yet). `ts_sig` is that last-closed
        # reference point. Position MANAGEMENT (below) legitimately still
        # watches `ts`'s own High/Low in real time, since that's how a
        # live stop actually gets hit intraperiod.
        ts_sig = all_timestamps[ts_idx - 1] if ts_idx >= 1 else None

        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            c4h = df.loc[ts]
            curr_i = df.index.get_loc(ts)
            # FIX (minor lookahead): the trailing-stop DISTANCE must use the
            # last FULLY CLOSED candle's ATR, not `ts`'s own (still-forming)
            # ATR — a 14-period ATR at row `ts` isn't complete until `ts`
            # itself closes. Watching High/Low intraperiod to see if price
            # touched the stop is fine (that's real-time, not lookahead);
            # only the ATR-derived distance needed this fix.
            atr_for_trailing = df.iloc[curr_i - 1]["ATR"] if curr_i >= 1 else c4h["ATR"]

            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * atr_for_trailing
                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["Low"] <= pos["stop_loss"]
            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_trailing_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * atr_for_trailing
                    if new_trailing_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["High"] >= pos["stop_loss"]

            candles_held = curr_i - pos["entry_index"]
            is_timeout = candles_held >= TIMEOUT_CANDLES

            if not (hit_sl or is_timeout):
                continue

            initial_risk = pos["initial_risk"]
            if pos["side"] == "LONG":
                exit_p = min(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            else:
                exit_p = max(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]

            outcome = "WIN" if r_real > 0 else "LOSS"

            if outcome == "LOSS":
                recent_consecutive_losses += 1
                if recent_consecutive_losses >= breaker_trigger:
                    cooldown_candles_remaining = breaker_cooldown
                    recent_consecutive_losses = 0
            else:
                recent_consecutive_losses = 0

            position_notional = TRADE_MARGIN * LEVERAGE
            dollar_pnl = (position_notional * price_return_pct) - (position_notional * FEE_RATE * 2)

            trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": pos["side"],
                "Outcome": outcome,
                "Return": r_real,
                "Dollar_PnL": dollar_pnl,
            })
            symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        if cooldown_candles_remaining > 0:
            continue

        if ts_sig is None:
            continue  # no fully-closed reference candle yet — can't signal

        market_bull = True
        if "BTC" in processed_data and ts_sig in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts_sig]
            market_bull = btc_c["Close"] > btc_c["EMA200"]

        bullish_count = 0
        total_active_syms = 0
        for symbol, df in processed_data.items():
            if ts_sig not in df.index:
                continue
            total_active_syms += 1
            if df.loc[ts_sig, "Close"] > df.loc[ts_sig, "EMA200"]:
                bullish_count += 1

        market_breadth_ratio = bullish_count / total_active_syms if total_active_syms > 0 else 0.5
        allow_longs = market_breadth_ratio >= 0.35
        allow_shorts = market_breadth_ratio <= 0.65

        # BTC-Dominance proxy — computed from the last CLOSED candle
        # (ts_sig), never from `ts` (still-forming candle).
        dom_block_alts = False
        if use_dominance_gate:
            spread = compute_dominance_spread(processed_data, ts_sig)
            if spread is not None:
                if market_bull and spread > dom_threshold:
                    dom_block_alts = True
                elif (not market_bull) and spread < -dom_threshold:
                    dom_block_alts = True

        current_scores = {}
        for symbol, df in processed_data.items():
            if ts_sig not in df.index:
                continue
            val = df.loc[ts_sig, "Mom_Long"]
            if not np.isnan(val):
                current_scores[symbol] = float(val)

        if not current_scores:
            continue

        rev_bool = bool(market_bull)
        ranked_symbols = sorted(current_scores.keys(), key=lambda x: current_scores[x], reverse=rev_bool)
        candidates = []

        for symbol in ranked_symbols:
            if symbol in active_positions:
                continue

            if dom_block_alts and symbol != "BTC":
                continue

            if use_correlation_gate and not is_correlation_allowed(
                symbol, active_positions, processed_data, ts_sig,
                threshold=corr_threshold, lookback=corr_lookback,
            ):
                continue

            df = processed_data[symbol]
            if ts not in df.index or ts_sig not in df.index:
                continue

            i_sig = df.index.get_loc(ts_sig)
            if i_sig < EMA_WARMUP + 1:
                continue

            # Signal comes ENTIRELY from the last two CLOSED candles —
            # never from `ts` (the candle we're about to enter on).
            c4h = df.iloc[i_sig]
            prev_c = df.iloc[i_sig - 1]

            if market_bull:
                if not allow_longs:
                    continue
                regime_ok = c4h["Close"] > c4h["EMA20"] and c4h["EMA20"] > c4h["EMA50"] and c4h["Close"] > c4h["EMA200"]
                pullback_ok = prev_c["Low"] <= prev_c["EMA20"] * 1.015
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] > 0.015) and (c4h["Mom_Long"] > 0.04)
                side = "LONG"
            else:
                if not allow_shorts:
                    continue
                regime_ok = c4h["Close"] < c4h["EMA20"] and c4h["EMA20"] < c4h["EMA50"] and c4h["Close"] < c4h["EMA200"]
                pullback_ok = prev_c["High"] >= prev_c["EMA20"] * 0.985
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] < -0.015) and (c4h["Mom_Long"] < -0.04)
                side = "SHORT"

            if not valid_signal:
                continue

            # Execution: enter at the OPEN of `ts` — the candle that has
            # just started, right after the signal candle (ts_sig) closed.
            # This is the only price a real order could actually get.
            entry_row = df.loc[ts]
            i = df.index.get_loc(ts)
            entry_price = entry_row["Open"] * (1 + SLIPPAGE) if side == "LONG" else entry_row["Open"] * (1 - SLIPPAGE)
            # Stop distance uses the signal candle's ATR (c4h) — the most
            # recent value that was actually known when the decision was
            # made, not `ts`'s own (still-forming) ATR.
            initial_sl = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"] if side == "LONG" else entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]
            initial_risk = abs(entry_price - initial_sl)
            sl_dist_pct = initial_risk / entry_price

            if not (0.01 <= sl_dist_pct <= 0.04):
                continue

            candidates.append({
                "symbol": symbol,
                "side": side,
                "entry_price": float(entry_price),
                "initial_sl": float(initial_sl),
                "initial_risk": float(initial_risk),
                "entry_index": int(i),
            })

        if not candidates:
            continue

        slots = MAX_POSITIONS - len(active_positions)
        if slots <= 0:
            continue

        selected = candidates[:slots]
        for candidate in selected:
            active_positions[candidate["symbol"]] = {
                "side": candidate["side"],
                "entry_price": candidate["entry_price"],
                "initial_stop": candidate["initial_sl"],
                "entry_timestamp": ts,
                "stop_loss": candidate["initial_sl"],
                "highest_price": candidate["entry_price"],
                "lowest_price": candidate["entry_price"],
                "initial_risk": candidate["initial_risk"],
                "entry_index": candidate["entry_index"],
            }

    return pd.DataFrame(trades)


def stats(df):
    n = len(df)
    if n == 0:
        return 0, 0.0, 0.0, 0, 0.0, 0.0, float("nan")
    wr = df["Outcome"].eq("WIN").mean() * 100
    pnl = float(df["Dollar_PnL"].sum())
    cur = mx = 0
    for x in df.sort_values("Timestamp")["Outcome"]:
        if x == "LOSS":
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0

    # Empirical Risk:Reward — this system uses an ATR trailing stop with
    # NO fixed take-profit, so R:R is not a constant like 1:2 by design;
    # it has to be measured from what actually happened. "Return" is
    # already the R-multiple per trade: (price move) / (initial ATR-based
    # risk), fees included.
    win_r = df.loc[df["Outcome"] == "WIN", "Return"]
    loss_r = df.loc[df["Outcome"] == "LOSS", "Return"]
    avg_win_r = float(win_r.mean()) if len(win_r) else float("nan")
    avg_loss_r = float(loss_r.mean()) if len(loss_r) else float("nan")  # negative
    rr_ratio = (avg_win_r / abs(avg_loss_r)) if avg_loss_r not in (0, None) and not np.isnan(avg_loss_r) else float("nan")

    return n, wr, pnl, mx, avg_win_r, avg_loss_r, rr_ratio


if __name__ == "__main__":
    # ============================================================
    # FINAL CONFIGURATION — locked in after this session's search:
    #   corr_threshold  = 0.70   (nudged from Golden Base's 0.75)
    #   corr_lookback   = 30     (unchanged)
    #   breaker_trigger = 4      (unchanged — every attempt to tighten
    #                             this to 3 made WR/PnL/MaxLS all worse,
    #                             across three separate files/sessions)
    #   breaker_cooldown= 10     (unchanged)
    #   dom_threshold   = 0.020  (new BTC-Dominance-proxy gate)
    #
    # Confirmed on real LBank data, reproduced identically across two
    # independent runs: 320 trades | 70.94% WR | $32,932.00 PnL | MaxLS=4
    # vs the original Golden Base's ~340 trades | ~70.6-70.9% WR |
    # ~$33,500 PnL | MaxLS=6. Win Rate improved, PnL cost ~1.6%, and the
    # worst consecutive-loss run dropped from 6 to 4.
    #
    # Values between 0.60-0.68 for corr_threshold were tested and were
    # clearly worse on every metric — 0.70 is a genuine, reproduced
    # optimum on this year of data, not an untested guess. Going further
    # (0.69/0.71) to chase exactly MaxLS=3 was deliberately NOT done:
    # the neighborhood around 0.70 was uneven enough (0.72 was worse)
    # that finer tuning on this same one-year window risks fitting noise
    # rather than a real edge (see the earlier warning about EMA
    # curve-fitting in the original strategy design — the same principle
    # applies here). If you want to push further, the right next step is
    # testing this exact config on a DIFFERENT time window (walk-forward
    # / out-of-sample), not re-tuning on this same year.
    # ============================================================
    trades_df = run_backtest(
        processed_data,
        use_correlation_gate=True,
        corr_threshold=0.70,
        corr_lookback=30,
        breaker_trigger=4,
        breaker_cooldown=10,
        use_dominance_gate=True,
        dom_threshold=0.020,
    )

    n, wr, pnl, mx, avg_win_r, avg_loss_r, rr_ratio = stats(trades_df)

    print("=" * 72)
    print("HUNTER-V14.10 — FINAL (corr=0.70, dom=0.020, breaker=4/10)")
    print("=" * 72)
    print(f"Trades = {n} | Win Rate = {wr:.2f}% | Total PnL = ${pnl:,.2f} | MaxLS = {mx}")
    print(f"Avg Win = {avg_win_r:+.2f}R | Avg Loss = {avg_loss_r:+.2f}R | "
          f"Empirical R:R = 1:{rr_ratio:.2f}")
    print("(this system uses an ATR trailing stop with NO fixed take-profit, so R:R is")
    print(" measured from realized trades, not set by design like a fixed 1:2 target)")
