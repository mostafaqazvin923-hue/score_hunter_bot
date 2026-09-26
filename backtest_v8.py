def run_backtest(all_symbol_data: dict[str, dict[str, pd.DataFrame]]) -> list[dict]:
    all_1h_times = sorted(list(set().union(*(dfs["1h"].index for dfs in all_symbol_data.values()))))

    active_position = None
    trades = []
    cooldowns = {sym: 0 for sym in all_symbol_data.keys()}

    # Pre-extract numpy arrays for lightning-fast lookups without dataframe slicing overhead
    symbol_arrays = {}
    for sym, dfs in all_symbol_data.items():
        h1 = dfs["1h"]
        h4 = dfs["4h"]
        d1 = dfs["1d"]
        symbol_arrays[sym] = {
            "h1_times": h1.index,
            "h1_df": h1,
            "h4_times": h4.index,
            "h4_df": h4,
            "d1_times": d1.index,
            "d1_df": d1,
        }

    for ts in all_1h_times:
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        # 1. Manage active position
        if active_position is not None:
            sym = active_position["symbol"]
            h1_df = symbol_arrays[sym]["h1_df"]
            if ts in h1_df.index:
                bar = h1_df.loc[ts]
                hi, lo = float(bar["high"]), float(bar["low"])
                side, sl, tp, entry = active_position["side"], active_position["sl"], active_position["tp"], active_position["entry"]
                notional = MARGIN_PER_TRADE * LEVERAGE

                hit_sl = lo <= sl if side == "LONG" else hi >= sl
                hit_tp = hi >= tp if side == "LONG" else lo <= tp

                if hit_sl or hit_tp:
                    if hit_sl and hit_tp:
                        outcome = "LOSS"
                        exit_price = sl
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_price = sl
                    else:
                        outcome = "WIN"
                        exit_price = tp

                    gross = (exit_price - entry) / entry * notional if side == "LONG" else (entry - exit_price) / entry * notional
                    fees = notional * FEE_RATE * 2.0
                    pnl = gross - fees

                    trades.append({
                        "symbol": sym, "side": side, "entry_ts": active_position["entry_ts"], "exit_ts": ts,
                        "outcome": outcome, "pnl": float(pnl), "entry": entry, "exit": exit_price,
                    })
                    cooldowns[sym] = 3
                    active_position = None

        # 2. Look for new entry if portfolio is free
        if active_position is None:
            candidates = []
            for symbol, data in symbol_arrays.items():
                if cooldowns[symbol] > 0:
                    continue

                h1_df = data["h1_df"]
                h4_df = data["h4_df"]
                d1_df = data["d1_df"]

                # Fast causal slicing using searchsorted (O(log N))
                d1_idx = data["d1_times"].searchsorted(ts, side="right") - 1
                h4_idx = data["h4_times"].searchsorted(ts, side="right") - 1
                h1_idx = data["h1_times"].searchsorted(ts, side="right") - 1

                if d1_idx < 0 or h4_idx < 0 or h1_idx < 25:
                    continue

                d1_row = d1_df.iloc[d1_idx]
                h4_sub = h4_df.iloc[:h4_idx]  # strictly completed < ts
                h1_sub = h1_df.iloc[:h1_idx]  # strictly completed < ts

                if len(h4_sub) == 0 or len(h1_sub) < 25:
                    continue

                h4_row = h4_sub.iloc[-1]

                # Daily Regime
                d1_close = d1_row["close"]
                ema50, ema200, slope = d1_row.get("ema50", np.nan), d1_row.get("ema200", np.nan), d1_row.get("ema50_slope", np.nan)
                if not all(np.isfinite([ema50, ema200, slope])):
                    continue

                daily_long = (d1_close > ema200) and (ema50 > ema200) and (slope > 0)
                daily_short = (d1_close < ema200) and (ema50 < ema200) and (slope < 0)

                if not (daily_long or daily_short):
                    continue

                # 4H Structure
                c_sh, c_sl, h4_close = h4_row.get("confirmed_sh", np.nan), h4_row.get("confirmed_sl", np.nan), h4_row["close"]
                if not all(np.isfinite([c_sh, c_sl])):
                    continue

                h4_completed = h4_sub
                if daily_long:
                    h4_ok = any(h4_completed["close"] > c_sh) and all(h4_completed["low"] >= c_sl) and (h4_close > c_sh)
                    side = "LONG"
                else:
                    h4_ok = any(h4_completed["close"] < c_sl) and all(h4_completed["high"] <= c_sh) and (h4_close < c_sl)
                    side = "SHORT"

                if not h4_ok:
                    continue

                # 1H Failed Auction & VWAP Reclaim
                prev_bar = h1_sub.iloc[-1]
                prev_prev_bar = h1_sub.iloc[-2]

                low24, high24 = prev_bar.get("low24", np.nan), prev_bar.get("high24", np.nan)
                atr = prev_bar.get("atr14", np.nan)
                vol_sma = prev_bar.get("vol_sma20", np.nan)
                vwap = prev_bar.get("vwap", np.nan)
                prev_vwap = prev_prev_bar.get("vwap", np.nan)

                if not all(np.isfinite([low24, high24, atr, vol_sma, vwap, prev_vwap])):
                    continue

                p_low, p_high, p_close, p_vol = prev_bar["low"], prev_bar["high"], prev_bar["close"], prev_bar["volume"]

                if side == "LONG":
                    sweep = (p_low < low24 - 0.10 * atr) and (p_low >= low24 - 1.00 * atr)
                    reclaim = p_close > low24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (prev_prev_bar["close"] <= prev_vwap) and (p_close > vwap)

                    if not (sweep and reclaim and vol_ok and vwap_reclaim):
                        continue
                    sl = p_low - 0.20 * atr
                else:
                    sweep = (p_high > high24 + 0.10 * atr) and (p_high <= high24 + 1.00 * atr)
                    reclaim = p_close < high24
                    vol_ok = p_vol >= 1.10 * vol_sma
                    vwap_reclaim = (prev_prev_bar["close"] >= prev_vwap) and (p_close < vwap)

                    if not (sweep and reclaim and vol_ok and vwap_reclaim):
                        continue
                    sl = p_high + 0.20 * atr

                candidates.append((symbol, side, prev_bar.name, sl, atr))

            if candidates:
                symbol, side, trigger_ts, sl, atr_val = candidates[0]
                h1_df = symbol_arrays[symbol]["h1_df"]
                next_indices = h1_df.index[h1_df.index > trigger_ts]
                if len(next_indices) == 0:
                    continue

                entry_ts = next_indices[0]
                raw_open = float(h1_df.loc[entry_ts, "open"])
                entry = raw_open * (1.0 + SLIPPAGE) if side == "LONG" else raw_open * (1.0 - SLIPPAGE)
                risk = (entry - sl) if side == "LONG" else (sl - entry)

                if risk <= 0:
                    continue

                tp = entry + (RR * risk) if side == "LONG" else entry - (RR * risk)
                active_position = {
                    "symbol": symbol, "side": side,
                    "entry_ts": entry_ts, "entry": entry, "sl": sl, "tp": tp,
                }

    return trades
