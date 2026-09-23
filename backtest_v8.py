    def evaluate_signal(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, idx_15m: int) -> Optional[Dict[str, Any]]:
        if idx_15m < 30:
            return None

        subset = df_15m.iloc[:idx_15m + 1]
        current_candle = subset.iloc[-1]
        
        regime = MarketContextAnalyzer.get_market_regime(df_4h, max(0, idx_15m // 16))
        if regime == "Neutral":
            return None

        score = 0
        direction = "long" if regime == "Bull" else "short"

        # 1. رژیم بازار (پایه اصلی) -> 3 امتیاز
        score += 3

        # 2. تشخیص نقدینگی و Sweep -> 3 امتیاز
        lookback_liq = 15
        recent_high = subset["high"].iloc[-lookback_liq:-1].max()
        recent_low = subset["low"].iloc[-lookback_liq:-1].min()

        liquidity_swept = False
        if direction == "long" and current_candle["low"] < recent_low and current_candle["close"] > recent_low:
            liquidity_swept = True
            score += 3
        elif direction == "short" and current_candle["high"] > recent_high and current_candle["close"] < recent_high:
            liquidity_swept = True
            score += 3

        # 3. شتاب و Displacement -> 2 امتیاز
        high_low = subset["high"] - subset["low"]
        atr = high_low.rolling(14).mean().iloc[-1]
        candle_range = current_candle["high"] - current_candle["low"]

        if candle_range > (1.1 * atr):
            score += 2

        # 4. حجم -> 2 امتیاز
        if "volume" in subset.columns:
            vol_mean = subset["volume"].rolling(14).mean().iloc[-1]
            if current_candle["volume"] > (1.1 * vol_mean):
                score += 2

        # حد نصاب منطقی (مثلاً ۵ از ۱۰ به جای ۷ از ۷ مطلق)
        if score < 5:
            return None
