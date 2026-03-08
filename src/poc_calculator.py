*** Begin Patch
*** Delete File: src/poc_calculator.py
*** Add File: src/poc_calculator.py
+from __future__ import annotations
+
+import pandas as pd
+
+
+VALID_PERIODS = {
+    "weekly": "W",
+    "monthly": "M",
+    "yearly": "Y",
+}
+
+
+def auto_tick_size(df: pd.DataFrame) -> float:
+    """
+    Odhad vhodné velikosti cenového bucketu podle průměrného denního range.
+    Cíl je držet přibližně 200 bucketů na průměrný denní rozsah.
+    """
+    if df.empty:
+        return 0.01
+
+    avg_range = (df["High"] - df["Low"]).mean()
+    if pd.isna(avg_range) or avg_range <= 0:
+        return 0.01
+
+    tick = avg_range / 200.0
+
+    if tick < 0.01:
+        return 0.01
+    if tick < 0.05:
+        return 0.01
+    if tick < 0.10:
+        return 0.05
+    if tick < 0.50:
+        return 0.10
+    if tick < 1.00:
+        return 0.50
+    if tick < 5.00:
+        return 1.00
+    return float(round(tick, 0))
+
+
+def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
+    """
+    Očekává standardní OHLCV dataframe se sloupci:
+    Date, Open, High, Low, Close, Volume
+    """
+    if df.empty:
+        return pd.DataFrame()
+
+    prepared = df.copy()
+
+    if "Date" not in prepared.columns:
+        raise ValueError("Input dataframe must contain a 'Date' column.")
+
+    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
+
+    for col in ["Open", "High", "Low", "Close", "Volume"]:
+        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
+
+    prepared = prepared.dropna(subset=["Date", "High", "Low", "Close", "Volume"])
+    prepared = prepared.sort_values("Date").reset_index(drop=True)
+
+    return prepared
+
+
+def _build_period_column(df: pd.DataFrame, period: str) -> pd.DataFrame:
+    if period not in VALID_PERIODS:
+        raise ValueError("period must be one of: weekly, monthly, yearly")
+
+    result = df.copy()
+    result["Period"] = result["Date"].dt.to_period(VALID_PERIODS[period])
+    return result
+
+
+def _calculate_single_period(group: pd.DataFrame, tick_size: float) -> dict:
+    period_high = float(group["High"].max())
+    period_low = float(group["Low"].min())
+
+    if tick_size <= 0:
+        tick_size = 0.01
+
+    span = max(period_high - period_low, tick_size)
+    steps = int(span / tick_size) + 2
+
+    price_levels = pd.Series(
+        [round(period_low + i * tick_size, 10) for i in range(steps)]
+    ).round(4)
+
+    volume_profile = pd.Series(0.0, index=price_levels)
+
+    for _, row in group.iterrows():
+        low = float(row["Low"])
+        high = float(row["High"])
+        volume = float(row["Volume"])
+
+        mask = (volume_profile.index >= low) & (volume_profile.index <= high)
+        n_buckets = int(mask.sum())
+
+        if n_buckets <= 0:
+            nearest_idx = (abs(volume_profile.index - low)).argmin()
+            volume_profile.iloc[nearest_idx] += volume
+            continue
+
+        volume_profile.loc[mask] += volume / n_buckets
+
+    poc_price = float(volume_profile.idxmax())
+    poc_volume = float(volume_profile.max())
+
+    return {
+        "Period": str(group["Period"].iloc[0]),
+        "PeriodStart": group["Date"].min().date().isoformat(),
+        "PeriodEnd": group["Date"].max().date().isoformat(),
+        "POC": round(poc_price, 4),
+        "POC_Volume": round(poc_volume, 0),
+        "Period_High": round(period_high, 4),
+        "Period_Low": round(period_low, 4),
+        "Period_Close": round(float(group["Close"].iloc[-1]), 4),
+    }
+
+
+def calculate_period_poc(
+    df: pd.DataFrame,
+    period: str,
+    tick_size: float | None = None,
+) -> pd.DataFrame:
+    prepared = _prepare_dataframe(df)
+    if prepared.empty:
+        return pd.DataFrame()
+
+    if tick_size is None:
+        tick_size = auto_tick_size(prepared)
+
+    prepared = _build_period_column(prepared, period)
+
+    rows = []
+    for _, group in prepared.groupby("Period"):
+        rows.append(_calculate_single_period(group, tick_size))
+
+    result = pd.DataFrame(rows)
+    if result.empty:
+        return result
+
+    return result.sort_values("PeriodStart").reset_index(drop=True)
+
+
+def filter_complete_periods(
+    df: pd.DataFrame,
+    period: str,
+    today: pd.Timestamp | None = None,
+) -> pd.DataFrame:
+    """
+    Odfiltruje aktuálně běžící (neuzavřenou) periodu.
+    """
+    if df.empty:
+        return df
+
+    now = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
+    result = df.copy()
+    period_end = pd.to_datetime(result["PeriodEnd"], errors="coerce")
+
+    if period == "weekly":
+        current_period_start = now - pd.Timedelta(days=now.weekday())
+        mask = period_end < current_period_start
+    elif period == "monthly":
+        current_period_start = now.replace(day=1)
+        mask = period_end < current_period_start
+    elif period == "yearly":
+        current_period_start = now.replace(month=1, day=1)
+        mask = period_end < current_period_start
+    else:
+        raise ValueError("period must be one of: weekly, monthly, yearly")
+
+    return result.loc[mask].reset_index(drop=True)
+
+
+def calculate_poc(
+    df: pd.DataFrame,
+    periods: list[str] | None = None,
+    keep_last: dict[str, int | None] | None = None,
+    tick_size: float | None = None,
+    today: pd.Timestamp | None = None,
+) -> pd.DataFrame:
+    """
+    Hlavní vstupní funkce pro projekt.
+
+    Vrací jeden sjednocený dataframe s POC levely za weekly/monthly/yearly.
+    Výchozí hodnoty jsou zvolené tak, aby fungovala přímo z aktuálního main.py.
+    """
+    prepared = _prepare_dataframe(df)
+    if prepared.empty:
+        return pd.DataFrame()
+
+    selected_periods = periods or ["weekly", "monthly", "yearly"]
+    keep_last = keep_last or {"weekly": 5, "monthly": 5, "yearly": 3}
+
+    inferred_tick_size = tick_size if tick_size is not None else auto_tick_size(prepared)
+
+    results = []
+    for period in selected_periods:
+        period_df = calculate_period_poc(prepared, period=period, tick_size=inferred_tick_size)
+        period_df = filter_complete_periods(period_df, period=period, today=today)
+
+        keep = keep_last.get(period)
+        if keep is not None:
+            period_df = period_df.tail(keep).reset_index(drop=True)
+
+        if period_df.empty:
+            continue
+
+        period_df.insert(0, "PeriodType", period)
+        results.append(period_df)
+
+    if not results:
+        return pd.DataFrame()
+
+    return pd.concat(results, ignore_index=True)
*** Update File: src/main.py
@@
-from config import load_settings
+from pathlib import Path
+
+from config import load_settings
 from utils import ensure_directory
 from symbol_loader import load_symbols
 from data_fetcher import (
@@
 )
 from poc_calculator import calculate_poc
@@
         describe_dataset(df, symbol)
         result = calculate_poc(df)
 
         if result.empty:
             print(f"[INFO] No data processed for {symbol}")
         else:
+            result.insert(0, "Symbol", symbol)
+            output_path = Path(processed_data_dir) / f"{symbol}_poc.csv"
+            result.to_csv(output_path, index=False)
             print(f"[OK] Processed data for {symbol}: {len(result)} rows")
+            print(f"[OK] Saved processed output: {output_path}")
*** End Patch
