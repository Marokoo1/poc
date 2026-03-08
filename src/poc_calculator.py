from __future__ import annotations

import pandas as pd


VALID_PERIODS = {
    "weekly": "W",
    "monthly": "M",
    "yearly": "Y",
}


def auto_tick_size(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.01

    avg_range = (df["High"] - df["Low"]).mean()
    if pd.isna(avg_range) or avg_range <= 0:
        return 0.01

    tick = avg_range / 200.0

    if tick < 0.01:
        return 0.01
    if tick < 0.05:
        return 0.01
    if tick < 0.10:
        return 0.05
    if tick < 0.50:
        return 0.10
    if tick < 1.00:
        return 0.50
    if tick < 5.00:
        return 1.00
    return float(round(tick, 0))


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    prepared = df.copy()

    if "Date" not in prepared.columns:
        raise ValueError("Input dataframe must contain a 'Date' column.")

    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared = prepared.dropna(subset=["Date", "High", "Low", "Close", "Volume"])
    prepared = prepared.sort_values("Date").reset_index(drop=True)

    return prepared


def _build_period_column(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period not in VALID_PERIODS:
        raise ValueError("period must be one of: weekly, monthly, yearly")

    result = df.copy()
    result["Period"] = result["Date"].dt.to_period(VALID_PERIODS[period])
    return result


def _calculate_single_period(group: pd.DataFrame, tick_size: float) -> dict:
    period_high = float(group["High"].max())
    period_low = float(group["Low"].min())

    if tick_size <= 0:
        tick_size = 0.01

    span = max(period_high - period_low, tick_size)
    steps = int(span / tick_size) + 2

    price_levels = pd.Series(
        [round(period_low + i * tick_size, 10) for i in range(steps)]
    ).round(4)

    volume_profile = pd.Series(0.0, index=price_levels)

    for _, row in group.iterrows():
        low = float(row["Low"])
        high = float(row["High"])
        volume = float(row["Volume"])

        mask = (volume_profile.index >= low) & (volume_profile.index <= high)
        n_buckets = int(mask.sum())

        if n_buckets <= 0:
            nearest_idx = (abs(volume_profile.index - low)).argmin()
            volume_profile.iloc[nearest_idx] += volume
            continue

        volume_profile.loc[mask] += volume / n_buckets

    poc_price = float(volume_profile.idxmax())
    poc_volume = float(volume_profile.max())

    return {
        "Period": str(group["Period"].iloc[0]),
        "PeriodStart": group["Date"].min().date().isoformat(),
        "PeriodEnd": group["Date"].max().date().isoformat(),
        "POC": round(poc_price, 4),
        "POC_Volume": round(poc_volume, 0),
        "Period_High": round(period_high, 4),
        "Period_Low": round(period_low, 4),
        "Period_Close": round(float(group["Close"].iloc[-1]), 4),
    }


def calculate_period_poc(
    df: pd.DataFrame,
    period: str,
    tick_size: float | None = None,
) -> pd.DataFrame:
    prepared = _prepare_dataframe(df)
    if prepared.empty:
        return pd.DataFrame()

    if tick_size is None:
        tick_size = auto_tick_size(prepared)

    prepared = _build_period_column(prepared, period)

    rows = []
    for _, group in prepared.groupby("Period"):
        rows.append(_calculate_single_period(group, tick_size))

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values("PeriodStart").reset_index(drop=True)


def filter_complete_periods(
    df: pd.DataFrame,
    period: str,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    now = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    result = df.copy()
    period_end = pd.to_datetime(result["PeriodEnd"], errors="coerce")

    if period == "weekly":
        current_period_start = now - pd.Timedelta(days=now.weekday())
        mask = period_end < current_period_start
    elif period == "monthly":
        current_period_start = now.replace(day=1)
        mask = period_end < current_period_start
    elif period == "yearly":
        current_period_start = now.replace(month=1, day=1)
        mask = period_end < current_period_start
    else:
        raise ValueError("period must be one of: weekly, monthly, yearly")

    return result.loc[mask].reset_index(drop=True)


def _get_current_side(last_close: float, level: float) -> str:
    if last_close > level:
        return "above"
    if last_close < level:
        return "below"
    return "at"


def enrich_poc_with_level_status(
    poc_df: pd.DataFrame,
    price_df: pd.DataFrame,
    track_touch: bool = True,
    track_cross: bool = True,
) -> pd.DataFrame:
    if poc_df.empty:
        return poc_df

    prices = _prepare_dataframe(price_df)
    if prices.empty:
        result = poc_df.copy()
        if track_touch:
            result["Touched"] = False
            result["FirstTouchDate"] = None
            result["TouchCount"] = 0
        if track_cross:
            result["Crossed"] = False
            result["FirstCrossDate"] = None
            result["CrossCount"] = 0
            result["LastCrossDirection"] = None
            result["CurrentSide"] = None
        return result

    result = poc_df.copy()
    prices = prices.sort_values("Date").reset_index(drop=True)

    touched_values = []
    first_touch_dates = []
    touch_counts = []

    crossed_values = []
    first_cross_dates = []
    cross_counts = []
    last_cross_directions = []
    current_sides = []

    for _, row in result.iterrows():
        level = float(row["POC"])
        activation_date = pd.to_datetime(row["PeriodEnd"], errors="coerce") + pd.Timedelta(days=1)

        future = prices.loc[prices["Date"] >= activation_date].copy()

        if future.empty:
            if track_touch:
                touched_values.append(False)
                first_touch_dates.append(None)
                touch_counts.append(0)

            if track_cross:
                crossed_values.append(False)
                first_cross_dates.append(None)
                cross_counts.append(0)
                last_cross_directions.append(None)
                current_sides.append(None)

            continue

        if track_touch:
            touch_mask = (future["Low"] <= level) & (future["High"] >= level)
            touch_rows = future.loc[touch_mask]

            touched_values.append(bool(touch_mask.any()))
            first_touch_dates.append(
                touch_rows["Date"].iloc[0].date().isoformat() if not touch_rows.empty else None
            )
            touch_counts.append(int(touch_mask.sum()))

        if track_cross:
            prev_close = future["Close"].shift(1)
            curr_close = future["Close"]

            cross_up_mask = (prev_close < level) & (curr_close > level)
            cross_down_mask = (prev_close > level) & (curr_close < level)
            cross_mask = cross_up_mask | cross_down_mask

            cross_rows = future.loc[cross_mask]

            crossed_values.append(bool(cross_mask.any()))
            first_cross_dates.append(
                cross_rows["Date"].iloc[0].date().isoformat() if not cross_rows.empty else None
            )
            cross_counts.append(int(cross_mask.sum()))

            if not cross_rows.empty:
                first_cross_idx = cross_rows.index[-1]
                if cross_up_mask.loc[first_cross_idx]:
                    last_cross_directions.append("up")
                elif cross_down_mask.loc[first_cross_idx]:
                    last_cross_directions.append("down")
                else:
                    last_cross_directions.append(None)
            else:
                last_cross_directions.append(None)

            current_sides.append(_get_current_side(float(future["Close"].iloc[-1]), level))

    if track_touch:
        result["Touched"] = touched_values
        result["FirstTouchDate"] = first_touch_dates
        result["TouchCount"] = touch_counts

    if track_cross:
        result["Crossed"] = crossed_values
        result["FirstCrossDate"] = first_cross_dates
        result["CrossCount"] = cross_counts
        result["LastCrossDirection"] = last_cross_directions
        result["CurrentSide"] = current_sides

    return result


def calculate_poc(
    df: pd.DataFrame,
    periods: list[str] | None = None,
    keep_last: dict[str, int | None] | None = None,
    tick_size: float | None = None,
    today: pd.Timestamp | None = None,
    include_level_status: bool = False,
    track_touch: bool = True,
    track_cross: bool = True,
) -> pd.DataFrame:
    prepared = _prepare_dataframe(df)
    if prepared.empty:
        return pd.DataFrame()

    selected_periods = periods or ["weekly", "monthly", "yearly"]
    keep_last = keep_last or {"weekly": 5, "monthly": 5, "yearly": 3}

    inferred_tick_size = tick_size if tick_size is not None else auto_tick_size(prepared)

    results = []
    for period in selected_periods:
        period_df = calculate_period_poc(prepared, period=period, tick_size=inferred_tick_size)
        period_df = filter_complete_periods(period_df, period=period, today=today)

        keep = keep_last.get(period)
        if keep is not None:
            period_df = period_df.tail(keep).reset_index(drop=True)

        if period_df.empty:
            continue

        period_df.insert(0, "PeriodType", period)
        results.append(period_df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    if include_level_status:
        combined = enrich_poc_with_level_status(
            combined,
            prepared,
            track_touch=track_touch,
            track_cross=track_cross,
        )

    return combined
