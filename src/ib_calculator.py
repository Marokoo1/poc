from __future__ import annotations

import pandas as pd

STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Date" not in out.columns:
        raise ValueError("Input dataframe must contain 'Date'.")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
    return out


def _first_row_after_date(prices: pd.DataFrame, dt: pd.Timestamp) -> pd.Timestamp | None:
    future = prices.loc[prices["Date"] > dt, "Date"]
    if future.empty:
        return None
    return pd.Timestamp(future.iloc[0])


def _level_row(*, ticker: str, period_type: str, period: str, level_name: str, level_badge: str, level_price: float,
               period_start: pd.Timestamp, period_end: pd.Timestamp, active_from: pd.Timestamp | None,
               ib_high: float, ib_low: float, ib_range: float, multiplier: float | None = None) -> dict:
    return {
        "Ticker": ticker,
        "PeriodType": period_type,
        "Period": period,
        "LevelSource": "IB",
        "LevelFamily": "ib",
        "LevelName": level_name,
        "LevelBadge": level_badge,
        "LevelPrice": round(float(level_price), 6),
        "POC": round(float(level_price), 6),
        "PeriodStart": pd.Timestamp(period_start),
        "PeriodEnd": pd.Timestamp(period_end),
        "ActiveFrom": pd.Timestamp(active_from) if active_from is not None else pd.NaT,
        "ValidUntil": pd.NaT,
        "IB_High": round(float(ib_high), 6),
        "IB_Low": round(float(ib_low), 6),
        "IB_Range": round(float(ib_range), 6),
        "Multiplier": float(multiplier) if multiplier is not None else None,
    }


def _normalize_standard_multipliers(settings: dict) -> list[float]:
    requested = settings.get("standard_multipliers", [1.5, 2.0, 3.0])
    allowed = []
    for m in requested:
        try:
            m = float(m)
        except Exception:
            continue
        if m in {1.5, 2.0, 3.0} and m not in allowed:
            allowed.append(m)
    if not allowed:
        allowed = [1.5, 2.0, 3.0]
    return sorted(allowed)


def _build_standard_rows(*, ticker: str, period_type: str, period: str, period_start: pd.Timestamp,
                         period_end: pd.Timestamp, active_from: pd.Timestamp | None,
                         ib_high: float, ib_low: float, ib_range: float, prefix: str,
                         multipliers: list[float]) -> list[dict]:
    rows: list[dict] = []

    # Requested set semantics:
    #   0    = IB low
    #   100  = IB high
    #   150  = high + 0.5 * range
    #   200  = high + 1.0 * range
    #   300  = high + 2.0 * range
    #   -150 = low  - 0.5 * range
    #   -200 = low  - 1.0 * range
    #   -300 = low  - 2.0 * range
    rows.append(_level_row(
        ticker=ticker, period_type=period_type, period=period,
        level_name=f"{prefix}_0", level_badge="0", level_price=ib_low,
        period_start=period_start, period_end=period_end, active_from=active_from,
        ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=0.0,
    ))
    rows.append(_level_row(
        ticker=ticker, period_type=period_type, period=period,
        level_name=f"{prefix}_100", level_badge="100", level_price=ib_high,
        period_start=period_start, period_end=period_end, active_from=active_from,
        ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=1.0,
    ))

    for m in multipliers:
        delta = m - 1.0
        label = str(int(round(m * 100)))
        rows.append(_level_row(
            ticker=ticker, period_type=period_type, period=period,
            level_name=f"{prefix}_{label}", level_badge=label,
            level_price=ib_high + (delta * ib_range),
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=m,
        ))
        rows.append(_level_row(
            ticker=ticker, period_type=period_type, period=period,
            level_name=f"{prefix}_NEG{label}", level_badge=f"-{label}",
            level_price=ib_low - (delta * ib_range),
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=-m,
        ))
    return rows


def _build_fib_rows(*, ticker: str, period_type: str, period: str, period_start: pd.Timestamp,
                    period_end: pd.Timestamp, active_from: pd.Timestamp | None,
                    ib_high: float, ib_low: float, ib_range: float, prefix: str,
                    multipliers: list[float]) -> list[dict]:
    rows: list[dict] = []
    for m in multipliers:
        try:
            m = float(m)
        except Exception:
            continue
        if m <= 1.0:
            continue
        delta = m - 1.0
        label = f"F{str(m).replace('.', '_')}"
        rows.append(_level_row(
            ticker=ticker, period_type=period_type, period=period,
            level_name=f"{prefix}_{label}", level_badge=label,
            level_price=ib_high + (delta * ib_range),
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=m,
        ))
        rows.append(_level_row(
            ticker=ticker, period_type=period_type, period=period,
            level_name=f"{prefix}_NEG{label}", level_badge=f"-{label}",
            level_price=ib_low - (delta * ib_range),
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, multiplier=-m,
        ))
    return rows


def calculate_monthly_ib_levels(df: pd.DataFrame, ticker: str, settings: dict) -> pd.DataFrame:
    prices = _prepare_dataframe(df)
    if prices.empty or not settings.get("monthly_enabled", True):
        return pd.DataFrame()
    prices["Month"] = prices["Date"].dt.to_period("M").astype(str)
    rows: list[dict] = []
    multipliers = _normalize_standard_multipliers(settings)
    for period, group in prices.groupby("Month", sort=True):
        window = group.head(5).copy()
        if len(window) < 5:
            continue
        ib_high = float(window["High"].max())
        ib_low = float(window["Low"].min())
        ib_range = ib_high - ib_low
        if ib_range <= 0:
            continue
        period_start = pd.Timestamp(window["Date"].min())
        period_end = pd.Timestamp(window["Date"].max())
        active_from = _first_row_after_date(prices, period_end)
        rows.extend(_build_standard_rows(
            ticker=ticker, period_type="monthly_ib", period=period,
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, prefix="M_IB", multipliers=multipliers,
        ))
        if settings.get("fibonacci_projection_enabled", False):
            rows.extend(_build_fib_rows(
                ticker=ticker, period_type="monthly_ib_fib", period=period,
                period_start=period_start, period_end=period_end, active_from=active_from,
                ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, prefix="M_IB_FIB",
                multipliers=list(settings.get("fibonacci_multipliers", [1.272, 1.618])),
            ))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)


def calculate_yearly_ib_levels(df: pd.DataFrame, ticker: str, settings: dict) -> pd.DataFrame:
    prices = _prepare_dataframe(df)
    if prices.empty or not settings.get("yearly_enabled", True):
        return pd.DataFrame()
    prices["Year"] = prices["Date"].dt.year.astype(int)
    prices["MonthNum"] = prices["Date"].dt.month.astype(int)
    rows: list[dict] = []
    multipliers = _normalize_standard_multipliers(settings)
    for year, group in prices.groupby("Year", sort=True):
        window = group[group["MonthNum"].isin([1, 2])].copy()
        if window.empty or set(window["MonthNum"].unique().tolist()) != {1, 2}:
            continue
        ib_high = float(window["High"].max())
        ib_low = float(window["Low"].min())
        ib_range = ib_high - ib_low
        if ib_range <= 0:
            continue
        period = str(year)
        period_start = pd.Timestamp(window["Date"].min())
        period_end = pd.Timestamp(window["Date"].max())
        active_from = _first_row_after_date(prices, period_end)
        rows.extend(_build_standard_rows(
            ticker=ticker, period_type="yearly_ib", period=period,
            period_start=period_start, period_end=period_end, active_from=active_from,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, prefix="Y_IB", multipliers=multipliers,
        ))
        if settings.get("fibonacci_projection_enabled", False):
            rows.extend(_build_fib_rows(
                ticker=ticker, period_type="yearly_ib_fib", period=period,
                period_start=period_start, period_end=period_end, active_from=active_from,
                ib_high=ib_high, ib_low=ib_low, ib_range=ib_range, prefix="Y_IB_FIB",
                multipliers=list(settings.get("fibonacci_multipliers", [1.272, 1.618])),
            ))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)


def calculate_all_ib_levels(df: pd.DataFrame, ticker: str, settings: dict | None = None) -> pd.DataFrame:
    settings = settings or {}
    frames = [
        calculate_monthly_ib_levels(df, ticker=ticker, settings=settings),
        calculate_yearly_ib_levels(df, ticker=ticker, settings=settings),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["Ticker", "PeriodType", "Period", "LevelName", "LevelPrice", "ActiveFrom"])
    return out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)
