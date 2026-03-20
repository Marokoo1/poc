from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_IB_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
FIB_IB_MULTIPLIERS: tuple[float, ...] = (0.618, 1.0, 1.272, 1.618, 2.618)


@dataclass(frozen=True)
class IBSettings:
    enabled: bool = True
    yearly_enabled: bool = True
    monthly_enabled: bool = True
    monthly_mode: str = "first_5_trading_days"
    standard_projection_enabled: bool = True
    standard_multipliers: tuple[float, ...] = STANDARD_IB_MULTIPLIERS
    fibonacci_projection_enabled: bool = False
    fibonacci_multipliers: tuple[float, ...] = FIB_IB_MULTIPLIERS
    round_decimals: int = 4


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return out


def _first_row_after_date(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    matches = df.loc[df["Date"] > dt, "Date"]
    if matches.empty:
        return pd.NaT
    return pd.Timestamp(matches.iloc[0]).normalize()


def _format_multiplier(mult: float) -> str:
    x = f"{mult:.3f}".rstrip("0").rstrip(".")
    return x.replace(".", "")


def _base_row(
    *,
    period: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    active_from: pd.Timestamp | pd.NaT,
    period_type: str,
    base_period_type: str,
    period_high: float,
    period_low: float,
    period_close: float,
    level_price: float,
    level_name: str,
    level_family: str,
    source_high: float,
    source_low: float,
    source_range: float,
    direction: str,
    projection_multiplier: float | None,
    decimals: int,
) -> dict:
    level_price = round(float(level_price), decimals)
    period_high = round(float(period_high), decimals)
    period_low = round(float(period_low), decimals)
    period_close = round(float(period_close), decimals)
    source_high = round(float(source_high), decimals)
    source_low = round(float(source_low), decimals)
    source_range = round(float(source_range), decimals)

    return {
        "Period": str(period),
        "PeriodStart": pd.Timestamp(period_start).date().isoformat(),
        "PeriodEnd": pd.Timestamp(period_end).date().isoformat(),
        "POC": level_price,  # compatibility alias for current backtest
        "POC_Volume": np.nan,
        "Period_High": period_high,
        "Period_Low": period_low,
        "Period_Close": period_close,
        "LevelPrice": level_price,
        "LevelName": level_name,
        "LevelSource": "IB",
        "LevelFamily": level_family,
        "Direction": direction,
        "ProjectionMultiplier": projection_multiplier,
        "SourceHigh": source_high,
        "SourceLow": source_low,
        "SourceRange": source_range,
        "BasePeriodType": base_period_type,
        "PeriodType": period_type,
        "ActiveFrom": pd.Timestamp(active_from).date().isoformat() if pd.notna(active_from) else None,
        "ValidUntil": None,
    }


def _build_projection_rows(
    *,
    period: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    active_from: pd.Timestamp | pd.NaT,
    base_period_type: str,
    base_period_label: str,
    period_high: float,
    period_low: float,
    period_close: float,
    source_high: float,
    source_low: float,
    multipliers: Iterable[float],
    family: str,
    period_type: str,
    decimals: int,
) -> list[dict]:
    rows: list[dict] = []
    source_range = float(source_high - source_low)
    for mult in multipliers:
        mult = float(mult)
        suffix = _format_multiplier(mult)
        up_level = source_high + source_range * mult
        dn_level = source_low - source_range * mult
        label_prefix = f"{base_period_label}_IB" if family == "core" else f"{base_period_label}_IB_{'FIB' if family == 'fib' else ''}".replace("__", "_")
        if family == "standard":
            label_up = f"{base_period_label}_IB_{suffix}_UP"
            label_dn = f"{base_period_label}_IB_{suffix}_DN"
        elif family == "fib":
            label_up = f"{base_period_label}_IB_FIB_{suffix}_UP"
            label_dn = f"{base_period_label}_IB_FIB_{suffix}_DN"
        else:
            label_up = f"{label_prefix}_{suffix}_UP"
            label_dn = f"{label_prefix}_{suffix}_DN"

        rows.append(
            _base_row(
                period=period,
                period_start=period_start,
                period_end=period_end,
                active_from=active_from,
                period_type=period_type,
                base_period_type=base_period_type,
                period_high=period_high,
                period_low=period_low,
                period_close=period_close,
                level_price=up_level,
                level_name=label_up,
                level_family=family,
                source_high=source_high,
                source_low=source_low,
                source_range=source_range,
                direction="up",
                projection_multiplier=mult,
                decimals=decimals,
            )
        )
        rows.append(
            _base_row(
                period=period,
                period_start=period_start,
                period_end=period_end,
                active_from=active_from,
                period_type=period_type,
                base_period_type=base_period_type,
                period_high=period_high,
                period_low=period_low,
                period_close=period_close,
                level_price=dn_level,
                level_name=label_dn,
                level_family=family,
                source_high=source_high,
                source_low=source_low,
                source_range=source_range,
                direction="down",
                projection_multiplier=mult,
                decimals=decimals,
            )
        )
    return rows


def _build_core_rows(
    *,
    period: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    active_from: pd.Timestamp | pd.NaT,
    base_period_type: str,
    base_period_label: str,
    period_high: float,
    period_low: float,
    period_close: float,
    source_high: float,
    source_low: float,
    period_type: str,
    decimals: int,
) -> list[dict]:
    source_range = float(source_high - source_low)
    return [
        _base_row(
            period=period,
            period_start=period_start,
            period_end=period_end,
            active_from=active_from,
            period_type=period_type,
            base_period_type=base_period_type,
            period_high=period_high,
            period_low=period_low,
            period_close=period_close,
            level_price=source_high,
            level_name=f"{base_period_label}_IBH",
            level_family="core",
            source_high=source_high,
            source_low=source_low,
            source_range=source_range,
            direction="up",
            projection_multiplier=None,
            decimals=decimals,
        ),
        _base_row(
            period=period,
            period_start=period_start,
            period_end=period_end,
            active_from=active_from,
            period_type=period_type,
            base_period_type=base_period_type,
            period_high=period_high,
            period_low=period_low,
            period_close=period_close,
            level_price=source_low,
            level_name=f"{base_period_label}_IBL",
            level_family="core",
            source_high=source_high,
            source_low=source_low,
            source_range=source_range,
            direction="down",
            projection_multiplier=None,
            decimals=decimals,
        ),
    ]


def _calculate_yearly(df: pd.DataFrame, settings: IBSettings) -> list[dict]:
    if not settings.yearly_enabled:
        return []

    rows: list[dict] = []
    years = sorted(df["Date"].dt.year.unique().tolist())
    for year in years:
        jan_feb = df[(df["Date"].dt.year == year) & (df["Date"].dt.month.isin([1, 2]))].copy()
        if jan_feb.empty or jan_feb["Date"].dt.month.nunique() < 2:
            continue

        period_start = pd.Timestamp(jan_feb["Date"].min()).normalize()
        period_end = pd.Timestamp(jan_feb["Date"].max()).normalize()
        active_from = _first_row_after_date(df, period_end)
        source_high = float(jan_feb["High"].max())
        source_low = float(jan_feb["Low"].min())
        period_close = float(jan_feb["Close"].iloc[-1])

        rows.extend(
            _build_core_rows(
                period=str(year),
                period_start=period_start,
                period_end=period_end,
                active_from=active_from,
                base_period_type="yearly",
                base_period_label="Y",
                period_high=float(jan_feb["High"].max()),
                period_low=float(jan_feb["Low"].min()),
                period_close=period_close,
                source_high=source_high,
                source_low=source_low,
                period_type="yearly_ib",
                decimals=settings.round_decimals,
            )
        )
        if settings.standard_projection_enabled:
            rows.extend(
                _build_projection_rows(
                    period=str(year),
                    period_start=period_start,
                    period_end=period_end,
                    active_from=active_from,
                    base_period_type="yearly",
                    base_period_label="Y",
                    period_high=float(jan_feb["High"].max()),
                    period_low=float(jan_feb["Low"].min()),
                    period_close=period_close,
                    source_high=source_high,
                    source_low=source_low,
                    multipliers=settings.standard_multipliers,
                    family="standard",
                    period_type="yearly_ib_std",
                    decimals=settings.round_decimals,
                )
            )
        if settings.fibonacci_projection_enabled:
            rows.extend(
                _build_projection_rows(
                    period=str(year),
                    period_start=period_start,
                    period_end=period_end,
                    active_from=active_from,
                    base_period_type="yearly",
                    base_period_label="Y",
                    period_high=float(jan_feb["High"].max()),
                    period_low=float(jan_feb["Low"].min()),
                    period_close=period_close,
                    source_high=source_high,
                    source_low=source_low,
                    multipliers=settings.fibonacci_multipliers,
                    family="fib",
                    period_type="yearly_ib_fib",
                    decimals=settings.round_decimals,
                )
            )
    return rows


def _calculate_monthly(df: pd.DataFrame, settings: IBSettings) -> list[dict]:
    if not settings.monthly_enabled:
        return []
    if settings.monthly_mode != "first_5_trading_days":
        raise ValueError("Only monthly_mode='first_5_trading_days' is currently implemented.")

    rows: list[dict] = []
    month_keys = sorted(df["Date"].dt.to_period("M").unique().tolist())
    for month_key in month_keys:
        month_df = df[df["Date"].dt.to_period("M") == month_key].copy()
        if len(month_df) < 5:
            continue
        ib_window = month_df.iloc[:5].copy()
        period_start = pd.Timestamp(ib_window["Date"].min()).normalize()
        period_end = pd.Timestamp(ib_window["Date"].max()).normalize()
        active_from = _first_row_after_date(df, period_end)
        source_high = float(ib_window["High"].max())
        source_low = float(ib_window["Low"].min())
        period_close = float(ib_window["Close"].iloc[-1])
        period_str = str(month_key)

        rows.extend(
            _build_core_rows(
                period=period_str,
                period_start=period_start,
                period_end=period_end,
                active_from=active_from,
                base_period_type="monthly",
                base_period_label="M",
                period_high=float(ib_window["High"].max()),
                period_low=float(ib_window["Low"].min()),
                period_close=period_close,
                source_high=source_high,
                source_low=source_low,
                period_type="monthly_ib",
                decimals=settings.round_decimals,
            )
        )
        if settings.standard_projection_enabled:
            rows.extend(
                _build_projection_rows(
                    period=period_str,
                    period_start=period_start,
                    period_end=period_end,
                    active_from=active_from,
                    base_period_type="monthly",
                    base_period_label="M",
                    period_high=float(ib_window["High"].max()),
                    period_low=float(ib_window["Low"].min()),
                    period_close=period_close,
                    source_high=source_high,
                    source_low=source_low,
                    multipliers=settings.standard_multipliers,
                    family="standard",
                    period_type="monthly_ib_std",
                    decimals=settings.round_decimals,
                )
            )
        if settings.fibonacci_projection_enabled:
            rows.extend(
                _build_projection_rows(
                    period=period_str,
                    period_start=period_start,
                    period_end=period_end,
                    active_from=active_from,
                    base_period_type="monthly",
                    base_period_label="M",
                    period_high=float(ib_window["High"].max()),
                    period_low=float(ib_window["Low"].min()),
                    period_close=period_close,
                    source_high=source_high,
                    source_low=source_low,
                    multipliers=settings.fibonacci_multipliers,
                    family="fib",
                    period_type="monthly_ib_fib",
                    decimals=settings.round_decimals,
                )
            )
    return rows


def calculate_ib_levels(
    df: pd.DataFrame,
    settings: IBSettings | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    settings = settings or IBSettings()
    if not settings.enabled:
        return pd.DataFrame()

    prepared = _prepare_dataframe(df)
    if prepared.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    rows.extend(_calculate_yearly(prepared, settings))
    rows.extend(_calculate_monthly(prepared, settings))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if ticker is not None:
        out.insert(0, "Ticker", ticker)

    out["PeriodStart"] = pd.to_datetime(out["PeriodStart"], errors="coerce")
    out["PeriodEnd"] = pd.to_datetime(out["PeriodEnd"], errors="coerce")
    out["ActiveFrom"] = pd.to_datetime(out["ActiveFrom"], errors="coerce")
    out["ValidUntil"] = pd.to_datetime(out["ValidUntil"], errors="coerce")

    out = out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)
    return out
