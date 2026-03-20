from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from poc_calculator import calculate_period_poc, filter_complete_periods
from ib_calculator import IBSettings, calculate_ib_levels


@dataclass(frozen=True)
class StrategyLevelSettings:
    signal_mode: str = "poc"  # poc, ib, poc_ib, ib_poc
    poc_periods: tuple[str, ...] = ("monthly", "yearly")
    confluence_max_atr: float = 0.25
    allow_ib_core: bool = True
    allow_ib_standard: bool = True
    allow_ib_fib: bool = False
    ib_settings: IBSettings = IBSettings()


def _standardize_poc_levels(ticker: str, price_df: pd.DataFrame, poc_periods: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for period_type in poc_periods:
        poc_df = calculate_period_poc(price_df.copy(), period=period_type)
        if poc_df.empty:
            continue
        poc_df = filter_complete_periods(poc_df, period=period_type)
        if poc_df.empty:
            continue

        poc_df = poc_df.copy()
        poc_df["Ticker"] = ticker
        poc_df["PeriodType"] = period_type
        poc_df["PeriodStart"] = pd.to_datetime(poc_df["PeriodStart"], errors="coerce")
        poc_df["PeriodEnd"] = pd.to_datetime(poc_df["PeriodEnd"], errors="coerce")
        poc_df["ActiveFrom"] = poc_df["PeriodEnd"] + pd.Timedelta(days=1)
        poc_df["ValidUntil"] = pd.NaT
        poc_df["LevelPrice"] = pd.to_numeric(poc_df["POC"], errors="coerce")
        poc_df["LevelName"] = "POC"
        poc_df["LevelSource"] = "POC"
        poc_df["LevelFamily"] = "core"
        poc_df["Direction"] = "neutral"
        poc_df["ProjectionMultiplier"] = np.nan
        poc_df["SourceHigh"] = pd.to_numeric(poc_df["Period_High"], errors="coerce")
        poc_df["SourceLow"] = pd.to_numeric(poc_df["Period_Low"], errors="coerce")
        poc_df["SourceRange"] = poc_df["SourceHigh"] - poc_df["SourceLow"]
        poc_df["BasePeriodType"] = poc_df["PeriodType"]
        frames.append(poc_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)
    return out


def _filter_ib_families(ib_levels: pd.DataFrame, settings: StrategyLevelSettings) -> pd.DataFrame:
    if ib_levels.empty:
        return ib_levels

    allow = set()
    if settings.allow_ib_core:
        allow.add("core")
    if settings.allow_ib_standard:
        allow.add("standard")
    if settings.allow_ib_fib:
        allow.add("fib")

    out = ib_levels[ib_levels["LevelFamily"].isin(allow)].copy()
    return out.reset_index(drop=True)


def _get_row_on_or_after(ohlcv: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    matches = ohlcv.loc[ohlcv["Date"] >= dt]
    if matches.empty:
        return None
    return matches.iloc[0]


def _get_atr_at_active_from(level_row: pd.Series, ohlcv: pd.DataFrame) -> float | None:
    active_from = pd.Timestamp(level_row["ActiveFrom"])
    row = _get_row_on_or_after(ohlcv, active_from)
    if row is None:
        return None
    atr = pd.to_numeric(pd.Series([row.get("ATR")]), errors="coerce").iloc[0]
    if pd.isna(atr) or float(atr) <= 0:
        return None
    return float(atr)


def _eligible_confirm_levels(confirm_levels: pd.DataFrame, signal_row: pd.Series) -> pd.DataFrame:
    if confirm_levels.empty:
        return confirm_levels
    active_from = pd.Timestamp(signal_row["ActiveFrom"])
    subset = confirm_levels.copy()
    subset = subset[pd.to_datetime(subset["ActiveFrom"], errors="coerce") <= active_from]
    if "ValidUntil" in subset.columns:
        valid_until = pd.to_datetime(subset["ValidUntil"], errors="coerce")
        subset = subset[valid_until.isna() | (valid_until >= active_from)]
    return subset


def _annotate_confluence(
    signal_levels: pd.DataFrame,
    confirm_levels: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    max_atr: float,
) -> pd.DataFrame:
    out = signal_levels.copy()
    if out.empty:
        return out

    out["trade_mode"] = "standalone"
    out["has_confluence"] = False
    out["confluence_count"] = 0
    out["nearest_confluence_price"] = np.nan
    out["nearest_confluence_distance_abs"] = np.nan
    out["nearest_confluence_distance_atr"] = np.nan
    out["confluence_period_types"] = ""
    out["confluence_level_names"] = ""
    out["confluence_sources"] = ""

    for idx, row in out.iterrows():
        eligible = _eligible_confirm_levels(confirm_levels, row)
        if eligible.empty:
            continue

        atr = _get_atr_at_active_from(row, ohlcv)
        if atr is None or atr <= 0:
            continue

        level_price = float(row["LevelPrice"])
        eligible = eligible.copy()
        eligible["distance_abs"] = (pd.to_numeric(eligible["LevelPrice"], errors="coerce") - level_price).abs()
        eligible["distance_atr"] = eligible["distance_abs"] / atr
        hits = eligible.loc[eligible["distance_atr"] <= max_atr].sort_values("distance_abs")
        if hits.empty:
            continue

        nearest = hits.iloc[0]
        out.at[idx, "has_confluence"] = True
        out.at[idx, "confluence_count"] = int(len(hits))
        out.at[idx, "nearest_confluence_price"] = float(nearest["LevelPrice"])
        out.at[idx, "nearest_confluence_distance_abs"] = float(nearest["distance_abs"])
        out.at[idx, "nearest_confluence_distance_atr"] = float(nearest["distance_atr"])
        out.at[idx, "confluence_period_types"] = ",".join(sorted(hits["PeriodType"].astype(str).unique().tolist()))
        out.at[idx, "confluence_level_names"] = ",".join(hits["LevelName"].astype(str).head(5).tolist())
        out.at[idx, "confluence_sources"] = ",".join(sorted(hits["LevelSource"].astype(str).unique().tolist()))

    return out


def build_strategy_levels_for_ticker(
    ticker: str,
    price_df: pd.DataFrame,
    settings: StrategyLevelSettings | None = None,
) -> pd.DataFrame:
    settings = settings or StrategyLevelSettings()

    poc_levels = _standardize_poc_levels(ticker, price_df, settings.poc_periods)
    ib_levels = calculate_ib_levels(price_df, settings=settings.ib_settings, ticker=ticker)
    ib_levels = _filter_ib_families(ib_levels, settings)

    mode = settings.signal_mode.lower()
    if mode == "poc":
        out = poc_levels.copy()
        if not out.empty:
            out["trade_mode"] = "poc"
            out["has_confluence"] = False
        return out

    if mode == "ib":
        out = ib_levels.copy()
        if not out.empty:
            out["trade_mode"] = "ib"
            out["has_confluence"] = False
        return out

    if mode == "poc_ib":
        if poc_levels.empty:
            return poc_levels
        out = _annotate_confluence(
            poc_levels,
            ib_levels,
            price_df,
            max_atr=float(settings.confluence_max_atr),
        )
        out = out.loc[out["has_confluence"]].copy()
        if not out.empty:
            out["trade_mode"] = "poc_ib"
        return out.reset_index(drop=True)

    if mode == "ib_poc":
        if ib_levels.empty:
            return ib_levels
        out = _annotate_confluence(
            ib_levels,
            poc_levels,
            price_df,
            max_atr=float(settings.confluence_max_atr),
        )
        out = out.loc[out["has_confluence"]].copy()
        if not out.empty:
            out["trade_mode"] = "ib_poc"
        return out.reset_index(drop=True)

    raise ValueError("signal_mode must be one of: 'poc', 'ib', 'poc_ib', 'ib_poc'")
