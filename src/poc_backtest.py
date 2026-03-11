# Updated poc_backtest.py with departure-from-level activation logic
# Generated from current repo version with configurable threshold.

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
CHARTS_DIR = RESULTS_DIR / "charts"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "DIA", "GLD", "TLT", "XLE"]
TIMEFRAME = "1d"
START_DATE = None
END_DATE = None

ENTRY_BUFFER_ATR = 0.30
SL_BUFFER_ATR = 0.20
TP_MULTIPLE = 1.5
TIME_EXIT_BARS = 15
ATR_PERIOD = 14
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_PCT = 0.0
MIN_ATR = 1e-9
MAX_OPEN_TRADES_PER_TICKER = 1
ALLOW_MULTIPLE_LEVELS_SAME_DAY = False

# --- New configurable activation / arming logic ---
REQUIRE_DEPARTURE_FROM_LEVEL = True
ACTIVATION_THRESHOLD_MODE = "atr"      # "atr" | "pct" | "absolute"
ACTIVATION_THRESHOLD_VALUE = 0.75       # e.g. 0.75 ATR, 0.5 %, or absolute points/pips
# --------------------------------------------------

LEVEL_COLORS = {
    "W": "WeeklyPOC",
    "M": "MonthlyPOC",
    "Y": "YearlyPOC",
}


@dataclass
class TradeResult:
    ticker: str
    level_type: str
    side: str
    level: float
    active_from: pd.Timestamp
    entry_date: pd.Timestamp | None
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str
    bars_held: int | None
    pnl_abs: float | None
    pnl_r: float | None
    mfe: float | None
    mae: float | None


def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}_{TIMEFRAME}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OHLCV file: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("Date").reset_index(drop=True)
    if START_DATE:
        df = df[df["Date"] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        df = df[df["Date"] <= pd.Timestamp(END_DATE)]
    return df.reset_index(drop=True)


def load_levels(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker}_poc_levels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing POC levels file: {path}")
    df = pd.read_csv(path)
    for c in ["PeriodStart", "PeriodEnd", "ActiveFrom"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(None)
    return df.sort_values(["ActiveFrom", "LevelType"]).reset_index(drop=True)


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(period, min_periods=1).mean()
    out["ATR"] = out["ATR"].clip(lower=MIN_ATR)
    return out


def side_touch(side: str, high: float, low: float, level: float, buffer_abs: float) -> bool:
    if side == "long":
        return low <= level + buffer_abs
    return high >= level - buffer_abs


def compute_activation_threshold(level: float, atr: float) -> float:
    mode = str(ACTIVATION_THRESHOLD_MODE).lower()
    value = float(ACTIVATION_THRESHOLD_VALUE)
    if value < 0:
        value = 0.0
    if mode == "atr":
        return max(value * max(float(atr), MIN_ATR), 0.0)
    if mode == "pct":
        return max(abs(level) * (value / 100.0), 0.0)
    if mode == "absolute":
        return max(value, 0.0)
    raise ValueError(f"Unsupported ACTIVATION_THRESHOLD_MODE: {ACTIVATION_THRESHOLD_MODE}")


def departure_reached(side: str, high: float, low: float, level: float, threshold_abs: float) -> bool:
    if threshold_abs <= 0:
        return True
    if side == "long":
        return high >= level + threshold_abs
    return low <= level - threshold_abs


def simulate_single_level(
    ticker: str,
    level_row: pd.Series,
    bars: pd.DataFrame,
) -> TradeResult:
    level = float(level_row["POC"])
    level_type = str(level_row["LevelType"])
    active_from = pd.Timestamp(level_row["ActiveFrom"])

    search_idx = bars.index[bars["Date"] >= active_from]
    if len(search_idx) == 0:
        return TradeResult(
            ticker=ticker,
            level_type=level_type,
            side="na",
            level=level,
            active_from=active_from,
            entry_date=None,
            entry_price=None,
            stop_price=None,
            target_price=None,
            exit_date=None,
            exit_price=None,
            exit_reason="not_active_in_data",
            bars_held=None,
            pnl_abs=None,
            pnl_r=None,
            mfe=None,
            mae=None,
        )

    start_i = int(search_idx[0])
    first_bar = bars.iloc[start_i]
    side = "long" if float(first_bar["Close"]) > level else "short"

    armed_i = start_i
    if REQUIRE_DEPARTURE_FROM_LEVEL:
        armed_i = None
        for i in range(start_i, len(bars)):
            bar = bars.iloc[i]
            threshold_abs = compute_activation_threshold(level, float(bar["ATR"]))
            if departure_reached(side, float(bar["High"]), float(bar["Low"]), level, threshold_abs):
                # arm only after the departure bar has completed
                armed_i = i + 1
                break
        if armed_i is None or armed_i >= len(bars):
            return TradeResult(
                ticker=ticker,
                level_type=level_type,
                side=side,
                level=level,
                active_from=active_from,
                entry_date=None,
                entry_price=None,
                stop_price=None,
                target_price=None,
                exit_date=None,
                exit_price=None,
                exit_reason="no_departure",
                bars_held=None,
                pnl_abs=None,
                pnl_r=None,
                mfe=None,
                mae=None,
            )

    entry_i = None
    entry_price = None
    stop_price = None
    target_price = None
    initial_risk = None

    for i in range(armed_i, len(bars)):
        bar = bars.iloc[i]
        atr = float(bar["ATR"])
        entry_buffer = ENTRY_BUFFER_ATR * atr
        if side_touch(side, float(bar["High"]), float(bar["Low"]), level, entry_buffer):
            entry_i = i
            entry_date = pd.Timestamp(bar["Date"])
            entry_price = level
            if side == "long":
                stop_price = level - SL_BUFFER_ATR * atr
                initial_risk = entry_price - stop_price
                target_price = entry_price + TP_MULTIPLE * initial_risk
            else:
                stop_price = level + SL_BUFFER_ATR * atr
                initial_risk = stop_price - entry_price
                target_price = entry_price - TP_MULTIPLE * initial_risk
            break

    if entry_i is None:
        return TradeResult(
            ticker=ticker,
            level_type=level_type,
            side=side,
            level=level,
            active_from=active_from,
            entry_date=None,
            entry_price=None,
            stop_price=None,
            target_price=None,
            exit_date=None,
            exit_price=None,
            exit_reason="no_touch",
            bars_held=None,
            pnl_abs=None,
            pnl_r=None,
            mfe=None,
            mae=None,
        )

    exit_i = None
    exit_price = None
    exit_reason = None
    mfe = 0.0
    mae = 0.0

    for i in range(entry_i, min(entry_i + TIME_EXIT_BARS + 1, len(bars))):
        bar = bars.iloc[i]
        high = float(bar["High"])
        low = float(bar["Low"])

        if side == "long":
            mfe = max(mfe, high - entry_price)
            mae = min(mae, low - entry_price)
            stop_hit = low <= stop_price
            target_hit = high >= target_price
            if stop_hit and target_hit:
                exit_i = i
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar"
                break
            if stop_hit:
                exit_i = i
                exit_price = stop_price
                exit_reason = "stop"
                break
            if target_hit:
                exit_i = i
                exit_price = target_price
                exit_reason = "target"
                break
        else:
            mfe = max(mfe, entry_price - low)
            mae = min(mae, entry_price - high)
            stop_hit = high >= stop_price
            target_hit = low <= target_price
            if stop_hit and target_hit:
                exit_i = i
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar"
                break
            if stop_hit:
                exit_i = i
                exit_price = stop_price
                exit_reason = "stop"
                break
            if target_hit:
                exit_i = i
                exit_price = target_price
                exit_reason = "target"
                break

    if exit_i is None:
        exit_i = min(entry_i + TIME_EXIT_BARS, len(bars) - 1)
        exit_bar = bars.iloc[exit_i]
        exit_price = float(exit_bar["Close"])
        exit_reason = "time_exit"

    exit_date = pd.Timestamp(bars.iloc[exit_i]["Date"])
    bars_held = exit_i - entry_i

    if side == "long":
        pnl_abs = exit_price - entry_price
    else:
        pnl_abs = entry_price - exit_price

    if SLIPPAGE_PCT:
        pnl_abs -= abs(entry_price) * (SLIPPAGE_PCT / 100.0)
        pnl_abs -= abs(exit_price) * (SLIPPAGE_PCT / 100.0)
    pnl_abs -= COMMISSION_PER_TRADE

    pnl_r = pnl_abs / initial_risk if initial_risk and initial_risk > 0 else math.nan

    return TradeResult(
        ticker=ticker,
        level_type=level_type,
        side=side,
        level=level,
        active_from=active_from,
        entry_date=entry_date,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        bars_held=bars_held,
        pnl_abs=pnl_abs,
        pnl_r=pnl_r,
        mfe=mfe,
        mae=mae,
    )


def run_backtest() -> pd.DataFrame:
    all_results: list[dict[str, Any]] = []

    for ticker in TICKERS:
        bars = add_atr(load_ohlcv(ticker))
        levels = load_levels(ticker)

        for _, level_row in levels.iterrows():
            result = simulate_single_level(ticker, level_row, bars)
            all_results.append(result.__dict__)

    return pd.DataFrame(all_results)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    trades = results.dropna(subset=["entry_date", "exit_date", "pnl_abs"]).copy()
    if trades.empty:
        return pd.DataFrame(
            [{"metric": "total_levels", "value": len(results)}, {"metric": "executed_trades", "value": 0}]
        )

    wins = (trades["pnl_abs"] > 0).sum()
    losses = (trades["pnl_abs"] <= 0).sum()
    summary = pd.DataFrame(
        [
            {"metric": "total_levels", "value": len(results)},
            {"metric": "executed_trades", "value": len(trades)},
            {"metric": "wins", "value": int(wins)},
            {"metric": "losses", "value": int(losses)},
            {"metric": "win_rate_pct", "value": float(wins / len(trades) * 100.0)},
            {"metric": "total_pnl_abs", "value": float(trades["pnl_abs"].sum())},
            {"metric": "avg_pnl_abs", "value": float(trades["pnl_abs"].mean())},
            {"metric": "avg_pnl_r", "value": float(trades["pnl_r"].mean())},
        ]
    )
    return summary


def main() -> None:
    results = run_backtest()
    results_path = REPORTS_DIR / "poc_backtest_results.csv"
    summary_path = REPORTS
