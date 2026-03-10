from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

ENRICHED_FILE = PROCESSED_DIR / "poc_levels_enriched.csv"
TRADES_FILE = PROCESSED_DIR / "poc_backtest_trades.csv"
SUMMARY_FILE = PROCESSED_DIR / "poc_backtest_summary.csv"

# ============================================================
# SETTINGS
# ============================================================
ATR_PERIOD = 14

MIN_SCORE = 50
ONLY_UNTESTED = False
ONLY_VALID_SIDES = True
ONLY_TREND_ALIGNED = False

ENTRY_BUFFER_ATR = 0.15
STOP_ATR = 1.00
TARGET_ATR = 1.50
MAX_HOLD_BARS = 10

AMBIGUOUS_EXIT = "ambiguous"

# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class TradeResult:
    ticker: str
    period_type: str
    period: str
    level_price: float
    side: str
    score: int
    trend_context: str
    trend_aligned: bool

    active_from: Optional[str]
    touch_date: Optional[str]
    entry_date: Optional[str]
    exit_date: Optional[str]

    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]

    exit_reason: str
    bars_held: Optional[int]

    pnl_abs: Optional[float]
    pnl_atr: Optional[float]
    return_pct: Optional[float]

    mfe_abs: Optional[float]
    mae_abs: Optional[float]


# ============================================================
# HELPERS
# ============================================================
def load_enriched_levels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {path}")

    df = pd.read_csv(path)

    required = {
        "Ticker",
        "PeriodType",
        "Period",
        "POC",
        "LevelSide",
        "IsTested",
        "Score",
        "TrendContext",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Enriched CSV nemá očekávané sloupce: {sorted(missing)}")

    for col in ["FirstTestDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} nemá požadované sloupce: {sorted(missing)}")

    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    prev_close = out["Close"].shift(1)
    tr_components = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["TR"] = tr_components.max(axis=1)
    out["ATR"] = out["TR"].rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()
    return out


def period_to_active_from(period_type: str, period: str) -> pd.Timestamp:
    period_type = str(period_type).lower()

    if period_type == "weekly":
        p = pd.Period(period, freq="W")
        return p.end_time.normalize() + pd.Timedelta(days=1)

    if period_type == "monthly":
        p = pd.Period(period, freq="M")
        return p.end_time.normalize() + pd.Timedelta(days=1)

    if period_type == "yearly":
        p = pd.Period(period, freq="Y")
        return p.end_time.normalize() + pd.Timedelta(days=1)

    raise ValueError(f"Neznámý PeriodType: {period_type}")


def first_index_on_or_after(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    valid = index[index >= dt]
    return valid[0] if len(valid) else None


def infer_trend_aligned(side: str, trend_context: str) -> bool:
    if side == "long" and trend_context == "up":
        return True
    if side == "short" and trend_context == "down":
        return True
    return False


def side_touch(bar: pd.Series, level: float, side: str, atr: float) -> bool:
    buffer_abs = ENTRY_BUFFER_ATR * atr

    if side == "long":
        return float(bar["Low"]) <= level + buffer_abs

    if side == "short":
        return float(bar["High"]) >= level - buffer_abs

    return False


def compute_mfe_mae(window: pd.DataFrame, entry: float, side: str) -> tuple[float, float]:
    if window.empty:
        return np.nan, np.nan

    if side == "long":
        mfe = float(window["High"].max()) - entry
        mae = float(window["Low"].min()) - entry
    else:
        mfe = entry - float(window["Low"].min())
        mae = entry - float(window["High"].max())

    return round(mfe, 6), round(mae, 6)


def simulate_trade(level_row: pd.Series, ohlcv: pd.DataFrame) -> TradeResult:
    ticker = str(level_row["Ticker"])
    period_type = str(level_row["PeriodType"])
    period = str(level_row["Period"])
    level = float(level_row["POC"])
    side = str(level_row["LevelSide"]).lower()
    score = int(level_row["Score"])
    trend_context = str(level_row["TrendContext"]).lower()
    trend_aligned = infer_trend_aligned(side, trend_context)

    active_from = period_to_active_from(period_type, period)
    start_dt = first_index_on_or_after(ohlcv.index, active_from)

    if side not in {"long", "short"}:
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            score=score,
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            exit_reason="invalid_side",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    if start_dt is None:
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            score=score,
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            exit_reason="no_data_after_active_from",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    df = ohlcv.loc[ohlcv.index >= start_dt].copy()
    if df.empty:
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            score=score,
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            exit_reason="empty_window",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    for i in range(len(df)):
        bar = df.iloc[i]
        atr = bar["ATR"]

        if pd.isna(atr) or atr <= 0:
            continue

        if not side_touch(bar, level, side, float(atr)):
            continue

        touch_date = df.index[i]
        entry_date = touch_date
        entry_price = level

        if side == "long":
            stop_price = entry_price - STOP_ATR * float(atr)
            target_price = entry_price + TARGET_ATR * float(atr)
        else:
            stop_price = entry_price + STOP_ATR * float(atr)
            target_price = entry_price - TARGET_ATR * float(atr)

        trade_window = df.iloc[i : i + MAX_HOLD_BARS + 1].copy()
        exit_reason = "time"
        exit_date = trade_window.index[-1]
        exit_price = float(trade_window.iloc[-1]["Close"])

        for j in range(len(trade_window)):
            row = trade_window.iloc[j]
            dt = trade_window.index[j]

            high = float(row["High"])
            low = float(row["Low"])

            if side == "long":
                hit_stop = low <= stop_price
                hit_target = high >= target_price
            else:
                hit_stop = high >= stop_price
                hit_target = low <= target_price

            if hit_stop and hit_target:
                exit_reason = AMBIGUOUS_EXIT
                exit_date = dt
                exit_price = np.nan
                break
            elif hit_stop:
                exit_reason = "stop"
                exit_date = dt
                exit_price = stop_price
                break
            elif hit_target:
                exit_reason = "target"
                exit_date = dt
                exit_price = target_price
                break

        held_window = trade_window.loc[:exit_date]
        mfe_abs, mae_abs = compute_mfe_mae(held_window, entry_price, side)

        if pd.isna(exit_price):
            pnl_abs = np.nan
            pnl_atr = np.nan
            return_pct = np.nan
        else:
            pnl_abs = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
            pnl_atr = pnl_abs / float(atr) if atr > 0 else np.nan
            return_pct = (pnl_abs / entry_price) * 100 if entry_price else np.nan

        bars_held = max(len(held_window) - 1, 0)

        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            score=score,
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            active_from=active_from.date().isoformat(),
            touch_date=touch_date.date().isoformat(),
            entry_date=entry_date.date().isoformat(),
            exit_date=exit_date.date().isoformat(),
            entry_price=round(entry_price, 6),
            exit_price=round(float(exit_price), 6) if pd.notna(exit_price) else np.nan,
            stop_price=round(stop_price, 6),
            target_price=round(target_price, 6),
            exit_reason=exit_reason,
            bars_held=bars_held,
            pnl_abs=round(float(pnl_abs), 6) if pd.notna(pnl_abs) else np.nan,
            pnl_atr=round(float(pnl_atr), 6) if pd.notna(pnl_atr) else np.nan,
            return_pct=round(float(return_pct), 6) if pd.notna(return_pct) else np.nan,
            mfe_abs=round(float(mfe_abs), 6) if pd.notna(mfe_abs) else np.nan,
            mae_abs=round(float(mae_abs), 6) if pd.notna(mae_abs) else np.nan,
        )

    return TradeResult(
        ticker=ticker,
        period_type=period_type,
        period=period,
        level_price=level,
        side=side,
        score=score,
        trend_context=trend_context,
        trend_aligned=trend_aligned,
        active_from=active_from.date().isoformat(),
        touch_date=None,
        entry_date=None,
        exit_date=None,
        entry_price=None,
        exit_price=None,
        stop_price=None,
        target_price=None,
        exit_reason="no_touch",
        bars_held=None,
        pnl_abs=None,
        pnl_atr=None,
        return_pct=None,
        mfe_abs=None,
        mae_abs=None,
    )


def build_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df = df[df["entry_date"].notna()].copy()
    df = df[df["exit_reason"] != AMBIGUOUS_EXIT].copy()

    if df.empty:
        return pd.DataFrame()

    df["win"] = df["pnl_abs"] > 0

    score_bins = [-np.inf, 39, 49, 59, 69, 79, np.inf]
    score_labels = ["<=39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df["score_bucket"] = pd.cut(df["score"], bins=score_bins, labels=score_labels)

    summary = (
        df.groupby(["period_type", "side", "trend_aligned", "score_bucket"], dropna=False)
        .agg(
            trades=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_pnl_abs=("pnl_abs", "mean"),
            median_pnl_abs=("pnl_abs", "median"),
            avg_pnl_atr=("pnl_atr", "mean"),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_mfe=("mfe_abs", "mean"),
            avg_mae=("mae_abs", "mean"),
            avg_bars_held=("bars_held", "mean"),
        )
        .reset_index()
    )

    summary["win_rate"] = (summary["win_rate"] * 100).round(2)
    return summary


def main() -> None:
    levels = load_enriched_levels(ENRICHED_FILE)

    if ONLY_VALID_SIDES:
        levels = levels[levels["LevelSide"].isin(["long", "short"])].copy()

    if ONLY_UNTESTED:
        levels = levels[levels["IsTested"] == False].copy()

    levels = levels[levels["Score"] >= MIN_SCORE].copy()

    if ONLY_TREND_ALIGNED:
        trend_mask = (
            ((levels["LevelSide"] == "long") & (levels["TrendContext"] == "up")) |
            ((levels["LevelSide"] == "short") & (levels["TrendContext"] == "down"))
        )
        levels = levels[trend_mask].copy()

    if levels.empty:
        print("⚠️ Po aplikaci filtrů nezůstaly žádné levely.")
        return

    tickers = sorted(levels["Ticker"].dropna().astype(str).unique().tolist())
    print(f"🎯 Backtest tickerů: {', '.join(tickers)}")

    results: list[dict] = []

    for ticker in tickers:
        print(f"Zpracovávám {ticker}...")
        ohlcv = load_ohlcv(ticker)

        if ohlcv.empty:
            print(f"  ⚠️ Chybí raw data pro {ticker}")
            continue

        ohlcv = add_indicators(ohlcv)
        ticker_levels = levels[levels["Ticker"] == ticker].copy()

        for _, level_row in ticker_levels.iterrows():
            result = simulate_trade(level_row, ohlcv)
            results.append(asdict(result))

        print(f"  ✅ Otestováno levelů: {len(ticker_levels)}")

    if not results:
        print("⚠️ Nevznikly žádné výsledky.")
        return

    trades = pd.DataFrame(results)
    summary = build_summary(trades)

    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_FILE, index=False)

    if not summary.empty:
        summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print(f"✅ Trades uloženy do: {TRADES_FILE}")
    if not summary.empty:
        print(f"✅ Summary uloženo do: {SUMMARY_FILE}")

    preview_cols = [
        "ticker", "period_type", "period", "side", "score",
        "entry_date", "exit_date", "exit_reason", "pnl_abs", "pnl_atr"
    ]
    preview_cols = [c for c in preview_cols if c in trades.columns]

    print("\nUkázka obchodů:")
    print(trades[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
