from __future__ import annotations

from dataclasses import dataclass
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
# BACKTEST SETTINGS
# ============================================================
ATR_PERIOD = 14

MIN_SCORE = 50
ONLY_VALID_NOW = True
ONLY_TREND_ALIGNED = False

ENTRY_BUFFER_ATR = 0.15
STOP_ATR = 1.00
TARGET_ATR = 1.50
MAX_HOLD_BARS = 10

# ambivalentní svíčka: zasáhne zároveň SL i PT
AMBIGUOUS_EXIT = "ambiguous"

# ============================================================
# DATA MODELS
# ============================================================
@dataclass
class TradeResult:
    ticker: str
    period_type: str
    period: str
    level_price: float
    side: str
    score: int
    trend_aligned: bool

    active_from: Optional[pd.Timestamp]
    touch_date: Optional[pd.Timestamp]
    entry_date: Optional[pd.Timestamp]
    exit_date: Optional[pd.Timestamp]

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
        "Ticker", "PeriodType", "Period", "POC",
        "LevelPrice", "LevelSide", "ActiveFrom",
        "ValidNow", "Score", "TrendAligned"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Enriched CSV nemá očekávané sloupce: {sorted(missing)}")

    for col in ["ActiveFrom", "FirstTestDate", "PeriodStart", "PeriodEnd"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} nemá požadované sloupce: {sorted(missing)}")

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

    return out


def first_index_on_or_after(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    valid = index[index >= dt]
    return valid[0] if len(valid) else None


def side_touch(row: pd.Series, level: float, side: str, atr: float) -> bool:
    buffer_abs = ENTRY_BUFFER_ATR * atr

    if side == "long":
        return float(row["Low"]) <= level + buffer_abs
    if side == "short":
        return float(row["High"]) >= level - buffer_abs

    return False


def compute_mfe_mae(window: pd.DataFrame, entry: float, side: str) -> tuple[float, float]:
    if side == "long":
        mfe = float(window["High"].max()) - entry
        mae = float(window["Low"].min()) - entry
    else:
        mfe = entry - float(window["Low"].min())
        mae = entry - float(window["High"].max())

    return round(mfe, 6), round(mae, 6)


def simulate_trade(
    ticker: str,
    level_row: pd.Series,
    ohlcv: pd.DataFrame,
) -> TradeResult:
    side = str(level_row["LevelSide"]).lower()
    level = float(level_row["LevelPrice"])
    score = int(level_row["Score"])
    trend_aligned = bool(level_row["TrendAligned"])
    period_type = str(level_row["PeriodType"])
    period = str(level_row["Period"])

    active_from = pd.Timestamp(level_row["ActiveFrom"]) if pd.notna(level_row["ActiveFrom"]) else None
    if active_from is None:
        return TradeResult(
            ticker, period_type, period, level, side, score, trend_aligned,
            None, None, None, None,
            None, None, None, None,
            "no_active_from", None,
            None, None, None,
            None, None
        )

    start_dt = first_index_on_or_after(ohlcv.index, active_from)
    if start_dt is None:
        return TradeResult(
            ticker, period_type, period, level, side, score, trend_aligned,
            active_from, None, None, None,
            None, None, None, None,
            "no_data_after_active_from", None,
            None, None, None,
            None, None
        )

    df = ohlcv.loc[ohlcv.index >= start_dt].copy()
    if df.empty:
        return TradeResult(
            ticker, period_type, period, level, side, score, trend_aligned,
            active_from, None, None, None,
            None, None, None, None,
            "empty_window", None,
            None, None, None,
            None, None
        )

    # 1) hledej první touch
    for i in range(len(df)):
        row = df.iloc[i]
        atr = row["ATR"]

        if pd.isna(atr) or atr <= 0:
            continue

        if not side_touch(row, level, side, float(atr)):
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

        # 2) sleduj další bary až do exit
        trade_window = df.iloc[i : i + MAX_HOLD_BARS + 1].copy()
        if trade_window.empty:
            break

        exit_reason = "time"
        exit_date = trade_window.index[-1]
        exit_price = float(trade_window.iloc[-1]["Close"])

        for j in range(len(trade_window)):
            bar = trade_window.iloc[j]
            bar_dt = trade_window.index[j]

            high = float(bar["High"])
            low = float(bar["Low"])

            if side == "long":
                hit_stop = low <= stop_price
                hit_target = high >= target_price
            else:
                hit_stop = high >= stop_price
                hit_target = low <= target_price

            if hit_stop and hit_target:
                exit_reason = AMBIGUOUS_EXIT
                exit_date = bar_dt
                exit_price = np.nan
                break
            elif hit_stop:
                exit_reason = "stop"
                exit_date = bar_dt
                exit_price = stop_price
                break
            elif hit_target:
                exit_reason = "target"
                exit_date = bar_dt
                exit_price = target_price
                break

        held_window = trade_window.loc[:exit_date] if pd.notna(exit_price) else trade_window.loc[:exit_date]
        mfe_abs, mae_abs = compute_mfe_mae(held_window, entry_price, side)

        if pd.isna(exit_price):
            pnl_abs = np.nan
            pnl_atr = np.nan
            return_pct = np.nan
        else:
            if side == "long":
                pnl_abs = exit_price - entry_price
            else:
                pnl_abs = entry_price - exit_price

            pnl_atr = pnl_abs / float(atr) if atr > 0 else np.nan
            return_pct = (pnl_abs / entry_price) * 100 if entry_price else np.nan

        bars_held = len(held_window) - 1

        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            score=score,
            trend_aligned=trend_aligned,

            active_from=active_from,
            touch_date=touch_date,
            entry_date=entry_date,
            exit_date=exit_date,

            entry_price=round(entry_price, 6),
            exit_price=round(float(exit_price), 6) if pd.notna(exit_price) else np.nan,
            stop_price=round(stop_price, 6),
            target_price=round(target_price, 6),

            exit_reason=exit_reason,
            bars_held=bars_held,

            pnl_abs=round(float(pnl_abs), 6) if pd.notna(pnl_abs) else np.nan,
            pnl_atr=round(float(pnl_atr), 6) if pd.notna(pnl_atr) else np.nan,
            return_pct=round(float(return_pct), 6) if pd.notna(return_pct) else np.nan,

            mfe_abs=round(mfe_abs, 6),
            mae_abs=round(mae_abs, 6),
        )

    return TradeResult(
        ticker=ticker,
        period_type=period_type,
        period=period,
        level_price=level,
        side=side,
        score=score,
        trend_aligned=trend_aligned,

        active_from=active_from,
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
    df = trades.copy()

    # jen skutečné obchody s exitem
    df = df[df["entry_date"].notna()].copy()
    df = df[df["exit_reason"] != AMBIGUOUS_EXIT].copy()

    if df.empty:
        return pd.DataFrame()

    df["win"] = df["pnl_abs"] > 0

    grouped = (
        df.groupby(["period_type", "side", "trend_aligned"], dropna=False)
        .agg(
            trades=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_pnl_abs=("pnl_abs", "mean"),
            median_pnl_abs=("pnl_abs", "median"),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_pnl_atr=("pnl_atr", "mean"),
            avg_mfe=("mfe_abs", "mean"),
            avg_mae=("mae_abs", "mean"),
            avg_bars_held=("bars_held", "mean"),
        )
        .reset_index()
    )

    grouped["win_rate"] = (grouped["win_rate"] * 100).round(2)
    return grouped


def main() -> None:
    levels = load_enriched_levels(ENRICHED_FILE)

    if ONLY_VALID_NOW:
        levels = levels[levels["ValidNow"] == True].copy()

    levels = levels[levels["LevelSide"].isin(["long", "short"])].copy()
    levels = levels[levels["Score"] >= MIN_SCORE].copy()

    if ONLY_TREND_ALIGNED:
        levels = levels[levels["TrendAligned"] == True].copy()

    if levels.empty:
        print("⚠️ Po aplikaci filtrů nezůstaly žádné levely.")
        return

    all_results: list[dict] = []

    tickers = sorted(levels["Ticker"].dropna().astype(str).unique().tolist())
    print(f"🎯 Backtest tickerů: {', '.join(tickers)}")

    for ticker in tickers:
        print(f"Zpracovávám {ticker}...")
        ohlcv = load_ohlcv(ticker)

        if ohlcv.empty:
            print(f"  ⚠️ Chybí raw data pro {ticker}")
            continue

        ohlcv = add_indicators(ohlcv)
        ticker_levels = levels[levels["Ticker"] == ticker].copy()

        for _, level_row in ticker_levels.iterrows():
            result = simulate_trade(ticker, level_row, ohlcv)
            all_results.append(result.__dict__)

        print(f"  ✅ Levelů: {len(ticker_levels)}")

    if not all_results:
        print("⚠️ Nevznikly žádné výsledky.")
        return

    trades = pd.DataFrame(all_results)
    summary = build_summary(trades)

    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_FILE, index=False)

    if not summary.empty:
        summary.to_csv(SUMMARY_FILE, index=False)

    print(f"\n✅ Trades uložen do: {TRADES_FILE}")
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
