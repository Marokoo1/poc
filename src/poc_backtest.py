from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from poc_calculator import calculate_period_poc, filter_complete_periods

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TRADES_FILE = PROCESSED_DIR / "poc_backtest_trades.csv"
SUMMARY_FILE = PROCESSED_DIR / "poc_backtest_summary.csv"
LEVELS_FILE = PROCESSED_DIR / "poc_backtest_levels.csv"

# ============================================================
# SETTINGS
# ============================================================
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

INCLUDE_PERIODS = ("weekly", "monthly", "yearly")
MIN_LEVEL_AGE_BARS = 1

ENTRY_BUFFER_ATR = 0.30

PERIOD_PARAMS = {
    "weekly": {
        "stop_atr": 1.0,
        "target_atr": 1.0,
        "max_hold_bars": 10,
    },
    "monthly": {
        "stop_atr": 1.5,
        "target_atr": 2.0,
        "max_hold_bars": 20,
    },
    "yearly": {
        "stop_atr": 2.0,
        "target_atr": 3.0,
        "max_hold_bars": 40,
    },
}

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
    active_from: str

    touch_date: Optional[str]
    entry_date: Optional[str]
    exit_date: Optional[str]

    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]

    trend_context: str
    trend_aligned: bool

    exit_reason: str
    bars_held: Optional[int]

    pnl_abs: Optional[float]
    pnl_atr: Optional[float]
    return_pct: Optional[float]

    mfe_abs: Optional[float]
    mae_abs: Optional[float]


# ============================================================
# LOADERS / INDICATORS
# ============================================================
def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} nemá požadované sloupce: {sorted(missing)}")

    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()


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
    out["EMA50"] = out["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    close = out["Close"]
    ema50 = out["EMA50"]
    ema200 = out["EMA200"]

    out["TrendContext"] = np.where(
        (close > ema50) & (ema50 > ema200),
        "up",
        np.where((close < ema50) & (ema50 < ema200), "down", "neutral"),
    )

    return out


# ============================================================
# HISTORICAL LEVEL GENERATION
# ============================================================
def build_all_levels_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for period_type in INCLUDE_PERIODS:
        poc_df = calculate_period_poc(price_df.copy(), period=period_type)
        if poc_df.empty:
            continue

        poc_df = filter_complete_periods(poc_df, period=period_type)
        if poc_df.empty:
            continue

        poc_df["Ticker"] = ticker
        poc_df["PeriodType"] = period_type
        poc_df["PeriodStart"] = pd.to_datetime(poc_df["PeriodStart"], errors="coerce")
        poc_df["PeriodEnd"] = pd.to_datetime(poc_df["PeriodEnd"], errors="coerce")
        poc_df["ActiveFrom"] = poc_df["PeriodEnd"] + pd.Timedelta(days=1)

        frames.append(poc_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ActiveFrom", "PeriodType", "Period"]).reset_index(drop=True)
    return out


# ============================================================
# BACKTEST HELPERS
# ============================================================
def get_first_row_on_or_after(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[int]:
    matches = df.index[df["Date"] >= dt].tolist()
    return matches[0] if matches else None


def infer_side_from_activation(close_at_activation: float, level: float) -> str:
    if pd.isna(close_at_activation) or pd.isna(level):
        return "unknown"
    if close_at_activation > level:
        return "long"
    if close_at_activation < level:
        return "short"
    return "at_level"


def is_trend_aligned(side: str, trend_context: str) -> bool:
    return (side == "long" and trend_context == "up") or (
        side == "short" and trend_context == "down"
    )


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


def get_period_params(period_type: str) -> dict:
    if period_type not in PERIOD_PARAMS:
        raise ValueError(f"Neznámé PeriodType pro params: {period_type}")
    return PERIOD_PARAMS[period_type]


def simulate_single_level(level_row: pd.Series, ohlcv: pd.DataFrame) -> TradeResult:
    ticker = str(level_row["Ticker"])
    period_type = str(level_row["PeriodType"])
    period = str(level_row["Period"])
    level = float(level_row["POC"])
    active_from = pd.Timestamp(level_row["ActiveFrom"])

    params = get_period_params(period_type)
    stop_atr = float(params["stop_atr"])
    target_atr = float(params["target_atr"])
    max_hold_bars = int(params["max_hold_bars"])

    start_idx = get_first_row_on_or_after(ohlcv, active_from)
    if start_idx is None:
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side="unknown",
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            trend_context="unknown",
            trend_aligned=False,
            exit_reason="no_data_after_active_from",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    search_idx = start_idx + MIN_LEVEL_AGE_BARS
    if search_idx >= len(ohlcv):
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side="unknown",
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            trend_context="unknown",
            trend_aligned=False,
            exit_reason="not_enough_future_bars",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    activation_row = ohlcv.iloc[start_idx]
    side = infer_side_from_activation(float(activation_row["Close"]), level)
    trend_context = str(activation_row["TrendContext"]).lower()
    trend_aligned = is_trend_aligned(side, trend_context)

    if side not in {"long", "short"}:
        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            active_from=active_from.date().isoformat(),
            touch_date=None,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stop_price=None,
            target_price=None,
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            exit_reason="invalid_side",
            bars_held=None,
            pnl_abs=None,
            pnl_atr=None,
            return_pct=None,
            mfe_abs=None,
            mae_abs=None,
        )

    future = ohlcv.iloc[search_idx:].copy()

    for i in range(len(future)):
        touch_bar = future.iloc[i]
        atr = touch_bar["ATR"]

        if pd.isna(atr) or atr <= 0:
            continue

        if not side_touch(touch_bar, level, side, float(atr)):
            continue

        touch_date = pd.Timestamp(touch_bar["Date"])
        entry_date = touch_date
        entry_price = level

        if side == "long":
            stop_price = entry_price - stop_atr * float(atr)
            target_price = entry_price + target_atr * float(atr)
        else:
            stop_price = entry_price + stop_atr * float(atr)
            target_price = entry_price - target_atr * float(atr)

        # Den vstupu:
        # - SL může být zasažen hned
        # - PT ve stejný den IGNORUJEME
        touch_high = float(touch_bar["High"])
        touch_low = float(touch_bar["Low"])

        if side == "long":
            same_day_stop_hit = touch_low <= stop_price
        else:
            same_day_stop_hit = touch_high >= stop_price

        if same_day_stop_hit:
            pnl_abs = (stop_price - entry_price) if side == "long" else (entry_price - stop_price)
            pnl_atr = pnl_abs / float(atr) if atr > 0 else np.nan
            return_pct = (pnl_abs / entry_price) * 100 if entry_price else np.nan

            held_window = future.iloc[i : i + 1].copy()
            mfe_abs, mae_abs = compute_mfe_mae(held_window, entry_price, side)

            return TradeResult(
                ticker=ticker,
                period_type=period_type,
                period=period,
                level_price=level,
                side=side,
                active_from=active_from.date().isoformat(),
                touch_date=touch_date.date().isoformat(),
                entry_date=entry_date.date().isoformat(),
                exit_date=touch_date.date().isoformat(),
                entry_price=round(entry_price, 6),
                exit_price=round(float(stop_price), 6),
                stop_price=round(stop_price, 6),
                target_price=round(target_price, 6),
                trend_context=trend_context,
                trend_aligned=trend_aligned,
                exit_reason="stop",
                bars_held=0,
                pnl_abs=round(float(pnl_abs), 6),
                pnl_atr=round(float(pnl_atr), 6) if pd.notna(pnl_atr) else np.nan,
                return_pct=round(float(return_pct), 6) if pd.notna(return_pct) else np.nan,
                mfe_abs=round(float(mfe_abs), 6) if pd.notna(mfe_abs) else np.nan,
                mae_abs=round(float(mae_abs), 6) if pd.notna(mae_abs) else np.nan,
            )

        # Od dalšího dne už sledujeme normálně SL i PT
        post_entry_window = future.iloc[i + 1 : i + 1 + max_hold_bars].copy()

        if post_entry_window.empty:
            held_window = future.iloc[i : i + 1].copy()
            mfe_abs, mae_abs = compute_mfe_mae(held_window, entry_price, side)

            return TradeResult(
                ticker=ticker,
                period_type=period_type,
                period=period,
                level_price=level,
                side=side,
                active_from=active_from.date().isoformat(),
                touch_date=touch_date.date().isoformat(),
                entry_date=entry_date.date().isoformat(),
                exit_date=touch_date.date().isoformat(),
                entry_price=round(entry_price, 6),
                exit_price=round(float(touch_bar["Close"]), 6),
                stop_price=round(stop_price, 6),
                target_price=round(target_price, 6),
                trend_context=trend_context,
                trend_aligned=trend_aligned,
                exit_reason="time",
                bars_held=0,
                pnl_abs=round(float((float(touch_bar["Close"]) - entry_price) if side == "long" else (entry_price - float(touch_bar["Close"]))), 6),
                pnl_atr=np.nan,
                return_pct=round(float((((float(touch_bar["Close"]) - entry_price) if side == "long" else (entry_price - float(touch_bar["Close"]))) / entry_price) * 100), 6),
                mfe_abs=round(float(mfe_abs), 6) if pd.notna(mfe_abs) else np.nan,
                mae_abs=round(float(mae_abs), 6) if pd.notna(mae_abs) else np.nan,
            )

        exit_reason = "time"
        last_bar = post_entry_window.iloc[-1]
        exit_date = pd.Timestamp(last_bar["Date"])
        exit_price = float(last_bar["Close"])

        for j in range(len(post_entry_window)):
            row = post_entry_window.iloc[j]
            dt = pd.Timestamp(row["Date"])
            high = float(row["High"])
            low = float(row["Low"])

            if side == "long":
                hit_stop = low <= stop_price
                hit_target = high >= target_price
            else:
                hit_stop = high >= stop_price
                hit_target = low <= target_price

            if hit_stop and hit_target:
                # konzervativně: ambiguous = stop
                exit_reason = "stop"
                exit_date = dt
                exit_price = stop_price
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

        held_window = future.iloc[i:].copy()
        held_window = held_window.loc[held_window["Date"] <= exit_date]
        mfe_abs, mae_abs = compute_mfe_mae(held_window, entry_price, side)

        pnl_abs = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
        pnl_atr = pnl_abs / float(atr) if atr > 0 else np.nan
        return_pct = (pnl_abs / entry_price) * 100 if entry_price else np.nan

        bars_held = max((exit_date - entry_date).days, 0)

        return TradeResult(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level_price=level,
            side=side,
            active_from=active_from.date().isoformat(),
            touch_date=touch_date.date().isoformat(),
            entry_date=entry_date.date().isoformat(),
            exit_date=exit_date.date().isoformat(),
            entry_price=round(entry_price, 6),
            exit_price=round(float(exit_price), 6),
            stop_price=round(stop_price, 6),
            target_price=round(target_price, 6),
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            exit_reason=exit_reason,
            bars_held=bars_held,
            pnl_abs=round(float(pnl_abs), 6),
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
        active_from=active_from.date().isoformat(),
        touch_date=None,
        entry_date=None,
        exit_date=None,
        entry_price=None,
        exit_price=None,
        stop_price=None,
        target_price=None,
        trend_context=trend_context,
        trend_aligned=trend_aligned,
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

    if df.empty:
        return pd.DataFrame()

    df["win"] = df["pnl_abs"] > 0

    summary = (
        df.groupby(["period_type", "side", "trend_aligned"], dropna=False, observed=False)
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


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"V {RAW_DIR} nebyly nalezeny žádné raw CSV soubory.")

    all_levels: list[pd.DataFrame] = []
    all_results: list[dict] = []

    tickers = [p.stem for p in raw_files]
    print(f"🎯 Historický backtest tickerů: {', '.join(tickers)}")

    for path in raw_files:
        ticker = path.stem
        print(f"Zpracovávám {ticker}...")

        ohlcv = load_ohlcv(ticker)
        if ohlcv.empty:
            print(f"  ⚠️ Chybí nebo jsou neplatná raw data")
            continue

        ohlcv = add_indicators(ohlcv)

        levels = build_all_levels_for_ticker(ticker, ohlcv)
        if levels.empty:
            print(f"  ⚠️ Nevznikly žádné historické levely")
            continue

        all_levels.append(levels)

        for _, level_row in levels.iterrows():
            result = simulate_single_level(level_row, ohlcv)
            all_results.append(asdict(result))

        print(f"  ✅ Historických levelů: {len(levels)}")

    if not all_results:
        print("⚠️ Nevznikly žádné výsledky.")
        return

    levels_df = pd.concat(all_levels, ignore_index=True) if all_levels else pd.DataFrame()
    trades_df = pd.DataFrame(all_results)
    summary_df = build_summary(trades_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    levels_df.to_csv(LEVELS_FILE, index=False)
    trades_df.to_csv(TRADES_FILE, index=False)
    if not summary_df.empty:
        summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print(f"✅ Levels uloženy do: {LEVELS_FILE}")
    print(f"✅ Trades uloženy do: {TRADES_FILE}")
    if not summary_df.empty:
        print(f"✅ Summary uloženo do: {SUMMARY_FILE}")

    print("\nUkázka obchodů:")
    preview_cols = [
        "ticker",
        "period_type",
        "period",
        "side",
        "entry_date",
        "exit_date",
        "exit_reason",
        "pnl_abs",
        "pnl_atr",
    ]
    preview_cols = [c for c in preview_cols if c in trades_df.columns]
    print(trades_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
