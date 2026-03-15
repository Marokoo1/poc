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

INCLUDE_PERIODS = ( "monthly", "yearly") #"weekly",
MIN_LEVEL_AGE_BARS = 3


# ============================================================
# Backtest preset configuration
# ============================================================

SUPERSESSION_ENABLED = True

SUPERSESSION_THRESHOLD_ATR = {
    "weekly": 0.75,
    "monthly": 1.00,
    "yearly": 1.25,
}


ENTRY_BUFFER_ATR = 0.20
BACKTEST_PRESET = "balanced"

PRESET_LIBRARY = {
    "loose": {
        "weekly": {
            "stop_atr": 1.3,
            "target_atr": 1.8,
            "max_hold_bars": 16,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 2.40,
        },
        "monthly": {
            "stop_atr": 1.8,
            "target_atr": 2.7,
            "max_hold_bars": 28,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 2.70,
        },
        "yearly": {
            "stop_atr": 2.7,
            "target_atr": 4.0,
            "max_hold_bars": 50,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 4.50,
        },
    },

    "balanced": {
        "weekly": {
            "stop_atr": 1.5,
            "target_atr": 2.0,
            "max_hold_bars": 20,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 3.00,
        },
        "monthly": {
            "stop_atr": 2.0,
            "target_atr": 3.0,
            "max_hold_bars": 35,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 3.30,
        },
        "yearly": {
            "stop_atr": 3.0,
            "target_atr": 4.5,
            "max_hold_bars": 60,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 5.5,
        },
    },

    "strict": {
        "weekly": {
            "stop_atr": 1.7,
            "target_atr": 2.3,
            "max_hold_bars": 24,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 3.60,
        },
        "monthly": {
            "stop_atr": 2.3,
            "target_atr": 3.5,
            "max_hold_bars": 42,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 4.00,
        },
        "yearly": {
            "stop_atr": 3.5,
            "target_atr": 5.2,
            "max_hold_bars": 75,
            "require_departure": True,
            "activation_threshold_mode": "atr",
            "activation_threshold_value": 6.50,
        },
    },
}

if BACKTEST_PRESET not in PRESET_LIBRARY:
    raise ValueError(f"Unknown BACKTEST_PRESET: {BACKTEST_PRESET}")

PERIOD_PARAMS = PRESET_LIBRARY[BACKTEST_PRESET]

print(f"Using backtest preset: {BACKTEST_PRESET}")
print(f"ENTRY_BUFFER_ATR: {ENTRY_BUFFER_ATR}")

if BACKTEST_PRESET not in PRESET_LIBRARY:
    raise ValueError(f"Unknown BACKTEST_PRESET: {BACKTEST_PRESET}")

PERIOD_PARAMS = PRESET_LIBRARY[BACKTEST_PRESET]

print(f"Using backtest preset: {BACKTEST_PRESET}")

AMBIGUOUS_EXIT = "ambiguous"

FIRST_TOUCH_ONLY = True
MAX_ENTRIES_PER_TICKER_PER_DAY = 1
DAILY_LIMIT_EXIT_REASON = "daily_limit"
FIRST_TOUCH_INVALID_EXIT_REASON = "first_touch_invalid"

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

def make_empty_trade_result(
    *,
    ticker: str,
    period_type: str,
    period: str,
    level: float,
    side: str,
    active_from: pd.Timestamp,
    trend_context: str,
    exit_reason: str,
) -> TradeResult:
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
        trend_aligned=False,
        exit_reason=exit_reason,
        bars_held=None,
        pnl_abs=None,
        pnl_atr=None,
        return_pct=None,
        mfe_abs=None,
        mae_abs=None,
    )


def touches_level_zone(bar: pd.Series, level: float, atr: float) -> bool:
    buffer_abs = ENTRY_BUFFER_ATR * float(atr)
    bar_high = float(bar["High"])
    bar_low = float(bar["Low"])
    return bar_low <= (level + buffer_abs) and bar_high >= (level - buffer_abs)




def detect_clean_touch(prev_bar: pd.Series, bar: pd.Series, level: float, atr: float) -> str | None:
    buffer_abs = ENTRY_BUFFER_ATR * atr

    prev_close = float(prev_bar["Close"])
    bar_open = float(bar["Open"])
    bar_high = float(bar["High"])
    bar_low = float(bar["Low"])

    # gap cross invalidace:
    # z nad levelu otevřít pod level = nebereme short
    # z pod levelu otevřít nad level = nebereme long
    if prev_close > level + buffer_abs and bar_open < level - buffer_abs:
        return "gap_cross"
    if prev_close < level - buffer_abs and bar_open > level + buffer_abs:
        return "gap_cross"

    # clean long = příchod shora
    if prev_close > level + buffer_abs and bar_open >= level - buffer_abs and bar_low <= level + buffer_abs:
        return "long"

    # clean short = příchod zdola
    if prev_close < level - buffer_abs and bar_open <= level + buffer_abs and bar_high >= level - buffer_abs:
        return "short"

    # bar už otevírá v zóně levelu = rotace / chop
    if (level - buffer_abs) <= bar_open <= (level + buffer_abs):
        return "rotation"

    return None


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

def compute_activation_threshold(level: float, atr: float, mode: str, value: float) -> float:
    mode = str(mode).lower()
    value = float(value)

    if value < 0:
        value = 0.0

    if mode == "atr":
        return value * float(atr)
    if mode == "pct":
        return abs(float(level)) * (value / 100.0)
    if mode == "absolute":
        return value

    raise ValueError(f"Neznámý activation_threshold_mode: {mode}")


def departure_reached(bar: pd.Series, level: float, side: str, threshold_abs: float) -> bool:
    if threshold_abs <= 0:
        return True

    high = float(bar["High"])
    low = float(bar["Low"])

    if side == "long":
        return high >= level + threshold_abs
    if side == "short":
        return low <= level - threshold_abs
    return False



def get_row_on_or_after(ohlcv: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    idx = get_first_row_on_or_after(ohlcv, dt)
    if idx is None:
        return None
    return ohlcv.iloc[idx]


def infer_level_side_at_active_from(level_row: pd.Series, ohlcv: pd.DataFrame) -> str:
    active_from = pd.Timestamp(level_row["ActiveFrom"])
    row = get_row_on_or_after(ohlcv, active_from)
    if row is None:
        return "unknown"

    close_px = float(row["Close"])
    level = float(level_row["POC"])
    return infer_side_from_activation(close_px, level)


def get_level_atr_at_active_from(level_row: pd.Series, ohlcv: pd.DataFrame) -> float | None:
    active_from = pd.Timestamp(level_row["ActiveFrom"])
    row = get_row_on_or_after(ohlcv, active_from)
    if row is None:
        return None

    atr = row.get("ATR")
    if pd.isna(atr) or atr is None or float(atr) <= 0:
        return None
    return float(atr)


def apply_level_supersession(levels: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if levels.empty or not SUPERSESSION_ENABLED:
        out = levels.copy()
        if "ValidUntil" not in out.columns:
            out["ValidUntil"] = pd.NaT
        return out

    out = levels.copy()
    out["ActiveFrom"] = pd.to_datetime(out["ActiveFrom"], errors="coerce")
    out["ValidUntil"] = pd.NaT

    # Metadata potřebná pro rozhodnutí, které levely jsou "stejná zóna".
    out["level_side"] = out.apply(lambda r: infer_level_side_at_active_from(r, ohlcv), axis=1)
    out["active_atr"] = out.apply(lambda r: get_level_atr_at_active_from(r, ohlcv), axis=1)

    out = out.sort_values(["ActiveFrom", "PeriodType", "POC"]).reset_index(drop=True)

    for new_idx in range(len(out)):
        new_row = out.iloc[new_idx]

        new_active_from = pd.Timestamp(new_row["ActiveFrom"])
        new_period_type = str(new_row["PeriodType"]).lower()
        new_side = str(new_row["level_side"]).lower()
        new_poc = float(new_row["POC"])
        new_atr = new_row["active_atr"]

        if pd.isna(new_active_from) or new_side == "unknown":
            continue

        threshold_mult = float(SUPERSESSION_THRESHOLD_ATR.get(new_period_type, 1.0))
        if pd.isna(new_atr) or new_atr is None or float(new_atr) <= 0:
            continue

        threshold_abs = float(new_atr) * threshold_mult

        # Hledej starší levely stejného tickeru / typu periody / směru ve stejné zóně.
        older_mask = (
            (out.index < new_idx)
            & (pd.to_datetime(out["ActiveFrom"], errors="coerce") < new_active_from)
            & (out["PeriodType"].astype(str).str.lower() == new_period_type)
            & (out["level_side"].astype(str).str.lower() == new_side)
        )

        if not older_mask.any():
            continue

        older_candidates = out.loc[older_mask].copy()
        if older_candidates.empty:
            continue

        older_candidates["price_distance"] = (older_candidates["POC"].astype(float) - new_poc).abs()
        same_zone = older_candidates["price_distance"] <= threshold_abs

        if not same_zone.any():
            continue

        same_zone_idx = older_candidates.loc[same_zone].index.tolist()

        for old_idx in same_zone_idx:
            old_valid_until = out.at[old_idx, "ValidUntil"]
            # Zkrať starý level jen dopředu v čase, nikdy ne zpětně.
            if pd.isna(old_valid_until) or pd.Timestamp(new_active_from) < pd.Timestamp(old_valid_until):
                out.at[old_idx, "ValidUntil"] = new_active_from

    out = out.drop(columns=["level_side", "active_atr"], errors="ignore")
    return out

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
    require_departure = bool(params.get("require_departure", False))
    activation_threshold_mode = str(params.get("activation_threshold_mode", "atr"))
    activation_threshold_value = float(params.get("activation_threshold_value", 0.0))

    start_idx = get_first_row_on_or_after(ohlcv, active_from)
    if start_idx is None:
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side="unknown",
            active_from=active_from,
            trend_context="unknown",
            exit_reason="no_data_after_active_from",
        )

    search_idx = start_idx + MIN_LEVEL_AGE_BARS
    if search_idx >= len(ohlcv):
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side="unknown",
            active_from=active_from,
            trend_context="unknown",
            exit_reason="not_enough_bars_after_active_from",
        )

    activation_row = ohlcv.iloc[start_idx]
    initial_side = infer_side_from_activation(float(activation_row["Close"]), level)
    trend_context = str(activation_row["TrendContext"]).lower()

    future = ohlcv.iloc[search_idx:].copy()
    if future.empty:
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason="no_future_bars",
        )

    entry_search_start = 0

    if require_departure:
        armed_index = None

        for i in range(len(future)):
            depart_bar = future.iloc[i]
            depart_atr = depart_bar["ATR"]

            if pd.isna(depart_atr) or depart_atr <= 0:
                continue

            threshold_abs = compute_activation_threshold(
                level=level,
                atr=float(depart_atr),
                mode=activation_threshold_mode,
                value=activation_threshold_value,
            )

            if departure_reached(depart_bar, level, initial_side, threshold_abs):
                armed_index = i + 1
                break

        if armed_index is None or armed_index >= len(future):
            return make_empty_trade_result(
                ticker=ticker,
                period_type=period_type,
                period=period,
                level=level,
                side=initial_side,
                active_from=active_from,
                trend_context=trend_context,
                exit_reason="no_departure",
            )

        entry_search_start = armed_index

    # ============================================================
    # FIRST TOUCH ONLY:
    # po departure bereme pouze první návrat ceny do level zóny
    # ============================================================
    
    first_touch_idx = None
    first_touch_bar = None
    first_touch_atr = None

    for i in range(entry_search_start, len(future)):
        touch_bar = future.iloc[i]
        atr = touch_bar["ATR"]

        if pd.isna(atr) or atr <= 0:
            continue

        if touches_level_zone(touch_bar, level, float(atr)):
            first_touch_idx = i
            first_touch_bar = touch_bar
            first_touch_atr = float(atr)
            break

    if first_touch_idx is None or first_touch_bar is None or first_touch_atr is None:
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason="no_touch",
        )

    if first_touch_idx == 0:
        if entry_search_start == 0:
            return make_empty_trade_result(
                ticker=ticker,
                period_type=period_type,
                period=period,
                level=level,
                side=initial_side,
                active_from=active_from,
                trend_context=trend_context,
                exit_reason=FIRST_TOUCH_INVALID_EXIT_REASON,
            )
        prev_bar = future.iloc[entry_search_start - 1]
    else:
        prev_bar = future.iloc[first_touch_idx - 1]

    touch_signal = detect_clean_touch(prev_bar, first_touch_bar, level, first_touch_atr)

    if touch_signal == "gap_cross":
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason="gap_cross",
        )

    if touch_signal == "rotation":
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason="rotation",
        )

    if touch_signal not in {"long", "short"}:
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason=FIRST_TOUCH_INVALID_EXIT_REASON,
        )

    side = touch_signal
    trend_aligned = is_trend_aligned(side, trend_context)

    touch_date = pd.Timestamp(first_touch_bar["Date"])
    entry_date = touch_date
    entry_price = level
    atr = first_touch_atr
    i = first_touch_idx

    if side == "long":
        stop_price = entry_price - stop_atr * atr
        target_price = entry_price + target_atr * atr
    else:
        stop_price = entry_price + stop_atr * atr
        target_price = entry_price - target_atr * atr

    # Den vstupu:
    # - SL může být zasažen hned
    # - PT ve stejný den ignorujeme
    touch_high = float(first_touch_bar["High"])
    touch_low = float(first_touch_bar["Low"])

    if side == "long":
        same_day_stop_hit = touch_low <= stop_price
    else:
        same_day_stop_hit = touch_high >= stop_price

    if same_day_stop_hit:
        pnl_abs = (stop_price - entry_price) if side == "long" else (entry_price - stop_price)
        pnl_atr = pnl_abs / atr if atr > 0 else np.nan
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

        close_px = float(first_touch_bar["Close"])
        pnl_abs = (close_px - entry_price) if side == "long" else (entry_price - close_px)
        return_pct = (pnl_abs / entry_price) * 100 if entry_price else np.nan

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
            exit_price=round(close_px, 6),
            stop_price=round(stop_price, 6),
            target_price=round(target_price, 6),
            trend_context=trend_context,
            trend_aligned=trend_aligned,
            exit_reason="time",
            bars_held=0,
            pnl_abs=round(float(pnl_abs), 6),
            pnl_atr=np.nan,
            return_pct=round(float(return_pct), 6) if pd.notna(return_pct) else np.nan,
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
    pnl_atr = pnl_abs / atr if atr > 0 else np.nan
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


def apply_daily_entry_limit(trades: pd.DataFrame, max_entries_per_day: int) -> pd.DataFrame:
    if trades.empty or max_entries_per_day <= 0:
        return trades

    df = trades.copy()

    if "entry_date" not in df.columns:
        return df

    entered_mask = df["entry_date"].notna()
    entered = df.loc[entered_mask].copy()
    skipped = df.loc[~entered_mask].copy()

    if entered.empty:
        return df

    entered["entry_date"] = pd.to_datetime(entered["entry_date"], errors="coerce")

    # Tvrdá priorita timeframe:
    # yearly > monthly > weekly
    period_priority = {"yearly": 0, "monthly": 1, "weekly": 2}
    entered["period_priority"] = entered["period_type"].map(period_priority).fillna(999)

    entered = entered.sort_values(
        ["ticker", "entry_date", "period_priority"],
        ascending=[True, True, True],
    ).copy()

    entered["entry_rank_in_day"] = (
        entered.groupby(["ticker", "entry_date"]).cumcount() + 1
    )

    over_limit_mask = entered["entry_rank_in_day"] > max_entries_per_day

    if over_limit_mask.any():
        cols_to_null = [
            "touch_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "bars_held",
            "pnl_abs",
            "pnl_atr",
            "return_pct",
            "mfe_abs",
            "mae_abs",
        ]
        for col in cols_to_null:
            if col in entered.columns:
                entered.loc[over_limit_mask, col] = np.nan

        entered.loc[over_limit_mask, "trend_aligned"] = False
        entered.loc[over_limit_mask, "exit_reason"] = DAILY_LIMIT_EXIT_REASON

    entered = entered.drop(
        columns=["entry_rank_in_day", "period_priority"],
        errors="ignore",
    )

    out = pd.concat([entered, skipped], ignore_index=True)
    out = out.sort_values(
        ["ticker", "active_from", "period_type", "period"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return out


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
    trades_df = apply_daily_entry_limit(
        trades_df,
        max_entries_per_day=MAX_ENTRIES_PER_TICKER_PER_DAY,
)

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
