from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import time

from poc_calculator import calculate_period_poc, filter_complete_periods
from ib_calculator import calculate_all_ib_levels

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
RUN_LOG_FILE = PROCESSED_DIR / "poc_backtest_run_log.csv"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def append_run_log_row(
    *,
    ticker: str,
    status: str,
    elapsed_sec: float,
    raw_rows: int,
    level_rows: int,
    trade_rows: int,
    summary_rows: int,
    error: str = "",
) -> None:
    row = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "status": status,
                "elapsed_sec": round(float(elapsed_sec), 3),
                "raw_rows": int(raw_rows),
                "level_rows": int(level_rows),
                "trade_rows": int(trade_rows),
                "summary_rows": int(summary_rows),
                "error": error,
                "logged_at": pd.Timestamp.now(),
            }
        ]
    )

    if RUN_LOG_FILE.exists():
        row.to_csv(RUN_LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(RUN_LOG_FILE, index=False)


def print_progress(
    *,
    idx: int,
    total: int,
    ticker: str,
    ticker_elapsed: float,
    started_at: float,
) -> None:
    elapsed_total = time.time() - started_at
    avg_per_ticker = elapsed_total / idx if idx else 0.0
    remaining = total - idx
    eta = remaining * avg_per_ticker

    print(
        f"[{idx}/{total}] {ticker} | "
        f"čas {format_seconds(ticker_elapsed)} | "
        f"průměr {format_seconds(avg_per_ticker)} / ticker | "
        f"ETA {format_seconds(eta)}"
    )



# ============================================================
# SETTINGS
# ============================================================
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

INCLUDE_PERIODS = ( "monthly", "yearly") #"weekly",
MIN_LEVEL_AGE_BARS = 3

# ============================================================
# SIGNAL / CONFLUENCE MODES
# ============================================================
SIGNAL_MODES = ["poc", "ib"]  # available: "poc", "ib"

# IB je teď jedna sada levelů. Není zvlášť "core" a "standard".
# Standardní sada znamená: 0, 100, 150, 200, -150, -200
# Fib zůstává volitelně oddělený přes period_type *_fib.
ALLOW_IB_STANDARD = True
ALLOW_IB_FIB = False

# Samostatný filtr pro IB standalone backtest
IB_SIGNAL_FILTERS = {
    "enabled": False,
    "allowed_period_types": [],        # např. ["monthly_ib", "yearly_ib_fib"] ; [] = vše
    "allowed_level_families": [],      # corrected IB používá hlavně "ib"; [] = vše
    "allowed_level_names": [],         # přesné názvy, např. ["M_IB_0", "M_IB_150", "M_IB_300", "M_IB_NEG150", "M_IB_NEG300"]
    "allowed_name_contains": [],       # částečné filtry, např. ["NEG", "300", "M_IB_"]
}

# Konfigurace pro anotaci POC obchodů aktivní IB podporou
POC_IB_CONTEXT_SETTINGS = {
    "enabled": True,
    "max_atr": 0.25,
    "allowed_period_types": [],        # [] = vše
    "allowed_level_families": [],      # corrected IB = jedna rodina
    "allowed_level_names": [],
    "allowed_name_contains": [],
}

IB_SETTINGS = {
    "enabled": True,
    "yearly_enabled": True,
    "monthly_enabled": True,
    "monthly_mode": "first_5_trading_days",
    "hold_until_tested": True,
    "standard_projection_enabled": True,
    # 0 a 100 jsou základ sady a dopočítají se uvnitř ib_calculator.py
    "standard_multipliers": [1.5, 2.0, 3.0],
    "fibonacci_projection_enabled": False,
    "fibonacci_multipliers": [1.272, 1.618, 2.618],
}


# ============================================================
# Backtest preset configuration
# ============================================================

SUPERSESSION_ENABLED = True

SUPERSESSION_THRESHOLD_ATR = {
    "weekly": 2.00,
    "monthly": 2.00,
    "yearly": 3.00,
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

ENTRY_VOLATILITY_FILTER_ENABLED = True
ENTRY_VOLATILITY_FAST_BARS = 2
ENTRY_VOLATILITY_MULTIPLIER = 1.35
ENTRY_VOLATILITY_EXIT_REASON = "entry_volatility_too_high"


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


def build_poc_levels_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
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
        poc_df["LevelSource"] = "POC"
        poc_df["LevelFamily"] = "poc"
        poc_df["LevelName"] = "POC"
        poc_df["LevelPrice"] = pd.to_numeric(poc_df["POC"], errors="coerce")
        poc_df["PeriodStart"] = pd.to_datetime(poc_df["PeriodStart"], errors="coerce")
        poc_df["PeriodEnd"] = pd.to_datetime(poc_df["PeriodEnd"], errors="coerce")
        poc_df["ActiveFrom"] = poc_df["PeriodEnd"] + pd.Timedelta(days=1)
        frames.append(poc_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ActiveFrom", "PeriodType", "Period"]).reset_index(drop=True)
    return out


def build_ib_levels_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    if not IB_SETTINGS.get("enabled", False):
        return pd.DataFrame()

    ib_df = calculate_all_ib_levels(price_df.copy(), ticker=ticker, settings=IB_SETTINGS)
    if ib_df.empty:
        return pd.DataFrame()

    # corrected calculator: standardní IB sada je jedna rodina "ib"
    # fib zůstává oddělený přes period_type obsahující "_fib"
    if not ALLOW_IB_FIB and "PeriodType" in ib_df.columns:
        ib_df = ib_df[~ib_df["PeriodType"].astype(str).str.contains("_fib", case=False, na=False)].copy()

    if not ALLOW_IB_STANDARD and "PeriodType" in ib_df.columns:
        ib_df = ib_df[ib_df["PeriodType"].astype(str).str.contains("_fib", case=False, na=False)].copy()

    if ib_df.empty:
        return pd.DataFrame()

    ib_df["POC"] = pd.to_numeric(ib_df["LevelPrice"], errors="coerce")
    ib_df = ib_df.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)
    return ib_df


def build_all_levels_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    poc_levels = build_poc_levels_for_ticker(ticker, price_df)
    ib_levels = build_ib_levels_for_ticker(ticker, price_df)

    frames = [df for df in [poc_levels, ib_levels] if not df.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    if "LevelPrice" not in out.columns and "POC" in out.columns:
        out["LevelPrice"] = pd.to_numeric(out["POC"], errors="coerce")
    if "POC" not in out.columns and "LevelPrice" in out.columns:
        out["POC"] = pd.to_numeric(out["LevelPrice"], errors="coerce")

    out = out.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)
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


def normalize_period_type(period_type: str) -> str:
    pt = str(period_type).lower()
    if pt.startswith("weekly"):
        return "weekly"
    if pt.startswith("monthly"):
        return "monthly"
    if pt.startswith("yearly"):
        return "yearly"
    return pt


def get_period_params(period_type: str) -> dict:
    normalized = normalize_period_type(period_type)
    if normalized not in PERIOD_PARAMS:
        raise ValueError(f"Neznámé PeriodType pro params: {period_type}")
    return PERIOD_PARAMS[normalized]

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
    level = float(level_row.get("LevelPrice", level_row.get("POC")))
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
    if levels.empty:
        out = levels.copy()
        if "ValidUntil" not in out.columns:
            out["ValidUntil"] = pd.NaT
        return out

    out = levels.copy()
    out["ActiveFrom"] = pd.to_datetime(out["ActiveFrom"], errors="coerce")
    if "ValidUntil" not in out.columns:
        out["ValidUntil"] = pd.NaT

    if not SUPERSESSION_ENABLED:
        return out

    poc_mask = out["LevelSource"].fillna("POC").astype(str).str.upper() == "POC"
    poc_levels = out.loc[poc_mask].copy()
    other_levels = out.loc[~poc_mask].copy()

    if poc_levels.empty:
        return out

    poc_levels["level_side"] = poc_levels.apply(lambda r: infer_level_side_at_active_from(r, ohlcv), axis=1)
    poc_levels["active_atr"] = poc_levels.apply(lambda r: get_level_atr_at_active_from(r, ohlcv), axis=1)

    poc_levels = poc_levels.sort_values(["ActiveFrom", "PeriodType", "POC"]).reset_index(drop=True)

    for new_idx in range(len(poc_levels)):
        new_row = poc_levels.iloc[new_idx]

        new_active_from = pd.Timestamp(new_row["ActiveFrom"])
        new_period_type = normalize_period_type(str(new_row["PeriodType"]).lower())
        new_side = str(new_row["level_side"]).lower()
        new_poc = float(new_row["POC"])
        new_atr = new_row["active_atr"]

        if pd.isna(new_active_from) or new_side == "unknown":
            continue

        threshold_mult = float(SUPERSESSION_THRESHOLD_ATR.get(new_period_type, 1.0))
        if pd.isna(new_atr) or new_atr is None or float(new_atr) <= 0:
            continue

        threshold_abs = float(new_atr) * threshold_mult

        older_mask = (
            (poc_levels.index < new_idx)
            & (pd.to_datetime(poc_levels["ActiveFrom"], errors="coerce") < new_active_from)
            & (poc_levels["PeriodType"].astype(str).apply(normalize_period_type) == new_period_type)
            & (poc_levels["level_side"].astype(str).str.lower() == new_side)
        )

        if not older_mask.any():
            continue

        older_candidates = poc_levels.loc[older_mask].copy()
        older_candidates["price_distance"] = (older_candidates["POC"].astype(float) - new_poc).abs()
        same_zone = older_candidates["price_distance"] <= threshold_abs

        if not same_zone.any():
            continue

        for old_idx in older_candidates.loc[same_zone].index.tolist():
            old_valid_until = poc_levels.at[old_idx, "ValidUntil"]
            if pd.isna(old_valid_until) or pd.Timestamp(new_active_from) < pd.Timestamp(old_valid_until):
                poc_levels.at[old_idx, "ValidUntil"] = new_active_from

    poc_levels = poc_levels.drop(columns=["level_side", "active_atr"], errors="ignore")
    combined = pd.concat([poc_levels, other_levels], ignore_index=True, sort=False)
    return combined.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)

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

def compute_true_range_series(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def entry_volatility_too_high(future: pd.DataFrame, touch_idx: int) -> bool:
    if not ENTRY_VOLATILITY_FILTER_ENABLED:
        return False

    if future.empty or touch_idx < 0 or touch_idx >= len(future):
        return False

    start_idx = max(0, touch_idx - ENTRY_VOLATILITY_FAST_BARS + 1)
    window = future.iloc[start_idx : touch_idx + 1].copy()

    if window.empty:
        return False

    if "TR" in window.columns:
        tr_mean = pd.to_numeric(window["TR"], errors="coerce").mean()
    else:
        tr_mean = compute_true_range_series(window).mean()

    touch_row = future.iloc[touch_idx]
    atr14 = pd.to_numeric(pd.Series([touch_row.get("ATR")]), errors="coerce").iloc[0]

    if pd.isna(tr_mean) or pd.isna(atr14) or atr14 <= 0:
        return False

    return float(tr_mean) > float(atr14) * ENTRY_VOLATILITY_MULTIPLIER

def simulate_single_level(level_row: pd.Series, ohlcv: pd.DataFrame) -> TradeResult:
    ticker = str(level_row["Ticker"])
    period_type = str(level_row["PeriodType"])
    period = str(level_row["Period"])
    level = float(level_row.get("LevelPrice", level_row.get("POC")))
    active_from = pd.Timestamp(level_row["ActiveFrom"])
    valid_until = pd.to_datetime(level_row.get("ValidUntil"), errors="coerce")

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
    
    if pd.notna(valid_until):
        future = future[future["Date"] < valid_until].copy()
        
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

    if entry_volatility_too_high(future, first_touch_idx):
        return make_empty_trade_result(
            ticker=ticker,
            period_type=period_type,
            period=period,
            level=level,
            side=initial_side,
            active_from=active_from,
            trend_context=trend_context,
            exit_reason=ENTRY_VOLATILITY_EXIT_REASON,
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


def normalize_signal_modes() -> list[str]:
    raw_modes = SIGNAL_MODES if isinstance(SIGNAL_MODES, (list, tuple, set)) else [SIGNAL_MODES]
    modes: list[str] = []
    for mode in raw_modes:
        mode_str = str(mode).strip().lower()
        if not mode_str:
            continue
        if mode_str not in {"poc", "ib"}:
            raise ValueError(f"Unknown signal mode in SIGNAL_MODES: {mode}")
        if mode_str not in modes:
            modes.append(mode_str)
    if not modes:
        raise ValueError("SIGNAL_MODES is empty")
    return modes


def infer_ib_level_side(level_name: str, multiplier=None, badge: str | None = None) -> str:
    if multiplier is not None and not pd.isna(multiplier):
        try:
            return "support" if float(multiplier) <= 0 else "resistance"
        except Exception:
            pass
    b = str(badge or "").strip().upper()
    if b.startswith("-") or b == "0":
        return "support"
    return "resistance"


def _normalize_filter_values(values) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    out = {str(v).strip().lower() for v in values if str(v).strip()}
    return out


def filter_ib_levels_for_use(
    ib_levels: pd.DataFrame,
    *,
    allowed_period_types=None,
    allowed_level_families=None,
    allowed_level_names=None,
    allowed_name_contains=None,
) -> pd.DataFrame:
    if ib_levels.empty:
        return ib_levels.copy()

    out = ib_levels.copy()
    period_types = _normalize_filter_values(allowed_period_types)
    families = _normalize_filter_values(allowed_level_families)
    names = _normalize_filter_values(allowed_level_names)
    name_contains = _normalize_filter_values(allowed_name_contains)

    if period_types:
        out = out[out["PeriodType"].astype(str).str.lower().isin(period_types)].copy()
    if families:
        out = out[out["LevelFamily"].astype(str).str.lower().isin(families)].copy()
    if names:
        out = out[out["LevelName"].astype(str).str.lower().isin(names)].copy()
    if name_contains:
        mask = out["LevelName"].astype(str).str.lower().apply(lambda x: any(s in x for s in name_contains))
        out = out[mask].copy()

    return out.reset_index(drop=True)


def compute_ib_tested_at_for_ticker(
    price_df: pd.DataFrame,
    ib_levels: pd.DataFrame,
) -> pd.DataFrame:
    if ib_levels.empty:
        out = ib_levels.copy()
        out["TestedAt"] = pd.NaT
        out["TouchesCount"] = 0
        out["IsTested"] = False
        return out

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.sort_values("Date").reset_index(drop=True)

    levels = ib_levels.copy()
    levels["ActiveFrom"] = pd.to_datetime(levels["ActiveFrom"], errors="coerce")
    levels["LevelPrice"] = pd.to_numeric(levels["LevelPrice"], errors="coerce")
    levels["LevelSide"] = levels.apply(lambda r: infer_ib_level_side(r.get("LevelName"), r.get("Multiplier"), r.get("LevelBadge")), axis=1)

    tested_at_list = []
    touches_list = []

    for _, lvl in levels.iterrows():
        active_from = lvl.get("ActiveFrom")
        level_price = lvl.get("LevelPrice")
        side = lvl.get("LevelSide")

        if pd.isna(active_from) or pd.isna(level_price):
            tested_at_list.append(pd.NaT)
            touches_list.append(0)
            continue

        future_prices = prices[prices["Date"] >= active_from]
        if future_prices.empty:
            tested_at_list.append(pd.NaT)
            touches_list.append(0)
            continue

        if side == "resistance":
            hits = future_prices[future_prices["High"] >= level_price]
        else:
            hits = future_prices[future_prices["Low"] <= level_price]

        if hits.empty:
            tested_at_list.append(pd.NaT)
            touches_list.append(0)
        else:
            tested_at_list.append(pd.Timestamp(hits.iloc[0]["Date"]))
            touches_list.append(int(len(hits)))

    levels["TestedAt"] = tested_at_list
    levels["TouchesCount"] = touches_list
    levels["IsTested"] = levels["TestedAt"].notna()
    levels["ValidUntil"] = levels["TestedAt"]
    return levels


def enrich_poc_trades_with_ib_context(
    poc_trades: pd.DataFrame,
    ib_levels: pd.DataFrame,
    ohlcv: pd.DataFrame,
    context_settings: dict | None = None,
) -> pd.DataFrame:
    settings = context_settings or {}
    context_enabled = bool(settings.get("enabled", True))
    confluence_max_atr = float(settings.get("max_atr", 0.25))

    out = poc_trades.copy()
    defaults = {
        "has_confluence": False,
        "has_ib_support": False,
        "confluence_count": 0,
        "ib_support_count": 0,
        "nearest_confluence_distance_abs": np.nan,
        "nearest_confluence_distance_atr": np.nan,
        "nearest_confluence_level": np.nan,
        "nearest_ib_distance_abs": np.nan,
        "nearest_ib_distance_atr": np.nan,
        "nearest_ib_level_name": "",
        "nearest_ib_level_family": "",
        "nearest_ib_period_type": "",
        "has_yearly_ib_support": False,
        "has_monthly_ib_support": False,
        "confluence_sources": "",
        "confluence_mode": "POC without IB support",
    }
    for k, v in defaults.items():
        out[k] = v

    if out.empty or ib_levels.empty or not context_enabled:
        return out

    ib = filter_ib_levels_for_use(
        ib_levels,
        allowed_period_types=settings.get("allowed_period_types", []),
        allowed_level_families=settings.get("allowed_level_families", []),
        allowed_level_names=settings.get("allowed_level_names", []),
        allowed_name_contains=settings.get("allowed_name_contains", []),
    )
    if ib.empty:
        return out

    trades = out.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["level_price"] = pd.to_numeric(trades["level_price"], errors="coerce")

    atr_map = ohlcv.copy()
    atr_map["Date"] = pd.to_datetime(atr_map["Date"], errors="coerce")
    atr_map["ATR"] = pd.to_numeric(atr_map["ATR"], errors="coerce")
    atr_by_date = atr_map.set_index("Date")["ATR"].to_dict()

    ib["ActiveFrom"] = pd.to_datetime(ib["ActiveFrom"], errors="coerce")
    ib["TestedAt"] = pd.to_datetime(ib.get("TestedAt"), errors="coerce")
    ib["LevelPrice"] = pd.to_numeric(ib["LevelPrice"], errors="coerce")

    for idx, tr in trades.iterrows():
        entry_time = tr.get("entry_date")
        signal_price = tr.get("level_price")
        atr = atr_by_date.get(entry_time, np.nan)

        if pd.isna(entry_time) or pd.isna(signal_price) or pd.isna(atr) or float(atr) <= 0:
            continue

        active_ib = ib[
            (ib["ActiveFrom"] <= entry_time)
            & (ib["TestedAt"].isna() | (entry_time < ib["TestedAt"]))
        ].copy()

        if active_ib.empty:
            continue

        active_ib["distance_abs"] = (active_ib["LevelPrice"] - float(signal_price)).abs()
        active_ib["distance_atr"] = active_ib["distance_abs"] / float(atr)
        supported = active_ib[active_ib["distance_atr"] <= confluence_max_atr].copy()

        if supported.empty:
            continue

        nearest = supported.sort_values(["distance_atr", "distance_abs"]).iloc[0]
        trades.at[idx, "has_confluence"] = True
        trades.at[idx, "has_ib_support"] = True
        trades.at[idx, "confluence_count"] = int(len(supported))
        trades.at[idx, "ib_support_count"] = int(len(supported))
        trades.at[idx, "nearest_confluence_distance_abs"] = float(nearest["distance_abs"])
        trades.at[idx, "nearest_confluence_distance_atr"] = float(nearest["distance_atr"])
        trades.at[idx, "nearest_confluence_level"] = float(nearest["LevelPrice"])
        trades.at[idx, "nearest_ib_distance_abs"] = float(nearest["distance_abs"])
        trades.at[idx, "nearest_ib_distance_atr"] = float(nearest["distance_atr"])
        trades.at[idx, "nearest_ib_level_name"] = str(nearest.get("LevelName", ""))
        trades.at[idx, "nearest_ib_level_family"] = str(nearest.get("LevelFamily", ""))
        trades.at[idx, "nearest_ib_period_type"] = str(nearest.get("PeriodType", ""))
        trades.at[idx, "has_yearly_ib_support"] = bool((supported["PeriodType"].astype(str).str.contains("yearly", case=False, na=False)).any())
        trades.at[idx, "has_monthly_ib_support"] = bool((supported["PeriodType"].astype(str).str.contains("monthly", case=False, na=False)).any())
        trades.at[idx, "confluence_sources"] = ", ".join(sorted(supported["LevelName"].astype(str).unique().tolist()))
        trades.at[idx, "confluence_mode"] = "POC supported by IB"

    trades["has_confluence"] = trades["has_confluence"].eq(True)
    trades["has_ib_support"] = trades["has_ib_support"].eq(True)
    trades["has_yearly_ib_support"] = trades["has_yearly_ib_support"].eq(True)
    trades["has_monthly_ib_support"] = trades["has_monthly_ib_support"].eq(True)
    trades["confluence_count"] = pd.to_numeric(trades["confluence_count"], errors="coerce").fillna(0).astype(int)
    trades["ib_support_count"] = pd.to_numeric(trades["ib_support_count"], errors="coerce").fillna(0).astype(int)
    return trades



def run_backtest_for_ticker_mode(
    ticker: str,
    ohlcv: pd.DataFrame,
    base_levels: pd.DataFrame,
    signal_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode = str(signal_mode).lower()
    poc_mask = base_levels["LevelSource"].fillna("POC").astype(str).str.upper() == "POC"
    ib_mask = base_levels["LevelSource"].fillna("").astype(str).str.upper() == "IB"

    if mode == "poc":
        signal_levels = base_levels.loc[poc_mask].copy()
        context_ib = filter_ib_levels_for_use(
            base_levels.loc[ib_mask].copy(),
            allowed_period_types=POC_IB_CONTEXT_SETTINGS.get("allowed_period_types", []),
            allowed_level_families=POC_IB_CONTEXT_SETTINGS.get("allowed_level_families", []),
            allowed_level_names=POC_IB_CONTEXT_SETTINGS.get("allowed_level_names", []),
            allowed_name_contains=POC_IB_CONTEXT_SETTINGS.get("allowed_name_contains", []),
        )
    elif mode == "ib":
        signal_levels = filter_ib_levels_for_use(
            base_levels.loc[ib_mask].copy(),
            allowed_period_types=IB_SIGNAL_FILTERS.get("allowed_period_types", []),
            allowed_level_families=IB_SIGNAL_FILTERS.get("allowed_level_families", []),
            allowed_level_names=IB_SIGNAL_FILTERS.get("allowed_level_names", []),
            allowed_name_contains=IB_SIGNAL_FILTERS.get("allowed_name_contains", []),
        )
        context_ib = pd.DataFrame()
    else:
        raise ValueError(f"Unknown signal mode: {signal_mode}")

    levels_out = base_levels.copy()
    levels_out["signal_mode"] = mode
    levels_out["UsedAsSignal"] = False
    levels_out["has_confluence"] = False
    levels_out["confluence_count"] = 0
    levels_out["nearest_confluence_distance_abs"] = np.nan
    levels_out["nearest_confluence_distance_atr"] = np.nan
    levels_out["nearest_confluence_level"] = np.nan
    levels_out["confluence_sources"] = ""
    levels_out["confluence_mode"] = np.where(
        levels_out["LevelSource"].fillna("POC").astype(str).str.upper() == "IB",
        "IB standalone",
        "POC without IB support",
    )

    if not signal_levels.empty:
        merge_cols = ["Ticker", "PeriodType", "Period", "LevelName", "ActiveFrom"]
        left = levels_out.copy()
        left["_merge_price"] = pd.to_numeric(left.get("LevelPrice", left.get("POC")), errors="coerce").round(8)
        right = signal_levels.copy()
        right["_merge_price"] = pd.to_numeric(right.get("LevelPrice", right.get("POC")), errors="coerce").round(8)
        levels_out = left.merge(
            right[merge_cols + ["_merge_price"]],
            how="left",
            on=merge_cols + ["_merge_price"],
            suffixes=("", "_sig"),
            indicator=True,
        )
        levels_out["UsedAsSignal"] = levels_out["_merge"] == "both"
        levels_out = levels_out.drop(columns=["_merge", "_merge_price"])
        levels_out["UsedAsSignal"] = levels_out["UsedAsSignal"].eq(True)

    trade_rows = []
    for _, level_row in signal_levels.iterrows():
        result = simulate_single_level(level_row, ohlcv)
        row = asdict(result)
        row["signal_mode"] = mode
        row["level_source"] = str(level_row.get("LevelSource", "POC"))
        row["level_family"] = str(level_row.get("LevelFamily", "poc"))
        row["level_name"] = str(level_row.get("LevelName", "POC"))
        row["has_confluence"] = False
        row["confluence_count"] = 0
        row["nearest_confluence_distance_abs"] = np.nan
        row["nearest_confluence_distance_atr"] = np.nan
        row["nearest_confluence_level"] = np.nan
        row["confluence_sources"] = ""
        row["confluence_mode"] = "IB standalone" if mode == "ib" else "POC without IB support"
        trade_rows.append(row)

    trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()
    if trades.empty:
        return levels_out, trades, pd.DataFrame()

    if mode == "poc":
        trades = enrich_poc_trades_with_ib_context(
            trades,
            ib_levels=context_ib,
            ohlcv=ohlcv,
            context_settings=POC_IB_CONTEXT_SETTINGS,
        )
        levels_out.loc[
            levels_out["UsedAsSignal"] & (levels_out["LevelSource"].fillna("POC").astype(str).str.upper() == "POC"),
            "confluence_mode",
        ] = "POC signal"
    else:
        trades["has_ib_support"] = False
        trades["ib_support_count"] = 0
        trades["nearest_ib_distance_abs"] = np.nan
        trades["nearest_ib_distance_atr"] = np.nan
        trades["nearest_ib_level_name"] = ""
        trades["nearest_ib_level_family"] = ""
        trades["nearest_ib_period_type"] = ""
        trades["has_yearly_ib_support"] = False
        trades["has_monthly_ib_support"] = False

    summary = (
        trades.groupby(["ticker", "signal_mode", "level_source", "period_type", "side", "exit_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    return levels_out, trades, summary


def run_backtest_for_ticker(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ohlcv = add_indicators(ohlcv)

    base_levels = build_all_levels_for_ticker(ticker, ohlcv)
    if base_levels.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_levels = apply_level_supersession(base_levels, ohlcv)

    ib_mask = base_levels["LevelSource"].fillna("").astype(str).str.upper() == "IB"
    if ib_mask.any():
        ib_levels = compute_ib_tested_at_for_ticker(ohlcv, base_levels.loc[ib_mask].copy())
        non_ib = base_levels.loc[~ib_mask].copy()
        base_levels = pd.concat([non_ib, ib_levels], ignore_index=True, sort=False)
        base_levels = base_levels.sort_values(["ActiveFrom", "PeriodType", "Period", "LevelPrice"]).reset_index(drop=True)

    mode_levels: list[pd.DataFrame] = []
    mode_trades: list[pd.DataFrame] = []
    mode_summaries: list[pd.DataFrame] = []

    for signal_mode in normalize_signal_modes():
        levels_out, trades, summary = run_backtest_for_ticker_mode(ticker, ohlcv, base_levels, signal_mode)
        if not levels_out.empty:
            mode_levels.append(levels_out)
        if not trades.empty:
            mode_trades.append(trades)
        if not summary.empty:
            mode_summaries.append(summary)

    final_levels = pd.concat(mode_levels, ignore_index=True) if mode_levels else pd.DataFrame()
    final_trades = pd.concat(mode_trades, ignore_index=True) if mode_trades else pd.DataFrame()
    final_summary = pd.concat(mode_summaries, ignore_index=True) if mode_summaries else pd.DataFrame()

    return final_levels, final_trades, final_summary


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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    tickers = sorted(
        [
            path.stem
            for path in RAW_DIR.glob("*.csv")
            if path.is_file()
        ]
    )

    if not tickers:
        print(f"❌ V {RAW_DIR} nebyly nalezeny žádné CSV soubory.")
        return

    all_levels: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    success_count = 0
    empty_count = 0
    failed_count = 0

    print(f"📂 Projekt: {PROJECT_DIR}")
    print(f"📁 Raw dir: {RAW_DIR}")
    print(f"📁 Processed dir: {PROCESSED_DIR}")
    print(f"📈 Tickery: {', '.join(tickers)}")
    print(f"🎯 Počet tickerů: {len(tickers)}")
    print(f"🧪 Preset: {BACKTEST_PRESET}")
    print(f"📆 Periody: {', '.join(INCLUDE_PERIODS)}")
    print(f"🎛️ Signal modes: {', '.join(normalize_signal_modes())}")
    print(f"🧭 POC+IB context enabled: {POC_IB_CONTEXT_SETTINGS.get('enabled', True)} | max ATR: {POC_IB_CONTEXT_SETTINGS.get('max_atr', 0.25)}")
    print()

    for i, ticker in enumerate(tickers, start=1):
        ticker_start = time.time()
        print("=" * 90)
        print(f"[{i}/{len(tickers)}] Zpracovávám {ticker}...")

        try:
            levels_df, trades_df, summary_df = run_backtest_for_ticker(ticker)

            raw_df = load_ohlcv(ticker)
            raw_rows = len(raw_df)

            if levels_df.empty and trades_df.empty and summary_df.empty:
                empty_count += 1
                append_run_log_row(
                    ticker=ticker,
                    status="empty",
                    elapsed_sec=time.time() - ticker_start,
                    raw_rows=raw_rows,
                    level_rows=0,
                    trade_rows=0,
                    summary_rows=0,
                )
                print(f"[{i}/{len(tickers)}] ⚠️ Bez výstupu pro {ticker}")
            else:
                success_count += 1

                if not levels_df.empty:
                    all_levels.append(levels_df)
                if not trades_df.empty:
                    all_trades.append(trades_df)
                if not summary_df.empty:
                    all_summary.append(summary_df)

                append_run_log_row(
                    ticker=ticker,
                    status="ok",
                    elapsed_sec=time.time() - ticker_start,
                    raw_rows=raw_rows,
                    level_rows=len(levels_df),
                    trade_rows=len(trades_df),
                    summary_rows=len(summary_df),
                )

                print(
                    f"[{i}/{len(tickers)}] ✅ {ticker} | "
                    f"raw: {raw_rows} | "
                    f"levels: {len(levels_df)} | "
                    f"trades: {len(trades_df)} | "
                    f"summary: {len(summary_df)}"
                )

        except Exception as e:
            failed_count += 1
            append_run_log_row(
                ticker=ticker,
                status="error",
                elapsed_sec=time.time() - ticker_start,
                raw_rows=0,
                level_rows=0,
                trade_rows=0,
                summary_rows=0,
                error=str(e),
            )
            print(f"[{i}/{len(tickers)}] ❌ Chyba u {ticker}: {e}")

        print_progress(
            idx=i,
            total=len(tickers),
            ticker=ticker,
            ticker_elapsed=time.time() - ticker_start,
            started_at=started_at,
        )

    final_levels = pd.concat(all_levels, ignore_index=True) if all_levels else pd.DataFrame()
    final_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    final_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    if not final_levels.empty:
        final_levels.to_csv(LEVELS_FILE, index=False)
    if not final_trades.empty:
        final_trades.to_csv(TRADES_FILE, index=False)
    if not final_summary.empty:
        final_summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 90)
    print("📌 Souhrn běhu")
    print(f"   Celkem tickerů: {len(tickers)}")
    print(f"   Úspěšně zpracováno: {success_count}")
    print(f"   Bez výstupu: {empty_count}")
    print(f"   S chybou: {failed_count}")
    print(f"   Level rows: {len(final_levels)}")
    print(f"   Trade rows: {len(final_trades)}")
    print(f"   Summary rows: {len(final_summary)}")
    print(f"   Celkový čas: {format_seconds(time.time() - started_at)}")
    print(f"   Log běhu: {RUN_LOG_FILE}")

    if not final_trades.empty:
        entered = final_trades["entry_date"].notna().sum() if "entry_date" in final_trades.columns else 0
        print(f"   Obchody se vstupem: {entered}")


if __name__ == "__main__":
    main()
