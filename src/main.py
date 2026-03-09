"""
POC Signals / Level Validity

Navazuje na POC_Calc.py a obohacuje poc_output.csv o obchodní metadata:
- jestli byl level po svém vzniku otestován
- kdy byl poprvé otestován
- zda je level nyní kandidát na long / short
- aktuální vzdálenost od ceny
- ATR normalizovanou vzdálenost
- jednoduché skóre pro screening

Instalace:
    pip install yfinance pandas numpy

Použití:
    python poc_signals.py

Vstup:
    poc_output.csv

Výstup:
    poc_levels_enriched.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# NASTAVENÍ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "poc_levels_enriched.csv"

START = "2020-01-01"

# První jednoduchá mechanická definice testu levelu:
# 1) cena se přiblíží k levelu v rámci TOUCH_BUFFER_ATR * ATR
# 2) během následujících LOOKAHEAD_BARS barů udělá reakci aspoň REACTION_ATR * ATR
ATR_PERIOD = 14
TOUCH_BUFFER_ATR = 0.15
REACTION_ATR = 0.50
LOOKAHEAD_BARS = 3

# Jednoduchý trend filtr / scoring
EMA_FAST = 50
EMA_SLOW = 200

# Používáme lokální CSV pro OHLCV ve složce data/raw.
# Očekávané sloupce: Date, Open, High, Low, Close, Volume.
LOCAL_OHLCV_DIR: Optional[str] = str(RAW_DIR)

# ============================================================


@dataclass
class LevelTestResult:
    is_tested: bool
    first_test_date: Optional[pd.Timestamp]
    touch_date: Optional[pd.Timestamp]
    reaction_size_abs: Optional[float]
    reaction_size_atr: Optional[float]


def load_all_poc_levels(processed_dir: Path) -> pd.DataFrame:
    files = sorted(processed_dir.glob("*_poc.csv"))

    if not files:
        raise FileNotFoundError(
            f"V {processed_dir} nebyly nalezeny žádné soubory *_poc.csv"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)

        required = {
            "Ticker",
            "PeriodType",
            "Period",
            "PeriodStart",
            "PeriodEnd",
            "POC",
            "POC_Volume",
            "Period_High",
            "Period_Low",
            "Period_Close",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} nemá očekávané sloupce: {sorted(missing)}")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["PeriodStart"] = pd.to_datetime(out["PeriodStart"])
    out["PeriodEnd"] = pd.to_datetime(out["PeriodEnd"])
    return out


def load_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    """Načte OHLCV buď z lokálního CSV, nebo z Yahoo Finance."""
    if LOCAL_OHLCV_DIR:
        local_path = Path(LOCAL_OHLCV_DIR) / f"{ticker}.csv"
        if local_path.exists():
            df = pd.read_csv(local_path, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            required = {"Open", "High", "Low", "Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{local_path} nemá požadované sloupce: {sorted(missing)}")
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    raw = yf.download(ticker, start=start, end=date.today().strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()


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
    return out


def next_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    valid = index[index > dt]
    return valid[0] if len(valid) else None


def classify_side(last_close: float, level: float) -> str:
    if pd.isna(last_close) or pd.isna(level):
        return "unknown"
    if last_close > level:
        return "long"
    if last_close < level:
        return "short"
    return "at_level"


def assess_trend(last_row: pd.Series) -> str:
    close = float(last_row["Close"])
    ema50 = float(last_row["EMA50"])
    ema200 = float(last_row["EMA200"])

    if close > ema50 > ema200:
        return "up"
    if close < ema50 < ema200:
        return "down"
    return "neutral"


def scan_level_test(
    ohlcv: pd.DataFrame,
    level: float,
    level_side: str,
    active_from: pd.Timestamp,
) -> LevelTestResult:
    """
    Hledá první smysluplný test levelu po jeho vzniku.

    long level:
        low <= level + buffer
        a následně max high v lookahead okně udělá reakci nahoru >= threshold

    short level:
        high >= level - buffer
        a následně min low v lookahead okně udělá reakci dolů >= threshold
    """
    df = ohlcv.loc[ohlcv.index >= active_from].copy()
    if df.empty:
        return LevelTestResult(False, None, None, None, None)

    for i in range(len(df)):
        row = df.iloc[i]
        atr = row["ATR"]
        if pd.isna(atr) or atr <= 0:
            continue

        buffer_abs = TOUCH_BUFFER_ATR * atr
        reaction_threshold_abs = REACTION_ATR * atr
        touch_date = df.index[i]

        if level_side == "long":
            touched = float(row["Low"]) <= level + buffer_abs
            if not touched:
                continue

            window = df.iloc[i : i + LOOKAHEAD_BARS + 1]
            max_high = float(window["High"].max())
            reaction_abs = max_high - level
            if reaction_abs >= reaction_threshold_abs:
                return LevelTestResult(
                    True,
                    touch_date,
                    touch_date,
                    round(reaction_abs, 6),
                    round(reaction_abs / atr, 6),
                )

        elif level_side == "short":
            touched = float(row["High"]) >= level - buffer_abs
            if not touched:
                continue

            window = df.iloc[i : i + LOOKAHEAD_BARS + 1]
            min_low = float(window["Low"].min())
            reaction_abs = level - min_low
            if reaction_abs >= reaction_threshold_abs:
                return LevelTestResult(
                    True,
                    touch_date,
                    touch_date,
                    round(reaction_abs, 6),
                    round(reaction_abs / atr, 6),
                )

    return LevelTestResult(False, None, None, None, None)


def compute_score(
    level_side: str,
    is_tested: bool,
    trend: str,
    distance_atr: Optional[float],
    period_type: str,
) -> int:
    score = 0

    if not is_tested:
        score += 40

    if level_side == "long" and trend == "up":
        score += 20
    elif level_side == "short" and trend == "down":
        score += 20
    elif trend == "neutral":
        score += 5

    period_bonus = {
        "weekly": 5,
        "monthly": 15,
        "yearly": 25,
    }
    score += period_bonus.get(period_type, 0)

    if distance_atr is not None and not pd.isna(distance_atr):
        # preferujeme rozumnou blízkost, ale ne úplně v levelu
        abs_dist = abs(float(distance_atr))
        if 0.25 <= abs_dist <= 1.50:
            score += 15
        elif 1.50 < abs_dist <= 3.00:
            score += 8
        elif abs_dist < 0.25:
            score += 3

    return int(min(score, 100))


def enrich_levels_for_ticker(ticker: str, levels_df: pd.DataFrame) -> pd.DataFrame:
    ohlcv = load_ohlcv(ticker, START)
    if ohlcv.empty:
        print(f"  ⚠️  Žádná OHLCV data pro {ticker}, přeskakuji enrichment.")
        return pd.DataFrame()

    ohlcv = add_indicators(ohlcv)
    last_row = ohlcv.iloc[-1]
    last_close = float(last_row["Close"])
    last_atr = float(last_row["ATR"]) if not pd.isna(last_row["ATR"]) else np.nan
    trend = assess_trend(last_row)

    rows: list[dict] = []

    for _, row in levels_df.sort_values(["PeriodType", "PeriodStart"]).iterrows():
        level = float(row["POC"])
        period_end = pd.Timestamp(row["PeriodEnd"])
        active_from = next_trading_day(ohlcv.index, period_end)

        level_side = classify_side(last_close, level)

        if active_from is None:
            test_result = LevelTestResult(False, None, None, None, None)
            days_untested = np.nan
        else:
            test_result = scan_level_test(ohlcv, level, level_side, active_from)
            if test_result.is_tested and test_result.first_test_date is not None:
                days_untested = (test_result.first_test_date.normalize() - active_from.normalize()).days
            else:
                days_untested = (ohlcv.index[-1].normalize() - active_from.normalize()).days

        distance_abs = last_close - level
        distance_pct = (distance_abs / level * 100.0) if level else np.nan
        distance_atr = (distance_abs / last_atr) if last_atr and not np.isnan(last_atr) else np.nan

        trend_alignment = (
            (level_side == "long" and trend == "up")
            or (level_side == "short" and trend == "down")
        )

        score = compute_score(
            level_side=level_side,
            is_tested=test_result.is_tested,
            trend=trend,
            distance_atr=distance_atr,
            period_type=str(row["PeriodType"]),
        )

        enriched = row.to_dict()
        enriched.update(
            {
                "LevelPrice": round(level, 4),
                "LevelSide": level_side,
                "ActiveFrom": active_from.date().isoformat() if active_from is not None else None,
                "IsTested": bool(test_result.is_tested),
                "FirstTestDate": test_result.first_test_date.date().isoformat()
                if test_result.first_test_date is not None
                else None,
                "DaysUntested": days_untested,
                "LastClose": round(last_close, 4),
                "LastATR14": round(last_atr, 4) if not np.isnan(last_atr) else np.nan,
                "DistanceToLastClose": round(distance_abs, 4),
                "DistancePct": round(distance_pct, 4) if not pd.isna(distance_pct) else np.nan,
                "DistanceATR": round(distance_atr, 4) if not pd.isna(distance_atr) else np.nan,
                "TrendContext": trend,
                "TrendAligned": bool(trend_alignment),
                "ReactionSizeAbs": test_result.reaction_size_abs,
                "ReactionSizeATR": test_result.reaction_size_atr,
                "ValidNow": not bool(test_result.is_tested),
                "Score": score,
            }
        )
        rows.append(enriched)

    return pd.DataFrame(rows)


def main() -> None:
    try:
        poc_df = load_all_poc_levels(PROCESSED_DIR)
    except Exception as e:
        print(f"❌ {e}")
        return

    if poc_df.empty:
        print("❌ Žádné POC levely k obohacení.")
        return

    all_frames: list[pd.DataFrame] = []
    tickers = sorted(poc_df["Ticker"].dropna().astype(str).unique().tolist())

    print(f"📂 Načteno {len(poc_df)} POC levelů")
    print(f"📈 Tickery: {', '.join(tickers)}")

    for ticker in tickers:
        print(f"Zpracovávám {ticker}...")
        ticker_levels = poc_df[poc_df["Ticker"] == ticker].copy()
        enriched = enrich_levels_for_ticker(ticker, ticker_levels)
        if not enriched.empty:
            all_frames.append(enriched)
            print(f"  ✅ Obohaceno {len(enriched)} levelů")

    if not all_frames:
        print("⚠️  Nic se nepodařilo zpracovat.")
        return

    final = pd.concat(all_frames, ignore_index=True)
    sort_cols = [c for c in ["Ticker", "PeriodType", "PeriodEnd"] if c in final.columns]
    if sort_cols:
        final = final.sort_values(sort_cols, ascending=[True, True, False]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)
    print(f"
✅ Hotovo! Výstup uložen do: {OUTPUT_FILE}")
    print(f"   Celkem řádků: {len(final)}")
    print("
Ukázka:")
    preview_cols = [
        "Ticker",
        "PeriodType",
        "Period",
        "POC",
        "LevelSide",
        "IsTested",
        "FirstTestDate",
        "LastClose",
        "DistanceATR",
        "TrendContext",
        "Score",
    ]
    preview_cols = [c for c in preview_cols if c in final.columns]
    print(final[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
