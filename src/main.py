from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import time

from data_fetcher import fetch_from_ib, fetch_yahoo_data, load_csv_data
from poc_calculator import (
    calculate_period_poc,
    enrich_poc_with_level_status,
    filter_complete_periods,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config nenalezen: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def ensure_directory(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_symbol_list(values: list[Any]) -> list[str]:
    symbols: list[str] = []

    for item in values:
        symbol = str(item).strip().upper()
        if symbol:
            symbols.append(symbol)

    return sorted(set(symbols))


def load_symbols_from_manual(cfg: dict[str, Any]) -> list[str]:
    symbols = cfg.get("symbols", [])
    if not isinstance(symbols, list):
        raise ValueError("universe.manual.symbols musí být seznam tickerů.")
    return normalize_symbol_list(symbols)


def load_symbols_from_txt(cfg: dict[str, Any]) -> list[str]:
    path_str = cfg.get("path")
    if not path_str:
        raise ValueError("universe.txt_list.path není nastaven.")

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"TXT watchlist nenalezen: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    symbols = [line.strip() for line in lines if line.strip()]
    return normalize_symbol_list(symbols)


def load_symbols_from_csv(cfg: dict[str, Any]) -> list[str]:
    path_str = cfg.get("path")
    column = cfg.get("column", "Symbol")

    if not path_str:
        raise ValueError("universe.csv_list.path není nastaven.")

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"CSV watchlist nenalezen: {path}")

    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(
            f"CSV watchlist nemá sloupec '{column}'. Dostupné sloupce: {list(df.columns)}"
        )

    symbols = df[column].dropna().astype(str).tolist()
    return normalize_symbol_list(symbols)


def load_symbols_from_tls(cfg: dict[str, Any]) -> list[str]:
    path_str = cfg.get("path")
    if not path_str:
        raise ValueError("universe.tls.path není nastaven.")

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"TLS watchlist nenalezen: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    symbols: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # jednoduchý a odolný parser: vezmi první token/sloupec
        token = line.split(",")[0].split(";")[0].strip()
        if token:
            symbols.append(token)

    return normalize_symbol_list(symbols)


def load_symbols(config: dict[str, Any]) -> list[str]:
    universe = config.get("universe", {})
    mode = universe.get("mode", "manual")

    if mode == "manual":
        return load_symbols_from_manual(universe.get("manual", {}))
    if mode == "txt_list":
        return load_symbols_from_txt(universe.get("txt_list", {}))
    if mode == "csv_list":
        return load_symbols_from_csv(universe.get("csv_list", {}))
    if mode == "tls":
        return load_symbols_from_tls(universe.get("tls", {}))

    raise ValueError(f"Neznámý universe.mode: {mode}")


def fetch_ohlcv_for_symbol(
    symbol: str,
    data_source_mode: str,
    config: dict[str, Any],
    raw_data_dir: str,
) -> pd.DataFrame:
    if data_source_mode == "yahoo":
        return fetch_yahoo_data(symbol, config.get("yahoo", {}), raw_data_dir)

    if data_source_mode == "csv":
        csv_cfg = config.get("csv", {})
        file_pattern = csv_cfg.get("file_pattern", "{symbol}.csv")
        return load_csv_data(symbol, raw_data_dir, file_pattern=file_pattern)

    if data_source_mode == "ib":
        return fetch_from_ib(symbol, config.get("ib", {}))

    raise ValueError(f"Neznámý data_source.mode: {data_source_mode}")


def apply_keep_last(df: pd.DataFrame, keep_last: int) -> pd.DataFrame:
    if df.empty:
        return df

    if keep_last <= 0:
        return pd.DataFrame(columns=df.columns)

    return df.tail(keep_last).reset_index(drop=True)


def maybe_enrich_level_status(
    poc_df: pd.DataFrame,
    price_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    level_status_cfg = config.get("poc", {}).get("level_status", {})
    enabled = bool(level_status_cfg.get("enabled", True))

    if not enabled:
        return poc_df

    track_touch = bool(level_status_cfg.get("track_touch", True))
    track_cross = bool(level_status_cfg.get("track_cross", True))

    return enrich_poc_with_level_status(
        poc_df=poc_df,
        price_df=price_df,
        track_touch=track_touch,
        track_cross=track_cross,
    )


def build_poc_for_symbol(symbol: str, price_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    poc_cfg = config.get("poc", {})
    periods_cfg = poc_cfg.get("periods", {})
    keep_last_cfg = poc_cfg.get("keep_last", {})

    period_plan = [
        ("weekly", bool(periods_cfg.get("weekly", True)), int(keep_last_cfg.get("weekly", 5))),
        ("monthly", bool(periods_cfg.get("monthly", True)), int(keep_last_cfg.get("monthly", 5))),
        ("yearly", bool(periods_cfg.get("yearly", True)), int(keep_last_cfg.get("yearly", 3))),
    ]

    frames: list[pd.DataFrame] = []

    for period_name, enabled, keep_last_n in period_plan:
        if not enabled:
            continue

        period_df = calculate_period_poc(price_df, period=period_name)
        if period_df.empty:
            print(f"[WARN] {symbol}: žádná POC data pro {period_name}")
            continue

        period_df = filter_complete_periods(period_df, period=period_name)
        period_df = apply_keep_last(period_df, keep_last=keep_last_n)
        period_df = maybe_enrich_level_status(period_df, price_df, config)

        if period_df.empty:
            print(f"[WARN] {symbol}: po filtraci nezůstala žádná data pro {period_name}")
            continue

        period_df["Ticker"] = symbol
        period_df["PeriodType"] = period_name

        ordered_cols = [
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
        ]

        extra_cols = [col for col in period_df.columns if col not in ordered_cols]
        period_df = period_df[ordered_cols + extra_cols]

        frames.append(period_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["PeriodType", "PeriodStart"]).reset_index(drop=True)


def save_poc_output(symbol: str, poc_df: pd.DataFrame, processed_dir: Path) -> Path:
    output_path = processed_dir / f"{symbol}_poc.csv"
    poc_df.to_csv(output_path, index=False)
    return output_path
    
def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"

def main() -> None:
    started_at = time.time()

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
    total_tickers = len(tickers)

    print(f"📂 Projekt: {PROJECT_DIR}")
    print(f"📁 Raw dir: {RAW_DIR}")
    print(f"📁 Processed dir: {PROCESSED_DIR}")
    print(f"📄 Output: {OUTPUT_FILE}")
    print(f"📊 Načteno {len(poc_df)} POC levelů")
    print(f"🎯 Počet tickerů: {total_tickers}")
    print(f"📈 Tickery: {', '.join(tickers)}")
    print()

    success_count = 0
    empty_count = 0
    failed_count = 0

    for i, ticker in enumerate(tickers, start=1):
        ticker_start = time.time()
        print("=" * 80)
        print(f"[{i}/{total_tickers}] Zpracovávám {ticker}...")

        try:
            ticker_levels = poc_df[poc_df["Ticker"] == ticker].copy()
            level_count = len(ticker_levels)
            print(f"[{i}/{total_tickers}] Levelů pro ticker: {level_count}")

            enriched = enrich_levels_for_ticker(ticker, ticker_levels)

            ticker_elapsed = time.time() - ticker_start

            if not enriched.empty:
                all_frames.append(enriched)
                success_count += 1
                print(
                    f"[{i}/{total_tickers}] ✅ Obohaceno {len(enriched)} levelů "
                    f"za {format_seconds(ticker_elapsed)}"
                )
            else:
                empty_count += 1
                print(
                    f"[{i}/{total_tickers}] ⚠️ Bez výstupu pro {ticker} "
                    f"({format_seconds(ticker_elapsed)})"
                )

        except Exception as e:
            failed_count += 1
            ticker_elapsed = time.time() - ticker_start
            print(
                f"[{i}/{total_tickers}] ❌ Chyba u {ticker}: {e} "
                f"(po {format_seconds(ticker_elapsed)})"
            )

        completed = i
        elapsed_total = time.time() - started_at
        avg_per_ticker = elapsed_total / completed if completed else 0
        remaining = total_tickers - completed
        eta = remaining * avg_per_ticker

        print(
            f"[{i}/{total_tickers}] 📊 Průběh: {completed}/{total_tickers} | "
            f"průměr {format_seconds(avg_per_ticker)} / ticker | "
            f"ETA {format_seconds(eta)}"
        )

    print("\n" + "=" * 80)
    print("📌 Souhrn běhu")
    print(f"   Celkem tickerů: {total_tickers}")
    print(f"   Úspěšně zpracováno: {success_count}")
    print(f"   Bez výstupu: {empty_count}")
    print(f"   S chybou: {failed_count}")
    print(f"   Celkový čas: {format_seconds(time.time() - started_at)}")

    if not all_frames:
        print("⚠️ Nic se nepodařilo zpracovat.")
        return

    final = pd.concat(all_frames, ignore_index=True)

    sort_cols = [c for c in ["Ticker", "PeriodType", "PeriodEnd"] if c in final.columns]
    if sort_cols:
        ascending = [True, True, False][: len(sort_cols)]
        final = final.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Hotovo! Výstup uložen do: {OUTPUT_FILE}")
    print(f"📦 Celkem řádků: {len(final)}")

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

    print("\nUkázka:")
    print(final[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
