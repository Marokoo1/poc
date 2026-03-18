from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import pandas as pd
import yaml

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
        config = load_config(CONFIG_PATH)
    except Exception as e:
        print(f"❌ {e}")
        return

    project_name = config.get("project", {}).get("name", "POC Project")
    data_source_mode = config.get("data_source", {}).get("mode", "yahoo")

    paths_cfg = config.get("paths", {})
    raw_dir = ensure_directory(paths_cfg.get("raw_data_dir", "data/raw"))
    processed_dir = ensure_directory(paths_cfg.get("processed_data_dir", "data/processed"))

    try:
        symbols = load_symbols(config)
    except Exception as e:
        print(f"❌ Chyba při načítání watchlistu: {e}")
        return

    if not symbols:
        print("❌ Watchlist je prázdný.")
        return

    total_symbols = len(symbols)
    success_count = 0
    empty_count = 0
    failed_count = 0
    total_rows = 0

    print(f"📂 Projekt: {project_name}")
    print(f"📈 Tickery: {', '.join(symbols)}")
    print(f"🗂 Raw dir: {raw_dir}")
    print(f"🗂 Processed dir: {processed_dir}")
    print(f"🔌 Data source: {data_source_mode}")
    print()

    for i, symbol in enumerate(symbols, start=1):
        symbol_start = time.time()
        print("=" * 80)
        print(f"[{i}/{total_symbols}] Zpracovávám {symbol}...")

        try:
            price_df = fetch_ohlcv_for_symbol(
                symbol=symbol,
                data_source_mode=data_source_mode,
                config=config,
                raw_data_dir=str(raw_dir),
            )

            if price_df.empty:
                empty_count += 1
                print(f"[{i}/{total_symbols}] ⚠️ Žádná OHLCV data pro {symbol}")
            else:
                poc_df = build_poc_for_symbol(symbol, price_df, config)

                if poc_df.empty:
                    empty_count += 1
                    print(f"[{i}/{total_symbols}] ⚠️ Žádná výsledná POC data pro {symbol}")
                else:
                    output_path = save_poc_output(symbol, poc_df, processed_dir)
                    success_count += 1
                    total_rows += len(poc_df)
                    print(f"[{i}/{total_symbols}] ✅ Uloženo {len(poc_df)} řádků do {output_path.name}")

        except Exception as e:
            failed_count += 1
            print(f"[{i}/{total_symbols}] ❌ Chyba u {symbol}: {e}")

        elapsed_symbol = time.time() - symbol_start
        elapsed_total = time.time() - started_at
        avg_per_symbol = elapsed_total / i
        remaining = total_symbols - i
        eta = remaining * avg_per_symbol

        print(
            f"[{i}/{total_symbols}] ⏱ {format_seconds(elapsed_symbol)} | "
            f"průměr {format_seconds(avg_per_symbol)} / ticker | "
            f"ETA {format_seconds(eta)}"
        )

    print("\n" + "=" * 80)
    print("📌 Souhrn běhu")
    print(f"   Celkem tickerů: {total_symbols}")
    print(f"   Úspěšně zpracováno: {success_count}")
    print(f"   Bez výstupu: {empty_count}")
    print(f"   S chybou: {failed_count}")
    print(f"   Celkem uložených řádků: {total_rows}")
    print(f"   Celkový čas: {format_seconds(time.time() - started_at)}")


if __name__ == "__main__":
    main()
