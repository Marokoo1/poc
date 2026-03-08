from pathlib import Path
import pandas as pd


def clean_symbols(symbols: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for symbol in symbols:
        if symbol is None:
            continue

        value = str(symbol).strip().upper()

        if not value:
            continue

        if value not in seen:
            cleaned.append(value)
            seen.add(value)

    return cleaned


def load_manual_symbols(universe_config: dict) -> list[str]:
    symbols = universe_config.get("manual", {}).get("symbols", [])
    return clean_symbols(symbols)


def load_txt_symbols(universe_config: dict) -> list[str]:
    path_str = universe_config.get("txt_list", {}).get("path", "")
    path = Path(path_str)

    if not path.exists():
        print(f"[WARN] TXT watchlist not found: {path}")
        return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[ERROR] Failed to read TXT watchlist {path}: {e}")
        return []

    return clean_symbols(lines)


def load_csv_symbols(universe_config: dict) -> list[str]:
    csv_config = universe_config.get("csv_list", {})
    path_str = csv_config.get("path", "")
    column_name = csv_config.get("column", "Symbol")

    path = Path(path_str)

    if not path.exists():
        print(f"[WARN] CSV watchlist not found: {path}")
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV watchlist {path}: {e}")
        return []

    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found in {path}")
        print(f"[INFO] Available columns: {list(df.columns)}")
        return []

    return clean_symbols(df[column_name].tolist())


def load_tls_symbols(universe_config: dict) -> list[str]:
    path_str = universe_config.get("tls", {}).get("path", "")
    path = Path(path_str)

    if not path.exists():
        print(f"[WARN] TLS watchlist not found: {path}")
        return []

    print(f"[INFO] TLS loading not implemented yet: {path}")
    return []


def load_symbols(universe_config: dict) -> list[str]:
    mode = universe_config.get("mode", "manual")

    if mode == "manual":
        symbols = load_manual_symbols(universe_config)
    elif mode == "txt_list":
        symbols = load_txt_symbols(universe_config)
    elif mode == "csv_list":
        symbols = load_csv_symbols(universe_config)
    elif mode == "tls":
        symbols = load_tls_symbols(universe_config)
    else:
        print(f"[ERROR] Unknown universe mode: {mode}")
        return []

    print(f"[INFO] Loaded {len(symbols)} symbols from universe mode '{mode}'")
    return symbols
