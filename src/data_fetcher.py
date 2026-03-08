from pathlib import Path
import pandas as pd
import yfinance as yf


STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance může vracet MultiIndex sloupce i pro jeden ticker.
    Tady je převedeme na obyčejné názvy typu Open, High, Low...
    """
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []

        for col in df.columns:
            # col bývá tuple, např. ('Close', 'SPY') nebo podobně
            parts = [str(part).strip() for part in col if str(part).strip() not in ("", "None")]
            if not parts:
                flattened.append("")
                continue

            # chceme vybrat hlavní OHLCV název
            chosen = None
            for part in parts:
                if part.lower() in COLUMN_ALIASES:
                    chosen = COLUMN_ALIASES[part.lower()]
                    break

            if chosen is None:
                # fallback: vezmi první část
                chosen = parts[0]

            flattened.append(chosen)

        df.columns = flattened

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        normalized = str(col).strip()
        lowered = normalized.lower()
        if lowered in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[lowered]

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def standardize_ohlcv_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        print(f"[WARN] Empty dataset for {symbol}")
        return pd.DataFrame()

    df = flatten_yahoo_columns(df)
    df = normalize_columns(df)

    missing_columns = [col for col in STANDARD_COLUMNS if col not in df.columns]
    if missing_columns:
        print(f"[ERROR] Missing required columns for {symbol}: {missing_columns}")
        print(f"[INFO] Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df[STANDARD_COLUMNS].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"] + numeric_columns)
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        print(f"[WARN] No usable rows after cleaning for {symbol}")
        return pd.DataFrame()

    return df


def save_dataframe_to_csv(df: pd.DataFrame, symbol: str, raw_data_dir: str) -> None:
    output_path = Path(raw_data_dir) / f"{symbol}.csv"
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved data for {symbol} to {output_path}")


def load_csv_data(symbol: str, raw_data_dir: str, file_pattern: str = "{symbol}.csv") -> pd.DataFrame:
    filename = file_pattern.format(symbol=symbol)
    file_path = Path(raw_data_dir) / filename

    if not file_path.exists():
        print(f"[WARN] CSV not found for {symbol}: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV for {symbol}: {e}")
        return pd.DataFrame()

    df = standardize_ohlcv_dataframe(df, symbol)

    if not df.empty:
        print(f"[OK] Loaded CSV for {symbol}: {file_path}")

    return df


def fetch_yahoo_data(symbol: str, yahoo_config: dict, raw_data_dir: str) -> pd.DataFrame:
    period = yahoo_config.get("period", "2y")
    interval = yahoo_config.get("interval", "1d")
    auto_adjust = yahoo_config.get("auto_adjust", False)
    save_downloaded_csv = yahoo_config.get("save_downloaded_csv", True)

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
        )
    except Exception as e:
        print(f"[ERROR] Yahoo download failed for {symbol}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"[WARN] Yahoo returned no data for {symbol}")
        return pd.DataFrame()

    df = df.reset_index()
    df = flatten_yahoo_columns(df)

    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    df = standardize_ohlcv_dataframe(df, symbol)

    if df.empty:
        return pd.DataFrame()

    print(f"[OK] Downloaded Yahoo data for {symbol}: {len(df)} rows")

    if save_downloaded_csv:
        save_dataframe_to_csv(df, symbol, raw_data_dir)

    return df


def describe_dataset(df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        print(f"[INFO] No usable data for {symbol}")
        return

    start_date = df["Date"].min().date()
    end_date = df["Date"].max().date()

    print(f"[INFO] {symbol}: {len(df)} rows")
    print(f"[INFO] {symbol}: date range {start_date} -> {end_date}")
    print(f"[INFO] {symbol}: columns = {list(df.columns)}")
    print("[INFO] First 3 rows:")
    print(df.head(3).to_string(index=False))


def fetch_from_ib(symbol: str, ib_config: dict) -> pd.DataFrame:
    print(
        f"[INFO] IB fetch placeholder for {symbol} "
        f"(host={ib_config['host']}, port={ib_config['port']}, client_id={ib_config['client_id']})"
    )
    return pd.DataFrame()
