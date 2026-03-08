import pandas as pd


def load_csv_data(symbol: str, raw_data_dir: str) -> pd.DataFrame:
    file_path = f"{raw_data_dir}/{symbol}.csv"

    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded CSV for {symbol}: {file_path}")
        return df
    except FileNotFoundError:
        print(f"[WARN] CSV not found for {symbol}: {file_path}")
        return pd.DataFrame()


def fetch_from_ib(symbol: str, ib_config: dict) -> pd.DataFrame:
    print(
        f"[INFO] IB fetch placeholder for {symbol} "
        f"(host={ib_config['host']}, port={ib_config['port']}, client_id={ib_config['client_id']})"
    )
    return pd.DataFrame()
