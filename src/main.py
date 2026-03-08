from config import load_settings
from utils import ensure_directory
from data_fetcher import load_csv_data, fetch_from_ib
from poc_calculator import calculate_poc


def main() -> None:
    settings = load_settings()

    raw_data_dir = settings["paths"]["raw_data_dir"]
    processed_data_dir = settings["paths"]["processed_data_dir"]

    ensure_directory(raw_data_dir)
    ensure_directory(processed_data_dir)

    mode = settings["data_source"]["mode"]
    symbols = settings["symbols"]

    print(f"Project: {settings['project']['name']}")
    print(f"Data mode: {mode}")
    print(f"Symbols: {symbols}")
    print("-" * 40)

    for symbol in symbols:
        if mode == "csv":
            df = load_csv_data(symbol, raw_data_dir)
        elif mode == "ib":
            df = fetch_from_ib(symbol, settings["ib"])
        else:
            print(f"[ERROR] Unknown data mode: {mode}")
            continue

        result = calculate_poc(df)

        if result.empty:
            print(f"[INFO] No data processed for {symbol}")
        else:
            print(f"[OK] Processed data for {symbol}: {len(result)} rows")

    print("-" * 40)
    print("Done.")


if __name__ == "__main__":
    main()
