from pathlib import Path

from config import load_settings
from utils import ensure_directory
from symbol_loader import load_symbols
from data_fetcher import (
    load_csv_data,
    fetch_yahoo_data,
    fetch_from_ib,
    describe_dataset,
)
from poc_calculator import calculate_poc


def main() -> None:
    settings = load_settings()

    raw_data_dir = settings["paths"]["raw_data_dir"]
    processed_data_dir = settings["paths"]["processed_data_dir"]

    ensure_directory(raw_data_dir)
    ensure_directory(processed_data_dir)

    universe_mode = settings["universe"]["mode"]
    data_mode = settings["data_source"]["mode"]

    symbols = load_symbols(settings["universe"])

    print(f"Project: {settings['project']['name']}")
    print(f"Universe mode: {universe_mode}")
    print(f"Data mode: {data_mode}")
    print(f"Symbols: {symbols}")

    poc_settings = settings.get("poc", {})
    period_flags = poc_settings.get(
        "periods",
        {
            "weekly": True,
            "monthly": True,
            "yearly": True,
        },
    )
    keep_last = poc_settings.get(
        "keep_last",
        {
            "weekly": 5,
            "monthly": 5,
            "yearly": 3,
        },
    )

    selected_periods = [name for name, enabled in period_flags.items() if enabled]

    print(f"POC periods enabled: {selected_periods}")
    print(f"POC keep_last: {keep_last}")
    print("-" * 50)

    if not symbols:
        print("[WARN] No symbols loaded. Exiting.")
        return

    if not selected_periods:
        print("[WARN] No POC periods enabled in settings. Exiting.")
        return

    for symbol in symbols:
        print(f"\n=== Processing {symbol} ===")

        if data_mode == "csv":
            df = load_csv_data(
                symbol=symbol,
                raw_data_dir=raw_data_dir,
                file_pattern=settings["csv"]["file_pattern"],
            )
        elif data_mode == "yahoo":
            df = fetch_yahoo_data(
                symbol=symbol,
                yahoo_config=settings["yahoo"],
                raw_data_dir=raw_data_dir,
            )
        elif data_mode == "ib":
            df = fetch_from_ib(symbol, settings["ib"])
        else:
            print(f"[ERROR] Unknown data mode: {data_mode}")
            continue

        describe_dataset(df, symbol)

        result = calculate_poc(
            df,
            periods=selected_periods,
            keep_last=keep_last,
        )

        if result.empty:
            print(f"[INFO] No data processed for {symbol}")
            continue

        output_path = Path(processed_data_dir) / f"{symbol}_poc.csv"
        result.to_csv(output_path, index=False)

        print(f"[OK] Processed data for {symbol}: {len(result)} rows")
        print(f"[OK] Saved processed POC for {symbol} to {output_path}")

    print("\n" + "-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
