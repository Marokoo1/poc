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
from poc_calculator import calculate_poc, find_nearest_untouched_levels


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

    level_status = poc_settings.get("level_status", {})
    level_status_enabled = level_status.get("enabled", False)
    track_touch = level_status.get("track_touch", True)
    track_cross = level_status.get("track_cross", True)

    nearest_cfg = poc_settings.get("nearest_untouched", {})
    nearest_enabled = nearest_cfg.get("enabled", False)
    save_summary_csv = nearest_cfg.get("save_summary_csv", True)
    summary_csv_path = nearest_cfg.get(
        "summary_csv_path",
        str(Path(processed_data_dir) / "poc_nearest_summary.csv"),
    )

    selected_periods = [name for name, enabled in period_flags.items() if enabled]

    print(f"POC periods enabled: {selected_periods}")
    print(f"POC keep_last: {keep_last}")
    print(f"Level status enabled: {level_status_enabled}")
    print(f"Track touch: {track_touch}")
    print(f"Track cross: {track_cross}")
    print(f"Nearest untouched enabled: {nearest_enabled}")
    print(f"Nearest untouched summary path: {summary_csv_path}")
    print("-" * 50)

    if not symbols:
        print("[WARN] No symbols loaded. Exiting.")
        return

    if not selected_periods:
        print("[WARN] No POC periods enabled in settings. Exiting.")
        return

    if nearest_enabled and not level_status_enabled:
        print("[WARN] nearest_untouched is enabled but level_status is disabled.")
        print("[WARN] Nearest untouched summary needs Touched metadata, so it will be skipped.")

    nearest_summary_rows = []

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
            include_level_status=level_status_enabled,
            track_touch=track_touch,
            track_cross=track_cross,
        )

        if result.empty:
            print(f"[INFO] No data processed for {symbol}")
            continue

        output_path = Path(processed_data_dir) / f"{symbol}_poc.csv"
        result.to_csv(output_path, index=False)

        print(f"[OK] Processed data for {symbol}: {len(result)} rows")
        print(f"[OK] Saved processed POC for {symbol} to {output_path}")

        if nearest_enabled and level_status_enabled and "Touched" in result.columns:
            last_close = float(df["Close"].iloc[-1])
            nearest = find_nearest_untouched_levels(result, last_close)
            nearest["Symbol"] = symbol
            nearest_summary_rows.append(nearest)

            print(
                f"[OK] Nearest untouched for {symbol}: "
                f"above={nearest['NearestUntouchedAbove']}, "
                f"below={nearest['NearestUntouchedBelow']}"
            )

    if nearest_enabled and level_status_enabled and save_summary_csv and nearest_summary_rows:
        summary_df = load_summary_dataframe(nearest_summary_rows)
        summary_output = Path(summary_csv_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output, index=False)
        print(f"\n[OK] Saved nearest untouched summary to {summary_output}")

    print("\n" + "-" * 50)
    print("Done.")


def load_summary_dataframe(rows: list[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)

    preferred_order = [
        "Symbol",
        "LastClose",
        "NearestUntouchedAbove",
        "NearestUntouchedAbove_PeriodType",
        "NearestUntouchedAbove_PeriodEnd",
        "NearestUntouchedAbove_Distance",
        "NearestUntouchedAbove_DistancePct",
        "NearestUntouchedBelow",
        "NearestUntouchedBelow_PeriodType",
        "NearestUntouchedBelow_PeriodEnd",
        "NearestUntouchedBelow_Distance",
        "NearestUntouchedBelow_DistancePct",
    ]

    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + remaining_cols]


if __name__ == "__main__":
    main()
