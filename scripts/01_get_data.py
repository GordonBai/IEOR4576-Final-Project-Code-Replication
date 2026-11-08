"""Validate the existing locked downloads; this project does not redownload data."""

from network_momentum.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def main() -> None:
    batches = sorted(RAW_DATA_DIR.glob("futures_batch_*.parquet"))
    required = [
        "paper64_returns_v3.parquet",
        "paper64_availability_v3.parquet",
        "paper64_prices_v3.parquet",
        "paper64_model_data_v3.parquet",
    ]
    missing = [name for name in required if not (PROCESSED_DATA_DIR / name).exists()]
    if len(batches) != 58 or missing:
        raise SystemExit(
            f"Data contract failed: raw_batches={len(batches)}, missing_processed={missing}"
        )
    print("Data contract PASS: 58 raw batches and all authoritative processed inputs exist.")


if __name__ == "__main__":
    main()
