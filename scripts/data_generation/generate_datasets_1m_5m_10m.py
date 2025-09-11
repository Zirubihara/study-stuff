#!/usr/bin/env python3
"""
Generate large datasets for better Spark performance testing.
"""

import csv
import os
import random
import time
from pathlib import Path


def generate_large_dataset(output_path: str, num_rows: int = 10000000):
    """Generate a large CSV dataset with specified number of rows."""
    print(f"Generating {num_rows:,} rows to {output_path}...")
    start_time = time.time()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Define categories for more realistic distribution
    categories1 = list(range(1, 21))  # 1-20
    categories2 = list(range(1, 11))  # 1-10
    categories3 = list(range(1, 6))  # 1-5

    # Generate year-months for 5 years
    year_months = []
    for year in range(2019, 2024):
        for month in range(1, 13):
            year_months.append(f"{year}{month:02d}")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_rows):
            row = [
                random.choice(year_months),
                random.choice(categories1),
                random.choice(categories2),
                random.choice(categories3),
                f"{random.randint(10000, 99999):05d}",  # 5-digit code with leading zeros
                random.randint(0, 1),
                random.randint(1, 1000),
                random.randint(1, 10000),
            ]
            writer.writerow(row)

            # Progress indicator
            if (i + 1) % 1000000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining_rows = num_rows - (i + 1)
                eta = remaining_rows / rate
                print(
                    f"  Progress: {i+1:,}/{num_rows:,} rows ({(i+1)/num_rows*100:.1f}%) - Rate: {rate:.0f} rows/sec - ETA: {eta/60:.1f}min"
                )

    elapsed = time.time() - start_time
    print(f"Completed! Generated {num_rows:,} rows in {elapsed:.1f} seconds")
    print(f"Rate: {num_rows/elapsed:.0f} rows/second")

    # Show file size
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / (1024**2):.1f} MB")
    print()


def main():
    """Generate multiple large datasets."""
    datasets = [
        (1000000, "../data/benchmark_1m.csv"),
        (5000000, "../data/benchmark_5m.csv"),
        (10000000, "../data/benchmark_10m.csv"),
        (50000000, "../data/benchmark_50m.csv"),  # ~1GB for serious Spark testing
    ]

    print("Large Dataset Generator for Spark Performance Testing")
    print("=" * 60)
    print(f"Will generate {len(datasets)} datasets:")
    for rows, filename in datasets:
        print(f"  - {rows:,} rows -> {filename}")
    print()

    total_start = time.time()

    for num_rows, output_path in datasets:
        generate_large_dataset(output_path, num_rows)

    total_elapsed = time.time() - total_start
    print(f"All datasets generated in {total_elapsed/60:.1f} minutes!")


if __name__ == "__main__":
    main()
