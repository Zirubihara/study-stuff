"""Generate large CSV dataset for comprehensive performance benchmarking."""

import csv
import random
import time
from pathlib import Path


def generate_large_dataset(output_path: str, num_rows: int = 1000000):
    """
    Generate large CSV data matching the expected schema for performance testing.

    Args:
        output_path: Path to save the CSV file
        num_rows: Number of rows to generate (default: 1M)
    """

    print(f"Generating {num_rows:,} rows of sample data...")
    print(f"Estimated file size: ~{num_rows * 50 / 1024 / 1024:.1f} MB")

    # Define realistic data generation parameters
    years = list(range(1988, 2021))  # 33 years
    months = list(range(1, 13))  # 12 months
    categories1 = list(range(1, 21))  # 20 categories
    categories2 = list(range(1, 51))  # 50 categories
    categories3 = list(range(1, 11))  # 10 categories

    # Pre-generate some commonly used codes for realism
    common_codes = [f"{i:04d}" for i in range(1, 1000)]

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_rows):
            # Generate year_month in YYYYMM format
            year = random.choice(years)
            month = random.choice(months)
            year_month = f"{year}{month:02d}"

            # Generate category values with realistic distributions
            category1 = random.choice(categories1)
            category2 = random.choice(categories2)
            category3 = random.choice(categories3)

            # Generate code with leading zeros (mix of common and random)
            if random.random() < 0.7:  # 70% common codes
                code = random.choice(common_codes)
            else:  # 30% random codes
                code = f"{random.randint(1, 9999):04d}"

            # Generate flag and values with realistic distributions
            flag = random.choices([0, 1], weights=[0.3, 0.7])[0]  # 70% flag=1

            # Value distributions with some correlation
            if category1 <= 10:  # Lower categories have lower values
                value1 = random.randint(1, 500)
                value2 = random.randint(100, 5000)
            else:  # Higher categories have higher values
                value1 = random.randint(500, 2000)
                value2 = random.randint(5000, 15000)

            # Add some seasonal variation based on month
            if month in [12, 1, 2]:  # Winter months
                value2 = int(value2 * random.uniform(1.1, 1.3))
            elif month in [6, 7, 8]:  # Summer months
                value2 = int(value2 * random.uniform(0.8, 1.0))

            writer.writerow(
                [
                    year_month,
                    category1,
                    category2,
                    category3,
                    code,
                    flag,
                    value1,
                    value2,
                ]
            )

            # Progress indicator
            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_rows - i - 1) / rate
                print(
                    f"Generated {i + 1:,} rows ({(i+1)/num_rows*100:.1f}%) - "
                    f"Rate: {rate:.0f} rows/sec - ETA: {eta:.0f}s"
                )

    elapsed = time.time() - start_time
    final_size = Path(output_path).stat().st_size / 1024 / 1024

    print(f"\nLarge dataset generation completed!")
    print(f"File: {output_path}")
    print(f"Rows: {num_rows:,}")
    print(f"Size: {final_size:.1f} MB")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Rate: {num_rows/elapsed:.0f} rows/second")


def generate_multiple_sizes():
    """Generate datasets of different sizes for scalability testing."""

    sizes = [
        (1000000, "../data/benchmark_1m.csv"),
        (5000000, "../data/benchmark_5m.csv"),
        (10000000, "../data/benchmark_10m.csv"),
        (20000000, "../data/benchmark_20m.csv"),
        (50000000, "../data/benchmark_50m.csv"),
    ]

    print("Generating multiple dataset sizes for scalability analysis...\n")

    for num_rows, filename in sizes:
        print(f"Creating {filename}...")
        generate_large_dataset(filename, num_rows)
        print()


if __name__ == "__main__":
    # Generate primary large dataset
    print("Large Dataset Generator for Performance Benchmarking")
    print("=" * 60)

    # Ask user for size preference or use default
    try:
        size_input = input(
            "\nEnter number of rows (default 10000000 for better Spark performance): "
        ).strip()
        if size_input:
            num_rows = int(size_input)
        else:
            num_rows = 10000000

        if num_rows < 10000:
            print(
                "Warning: Very small dataset may not show meaningful performance differences"
            )
        elif num_rows > 5000000:
            print(
                "Warning: Very large dataset may take significant time and disk space"
            )

    except ValueError:
        print("Invalid input, using default size of 10,000,000 rows")
        num_rows = 10000000

    # Generate the dataset (default to 10M rows for better Spark performance)
    output_file = "../data/large_data.csv"
    generate_large_dataset(output_file, num_rows)

    # Option to generate multiple sizes
    generate_multiple = (
        input("\nGenerate multiple sizes for scalability testing? (y/N): ")
        .strip()
        .lower()
    )
    if generate_multiple == "y":
        generate_multiple_sizes()

    print("\nDataset generation complete! Ready for benchmarking.")
