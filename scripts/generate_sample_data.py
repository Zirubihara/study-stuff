"""Generate sample CSV data for performance benchmarking."""

import csv
import random
from pathlib import Path


def generate_sample_data(output_path: str, num_rows: int = 100000):
    """Generate sample CSV data matching the expected schema."""

    # Define the data generation parameters
    years = list(range(1988, 2021))
    months = list(range(1, 13))

    print(f"Generating {num_rows} rows of sample data...")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_rows):
            # Generate year_month in YYYYMM format
            year = random.choice(years)
            month = random.choice(months)
            year_month = f"{year}{month:02d}"

            # Generate category values
            category1 = random.randint(1, 10)
            category2 = random.randint(1, 20)
            category3 = random.randint(1, 5)

            # Generate code with leading zeros
            code = f"{random.randint(1, 9999):04d}"

            # Generate flag and values
            flag = random.randint(0, 1)
            value1 = random.randint(1, 1000)
            value2 = random.randint(1, 10000)

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

            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1} rows...")

    print(f"Sample data saved to: {output_path}")


if __name__ == "__main__":
    # Create sample data file
    output_file = "../data/sample_data.csv"
    generate_sample_data(
        output_file, num_rows=50000
    )  # 50K rows for reasonable test size
