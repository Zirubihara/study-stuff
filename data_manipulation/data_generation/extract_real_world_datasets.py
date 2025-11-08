"""
Extract Real-World Dataset Subsets from custom_1988_2020.csv

This script creates dataset subsets (1M, 5M, 10M, 50M, 100M rows) from the original
real-world custom_1988_2020.csv file for benchmarking purposes.
"""

import os
import sys

def extract_dataset_subset(input_file, output_file, num_rows):
    """
    Extract the first num_rows from input_file and write to output_file.
    Includes header line.

    Args:
        input_file: Path to source CSV file
        output_file: Path to destination CSV file
        num_rows: Number of data rows to extract (excluding header)
    """
    print(f"Extracting {num_rows:,} rows to {os.path.basename(output_file)}...")

    rows_written = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Read and write each line
        for i, line in enumerate(infile):
            outfile.write(line)
            rows_written += 1

            # Progress reporting
            if rows_written % 1_000_000 == 0:
                print(f"  Processed {rows_written:,} rows...")

            # Stop after writing the desired number of rows
            if rows_written >= num_rows:
                break

    print(f"  Completed: {rows_written:,} rows written")

    # Get file size
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

    if file_size_gb >= 1:
        print(f"  File size: {file_size_gb:.2f} GB")
    else:
        print(f"  File size: {file_size_mb:.1f} MB")
    print()

def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    input_file = os.path.join(project_root, 'custom_1988_2020.csv')
    data_dir = os.path.join(project_root, 'data')

    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure custom_1988_2020.csv is in the project root directory.")
        sys.exit(1)

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Get input file info
    input_size_bytes = os.path.getsize(input_file)
    input_size_gb = input_size_bytes / (1024 * 1024 * 1024)

    print("=" * 70)
    print("Real-World Dataset Extraction")
    print("=" * 70)
    print(f"Source file: {os.path.basename(input_file)}")
    print(f"Source size: {input_size_gb:.2f} GB")
    print(f"Output directory: {data_dir}")
    print("=" * 70)
    print()

    # Dataset sizes to extract
    datasets = [
        (1_000_000, 'benchmark_1m.csv'),
        (5_000_000, 'benchmark_5m.csv'),
        (10_000_000, 'benchmark_10m.csv'),
        (50_000_000, 'benchmark_50m.csv'),
        (100_000_000, 'benchmark_100m.csv'),
    ]

    # Extract each dataset
    for num_rows, filename in datasets:
        output_file = os.path.join(data_dir, filename)
        extract_dataset_subset(input_file, output_file, num_rows)

    print("=" * 70)
    print("All datasets extracted successfully!")
    print("=" * 70)
    print("\nGenerated datasets:")
    for num_rows, filename in datasets:
        output_file = os.path.join(data_dir, filename)
        if os.path.exists(output_file):
            size_bytes = os.path.getsize(output_file)
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 * 1024 * 1024)

            if size_gb >= 1:
                size_str = f"{size_gb:.2f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"

            print(f"  - {filename:25s} {num_rows:>12,} rows  {size_str:>10s}")

if __name__ == "__main__":
    main()
