#!/usr/bin/env python3
"""
Generate massive CSV datasets (100M, 200M rows) for extreme performance benchmarking.
WARNING: These will create very large files (3-7GB each)!
"""

import csv
import random
import time
from pathlib import Path


def generate_massive_dataset(output_path: str, num_rows: int = 100000000):
    """
    Generate massive CSV data for extreme performance testing.
    
    Args:
        output_path: Path to save the CSV file
        num_rows: Number of rows to generate (default: 100M)
    """
    
    estimated_size_gb = num_rows * 50 / 1024 / 1024 / 1024
    print(f"Generating {num_rows:,} rows of sample data...")
    print(f"Estimated file size: ~{estimated_size_gb:.1f} GB")
    print("WARNING: This will take significant time and disk space!")
    
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
        
        # Progress tracking
        progress_interval = num_rows // 20  # Show progress every 5%
        
        for i in range(num_rows):
            # Generate realistic data
            year = random.choice(years)
            month = random.choice(months)
            year_month = f"{year}{month:02d}"
            
            category1 = random.choice(categories1)
            category2 = random.choice(categories2)
            category3 = random.choice(categories3)
            
            # Use common codes most of the time, random codes occasionally
            if random.random() < 0.8:
                code = random.choice(common_codes)
            else:
                code = f"{random.randint(1, 9999):04d}"
            
            flag = random.randint(0, 1)
            
            # Generate correlated values for more realistic data
            base_value = random.randint(100, 10000)
            value1 = base_value + random.randint(-50, 50)
            value2 = base_value + random.randint(-100, 100)
            
            writer.writerow([
                year_month,
                category1,
                category2,
                category3,
                code,
                flag,
                value1,
                value2
            ])
            
            # Progress reporting
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                percent = ((i + 1) / num_rows) * 100
                rate = (i + 1) / elapsed
                eta = (num_rows - i - 1) / rate
                print(f"Progress: {percent:5.1f}% ({i+1:,}/{num_rows:,}) - "
                      f"Rate: {rate:,.0f} rows/sec - ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_time
    file_size_gb = Path(output_path).stat().st_size / (1024**3)
    
    print(f"\nDataset generation completed!")
    print(f"File: {output_path}")
    print(f"Rows: {num_rows:,}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Rate: {num_rows/total_time:,.0f} rows/second")


def main():
    """Generate multiple massive datasets."""
    
    # Ensure data directory exists
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    datasets = [
        ("benchmark_100m.csv", 100_000_000),  # ~3.4 GB
        # ("benchmark_200m.csv", 200_000_000),  # ~6.8 GB - uncomment if you want this too
    ]
    
    for filename, num_rows in datasets:
        output_path = data_dir / filename
        
        if output_path.exists():
            print(f"Dataset {filename} already exists ({output_path.stat().st_size / (1024**3):.2f} GB)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Creating {filename}")
        print(f"{'='*60}")
        
        try:
            generate_massive_dataset(str(output_path), num_rows)
        except KeyboardInterrupt:
            print(f"\nGeneration cancelled by user")
            if output_path.exists():
                print(f"Removing incomplete file: {output_path}")
                output_path.unlink()
            break
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            if output_path.exists():
                print(f"Removing incomplete file: {output_path}")
                output_path.unlink()


if __name__ == "__main__":
    print("Massive Dataset Generator")
    print("This will create extremely large files!")
    print("100M rows = ~3.4 GB")
    print("200M rows = ~6.8 GB")
    print()
    print("Make sure you have sufficient disk space.")
    print("Press Ctrl+C to cancel within 5 seconds...")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        exit(0)
    
    main()