#!/usr/bin/env python3
"""
Generate a massive 50M row dataset (~1GB) for serious Spark performance testing.
"""

import csv
import os
import random
import time
from pathlib import Path

def generate_massive_dataset():
    """Generate a 50M row dataset (~1GB) optimized for Spark testing."""
    num_rows = 50000000
    output_path = "../data/benchmark_50m.csv"
    
    print("Massive Dataset Generator for Spark Performance Testing")
    print("=" * 65)
    print(f"Generating {num_rows:,} rows (~1GB) to {output_path}")
    print("This will take several minutes - perfect time for a coffee!")
    print()
    
    start_time = time.time()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define categories for realistic distribution
    categories1 = list(range(1, 51))   # 1-50 (more variety for large dataset)
    categories2 = list(range(1, 21))   # 1-20  
    categories3 = list(range(1, 11))   # 1-10
    
    # Generate year-months for 10 years (more historical data)
    year_months = []
    for year in range(2014, 2024):
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
                f"{random.randint(10000, 99999):05d}",  # 5-digit code
                random.randint(0, 1),
                random.randint(1, 10000),   # Larger value range for variety
                random.randint(1, 50000),   # Much larger range for better aggregations
            ]
            writer.writerow(row)
            
            # Progress indicator every 1M rows
            if (i + 1) % 1000000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining_rows = num_rows - (i + 1)
                eta = remaining_rows / rate
                progress_pct = (i + 1) / num_rows * 100
                print(f"  Progress: {i+1:,}/{num_rows:,} rows ({progress_pct:.1f}%) - Rate: {rate:.0f} rows/sec - ETA: {eta/60:.1f}min")
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path)
    
    print()
    print("*** MASSIVE DATASET COMPLETE! ***")
    print(f"Generated {num_rows:,} rows in {elapsed/60:.1f} minutes")
    print(f"Rate: {num_rows/elapsed:.0f} rows/second")
    print(f"File size: {file_size / (1024**3):.2f} GB ({file_size / (1024**2):.1f} MB)")
    print()
    print("This dataset is perfect for:")
    print("- Demonstrating Spark's distributed processing advantages")
    print("- Testing memory management across different libraries")
    print("- Benchmarking scalability limits")
    print("- Real-world big data performance scenarios")

if __name__ == "__main__":
    generate_massive_dataset()