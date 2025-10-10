"""
Preprocessing Pipeline for Japanese Trade Dataset
Based on models.md specifications - using Polars for fast preprocessing
"""

import polars as pl
import numpy as np
from pathlib import Path

def preprocess_trade_data(input_file: str, output_dir: str = "../processed"):
    """
    Complete preprocessing pipeline for Japanese trade dataset

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed data
    """
    print(f"Starting preprocessing pipeline for: {input_file}")
    print("=" * 80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: File Loading
    print("\n[1/9] Loading CSV file with Polars...")
    df = pl.read_csv(
        input_file,
        schema_overrides={
            'year_month': pl.Utf8,
            'category1': pl.Int64,
            'category2': pl.Int64,
            'category3': pl.Int64,
            'code': pl.Utf8,
            'flag': pl.Int64,
            'value1': pl.Int64,
            'value2': pl.Int64
        }
    )
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"   Columns: {df.columns}")

    # Step 2: Handling Missing Data
    print("\n[2/9] Handling missing data...")
    null_counts = df.null_count()
    total_nulls = sum([null_counts[col][0] for col in null_counts.columns])
    print(f"   Total null values: {total_nulls}")

    # Fill numeric columns with median
    numeric_cols = ['category1', 'category2', 'category3', 'flag', 'value1', 'value2']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median_val))

    # Fill categorical columns with "Unknown"
    categorical_cols = ['year_month', 'code']
    for col in categorical_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null("Unknown"))

    print(f"   Missing values filled")

    # Step 3: Data Type Corrections
    print("\n[3/9] Correcting data types...")
    df = df.with_columns([
        # Ensure numeric types
        pl.col('category1').cast(pl.Int64),
        pl.col('category2').cast(pl.Int64),
        pl.col('category3').cast(pl.Int64),
        pl.col('flag').cast(pl.Int64),
        pl.col('value1').cast(pl.Float64),
        pl.col('value2').cast(pl.Float64),
        # Keep strings as categorical
        pl.col('year_month').cast(pl.Categorical),
        pl.col('code').cast(pl.Categorical)
    ])
    print(f"   Data types corrected")

    # Step 4: Encoding Categorical Features
    print("\n[4/9] Encoding categorical features...")

    # Check unique values for categorical columns
    unique_categories = {
        'category1': df['category1'].n_unique(),
        'category2': df['category2'].n_unique(),
        'category3': df['category3'].n_unique(),
        'flag': df['flag'].n_unique(),
        'year_month': df['year_month'].n_unique(),
        'code': df['code'].n_unique()
    }
    print(f"   Unique values: {unique_categories}")

    # Create label encodings (preserve originals for reference)
    df = df.with_columns([
        pl.col('year_month').to_physical().alias('year_month_encoded'),
        pl.col('code').to_physical().alias('code_encoded')
    ])
    print(f"   Categorical encoding completed")

    # Step 5: Scaling/Normalizing Numeric Columns
    print("\n[5/9] Scaling numeric columns (min-max normalization)...")

    numeric_to_scale = ['value1', 'value2']
    for col in numeric_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val

        if range_val > 0:
            df = df.with_columns(
                ((pl.col(col) - min_val) / range_val).alias(f'{col}_normalized')
            )
            print(f"   {col}: min={min_val}, max={max_val}, range={range_val}")

    # Step 6: Date Parsing and Feature Expansion
    print("\n[6/9] Parsing dates and extracting features...")

    # Parse year_month (YYYYMM format) and extract features
    df = df.with_columns([
        pl.col('year_month').cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias('year'),
        pl.col('year_month').cast(pl.Utf8).str.slice(4, 2).cast(pl.Int32).alias('month')
    ])

    # Add quarter
    df = df.with_columns(
        ((pl.col('month') - 1) // 3 + 1).alias('quarter')
    )

    print(f"   Extracted features: year, month, quarter")

    # Step 7: Outlier Detection and Handling
    print("\n[7/9] Detecting outliers...")

    for col in ['value1', 'value2']:
        mean_val = df[col].mean()
        std_val = df[col].std()

        # Count values beyond 3 standard deviations
        outliers = df.filter(
            (pl.col(col) < mean_val - 3 * std_val) |
            (pl.col(col) > mean_val + 3 * std_val)
        ).height

        print(f"   {col}: mean={mean_val:.2f}, std={std_val:.2f}, outliers={outliers:,} ({outliers/len(df)*100:.2f}%)")

    print(f"   Keeping outliers for anomaly detection analysis")

    # Step 8: Feature Summary
    print("\n[8/9] Final feature summary...")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Columns: {df.columns}")

    # Step 9: Export for ML/DL Analysis
    print("\n[9/9] Exporting processed data...")

    # Save as Parquet (more efficient for ML)
    output_parquet = f"{output_dir}/processed_data.parquet"
    df.write_parquet(output_parquet)
    print(f"   Saved Parquet: {output_parquet}")

    # Save as CSV for compatibility
    output_csv = f"{output_dir}/processed_data.csv"
    df.write_csv(output_csv)
    print(f"   Saved CSV: {output_csv}")

    # Save summary statistics
    stats = df.describe()
    stats_file = f"{output_dir}/summary_statistics.csv"
    stats.write_csv(stats_file)
    print(f"   Saved statistics: {stats_file}")

    # Save encoding mappings for reference
    encoding_info = {
        'year_month_unique': df.select(['year_month', 'year_month_encoded']).unique().sort('year_month_encoded'),
        'code_unique': df.select(['code', 'code_encoded']).unique().sort('code_encoded')
    }

    encoding_info['year_month_unique'].write_csv(f"{output_dir}/year_month_encoding.csv")
    encoding_info['code_unique'].write_csv(f"{output_dir}/code_encoding.csv")
    print(f"   Saved encoding mappings")

    print("\n" + "=" * 80)
    print("Preprocessing complete! Data ready for ML/DL anomaly detection.")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    return df

if __name__ == "__main__":
    # Test with benchmark_10m.csv
    input_file = "../data/benchmark_10m.csv"
    output_dir = "processed"

    processed_df = preprocess_trade_data(input_file, output_dir)

    print("\nSample of processed data (first 5 rows):")
    print(processed_df.head())
