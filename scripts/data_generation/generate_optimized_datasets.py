#!/usr/bin/env python3
"""
Generate specialized datasets to showcase each technology's strengths and weaknesses.
Each dataset is designed to highlight specific advantages of different libraries.
"""

import csv
import json
import random
import string
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def generate_pandas_showcase_dataset(output_path: str, num_rows: int = 500000):
    """
    Dataset showcasing Pandas strengths:
    - Complex string operations
    - DateTime manipulations
    - Missing data handling
    - Mixed data types
    - Hierarchical indexing scenarios
    """
    print(f"Generating Pandas showcase dataset: {num_rows:,} rows")
    print("Features: Complex strings, dates, missing data, mixed types")

    # Complex data for Pandas strengths
    cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
    ]
    departments = ["Sales", "Marketing", "Engineering", "HR", "Finance", "Operations"]
    statuses = ["Active", "Inactive", "Pending", "Suspended"]

    # Generate realistic names
    first_names = [
        "John",
        "Jane",
        "Michael",
        "Sarah",
        "David",
        "Emily",
        "Robert",
        "Lisa",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Brown",
        "Davis",
        "Miller",
        "Wilson",
        "Moore",
        "Taylor",
    ]

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(
            [
                "employee_id",
                "full_name",
                "email",
                "phone",
                "hire_date",
                "birth_date",
                "department",
                "city",
                "salary",
                "bonus",
                "status",
                "notes",
                "skills",
                "last_login",
                "project_hours",
                "rating",
            ]
        )

        for i in range(num_rows):
            # Complex string data
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            full_name = f"{first_name} {last_name}"
            email = f"{first_name.lower()}.{last_name.lower()}@company.com"

            # Phone with various formats (messy data for cleaning)
            phone_formats = [
                f"({random.randint(100,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
                f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}",
                f"{random.randint(1000000000,9999999999)}",
            ]
            phone = random.choice(phone_formats)

            # DateTime data
            hire_date = datetime(2010, 1, 1) + timedelta(days=random.randint(0, 5000))
            birth_date = datetime(1960, 1, 1) + timedelta(days=random.randint(0, 15000))

            # Mixed and missing data
            department = random.choice(departments)
            city = random.choice(cities)

            # Salary with some missing values
            salary = random.randint(40000, 150000) if random.random() > 0.05 else None
            bonus = random.randint(0, 20000) if random.random() > 0.3 else None

            status = random.choice(statuses)

            # Complex text data
            notes = (
                f"Employee in {department}. "
                + ("High performer. " if random.random() > 0.7 else "")
                + ("Remote worker. " if random.random() > 0.6 else "")
                + ("Needs training. " if random.random() > 0.8 else "")
            )

            # JSON-like skills data
            all_skills = [
                "Python",
                "SQL",
                "Excel",
                "PowerBI",
                "Tableau",
                "R",
                "Java",
                "JavaScript",
            ]
            skills = json.dumps(random.sample(all_skills, random.randint(1, 4)))

            # Recent datetime
            last_login = (
                datetime.now() - timedelta(days=random.randint(0, 30))
                if random.random() > 0.1
                else None
            )

            # Numerical data
            project_hours = round(random.uniform(0, 80), 1)
            rating = round(random.uniform(1, 5), 1)

            writer.writerow(
                [
                    f"EMP{i+1:06d}",
                    full_name,
                    email,
                    phone,
                    hire_date.strftime("%Y-%m-%d"),
                    birth_date.strftime("%Y-%m-%d"),
                    department,
                    city,
                    salary,
                    bonus,
                    status,
                    notes,
                    skills,
                    last_login.strftime("%Y-%m-%d %H:%M:%S") if last_login else None,
                    project_hours,
                    rating,
                ]
            )

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1:,}/{num_rows:,}")

    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  Completed in {elapsed:.1f}s, Size: {file_size_mb:.1f} MB\n")


def generate_pyarrow_showcase_dataset(output_path: str, num_rows: int = 5000000):
    """
    Dataset showcasing PyArrow strengths:
    - Pure numerical data
    - Columnar operations
    - Aggregations
    - Filtering
    - Large-scale analytics
    """
    print(f"Generating PyArrow showcase dataset: {num_rows:,} rows")
    print("Features: Pure numerical, columnar analytics, aggregations")

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header - all numerical/categorical for optimal columnar processing
        writer.writerow(
            [
                "transaction_id",
                "timestamp",
                "user_id",
                "product_id",
                "category_id",
                "quantity",
                "unit_price",
                "total_amount",
                "discount",
                "tax",
                "region_id",
                "channel_id",
                "is_premium",
                "customer_score",
            ]
        )

        progress_interval = num_rows // 10

        for i in range(num_rows):
            # Optimized for columnar operations
            transaction_id = i + 1
            timestamp = int(time.time()) - random.randint(0, 31536000)  # Last year
            user_id = random.randint(1, 100000)
            product_id = random.randint(1, 10000)
            category_id = random.randint(1, 50)

            # Numerical data perfect for vectorized operations
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(5.0, 500.0), 2)
            total_amount = round(quantity * unit_price, 2)
            discount = round(total_amount * random.uniform(0, 0.3), 2)
            tax = round((total_amount - discount) * 0.08, 2)

            region_id = random.randint(1, 10)
            channel_id = random.randint(1, 5)
            is_premium = random.randint(0, 1)
            customer_score = random.randint(1, 100)

            writer.writerow(
                [
                    transaction_id,
                    timestamp,
                    user_id,
                    product_id,
                    category_id,
                    quantity,
                    unit_price,
                    total_amount,
                    discount,
                    tax,
                    region_id,
                    channel_id,
                    is_premium,
                    customer_score,
                ]
            )

            if (i + 1) % progress_interval == 0:
                print(f"  Progress: {i+1:,}/{num_rows:,}")

    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  Completed in {elapsed:.1f}s, Size: {file_size_mb:.1f} MB\n")


def generate_polars_showcase_dataset(output_path: str, num_rows: int = 2000000):
    """
    Dataset showcasing Polars strengths:
    - Complex query optimization
    - Group operations
    - Window functions
    - Time series analysis
    - Memory-efficient processing
    """
    print(f"Generating Polars showcase dataset: {num_rows:,} rows")
    print("Features: Time series, complex queries, window operations")

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header optimized for complex queries
        writer.writerow(
            [
                "device_id",
                "timestamp",
                "sensor_type",
                "measurement_value",
                "location_lat",
                "location_lon",
                "temperature",
                "humidity",
                "battery_level",
                "signal_strength",
                "error_code",
                "quality_score",
            ]
        )

        # IoT sensor data - perfect for Polars time series + groupby operations
        device_count = 1000
        sensor_types = ["temp", "humidity", "pressure", "light", "motion"]

        progress_interval = num_rows // 10

        for i in range(num_rows):
            device_id = f"device_{random.randint(1, device_count):04d}"

            # Time series data
            base_time = datetime(2024, 1, 1)
            timestamp = base_time + timedelta(minutes=i // device_count)

            sensor_type = random.choice(sensor_types)

            # Sensor readings with realistic correlations
            measurement_value = round(random.gauss(50, 15), 2)
            location_lat = round(random.uniform(40.0, 41.0), 6)
            location_lon = round(random.uniform(-74.0, -73.0), 6)

            # Correlated environmental data
            base_temp = 20 + random.gauss(0, 10)
            temperature = round(base_temp, 1)
            humidity = round(
                max(0, min(100, 50 + (25 - base_temp) * 2 + random.gauss(0, 5))), 1
            )

            # Device status
            battery_level = max(0, min(100, random.gauss(70, 20)))
            signal_strength = max(-100, min(0, random.gauss(-60, 15)))
            error_code = random.choice([0, 0, 0, 0, 1, 2, 3])  # Mostly no errors
            quality_score = round(random.uniform(0.7, 1.0), 3)

            writer.writerow(
                [
                    device_id,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    sensor_type,
                    measurement_value,
                    location_lat,
                    location_lon,
                    temperature,
                    humidity,
                    round(battery_level, 1),
                    round(signal_strength, 1),
                    error_code,
                    quality_score,
                ]
            )

            if (i + 1) % progress_interval == 0:
                print(f"  Progress: {i+1:,}/{num_rows:,}")

    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  Completed in {elapsed:.1f}s, Size: {file_size_mb:.1f} MB\n")


def generate_dask_showcase_dataset(output_path: str, num_rows: int = 10000000):
    """
    Dataset showcasing Dask strengths:
    - Large-scale data that doesn't fit in memory
    - Parallel processing
    - Out-of-core computations
    - Scientific computing scenarios
    """
    print(f"Generating Dask showcase dataset: {num_rows:,} rows")
    print("Features: Large-scale, parallel processing, scientific data")

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header for large-scale scientific/financial data
        writer.writerow(
            [
                "record_id",
                "timestamp",
                "experiment_id",
                "sample_id",
                "measurement_1",
                "measurement_2",
                "measurement_3",
                "measurement_4",
                "measurement_5",
                "measurement_6",
                "measurement_7",
                "measurement_8",
                "control_group",
                "treatment_group",
                "batch_id",
                "lab_id",
            ]
        )

        progress_interval = num_rows // 20

        for i in range(num_rows):
            record_id = i + 1
            timestamp = int(time.time()) - random.randint(0, 86400 * 365)  # Last year
            experiment_id = random.randint(1, 100)
            sample_id = f"sample_{random.randint(1, 10000):06d}"

            # Scientific measurements with realistic correlations
            base_value = random.gauss(100, 20)
            measurement_1 = round(base_value + random.gauss(0, 5), 4)
            measurement_2 = round(base_value * 1.1 + random.gauss(0, 8), 4)
            measurement_3 = round(base_value * 0.9 + random.gauss(0, 6), 4)
            measurement_4 = round(base_value + random.gauss(0, 10), 4)
            measurement_5 = round(
                measurement_1 * measurement_2 / 100 + random.gauss(0, 3), 4
            )
            measurement_6 = round(
                np.sqrt(abs(measurement_3)) * 10 + random.gauss(0, 2), 4
            )
            measurement_7 = round(random.uniform(0, 1000), 4)
            measurement_8 = round(random.expovariate(1 / 50), 4)

            control_group = random.randint(0, 1)
            treatment_group = random.randint(1, 5) if not control_group else 0
            batch_id = random.randint(1, 1000)
            lab_id = random.randint(1, 20)

            writer.writerow(
                [
                    record_id,
                    timestamp,
                    experiment_id,
                    sample_id,
                    measurement_1,
                    measurement_2,
                    measurement_3,
                    measurement_4,
                    measurement_5,
                    measurement_6,
                    measurement_7,
                    measurement_8,
                    control_group,
                    treatment_group,
                    batch_id,
                    lab_id,
                ]
            )

            if (i + 1) % progress_interval == 0:
                print(f"  Progress: {i+1:,}/{num_rows:,}")

    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  Completed in {elapsed:.1f}s, Size: {file_size_mb:.1f} MB\n")


def generate_spark_showcase_dataset(output_path: str, num_rows: int = 20000000):
    """
    Dataset showcasing PySpark strengths:
    - Massive distributed data
    - Complex joins and aggregations
    - Machine learning features
    - Enterprise-scale analytics
    """
    print(f"Generating PySpark showcase dataset: {num_rows:,} rows")
    print("Features: Massive scale, complex joins, ML features")

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header for enterprise-scale data
        writer.writerow(
            [
                "customer_id",
                "transaction_date",
                "product_category",
                "product_subcategory",
                "brand_id",
                "store_id",
                "sales_amount",
                "quantity",
                "discount_amount",
                "customer_age",
                "customer_income",
                "customer_segment",
                "geography_region",
                "marketing_channel",
                "is_weekend",
                "season",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "target_variable",
            ]
        )

        # Large-scale retail/enterprise data
        categories = [
            "Electronics",
            "Clothing",
            "Home",
            "Sports",
            "Books",
            "Beauty",
            "Food",
        ]
        subcategories = {
            "Electronics": ["Phones", "Laptops", "TVs", "Gaming"],
            "Clothing": ["Shirts", "Pants", "Shoes", "Accessories"],
            "Home": ["Furniture", "Kitchen", "Decor", "Tools"],
            "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports"],
            "Books": ["Fiction", "Non-fiction", "Educational", "Children"],
            "Beauty": ["Skincare", "Makeup", "Hair", "Fragrance"],
            "Food": ["Organic", "Snacks", "Beverages", "Frozen"],
        }

        segments = ["Premium", "Standard", "Budget", "Enterprise"]
        regions = ["North", "South", "East", "West", "Central"]
        channels = ["Online", "Store", "Mobile App", "Phone", "Partner"]
        seasons = ["Spring", "Summer", "Fall", "Winter"]

        progress_interval = num_rows // 20

        for i in range(num_rows):
            customer_id = random.randint(1, 1000000)

            # Date range for time series analysis
            base_date = datetime(2020, 1, 1)
            transaction_date = base_date + timedelta(
                days=random.randint(0, 1460)
            )  # 4 years

            category = random.choice(categories)
            subcategory = random.choice(subcategories[category])
            brand_id = random.randint(1, 500)
            store_id = random.randint(1, 2000)

            # Realistic sales data with correlations
            base_price = random.uniform(10, 1000)
            quantity = random.randint(1, 5)
            discount_rate = random.uniform(0, 0.5) if random.random() > 0.7 else 0
            sales_amount = round(base_price * quantity, 2)
            discount_amount = round(sales_amount * discount_rate, 2)

            # Customer demographics
            customer_age = max(18, min(80, int(random.gauss(40, 15))))
            customer_income = max(20000, int(random.gauss(60000, 25000)))
            customer_segment = random.choice(segments)
            geography_region = random.choice(regions)
            marketing_channel = random.choice(channels)

            # Time-based features
            is_weekend = 1 if transaction_date.weekday() >= 5 else 0
            season = seasons[(transaction_date.month - 1) // 3]

            # ML features
            feature_1 = round(random.gauss(0, 1), 4)
            feature_2 = round(random.uniform(-2, 2), 4)
            feature_3 = round(random.expovariate(1), 4)
            feature_4 = round(customer_age * feature_1 + random.gauss(0, 5), 4)

            # Target variable for ML
            target_variable = (
                1 if (sales_amount > 100 and customer_income > 50000) else 0
            )

            writer.writerow(
                [
                    customer_id,
                    transaction_date.strftime("%Y-%m-%d"),
                    category,
                    subcategory,
                    brand_id,
                    store_id,
                    sales_amount,
                    quantity,
                    discount_amount,
                    customer_age,
                    customer_income,
                    customer_segment,
                    geography_region,
                    marketing_channel,
                    is_weekend,
                    season,
                    feature_1,
                    feature_2,
                    feature_3,
                    feature_4,
                    target_variable,
                ]
            )

            if (i + 1) % progress_interval == 0:
                print(f"  Progress: {i+1:,}/{num_rows:,}")

    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  Completed in {elapsed:.1f}s, Size: {file_size_mb:.1f} MB\n")


def main():
    """Generate all specialized datasets."""
    print("GENERATING SPECIALIZED DATASETS")
    print("=" * 60)
    print("Each dataset showcases different technology strengths:")
    print("- Pandas: Complex data types, strings, dates, missing data")
    print("- PyArrow: Pure numerical, columnar analytics")
    print("- Polars: Time series, complex queries, memory efficiency")
    print("- Dask: Large-scale, parallel processing")
    print("- PySpark: Massive enterprise data, ML features")
    print("=" * 60)

    # Ensure data directory exists
    data_dir = Path("../data/specialized")
    data_dir.mkdir(exist_ok=True, parents=True)

    datasets = [
        ("pandas_showcase.csv", generate_pandas_showcase_dataset, 500000),
        ("pyarrow_showcase.csv", generate_pyarrow_showcase_dataset, 5000000),
        ("polars_showcase.csv", generate_polars_showcase_dataset, 2000000),
        ("dask_showcase.csv", generate_dask_showcase_dataset, 10000000),
        ("spark_showcase.csv", generate_spark_showcase_dataset, 20000000),
    ]

    total_start = time.time()

    for filename, generator_func, num_rows in datasets:
        output_path = data_dir / filename

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            print(f"EXISTS: {filename} already exists ({size_mb:.1f} MB)")
            continue

        print(f"\nCreating {filename}")
        print("-" * 40)

        try:
            generator_func(str(output_path), num_rows)
        except KeyboardInterrupt:
            print(f"\nGeneration cancelled by user")
            if output_path.exists():
                output_path.unlink()
            break
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            if output_path.exists():
                output_path.unlink()

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("SPECIALIZED DATASETS SUMMARY")
    print("=" * 60)

    total_size = 0
    for filename, _, _ in datasets:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            total_size += size_mb
            print(f"SUCCESS: {filename:<25} | {size_mb:>8.1f} MB")
        else:
            print(f"MISSING: {filename:<25} | Missing")

    print("-" * 60)
    print(f"Total size: {total_size:.1f} MB")
    print(f"Generation time: {total_time:.1f} seconds")
    print(f"Storage location: {data_dir}")

    print("\nNext steps:")
    print("1. Create benchmark scripts for each specialized dataset")
    print("2. Run comparative analysis to show each technology's strengths")
    print("3. Generate visualization comparing performance on optimized data")


if __name__ == "__main__":
    main()
