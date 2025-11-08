# Python Data Processing Speed Test 🚀

**TL;DR**: This project tests how fast different Python libraries process large datasets. **Spoiler alert: Polars wins by a lot!**

## What This Project Does

Imagine you have a huge CSV file with millions of rows (like sales data, user logs, etc.). This project tests 5 different Python libraries to see which one processes the data fastest:

## 📊 Dataset Source (REAL-WORLD DATA)

This project uses **REAL production data** from the **Japanese Trade Statistics dataset (1988-2020)** for benchmarking:

- **Source**: [100 million data (csv) on Kaggle](https://www.kaggle.com/datasets/zanjibar/100-million-data-csv/data) by TadashiNagao
- **Original Data**: [Japan Customs](https://www.customs.go.jp/toukei/info/index_e.htm)
- **License**: CC BY-SA 4.0
- **Original Size**: 4.23 GB, 113.6 million rows (custom_1988_2020.csv)
- **Content**: Real Japanese import/export trade records from 1988-2020 with anonymized column names for universal benchmarking

**Why Real Data Matters**: Unlike synthetic benchmarks, this uses actual production data with real-world patterns, distributions, and characteristics - making the performance results much more credible and applicable to real scenarios.

**Column mapping** (anonymized for general benchmarking):
- `year_month` ← ym (year + month)
- `category1` ← exp_imp (export/import indicator)
- `category2` ← hs9 (Harmonized System product code)
- `category3` ← Customs (customs point code)
- `code` ← Country (country code)
- `flag` ← additional classifier
- `value1` ← Q1/Q2 (quantity)
- `value2` ← Value (in thousands of yen)

The dataset has been processed with generalized column names to focus on **data processing performance** rather than domain-specific analysis.

## Libraries Tested

- **Pandas** 🐼 - The most popular one everyone knows
- **Polars** ⚡ - The new super-fast kid on the block
- **PyArrow** 🏹 - Good for pure numbers
- **Dask** 🌪️ - For when your data is too big for memory
- **PySpark** ⚡ - For truly massive datasets

## 🏆 The Results (Spoiler: Polars Dominates!)

We tested all libraries on different dataset sizes. Here's what we found:

### 📊 Speed Comparison (Real-World Data Results)

**Updated with real production data from custom_1988_2020.csv (1988-2020 trade statistics):**

| Dataset Size | Winner | 2nd Place | Pandas Time | Winner Time | Speed Boost |
|-------------|--------|-----------|-------------|-------------|-------------|
| 1M rows     | **Polars** | PyArrow   | 1.51s       | 0.42s       | **3.6x faster** |
| 5M rows     | **Polars** | PyArrow   | 7.69s       | 1.84s       | **4.2x faster** |
| 10M rows    | **Polars** | PyArrow   | 16.95s      | 2.87s       | **5.9x faster** |
| 50M rows    | **Polars** | PyArrow   | 122.75s     | 22.95s      | **5.3x faster** |

*Note: All benchmarks use real-world data extracted from the original 113.6M row dataset, not synthetic data.*

### 💾 Memory Usage

**Polars uses 50-60% less memory than Pandas!**

- 10M rows: Pandas = 1.46GB, Polars = 0.55GB
- 50M rows: Pandas = 7.22GB, Polars = 2.66GB

## 🎯 When Should You Use What?

### Use **Polars** when:
- ✅ You want the fastest performance (almost always)
- ✅ You care about memory usage
- ✅ You have any dataset from 100K to 50M+ rows
- ✅ You want modern, clean syntax

### Use **Pandas** when:
- ✅ You're working with small datasets (< 1M rows)
- ✅ Your team already knows Pandas well
- ✅ You need specific Pandas-only features
- ⚠️ **Warning**: Gets slow and memory-hungry with large data

### Use **PyArrow** when:
- ✅ You have pure numerical data (no text processing)
- ✅ You need to work with Apache ecosystem tools
- ✅ You want good performance with numerical operations

### Use **PySpark** when:
- ✅ You have truly massive datasets (100M+ rows)
- ✅ You have a cluster of computers
- ✅ You need distributed processing

### Use **Dask** when:
- ✅ Your data doesn't fit in memory
- ✅ You want to scale existing Pandas code
- ⚠️ **Note**: Usually slower than other options

## 🚀 Quick Start - Run the Tests Yourself!

### Step 1: Setup (5 minutes)

```bash
# 1. Clone this project
git clone <your-repo-url>
cd study-stuff

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install required packages
pip install -r requirements.txt
```

### Step 2: Extract Real-World Data Subsets (5 minutes, ONE TIME)

```bash
# First time only: Extract real-world data subsets from the original dataset
cd data_manipulation/data_generation
python extract_real_world_datasets.py

# This creates benchmark datasets (1M, 5M, 10M, 50M, 100M rows) from custom_1988_2020.csv
```

### Step 3: Run Benchmarks (10-20 minutes total)

```bash
# Navigate to the benchmark folder
cd ../benchmarks/dataset_specific

# Run tests on different dataset sizes (now using REAL data!)
python benchmark_1m_simple.py     # Takes ~30 seconds
python benchmark_5m_simple.py     # Takes ~2 minutes
python benchmark_10m_simple.py    # Takes ~3 minutes
python benchmark_50m_simple.py    # Takes ~8 minutes
python benchmark_100m_simple.py   # Takes ~15 minutes (for big data comparison)
```

### Step 4: See Your Results

```bash
# Go back to main folder
cd ../../..

# Look at the results
ls results/

# You'll see files like:
# performance_metrics_polars_10m.json
# performance_metrics_pandas_10m.json
# etc.
```

## 📁 Where to Find Everything

```
study-stuff/
├── data_manipulation/benchmarks/dataset_specific/    ← The benchmark tests are here
├── results/                                          ← Results appear here as JSON files
├── data/                                             ← CSV datasets are stored here
├── charts/                                           ← Generated charts go here
└── README.md                                         ← You are here!
```

## 📊 Understanding the Results

Each test creates a JSON file with timing results. Here's what the numbers mean:

```json
{
    "total_operation_time_seconds": 2.87,    ← Total time to process everything
    "loading_time_seconds": 0.503,          ← Time to load the CSV file
    "cleaning_time_seconds": 0.005,         ← Time to clean missing data
    "aggregation_time_seconds": 0.780,      ← Time to group and calculate averages
    "sorting_time_seconds": 1.008,          ← Time to sort the data
    "filtering_time_seconds": 0.057,        ← Time to filter rows
    "correlation_time_seconds": 0.515,      ← Time to calculate correlations
    "memory_size_gb": 0.549,                ← Memory used (in GB)
    "row_count": 10000000                   ← Number of rows processed
}
```

**Lower numbers = faster performance! 🏃‍♂️💨**

## 🎨 Generate Pretty Charts

```bash
cd data_manipulation/visualization
python create_presentation_charts.py

# Charts will appear in ../../charts/ folder
```

## 🔧 Troubleshooting (Common Issues)

### "Java Error" with PySpark
If you see Java version errors:

1. **Install Java 17+**:
   - Windows: Download from [Oracle](https://www.oracle.com/java/technologies/downloads/)
   - Mac: `brew install openjdk@17`
   - Linux: `sudo apt install openjdk-17-jdk`

2. **Set JAVA_HOME**:
   ```bash
   # Windows
   set JAVA_HOME=C:\Program Files\Java\jdk-17

   # Mac/Linux
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
   ```

### "Out of Memory" Errors
- Close other programs
- Start with smaller datasets (1M, 5M) first
- Make sure you have at least 8GB RAM for 50M row tests

### "File Not Found" Errors
Make sure you have the original custom_1988_2020.csv file and run the extraction script:

```bash
cd data_manipulation/data_generation
python extract_real_world_datasets.py
```

## 🤓 What Operations Do We Test?

Each benchmark runs these common data processing tasks:

1. **Loading** - Reading a CSV file into memory
2. **Cleaning** - Handling missing/null values
3. **Aggregating** - Grouping data and calculating averages
4. **Sorting** - Ordering data by values
5. **Filtering** - Selecting rows that match conditions
6. **Correlations** - Finding relationships between columns

These are the bread-and-butter operations you'll do with any real dataset!

## 🎯 Key Takeaways for Beginners

1. **Polars is usually your best choice** - It's fast, memory-efficient, and has a clean API
2. **Pandas is fine for small data** - But switch to Polars when you hit performance issues
3. **Size matters** - What works for 100K rows might be unusably slow for 10M rows
4. **Memory usage matters** - Some libraries use 3x more RAM than others
5. **Real data tells the truth** - These benchmarks use actual production data (113.6M rows from Japanese trade statistics 1988-2020), not synthetic data
6. **Test with your own data** - While our results use real-world data, your specific use case may vary

## 🤝 Need Help?

- **New to data processing?** Start with the 1M row benchmark to see the basics
- **Working with big data?** Focus on the 50M row results
- **Questions?** Open an issue in this repository

## 📚 Want to Learn More?

- [Polars User Guide](https://pola-rs.github.io/polars-book/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

---

**Happy data processing! 🎉**

*Remember: The best library is the one that solves your specific problem efficiently. But if you're unsure, Polars is a pretty safe bet these days!*

---

## 📝 About This Dataset

All benchmarks in this project use **real production data** extracted from the custom_1988_2020.csv dataset:
- **113.6 million rows** of actual Japanese trade statistics (1988-2020)
- **4.23 GB** of real-world data with genuine patterns and distributions
- Subsets extracted: 1M, 5M, 10M, 50M, 100M rows for scalability testing
- Much more credible than synthetic benchmarks for research and production use cases