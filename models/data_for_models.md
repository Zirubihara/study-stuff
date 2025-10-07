# Preprocessing Plan for "100 million data (csv)" – Japanese Trade Dataset (Polars Workflow)

## Goal

Prepare the dataset for machine learning and deep learning anomaly detection, using Polars for fast and reproducible analysis.

---

## Dataset Description

- Source: "100 million data (csv)" – Japanese customs/import/export data since 1988
- Structure: Large tabular file, numeric, categorical, and date fields (e.g. product code, value, amount, country, date, description)
- No images or multimedia

---

## Preprocessing Steps

### 1. File Loading
- Read the CSV file into a Polars DataFrame (`pl.read_csv()`)

### 2. Handling Missing Data
- Check missing values in each column (`df.null_count()`)
- For numeric columns (e.g. value, amount): fill with mean or median
- For categorical columns (e.g. country, product code): fill with "Unknown" or the most frequent value

### 3. Data Type Corrections
- Ensure:
  - Numeric columns are `Float64`/`Int64`
  - Dates are properly parsed to `pl.Date`
  - Categories are set as `Categorical`

### 4. Encoding Categorical Features
- Country:
  - If fewer than 100 countries, use one-hot encoding (`df.to_dummies()`)
  - Otherwise, use categorical encoding
- Product code/type: categorical encoding or embeddings for deep learning

### 5. Scaling/Normalizing Numeric Columns
- Apply min-max scaling or standardization to value and amount columns

### 6. Date Parsing and Feature Expansion
- Parse date column to correct format
- Extract year, month, quarter to create additional seasonal features (`df.with_columns([pl.col('date').dt.year(), ...])`)

### 7. Outlier Detection and Handling
- Check distributions of key numeric columns
- Remove only obvious data entry errors; keep genuine anomalies for analysis

### 8. Feature Reduction After Encoding
- If one-hot encoding creates too many columns, keep only the most relevant ones or revert to categorical encoding

### 9. Export for ML/DL Analysis
- Save the processed DataFrame to CSV or Parquet (`df.write_csv()`, `df.write_parquet()`)

---

## Example for Typical Columns

| Column         | Target Type  | Processing                 |
|----------------|--------------|----------------------------|
| code           | Categorical  | Label/categorical encoding |
| value          | Float64      | Impute, normalize/scale    |
| amount         | Float64/Int  | Impute, normalize/scale    |
| country_from   | Categorical  | Label/one-hot encoding     |
| country_to     | Categorical  | Label/one-hot encoding     |
| date           | Date         | Parse, extract features    |
| description    | String       | Impute/drop/embed          |
| note           | String       | Impute/drop                |

---

## Final Note

This pipeline ensures your data are ready for anomaly detection with scikit-learn, PyTorch, TensorFlow, JAX, MXNet, and other ML/DL libraries.  
Polars enables efficient preprocessing even for very large CSV files.

_Pass this file directly to your next AI agent or teammate for streamlined implementation!_
