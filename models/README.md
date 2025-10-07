# Anomaly Detection in Japanese Trade Data

## Overview

This project implements and compares multiple machine learning and deep learning frameworks for anomaly detection on large-scale Japanese trade data (custom_1988_2020.csv dataset with 113.6M rows).

## Project Structure

```
models/
├── README.md                              # This file
├── data_science.md                        # Comparative modeling plan
├── models.md                              # Preprocessing plan
├── preprocess_polars.py                   # Data preprocessing implementation
├── anomaly_detection_sklearn.py           # Scikit-learn implementation (COMPLETED)
├── visualize_sklearn_results.py           # Visualization script
├── processed/                             # Preprocessed data
│   ├── processed_data.parquet             # Main preprocessed dataset (10M rows)
│   ├── processed_data.csv                 # CSV format
│   ├── summary_statistics.csv             # Statistical summary
│   ├── year_month_encoding.csv            # Encoding mappings
│   └── code_encoding.csv                  # Encoding mappings
├── results/                               # Performance metrics and predictions
│   ├── sklearn_anomaly_detection_results.json
│   └── sklearn_predictions.csv
└── charts/                                # Visualizations
    ├── sklearn_comparison.png
    └── sklearn_metrics_table.png
```

## Completed Implementations

### ✅ Scikit-learn (Isolation Forest + LOF)

**Status:** COMPLETE
**Implementation:** [anomaly_detection_sklearn.py](anomaly_detection_sklearn.py)

**Results (1M sample):**
- **Isolation Forest:**
  - Training: 7.71s
  - Inference: 1.36s (109,941 samples/sec)
  - Memory: 0.005 GB
  - Anomalies: 1,488 (0.99%)

- **Local Outlier Factor (LOF):**
  - Training: 49.29s
  - Inference: 8.67s (17,293 samples/sec)
  - Memory: 0.13 GB
  - Anomalies: 1,539 (1.03%)

**Key Findings:**
- Isolation Forest is 6.4x faster than LOF
- Models agree 98% of the time
- Both detect ~1% anomaly rate (as configured)

## Planned Implementations

### 🔄 PyTorch (MLP Autoencoder)
**Status:** PENDING
Deep learning approach with flexible neural network architecture

### 🔄 TensorFlow/Keras (MLP Autoencoder)
**Status:** PENDING
High-level API with production scalability

### 🔄 MXNet (MLP Autoencoder)
**Status:** PENDING
Distributed computing and large-scale support

### 🔄 JAX (MLP Autoencoder)
**Status:** PENDING
Modern, high-performance deep learning

## Dataset Information

**Source:** Japanese customs/trade data (1988-2020)
**Original Size:** 113.6M rows, 4.23GB
**Preprocessed Sample:** 1M rows for model training

**Features (11 total):**
- `category1`, `category2`, `category3` - Product categories
- `flag` - Binary flag
- `value1_normalized`, `value2_normalized` - Scaled trade values
- `year`, `month`, `quarter` - Temporal features
- `year_month_encoded`, `code_encoded` - Categorical encodings

## How to Run

### 1. Preprocess Data (if needed)

```bash
cd models
python preprocess_polars.py
```

This creates preprocessed data in `processed/` directory.

### 2. Run Scikit-learn Anomaly Detection

```bash
cd models
python anomaly_detection_sklearn.py
```

Results saved to `results/sklearn_anomaly_detection_results.json`

### 3. Generate Visualizations

```bash
cd models
python visualize_sklearn_results.py
```

Charts saved to `charts/` directory.

## Configuration

### Sample Size
Default: 1M rows (for faster processing)
Modify in `anomaly_detection_sklearn.py`:

```python
results = detector.run_full_comparison(
    data_path,
    output_dir,
    sample_size=1_000_000  # Change this value
)
```

### Contamination Rate
Default: 1% (0.01)
Modify in `anomaly_detection_sklearn.py`:

```python
detector = AnomalyDetectorSklearn(
    contamination=0.01,  # Change this value
    random_state=42
)
```

## Evaluation Metrics

For each model, we track:
- **Training Time** - Time to fit the model
- **Inference Time** - Time to predict on test set
- **Memory Usage** - RAM consumption during training
- **Inference Speed** - Samples processed per second
- **Anomalies Detected** - Number and percentage of anomalies
- **Model Agreement** - How well different models agree on anomalies

## Requirements

```bash
pip install polars numpy scikit-learn matplotlib psutil
```

All dependencies are in the main project's `requirements.txt`.

## Research Goals

This project aims to provide:
1. **Fair comparison** of ML/DL frameworks for anomaly detection
2. **Practical insights** for real-world business applications
3. **Performance benchmarks** on large-scale data
4. **Reproducible results** for academic research

## Next Steps

1. ✅ Complete scikit-learn baseline (DONE)
2. 🔄 Implement PyTorch autoencoder
3. 🔄 Implement TensorFlow/Keras autoencoder
4. 🔄 Implement MXNet autoencoder
5. 🔄 Implement JAX autoencoder
6. 🔄 Create comprehensive framework comparison
7. 🔄 Generate final research visualizations

## Output Files

### Results JSON Format
```json
{
  "dataset_info": {
    "total_samples": 1000000,
    "n_features": 11,
    "train_samples": 700000,
    "val_samples": 150000,
    "test_samples": 150000
  },
  "isolation_forest": {
    "training_time": 7.71,
    "inference_time": 1.36,
    "memory_usage_gb": 0.005,
    "n_anomalies": 1488,
    "anomaly_rate": 0.99
  },
  "local_outlier_factor": {
    ...
  }
}
```

### Predictions CSV Format
```csv
isolation_forest_anomaly,lof_anomaly,both_agree,either_anomaly
0,0,1,0
1,1,1,1
0,1,0,1
...
```

## Notes

- Uses Polars for fast data preprocessing
- Implements data splitting (70/15/15 train/val/test)
- Tracks system resources (CPU, RAM)
- Supports sampling for faster prototyping
- All results reproducible (random_state=42)

## Contact & Support

This project is part of academic research on large-scale anomaly detection for Japanese trade data analysis.

---

**Last Updated:** 2025-10-07
**Status:** Phase 1 (scikit-learn) COMPLETE
