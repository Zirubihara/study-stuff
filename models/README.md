# Anomaly Detection Framework Comparison

Comparative analysis of 5 machine learning frameworks for anomaly detection on Japanese trade data (1988-2020).

## Project Structure

```
models/
├── run_all_models.py                 # Unified runner (runs all frameworks)
├── README.md                         # This file
│
├── sklearn/                          # Classical Machine Learning
│   └── anomaly_detection_sklearn.py  # Isolation Forest + LOF
├── xgboost/                          # Gradient Boosting
│   └── anomaly_detection_xgboost.py  # XGBoost Detector
├── pytorch/                          # Deep Learning Framework
│   └── anomaly_detection_pytorch.py  # MLP Autoencoder
├── tensorflow/                       # Deep Learning Framework
│   └── anomaly_detection_tensorflow.py # MLP Autoencoder
├── jax/                              # Modern Deep Learning
│   └── anomaly_detection_jax.py      # MLP Autoencoder with JIT
│
├── preprocessing/                    # Data Preparation
│   └── preprocess_polars.py          # Polars-based preprocessing
├── visualization/                    # Results Visualization
│   ├── compare_all_results.py        # Framework comparison charts
│   └── visualize_sklearn_results.py  # Sklearn-specific charts
│
├── results/                          # JSON results & CSV predictions
├── charts/                           # Generated visualizations
└── processed/                        # Preprocessed data
```

## Usage

### Running All Frameworks

The recommended approach is to run all frameworks sequentially using the unified runner:

```bash
cd models
python run_all_models.py
```

This command will:
1. Run all 5 frameworks sequentially (sklearn, xgboost, pytorch, tensorflow, jax)
2. Show real-time progress with colored output
3. Handle errors gracefully (continues even if one framework fails)
4. Automatically generate comparison visualizations
5. Print a comprehensive summary with all metrics

**Expected Duration:** 5-15 minutes (depending on hardware)

### Advanced Options

#### Skip Specific Frameworks

```bash
# Skip PyTorch and TensorFlow
python run_all_models.py --skip pytorch,tensorflow

# Skip only JAX
python run_all_models.py --skip jax
```

#### Run Only Specific Frameworks

```bash
# Run only classical ML frameworks
python run_all_models.py --only sklearn,xgboost

# Run only deep learning frameworks
python run_all_models.py --only pytorch,tensorflow,jax
```

### Manual Execution

For individual framework execution:

```bash
# Step 1: Preprocess data (if not done yet)
cd preprocessing
python preprocess_polars.py

# Step 2: Run individual frameworks
cd ../sklearn
python anomaly_detection_sklearn.py

cd ../xgboost
python anomaly_detection_xgboost.py

cd ../pytorch
python anomaly_detection_pytorch.py

cd ../tensorflow
python anomaly_detection_tensorflow.py

cd ../jax
python anomaly_detection_jax.py

# Step 3: Generate comparison visualizations
cd ../visualization
python compare_all_results.py
```

## Framework Classification

### Machine Learning (Classical)
1. **Scikit-learn** - Isolation Forest, Local Outlier Factor (LOF)
2. **XGBoost** - Gradient Boosting-based anomaly detection

### Deep Learning
3. **PyTorch** - MLP Autoencoder
4. **TensorFlow** - MLP Autoencoder
5. **JAX** - MLP Autoencoder (JIT-compiled)

## Results Summary

Performance metrics on 5,000,000 samples:

| Framework | Training Time | Inference Speed | Anomalies Detected |
|-----------|---------------|-----------------|-------------------|
| XGBoost | 13.24s | 2.36M samples/s | 7,554 |
| Isolation Forest | 15.96s | 236K samples/s | 7,552 |
| JAX | 45.12s | 646K samples/s | 7,655 |
| TensorFlow | 60.86s | 44K samples/s | 7,654 |
| PyTorch | 345.75s | 114K samples/s | 7,550 |

**Key Findings:**
- High agreement across all frameworks (~7,550 anomalies, 1.01-1.02%)
- XGBoost demonstrates fastest overall performance (training + inference)
- JAX achieves best performance among deep learning frameworks
- All frameworks validated on identical 5M dataset ensuring fair comparison

## Configuration

All frameworks use identical experimental settings:
- **Dataset**: 5,000,000 samples from Japanese trade data
- **Split**: 70% train / 15% validation / 15% test
- **Contamination**: 1% expected anomalies
- **Random State**: 42 (for reproducibility)
- **Features**: 11 features (normalized values, encoded categories, temporal features)

## Output Files

### Results (JSON)
- `results/sklearn_anomaly_detection_results.json`
- `results/pytorch_anomaly_detection_results.json`
- `results/tensorflow_anomaly_detection_results.json`
- `results/jax_anomaly_detection_results.json`
- `results/xgboost_anomaly_detection_results.json`

### Predictions (CSV)
- `results/sklearn_predictions.csv`
- `results/pytorch_predictions.csv`
- `results/tensorflow_predictions.csv`
- `results/jax_predictions.csv`
- `results/xgboost_predictions.csv`

### Visualizations (PNG)
- `charts/framework_comparison.png` - 6-panel comparison chart
- `charts/framework_comparison_table.png` - Summary table with rankings

### Logs
- `sklearn/sklearn_run.log`
- `pytorch/pytorch_run.log`
- `tensorflow/tensorflow_run.log`
- `jax/jax_run.log`
- `xgboost/xgboost_run.log`

## Prerequisites

Required components:
1. Preprocessed data: `models/processed/processed_data.parquet`
2. All dependencies installed: `pip install -r ../requirements.txt`
3. Python 3.8+ with appropriate ML libraries

If preprocessed data doesn't exist, run:
```bash
cd preprocessing
python preprocess_polars.py
```

## Troubleshooting

### "Script not found" error
Ensure you're in the `models/` directory:
```bash
cd models
python run_all_models.py
```

### "No preprocessed data" error
Run preprocessing first:
```bash
cd preprocessing
python preprocess_polars.py
cd ..
python run_all_models.py
```

### Out of memory error
- Skip memory-intensive frameworks: `--skip pytorch`
- Or run frameworks individually with more time between executions

### Framework fails but others continue
- This is expected behavior - the script continues even if one framework fails
- Check the framework's log file for details
- The summary will show which frameworks succeeded/failed

## Validation

All 5 frameworks have been validated and ensure:
1. **Fair comparison** - identical 5M sample dataset and configuration
2. **High consistency** - all frameworks detect approximately 1.01% anomalies
3. **Performance benchmarks** - comprehensive speed and accuracy metrics
4. **Reproducibility** - fixed random seeds, documented parameters
5. **Scalability** - tested at scale with real-world data

---

**Project**: Comparative ML/DL Framework Analysis for Anomaly Detection  
**Dataset**: Japanese Trade Dataset (1988-2020, 113.6M rows original)  
**Sample Size**: 5,000,000 rows for all benchmarks
