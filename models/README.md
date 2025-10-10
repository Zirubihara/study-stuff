# Anomaly Detection Framework Comparison

Comparative analysis of 5 machine learning frameworks for anomaly detection on Japanese trade data (1988-2020).

## 📁 Project Structure

```
models/
├── sklearn/                          # Classical Machine Learning
│   └── anomaly_detection_sklearn.py  # Isolation Forest + LOF
├── pytorch/                          # Deep Learning Framework
│   └── anomaly_detection_pytorch.py  # MLP Autoencoder
├── tensorflow/                       # Deep Learning Framework (Google)
│   └── anomaly_detection_tensorflow.py # MLP Autoencoder
├── jax/                              # Modern Deep Learning
│   └── anomaly_detection_jax.py      # MLP Autoencoder with JIT
├── xgboost/                          # Gradient Boosting
│   └── anomaly_detection_xgboost.py  # XGBoost Detector
├── preprocessing/                    # Data Preparation
│   └── preprocess_polars.py          # Polars-based preprocessing
├── visualization/                    # Results Visualization
│   ├── compare_all_results.py        # Framework comparison charts
│   └── visualize_sklearn_results.py  # Sklearn-specific charts
├── results/                          # JSON results files
├── charts/                           # Generated visualizations
├── processed/                        # Preprocessed data
└── docs/                             # Documentation
    ├── README.md                     # English documentation
    ├── README_PL.md                  # Polish documentation
    ├── SUMMARY.md                    # English summary
    ├── PODSUMOWANIE.md               # Polish summary
    ├── FINAL_THESIS_RESULTS.md       # Thesis results
    ├── data_science.md               # Modeling plan
    └── data_for_models.md            # Preprocessing plan
```

## 🚀 Quick Start

### 1. Preprocess Data
```bash
cd preprocessing
python preprocess_polars.py
```

### 2. Run Individual Frameworks
```bash
# Scikit-learn
cd sklearn
python anomaly_detection_sklearn.py

# PyTorch
cd pytorch
python anomaly_detection_pytorch.py

# TensorFlow
cd tensorflow
python anomaly_detection_tensorflow.py

# JAX
cd jax
python anomaly_detection_jax.py

# XGBoost
cd xgboost
python anomaly_detection_xgboost.py
```

### 3. Generate Comparison Charts
```bash
cd visualization
python compare_all_results.py
```

## 📊 Framework Classification

### Machine Learning (Classical)
1. **Scikit-learn** - Isolation Forest, LOF
2. **XGBoost** - Gradient Boosting

### Deep Learning
3. **PyTorch** - MLP Autoencoder
4. **TensorFlow** - MLP Autoencoder
5. **JAX** - MLP Autoencoder (JIT-compiled)

## 🎯 Results Summary (5M Samples)

| Framework | Training Time | Inference Speed | Anomalies Detected |
|-----------|---------------|-----------------|-------------------|
| XGBoost | 13.24s | 2.36M samples/s | 7,554 |
| Isolation Forest | 15.96s | 236K samples/s | 7,552 |
| JAX | 45.12s | 646K samples/s | 7,655 |
| TensorFlow | 60.86s | 44K samples/s | 7,654 |
| PyTorch | 345.75s | 114K samples/s | 7,550 |

**Key Findings:**
- ✅ High agreement across all frameworks (~7,550 anomalies, 1.01-1.02%)
- ⚡ XGBoost fastest overall (training + inference)
- 🔥 JAX fastest among deep learning frameworks
- 📊 All frameworks validated on identical 5M dataset (fair comparison)

## 📝 Configuration

All frameworks use identical settings:
- **Dataset**: 5,000,000 samples from Japanese trade data
- **Split**: 70% train / 15% validation / 15% test
- **Contamination**: 1% expected anomalies
- **Random State**: 42 (reproducibility)
- **Features**: 11 features (normalized values, encoded categories, temporal features)

## 🔧 Requirements

See main project [requirements.txt](../requirements.txt) for all dependencies.

## 📖 Documentation

Full documentation available in [docs/](docs/) folder:
- English: [README.md](docs/README.md), [SUMMARY.md](docs/SUMMARY.md)
- Polish: [README_PL.md](docs/README_PL.md), [PODSUMOWANIE.md](docs/PODSUMOWANIE.md)
- Thesis: [FINAL_THESIS_RESULTS.md](docs/FINAL_THESIS_RESULTS.md)

## 🎓 For Thesis

All 5 frameworks are validated and ready for academic use. Results demonstrate:
1. **Fair comparison** - identical data and configuration
2. **High consistency** - all frameworks detect similar anomalies
3. **Performance trade-offs** - speed vs flexibility documented
4. **Reproducible** - fixed random seeds, documented parameters

---

**Project**: Comparative ML/DL Framework Analysis for Anomaly Detection
**Data**: Japanese Trade Dataset (1988-2020, 113.6M rows original)
**Sample Size**: 5M rows for all benchmarks
