# Anomaly Detection Framework Comparison

Comparative analysis of 5 machine learning frameworks for anomaly detection on Japanese trade data (1988-2020).

## 📁 Project Structure

```
models/
├── run_all_models.py                 # Unified runner (runs all frameworks) ⭐
├── THESIS_SUMMARY.md                 # Complete thesis documentation ⭐
├── QUICK_START.md                    # Usage guide
├── README.md                         # This file
│
├── sklearn/                          # Classical Machine Learning
│   └── anomaly_detection_sklearn.py  # Isolation Forest + LOF
├── xgboost/                          # Gradient Boosting
│   └── anomaly_detection_xgboost.py  # XGBoost Detector
├── pytorch/                          # Deep Learning Framework
│   └── anomaly_detection_pytorch.py  # MLP Autoencoder
├── tensorflow/                       # Deep Learning Framework (Google)
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
├── processed/                        # Preprocessed data
└── docs/                             # Polish documentation
    ├── README_PL.md                  # Polski README
    └── PODSUMOWANIE.md               # Polskie podsumowanie
```

## 🚀 Quick Start

### **Easy Way - Run All Frameworks at Once** ⭐

```bash
cd models
python run_all_models.py
```

This runs all 5 frameworks, generates comparison charts, and prints a comprehensive summary!

**Duration:** 5-15 minutes | **See:** [QUICK_START.md](QUICK_START.md) for advanced options

---

### Manual Way - Run Individual Frameworks

```bash
# 1. Preprocess Data (if needed)
cd preprocessing
python preprocess_polars.py

# 2. Run Individual Frameworks
cd sklearn
python anomaly_detection_sklearn.py

cd ../xgboost
python anomaly_detection_xgboost.py

cd ../pytorch
python anomaly_detection_pytorch.py

cd ../tensorflow
python anomaly_detection_tensorflow.py

cd ../jax
python anomaly_detection_jax.py

# 3. Generate Comparison Charts
cd ../visualization
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

**Main Documentation:**
- 📊 **[THESIS_SUMMARY.md](THESIS_SUMMARY.md)** - Complete thesis summary with all results ⭐
- 🚀 **[QUICK_START.md](QUICK_START.md)** - Quick start guide for running frameworks
- 📝 **[README.md](README.md)** - This file (project overview)

**Polish Documentation:**
- [README_PL.md](docs/README_PL.md) - Polski README
- [PODSUMOWANIE.md](docs/PODSUMOWANIE.md) - Polskie podsumowanie

## 🎓 For Thesis

**📊 [→ See THESIS_SUMMARY.md for complete thesis documentation](THESIS_SUMMARY.md)**

All 5 frameworks validated and ready for academic use:
1. **Fair comparison** - identical 5M sample dataset and configuration
2. **High consistency** - all frameworks detect ~1.01% anomalies
3. **Performance benchmarks** - comprehensive speed and accuracy metrics
4. **Reproducible** - fixed random seeds, documented parameters
5. **Production-ready** - tested at scale with real-world data

---

**Project**: Comparative ML/DL Framework Analysis for Anomaly Detection
**Data**: Japanese Trade Dataset (1988-2020, 113.6M rows original)
**Sample Size**: 5M rows for all benchmarks
