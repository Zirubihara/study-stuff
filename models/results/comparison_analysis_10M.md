# ML/DL Framework Comparison - 10M Dataset Anomaly Detection

**Dataset**: 10M rows (9,999,999 samples)
**Test Set**: 1.5M samples
**Date**: October 12, 2025

---

## Executive Summary

Performance comparison of 5 ML/DL frameworks for anomaly detection on a 10M row dataset. XGBoost dominates in both training and inference speed, while JAX leads among deep learning frameworks.

---

## Training Performance

| Rank | Framework | Algorithm | Training Time | Memory Usage | Notes |
|------|-----------|-----------|---------------|--------------|-------|
| 🥇 | **XGBoost** | Gradient Boosting | **26.8s** | 0.42 GB | Fastest overall |
| 🥈 | **scikit-learn** | Isolation Forest | **64.1s** | -0.27 GB | 2nd fastest |
| 🥉 | **JAX** | Autoencoder | 141.3s | 0.48 GB | Best DL framework |
| 4 | **TensorFlow** | Autoencoder | 252.6s | 0.59 GB | Moderate speed |
| 5 | **scikit-learn** | LOF | 526.4s | 1.25 GB | Highest memory |
| 6 | **PyTorch** | Autoencoder | **1,183.9s** | 0.12 GB | **20 minutes!** |

**Key Findings:**
- XGBoost is **44x faster** than PyTorch for training
- PyTorch takes over 20 minutes to train (slowest)
- Isolation Forest has negative memory usage (likely measurement artifact)

---

## Inference Performance

| Rank | Framework | Algorithm | Inference Time | Speed (samples/s) | Speedup vs Slowest |
|------|-----------|-----------|----------------|-------------------|-------------------|
| 🥇 | **XGBoost** | Gradient Boosting | **0.76s** | **1,980,967** | **120x** |
| 🥈 | **JAX** | Autoencoder | **6.38s** | **235,212** | **14x** |
| 🥉 | **PyTorch** | Autoencoder | 7.85s | 191,193 | **12x** |
| 4 | **scikit-learn** | Isolation Forest | 11.79s | 127,203 | **8x** |
| 5 | **TensorFlow** | Autoencoder | 80.21s | 18,701 | **1.1x** |
| 6 | **scikit-learn** | LOF | **90.81s** | **16,518** | **1.0x** |

**Key Findings:**
- XGBoost processes **1.98 million samples per second** (fastest)
- JAX is **12.6x faster** than TensorFlow for inference
- LOF takes 91 seconds for 1.5M samples (not scalable)

---

## Anomaly Detection Results

| Framework | Anomalies Detected | Anomaly Rate | Agreement with Mean |
|-----------|-------------------|--------------|---------------------|
| **JAX** | 14,835 | 0.989% | Most conservative |
| **scikit-learn** (IF) | 14,886 | 0.992% | -0.8% |
| **XGBoost** | 15,053 | 1.004% | +0.4% |
| **PyTorch** | 15,082 | 1.005% | +0.6% |
| **TensorFlow** | 15,125 | 1.008% | +0.8% |
| **scikit-learn** (LOF) | 15,151 | 1.010% | Most aggressive |

**Consistency:**
- All models detect ~1% anomalies (target contamination rate)
- Variance: 316 samples (2.1% difference between min/max)
- All models show excellent calibration

---

## Framework Rankings

### Overall Performance (Speed + Accuracy)
1. **XGBoost** ⭐⭐⭐⭐⭐
   - Best training speed (26.8s)
   - Best inference speed (0.76s)
   - Good accuracy (1.004% anomaly rate)
   - **Recommended for production**

2. **JAX** ⭐⭐⭐⭐
   - Best deep learning performance
   - 2nd fastest inference (6.38s)
   - Good training speed (141.3s)
   - **Recommended for deep learning**

3. **scikit-learn (Isolation Forest)** ⭐⭐⭐⭐
   - Fast training (64.1s)
   - Decent inference (11.79s)
   - Simple to use
   - **Good general-purpose choice**

4. **PyTorch** ⭐⭐⭐
   - Very slow training (1,184s)
   - Good inference (7.85s)
   - Lowest memory (0.12 GB)
   - **Not recommended for CPU**

5. **TensorFlow** ⭐⭐
   - Slow inference (80.2s)
   - Moderate training (252.6s)
   - **Poor CPU performance**

6. **scikit-learn (LOF)** ⭐
   - Extremely slow (526s train, 91s inference)
   - Highest memory (1.25 GB)
   - **Not scalable for large datasets**

---

## Deep Learning Framework Comparison

| Metric | JAX | PyTorch | TensorFlow |
|--------|-----|---------|------------|
| **Training Time** | 141.3s | 1,183.9s ❌ | 252.6s |
| **Inference Time** | **6.38s** ✅ | 7.85s | 80.21s ❌ |
| **Memory Usage** | 0.48 GB | **0.12 GB** ✅ | 0.59 GB |
| **Inference Speed** | **235K/s** ✅ | 191K/s | 18.7K/s ❌ |
| **Training Speed Ratio** | 1.0x | **8.4x slower** ❌ | 1.8x slower |
| **Inference Speed Ratio** | 1.0x | 1.2x slower | **12.6x slower** ❌ |

**Winner: JAX**
- 8.4x faster training than PyTorch
- 12.6x faster inference than TensorFlow
- Best overall deep learning performance

---

## Use Case Recommendations

### 🏢 Production Deployment
**Choose: XGBoost**
- Ultra-fast inference (0.76s)
- Quick training (26.8s)
- High throughput (2M samples/sec)
- Low latency

### 🧠 Deep Learning Research
**Choose: JAX**
- Best DL performance (6.38s inference)
- Functional programming paradigm
- GPU/TPU ready
- Fast training (141.3s)

### 📊 Quick Prototyping
**Choose: scikit-learn Isolation Forest**
- Simple API
- Fast training (64.1s)
- No deep learning complexity
- Decent performance

### ❌ Avoid for Large Datasets
- **LOF**: Too slow (91s inference)
- **TensorFlow on CPU**: Poor inference performance (80s)
- **PyTorch on CPU**: Very slow training (20 minutes)

---

## Performance Gaps

### Training Time Gaps
- **XGBoost vs PyTorch**: 44.2x faster ⚡
- **Isolation Forest vs LOF**: 8.2x faster
- **JAX vs TensorFlow**: 1.8x faster

### Inference Time Gaps
- **XGBoost vs LOF**: 119.5x faster ⚡
- **JAX vs TensorFlow**: 12.6x faster
- **XGBoost vs JAX**: 8.4x faster

---

## Statistical Summary

### Training Time Statistics
- **Mean**: 365.9s
- **Median**: 196.95s
- **Min**: 26.8s (XGBoost)
- **Max**: 1,183.9s (PyTorch)
- **Std Dev**: 415.7s

### Inference Time Statistics
- **Mean**: 33.3s
- **Median**: 9.82s
- **Min**: 0.76s (XGBoost)
- **Max**: 90.81s (LOF)
- **Std Dev**: 37.8s

---

## Conclusions

1. **XGBoost dominates** for traditional ML anomaly detection with 120x faster inference
2. **JAX is the best deep learning framework** with 12.6x faster inference than TensorFlow
3. **PyTorch has slow training** (20 min) but acceptable inference on CPU
4. **TensorFlow struggles on CPU** with 80s inference time
5. **LOF doesn't scale** well for large datasets (91s for 1.5M samples)
6. All models show **consistent anomaly detection** (~1% rate)

### Final Recommendation Matrix

| Scenario | 1st Choice | 2nd Choice | Avoid |
|----------|-----------|-----------|-------|
| Production | XGBoost | JAX | LOF, TensorFlow |
| Deep Learning | JAX | PyTorch | TensorFlow (CPU) |
| CPU-only | XGBoost | Isolation Forest | PyTorch, TensorFlow |
| Memory-constrained | PyTorch | XGBoost | LOF |
| Quick training | XGBoost | Isolation Forest | PyTorch |
| Real-time inference | XGBoost | JAX | LOF, TensorFlow |

---

**Report Generated**: October 12, 2025
**Dataset Size**: 10,000,000 rows
**Test Set**: 1,500,000 samples
**Frameworks Tested**: 6 algorithms across 5 frameworks
