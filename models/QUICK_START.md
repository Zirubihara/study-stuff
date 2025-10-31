# Quick Start Guide - Anomaly Detection Frameworks

## 🚀 Run All Frameworks at Once

The easiest way to run all frameworks and generate comparison visualizations:

```bash
cd models
python run_all_models.py
```

This single command will:
1. ✅ Run all 5 frameworks sequentially (sklearn, xgboost, pytorch, tensorflow, jax)
2. ✅ Show real-time progress with colored output
3. ✅ Handle errors gracefully (continues even if one fails)
4. ✅ Automatically generate comparison visualizations
5. ✅ Print a comprehensive summary with all metrics

**Expected Duration:** 5-15 minutes (depending on your hardware)

---

## 📋 Advanced Usage

### Skip Specific Frameworks
```bash
# Skip PyTorch and TensorFlow (they take longer)
python run_all_models.py --skip pytorch,tensorflow

# Skip only JAX
python run_all_models.py --skip jax
```

### Run Only Specific Frameworks
```bash
# Run only the fastest frameworks
python run_all_models.py --only sklearn,xgboost

# Run only deep learning frameworks
python run_all_models.py --only pytorch,tensorflow,jax
```

---

## 📂 Output Files

After running, you'll find:

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

---

## 🔧 Manual Execution (Old Way)

If you prefer to run frameworks individually:

```bash
# Step 1: Preprocess data (if not done yet)
cd preprocessing
python preprocess_polars.py

# Step 2: Run each framework
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

---

## 📊 Expected Results (5M samples)

| Framework | Training Time | Inference Speed | Anomalies Detected |
|-----------|---------------|-----------------|-------------------|
| XGBoost | ~13s | 2.36M samples/s | 7,554 |
| Isolation Forest | ~16s | 236K samples/s | 7,552 |
| JAX | ~45s | 646K samples/s | 7,655 |
| TensorFlow | ~61s | 44K samples/s | 7,654 |
| PyTorch | ~346s | 114K samples/s | 7,550 |

---

## ⚠️ Prerequisites

Make sure you have:
1. Preprocessed data: `models/processed/processed_data.parquet`
2. All dependencies installed: `pip install -r ../requirements.txt`
3. Python 3.8+ with appropriate ML libraries

If preprocessed data doesn't exist, run:
```bash
cd preprocessing
python preprocess_polars.py
```

---

## 🎓 For Your Thesis

The `run_all_models.py` script ensures:
- ✅ **Fair comparison** - All frameworks run on identical data
- ✅ **Reproducibility** - Fixed random seeds (42)
- ✅ **Complete results** - JSON files with all metrics
- ✅ **Professional visualizations** - Ready for thesis inclusion
- ✅ **Consistency validation** - All frameworks detect ~1% anomalies

---

## 💡 Tips

1. **First Run:** Run all frameworks to establish baseline
2. **Quick Tests:** Use `--only sklearn,xgboost` for fastest results
3. **Deep Learning Only:** Use `--only pytorch,tensorflow,jax` to compare DL frameworks
4. **Troubleshooting:** Check individual logs in each framework's directory

---

## 🐛 Troubleshooting

### "Script not found" error
Make sure you're in the `models/` directory:
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
- Skip the slowest frameworks: `--skip pytorch`
- Or run them individually with more time between executions

### Framework fails but others continue
- This is expected! The script continues even if one fails
- Check the framework's log file for details
- The summary will show which succeeded/failed

---

**Need help?** Check the detailed documentation in `docs/README.md`






