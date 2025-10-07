# Comparative Modeling Plan for Anomaly Detection in Japanese Trade Data

## Goal

Systematic comparison of the most popular machine learning and deep learning libraries for large-scale anomaly detection: scikit-learn, PyTorch, TensorFlow, MXNet, JAX.  
Emphasis on practical, reproducible, and business-relevant evaluation.

---

## Steps

### 1. Data Preparation

- Use cleaned, encoded, and scaled dataset (see processed_data.parquet/CSV).
- Split data into train, validation, and test sets (70/15/15%).

### 2. Task & Metrics

- Unsupervised anomaly detection: Find records statistically different from typical entries.
- Evaluation metrics:
    - Precision, recall, F1 (using manually labeled/verified subset if available)
    - Number of anomalies detected
    - Training and inference time
    - RAM/CPU/GPU resource consumption
    - Coding/workflow ease (subjective score)

### 3. Model Implementation

| Framework      | Model                    | Notes                                 |
|----------------|--------------------------|---------------------------------------|
| scikit-learn   | Isolation Forest, LOF    | Classic ML algorithms                 |
| PyTorch        | MLP Autoencoder          | Flexible, deep learning workflow      |
| TensorFlow     | MLP Autoencoder (Keras)  | High-level API, scalable              |
| MXNet          | MLP Autoencoder          | Distributed/large-scale support       |
| JAX            | MLP Autoencoder          | Modern, performant DL                 |

- Use the same features, hyperparameters, and data chunks wherever possible.

### 4. Comparative Evaluation

- Collect for each library:
    - Detection metrics (precision/recall/F1)
    - Training/inference time and resources
    - Coding complexity (lines, errors, subjective notes)
- Present in tables and graphs for clear side-by-side comparison.

### 5. Conclusion

- Assess which library is best for large-scale anomaly detection.
- Discuss practical pros, cons, pitfalls, and ideal use cases.
- Give recommendations for real business, compliance, and research settings.

---

## Notes

- All tests can be run on a modern Intel laptop with 32GB RAM (no GPU required, but faster with one).
- Data chunking/batching recommended for big files.

---

## Focus

_This plan is created for a fair, direct side-by-side comparison to inform both ML researchers and business users about the strengths and limitations of each solution for practical anomaly detection._
