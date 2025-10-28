"""
Anomaly Detection using XGBoost
Implementation of XGBoost for anomaly detection (alternative to MXNet)
Based on data_science.md comparative modeling plan
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AnomalyDetectorXGBoost:
    """XGBoost-based anomaly detection using One-Class classification approach"""

    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize XGBoost anomaly detector

        Args:
            contamination: Expected proportion of outliers (for threshold)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

        np.random.seed(random_state)
        print(f"XGBoost version: {xgb.__version__}")

    def load_data(self, data_path: str, sample_size: int = None):
        """Load preprocessed data"""
        print(f"Loading data from {data_path}...")
        start_time = time.time()

        # Load with Polars
        if data_path.endswith(".parquet"):
            df = pl.read_parquet(data_path)
        else:
            df = pl.read_csv(data_path)

        # Sample if needed
        if sample_size and len(df) > sample_size:
            print(f"   Sampling {sample_size:,} rows from {len(df):,} total rows...")
            df = df.sample(n=sample_size, seed=self.random_state)

        load_time = time.time() - start_time
        print(f"   Loaded {len(df):,} rows in {load_time:.2f}s")

        return df

    def prepare_features(self, df):
        """Prepare feature matrix"""
        print("\nPreparing features...")

        # Select numeric features
        feature_cols = [
            "category1",
            "category2",
            "category3",
            "flag",
            "value1_normalized",
            "value2_normalized",
            "year",
            "month",
            "quarter",
            "year_month_encoded",
            "code_encoded",
        ]

        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Using {len(available_cols)} features: {available_cols}")

        X = df.select(available_cols).to_numpy()
        print(f"   Feature matrix shape: {X.shape}")

        return X, available_cols

    def split_data(self, X, test_size=0.15, val_size=0.15):
        """Split data into train/val/test sets (70/15/15)"""
        print("\nSplitting data...")

        X_temp, X_test = train_test_split(
            X, test_size=test_size, random_state=self.random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(
            X_temp, test_size=val_ratio, random_state=self.random_state
        )

        print(
            f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)"
        )
        print(
            f"   Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)"
        )
        print(
            f"   Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)"
        )

        return X_train, X_val, X_test

    def train_model(self, X_train, X_val):
        """Train XGBoost model for anomaly detection using ensemble approach"""
        print("\n" + "=" * 80)
        print("XGBOOST ANOMALY DETECTION")
        print("=" * 80)

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024

        print("Initializing ensemble model...")

        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # POPRAWKA: Trenuj jeden model dla KAŻDEJ cechy
        # Anomaly score = średni błąd rekonstrukcji WSZYSTKICH cech
        self.models = []

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        print(
            f"\nTraining {X_train_scaled.shape[1]} XGBoost models (one per feature)..."
        )
        start_time = time.time()

        for i in range(X_train_scaled.shape[1]):
            # Trenuj model do predykcji i-tej cechy z pozostałych
            X_without_i = np.delete(X_train_scaled, i, axis=1)
            y_i = X_train_scaled[:, i]

            model = xgb.XGBRegressor(**params)
            model.fit(X_without_i, y_i, verbose=False)
            self.models.append(model)

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024
        mem_used = mem_after - mem_before

        print(f"\n   Training completed in {training_time:.2f}s")
        print(f"   Trained {len(self.models)} models")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used, X_train_scaled, X_val_scaled

    def calculate_anomaly_scores(self, X_scaled):
        """Calculate anomaly scores as mean reconstruction error across all features"""
        total_error = np.zeros(len(X_scaled))

        for i, model in enumerate(self.models):
            X_without_i = np.delete(X_scaled, i, axis=1)
            predictions = model.predict(X_without_i)
            errors = np.abs(predictions - X_scaled[:, i])
            total_error += errors

        # Average error across all features
        anomaly_scores = total_error / len(self.models)
        return anomaly_scores

    def set_threshold(self, X_val_scaled):
        """Set anomaly threshold based on validation set"""
        print("\nCalculating anomaly threshold...")
        val_scores = self.calculate_anomaly_scores(X_val_scaled)
        self.threshold = np.percentile(val_scores, (1 - self.contamination) * 100)
        print(f"   Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def evaluate_model(self, X_test_scaled):
        """Evaluate model on test set"""
        print(f"\nEvaluating XGBoost...")

        # Calculate anomaly scores
        start_time = time.time()
        test_scores = self.calculate_anomaly_scores(X_test_scaled)
        inference_time = time.time() - start_time

        # Detect anomalies (scores above threshold)
        y_pred = (test_scores > self.threshold).astype(int)

        # Count anomalies
        n_anomalies = np.sum(y_pred)
        anomaly_rate = n_anomalies / len(y_pred) * 100

        print(f"   Inference time: {inference_time:.2f}s")
        print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_rate:.2f}%)")
        print(
            f"   Inference speed: {len(X_test_scaled)/inference_time:,.0f} samples/sec"
        )
        print(f"   Mean anomaly score: {np.mean(test_scores):.6f}")
        print(f"   Max anomaly score: {np.max(test_scores):.6f}")

        results = {
            "model_name": "XGBoost Anomaly Detector",
            "n_samples": len(X_test_scaled),
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(anomaly_rate),
            "inference_time": float(inference_time),
            "inference_speed": float(len(X_test_scaled) / inference_time),
            "mean_anomaly_score": float(np.mean(test_scores)),
            "max_anomaly_score": float(np.max(test_scores)),
            "threshold": float(self.threshold),
        }

        return results, y_pred, test_scores

    def run_full_comparison(
        self, data_path: str, output_dir: str = "../results", sample_size: int = None
    ):
        """Run complete XGBoost anomaly detection"""
        print("=" * 80)
        print("XGBOOST ANOMALY DETECTION")
        print("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = self.load_data(data_path, sample_size=sample_size)
        X, feature_names = self.prepare_features(df)
        X_train, X_val, X_test = self.split_data(X)

        # Train model
        training_time, mem_used, X_train_scaled, X_val_scaled = self.train_model(
            X_train, X_val
        )

        # Set threshold from validation set
        self.set_threshold(X_val_scaled)

        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        results, predictions, scores = self.evaluate_model(X_test_scaled)
        results["training_time"] = training_time
        results["memory_usage_gb"] = mem_used

        # Save results
        comparison = {
            "dataset_info": {
                "data_path": data_path,
                "total_samples": int(len(df)),
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "test_samples": int(X_test.shape[0]),
            },
            "xgboost_detector": results,
            "configuration": {
                "contamination": self.contamination,
                "random_state": self.random_state,
                "max_depth": 6,
                "n_estimators": 100,
                "xgboost_version": xgb.__version__,
            },
        }

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Training Time':<30} {training_time:.2f}s")
        print(f"{'Memory Usage':<30} {mem_used:.2f} GB")
        print(f"{'Inference Time':<30} {results['inference_time']:.2f}s")
        print(
            f"{'Anomalies Detected':<30} {results['n_anomalies']:,} ({results['anomaly_rate']:.2f}%)"
        )
        print(f"{'Inference Speed':<30} {results['inference_speed']:,.0f} samples/s")

        # Save results
        output_file = f"{output_dir}/xgboost_anomaly_detection_results.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame(
            {"xgboost_anomaly": predictions, "anomaly_score": scores}
        )
        predictions_file = f"{output_dir}/xgboost_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        print("\n" + "=" * 80)
        print("XGBOOST ANALYSIS COMPLETE")
        print("=" * 80)

        return comparison


if __name__ == "__main__":
    # Run XGBoost anomaly detection on preprocessed data
    data_path = "../processed/processed_data.parquet"
    output_dir = "../results"

    # Use 1M sample (same as deep learning frameworks for fair comparison)
    detector = AnomalyDetectorXGBoost(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(
        data_path, output_dir, sample_size=10_000_000
    )

    print("\n[SUCCESS] XGBoost anomaly detection complete!")
    print("[COMPLETE] All 5 frameworks implemented!")
