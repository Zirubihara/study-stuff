"""
Anomaly Detection using scikit-learn
Implementation of Isolation Forest and Local Outlier Factor (LOF)
Based on data_science.md comparative modeling plan
"""

import polars as pl
import numpy as np
import time
import psutil
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectorSklearn:
    """Scikit-learn based anomaly detection using Isolation Forest and LOF"""

    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize anomaly detectors

        Args:
            contamination: Expected proportion of outliers (default 1%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.lof = None
        self.results = {}

    def load_data(self, data_path: str, sample_size: int = None):
        """Load preprocessed data"""
        print(f"Loading data from {data_path}...")
        start_time = time.time()

        # Load with Polars for efficiency
        if data_path.endswith('.parquet'):
            df = pl.read_parquet(data_path)
        else:
            df = pl.read_csv(data_path)

        # Sample for faster processing if needed
        if sample_size and len(df) > sample_size:
            print(f"   Sampling {sample_size:,} rows from {len(df):,} total rows...")
            df = df.sample(n=sample_size, seed=self.random_state)

        load_time = time.time() - start_time
        print(f"   Loaded {len(df):,} rows in {load_time:.2f}s")

        return df

    def prepare_features(self, df):
        """Prepare feature matrix for anomaly detection"""
        print("\nPreparing features...")

        # Select numeric features for anomaly detection
        feature_cols = [
            'category1', 'category2', 'category3', 'flag',
            'value1_normalized', 'value2_normalized',
            'year', 'month', 'quarter',
            'year_month_encoded', 'code_encoded'
        ]

        # Check which columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Using {len(available_cols)} features: {available_cols}")

        # Convert to numpy array
        X = df.select(available_cols).to_numpy()

        print(f"   Feature matrix shape: {X.shape}")
        return X, available_cols

    def split_data(self, X, test_size=0.15, val_size=0.15):
        """Split data into train/val/test sets (70/15/15)"""
        print("\nSplitting data...")

        # First split: train+val vs test
        X_temp, X_test = train_test_split(
            X, test_size=test_size, random_state=self.random_state
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(
            X_temp, test_size=val_ratio, random_state=self.random_state
        )

        print(f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

        return X_train, X_val, X_test

    def train_isolation_forest(self, X_train):
        """Train Isolation Forest model"""
        print("\n" + "="*80)
        print("ISOLATION FOREST")
        print("="*80)

        # Track resources
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        print("Training Isolation Forest...")
        start_time = time.time()

        # Initialize and train
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1,
            verbose=0
        )

        self.isolation_forest.fit(X_train)

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        mem_used = mem_after - mem_before

        print(f"   Training completed in {training_time:.2f}s")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used

    def train_lof(self, X_train):
        """Train Local Outlier Factor model"""
        print("\n" + "="*80)
        print("LOCAL OUTLIER FACTOR (LOF)")
        print("="*80)

        # Track resources
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        print("Training LOF...")
        start_time = time.time()

        # Initialize and train
        self.lof = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=20,
            novelty=True,  # Enable prediction on new data
            n_jobs=-1
        )

        self.lof.fit(X_train)

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        mem_used = mem_after - mem_before

        print(f"   Training completed in {training_time:.2f}s")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used

    def evaluate_model(self, model, X_test, model_name):
        """Evaluate model on test set"""
        print(f"\nEvaluating {model_name}...")

        # Inference time
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time

        # Convert to binary (1=normal, -1=anomaly in sklearn)
        # Convert to (1=anomaly, 0=normal) for standard metrics
        y_pred = (predictions == -1).astype(int)

        # Count anomalies
        n_anomalies = np.sum(y_pred)
        anomaly_rate = n_anomalies / len(y_pred) * 100

        print(f"   Inference time: {inference_time:.2f}s")
        print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_rate:.2f}%)")
        print(f"   Inference speed: {len(X_test)/inference_time:,.0f} samples/sec")

        results = {
            'model_name': model_name,
            'n_samples': len(X_test),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'inference_time': float(inference_time),
            'inference_speed': float(len(X_test)/inference_time)
        }

        return results, y_pred

    def run_full_comparison(self, data_path: str, output_dir: str = "models/results", sample_size: int = None):
        """Run complete anomaly detection comparison"""
        print("="*80)
        print("SCIKIT-LEARN ANOMALY DETECTION COMPARISON")
        print("="*80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = self.load_data(data_path, sample_size=sample_size)
        X, feature_names = self.prepare_features(df)
        X_train, X_val, X_test = self.split_data(X)

        # Train Isolation Forest
        if_train_time, if_mem = self.train_isolation_forest(X_train)
        if_results, if_predictions = self.evaluate_model(
            self.isolation_forest, X_test, "Isolation Forest"
        )
        if_results['training_time'] = if_train_time
        if_results['memory_usage_gb'] = if_mem

        # Train LOF
        lof_train_time, lof_mem = self.train_lof(X_train)
        lof_results, lof_predictions = self.evaluate_model(
            self.lof, X_test, "Local Outlier Factor"
        )
        lof_results['training_time'] = lof_train_time
        lof_results['memory_usage_gb'] = lof_mem

        # Compare results
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        comparison = {
            'dataset_info': {
                'data_path': data_path,
                'total_samples': int(len(df)),
                'n_features': len(feature_names),
                'feature_names': feature_names,
                'train_samples': int(X_train.shape[0]),
                'val_samples': int(X_val.shape[0]),
                'test_samples': int(X_test.shape[0])
            },
            'isolation_forest': if_results,
            'local_outlier_factor': lof_results,
            'configuration': {
                'contamination': self.contamination,
                'random_state': self.random_state
            }
        }

        # Print comparison table
        print(f"\n{'Metric':<30} {'Isolation Forest':<20} {'LOF':<20}")
        print("-"*70)
        print(f"{'Training Time (s)':<30} {if_train_time:<20.2f} {lof_train_time:<20.2f}")
        print(f"{'Memory Usage (GB)':<30} {if_mem:<20.2f} {lof_mem:<20.2f}")
        print(f"{'Inference Time (s)':<30} {if_results['inference_time']:<20.2f} {lof_results['inference_time']:<20.2f}")
        print(f"{'Anomalies Detected':<30} {if_results['n_anomalies']:<20,} {lof_results['n_anomalies']:<20,}")
        print(f"{'Anomaly Rate (%)':<30} {if_results['anomaly_rate']:<20.2f} {lof_results['anomaly_rate']:<20.2f}")
        print(f"{'Inference Speed (samples/s)':<30} {if_results['inference_speed']:<20,.0f} {lof_results['inference_speed']:<20,.0f}")

        # Save results
        output_file = f"{output_dir}/sklearn_anomaly_detection_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame({
            'isolation_forest_anomaly': if_predictions,
            'lof_anomaly': lof_predictions,
            'both_agree': (if_predictions == lof_predictions).astype(int),
            'either_anomaly': ((if_predictions == 1) | (lof_predictions == 1)).astype(int)
        })

        predictions_file = f"{output_dir}/sklearn_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        # Agreement analysis
        agreement = np.sum(if_predictions == lof_predictions) / len(if_predictions) * 100
        both_anomaly = np.sum((if_predictions == 1) & (lof_predictions == 1))

        print(f"\n{'Agreement Analysis':<30}")
        print("-"*70)
        print(f"{'Models agree':<30} {agreement:.2f}%")
        print(f"{'Both detect anomaly':<30} {both_anomaly:,} samples")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        return comparison


if __name__ == "__main__":
    # Run anomaly detection on preprocessed data
    data_path = "processed/processed_data.parquet"
    output_dir = "results"

    # Use 1M sample for faster processing (adjust as needed)
    detector = AnomalyDetectorSklearn(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(data_path, output_dir, sample_size=1_000_000)

    print("\n[SUCCESS] Scikit-learn anomaly detection complete!")
    print(f"[NEXT] Implement PyTorch, TensorFlow, MXNet, JAX models")
