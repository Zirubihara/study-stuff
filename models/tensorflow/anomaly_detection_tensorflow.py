"""
Anomaly Detection using TensorFlow/Keras
Implementation of MLP Autoencoder for anomaly detection
Based on data_science.md comparative modeling plan
"""

import polars as pl
import numpy as np
import time
import psutil
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging


class MLPAutoencoderKeras(Model):
    """Keras MLP Autoencoder for anomaly detection"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        """
        Initialize autoencoder architecture

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (encoder)
        """
        super(MLPAutoencoderKeras, self).__init__()

        # Encoder
        self.encoder_layers = []
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.encoder_layers.append(layers.Dropout(0.2))

        # Decoder (mirror of encoder)
        self.decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for i, hidden_dim in enumerate(reversed_dims):
            if i < len(reversed_dims) - 1:
                self.decoder_layers.append(layers.Dense(hidden_dim, activation='relu'))
                self.decoder_layers.append(layers.Dropout(0.2))
            else:
                self.decoder_layers.append(layers.Dense(hidden_dim, activation='linear'))

    def call(self, x, training=False):
        """Forward pass through autoencoder"""
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x, training=training)

        return x


class AnomalyDetectorTensorFlow:
    """TensorFlow/Keras-based anomaly detection using MLP Autoencoder"""

    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize TensorFlow anomaly detector

        Args:
            contamination: Expected proportion of outliers (for threshold)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    def load_data(self, data_path: str, sample_size: int = None):
        """Load preprocessed data"""
        print(f"Loading data from {data_path}...")
        start_time = time.time()

        # Load with Polars
        if data_path.endswith('.parquet'):
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
            'category1', 'category2', 'category3', 'flag',
            'value1_normalized', 'value2_normalized',
            'year', 'month', 'quarter',
            'year_month_encoded', 'code_encoded'
        ]

        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Using {len(available_cols)} features: {available_cols}")

        X = df.select(available_cols).to_numpy()
        print(f"   Feature matrix shape: {X.shape}")

        return X, available_cols

    def split_data(self, X, test_size=0.15, val_size=0.15):
        """Split data into train/val/test sets (70/15/15)"""
        print("\nSplitting data...")

        X_temp, X_test = train_test_split(X, test_size=test_size, random_state=self.random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(X_temp, test_size=val_ratio, random_state=self.random_state)

        print(f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

        return X_train, X_val, X_test

    def train_model(self, X_train, X_val, input_dim, epochs=5, batch_size=1024):
        """Train the autoencoder model"""
        print("\n" + "="*80)
        print("TENSORFLOW/KERAS MLP AUTOENCODER")
        print("="*80)

        # Track resources
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        print("Initializing model...")

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build model
        self.model = MLPAutoencoderKeras(input_dim=input_dim)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

        # Build the model by calling it once
        self.model.build(input_shape=(None, input_dim))

        # Count parameters
        total_params = self.model.count_params()
        print(f"   Total parameters: {total_params:,}")

        print(f"\nTraining for {epochs} epochs...")
        start_time = time.time()

        # Train model
        history = self.model.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, X_val_scaled),
            verbose=0,
            shuffle=True
        )

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        mem_used = mem_after - mem_before

        # Print training progress
        train_losses = history.history['loss']
        val_losses = history.history['val_loss']

        print(f"   Epoch [5/{epochs}] - Train Loss: {train_losses[4]:.6f}, Val Loss: {val_losses[4]:.6f}")
        print(f"   Epoch [{epochs}/{epochs}] - Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

        print(f"\n   Training completed in {training_time:.2f}s")
        print(f"   Final train loss: {train_losses[-1]:.6f}")
        print(f"   Final val loss: {val_losses[-1]:.6f}")
        print(f"   Best val loss: {min(val_losses):.6f}")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used, train_losses, val_losses, X_train_scaled, X_val_scaled

    def calculate_reconstruction_errors(self, X_scaled):
        """Calculate reconstruction errors for anomaly detection"""
        predictions = self.model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - predictions) ** 2, axis=1)
        return errors

    def set_threshold(self, X_val_scaled):
        """Set anomaly threshold based on validation set"""
        print("\nCalculating anomaly threshold...")
        val_errors = self.calculate_reconstruction_errors(X_val_scaled)
        self.threshold = np.percentile(val_errors, (1 - self.contamination) * 100)
        print(f"   Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def evaluate_model(self, X_test_scaled):
        """Evaluate model on test set"""
        print(f"\nEvaluating TensorFlow/Keras Autoencoder...")

        # Calculate reconstruction errors
        start_time = time.time()
        test_errors = self.calculate_reconstruction_errors(X_test_scaled)
        inference_time = time.time() - start_time

        # Detect anomalies (errors above threshold)
        y_pred = (test_errors > self.threshold).astype(int)

        # Count anomalies
        n_anomalies = np.sum(y_pred)
        anomaly_rate = n_anomalies / len(y_pred) * 100

        print(f"   Inference time: {inference_time:.2f}s")
        print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_rate:.2f}%)")
        print(f"   Inference speed: {len(X_test_scaled)/inference_time:,.0f} samples/sec")
        print(f"   Mean reconstruction error: {np.mean(test_errors):.6f}")
        print(f"   Max reconstruction error: {np.max(test_errors):.6f}")

        results = {
            'model_name': 'TensorFlow/Keras MLP Autoencoder',
            'n_samples': len(X_test_scaled),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'inference_time': float(inference_time),
            'inference_speed': float(len(X_test_scaled)/inference_time),
            'mean_reconstruction_error': float(np.mean(test_errors)),
            'max_reconstruction_error': float(np.max(test_errors)),
            'threshold': float(self.threshold)
        }

        return results, y_pred, test_errors

    def run_full_comparison(self, data_path: str, output_dir: str = "../results", sample_size: int = None):
        """Run complete TensorFlow anomaly detection"""
        print("="*80)
        print("TENSORFLOW/KERAS ANOMALY DETECTION")
        print("="*80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = self.load_data(data_path, sample_size=sample_size)
        X, feature_names = self.prepare_features(df)
        X_train, X_val, X_test = self.split_data(X)

        # Train model
        training_time, mem_used, train_losses, val_losses, X_train_scaled, X_val_scaled = self.train_model(
            X_train, X_val, input_dim=X.shape[1], epochs=5
        )

        # Set threshold from validation set
        self.set_threshold(X_val_scaled)

        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        results, predictions, errors = self.evaluate_model(X_test_scaled)
        results['training_time'] = training_time
        results['memory_usage_gb'] = mem_used

        # Save results
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
            'tensorflow_autoencoder': results,
            'configuration': {
                'contamination': self.contamination,
                'random_state': self.random_state,
                'epochs': 10,
                'batch_size': 1024,
                'tensorflow_version': tf.__version__
            }
        }

        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"{'Training Time':<30} {training_time:.2f}s")
        print(f"{'Memory Usage':<30} {mem_used:.2f} GB")
        print(f"{'Inference Time':<30} {results['inference_time']:.2f}s")
        print(f"{'Anomalies Detected':<30} {results['n_anomalies']:,} ({results['anomaly_rate']:.2f}%)")
        print(f"{'Inference Speed':<30} {results['inference_speed']:,.0f} samples/s")

        # Save results
        output_file = f"{output_dir}/tensorflow_anomaly_detection_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame({
            'tensorflow_anomaly': predictions,
            'reconstruction_error': errors
        })
        predictions_file = f"{output_dir}/tensorflow_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        print("\n" + "="*80)
        print("TENSORFLOW/KERAS ANALYSIS COMPLETE")
        print("="*80)

        return comparison


if __name__ == "__main__":
    # Run TensorFlow anomaly detection on preprocessed data
    data_path = "../processed/processed_data.parquet"
    output_dir = "../results"

    # Use 1M sample (same as PyTorch for fair comparison)
    detector = AnomalyDetectorTensorFlow(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(data_path, output_dir, sample_size=10_000_000)

    print("\n[SUCCESS] TensorFlow/Keras anomaly detection complete!")
    print("[NEXT] Compare with scikit-learn and PyTorch results")
