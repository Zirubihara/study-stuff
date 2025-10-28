"""
Anomaly Detection using JAX
Implementation of MLP Autoencoder for anomaly detection
FIXED VERSION: 50 epochs + early stopping
"""

import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
import psutil
from jax import grad, jit, random, vmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class MLPAutoencoderJAX:
    """JAX MLP Autoencoder for anomaly detection"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], key=None):
        """
        Initialize autoencoder architecture

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (encoder)
            key: JAX random key
        """
        if key is None:
            key = random.PRNGKey(42)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.params = self._initialize_params(key)

    def _initialize_params(self, key):
        """Initialize network parameters"""
        params = {}
        layer_dims = [self.input_dim] + self.hidden_dims

        # Encoder parameters
        for i in range(len(self.hidden_dims)):
            key, subkey1, subkey2 = random.split(key, 3)
            params[f'encoder_w{i}'] = random.normal(subkey1, (layer_dims[i], layer_dims[i+1])) * 0.1
            params[f'encoder_b{i}'] = jnp.zeros(layer_dims[i+1])

        # Decoder parameters (mirror of encoder)
        reversed_dims = list(reversed(self.hidden_dims)) + [self.input_dim]
        for i in range(len(reversed_dims) - 1):
            key, subkey1, subkey2 = random.split(key, 3)
            params[f'decoder_w{i}'] = random.normal(subkey1, (reversed_dims[i], reversed_dims[i+1])) * 0.1
            params[f'decoder_b{i}'] = jnp.zeros(reversed_dims[i+1])

        return params

    def forward(self, params, x):
        """Forward pass through autoencoder"""
        # Encoder
        h = x
        for i in range(len(self.hidden_dims)):
            h = jnp.dot(h, params[f'encoder_w{i}']) + params[f'encoder_b{i}']
            h = jax.nn.relu(h)

        # Decoder
        reversed_dims = list(reversed(self.hidden_dims)) + [self.input_dim]
        for i in range(len(reversed_dims) - 1):
            h = jnp.dot(h, params[f'decoder_w{i}']) + params[f'decoder_b{i}']
            if i < len(reversed_dims) - 2:
                h = jax.nn.relu(h)

        return h

    def loss_fn(self, params, x):
        """MSE loss function"""
        pred = self.forward(params, x)
        return jnp.mean((x - pred) ** 2)

    def count_params(self):
        """Count total parameters"""
        total = 0
        for key, value in self.params.items():
            total += value.size
        return total


class AnomalyDetectorJAX:
    """JAX-based anomaly detection using MLP Autoencoder"""

    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize JAX anomaly detector

        Args:
            contamination: Expected proportion of outliers (for threshold)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.key = random.PRNGKey(random_state)
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

        # Set reproducibility
        np.random.seed(random_state)

        print(f"JAX version: {jax.__version__}")
        print(f"JAX backend: {jax.default_backend()}")

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
        
        if len(available_cols) == 0:
            raise ValueError(f"No features found! Available columns: {df.columns.to_list()}")
        
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

    def create_batches(self, X, batch_size=1024, shuffle=True):
        """Create batches from data"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            yield X[indices[start_idx:end_idx]]

    def train_model(self, X_train, X_val, input_dim, epochs=50, batch_size=1024, lr=0.001):
        """Train the autoencoder model with early stopping"""
        print("\n" + "="*80)
        print("JAX MLP AUTOENCODER")
        print("="*80)

        # Track resources
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        print("Initializing model...")

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)

        # Initialize model
        self.model = MLPAutoencoderJAX(input_dim=input_dim, key=self.key)
        total_params = self.model.count_params()
        print(f"   Total parameters: {total_params:,}")

        # Optimizer with exponential decay schedule
        scheduler = optax.exponential_decay(
            init_value=lr,
            transition_steps=1000,
            decay_rate=0.95
        )
        optimizer = optax.adam(scheduler)
        opt_state = optimizer.init(self.model.params)

        # JIT compile the update function
        @jit
        def update(params, opt_state, batch):
            loss_value, grads = jax.value_and_grad(self.model.loss_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        print(f"\nTraining for up to {epochs} epochs (with early stopping)...")
        print(f"   Early stopping patience: 7 epochs")
        start_time = time.time()

        # FIXED: Early stopping variables
        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0
        train_losses = []
        val_losses = []
        best_params = None

        for epoch in range(epochs):  # FIXED: Changed from 5 to 50
            # Training phase
            train_loss = 0.0
            n_batches = 0
            for batch in self.create_batches(X_train_scaled, batch_size, shuffle=True):
                batch_jax = jnp.array(batch)
                self.model.params, opt_state, loss = update(self.model.params, opt_state, batch_jax)
                train_loss += float(loss)
                n_batches += 1

            train_loss /= n_batches
            train_losses.append(train_loss)

            # Validation phase
            val_loss = 0.0
            n_val_batches = 0
            for batch in self.create_batches(X_val_scaled, batch_size, shuffle=False):
                batch_jax = jnp.array(batch)
                loss = self.model.loss_fn(self.model.params, batch_jax)
                val_loss += float(loss)
                n_val_batches += 1

            val_loss /= n_val_batches
            val_losses.append(val_loss)

            # FIXED: Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best params (deep copy)
                best_params = {k: jnp.array(v) for k, v in self.model.params.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    # Restore best params
                    self.model.params = best_params
                    break

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        mem_used = mem_after - mem_before

        actual_epochs = len(train_losses)
        print(f"\n   Training stopped after {actual_epochs} epochs")
        print(f"   Training completed in {training_time:.2f}s")
        print(f"   Final train loss: {train_losses[-1]:.6f}")
        print(f"   Final val loss: {val_losses[-1]:.6f}")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used, train_losses, val_losses, X_train_scaled, X_val_scaled

    def calculate_reconstruction_errors(self, X_scaled):
        """Calculate reconstruction errors for anomaly detection"""
        # Process in batches for efficiency
        errors = []
        for i in range(0, len(X_scaled), 1024):
            batch = jnp.array(X_scaled[i:i+1024])
            pred = self.model.forward(self.model.params, batch)
            batch_errors = jnp.mean((batch - pred) ** 2, axis=1)
            errors.extend(batch_errors.tolist())

        return np.array(errors)

    def set_threshold(self, X_val_scaled):
        """Set anomaly threshold based on validation set"""
        print("\nCalculating anomaly threshold...")
        val_errors = self.calculate_reconstruction_errors(X_val_scaled)
        self.threshold = np.percentile(val_errors, (1 - self.contamination) * 100)
        print(f"   Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def evaluate_model(self, X_test_scaled):
        """Evaluate model on test set"""
        print(f"\nEvaluating JAX Autoencoder...")

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
            'model_name': 'JAX MLP Autoencoder',
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
        """Run complete JAX anomaly detection"""
        print("="*80)
        print("JAX ANOMALY DETECTION")
        print("="*80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = self.load_data(data_path, sample_size=sample_size)
        X, feature_names = self.prepare_features(df)
        X_train, X_val, X_test = self.split_data(X)

        # Train model (FIXED: now uses 50 epochs with early stopping)
        training_time, mem_used, train_losses, val_losses, X_train_scaled, X_val_scaled = self.train_model(
            X_train, X_val, input_dim=X.shape[1], epochs=50
        )

        # Set threshold from validation set
        self.set_threshold(X_val_scaled)

        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
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
            'jax_autoencoder': results,
            'configuration': {
                'contamination': self.contamination,
                'random_state': self.random_state,
                'max_epochs': 50,
                'actual_epochs': len(train_losses),
                'batch_size': 1024,
                'early_stopping_patience': 7,
                'jax_version': jax.__version__,
                'backend': jax.default_backend()
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
        output_file = f"{output_dir}/jax_anomaly_detection_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame({
            'jax_anomaly': predictions,
            'reconstruction_error': errors
        })
        predictions_file = f"{output_dir}/jax_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        print("\n" + "="*80)
        print("JAX ANALYSIS COMPLETE")
        print("="*80)

        return comparison


if __name__ == "__main__":
    # Run JAX anomaly detection on preprocessed data
    data_path = "../processed/processed_data.parquet"
    output_dir = "../results"

    # Use 10M sample
    detector = AnomalyDetectorJAX(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(data_path, output_dir, sample_size=10_000_000)

    print("\n[SUCCESS] JAX anomaly detection complete!")
    print("[FIXED] Now using 50 epochs with early stopping")
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
        output_file = f"{output_dir}/jax_anomaly_detection_results.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame(
            {"jax_anomaly": predictions, "reconstruction_error": errors}
        )
        predictions_file = f"{output_dir}/jax_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        print("\n" + "=" * 80)
        print("JAX ANALYSIS COMPLETE")
        print("=" * 80)

        return comparison


if __name__ == "__main__":
    # Run JAX anomaly detection on preprocessed data
    data_path = "../processed/processed_data.parquet"
    output_dir = "../results"

    # Use 10M sample
    detector = AnomalyDetectorJAX(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(
        data_path, output_dir, sample_size=10_000_000
    )

    print("\n[SUCCESS] JAX anomaly detection complete!")
    print("[FIXED] Now using 50 epochs with early stopping")
