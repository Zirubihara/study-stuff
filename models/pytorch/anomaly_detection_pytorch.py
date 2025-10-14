"""
Anomaly Detection using PyTorch
Implementation of MLP Autoencoder for anomaly detection
Based on data_science.md comparative modeling plan
"""

import polars as pl
import numpy as np
import time
import psutil
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MLPAutoencoder(nn.Module):
    """Multi-Layer Perceptron Autoencoder for anomaly detection"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        """
        Initialize autoencoder architecture

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (encoder)
        """
        super(MLPAutoencoder, self).__init__()

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(reversed_dims) - 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetectorPyTorch:
    """PyTorch-based anomaly detection using MLP Autoencoder"""

    def __init__(self, contamination=0.01, random_state=42, device=None):
        """
        Initialize PyTorch anomaly detector

        Args:
            contamination: Expected proportion of outliers (for threshold)
            random_state: Random seed for reproducibility
            device: Device to use (cuda/cpu)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        print(f"Using device: {self.device}")

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

    def create_data_loaders(self, X_train, X_val, X_test, batch_size=1024):
        """Create PyTorch data loaders"""
        print("\nCreating data loaders...")

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to tensors
        train_tensor = torch.FloatTensor(X_train_scaled)
        val_tensor = torch.FloatTensor(X_val_scaled)
        test_tensor = torch.FloatTensor(X_test_scaled)

        # Create datasets
        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        test_dataset = TensorDataset(test_tensor, test_tensor)

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader, X_test_scaled

    def train_model(self, train_loader, val_loader, input_dim, epochs=20, lr=0.001):
        """Train the autoencoder model"""
        print("\n" + "="*80)
        print("PYTORCH MLP AUTOENCODER")
        print("="*80)

        # Track resources
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        print("Initializing model...")
        self.model = MLPAutoencoder(input_dim=input_dim).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total parameters: {total_params:,}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"\nTraining for {epochs} epochs...")
        start_time = time.time()

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Could save model here if needed

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        training_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        mem_used = mem_after - mem_before

        print(f"\n   Training completed in {training_time:.2f}s")
        print(f"   Final train loss: {train_losses[-1]:.6f}")
        print(f"   Final val loss: {val_losses[-1]:.6f}")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Memory used: {mem_used:.2f} GB")

        return training_time, mem_used, train_losses, val_losses

    def calculate_reconstruction_errors(self, data_loader, X_scaled):
        """Calculate reconstruction errors for anomaly detection"""
        self.model.eval()
        reconstruction_errors = []

        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)

                # Calculate MSE for each sample
                errors = torch.mean((batch_X - outputs) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())

        return np.array(reconstruction_errors)

    def set_threshold(self, val_loader, X_val_scaled):
        """Set anomaly threshold based on validation set"""
        print("\nCalculating anomaly threshold...")
        val_errors = self.calculate_reconstruction_errors(val_loader, X_val_scaled)
        self.threshold = np.percentile(val_errors, (1 - self.contamination) * 100)
        print(f"   Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def evaluate_model(self, test_loader, X_test_scaled):
        """Evaluate model on test set"""
        print(f"\nEvaluating PyTorch Autoencoder...")

        # Calculate reconstruction errors
        start_time = time.time()
        test_errors = self.calculate_reconstruction_errors(test_loader, X_test_scaled)
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
            'model_name': 'PyTorch MLP Autoencoder',
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
        """Run complete PyTorch anomaly detection"""
        print("="*80)
        print("PYTORCH ANOMALY DETECTION")
        print("="*80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        df = self.load_data(data_path, sample_size=sample_size)
        X, feature_names = self.prepare_features(df)
        X_train, X_val, X_test = self.split_data(X)

        # Create data loaders
        train_loader, val_loader, test_loader, X_test_scaled = self.create_data_loaders(
            X_train, X_val, X_test
        )

        # Train model (5 epochs for 5M dataset to reduce time)
        training_time, mem_used, train_losses, val_losses = self.train_model(
            train_loader, val_loader, input_dim=X.shape[1], epochs=5
        )

        # Set threshold from validation set
        _, _, X_val_scaled = X_train, X_val, X_test
        X_val_scaled = self.scaler.transform(X_val)
        val_tensor = torch.FloatTensor(X_val_scaled)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        val_loader_threshold = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        self.set_threshold(val_loader_threshold, X_val_scaled)

        # Evaluate on test set
        results, predictions, errors = self.evaluate_model(test_loader, X_test_scaled)
        results['training_time'] = training_time
        results['memory_usage_gb'] = mem_used
        results['device'] = str(self.device)

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
            'pytorch_autoencoder': results,
            'configuration': {
                'contamination': self.contamination,
                'random_state': self.random_state,
                'device': str(self.device),
                'epochs': 10,
                'batch_size': 1024
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
        output_file = f"{output_dir}/pytorch_anomaly_detection_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Save predictions
        predictions_df = pl.DataFrame({
            'pytorch_anomaly': predictions,
            'reconstruction_error': errors
        })
        predictions_file = f"{output_dir}/pytorch_predictions.csv"
        predictions_df.write_csv(predictions_file)
        print(f"Predictions saved to: {predictions_file}")

        print("\n" + "="*80)
        print("PYTORCH ANALYSIS COMPLETE")
        print("="*80)

        return comparison


if __name__ == "__main__":
    # Run PyTorch anomaly detection on preprocessed data
    data_path = "../processed/processed_data.parquet"
    output_dir = "../results"

    # Use 5M sample for fair comparison across all frameworks
    detector = AnomalyDetectorPyTorch(contamination=0.01, random_state=42)
    results = detector.run_full_comparison(data_path, output_dir, sample_size=10_000_000)

    print("\n[SUCCESS] PyTorch anomaly detection complete!")
    print("[NEXT] Compare with scikit-learn results")
