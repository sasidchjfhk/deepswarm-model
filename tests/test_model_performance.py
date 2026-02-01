# tests/test_model_performance.py
"""
Performance and load testing for XGBoost model in production.
"""

import pytest
import pandas as pd
import numpy as np
import time
import threading
import concurrent.futures
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.xgboost_model import SwarmXGBoost
from src.data.preprocessors import NetworkFeatureEngineer, LabelEncoder


class TestModelPerformance:
    """Production performance testing."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for performance testing."""
        np.random.seed(42)
        
        # Create larger dataset for performance testing
        n_samples = 5000
        data = {
            'Flow Duration': np.random.exponential(1000, n_samples),
            'Total Fwd Packets': np.random.poisson(10, n_samples),
            'Total Backward Packets': np.random.poisson(8, n_samples),
            'Fwd Packet Length Mean': np.random.normal(100, 50, n_samples),
            'Bwd Packet Length Mean': np.random.normal(80, 40, n_samples),
            'Flow Bytes/s': np.random.exponential(10000, n_samples),
            'Flow Packets/s': np.random.exponential(100, n_samples),
            'Label': np.random.choice(['Benign', 'Attack'], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        X = df.drop(columns=['Label'])
        y = df['Label']
        
        # Preprocess
        feature_engineer = NetworkFeatureEngineer(feature_selection=True, k_best_features=10)
        label_encoder = LabelEncoder()
        
        y_encoded = label_encoder.encode(y)
        X_processed = feature_engineer.fit_transform(X, y_encoded)
        
        feature_names = feature_engineer.selected_features
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Train model
        model = SwarmXGBoost(num_classes=2, params={'base_score': 0.5})
        model.fit(X_df, y_encoded, verbose=False)
        
        return model, feature_engineer
    
    def test_batch_prediction_performance(self, trained_model):
        """Test batch prediction performance."""
        model, feature_engineer = trained_model
        
        # Create test data
        np.random.seed(42)
        batch_sizes = [1, 10, 100, 1000, 5000]
        
        for batch_size in batch_sizes:
            test_data = pd.DataFrame({
                'Flow Duration': np.random.exponential(1000, batch_size),
                'Total Fwd Packets': np.random.poisson(10, batch_size),
                'Total Backward Packets': np.random.poisson(8, batch_size),
                'Fwd Packet Length Mean': np.random.normal(100, 50, batch_size),
                'Bwd Packet Length Mean': np.random.normal(80, 40, batch_size),
                'Flow Bytes/s': np.random.exponential(10000, batch_size),
                'Flow Packets/s': np.random.exponential(100, batch_size),
            })
            
            # Transform and predict
            X_processed = feature_engineer.transform(test_data)
            X_df = pd.DataFrame(X_processed, columns=feature_engineer.selected_features)
            
            # Measure prediction time
            start_time = time.time()
            predictions = model.predict(X_df)
            prediction_time = time.time() - start_time
            
            # Performance assertions
            assert len(predictions) == batch_size
            
            # Throughput requirements (samples per second)
            throughput = batch_size / prediction_time
            print(f"Batch size {batch_size}: {throughput:.0f} samples/sec")
            
            # Should handle at least 1000 samples/sec for production
            if batch_size >= 100:
                assert throughput >= 1000, f"Throughput too low: {throughput:.0f} samples/sec"
    
    def test_concurrent_predictions(self, trained_model):
        """Test model performance under concurrent load."""
        model, feature_engineer = trained_model
        
        def predict_worker():
            """Worker function for concurrent prediction."""
            np.random.seed(threading.get_ident())
            
            # Create test data
            test_data = pd.DataFrame({
                'Flow Duration': np.random.exponential(1000, 100),
                'Total Fwd Packets': np.random.poisson(10, 100),
                'Total Backward Packets': np.random.poisson(8, 100),
                'Fwd Packet Length Mean': np.random.normal(100, 50, 100),
                'Bwd Packet Length Mean': np.random.normal(80, 40, 100),
                'Flow Bytes/s': np.random.exponential(10000, 100),
                'Flow Packets/s': np.random.exponential(100, 100),
            })
            
            X_processed = feature_engineer.transform(test_data)
            X_df = pd.DataFrame(X_processed, columns=feature_engineer.selected_features)
            
            return model.predict(X_df)
        
        # Test with multiple concurrent threads
        num_threads = 4
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(predict_worker) for _ in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Verify all predictions completed
        assert len(results) == num_threads
        for result in results:
            assert len(result) == 100
        
        # Should handle concurrent requests efficiently
        total_samples = num_threads * 100
        throughput = total_samples / total_time
        print(f"Concurrent prediction: {throughput:.0f} samples/sec with {num_threads} threads")
        
        assert throughput >= 500, f"Concurrent throughput too low: {throughput:.0f} samples/sec"
    
    def test_memory_usage(self, trained_model):
        """Test memory usage during prediction."""
        import psutil
        import os
        
        model, feature_engineer = trained_model
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Large batch prediction
        np.random.seed(42)
        large_batch = pd.DataFrame({
            'Flow Duration': np.random.exponential(1000, 10000),
            'Total Fwd Packets': np.random.poisson(10, 10000),
            'Total Backward Packets': np.random.poisson(8, 10000),
            'Fwd Packet Length Mean': np.random.normal(100, 50, 10000),
            'Bwd Packet Length Mean': np.random.normal(80, 40, 10000),
            'Flow Bytes/s': np.random.exponential(10000, 10000),
            'Flow Packets/s': np.random.exponential(100, 10000),
        })
        
        X_processed = feature_engineer.transform(large_batch)
        X_df = pd.DataFrame(X_processed, columns=feature_engineer.selected_features)
        
        # Predict and measure memory
        predictions = model.predict(X_df)
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = peak_memory - baseline_memory
        
        # Memory assertions
        assert len(predictions) == 10000
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.1f} MB"
        
        print(f"Memory usage: {memory_increase:.1f} MB increase for 10k samples")
    
    def test_model_loading_performance(self, trained_model, tmp_path):
        """Test model loading and saving performance."""
        model, feature_engineer = trained_model
        
        # Save model
        model_dir = tmp_path / "perf_test"
        save_start = time.time()
        model.save(str(model_dir))
        save_time = time.time() - save_start
        
        # Load model multiple times
        load_times = []
        for _ in range(5):
            load_start = time.time()
            import joblib
            loaded_model = joblib.load(model_dir / "swarm_xgboost_v1.pkl")
            load_time = time.time() - load_start
            load_times.append(load_time)
        
        avg_load_time = np.mean(load_times)
        
        # Performance assertions
        assert save_time < 5.0, f"Model saving too slow: {save_time:.2f}s"
        assert avg_load_time < 1.0, f"Model loading too slow: {avg_load_time:.2f}s"
        
        print(f"Save time: {save_time:.2f}s, Avg load time: {avg_load_time:.2f}s")
    
    def test_prediction_consistency(self, trained_model):
        """Test prediction consistency across multiple runs."""
        model, feature_engineer = trained_model
        
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'Flow Duration': np.random.exponential(1000, 100),
            'Total Fwd Packets': np.random.poisson(10, 100),
            'Total Backward Packets': np.random.poisson(8, 100),
            'Fwd Packet Length Mean': np.random.normal(100, 50, 100),
            'Bwd Packet Length Mean': np.random.normal(80, 40, 100),
            'Flow Bytes/s': np.random.exponential(10000, 100),
            'Flow Packets/s': np.random.exponential(100, 100),
        })
        
        X_processed = feature_engineer.transform(test_data)
        X_df = pd.DataFrame(X_processed, columns=feature_engineer.selected_features)
        
        # Multiple predictions
        predictions_1 = model.predict(X_df)
        predictions_2 = model.predict(X_df)
        predictions_3 = model.predict(X_df)
        
        # Should be identical
        np.testing.assert_array_equal(predictions_1, predictions_2)
        np.testing.assert_array_equal(predictions_2, predictions_3)
        
        # Test probabilities consistency
        probs_1 = model.predict_proba(X_df)
        probs_2 = model.predict_proba(X_df)
        
        np.testing.assert_array_almost_equal(probs_1, probs_2, decimal=10)
    
    def test_scalability_limits(self, trained_model):
        """Test model behavior at scalability limits."""
        model, feature_engineer = trained_model
        
        # Test very large batch
        np.random.seed(42)
        very_large_batch = pd.DataFrame({
            'Flow Duration': np.random.exponential(1000, 50000),
            'Total Fwd Packets': np.random.poisson(10, 50000),
            'Total Backward Packets': np.random.poisson(8, 50000),
            'Fwd Packet Length Mean': np.random.normal(100, 50, 50000),
            'Bwd Packet Length Mean': np.random.normal(80, 40, 50000),
            'Flow Bytes/s': np.random.exponential(10000, 50000),
            'Flow Packets/s': np.random.exponential(100, 50000),
        })
        
        X_processed = feature_engineer.transform(very_large_batch)
        X_df = pd.DataFrame(X_processed, columns=feature_engineer.selected_features)
        
        # Should handle large batches without memory issues
        start_time = time.time()
        predictions = model.predict(X_df)
        prediction_time = time.time() - start_time
        
        assert len(predictions) == 50000
        assert prediction_time < 30.0, f"Large batch prediction too slow: {prediction_time:.2f}s"
        
        throughput = 50000 / prediction_time
        print(f"Large batch (50k): {throughput:.0f} samples/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
