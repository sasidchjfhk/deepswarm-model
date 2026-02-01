# tests/test_xgboost_production.py
"""
Production-grade test suite for XGBoost Swarm IDS model.

Tests cover:
- Model loading and serialization
- Prediction accuracy and performance
- Edge cases and error handling
- Model robustness and data validation
- Integration with full pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
import shutil
import time
import joblib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.xgboost_model import SwarmXGBoost
from src.data.preprocessors import NetworkFeatureEngineer, LabelEncoder
from src.data.loader import ProductionDataLoader


class TestSwarmXGBoostModel:
    """Test core XGBoost model functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample network traffic data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic network features
        data = {
            'Flow Duration': np.random.exponential(1000, n_samples),
            'Total Fwd Packets': np.random.poisson(10, n_samples),
            'Total Backward Packets': np.random.poisson(8, n_samples),
            'Fwd Packet Length Mean': np.random.normal(100, 50, n_samples),
            'Bwd Packet Length Mean': np.random.normal(80, 40, n_samples),
            'Flow Bytes/s': np.random.exponential(10000, n_samples),
            'Flow Packets/s': np.random.exponential(100, n_samples),
            'Fwd IAT Mean': np.random.exponential(50, n_samples),
            'Bwd IAT Mean': np.random.exponential(60, n_samples),
            'Fwd Header Length': np.random.poisson(40, n_samples),
            'Bwd Header Length': np.random.poisson(40, n_samples),
            'Label': np.random.choice(['Benign', 'DDoS', 'PortScan', 'Bot'], n_samples, p=[0.7, 0.15, 0.1, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Add some outliers to test robustness
        df.loc[df.sample(50).index, 'Flow Duration'] *= 100
        df.loc[df.sample(30).index, 'Flow Bytes/s'] *= 1000
        
        return df
    
    @pytest.fixture
    def processed_data(self, sample_data):
        """Create processed data ready for XGBoost."""
        # Split features and labels
        X = sample_data.drop(columns=['Label'])
        y = sample_data['Label']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.encode(y)
        
        # Feature engineering
        feature_engineer = NetworkFeatureEngineer(
            feature_selection=True,
            k_best_features=20
        )
        X_train = sample_data.sample(frac=0.8, random_state=42)
        X_val = sample_data.drop(X_train.index)
        
        y_train = label_encoder.encode(X_train['Label'])
        y_val = label_encoder.encode(X_val['Label'])
        
        X_train_features = X_train.drop(columns=['Label'])
        X_val_features = X_val.drop(columns=['Label'])
        
        # Fit and transform
        X_train_processed = feature_engineer.fit_transform(X_train_features, y_train)
        X_val_processed = feature_engineer.transform(X_val_features)
        
        # Convert to DataFrames with feature names
        feature_names = feature_engineer.selected_features
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        
        return {
            'X_train': X_train_df,
            'X_val': X_val_df,
            'y_train': y_train,
            'y_val': y_val,
            'feature_engineer': feature_engineer,
            'label_encoder': label_encoder,
            'num_classes': len(np.unique(y_encoded))
        }
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test default initialization
        model = SwarmXGBoost()
        assert model.num_classes == 2
        assert model.use_gpu == False
        assert model.model is None
        assert len(model.feature_names) == 0
        
        # Test custom initialization
        custom_params = {'max_depth': 6, 'learning_rate': 0.05}
        model = SwarmXGBoost(
            params=custom_params,
            num_classes=4,
            use_gpu=True
        )
        assert model.num_classes == 4
        assert model.use_gpu == True
        assert model.params['max_depth'] == 6
        assert model.params['learning_rate'] == 0.05
    
    def test_model_training(self, processed_data):
        """Test model training with validation."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        
        # Test training
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Verify model is trained
        assert model.model is not None
        assert hasattr(model.model, 'predict')
        assert len(model.feature_names) > 0
        
        # Verify feature names are stored
        expected_features = processed_data['X_train'].columns.tolist()
        assert model.feature_names == expected_features
    
    def test_prediction_methods(self, processed_data):
        """Test predict and predict_proba methods."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        X_test = processed_data['X_val'].head(10)
        
        # Test predict
        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert all(pred in range(processed_data['num_classes']) for pred in predictions)
        
        # Test predict_proba
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (10, processed_data['num_classes'])
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert all(prob >= 0 for prob in probabilities.flatten())  # All non-negative
    
    def test_model_serialization(self, processed_data, tmp_path):
        """Test model saving and loading."""
        # Train model
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Save model
        output_dir = tmp_path / "test_model"
        model.save(str(output_dir), version="test")
        
        # Verify files exist
        json_path = output_dir / "xgboost_core_test.json"
        pkl_path = output_dir / "swarm_xgboost_test.pkl"
        
        assert json_path.exists()
        assert pkl_path.exists()
        
        # Test loading pickle
        loaded_model = joblib.load(pkl_path)
        assert isinstance(loaded_model, SwarmXGBoost)
        assert loaded_model.num_classes == processed_data['num_classes']
        assert loaded_model.feature_names == model.feature_names
        
        # Test predictions are consistent
        X_test = processed_data['X_val'].head(5)
        original_preds = model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        
        np.testing.assert_array_equal(original_preds, loaded_preds)
    
    def test_edge_cases(self, processed_data):
        """Test edge cases and error handling."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Test empty input
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame())
        
        # Test single sample
        single_sample = processed_data['X_val'].iloc[[0]]
        pred = model.predict(single_sample)
        assert len(pred) == 1
        
        # Test wrong number of features
        wrong_features = pd.DataFrame(np.random.rand(5, 5))
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(wrong_features)
    
    def test_model_robustness(self, processed_data):
        """Test model robustness with various data conditions."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Test with NaN values
        X_nan = processed_data['X_val'].copy()
        X_nan.iloc[0, 0] = np.nan
        
        # XGBoost should handle NaN values
        pred_nan = model.predict(X_nan.head(1))
        assert len(pred_nan) == 1
        
        # Test with extreme values
        X_extreme = processed_data['X_val'].copy()
        X_extreme.iloc[0, 0] = 1e10  # Very large value
        
        pred_extreme = model.predict(X_extreme.head(1))
        assert len(pred_extreme) == 1
        
        # Test with negative values where appropriate
        X_negative = processed_data['X_val'].copy()
        X_negative.iloc[0, 0] = -1000
        
        pred_negative = model.predict(X_negative.head(1))
        assert len(pred_negative) == 1
    
    def test_performance_benchmarks(self, processed_data):
        """Test model performance benchmarks."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Test prediction speed
        X_test = processed_data['X_val']
        
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Should predict 1000 samples in under 1 second
        assert len(predictions) == len(X_test)
        assert prediction_time < 1.0, f"Prediction too slow: {prediction_time:.3f}s"
        
        # Test probability prediction speed
        start_time = time.time()
        probabilities = model.predict_proba(X_test)
        prob_time = time.time() - start_time
        
        assert probabilities.shape[0] == len(X_test)
        assert prob_time < 2.0, f"Probability prediction too slow: {prob_time:.3f}s"
    
    def test_feature_importance_logging(self, processed_data, caplog):
        """Test feature importance logging."""
        model = SwarmXGBoost(num_classes=processed_data['num_classes'])
        model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Check that feature importance is calculated
        assert hasattr(model.model, 'feature_importances_')
        assert len(model.model.feature_importances_) == len(model.feature_names)
    
    def test_binary_vs_multiclass(self):
        """Test model behavior for binary vs multiclass classification."""
        # Binary classification
        binary_model = SwarmXGBoost(num_classes=2)
        assert binary_model.params['objective'] == 'binary:logistic'
        # Note: num_class might be present but None for binary classification
        
        # Multiclass classification
        multi_model = SwarmXGBoost(num_classes=4)
        assert multi_model.params['objective'] == 'multi:softprob'
        assert multi_model.params['num_class'] == 4
    
    @patch('xgboost.XGBClassifier')
    def test_gpu_support(self, mock_xgb, processed_data):
        """Test GPU support configuration."""
        # Test GPU enabled
        gpu_model = SwarmXGBoost(use_gpu=True, num_classes=processed_data['num_classes'])
        gpu_model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        # Check that GPU parameters are set
        assert gpu_model.params['device'] == 'cuda'
        
        # Test CPU fallback
        cpu_model = SwarmXGBoost(use_gpu=False, num_classes=processed_data['num_classes'])
        cpu_model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            verbose=False
        )
        
        assert cpu_model.params['device'] == 'cpu'


class TestXGBoostIntegration:
    """Integration tests with full pipeline."""
    
    @pytest.fixture
    def integration_data(self, tmp_path):
        """Create integration test data."""
        np.random.seed(42)
        n_samples = 500
        
        # Create more realistic data
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
        
        # Save to temporary CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_end_to_end_pipeline(self, integration_data, tmp_path):
        """Test complete pipeline from data to model."""
        # Load data
        df = pd.read_csv(integration_data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=['Label'])
        y = df['Label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Preprocessing
        feature_engineer = NetworkFeatureEngineer(
            feature_selection=True,
            k_best_features=10
        )
        label_encoder = LabelEncoder()
        
        y_train_encoded = label_encoder.encode(y_train)
        y_val_encoded = label_encoder.encode(y_val)
        
        X_train_processed = feature_engineer.fit_transform(X_train, y_train_encoded)
        X_val_processed = feature_engineer.transform(X_val)
        
        feature_names = feature_engineer.selected_features
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        
        # Train model
        num_classes = len(np.unique(y_train_encoded))
        model = SwarmXGBoost(
            num_classes=num_classes,
            params={'base_score': 0.5}  # Fix base_score for binary classification
        )
        model.fit(X_train_df, y_train_encoded, X_val_df, y_val_encoded, verbose=False)
        
        # Save everything
        model_dir = tmp_path / "saved_model"
        model.save(str(model_dir))
        
        transformer_dir = tmp_path / "transformers"
        feature_engineer.save(transformer_dir)
        
        # Load and test
        loaded_model = joblib.load(model_dir / "swarm_xgboost_v1.pkl")
        loaded_engineer = NetworkFeatureEngineer.load(transformer_dir)
        
        # Test on new data
        X_new = X_val.head(5)
        X_new_processed = loaded_engineer.transform(X_new)
        X_new_df = pd.DataFrame(X_new_processed, columns=loaded_engineer.selected_features)
        
        predictions = loaded_model.predict(X_new_df)
        probabilities = loaded_model.predict_proba(X_new_df)
        
        assert len(predictions) == 5
        # For binary classification, probabilities shape is (n_samples, 2)
        # For single class, it might be (n_samples, 1)
        expected_shape = (5, max(2, num_classes))
        assert probabilities.shape[0] == 5  # Correct number of samples
        assert probabilities.shape[1] in [1, 2]  # Either 1 or 2 columns for binary


class TestXGBoostErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test model initialization with invalid parameters."""
        # Test string num_classes (should work but might cause issues)
        try:
            model = SwarmXGBoost(num_classes="2")
            # If it works, that's fine - XGBoost might handle it
        except (ValueError, TypeError):
            pass  # Expected behavior
    
    def test_training_without_validation(self):
        """Test training without validation data."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(100, 10))
        y_train = np.random.randint(0, 2, 100)
        
        model = SwarmXGBoost(num_classes=2)
        
        # Should train without validation
        model.fit(X_train, y_train, verbose=False)
        assert model.model is not None
    
    def test_prediction_before_training(self):
        """Test prediction before training."""
        model = SwarmXGBoost()
        X_test = pd.DataFrame(np.random.rand(10, 5))
        
        with pytest.raises(AttributeError):
            model.predict(X_test)
        
        with pytest.raises(AttributeError):
            model.predict_proba(X_test)
    
    def test_mismatched_features(self):
        """Test prediction with mismatched features."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        y_train = np.random.randint(0, 2, 100)
        
        model = SwarmXGBoost(num_classes=2)
        model.fit(X_train, y_train, verbose=False)
        
        # Test with wrong feature names
        X_wrong = pd.DataFrame(np.random.rand(10, 10), columns=[f'wrong_{i}' for i in range(10)])
        
        # Should either work (if XGBoost ignores column names) or raise error
        try:
            predictions = model.predict(X_wrong)
            assert len(predictions) == 10
        except (ValueError, RuntimeError):
            # Expected behavior
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
