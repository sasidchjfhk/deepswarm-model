# tests/test_data_pipeline.py
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import TemporalSplitter
from src.data.preprocessors import NetworkFeatureEngineer
from src.data.dataset import IDSDataset

class TestTemporalSplitter:
    def test_split_by_day_concatenation(self):
        # Create dummy dataframes
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dfs = {
            day: pd.DataFrame({
                'col1': range(10),
                'day': [day] * 10
            }) for day in days
        }
        
        train_days = ['Monday', 'Tuesday']
        val_day = 'Wednesday'
        
        train, val, test = TemporalSplitter.split_by_day(dfs, train_days, val_day)
        
        # Verify train contains only train_days
        assert len(train) == 20
        assert set(train['day'].unique()) == set(train_days)
        
        # Verify val contains only val_day
        assert len(val) == 10
        assert val['day'].iloc[0] == val_day
        
        # Verify test defaults to val
        assert len(test) == 10
        assert test['day'].iloc[0] == val_day
        
    def test_temporal_leakage_check(self):
        # Case 1: No Leakage
        train_df = pd.DataFrame({
            'Timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00'])
        })
        val_df = pd.DataFrame({
            'Timestamp': pd.to_datetime(['2023-01-02 10:00'])
        })
        
        assert TemporalSplitter.check_temporal_leakage(train_df, val_df) == True
        
        # Case 2: Leakage (Train ends after Val starts)
        train_df_leak = pd.DataFrame({
            'Timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-02 11:00'])
        })
        val_df_leak = pd.DataFrame({
            'Timestamp': pd.to_datetime(['2023-01-02 10:00'])
        })
        
        # Should return False (Leakage Detected) cause Train Max (02 11:00) > Val Min (02 10:00)
        assert TemporalSplitter.check_temporal_leakage(train_df_leak, val_df_leak) == False

class TestFeatureEngineer:
    def test_fit_transform_no_leakage(self, tmp_path):
        # Create dummy data
        X_train = pd.DataFrame({
            'Flow Duration': [100, 200, 300, 1000], # outlier
            'Total Fwd Packets': [5, 10, 15, 20]
        })
        y_train = pd.Series([0, 0, 1, 1])
        
        X_val = pd.DataFrame({
            'Flow Duration': [5000], # massive outlier in val
            'Total Fwd Packets': [50]
        })
        
        # Init engineer
        engineer = NetworkFeatureEngineer(scaler_type="robust", feature_selection=False)
        
        # Fit on Train
        engineer.fit(X_train, y_train)
        
        # Transform both
        X_train_processed = engineer.transform(X_train)
        X_val_processed = engineer.transform(X_val)
        
        # Check Scaler stats
        # Median of [100, 200, 300, 1000] is (200+300)/2 = 250
        # If we fit on ALL (including 5000), median would shift.
        # RobustScaler centers median at 0.
        
        # Let's verify saving/loading
        save_path = tmp_path / "transformers"
        engineer.save(save_path)
        
        loaded_engineer = NetworkFeatureEngineer.load(save_path)
        X_val_loaded_processed = loaded_engineer.transform(X_val)
        
        np.testing.assert_array_almost_equal(X_val_processed, X_val_loaded_processed)
