# src/data/preprocessors.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
from pathlib import Path
from loguru import logger

class NetworkFeatureEngineer:
    """
    Feature engineering for network intrusion detection.
    
    Principles:
    - Fit on train, transform on val/test (NO LEAKAGE)
    - Save all transformers for production
    - Versioned feature schemas
    """
    
    # Critical features identified through domain knowledge + feature importance
    CORE_FEATURES = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Flow IAT Mean',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'Fwd PSH Flags',
        'Bwd PSH Flags',
        'Fwd URG Flags',
        'Bwd URG Flags',
        'SYN Flag Count',
        'RST Flag Count',
        'PSH Flag Count',
        'ACK Flag Count',
        'URG Flag Count',
        'Average Packet Size',
        'Avg Fwd Segment Size',
        'Avg Bwd Segment Size',
    ]
    
    def __init__(
        self,
        scaler_type: str = "robust",
        feature_selection: bool = False, # Default OFF for XGBoost stability
        k_best_features: int = 50
    ):
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.k_best_features = k_best_features
        
        # Transformers (fitted on training data only)
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = {}
        
        # Feature metadata
        self.feature_names = None
        self.selected_features = None
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> 'NetworkFeatureEngineer':
        """
        Fit all transformers on training data.
        
        CRITICAL: Only call this ONCE on training data!
        """
        
        logger.info("Fitting feature engineering pipeline...")
        
        # 1. Select features
        X_train = self._select_core_features(X_train)
        
        # 2. Engineer new features
        X_train = self._engineer_features(X_train)
        
        # 3. Fit scaler
        self.scaler = self._create_scaler()
        self.scaler.fit(X_train)
        logger.info(f"✓ Fitted {self.scaler_type} scaler")
        
        # 4. Fit feature selector (optional)
        if self.feature_selection:
            logger.info("Fitting feature selector (this may take a while)...")
            X_scaled = self.scaler.transform(X_train)
            
            # Subsample for feature selection if data is too large to speed up dev
            if len(X_scaled) > 100000:
                 logger.info("Subsampling for feature selection speed...")
                 idx = np.random.choice(len(X_scaled), 50000, replace=False)
                 X_for_sel = X_scaled[idx]
                 # Handle both pandas Series and numpy arrays
                 if isinstance(y_train, pd.Series):
                     y_for_sel = y_train.iloc[idx]
                 else:
                     y_for_sel = y_train[idx]
            else:
                 X_for_sel = X_scaled
                 y_for_sel = y_train
                 
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.k_best_features, X_scaled.shape[1])
            )
            self.feature_selector.fit(X_for_sel, y_for_sel)
            
            # Store selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X_train.columns[selected_mask].tolist()
            logger.info(f"✓ Selected {len(self.selected_features)} features")
        else:
            self.selected_features = X_train.columns.tolist()
            
        self.feature_names = X_train.columns.tolist()
        
        return self
        
    def transform(self, X: pd.DataFrame, return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted transformers.
        
        Args:
            X: Input dataframe
            return_df: If True, returns pandas DataFrame with column names (Good for XGBoost)
        """
        
        if self.scaler is None:
            raise ValueError("Must call fit() before transform()")
            
        # 1. Select features (same as training)
        X = self._select_core_features(X)
        
        # 2. Engineer features (same as training)
        X = self._engineer_features(X)
        
        # 3. Scale
        X_scaled = self.scaler.transform(X)
        
        # 4. Feature selection
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
            
        if return_df:
            return pd.DataFrame(X_scaled, columns=self.selected_features)
            
        return X_scaled
        
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        return_df: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform training data."""
        self.fit(X_train, y_train)
        return self.transform(X_train, return_df=return_df)
        
    def _select_core_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select core features, handling missing columns."""
        
        available_features = [f for f in self.CORE_FEATURES if f in X.columns]
        
        if len(available_features) < len(self.CORE_FEATURES):
            missing = set(self.CORE_FEATURES) - set(available_features)
            # logger.warning(f"Missing features: {missing}")
            
        return X[available_features].copy()
        
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        
        X = X.copy()
        
        # Ratio features (avoid division by zero)
        epsilon = 1e-9
        
        if 'Total Fwd Packets' in X.columns and 'Total Backward Packets' in X.columns:
            X['fwd_bwd_packet_ratio'] = (
                X['Total Fwd Packets'] / (X['Total Backward Packets'] + epsilon)
            )
            
        if 'Total Length of Fwd Packets' in X.columns and 'Total Length of Bwd Packets' in X.columns:
            X['fwd_bwd_length_ratio'] = (
                X['Total Length of Fwd Packets'] / 
                (X['Total Length of Bwd Packets'] + epsilon)
            )
        
        # Normalized features
        if 'Average Packet Size' in X.columns and 'Flow Duration' in X.columns:
            X['avg_packet_size_normalized'] = (
                X['Average Packet Size'] / (X['Flow Duration'] + epsilon)
            )
        
        # Flag ratios
        if 'Total Fwd Packets' in X.columns and 'Total Backward Packets' in X.columns:
            total_packets = X['Total Fwd Packets'] + X['Total Backward Packets'] + epsilon
            
            if 'SYN Flag Count' in X.columns:
                X['syn_ratio'] = X['SYN Flag Count'] / total_packets
            if 'RST Flag Count' in X.columns:
                X['rst_ratio'] = X['RST Flag Count'] / total_packets
            if 'ACK Flag Count' in X.columns:
                X['ack_ratio'] = X['ACK Flag Count'] / total_packets
        
        # Handle any infinities from division
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        return X
        
    def _create_scaler(self):
        """Create appropriate scaler."""
        
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            # Robust to outliers (better for network data)
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
            
    def save(self, path: Path):
        """Save all transformers for production."""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, path / "scaler.pkl")
            
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, path / "feature_selector.pkl")
            
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "scaler_type": self.scaler_type,
            "k_best_features": self.k_best_features
        }
        joblib.dump(metadata, path / "metadata.pkl")
        
        logger.info(f"✓ Saved feature engineering pipeline to {path}")
        
    @classmethod
    def load(cls, path: Path) -> 'NetworkFeatureEngineer':
        """Load saved transformers."""
        
        path = Path(path)
        
        # Load metadata
        metadata = joblib.load(path / "metadata.pkl")
        
        # Create instance
        instance = cls(
            scaler_type=metadata["scaler_type"],
            k_best_features=metadata["k_best_features"]
        )
        
        # Load transformers
        instance.scaler = joblib.load(path / "scaler.pkl")
        
        if (path / "feature_selector.pkl").exists():
            instance.feature_selector = joblib.load(path / "feature_selector.pkl")
            
        instance.feature_names = metadata["feature_names"]
        instance.selected_features = metadata["selected_features"]
        
        logger.info(f"✓ Loaded feature engineering pipeline from {path}")
        
        return instance


class LabelEncoder:
    """
    Encode attack labels to integers.
    """
    
    # CICIDS2017 attack types
    # CICIDS2017 attack types (Canonical + Variations)
    ATTACK_TYPES = {
        'BENIGN': 0,
        # DoS
        'DoS Hulk': 1,
        'DoS GoldenEye': 2,
        'DoS slowloris': 3,
        'DoS Slowhttptest': 4,
        # PortScan
        'PortScan': 5,
        'Portscan': 5,  # Variation
        # DDoS
        'DDoS': 6,
        # Brute Force
        'FTP-Patator': 7,
        'FTP-Patator - Attempted': 7,
        'SSH-Patator': 8,
        'SSH-Patator - Attempted': 8,
        # Botnet
        'Bot': 9,
        'Botnet': 9,
        'Botnet - Attempted': 9,
        # Web Attacks
        'Web Attack – Brute Force': 10,
        'Web Attack - Brute Force': 10,
        'Web Attack - Brute Force - Attempted': 10,
        'Web Attack – XSS': 11,
        'Web Attack - XSS': 11,
        'Web Attack - XSS - Attempted': 11,
        'Web Attack – Sql Injection': 12,
        'Web Attack - SQL Injection': 12,
        'Web Attack - SQL Injection - Attempted': 12,
        # Infiltration
        'Infiltration': 13,
        'Infiltration - Attempted': 13,
        'Infiltration - Portscan': 13,  # Map to Infiltration? Or PortScan? Let's treat as Infiltration logic
        'Heartbleed': 14,
    }
    
    def __init__(self):
        self.label_to_idx = self.ATTACK_TYPES.copy()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def encode(self, labels: pd.Series) -> np.ndarray:
        """
        Convert string labels to integers and ensure they are contiguous 
        (0, 1, 2...) for XGBoost compatibility.
        """
        
        # Clean labels
        labels = labels.str.strip()
        
        # 1. Map to raw indices using big dictionary
        raw_encoded = labels.map(self.label_to_idx)
        
        # Check for unknown labels
        unknown_mask = raw_encoded.isna()
        if unknown_mask.any():
            unknown_labels = labels[unknown_mask].unique()
            logger.warning(f"Unknown labels found: {unknown_labels}")
            raw_encoded = raw_encoded.fillna(self.label_to_idx['BENIGN'])
            
        # 2. Force contiguity (0, 1, 2... N)
        # This is vital for XGBoost 2.0+
        unique_raw = np.sort(raw_encoded.unique())
        self.relabel_map = {old: new for new, old in enumerate(unique_raw)}
        
        # Update idx_to_label to reflect the NEW contiguous indices
        # We take the first string label found for each new index
        new_idx_to_label = {}
        for old_idx, new_idx in self.relabel_map.items():
            original_label = self.idx_to_label.get(old_idx, "UNKNOWN")
            new_idx_to_label[new_idx] = original_label
        
        self.idx_to_label = new_idx_to_label
        
        # Apply the final mapping
        final_encoded = raw_encoded.map(self.relabel_map)
        
        return final_encoded.values.astype(int)
        
    def decode(self, indices: np.ndarray) -> List[str]:
        """Convert integer indices to string labels."""
        return [self.idx_to_label.get(idx, "UNKNOWN") for idx in indices]
        
    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)
