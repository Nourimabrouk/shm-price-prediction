"""Feature engineering module for SHM Heavy Equipment Price Prediction.

This module provides leak-proof feature engineering with:
- Strict temporal validation (no future information leakage)
- Robust categorical encoding
- Feature selection and importance analysis
- Comprehensive feature documentation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json

# Configure logging for Windows compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class LeakProofFeatureEngineer:
    """Feature engineer that prevents data leakage and maintains temporal validity."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.engineered_features = []
        
        # Features that are definitively leakage (target or derived from target)
        self.forbidden_features = [
            'Sales Price',  # This is the target
            'log1p_price',  # Derived from target
            'price_log',    # Derived from target
            'log_price',    # Derived from target
            'target',       # Alternative target name
            'price'         # Alternative target name
        ]
        
        # Temporal features that could cause leakage if not handled carefully
        self.temporal_risk_features = [
            'Sales date',
            'date',
            'sale_date',
            'timestamp'
        ]
    
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify and categorize feature types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping feature types to column names
        """
        feature_types = {
            'forbidden': [],
            'temporal': [],
            'categorical': [],
            'numerical': [],
            'high_cardinality': [],
            'low_variance': []
        }
        
        for col in df.columns:
            # Check for forbidden features
            if any(forbidden in col.lower() for forbidden in ['price', 'target', 'log1p']):
                feature_types['forbidden'].append(col)
                continue
            
            # Check for temporal features
            if any(temp in col.lower() for temp in ['date', 'time']):
                feature_types['temporal'].append(col)
                continue
            
            # Analyze column characteristics
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                if unique_count > 50:  # High cardinality categorical
                    feature_types['high_cardinality'].append(col)
                else:
                    feature_types['categorical'].append(col)
            else:
                # Numerical feature
                if df[col].var() < 1e-6:  # Very low variance
                    feature_types['low_variance'].append(col)
                else:
                    feature_types['numerical'].append(col)
        
        return feature_types
    
    def remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features that could cause data leakage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with leakage features removed
        """
        df_clean = df.copy()
        removed_features = []
        
        for col in df.columns:
            # Remove forbidden features
            if any(forbidden in col.lower() for forbidden in 
                   ['price', 'target', 'log1p', 'log_price', 'price_log']):
                if col in df_clean.columns:
                    df_clean = df_clean.drop(columns=[col])
                    removed_features.append(col)
        
        if removed_features:
            logger.warning(f"Removed potential leakage features: {removed_features}")
        
        return df_clean
    
    def engineer_temporal_features(self, df: pd.DataFrame, 
                                 date_column: str = 'Sales date') -> pd.DataFrame:
        """Engineer temporal features without causing leakage.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with temporal features
        """
        df_temporal = df.copy()
        
        if date_column in df_temporal.columns:
            # Convert to datetime
            df_temporal[date_column] = pd.to_datetime(df_temporal[date_column], errors='coerce')
            
            # Extract basic temporal components
            df_temporal['sale_year'] = df_temporal[date_column].dt.year
            df_temporal['sale_month'] = df_temporal[date_column].dt.month
            df_temporal['sale_quarter'] = df_temporal[date_column].dt.quarter
            df_temporal['sale_dayofweek'] = df_temporal[date_column].dt.dayofweek
            df_temporal['sale_dayofyear'] = df_temporal[date_column].dt.dayofyear
            
            # Market timing features (seasons, economic cycles)
            df_temporal['is_q4'] = (df_temporal['sale_quarter'] == 4).astype(int)
            df_temporal['is_winter'] = df_temporal['sale_month'].isin([12, 1, 2]).astype(int)
            df_temporal['is_spring'] = df_temporal['sale_month'].isin([3, 4, 5]).astype(int)
            df_temporal['is_summer'] = df_temporal['sale_month'].isin([6, 7, 8]).astype(int)
            df_temporal['is_fall'] = df_temporal['sale_month'].isin([9, 10, 11]).astype(int)
            
            self.engineered_features.extend([
                'sale_year', 'sale_month', 'sale_quarter', 'sale_dayofweek', 'sale_dayofyear',
                'is_q4', 'is_winter', 'is_spring', 'is_summer', 'is_fall'
            ])
            
            # Drop the original date column to avoid dtype issues
            df_temporal = df_temporal.drop(columns=[date_column])
            
            logger.info("Temporal features engineered successfully")
        
        return df_temporal
    
    def engineer_equipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer equipment-specific features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with equipment features
        """
        df_equipment = df.copy()
        
        # Age calculation (if Year Made is available)
        if 'Year Made' in df_equipment.columns and 'sale_year' in df_equipment.columns:
            df_equipment['equipment_age'] = df_equipment['sale_year'] - df_equipment['Year Made']
            df_equipment['equipment_age'] = df_equipment['equipment_age'].clip(lower=0)  # No negative ages
            self.engineered_features.append('equipment_age')
        
        # Usage intensity features
        if 'MachineHours CurrentMeter' in df_equipment.columns:
            hours = pd.to_numeric(df_equipment['MachineHours CurrentMeter'], errors='coerce').fillna(0)
            df_equipment['log_machine_hours'] = np.log1p(hours)
            age = df_equipment.get('equipment_age', pd.Series([1]*len(df_equipment))).replace(0, 1)
            df_equipment['hours_per_year'] = hours / age
            self.engineered_features.extend(['log_machine_hours', 'hours_per_year'])
        
        # Equipment size and power features
        if 'Engine Horsepower' in df_equipment.columns:
            hp = pd.to_numeric(df_equipment['Engine Horsepower'], errors='coerce')
            df_equipment['log_horsepower'] = np.log1p(hp.fillna(0))
            self.engineered_features.append('log_horsepower')
        
        # Machine condition indicators
        if 'Usage Band' in df_equipment.columns:
            usage_map = {'Low': 1, 'Medium': 2, 'High': 3}
            df_equipment['usage_intensity'] = df_equipment['Usage Band'].map(usage_map).fillna(2)
            self.engineered_features.append('usage_intensity')
        
        logger.info("Equipment features engineered successfully")
        return df_equipment
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  fit_transform: bool = True) -> pd.DataFrame:
        """Encode categorical features with proper handling of unseen categories.
        
        Args:
            df: Input DataFrame
            fit_transform: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove date columns from categorical encoding
        categorical_columns = [col for col in categorical_columns 
                             if not any(temp in col.lower() for temp in ['date', 'time'])]
        
        self.categorical_features = categorical_columns
        
        for col in categorical_columns:
            if col not in df_encoded.columns:
                continue
                
            # Handle high cardinality features differently
            unique_count = df_encoded[col].nunique()
            
            if unique_count > 20:  # High cardinality - use target encoding or frequency encoding
                if fit_transform:
                    # Simple frequency encoding for high cardinality
                    freq_map = df_encoded[col].value_counts().to_dict()
                    self.label_encoders[f"{col}_freq"] = freq_map
                    df_encoded[f"{col}_frequency"] = df_encoded[col].map(freq_map).fillna(0)
                else:
                    freq_map = self.label_encoders.get(f"{col}_freq", {})
                    df_encoded[f"{col}_frequency"] = df_encoded[col].map(freq_map).fillna(0)
                
                # Keep only top categories, group rest as 'Other'
                if fit_transform:
                    top_categories = df_encoded[col].value_counts().head(10).index.tolist()
                    self.label_encoders[f"{col}_top"] = top_categories
                else:
                    top_categories = self.label_encoders.get(f"{col}_top", [])
                
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in top_categories else 'Other'
                )
            
            # Standard label encoding
            if fit_transform:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit with unique values plus 'Unknown' for unseen categories
                    all_values = df_encoded[col].dropna().unique().tolist() + ['Unknown']
                    self.label_encoders[col].fit(all_values)
                
                # Transform with handling for unseen categories
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                try:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                except ValueError:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
            else:
                # Transform mode
                if col in self.label_encoders:
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               fit_transform: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame
            fit_transform: Whether to fit scaler (True for training, False for test)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        # Identify numerical columns (excluding categorical encoded ones)
        numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        self.numerical_features = numerical_columns
        
        if numerical_columns:
            if fit_transform:
                df_scaled[numerical_columns] = self.scaler.fit_transform(df_scaled[numerical_columns])
            else:
                df_scaled[numerical_columns] = self.scaler.transform(df_scaled[numerical_columns])
            
            logger.info(f"Scaled {len(numerical_columns)} numerical features")
        
        return df_scaled
    
    def select_features(self, df: pd.DataFrame, target: pd.Series, 
                       k_features: int = 50, fit_transform: bool = True) -> pd.DataFrame:
        """Select top k features based on statistical significance.
        
        Args:
            df: Input DataFrame
            target: Target variable
            k_features: Number of features to select
            fit_transform: Whether to fit selector (True for training, False for test)
            
        Returns:
            DataFrame with selected features
        """
        if fit_transform:
            # Ensure we don't select more features than available
            k_features = min(k_features, df.shape[1])
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            df_selected = pd.DataFrame(
                self.feature_selector.fit_transform(df, target),
                index=df.index,
                columns=df.columns[self.feature_selector.get_support()]
            )
            
            # Store selected feature names
            self.feature_names = df_selected.columns.tolist()
            
            logger.info(f"Selected {len(self.feature_names)} features out of {df.shape[1]}")
        else:
            if self.feature_selector is not None:
                df_selected = df[self.feature_names]
            else:
                df_selected = df
                logger.warning("Feature selector not fitted, returning all features")
        
        return df_selected
    
    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Fit and transform features for training data.
        
        Args:
            df: Input DataFrame
            target: Target variable
            
        Returns:
            Transformed feature DataFrame
        """
        logger.info("Starting feature engineering (fit_transform)")
        
        # Remove leakage features
        df_clean = self.remove_leakage_features(df)
        
        # Engineer temporal features
        df_temporal = self.engineer_temporal_features(df_clean)
        
        # Engineer equipment features
        df_equipment = self.engineer_equipment_features(df_temporal)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_equipment, fit_transform=True)
        
        # Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded, fit_transform=True)
        
        # Feature selection
        df_selected = self.select_features(df_scaled, target, fit_transform=True)
        
        logger.info(f"Feature engineering completed: {df_selected.shape[1]} features")
        return df_selected
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features for validation/test data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed feature DataFrame
        """
        logger.info("Starting feature engineering (transform)")
        
        # Remove leakage features
        df_clean = self.remove_leakage_features(df)
        
        # Engineer temporal features
        df_temporal = self.engineer_temporal_features(df_clean)
        
        # Engineer equipment features
        df_equipment = self.engineer_equipment_features(df_temporal)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_equipment, fit_transform=False)
        
        # Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded, fit_transform=False)
        
        # Feature selection
        df_selected = self.select_features(df_scaled, None, fit_transform=False)
        
        logger.info(f"Feature transformation completed: {df_selected.shape[1]} features")
        return df_selected
    
    def get_feature_info(self) -> Dict:
        """Get information about engineered features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'total_features': len(self.feature_names),
            'selected_features': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'engineered_features': self.engineered_features,
            'forbidden_features_removed': self.forbidden_features
        }
    
    def save_feature_artifacts(self, artifacts_dir: str):
        """Save feature engineering artifacts.
        
        Args:
            artifacts_dir: Directory to save artifacts
        """
        artifacts_path = Path(artifacts_dir) / "models"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature information
        feature_info = self.get_feature_info()
        with open(artifacts_path / "feature_list.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save encoders and scalers
        import joblib
        joblib.dump(self.label_encoders, artifacts_path / "label_encoders.pkl")
        joblib.dump(self.scaler, artifacts_path / "scaler.pkl")
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, artifacts_path / "feature_selector.pkl")
        
        logger.info(f"Feature artifacts saved to {artifacts_path}")


if __name__ == "__main__":
    # Example usage
    from src.data import load_and_validate_data
    
    data_path = "data/raw/Bit_SHM_data.csv"
    df, _ = load_and_validate_data(data_path)
    
    if 'Sales Price' in df.columns:
        target = df['Sales Price']
        
        engineer = LeakProofFeatureEngineer()
        features = engineer.fit_transform(df, target)
        
        print(f"Original features: {df.shape[1]}")
        print(f"Engineered features: {features.shape[1]}")
        print(f"Selected features: {engineer.feature_names[:10]}...")  # Show first 10