"""Data validation utilities for SHM price prediction pipeline.

This module provides comprehensive data validation to ensure data quality
and catch issues early in the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from .config import TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
except ImportError:
    from src.config import TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class DataValidator:
    """Comprehensive data validation for production ML pipelines."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, raise exceptions on validation failures.
                        If False, log warnings and attempt recovery.
        """
        self.strict_mode = strict_mode
        self.validation_report = {}
        
    def validate_input_data(self, df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          target_column: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Validate input DataFrame for ML pipeline.
        
        Args:
            df: Input DataFrame to validate
            required_columns: List of required column names
            target_column: Name of target column (for additional checks)
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check DataFrame is not empty
        if df.empty:
            report['errors'].append("DataFrame is empty")
            report['is_valid'] = False
            return report['is_valid'], report
            
        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                report['errors'].append(f"Missing required columns: {missing_cols}")
                report['is_valid'] = False
                
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            report['warnings'].append(f"Found {n_duplicates} duplicate rows")
            report['statistics']['n_duplicates'] = n_duplicates
            
        # Check for missing values
        missing_stats = df.isnull().sum()
        high_missing = missing_stats[missing_stats > len(df) * 0.5]
        if not high_missing.empty:
            report['warnings'].append(f"Columns with >50% missing: {high_missing.to_dict()}")
            
        report['statistics']['missing_values'] = missing_stats.to_dict()
        
        # Validate target column if specified
        if target_column and target_column in df.columns:
            target_validation = self._validate_target(df[target_column])
            report['target_validation'] = target_validation
            if not target_validation['is_valid']:
                report['is_valid'] = False
                report['errors'].extend(target_validation['errors'])
                
        # Check data types
        dtype_issues = self._check_data_types(df)
        if dtype_issues:
            report['warnings'].extend(dtype_issues)
            
        # Check for data leakage indicators
        leakage_risks = self._check_data_leakage(df)
        if leakage_risks:
            report['warnings'].extend(leakage_risks)
            
        return report['is_valid'], report
    
    def _validate_target(self, target: pd.Series) -> Dict[str, Any]:
        """Validate target variable for regression.
        
        Args:
            target: Target series to validate
            
        Returns:
            Validation report for target
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for nulls in target
        n_nulls = target.isnull().sum()
        if n_nulls > 0:
            report['errors'].append(f"Target has {n_nulls} missing values")
            report['is_valid'] = False
            
        # Check for negative prices (shouldn't happen for equipment)
        if (target < 0).any():
            n_negative = (target < 0).sum()
            report['errors'].append(f"Target has {n_negative} negative values")
            report['is_valid'] = False
            
        # Check for outliers using IQR method
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        n_outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        if n_outliers > 0:
            report['warnings'].append(f"Target has {n_outliers} extreme outliers")
            
        # Statistics
        report['statistics'] = {
            'mean': target.mean(),
            'median': target.median(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'n_unique': target.nunique(),
            'n_outliers': n_outliers
        }
        
        return report
    
    def _check_data_types(self, df: pd.DataFrame) -> List[str]:
        """Check for data type issues.
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of data type warnings
        """
        warnings = []
        
        # Check for object columns that might be numeric
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                non_numeric_ratio = pd.to_numeric(df[col], errors='coerce').isnull().mean()
                if non_numeric_ratio < 0.1:  # Less than 10% non-numeric
                    warnings.append(f"Column '{col}' appears to be numeric but stored as object")
            except:
                pass
                
        # Check for high cardinality categoricals
        for col in df.select_dtypes(include=['object']).columns:
            n_unique = df[col].nunique()
            if n_unique > 0.5 * len(df):
                warnings.append(f"Column '{col}' has very high cardinality ({n_unique} unique values)")
                
        return warnings
    
    def _check_data_leakage(self, df: pd.DataFrame) -> List[str]:
        """Check for potential data leakage indicators.
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of data leakage warnings
        """
        warnings = []
        
        # Check for ID columns that might leak information
        id_pattern_cols = [col for col in df.columns if 'id' in col.lower()]
        for col in id_pattern_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Check if ID is correlated with index (temporal leakage)
                if len(df) > 1:
                    correlation = df[col].corr(pd.Series(range(len(df))))
                    if abs(correlation) > 0.9:
                        warnings.append(f"Column '{col}' highly correlated with index (potential temporal leakage)")
                        
        return warnings
    
    def validate_prediction_input(self, X: pd.DataFrame, 
                                 trained_features: List[str]) -> Tuple[bool, pd.DataFrame]:
        """Validate input for prediction.
        
        Args:
            X: Input features for prediction
            trained_features: List of features the model was trained on
            
        Returns:
            Tuple of (is_valid, processed_X)
        """
        # Check for missing features
        missing_features = set(trained_features) - set(X.columns)
        if missing_features:
            if self.strict_mode:
                raise ValueError(f"Missing features for prediction: {missing_features}")
            else:
                # Add missing features with default values
                for feat in missing_features:
                    X[feat] = 0
                    warnings.warn(f"Added missing feature '{feat}' with default value 0")
                    
        # Check for extra features
        extra_features = set(X.columns) - set(trained_features)
        if extra_features:
            # Remove extra features
            X = X[trained_features]
            
        # Ensure column order matches training
        X = X[trained_features]
        
        return True, X
    
    def validate_time_series_split(self, df: pd.DataFrame, 
                                  date_column: str,
                                  train_size: float = 0.8) -> Dict[str, Any]:
        """Validate temporal split for time series data.
        
        Args:
            df: DataFrame with temporal data
            date_column: Name of date column
            train_size: Proportion of data for training
            
        Returns:
            Validation report for temporal split
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if date_column not in df.columns:
            report['errors'].append(f"Date column '{date_column}' not found")
            report['is_valid'] = False
            return report
            
        # Check if dates are properly sorted
        dates = pd.to_datetime(df[date_column])
        if not dates.is_monotonic_increasing:
            report['warnings'].append("Dates are not monotonically increasing")
            
        # Calculate split point
        split_idx = int(len(df) * train_size)
        train_dates = dates.iloc[:split_idx]
        test_dates = dates.iloc[split_idx:]
        
        # Check for overlap
        if train_dates.max() >= test_dates.min():
            report['errors'].append("Train and test sets have temporal overlap")
            report['is_valid'] = False
            
        # Check for gaps
        date_diff = (test_dates.min() - train_dates.max()).days
        if date_diff > 30:
            report['warnings'].append(f"Large gap ({date_diff} days) between train and test sets")
            
        report['statistics'] = {
            'train_start': str(train_dates.min()),
            'train_end': str(train_dates.max()),
            'test_start': str(test_dates.min()),
            'test_end': str(test_dates.max()),
            'train_size': len(train_dates),
            'test_size': len(test_dates),
            'gap_days': date_diff
        }
        
        return report


def validate_model_inputs(X_train: pd.DataFrame, y_train: np.ndarray,
                         X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
    """Quick validation of model inputs.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Validation report
    """
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check shapes
    if len(X_train) != len(y_train):
        report['errors'].append(f"X_train and y_train length mismatch: {len(X_train)} != {len(y_train)}")
        report['is_valid'] = False
        
    if len(X_test) != len(y_test):
        report['errors'].append(f"X_test and y_test length mismatch: {len(X_test)} != {len(y_test)}")
        report['is_valid'] = False
        
    # Check features match
    if set(X_train.columns) != set(X_test.columns):
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        missing_in_train = set(X_test.columns) - set(X_train.columns)
        
        if missing_in_test:
            report['errors'].append(f"Features in train but not test: {missing_in_test}")
            report['is_valid'] = False
        if missing_in_train:
            report['errors'].append(f"Features in test but not train: {missing_in_train}")
            report['is_valid'] = False
            
    # Check for NaN values
    if X_train.isnull().any().any():
        report['warnings'].append("X_train contains NaN values")
    if X_test.isnull().any().any():
        report['warnings'].append("X_test contains NaN values")
    if pd.Series(y_train).isnull().any():
        report['errors'].append("y_train contains NaN values")
        report['is_valid'] = False
    if pd.Series(y_test).isnull().any():
        report['errors'].append("y_test contains NaN values")
        report['is_valid'] = False
        
    return report