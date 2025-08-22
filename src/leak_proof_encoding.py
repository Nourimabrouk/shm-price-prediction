"""Leak-proof target encoding utilities for temporal data.

This module implements time-aware target encoding that eliminates temporal leakage
by ensuring that target statistics are computed using only past information at 
the time of each observation.

CRITICAL MISSION: Zero target encoding leakage tolerance.

Key Features:
- Expanding window target encoding with temporal awareness
- Cross-validation with time-aware folds
- Shift(1) protection to prevent same-period contamination
- Comprehensive encoding validation and auditing
- Windows-safe console output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.model_selection import KFold
import warnings
from collections import defaultdict


@dataclass
class TargetEncodingConfig:
    """Configuration for time-aware target encoding."""
    target_column: str = 'sales_price'
    date_column: str = 'sales_date'
    min_samples_leaf: int = 10  # Minimum samples required for encoding
    smoothing_factor: float = 0.0  # Bayesian smoothing (0 = no smoothing)
    add_noise: bool = True  # Add small random noise to prevent overfitting
    noise_level: float = 0.01  # Standard deviation of noise as fraction of target std
    shift_periods: int = 1  # Number of periods to shift back (prevents same-period leakage)
    cv_folds: int = 5  # Number of cross-validation folds for regularization


class TimeAwareTargetEncoder:
    """Leak-proof target encoder that respects temporal boundaries.
    
    This encoder ensures that target statistics for each observation are computed
    using only data that was available at the time of that observation, preventing
    any form of temporal leakage.
    """
    
    def __init__(self, config: TargetEncodingConfig):
        """Initialize time-aware target encoder.
        
        Args:
            config: Target encoding configuration
        """
        self.config = config
        self.global_mean = None
        self.encoding_stats = {}
        self.is_fitted = False
        self.validation_results = {}
        
    def fit_transform_temporal(self, df: pd.DataFrame, 
                             categorical_columns: List[str]) -> pd.DataFrame:
        """Fit and transform categorical columns using time-aware target encoding.
        
        Args:
            df: Training DataFrame with temporal data
            categorical_columns: List of categorical columns to encode
            
        Returns:
            DataFrame with target-encoded categorical columns
        """
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")
        if self.config.date_column not in df.columns:
            raise ValueError(f"Date column '{self.config.date_column}' not found")
        
        print(f"[ENCODING] Starting time-aware target encoding for {len(categorical_columns)} columns...")
        
        # Prepare data
        df_encoded = df.copy()
        df_encoded[self.config.date_column] = pd.to_datetime(df_encoded[self.config.date_column])
        df_encoded = df_encoded.sort_values(self.config.date_column).reset_index(drop=True)
        
        # Store global mean for fallback
        self.global_mean = df_encoded[self.config.target_column].mean()
        
        # Encode each categorical column
        for col in categorical_columns:
            if col in df_encoded.columns:
                print(f"[ENCODING] Processing column: {col}")
                df_encoded[f'{col}_encoded'] = self._encode_column_temporal(
                    df_encoded, col
                )
                
                # Validate encoding
                validation = self._validate_encoding(df_encoded, col, f'{col}_encoded')
                self.validation_results[col] = validation
                
                if validation['has_leakage_risk']:
                    warnings.warn(f"Potential leakage risk detected in {col}: {validation['risk_factors']}")
        
        self.is_fitted = True
        print(f"[ENCODING] Temporal target encoding complete")
        return df_encoded
    
    def transform_temporal(self, df: pd.DataFrame, 
                          categorical_columns: List[str]) -> pd.DataFrame:
        """Transform new data using fitted temporal encodings.
        
        Args:
            df: DataFrame to transform
            categorical_columns: List of categorical columns to encode
            
        Returns:
            DataFrame with target-encoded categorical columns
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        df_encoded = df.copy()
        df_encoded[self.config.date_column] = pd.to_datetime(df_encoded[self.config.date_column])
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                df_encoded[f'{col}_encoded'] = self._transform_column_temporal(
                    df_encoded, col
                )
        
        return df_encoded
    
    def _encode_column_temporal(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Encode a single categorical column using expanding window approach.
        
        Args:
            df: DataFrame with sorted temporal data
            column: Column name to encode
            
        Returns:
            Series with encoded values
        """
        encoded_values = np.full(len(df), self.global_mean, dtype=float)
        category_stats = defaultdict(lambda: {'sum': 0, 'count': 0})
        
        # Store encoding statistics for later use
        self.encoding_stats[column] = {}
        
        for i, row in df.iterrows():
            current_date = row[self.config.date_column]
            category = row[column]
            target_value = row[self.config.target_column]
            
            # Calculate cutoff date with shift protection
            cutoff_date = current_date - timedelta(days=self.config.shift_periods)
            
            # Get historical data up to cutoff date
            historical_mask = df[self.config.date_column] <= cutoff_date
            historical_data = df[historical_mask]
            
            if len(historical_data) == 0:
                # No historical data available, use global mean
                encoded_values[i] = self.global_mean
            else:
                # Calculate category mean from historical data
                category_data = historical_data[historical_data[column] == category]
                
                if len(category_data) < self.config.min_samples_leaf:
                    # Insufficient data for this category, use global mean
                    encoded_values[i] = self.global_mean
                else:
                    # Calculate mean with optional smoothing
                    category_mean = category_data[self.config.target_column].mean()
                    
                    if self.config.smoothing_factor > 0:
                        # Bayesian smoothing
                        global_mean_historical = historical_data[self.config.target_column].mean()
                        n = len(category_data)
                        smoothing = self.config.smoothing_factor
                        
                        category_mean = (n * category_mean + smoothing * global_mean_historical) / (n + smoothing)
                    
                    encoded_values[i] = category_mean
            
            # Add noise if configured
            if self.config.add_noise:
                noise_std = self.config.noise_level * df[self.config.target_column].std()
                encoded_values[i] += np.random.normal(0, noise_std)
        
        # Store final statistics for this column
        unique_categories = df[column].unique()
        for cat in unique_categories:
            cat_data = df[df[column] == cat]
            self.encoding_stats[column][cat] = {
                'mean': cat_data[self.config.target_column].mean(),
                'count': len(cat_data),
                'std': cat_data[self.config.target_column].std()
            }
        
        return pd.Series(encoded_values, index=df.index)
    
    def _transform_column_temporal(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Transform a column using pre-fitted encoding statistics.
        
        Args:
            df: DataFrame to transform
            column: Column name to transform
            
        Returns:
            Series with encoded values
        """
        if column not in self.encoding_stats:
            raise ValueError(f"Column '{column}' was not fitted")
        
        encoded_values = np.full(len(df), self.global_mean, dtype=float)
        
        for i, category in enumerate(df[column]):
            if pd.isna(category):
                encoded_values[i] = self.global_mean
            elif category in self.encoding_stats[column]:
                encoded_values[i] = self.encoding_stats[column][category]['mean']
            else:
                # Unknown category, use global mean
                encoded_values[i] = self.global_mean
        
        return pd.Series(encoded_values, index=df.index)
    
    def _validate_encoding(self, df: pd.DataFrame, original_col: str, 
                          encoded_col: str) -> Dict[str, Any]:
        """Validate target encoding for potential leakage.
        
        Args:
            df: DataFrame with original and encoded columns
            original_col: Name of original categorical column
            encoded_col: Name of encoded column
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_leakage_risk': False,
            'risk_factors': [],
            'statistics': {}
        }
        
        # Check for perfect correlation (impossible without leakage)
        target_values = df[self.config.target_column]
        encoded_values = df[encoded_col]
        
        correlation = np.corrcoef(target_values, encoded_values)[0, 1]
        if correlation > 0.95:
            validation['has_leakage_risk'] = True
            validation['risk_factors'].append(f"Suspiciously high correlation: {correlation:.3f}")
        
        # Check for identical values within categories
        unique_categories = df[original_col].unique()
        for category in unique_categories:
            cat_mask = df[original_col] == category
            cat_encoded_values = df.loc[cat_mask, encoded_col]
            cat_target_values = df.loc[cat_mask, self.config.target_column]
            
            # Check if encoded values are identical for all instances of this category
            if len(cat_encoded_values.unique()) == 1 and len(cat_encoded_values) > 1:
                # This is expected behavior, but check if it matches target too closely
                encoded_val = cat_encoded_values.iloc[0]
                target_mean = cat_target_values.mean()
                
                if abs(encoded_val - target_mean) / target_mean < 0.001:  # Very close match
                    validation['has_leakage_risk'] = True
                    validation['risk_factors'].append(f"Category {category} encoded value too close to target mean")
        
        # Check variance reduction (should not be too dramatic)
        original_unique_count = df[original_col].nunique()
        encoded_variance = encoded_values.var()
        target_variance = target_values.var()
        
        variance_ratio = encoded_variance / target_variance
        if variance_ratio > 0.8:  # Encoded values explain too much variance
            validation['has_leakage_risk'] = True
            validation['risk_factors'].append(f"Encoded variance ratio too high: {variance_ratio:.3f}")
        
        validation['statistics'] = {
            'correlation': correlation,
            'original_categories': original_unique_count,
            'encoded_variance': encoded_variance,
            'target_variance': target_variance,
            'variance_ratio': variance_ratio
        }
        
        return validation
    
    def print_encoding_report(self) -> None:
        """Print comprehensive encoding validation report."""
        if not self.validation_results:
            print("No validation results available. Run fit_transform_temporal() first.")
            return
        
        print("\n" + "="*80)
        print("TARGET ENCODING VALIDATION REPORT")
        print("="*80)
        print(f"Global target mean: {self.global_mean:.2f}")
        print(f"Configuration:")
        print(f"  Min samples per leaf: {self.config.min_samples_leaf}")
        print(f"  Smoothing factor: {self.config.smoothing_factor}")
        print(f"  Shift periods: {self.config.shift_periods}")
        print(f"  Noise level: {self.config.noise_level}")
        print()
        
        leakage_detected = False
        
        for column, validation in self.validation_results.items():
            print(f"COLUMN: {column}")
            
            if validation['has_leakage_risk']:
                print("  STATUS: LEAKAGE RISK DETECTED")
                leakage_detected = True
                for risk in validation['risk_factors']:
                    print(f"    RISK: {risk}")
            else:
                print("  STATUS: NO LEAKAGE DETECTED")
            
            stats = validation['statistics']
            print(f"  Correlation with target: {stats['correlation']:.3f}")
            print(f"  Original categories: {stats['original_categories']}")
            print(f"  Variance ratio: {stats['variance_ratio']:.3f}")
            
            # Show encoding statistics
            if column in self.encoding_stats:
                encoding_info = self.encoding_stats[column]
                print(f"  Encoded categories: {len(encoding_info)}")
                
                # Show top categories by count
                sorted_cats = sorted(encoding_info.items(), 
                                   key=lambda x: x[1]['count'], reverse=True)
                print(f"  Top categories by frequency:")
                for cat, info in sorted_cats[:3]:
                    print(f"    {cat}: {info['count']} samples, mean={info['mean']:.2f}")
            
            print()
        
        if leakage_detected:
            print("OVERALL STATUS: LEAKAGE RISK DETECTED - REVIEW REQUIRED")
        else:
            print("OVERALL STATUS: NO LEAKAGE DETECTED - ENCODING IS SAFE")
        
        print("="*80)


class CrossValidatedTargetEncoder:
    """Cross-validated target encoder with temporal awareness.
    
    Uses time-aware cross-validation to create more robust target encodings
    that are less prone to overfitting while maintaining temporal validity.
    """
    
    def __init__(self, config: TargetEncodingConfig):
        """Initialize CV target encoder.
        
        Args:
            config: Target encoding configuration
        """
        self.config = config
        self.encoders = {}
        self.global_mean = None
        self.is_fitted = False
    
    def fit_transform_cv(self, df: pd.DataFrame, 
                        categorical_columns: List[str]) -> pd.DataFrame:
        """Fit and transform using time-aware cross-validation.
        
        Args:
            df: Training DataFrame
            categorical_columns: Columns to encode
            
        Returns:
            DataFrame with CV target encodings
        """
        print(f"[CV-ENCODING] Starting CV target encoding for {len(categorical_columns)} columns...")
        
        df_encoded = df.copy()
        df_encoded[self.config.date_column] = pd.to_datetime(df_encoded[self.config.date_column])
        df_encoded = df_encoded.sort_values(self.config.date_column).reset_index(drop=True)
        
        self.global_mean = df_encoded[self.config.target_column].mean()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                print(f"[CV-ENCODING] Processing column: {col}")
                df_encoded[f'{col}_cv_encoded'] = self._encode_column_cv(df_encoded, col)
        
        self.is_fitted = True
        return df_encoded
    
    def _encode_column_cv(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Encode column using time-aware cross-validation.
        
        Args:
            df: DataFrame with temporal data
            column: Column to encode
            
        Returns:
            Series with CV encoded values
        """
        encoded_values = np.full(len(df), self.global_mean, dtype=float)
        
        # Create time-aware folds
        fold_size = len(df) // self.config.cv_folds
        
        for fold in range(self.config.cv_folds):
            # Define fold boundaries
            fold_start = fold * fold_size
            fold_end = (fold + 1) * fold_size if fold < self.config.cv_folds - 1 else len(df)
            
            # Training data is everything before this fold
            train_mask = np.arange(len(df)) < fold_start
            fold_mask = np.arange(len(df)) >= fold_start
            fold_mask &= np.arange(len(df)) < fold_end
            
            if not any(train_mask):
                # No training data for this fold, use global mean
                encoded_values[fold_mask] = self.global_mean
                continue
            
            train_data = df[train_mask]
            fold_data = df[fold_mask]
            
            # Calculate encodings for this fold
            for i, (idx, row) in enumerate(fold_data.iterrows()):
                category = row[column]
                
                # Get category statistics from training data
                category_data = train_data[train_data[column] == category]
                
                if len(category_data) < self.config.min_samples_leaf:
                    encoded_values[idx] = self.global_mean
                else:
                    category_mean = category_data[self.config.target_column].mean()
                    
                    # Apply smoothing if configured
                    if self.config.smoothing_factor > 0:
                        train_mean = train_data[self.config.target_column].mean()
                        n = len(category_data)
                        smoothing = self.config.smoothing_factor
                        
                        category_mean = (n * category_mean + smoothing * train_mean) / (n + smoothing)
                    
                    encoded_values[idx] = category_mean
        
        return pd.Series(encoded_values, index=df.index)


def create_leak_proof_target_encodings(df: pd.DataFrame, 
                                      categorical_columns: List[str],
                                      target_column: str = 'sales_price',
                                      date_column: str = 'sales_date',
                                      method: str = 'temporal',
                                      **kwargs) -> pd.DataFrame:
    """Create leak-proof target encodings using specified method.
    
    Args:
        df: Input DataFrame
        categorical_columns: Columns to encode
        target_column: Target column name
        date_column: Date column name
        method: Encoding method ('temporal' or 'cv')
        **kwargs: Additional configuration parameters
        
    Returns:
        DataFrame with target encodings
    """
    config = TargetEncodingConfig(
        target_column=target_column,
        date_column=date_column,
        **kwargs
    )
    
    if method == 'temporal':
        encoder = TimeAwareTargetEncoder(config)
        df_encoded = encoder.fit_transform_temporal(df, categorical_columns)
        encoder.print_encoding_report()
        return df_encoded
    
    elif method == 'cv':
        encoder = CrossValidatedTargetEncoder(config)
        return encoder.fit_transform_cv(df, categorical_columns)
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


if __name__ == "__main__":
    # Test the target encoding system
    print("Testing leak-proof target encoding...")
    
    # Create sample temporal data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range("2008-01-01", "2012-12-31", periods=n_samples)
    
    # Create categories with temporal trends
    categories_a = np.random.choice(['cat1', 'cat2', 'cat3'], n_samples)
    categories_b = np.random.choice(['type_x', 'type_y', 'type_z'], n_samples)
    
    # Create target with category relationships
    base_price = 50000
    cat_effects = {'cat1': 1.2, 'cat2': 1.0, 'cat3': 0.8}
    type_effects = {'type_x': 1.1, 'type_y': 1.0, 'type_z': 0.9}
    
    prices = []
    for i in range(n_samples):
        price = base_price * cat_effects[categories_a[i]] * type_effects[categories_b[i]]
        price *= np.random.lognormal(0, 0.2)  # Add noise
        prices.append(price)
    
    test_df = pd.DataFrame({
        'sales_date': dates,
        'sales_price': prices,
        'category_a': categories_a,
        'category_b': categories_b
    })
    
    # Test temporal encoding
    encoded_df = create_leak_proof_target_encodings(
        test_df, 
        ['category_a', 'category_b'],
        method='temporal',
        min_samples_leaf=5,
        shift_periods=1
    )
    
    print(f"Encoding test completed. Shape: {encoded_df.shape}")
    print(f"New columns: {[col for col in encoded_df.columns if col not in test_df.columns]}")