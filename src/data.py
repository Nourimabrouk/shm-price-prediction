"""Data loading and validation module for SHM Heavy Equipment Price Prediction.

This module provides robust data loading, validation, and preprocessing with:
- Windows encoding compatibility 
- Data quality checks and validation
- Temporal validation for price prediction
- Reproducible train/validation/test splits
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging for Windows compatibility (ASCII only)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DataValidator:
    """Validates data quality and prepares for model training."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data with Windows encoding compatibility.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Try UTF-8 first, fallback to latin-1 for Windows compatibility
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, trying latin-1 encoding")
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            logger.warning(f"Standard encoding failed, trying cp1252: {e}")
            df = pd.read_csv(file_path, encoding='cp1252')
            
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'price_validation': {},
            'temporal_validation': {}
        }
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        validation_results['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_percentage': (missing_counts / len(df) * 100).round(2).to_dict()
        }
        
        # Data type analysis
        validation_results['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Price validation
        if 'Sales Price' in df.columns:
            prices = df['Sales Price'].dropna()
            validation_results['price_validation'] = {
                'min_price': float(prices.min()),
                'max_price': float(prices.max()),
                'mean_price': float(prices.mean()),
                'median_price': float(prices.median()),
                'zero_prices': int((prices == 0).sum()),
                'negative_prices': int((prices < 0).sum())
            }
        
        # Temporal validation
        if 'Sales date' in df.columns:
            try:
                dates = pd.to_datetime(df['Sales date'], errors='coerce')
                validation_results['temporal_validation'] = {
                    'date_range': {
                        'min_date': str(dates.min()),
                        'max_date': str(dates.max())
                    },
                    'invalid_dates': int(dates.isnull().sum()),
                    'unique_dates': int(dates.nunique())
                }
            except Exception as e:
                validation_results['temporal_validation'] = {'error': str(e)}
        
        logger.info("Data validation completed successfully")
        return validation_results
    
    def prepare_temporal_splits(self, df: pd.DataFrame, 
                              date_column: str = 'Sales date',
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create temporal train/validation/test splits.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx)
        """
        # Convert dates and sort
        df_sorted = df.copy()
        df_sorted[date_column] = pd.to_datetime(df_sorted[date_column], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[date_column])
        df_sorted = df_sorted.sort_values(date_column).reset_index(drop=True)
        
        n_samples = len(df_sorted)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_idx = df_sorted.index[:train_end].values
        val_idx = df_sorted.index[train_end:val_end].values
        test_idx = df_sorted.index[val_end:].values
        
        logger.info(f"Temporal splits created - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_idx, val_idx, test_idx
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic preprocessing while preserving temporal structure.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Convert date column
        if 'Sales date' in df_processed.columns:
            df_processed['Sales date'] = pd.to_datetime(df_processed['Sales date'], errors='coerce')
        
        # Handle missing values in key columns
        if 'Sales Price' in df_processed.columns:
            # Remove rows with missing prices (can't train without target)
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna(subset=['Sales Price'])
            dropped_rows = initial_rows - len(df_processed)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing prices")
        
        # Remove obvious outliers (prices <= 0)
        if 'Sales Price' in df_processed.columns:
            initial_rows = len(df_processed)
            df_processed = df_processed[df_processed['Sales Price'] > 0]
            dropped_rows = initial_rows - len(df_processed)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with invalid prices (<= 0)")
        
        # Handle missing values in numeric columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Sales Price':  # Don't fill target variable
                median_value = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_value)
        
        # Handle missing values in categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Sales date':  # Don't fill date column
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        logger.info(f"Data preprocessing completed: {len(df_processed)} rows remaining")
        return df_processed


def load_and_validate_data(data_path: str, 
                          artifacts_dir: str = "artifacts",
                          random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """Main function to load and validate data with temporal splits.
    
    Args:
        data_path: Path to the raw data file
        artifacts_dir: Directory to save artifacts
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (processed_dataframe, validation_results)
    """
    validator = DataValidator(random_state=random_state)
    
    # Load data
    df = validator.load_data(data_path)
    
    # Validate data quality
    validation_results = validator.validate_data_quality(df)
    
    # Preprocess data
    df_processed = validator.preprocess_data(df)
    
    # Create temporal splits
    train_idx, val_idx, test_idx = validator.prepare_temporal_splits(df_processed)
    
    # Save splits to artifacts
    splits_dir = Path(artifacts_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({'index': train_idx}).to_csv(splits_dir / "train_idx.csv", index=False)
    pd.DataFrame({'index': val_idx}).to_csv(splits_dir / "val_idx.csv", index=False)
    pd.DataFrame({'index': test_idx}).to_csv(splits_dir / "test_idx.csv", index=False)
    
    # Update validation results with split information
    validation_results['splits'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'total_size': len(df_processed)
    }
    
    logger.info("Data loading and validation completed successfully")
    
    return df_processed, validation_results


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/Bit_SHM_data.csv"
    df, results = load_and_validate_data(data_path)
    print("Data validation results:")
    print(f"Total rows: {results['total_rows']}")
    print(f"Total columns: {results['total_columns']}")
    print(f"Train/Val/Test sizes: {results['splits']}")