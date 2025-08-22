"""Unit tests for data validation and preprocessing."""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.temporal_validation import TemporalSplitter, TemporalSplitConfig
from src.data_loader import SHMDataLoader


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator(strict_mode=False)
        
        # Create sample valid DataFrame
        self.valid_df = pd.DataFrame({
            'sales_price': [10000, 15000, 20000, 25000, 30000],
            'year_made': [2010, 2012, 2015, 2018, 2020],
            'manufacturer': ['CAT', 'JOHN DEERE', 'CAT', 'KOMATSU', 'JOHN DEERE'],
            'sales_date': pd.date_range('2021-01-01', periods=5, freq='M')
        })
        
        # Create problematic DataFrame
        self.problem_df = pd.DataFrame({
            'sales_price': [10000, -5000, None, 25000, 30000],  # Negative and null values
            'year_made': [2010, 2012, 2015, 2018, 2020],
            'manufacturer': ['CAT', 'JOHN DEERE', 'CAT', 'KOMATSU', 'JOHN DEERE']
        })
        
    def test_valid_data_validation(self):
        """Test validation of valid data."""
        is_valid, report = self.validator.validate_input_data(
            self.valid_df,
            required_columns=['sales_price', 'year_made'],
            target_column='sales_price'
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(len(report['errors']), 0)
        
    def test_missing_columns_validation(self):
        """Test validation with missing required columns."""
        is_valid, report = self.validator.validate_input_data(
            self.valid_df,
            required_columns=['sales_price', 'year_made', 'non_existent_column']
        )
        
        self.assertFalse(is_valid)
        self.assertTrue(any('Missing required columns' in err for err in report['errors']))
        
    def test_target_validation(self):
        """Test target variable validation."""
        # Test with problematic target
        is_valid, report = self.validator.validate_input_data(
            self.problem_df,
            target_column='sales_price'
        )
        
        self.assertFalse(is_valid)
        self.assertTrue(any('negative values' in err for err in report['errors']))
        self.assertTrue(any('missing values' in err for err in report['errors']))
        
    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()
        is_valid, report = self.validator.validate_input_data(empty_df)
        
        self.assertFalse(is_valid)
        self.assertTrue(any('empty' in err.lower() for err in report['errors']))
        
    def test_duplicate_detection(self):
        """Test duplicate row detection."""
        df_with_dupes = pd.concat([self.valid_df, self.valid_df.iloc[[0]]], ignore_index=True)
        is_valid, report = self.validator.validate_input_data(df_with_dupes)
        
        self.assertTrue(any('duplicate' in warn.lower() for warn in report['warnings']))
        self.assertEqual(report['statistics']['n_duplicates'], 1)
        
    def test_high_missing_value_warning(self):
        """Test warning for columns with high missing values."""
        df_high_missing = self.valid_df.copy()
        df_high_missing['mostly_missing'] = [1, None, None, None, None]
        
        is_valid, report = self.validator.validate_input_data(df_high_missing)
        
        self.assertTrue(any('>50% missing' in warn for warn in report['warnings']))
        
    def test_temporal_split_validation(self):
        """Test temporal split validation."""
        report = self.validator.validate_time_series_split(
            self.valid_df,
            date_column='sales_date',
            train_size=0.6
        )
        
        self.assertTrue(report['is_valid'])
        self.assertEqual(report['statistics']['train_size'], 3)
        self.assertEqual(report['statistics']['test_size'], 2)
        
    def test_temporal_split_overlap_detection(self):
        """Test detection of temporal overlap in splits."""
        # Create DataFrame with non-monotonic dates
        df_overlap = self.valid_df.copy()
        df_overlap['sales_date'] = [
            pd.Timestamp('2021-01-01'),
            pd.Timestamp('2021-03-01'),
            pd.Timestamp('2021-02-01'),  # Out of order
            pd.Timestamp('2021-04-01'),
            pd.Timestamp('2021-05-01')
        ]
        
        report = self.validator.validate_time_series_split(
            df_overlap,
            date_column='sales_date',
            train_size=0.6
        )
        
        self.assertTrue(any('not monotonically increasing' in warn for warn in report['warnings']))
        

class TestModelInputValidation(unittest.TestCase):
    """Test model input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample train/test data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.y_train = np.array([100, 200, 300, 400, 500])
        
        self.X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70]
        })
        self.y_test = np.array([600, 700])
        
    def test_valid_model_inputs(self):
        """Test validation of valid model inputs."""
        report = validate_model_inputs(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        self.assertTrue(report['is_valid'])
        self.assertEqual(len(report['errors']), 0)
        
    def test_shape_mismatch_detection(self):
        """Test detection of shape mismatches."""
        # Create mismatched y_train
        y_train_wrong = np.array([100, 200, 300])  # Wrong length
        
        report = validate_model_inputs(
            self.X_train, y_train_wrong,
            self.X_test, self.y_test
        )
        
        self.assertFalse(report['is_valid'])
        self.assertTrue(any('length mismatch' in err for err in report['errors']))
        
    def test_feature_mismatch_detection(self):
        """Test detection of feature mismatches."""
        # Create X_test with different features
        X_test_wrong = pd.DataFrame({
            'feature1': [6, 7],
            'feature3': [60, 70]  # Different feature name
        })
        
        report = validate_model_inputs(
            self.X_train, self.y_train,
            X_test_wrong, self.y_test
        )
        
        self.assertFalse(report['is_valid'])
        self.assertTrue(any('Features in train but not test' in err for err in report['errors']))
        
    def test_nan_detection(self):
        """Test detection of NaN values."""
        # Add NaN to X_train
        X_train_nan = self.X_train.copy()
        X_train_nan.iloc[0, 0] = np.nan
        
        # Add NaN to y_train  
        y_train_nan = self.y_train.copy().astype(float)
        y_train_nan[0] = np.nan
        
        report = validate_model_inputs(
            X_train_nan, y_train_nan,
            self.X_test, self.y_test
        )
        
        self.assertFalse(report['is_valid'])
        self.assertTrue(any('X_train contains NaN' in warn for warn in report['warnings']))
        self.assertTrue(any('y_train contains NaN' in err for err in report['errors']))
        

class TestPredictionValidation(unittest.TestCase):
    """Test prediction input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator(strict_mode=False)
        self.trained_features = ['feature1', 'feature2', 'feature3']
        
    def test_missing_features_handling(self):
        """Test handling of missing features in prediction input."""
        # Create input with missing feature
        X_pred = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [10, 20]
            # feature3 is missing
        })
        
        is_valid, X_processed = self.validator.validate_prediction_input(
            X_pred, self.trained_features
        )
        
        self.assertTrue(is_valid)
        self.assertIn('feature3', X_processed.columns)
        self.assertEqual(X_processed['feature3'].iloc[0], 0)  # Default value
        
    def test_extra_features_removal(self):
        """Test removal of extra features in prediction input."""
        # Create input with extra feature
        X_pred = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [10, 20],
            'feature3': [100, 200],
            'feature4': [1000, 2000]  # Extra feature
        })
        
        is_valid, X_processed = self.validator.validate_prediction_input(
            X_pred, self.trained_features
        )
        
        self.assertTrue(is_valid)
        self.assertNotIn('feature4', X_processed.columns)
        self.assertEqual(list(X_processed.columns), self.trained_features)
        
    def test_column_order_preservation(self):
        """Test that column order matches training."""
        # Create input with different column order
        X_pred = pd.DataFrame({
            'feature3': [100, 200],
            'feature1': [1, 2],
            'feature2': [10, 20]
        })
        
        is_valid, X_processed = self.validator.validate_prediction_input(
            X_pred, self.trained_features
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(list(X_processed.columns), self.trained_features)


if __name__ == '__main__':
    unittest.main()