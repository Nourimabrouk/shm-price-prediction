"""Model training pipeline for equipment price prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available. Using RandomForest as fallback.")
    CatBoostRegressor = None
    Pool = None
    CATBOOST_AVAILABLE = False
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime
import joblib
import time
from itertools import product
import random
from abc import ABC, abstractmethod
from copy import deepcopy

warnings.filterwarnings('ignore')


class ConformalPredictor:
    """Industry-standard conformal prediction with guaranteed coverage.
    
    Provides uncertainty quantification with theoretical guarantees for any base model.
    Uses the split conformal prediction framework with calibration on held-out data.
    """
    
    def __init__(self, base_model, alpha: float = 0.1):
        """Initialize conformal predictor.
        
        Args:
            base_model: Trained ML model (CatBoost, RandomForest, etc.)
            alpha: Miscoverage rate (0.1 = 90% coverage, 0.2 = 80% coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self.calibration_scores = None
        self.quantile_threshold = None
        self.is_calibrated = False
        
    def calibrate(self, X_cal: pd.DataFrame, y_cal: np.ndarray):
        """Calibrate conformal scores on held-out calibration set.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get base model predictions on calibration set
        y_cal_pred = self.base_model.predict(X_cal)
        
        # Compute conformal scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - y_cal_pred)
        
        # Calculate the quantile threshold for desired coverage
        n_cal = len(self.calibration_scores)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0
        
        self.quantile_threshold = np.quantile(self.calibration_scores, quantile_level)
        self.is_calibrated = True
        
        print(f"Conformal predictor calibrated for {self.confidence_level:.0%} coverage")
        print(f"   Calibration set size: {n_cal:,}")
        print(f"   Quantile threshold: ${self.quantile_threshold:,.0f}")
        
    def predict_with_intervals(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with conformal prediction intervals.
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate conformal predictor before making interval predictions")
        
        # Base model predictions
        y_pred = self.base_model.predict(X_test)
        
        # Conformal prediction intervals
        y_lower = y_pred - self.quantile_threshold
        y_upper = y_pred + self.quantile_threshold
        
        # Ensure non-negative predictions for price data
        y_lower = np.maximum(y_lower, 0)
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred, y_lower, y_upper
    
    def validate_coverage(self, X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, float]:
        """Validate prediction interval coverage on validation set.
        
        Args:
            X_val: Validation features  
            y_val: Validation targets
            
        Returns:
            Dictionary with coverage statistics
        """
        y_pred, y_lower, y_upper = self.predict_with_intervals(X_val)
        
        # Calculate empirical coverage
        in_interval = (y_val >= y_lower) & (y_val <= y_upper)
        empirical_coverage = np.mean(in_interval)
        
        # Calculate interval metrics
        avg_width = np.mean(y_upper - y_lower)
        relative_width = np.mean((y_upper - y_lower) / np.maximum(y_pred, 1)) * 100
        
        coverage_stats = {
            'target_coverage': self.confidence_level,
            'empirical_coverage': empirical_coverage,
            'coverage_error': abs(empirical_coverage - self.confidence_level),
            'avg_interval_width': avg_width,
            'relative_width_percent': relative_width,
            'samples_in_interval': np.sum(in_interval),
            'total_samples': len(y_val)
        }
        
        print(f"\nConformal Prediction Validation ({self.confidence_level:.0%} target coverage):")
        print(f"   Empirical coverage: {empirical_coverage:.1%}")
        print(f"   Coverage error: {coverage_stats['coverage_error']:.1%}")
        print(f"   Average interval width: ${avg_width:,.0f}")
        print(f"   Relative width: {relative_width:.1f}%")
        
        return coverage_stats


class EnsembleOrchestrator:
    """Advanced multi-model orchestration with stacking capabilities.
    
    Coordinates multiple models through weighted averaging or meta-learning,
    providing robust predictions and uncertainty quantification.
    """
    
    def __init__(self, combination_method: str = 'weighted', random_state: int = 42):
        """Initialize ensemble orchestrator.
        
        Args:
            combination_method: 'weighted', 'average', or 'stacking'
            random_state: Random seed for reproducibility
        """
        self.combination_method = combination_method
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.meta_learner = None
        self.is_fitted = False
        
        if combination_method == 'stacking':
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0, random_state=random_state)
    
    def register_model(self, name: str, model: any, weight: float = 1.0):
        """Register a model with the ensemble.
        
        Args:
            name: Model identifier
            model: Trained model object
            weight: Model weight for weighted averaging (ignored for stacking)
        """
        self.models[name] = model
        self.weights[name] = weight
        print(f"Registered model '{name}' with weight {weight}")
    
    def fit_ensemble(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_val: pd.DataFrame, y_val: np.ndarray):
        """Fit the ensemble combination method.
        
        Args:
            X_train: Training features (used for stacking meta-learner)
            y_train: Training targets
            X_val: Validation features (used for weight optimization)
            y_val: Validation targets
        """
        if not self.models:
            raise ValueError("No models registered. Use register_model() first.")
        
        if self.combination_method == 'weighted':
            self._optimize_weights(X_val, y_val)
        elif self.combination_method == 'stacking':
            self._fit_meta_learner(X_train, y_train)
        # For 'average', no fitting needed
        
        self.is_fitted = True
        print(f"Ensemble fitted using {self.combination_method} method")
    
    def _optimize_weights(self, X_val: pd.DataFrame, y_val: np.ndarray):
        """Optimize model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        # Get predictions from all models
        val_preds = {}
        for name, model in self.models.items():
            val_preds[name] = model.predict(X_val)
        
        # Simple weight optimization based on individual RMSE
        rmse_scores = {}
        for name, pred in val_preds.items():
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            rmse_scores[name] = rmse
        
        # Inverse RMSE weighting (better models get higher weight)
        inv_rmse_sum = sum(1/rmse for rmse in rmse_scores.values())
        for name in self.models.keys():
            self.weights[name] = (1/rmse_scores[name]) / inv_rmse_sum
        
        print(f"   Optimized weights: {', '.join([f'{name}: {w:.3f}' for name, w in self.weights.items()])}")
    
    def _fit_meta_learner(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit meta-learner for stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Generate meta-features using cross-validation predictions
        from sklearn.model_selection import cross_val_predict
        
        meta_features = []
        for name, model in self.models.items():
            # For already-fitted models, we approximate CV predictions
            cv_pred = model.predict(X_train)  # Simplified - would use proper CV in production
            meta_features.append(cv_pred)
        
        X_meta = np.column_stack(meta_features)
        self.meta_learner.fit(X_meta, y_train)
        
        print(f"   Meta-learner fitted with {len(self.models)} base model predictions")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict method for compatibility with other interfaces."""
        return self.predict_ensemble(X_test)
    
    def predict_ensemble(self, X_test: pd.DataFrame, method: str = None) -> np.ndarray:
        """Generate ensemble predictions.
        
        Args:
            X_test: Test features
            method: Override combination method for this prediction
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        method = method or self.combination_method
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        if method == 'average':
            # Simple averaging
            pred_matrix = np.column_stack(list(predictions.values()))
            return np.mean(pred_matrix, axis=1)
        
        elif method == 'weighted':
            # Weighted averaging
            ensemble_pred = np.zeros(len(X_test))
            for name, pred in predictions.items():
                ensemble_pred += self.weights[name] * pred
            return ensemble_pred
        
        elif method == 'stacking':
            # Meta-learner prediction
            X_meta = np.column_stack(list(predictions.values()))
            return self.meta_learner.predict(X_meta)
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def predict_with_uncertainty(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with ensemble-based uncertainty estimates.
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        # Get predictions from all models
        all_predictions = []
        for model in self.models.values():
            all_predictions.append(model.predict(X_test))
        
        pred_matrix = np.column_stack(all_predictions)
        
        # Ensemble prediction
        ensemble_pred = self.predict_ensemble(X_test)
        
        # Uncertainty as standard deviation across models
        uncertainty = np.std(pred_matrix, axis=1)
        
        return ensemble_pred, uncertainty
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, any]:
        """Comprehensive ensemble evaluation.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with ensemble performance metrics
        """
        # Individual model performance
        individual_performance = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            individual_performance[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        # Ensemble performance
        ensemble_pred = self.predict_ensemble(X_test)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        # Ensemble improvement
        best_individual_rmse = min(perf['rmse'] for perf in individual_performance.values())
        improvement = (best_individual_rmse - ensemble_rmse) / best_individual_rmse * 100
        
        return {
            'individual_performance': individual_performance,
            'ensemble_performance': {
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2
            },
            'ensemble_improvement_percent': improvement,
            'combination_method': self.combination_method,
            'model_weights': self.weights.copy()
        }

class EquipmentPricePredictor:
    """Equipment price prediction pipeline with multiple model options."""
    
    def __init__(self, model_type: str = 'catboost', random_state: int = 42):
        """Initialize the price prediction pipeline.
        
        Args:
            model_type: Type of model to use ('random_forest', 'catboost')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_features = None
        self.target_column = 'sales_price'
        self.is_fitted = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model with optimal parameters."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                print("Warning: CatBoost not available. Falling back to RandomForest.")
                self.model_type = 'random_forest'
                self._initialize_model()
                return
            
            self.model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.1,
                depth=8,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features with econometric feature awareness.
        
        Enhanced to properly handle sophisticated econometric features from advanced
        feature engineering pipeline.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        # Exclude target and ID columns
        exclude_cols = [self.target_column, 'sales_id', 'machine_id', 'sales_date']
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Initialize feature lists
        categorical_features = []
        numerical_features = []
        
        # Econometric feature categories (all numerical/continuous)
        econometric_numerical_patterns = [
            '_squared', '_log', 'log1p_', '_sin', '_cos', '_trend', '_interaction', 
            '_z_by_', '_x_', '_na', 'completeness', '_bucket'
        ]
        
        for col in available_cols:
            # Force econometric features to be treated as numerical
            is_econometric = any(pattern in col for pattern in econometric_numerical_patterns)
            
            if is_econometric:
                numerical_features.append(col)
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_features.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality integers)
                unique_vals = df[col].nunique()
                # Be more conservative with econometric buckets
                if unique_vals < 20 and df[col].dtype == 'int64' and not col.endswith('_bucket'):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        return categorical_features, numerical_features
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Minimal preprocessing for modeling.
        
        Args:
            df: Raw DataFrame
            is_training: Whether this is training data (affects encoder fitting)
            
        Returns:
            Preprocessed DataFrame ready for modeling
        """
        df_processed = df.copy()
        
        # Remove rows with missing target (for training)
        if is_training and self.target_column in df_processed.columns:
            df_processed = df_processed.dropna(subset=[self.target_column])
            
            # Log-transform target to handle skewness and improve RMSLE
            df_processed[f'{self.target_column}_log'] = np.log1p(df_processed[self.target_column])
        
        # Handle temporal features
        if 'sales_date' in df_processed.columns:
            df_processed['sales_year'] = df_processed['sales_date'].dt.year
            df_processed['sales_month'] = df_processed['sales_date'].dt.month
            df_processed['sales_quarter'] = df_processed['sales_date'].dt.quarter
            
            # Calculate equipment age
            if 'year_made' in df_processed.columns:
                df_processed['age_at_sale'] = df_processed['sales_year'] - df_processed['year_made']
                # Cap unrealistic ages
                df_processed['age_at_sale'] = df_processed['age_at_sale'].clip(0, 50)
        
        # Identify feature types
        if is_training:
            self.categorical_features, numerical_features = self._identify_feature_types(df_processed)
            self.feature_columns = self.categorical_features + numerical_features
        
        # Handle missing values in categorical features
        for col in self.categorical_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Handle missing values in numerical features
        numerical_cols = [col for col in df_processed.columns if col not in self.categorical_features + [self.target_column]]
        for col in numerical_cols:
            if col in df_processed.columns:
                if col == 'machinehours_currentmeter':
                    # Use median for machine hours
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    # Use median for other numerical features
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Encode categorical variables for Random Forest
        if self.model_type == 'random_forest':
            for col in self.categorical_features:
                if col in df_processed.columns:
                    if is_training:
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            known_categories = set(self.label_encoders[col].classes_)
                            df_processed[col] = df_processed[col].astype(str)
                            unknown_mask = ~df_processed[col].isin(known_categories)
                            df_processed.loc[unknown_mask, col] = 'Unknown'
                            
                            # Add 'Unknown' to encoder if not present
                            if 'Unknown' not in known_categories:
                                self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                            
                            df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features and target for modeling.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        # Select features
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Run preprocess_data with is_training=True first.")
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Handle target
        y = None
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
        
        return X, y
    
    def temporal_split_with_audit(self, df: pd.DataFrame, test_size: float = 0.2, 
                                 date_col: str = 'sales_date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-aware split with audit trail to prevent temporal data leakage.
        
        Enhanced from internal/pipeline.py for production-grade temporal validation.
        
        Args:
            df: DataFrame with temporal data
            test_size: Fraction for validation set
            date_col: Name of the date column
            
        Returns:
            Tuple of (train_df, validation_df)
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Stable temporal sorting (mergesort maintains order for equal dates)
        order = df[date_col].argsort(kind="mergesort")
        cutoff = int(len(df) * (1 - test_size))
        
        train_idx = order[:cutoff]
        val_idx = order[cutoff:]
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Audit trail for temporal split integrity
        start_train, end_train = train_df[date_col].min(), train_df[date_col].max()
        start_val, end_val = val_df[date_col].min(), val_df[date_col].max()
        
        print(f"[TIME] [TEMPORAL AUDIT] Train: {start_train.strftime('%Y-%m-%d')} -> {end_train.strftime('%Y-%m-%d')}")
        print(f"[TIME] [TEMPORAL AUDIT] Validation: {start_val.strftime('%Y-%m-%d')} -> {end_val.strftime('%Y-%m-%d')}")
        print(f"[TIME] [TEMPORAL AUDIT] Split integrity: {'[OK] VALID' if end_train <= start_val else '[ERROR] DATA LEAKAGE DETECTED'}")
        
        # Store audit info for later reference
        self.split_audit_info = {
            'train_start': start_train,
            'train_end': end_train,
            'val_start': start_val,
            'val_end': end_val,
            'split_valid': end_train <= start_val,
            'train_samples': len(train_df),
            'val_samples': len(val_df)
        }
        
        return train_df, val_df

    def train(self, df: pd.DataFrame, validation_split: float = 0.2, 
              use_time_split: bool = True) -> Dict[str, any]:
        """Train model and return metrics.
        
        Args:
            df: Training DataFrame
            validation_split: Fraction of data to use for validation
            use_time_split: Whether to use temporal validation split
            
        Returns:
            Dictionary with training metrics and model info
        """
        print(f"Training {self.model_type} model...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, is_training=True)
        
        # Enhanced time-based splitting with audit trail
        if use_time_split and 'sales_date' in df.columns:
            print("[TIME] Using temporal validation split (prevents data leakage)")
            train_df, val_df = self.temporal_split_with_audit(df_processed, validation_split)
            
            X_train, y_train = self.prepare_features_target(train_df)
            X_val, y_val = self.prepare_features_target(val_df)
        else:
            print("[WARN]  Using random split (may cause data leakage in temporal data)")
            X, y = self.prepare_features_target(df_processed)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state
            )
        
        # Train model
        if self.model_type == 'catboost':
            # CatBoost handles categorical features automatically
            cat_feature_indices = [X_train.columns.get_loc(col) for col in self.categorical_features 
                                 if col in X_train.columns]
            
            self.model.fit(
                X_train, y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            # Random Forest training
            self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        self.is_fitted = True
        
        # Compile results
        results = {
            'model_type': self.model_type,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features_used': len(X_train.columns),
            'categorical_features': len([col for col in self.categorical_features if col in X_train.columns]),
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'feature_names': list(X_train.columns)
        }
        
        # Add feature importance with econometric analysis
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance_df.head(10).to_dict('records')
            
            # Analyze econometric feature performance
            econometric_analysis = self._analyze_econometric_features(importance_df, X_train.columns)
            results['econometric_analysis'] = econometric_analysis
        
        print(f"Training completed. Validation RMSE: ${val_metrics['rmse']:,.0f}")
        
        # Display econometric feature insights
        if 'econometric_analysis' in results:
            self._display_econometric_insights(results['econometric_analysis'])
        
        return results

    def fit_on_pre_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, any]:
        """Fit the model on explicitly provided train/validation DataFrames without re-splitting.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
        
        Returns:
            Dictionary containing train/validation metrics and feature metadata
        """
        # Preprocess separately to avoid leakage
        train_processed = self.preprocess_data(train_df, is_training=True)
        val_processed = self.preprocess_data(val_df, is_training=False)

        X_train, y_train = self.prepare_features_target(train_processed)
        X_val, y_val = self.prepare_features_target(val_processed)

        if self.model_type == 'catboost':
            categorical_indices = [X_train.columns.get_loc(col) for col in self.categorical_features
                                   if col in X_train.columns]
            self.model.fit(
                X_train, y_train,
                cat_features=categorical_indices,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        # Predictions and metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)

        self.is_fitted = True

        results: Dict[str, any] = {
            'model_type': self.model_type,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features_used': len(X_train.columns),
            'categorical_features': len([c for c in self.categorical_features if c in X_train.columns]),
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'feature_names': list(X_train.columns)
        }

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance_df.head(10).to_dict('records')

            econometric_analysis = self._analyze_econometric_features(importance_df, X_train.columns)
            results['econometric_analysis'] = econometric_analysis

        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of price predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not initialized. Model must be trained first.")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, is_training=False)
        
        # Validate features are available
        missing_features = set(self.feature_columns) - set(df_processed.columns)
        if missing_features:
            warnings.warn(f"Missing features for prediction: {missing_features}. Will use default values.")
            for feat in missing_features:
                df_processed[feat] = 0
        
        X, _ = self.prepare_features_target(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Ensure positive predictions for price data
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure positive predictions
        y_pred_pos = np.maximum(y_pred, 1)
        
        # Standard regression metrics
        mae = mean_absolute_error(y_true, y_pred_pos)
        mse = mean_squared_error(y_true, y_pred_pos)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_pos)
        
        # Business-relevant metrics
        mape = np.mean(np.abs((y_true - y_pred_pos) / y_true)) * 100
        
        # Root Mean Squared Logarithmic Error (RMSLE) 
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))
        
        # Within tolerance accuracy (business metric)
        tolerance_15_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.15) * 100
        tolerance_25_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.25) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'rmsle': rmsle,
            'within_15_pct': tolerance_15_pct,
            'within_25_pct': tolerance_25_pct
        }
    
    def _analyze_econometric_features(self, importance_df: pd.DataFrame, all_features: List[str]) -> Dict[str, any]:
        """Analyze the contribution and effectiveness of econometric features.
        
        Args:
            importance_df: DataFrame with feature importances
            all_features: List of all feature names
            
        Returns:
            Dictionary with econometric feature analysis
        """
        # Categorize features by econometric type
        feature_categories = {
            'depreciation': ['age_squared', 'log1p_age'],
            'seasonality': ['sale_month_sin', 'sale_month_cos', 'year_trend', 'is_2008_2009'],
            'interactions': ['usage_age_interaction', 'age_x_hp'],
            'binning': ['age_bucket', 'hours_bucket'],
            'normalization': [f for f in all_features if '_z_by_' in f],
            'data_quality': ['info_completeness'] + [f for f in all_features if f.endswith('_na')],
            'basic': []  # Will be populated with non-econometric features
        }
        
        # Identify which features are present and classify them
        present_features = set(all_features)
        category_performance = {}
        
        for category, patterns in feature_categories.items():
            if category == 'basic':
                continue
                
            # Find matching features for this category
            if category in ['normalization', 'data_quality']:
                matching_features = patterns  # Already filtered above
            else:
                matching_features = [f for f in patterns if f in present_features]
            
            if matching_features:
                # Get importance scores for these features
                cat_importance = importance_df[importance_df['feature'].isin(matching_features)]
                
                category_performance[category] = {
                    'features': matching_features,
                    'count': len(matching_features),
                    'total_importance': cat_importance['importance'].sum() if not cat_importance.empty else 0,
                    'avg_importance': cat_importance['importance'].mean() if not cat_importance.empty else 0,
                    'top_feature': cat_importance.iloc[0]['feature'] if not cat_importance.empty else None,
                    'top_importance': cat_importance.iloc[0]['importance'] if not cat_importance.empty else 0
                }
        
        # Calculate overall econometric contribution
        econometric_features = []
        for category_data in category_performance.values():
            econometric_features.extend(category_data['features'])
        
        econometric_importance = importance_df[importance_df['feature'].isin(econometric_features)]
        basic_importance = importance_df[~importance_df['feature'].isin(econometric_features)]
        
        total_econometric = econometric_importance['importance'].sum() if not econometric_importance.empty else 0
        total_basic = basic_importance['importance'].sum() if not basic_importance.empty else 0
        total_all = total_econometric + total_basic
        
        analysis = {
            'category_performance': category_performance,
            'overall_contribution': {
                'econometric_features': len(econometric_features),
                'econometric_importance_share': (total_econometric / total_all * 100) if total_all > 0 else 0,
                'basic_features': len(basic_importance),
                'basic_importance_share': (total_basic / total_all * 100) if total_all > 0 else 0,
                'top_econometric_features': econometric_importance.head(5).to_dict('records') if not econometric_importance.empty else []
            },
            'sophistication_metrics': {
                'categories_present': len([c for c in category_performance if category_performance[c]['count'] > 0]),
                'total_econometric_features': len(econometric_features),
                'advanced_feature_density': len(econometric_features) / len(all_features) * 100 if all_features else 0
            }
        }
        
        return analysis
    
    def _display_econometric_insights(self, analysis: Dict[str, any]) -> None:
        """Display formatted insights about econometric feature performance.
        
        Args:
            analysis: Output from _analyze_econometric_features
        """
        print("\n" + "="*80)
        print("[BRAIN] ECONOMETRIC FEATURE ANALYSIS")
        print("="*80)
        
        # Overall contribution
        overall = analysis['overall_contribution']
        sophistication = analysis['sophistication_metrics']
        
        print(f"[DATA] Feature Composition:")
        print(f"  • Econometric features: {overall['econometric_features']} ({sophistication['advanced_feature_density']:.1f}% of all features)")
        print(f"  • Traditional features: {overall['basic_features']}")
        print(f"  • Econometric importance share: {overall['econometric_importance_share']:.1f}%")
        
        print(f"\n[TARGET] Sophistication Metrics:")
        print(f"  • Categories implemented: {sophistication['categories_present']}/6 econometric categories")
        print(f"  • Advanced feature density: {sophistication['advanced_feature_density']:.1f}%")
        
        # Category performance
        print(f"\n[SEARCH] Category Performance:")
        category_perf = analysis['category_performance']
        
        for category, data in sorted(category_perf.items(), key=lambda x: x[1]['total_importance'], reverse=True):
            if data['count'] > 0:
                print(f"  • {category.title()}: {data['count']} features, {data['total_importance']:.3f} total importance")
                if data['top_feature']:
                    print(f"    └─ Best: {data['top_feature']} ({data['top_importance']:.3f})")
        
        # Top econometric features
        if overall['top_econometric_features']:
            print(f"\n[STAR] Top Econometric Features:")
            for i, feature_data in enumerate(overall['top_econometric_features'][:3], 1):
                print(f"  {i}. {feature_data['feature']}: {feature_data['importance']:.3f}")
        
        print("="*80)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")


def train_competition_grade_models(df: pd.DataFrame, use_optimization: bool = False, time_budget: int = 15) -> Dict[str, any]:
    """Train competition-grade models with comprehensive evaluation.
    
    Args:
        df: Training DataFrame
        use_optimization: Whether to use hyperparameter optimization for CatBoost
        time_budget: Time budget in minutes for optimization
        
    Returns:
        Dictionary with results from all models
    """
    print("Training Competition-Grade Models...")
    
    # Train baseline Random Forest
    print("\n1. Training Baseline Random Forest...")
    rf_model = EquipmentPricePredictor(model_type='random_forest', random_state=42)
    rf_results = rf_model.train(df.sample(min(5000, len(df)), random_state=42), validation_split=0.2, use_time_split=True)
    
    # Train advanced CatBoost
    print("\n2. Training Advanced CatBoost...")
    if use_optimization:
        print("Using hyperparameter optimization...")
        cb_results = train_optimized_catboost_pipeline(df, time_budget_minutes=time_budget)
    else:
        cb_model = EquipmentPricePredictor(model_type='catboost', random_state=42)
        cb_results = cb_model.train(df.sample(min(5000, len(df)), random_state=42), validation_split=0.2, use_time_split=True)
    
    results = {
        'Random Forest': rf_results,
        'CatBoost': cb_results
    }
    
    # Compare results
    comparison_data = []
    for name, result in results.items():
        val_metrics = result['validation_metrics']
        comparison_data.append({
            'Model': name,
            'RMSE': f"${val_metrics['rmse']:,.0f}",
            'MAE': f"${val_metrics['mae']:,.0f}", 
            'R²': f"{val_metrics['r2']:.3f}",
            'MAPE': f"{val_metrics['mape']:.1f}%",
            'Within 15%': f"{val_metrics['within_15_pct']:.1f}%",
            'Within 25%': f"{val_metrics['within_25_pct']:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("COMPETITION-GRADE MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return results


def get_catboost_smart_defaults(n_samples: int, n_features: int, n_categoricals: int) -> dict:
    """
    Dataset-aware CatBoost defaults optimized for heavy equipment data.
    Based on 412K samples, 66 features, 5K+ categorical levels.
    """
    
    # Scale iterations based on dataset size
    if n_samples > 300000:
        iterations = 1000  # Large dataset: more iterations needed
    elif n_samples > 100000:
        iterations = 750   # Medium dataset
    else:
        iterations = 500   # Small dataset
    
    # Scale depth based on feature count and categorical complexity
    if n_categoricals > 1000:  # High-cardinality categoricals like equipment models
        depth = 8              # Deeper trees for complex categoricals
    elif n_categoricals > 100:
        depth = 7              # Medium depth
    else:
        depth = 6              # Standard depth
    
    # Learning rate scaled by dataset size (larger datasets can handle smaller LR)
    learning_rate = 0.05 if n_samples > 200000 else 0.08
    
    return {
        'iterations': iterations,
        'learning_rate': learning_rate,
        'depth': depth,
        'l2_leaf_reg': 3,           # Moderate regularization for price data
        'border_count': 128,        # Higher for continuous price features
        'bagging_temperature': 1,   # Standard stochasticity
        'random_strength': 1,       # Standard randomness
        'early_stopping_rounds': 50, # Aggressive early stopping for time savings
        'verbose': False,
        'eval_metric': 'RMSE',      # RMSE for log-price (equivalent to RMSLE)
        'loss_function': 'RMSE',
        'task_type': 'CPU',         # Assume CPU environment
        'random_seed': 42
        # Remove cat_features='auto' to avoid conflicts
    }


def optimize_catboost_coarse(X_train, y_train, X_val, y_val, cat_features=None):
    """
    Fast coarse hyperparameter optimization focusing on high-impact parameters.
    Optimized for heavy equipment dataset characteristics.
    """
    
    base_params = get_catboost_smart_defaults(len(X_train), len(X_train.columns), 
                                            len(cat_features) if cat_features else 10)
    
    # Focus on parameters with highest impact for this dataset type
    param_grid = {
        'learning_rate': [0.03, 0.05, 0.08],      # 3 values: small range around optimal
        'depth': [6, 7, 8, 9],                    # 4 values: critical for categorical data
        'l2_leaf_reg': [1, 3, 5, 10],            # 4 values: important for price regression
        'iterations': [800, 1000, 1200]          # 3 values: scaled for large dataset
    }
    
    best_score = float('inf')
    best_params = None
    results = []
    
    print(f"Coarse grid search: {len(list(ParameterGrid(param_grid)))} combinations")
    start_time = time.time()
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        # Update base params with grid search params
        model_params = {**base_params, **params}
        model_params['early_stopping_rounds'] = 30  # More aggressive for grid search
        
        try:
            model = CatBoostRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                verbose=False,
                plot=False
            )
            
            # Get validation score
            val_pred = model.predict(X_val)
            score = np.sqrt(np.mean((y_val - val_pred) ** 2))  # RMSE on log-price
            
            results.append({
                'params': params,
                'score': score,
                'iterations': model.tree_count_
            })
            
            if score < best_score:
                best_score = score
                best_params = params
                
            # Time constraint check
            if time.time() - start_time > 900:  # 15 minutes max
                print(f"Time limit reached after {i+1} combinations")
                break
                
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    print(f"Best coarse score: {best_score:.4f}")
    print(f"Best coarse params: {best_params}")
    
    return best_params, best_score, results


def optimize_catboost_fine(X_train, y_train, X_val, y_val, best_coarse_params, cat_features=None):
    """
    Fine-tune around best coarse parameters with focused ranges.
    """
    
    base_params = get_catboost_smart_defaults(len(X_train), len(X_train.columns), 
                                            len(cat_features) if cat_features else 10)
    
    # Fine-tune around best coarse parameters
    fine_grid = {}
    
    # Learning rate fine-tuning
    best_lr = best_coarse_params['learning_rate']
    fine_grid['learning_rate'] = [
        max(0.01, best_lr * 0.8),
        best_lr,
        min(0.15, best_lr * 1.2)
    ]
    
    # Depth fine-tuning
    best_depth = best_coarse_params['depth']
    fine_grid['depth'] = [
        max(4, best_depth - 1),
        best_depth,
        min(12, best_depth + 1)
    ]
    
    # L2 regularization fine-tuning
    best_l2 = best_coarse_params['l2_leaf_reg']
    fine_grid['l2_leaf_reg'] = [
        max(0.1, best_l2 * 0.5),
        best_l2,
        best_l2 * 2
    ]
    
    # Additional parameters for fine-tuning
    fine_grid.update({
        'bagging_temperature': [0.8, 1.0, 1.2],  # Stochasticity tuning
        'border_count': [64, 128, 254]           # Continuous feature splits
    })
    
    best_score = float('inf')
    best_params = None
    
    print(f"Fine tuning around best coarse parameters...")
    
    # Try smaller combinations for fine-tuning
    
    # Sample combinations rather than full grid (time constraint)
    param_combinations = list(product(
        fine_grid['learning_rate'],
        fine_grid['depth'], 
        fine_grid['l2_leaf_reg'][:2],  # Reduce combinations
        fine_grid['bagging_temperature'][:2],  # Reduce combinations
        fine_grid['border_count'][:2]  # Reduce combinations
    ))
    
    # Randomly sample if too many combinations
    if len(param_combinations) > 20:
        random.seed(42)
        param_combinations = random.sample(param_combinations, 20)
    
    for lr, depth, l2, bagging_temp, border_count in param_combinations:
        model_params = {**base_params, **best_coarse_params}
        model_params.update({
            'learning_rate': lr,
            'depth': depth,
            'l2_leaf_reg': l2,
            'bagging_temperature': bagging_temp,
            'border_count': border_count,
            'early_stopping_rounds': 40
        })
        
        try:
            model = CatBoostRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                verbose=False
            )
            
            val_pred = model.predict(X_val)
            score = np.sqrt(np.mean((y_val - val_pred) ** 2))
            
            if score < best_score:
                best_score = score
                best_params = {
                    'learning_rate': lr,
                    'depth': depth,
                    'l2_leaf_reg': l2,
                    'bagging_temperature': bagging_temp,
                    'border_count': border_count
                }
                
        except Exception as e:
            continue
    
    print(f"Best fine-tuned score: {best_score:.4f}")
    print(f"Best fine-tuned params: {best_params}")
    
    # Combine with base params
    final_params = {**base_params, **best_coarse_params, **best_params}
    
    return final_params, best_score


def train_optimized_catboost(X_train, y_train, X_val, y_val, X_test=None, y_test=None, 
                           cat_features=None, time_budget_minutes=25):
    """
    Complete CatBoost optimization pipeline for heavy equipment pricing.
    
    Args:
        time_budget_minutes: Total time budget for optimization (default 25 min)
    
    Returns:
        dict: {
            'model': trained CatBoostRegressor,
            'best_params': optimized parameters,
            'scores': {'train': ..., 'val': ..., 'test': ...},
            'feature_importance': feature importance rankings,
            'optimization_history': parameter search results
        }
    """
    
    print("Starting CatBoost hyperparameter optimization...")
    print(f"Dataset: {len(X_train):,} train, {len(X_val):,} val samples")
    print(f"Time budget: {time_budget_minutes} minutes")
    
    start_time = time.time()
    
    # Stage 1: Smart defaults (immediate)
    base_params = get_catboost_smart_defaults(
        len(X_train), len(X_train.columns), 
        len(cat_features) if cat_features else 0
    )
    print(f"Smart defaults configured")
    
    # Stage 2: Coarse optimization (15 minutes)
    print("Stage 2: Coarse grid search...")
    best_coarse_params, best_coarse_score, coarse_results = optimize_catboost_coarse(
        X_train, y_train, X_val, y_val, cat_features
    )
    
    # Stage 3: Fine tuning (remaining time)
    remaining_time = time_budget_minutes - (time.time() - start_time) / 60
    if remaining_time > 5 and best_coarse_params is not None:  # At least 5 minutes for fine tuning
        print("Stage 3: Fine tuning...")
        final_params, best_score = optimize_catboost_fine(
            X_train, y_train, X_val, y_val, best_coarse_params, cat_features
        )
    else:
        if best_coarse_params is not None:
            final_params = {**base_params, **best_coarse_params}
            best_score = best_coarse_score
        else:
            # Fallback to base params if optimization failed
            final_params = base_params
            best_score = float('inf')
    
    # Train final model with more iterations
    print("Training final optimized model...")
    final_params['iterations'] = min(2000, final_params['iterations'] * 2)
    final_params['early_stopping_rounds'] = 100
    
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        verbose=100  # Show progress for final training
    )
    
    # Evaluate final model
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    
    scores = {
        'train_rmse': np.sqrt(np.mean((y_train - train_pred) ** 2)),
        'val_rmse': np.sqrt(np.mean((y_val - val_pred) ** 2)),
        'train_mae': np.mean(np.abs(y_train - train_pred)),
        'val_mae': np.mean(np.abs(y_val - val_pred))
    }
    
    # Test evaluation if provided
    if X_test is not None and y_test is not None:
        test_pred = final_model.predict(X_test)
        scores['test_rmse'] = np.sqrt(np.mean((y_test - test_pred) ** 2))
        scores['test_mae'] = np.mean(np.abs(y_test - test_pred))
    
    # Feature importance
    feature_importance = pd.Series(
        final_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    total_time = (time.time() - start_time) / 60
    print(f"Optimization complete in {total_time:.1f} minutes")
    print(f"Best validation RMSE: {scores['val_rmse']:.4f}")
    
    return {
        'model': final_model,
        'best_params': final_params,
        'scores': scores,
        'feature_importance': feature_importance,
        'optimization_history': coarse_results,
        'optimization_time': total_time
    }


def train_optimized_catboost_pipeline(df: pd.DataFrame, time_budget_minutes: int = 15) -> Dict:
    """
    Complete pipeline for training optimized CatBoost with time constraints.
    Integrates with existing evaluation framework.
    """
    from src.evaluation import ModelEvaluator
    
    # Sample data for optimization (use full dataset for production)
    sample_size = min(20000, len(df))  # Larger sample for optimization
    df_sample = df.sample(sample_size, random_state=42)
    
    # Prepare features - simple approach for optimization
    # Remove non-feature columns
    exclude_cols = ['sales_id', 'sales_date', 'machine_id']
    feature_cols = [col for col in df_sample.columns if col not in exclude_cols + ['sales_price']]
    
    X = df_sample[feature_cols]
    y = df_sample['sales_price']
    
    # Identify categorical features (keep raw strings for CatBoost)
    categorical_features = [
        col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category'
    ]

    # Fill missing values: categoricals with 'Missing', numericals with median
    X_encoded = X.copy()
    for col in categorical_features:
        X_encoded[col] = X_encoded[col].fillna('Missing')
    X_encoded = X_encoded.apply(lambda s: s.fillna(s.median()) if s.dtype != object and s.dtype.kind in 'fcmi' else s)
    
    # Time-aware split for validation
    split_date = df_sample['sales_date'].quantile(0.8)
    train_mask = df_sample['sales_date'] <= split_date
    
    X_train = X_encoded[train_mask]
    y_train = y[train_mask]
    X_val = X_encoded[~train_mask]
    y_val = y[~train_mask]
    
    print(f"Training on {len(X_train):,} samples, validating on {len(X_val):,} samples")
    
    # Run optimization
    # CatBoost can accept categorical columns by index
    cat_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]

    opt_results = train_optimized_catboost(
        X_train, y_train, X_val, y_val,
        cat_features=cat_indices,
        time_budget_minutes=time_budget_minutes
    )
    
    # Evaluate using comprehensive evaluator
    from src.evaluation import evaluate_model_comprehensive
    train_pred = opt_results['model'].predict(X_train)
    val_pred = opt_results['model'].predict(X_val)

    train_eval = evaluate_model_comprehensive(y_train.values, train_pred, "Optimized CatBoost - Train")
    val_eval = evaluate_model_comprehensive(y_val.values, val_pred, "Optimized CatBoost - Validation")
    
    # Return in format compatible with existing pipeline
    return {
        'model': opt_results['model'],
        'training_metrics': train_eval['metrics'],
        'validation_metrics': val_eval['metrics'],
        'feature_importance': opt_results['feature_importance'].to_dict(),
        'optimization_results': opt_results,
        'training_time': opt_results['optimization_time'],
        'sample_size': sample_size
    }


if __name__ == "__main__":
    # Test the models module
    from data_loader import load_shm_data
    
    df, validation_report = load_shm_data()
    results = train_competition_grade_models(df)
    
    print(f"\nAll models training completed successfully!")