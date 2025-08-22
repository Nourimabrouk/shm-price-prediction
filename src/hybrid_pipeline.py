# flake8: noqa
"""Hybrid Pipeline - Consolidation of src/ and internal/ features.

This module combines the best of both implementations:
- src/: Advanced ML with optimization and visualization
- internal/: Robust temporal validation and business logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

try:
    from .data_loader import SHMDataLoader
    from .models import EquipmentPricePredictor, train_optimized_catboost
    from .evaluation import ModelEvaluator
    from .eda import analyze_shm_dataset
except ImportError:
    # Fallback to absolute imports when running as main module
    from src.data_loader import SHMDataLoader
    from src.models import EquipmentPricePredictor, train_optimized_catboost
    from src.evaluation import ModelEvaluator
    from src.eda import analyze_shm_dataset


class HybridEquipmentPredictor:
    """Combined equipment price predictor using src/ and internal/ approaches."""
    
    def __init__(self, optimization_enabled: bool = True, time_budget: int = 15):
        """Initialize hybrid predictor.
        
        Args:
            optimization_enabled: Whether to use hyperparameter optimization
            time_budget: Time budget for optimization (minutes)
        """
        self.optimization_enabled = optimization_enabled
        self.time_budget = time_budget
        self.baseline_model = None
        self.advanced_model = None
        self.split_audit_info = None
        self.data_audit_info = None
        
    def load_and_audit_data(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data with comprehensive auditing.
        
        Args:
            file_path: Path to SHM dataset
            
        Returns:
            Tuple of (cleaned DataFrame, audit report)
        """
        print("[DATA] Loading data with enhanced auditing...")
        
        # Use enhanced data loader
        loader = SHMDataLoader(Path(file_path))
        df = loader.load_data()
        validation_report = loader.validate_data(df)
        
        # Store audit information
        self.data_audit_info = {
            'original_shape': df.shape,
            'missing_data_patterns': validation_report.get('missing_data', {}),
            'temporal_range': validation_report.get('temporal_analysis', {}),
            'data_quality_issues': []
        }
        
        return df, validation_report
    
    def temporal_split_with_comprehensive_audit(self, df: pd.DataFrame, 
                                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced temporal split with comprehensive business auditing.
        
        Args:
            df: DataFrame with temporal data
            test_size: Fraction for validation set
            
        Returns:
            Tuple of (train_df, validation_df)
        """
        print("[TIME] Performing temporal split with comprehensive audit...")
        
        # Find date column using robust detection
        loader = SHMDataLoader(Path("dummy"))  # Just for methods
        date_col = loader.find_column_robust(loader.DATE_CANDIDATES, df)
        
        if not date_col:
            raise ValueError("No date column found for temporal splitting")
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove rows with invalid dates
        valid_dates_before = df[date_col].notna().sum()
        df_clean = df.dropna(subset=[date_col])
        valid_dates_after = len(df_clean)
        
        if valid_dates_after < valid_dates_before:
            print(f"[WARN]  [DATE AUDIT] Removed {valid_dates_before - valid_dates_after} rows with invalid dates")
        
        # Stable temporal sorting
        order = df_clean[date_col].argsort(kind="mergesort")
        cutoff = int(len(df_clean) * (1 - test_size))
        
        train_idx = order[:cutoff]
        val_idx = order[cutoff:]
        
        train_df = df_clean.iloc[train_idx].copy()
        val_df = df_clean.iloc[val_idx].copy()
        
        # Comprehensive audit trail
        start_train, end_train = train_df[date_col].min(), train_df[date_col].max()
        start_val, end_val = val_df[date_col].min(), val_df[date_col].max()
        
        # Business logic validation
        temporal_gap = (start_val - end_train).days
        split_valid = end_train <= start_val
        
        print(f"[TIME] [TEMPORAL AUDIT] Train period: {start_train.strftime('%Y-%m-%d')} -> {end_train.strftime('%Y-%m-%d')}")
        print(f"[TIME] [TEMPORAL AUDIT] Validation period: {start_val.strftime('%Y-%m-%d')} -> {end_val.strftime('%Y-%m-%d')}")
        print(f"[TIME] [TEMPORAL AUDIT] Temporal gap: {temporal_gap} days")
        print(f"[TIME] [TEMPORAL AUDIT] Split integrity: {'[OK] VALID' if split_valid else '[ERROR] DATA LEAKAGE DETECTED'}")
        
        # Market regime analysis
        train_years = train_df[date_col].dt.year.unique()
        val_years = val_df[date_col].dt.year.unique()
        crisis_years = set([2008, 2009, 2010])
        
        train_has_crisis = bool(set(train_years) & crisis_years)
        val_has_crisis = bool(set(val_years) & crisis_years)
        
        print(f"[TIME] [MARKET AUDIT] Train includes crisis years: {'Yes' if train_has_crisis else 'No'}")
        print(f"[TIME] [MARKET AUDIT] Validation includes crisis years: {'Yes' if val_has_crisis else 'No'}")
        
        # Store comprehensive audit info
        self.split_audit_info = {
            'train_start': start_train,
            'train_end': end_train,
            'val_start': start_val,
            'val_end': end_val,
            'temporal_gap_days': temporal_gap,
            'split_valid': split_valid,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'train_years': sorted(train_years),
            'val_years': sorted(val_years),
            'train_has_crisis': train_has_crisis,
            'val_has_crisis': val_has_crisis,
            'date_column_used': date_col
        }
        
        return train_df, val_df
    
    def train_hybrid_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train hybrid models with best practices from both implementations.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dictionary with comprehensive results from both models
        """
        print("[MODEL] Training hybrid models with enhanced capabilities...")
        
        # Temporal split with comprehensive audit
        train_df, val_df = self.temporal_split_with_comprehensive_audit(df)
        
        hybrid_results = {}
        
        # Train baseline Random Forest (internal/ approach)
        print("\n[1] Training Baseline Random Forest (Production Pipeline)")
        baseline_predictor = EquipmentPricePredictor(model_type='random_forest', random_state=42)
        
        # Use smaller sample for baseline
        sample_size = min(5000, len(train_df))
        train_sample = train_df.sample(sample_size, random_state=42)
        val_sample = val_df.sample(min(1000, len(val_df)), random_state=42)
        
        # Respect explicit temporal split: fit without re-splitting
        baseline_results = baseline_predictor.fit_on_pre_split(train_sample, val_sample)
        
        # Get predictions for evaluation
        baseline_pred_train = baseline_predictor.predict(train_sample)
        baseline_pred_val = baseline_predictor.predict(val_sample)
        
        # Enhanced evaluation with prediction intervals
        evaluator = ModelEvaluator()
        train_intervals = evaluator.compute_prediction_intervals(
            train_sample['sales_price'].values, baseline_pred_train
        )
        val_intervals = evaluator.compute_prediction_intervals(
            val_sample['sales_price'].values, baseline_pred_val  
        )
        
        baseline_results['prediction_intervals'] = {
            'train': train_intervals,
            'validation': val_intervals
        }
        
        hybrid_results['baseline_random_forest'] = baseline_results
        
        # Train advanced CatBoost (src/ optimization approach)
        print("\n[2] Training Advanced CatBoost")
        
        if self.optimization_enabled:
            print(f"[FAST] Using hyperparameter optimization (budget: {self.time_budget} minutes)")
            
            # Use larger sample for optimization
            opt_sample_size = min(20000, len(train_df))
            train_opt_sample = train_df.sample(opt_sample_size, random_state=42)
            val_opt_sample = val_df.sample(min(4000, len(val_df)), random_state=42)
            
            # Prepare data for optimization
            exclude_cols = ['sales_id', 'sales_date', 'machine_id', 'sales_price']
            feature_cols = [col for col in train_opt_sample.columns if col not in exclude_cols]
            
            X_train = train_opt_sample[feature_cols]
            y_train = train_opt_sample['sales_price']
            X_val = val_opt_sample[feature_cols]
            y_val = val_opt_sample['sales_price']
            
            # Identify categorical features
            categorical_features = []
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                    categorical_features.append(col)
            
            # Fill missing values for optimization
            for col in categorical_features:
                X_train[col] = X_train[col].fillna('Missing')
                X_val[col] = X_val[col].fillna('Missing')
            
            X_train = X_train.fillna(X_train.median())
            X_val = X_val.fillna(X_train.median())  # Use training medians
            
            # Run optimization
            opt_results = train_optimized_catboost(
                X_train, y_train, X_val, y_val,
                cat_features=categorical_features,
                time_budget_minutes=self.time_budget
            )
            
            # Add prediction intervals
            train_pred_opt = opt_results['model'].predict(X_train)
            val_pred_opt = opt_results['model'].predict(X_val)
            
            opt_train_intervals = evaluator.compute_prediction_intervals(y_train.values, train_pred_opt)
            opt_val_intervals = evaluator.compute_prediction_intervals(y_val.values, val_pred_opt)
            
            opt_results['prediction_intervals'] = {
                'train': opt_train_intervals,
                'validation': opt_val_intervals
            }
            
            hybrid_results['advanced_catboost'] = opt_results
            
        else:
            # Standard CatBoost training
            advanced_predictor = EquipmentPricePredictor(model_type='catboost', random_state=42)
            advanced_results = advanced_predictor.fit_on_pre_split(train_sample, val_sample)
            
            # Add prediction intervals
            adv_pred_train = advanced_predictor.predict(train_sample)
            adv_pred_val = advanced_predictor.predict(val_sample)
            
            adv_train_intervals = evaluator.compute_prediction_intervals(
                train_sample['sales_price'].values, adv_pred_train
            )
            adv_val_intervals = evaluator.compute_prediction_intervals(
                val_sample['sales_price'].values, adv_pred_val
            )
            
            advanced_results['prediction_intervals'] = {
                'train': adv_train_intervals,
                'validation': adv_val_intervals
            }
            
            hybrid_results['advanced_catboost'] = advanced_results
        
        # Add audit information to results
        hybrid_results['audit_info'] = {
            'temporal_split': self.split_audit_info,
            'data_quality': self.data_audit_info
        }
        
        return hybrid_results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive business report.
        
        Args:
            results: Results from train_hybrid_models
        """
        print("\n" + "="*80)
        print("[TARGET] HYBRID EQUIPMENT PRICE PREDICTION REPORT")
        print("="*80)
        
        # Audit summary
        if 'audit_info' in results:
            audit = results['audit_info']
            temporal = audit.get('temporal_split', {})
            
            print(f"\n[TIME] TEMPORAL VALIDATION AUDIT:")
            print(f"   Split integrity: {'[OK] VALID' if temporal.get('split_valid') else '[ERROR] INVALID'}")
            print(f"   Training samples: {temporal.get('train_samples', 0):,}")
            print(f"   Validation samples: {temporal.get('val_samples', 0):,}")
            print(f"   Temporal gap: {temporal.get('temporal_gap_days', 0)} days")
        
        # Model comparison
        print(f"\n[MODEL] MODEL PERFORMANCE COMPARISON:")
        for model_name, model_results in results.items():
            if model_name == 'audit_info':
                continue
                
            if 'validation_metrics' in model_results:
                metrics = model_results['validation_metrics']
                print(f"\n   {model_name.upper()}:")
                print(f"      RMSE: ${metrics.get('rmse', 0):,.0f}")
                print(f"      Within 15%: {metrics.get('within_15_pct', 0):.1f}%")
                print(f"      R2 Score: {metrics.get('r2', 0):.3f}")
                
                # Prediction intervals summary
                if 'prediction_intervals' in model_results:
                    intervals = model_results['prediction_intervals']
                    if 'validation' in intervals:
                        val_lower, val_upper = intervals['validation']
                        avg_width = np.mean(val_upper - val_lower)
                        print(f"      Avg Uncertainty: +/-${avg_width:,.0f}")
        
        print("\n[SUCCESS] CONSOLIDATION SUMMARY:")
        print("   [OK] Temporal data leakage prevention (from internal/)")
        print("   [OK] Hyperparameter optimization (from src/)")
        print("   [OK] Business-aware data preprocessing (enhanced)")
        print("   [OK] Prediction intervals for uncertainty quantification")
        print("   [OK] Audit trail and validation")
        print("="*80)


def run_hybrid_pipeline(file_path: str, optimize: bool = True, time_budget: int = 15) -> Dict[str, Any]:
    """Run the complete hybrid pipeline.
    
    Args:
        file_path: Path to SHM dataset
        optimize: Whether to use hyperparameter optimization
        time_budget: Time budget for optimization (minutes)
        
    Returns:
        Dictionary with all pipeline results
    """
    predictor = HybridEquipmentPredictor(optimization_enabled=optimize, 
                                        time_budget=time_budget)
    
    # Load and audit data
    df, validation_report = predictor.load_and_audit_data(file_path)
    
    # Run EDA for business insights
    key_findings, eda_analysis = analyze_shm_dataset(df)
    
    # Train hybrid models
    results = predictor.train_hybrid_models(df)
    
    # Generate comprehensive report
    predictor.generate_comprehensive_report(results)
    
    return {
        'data_validation': validation_report,
        'eda_findings': key_findings,
        'eda_analysis': eda_analysis,
        'model_results': results,
        'audit_info': results.get('audit_info', {})
    }


if __name__ == "__main__":
    # Test the hybrid pipeline in a safe way without saving artifacts
    try:
        results = run_hybrid_pipeline("./data/raw/Bit_SHM_data.csv", optimize=True, time_budget=10)
        print("[OK] Hybrid pipeline test completed!")
    except Exception as e:
        print(f"[WARN] Hybrid pipeline self-test failed: {e}")