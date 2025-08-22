#!/usr/bin/env python3
"""
PRODUCTION MODEL TRAINING PIPELINE
==================================

Fixed implementation following Blue Book winning strategies for SHM heavy equipment pricing.
Addresses critical temporal validation issues and implements honest performance evaluation.

Mission: Train production-grade ML models achieving >70% accuracy within 15% tolerance.
Strategy: STRICT temporal validation with ZERO data leakage between periods.

Author: ML Engineering Team
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import time

# Local imports
from src.data_loader import load_shm_data
from src.models import EquipmentPricePredictor
from src.evaluation import evaluate_model_comprehensive
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

class ProductionModelTrainer:
    """
    Production-grade model training with Blue Book temporal validation strategies.
    
    Key Features:
    - STRICT temporal validation (Train: ≤2009, Val: 2010-2011, Test: ≥2012)
    - Zero data leakage prevention
    - Blue Book CatBoost configuration with log-price transformation
    - Honest performance metrics aligned with competition standards
    """
    
    def __init__(self, output_dir: Path = Path("./outputs/models")):
        """
        Initialize production trainer.
        
        Args:
            output_dir: Directory to save model artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.models = {}
        self.performance_summary = {}
        
        print("="*80)
        print("PRODUCTION MODEL TRAINER - Blue Book Temporal Validation")
        print("="*80)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load SHM data with comprehensive preprocessing.
        
        Returns:
            Processed DataFrame ready for temporal splitting
        """
        print("\n[STEP 1] Loading and preparing SHM dataset...")
        
        # Load data using the production data loader
        df, validation_report = load_shm_data()
        
        print(f"[DATA] Loaded {len(df):,} total records")
        print(f"[DATA] Dataset span: {df['sales_date'].min()} to {df['sales_date'].max()}")
        
        # Validate we have the required temporal range
        df['year'] = df['sales_date'].dt.year
        year_counts = df['year'].value_counts().sort_index()
        
        print(f"[AUDIT] Year distribution:")
        for year in range(2004, 2013):
            if year in year_counts.index:
                print(f"   {year}: {year_counts[year]:,} records")
            else:
                print(f"   {year}: 0 records")
        
        # Ensure we have sufficient data for all periods
        train_data = df[df['year'] <= 2009]
        val_data = df[(df['year'] >= 2010) & (df['year'] <= 2011)]
        test_data = df[df['year'] >= 2012]
        
        print(f"\n[CRITICAL] Temporal split validation:")
        print(f"   Train (<=2009): {len(train_data):,} records")
        print(f"   Validation (2010-2011): {len(val_data):,} records")
        print(f"   Test (>=2012): {len(test_data):,} records")
        
        if len(train_data) < 10000:
            raise ValueError(f"Insufficient training data: {len(train_data)} records")
        if len(val_data) < 1000:
            raise ValueError(f"Insufficient validation data: {len(val_data)} records")
        if len(test_data) < 1000:
            raise ValueError(f"Insufficient test data: {len(test_data)} records")
        
        print("[OK] Temporal data validation passed")
        
        return df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create STRICT temporal splits following Blue Book strategy.
        
        Args:
            df: Complete dataset
            
        Returns:
            Tuple of (train_df, val_df, test_df) with NO temporal overlap
        """
        print("\n[STEP 2] Creating STRICT temporal splits...")
        
        # STRICT chronological boundaries - NO OVERLAP
        train_df = df[df['sales_date'].dt.year <= 2009].copy()
        val_df = df[(df['sales_date'].dt.year >= 2010) & (df['sales_date'].dt.year <= 2011)].copy()
        test_df = df[df['sales_date'].dt.year >= 2012].copy()
        
        print(f"[SPLIT] Train: {train_df['sales_date'].min()} to {train_df['sales_date'].max()}")
        print(f"[SPLIT] Val: {val_df['sales_date'].min()} to {val_df['sales_date'].max()}")
        print(f"[SPLIT] Test: {test_df['sales_date'].min()} to {test_df['sales_date'].max()}")
        
        # Verify NO temporal leakage
        train_max = train_df['sales_date'].max()
        val_min = val_df['sales_date'].min()
        val_max = val_df['sales_date'].max()
        test_min = test_df['sales_date'].min()
        
        assert train_max < val_min, f"LEAKAGE: Train max {train_max} >= Val min {val_min}"
        assert val_max < test_min, f"LEAKAGE: Val max {val_max} >= Test min {test_min}"
        
        print("[OK] NO TEMPORAL LEAKAGE - Splits are valid")
        
        return train_df, val_df, test_df
    
    def train_baseline_random_forest(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """
        Train baseline Random Forest with proper temporal validation.
        
        Args:
            train_df: Training data (≤2009)
            val_df: Validation data (2010-2011)
            
        Returns:
            Dictionary with training results
        """
        print("\n[STEP 3] Training Baseline Random Forest...")
        
        # Use substantial sample size (Blue Book requirement)
        sample_size = min(50000, len(train_df))
        if len(train_df) > sample_size:
            train_sample = train_df.sample(sample_size, random_state=42)
            print(f"[SAMPLE] Using {sample_size:,} training samples from {len(train_df):,} available")
        else:
            train_sample = train_df
            print(f"[SAMPLE] Using all {len(train_sample):,} training samples")
        
        # Initialize Random Forest model
        rf_model = EquipmentPricePredictor(model_type='random_forest', random_state=42)
        
        # Train using pre-split data (prevents re-splitting)
        start_time = time.time()
        results = rf_model.fit_on_pre_split(train_sample, val_df)
        training_time = time.time() - start_time
        
        results['training_time'] = training_time
        results['model'] = rf_model
        
        # Store model
        self.models['random_forest'] = rf_model
        
        # Display results
        val_metrics = results['validation_metrics']
        print(f"[RESULTS] Random Forest - Training Time: {training_time:.1f}s")
        print(f"   RMSE: ${val_metrics['rmse']:,.0f}")
        print(f"   MAE: ${val_metrics['mae']:,.0f}")
        print(f"   R²: {val_metrics['r2']:.3f}")
        print(f"   RMSLE: {val_metrics.get('rmsle_primary', val_metrics.get('rmsle', 'N/A')):.3f}")
        print(f"   Within 15%: {val_metrics['within_15_pct']:.1f}%")
        
        return results
    
    def train_catboost_blue_book(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """
        Train CatBoost with Blue Book configuration (log-price optimization).
        
        Args:
            train_df: Training data (≤2009)
            val_df: Validation data (2010-2011)
            
        Returns:
            Dictionary with training results
        """
        print("\n[STEP 4] Training CatBoost with Blue Book Configuration...")
        
        # Use substantial sample size for CatBoost
        sample_size = min(50000, len(train_df))
        if len(train_df) > sample_size:
            train_sample = train_df.sample(sample_size, random_state=42)
            print(f"[SAMPLE] Using {sample_size:,} training samples from {len(train_df):,} available")
        else:
            train_sample = train_df
            print(f"[SAMPLE] Using all {len(train_sample):,} training samples")
        
        # Initialize CatBoost with Blue Book configuration
        print("[CONFIG] Blue Book CatBoost: log-price transformation + RMSLE optimization")
        cb_model = EquipmentPricePredictor(model_type='catboost', random_state=42)
        
        # Train using pre-split data
        start_time = time.time()
        results = cb_model.fit_on_pre_split(train_sample, val_df)
        training_time = time.time() - start_time
        
        results['training_time'] = training_time
        results['model'] = cb_model
        
        # Store model
        self.models['catboost'] = cb_model
        
        # Display results
        val_metrics = results['validation_metrics']
        print(f"[RESULTS] CatBoost - Training Time: {training_time:.1f}s")
        print(f"   RMSE: ${val_metrics['rmse']:,.0f}")
        print(f"   MAE: ${val_metrics['mae']:,.0f}")
        print(f"   R²: {val_metrics['r2']:.3f}")
        print(f"   RMSLE: {val_metrics.get('rmsle_primary', val_metrics.get('rmsle', 'N/A')):.3f}")
        print(f"   Within 15%: {val_metrics['within_15_pct']:.1f}%")
        
        return results
    
    def evaluate_on_test_set(self, test_df: pd.DataFrame) -> Dict:
        """
        Final evaluation on held-out test set (≥2012).
        
        Args:
            test_df: Test data (≥2012)
            
        Returns:
            Dictionary with test results for all models
        """
        print("\n[STEP 5] Final Evaluation on Test Set (>=2012)...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n[TEST] Evaluating {model_name}...")
            
            # Make predictions on test set
            predictions = model.predict(test_df)
            actual = test_df['sales_price'].values
            
            # Calculate comprehensive metrics
            test_metrics = self._calculate_honest_metrics(actual, predictions)
            test_results[model_name] = test_metrics
            
            # Display key metrics
            print(f"   RMSE: ${test_metrics['rmse']:,.0f}")
            print(f"   RMSLE: {test_metrics['rmsle']:.3f}")
            print(f"   Within 15%: {test_metrics['within_15_pct']:.1f}%")
            print(f"   Business Ready: {'YES' if test_metrics['within_15_pct'] >= 65 else 'NO'}")
        
        return test_results
    
    def _calculate_honest_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate honest metrics aligned with Blue Book competition standards.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of honest performance metrics
        """
        # Ensure positive predictions
        y_pred_pos = np.maximum(y_pred, 1)
        
        # Standard regression metrics
        mae = mean_absolute_error(y_true, y_pred_pos)
        mse = mean_squared_error(y_true, y_pred_pos)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_pos)
        
        # Business metrics
        mape = np.mean(np.abs((y_true - y_pred_pos) / y_true)) * 100
        
        # RMSLE (primary Blue Book metric)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))
        
        # Within tolerance accuracy (business critical)
        tolerance_10_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.10) * 100
        tolerance_15_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.15) * 100
        tolerance_25_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.25) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'rmsle': rmsle,
            'within_10_pct': tolerance_10_pct,
            'within_15_pct': tolerance_15_pct,
            'within_25_pct': tolerance_25_pct,
            'samples': len(y_true)
        }
    
    def create_model_comparison(self, rf_results: Dict, cb_results: Dict, test_results: Dict) -> pd.DataFrame:
        """
        Create comprehensive model comparison table.
        
        Args:
            rf_results: Random Forest results
            cb_results: CatBoost results
            test_results: Test set results for both models
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        # Random Forest
        val_metrics = rf_results['validation_metrics']
        test_metrics = test_results.get('random_forest', {})
        
        comparison_data.append({
            'Model': 'Random Forest',
            'Type': 'Baseline',
            'Val_RMSE': f"${val_metrics['rmse']:,.0f}",
            'Val_RMSLE': f"{val_metrics.get('rmsle_primary', val_metrics.get('rmsle', 0)):.3f}",
            'Val_Within_15pct': f"{val_metrics['within_15_pct']:.1f}%",
            'Test_RMSE': f"${test_metrics.get('rmse', 0):,.0f}",
            'Test_RMSLE': f"{test_metrics.get('rmsle', 0):.3f}",
            'Test_Within_15pct': f"{test_metrics.get('within_15_pct', 0):.1f}%",
            'Business_Ready': 'YES' if test_metrics.get('within_15_pct', 0) >= 65 else 'NO'
        })
        
        # CatBoost
        val_metrics = cb_results['validation_metrics']
        test_metrics = test_results.get('catboost', {})
        
        comparison_data.append({
            'Model': 'CatBoost',
            'Type': 'Blue Book',
            'Val_RMSE': f"${val_metrics['rmse']:,.0f}",
            'Val_RMSLE': f"{val_metrics.get('rmsle_primary', val_metrics.get('rmsle', 0)):.3f}",
            'Val_Within_15pct': f"{val_metrics['within_15_pct']:.1f}%",
            'Test_RMSE': f"${test_metrics.get('rmse', 0):,.0f}",
            'Test_RMSLE': f"{test_metrics.get('rmsle', 0):.3f}",
            'Test_Within_15pct': f"{test_metrics.get('within_15_pct', 0):.1f}%",
            'Business_Ready': 'YES' if test_metrics.get('within_15_pct', 0) >= 65 else 'NO'
        })
        
        return pd.DataFrame(comparison_data)
    
    def save_model_artifacts(self, rf_results: Dict, cb_results: Dict, test_results: Dict):
        """
        Save all model artifacts and performance metrics.
        
        Args:
            rf_results: Random Forest results
            cb_results: CatBoost results
            test_results: Test set results
        """
        print("\n[STEP 6] Saving Model Artifacts...")
        
        # Save trained models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Random Forest
        rf_path = self.output_dir / f"random_forest_model_{timestamp}.joblib"
        joblib.dump(self.models['random_forest'], rf_path)
        print(f"[SAVE] Random Forest: {rf_path}")
        
        # CatBoost
        cb_path = self.output_dir / f"catboost_model_{timestamp}.joblib"
        joblib.dump(self.models['catboost'], cb_path)
        print(f"[SAVE] CatBoost: {cb_path}")
        
        # Save performance metrics
        performance_data = {
            'timestamp': timestamp,
            'training_strategy': 'Blue Book Temporal Validation',
            'temporal_splits': {
                'train_period': '<=2009',
                'validation_period': '2010-2011',
                'test_period': '>=2012'
            },
            'models': {
                'random_forest': {
                    'validation_metrics': rf_results['validation_metrics'],
                    'test_metrics': test_results.get('random_forest', {}),
                    'training_time': rf_results['training_time'],
                    'model_path': str(rf_path)
                },
                'catboost': {
                    'validation_metrics': cb_results['validation_metrics'],
                    'test_metrics': test_results.get('catboost', {}),
                    'training_time': cb_results['training_time'],
                    'model_path': str(cb_path)
                }
            },
            'business_assessment': self._create_business_assessment(test_results)
        }
        
        metrics_path = self.output_dir / f"production_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        print(f"[SAVE] Performance metrics: {metrics_path}")
        
        # Save comparison table
        comparison_df = self.create_model_comparison(rf_results, cb_results, test_results)
        comparison_path = self.output_dir / f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"[SAVE] Model comparison: {comparison_path}")
        
        return performance_data
    
    def _create_business_assessment(self, test_results: Dict) -> Dict:
        """
        Create business readiness assessment.
        
        Args:
            test_results: Test set results for all models
            
        Returns:
            Business assessment dictionary
        """
        # Find best performing model
        best_model = None
        best_score = 0
        
        for model_name, metrics in test_results.items():
            within_15_pct = metrics.get('within_15_pct', 0)
            if within_15_pct > best_score:
                best_score = within_15_pct
                best_model = model_name
        
        # Business readiness criteria
        business_ready = best_score >= 65  # At least 65% within 15% tolerance
        production_ready = best_score >= 70  # At least 70% within 15% tolerance
        
        assessment = {
            'best_model': best_model,
            'best_within_15_pct': best_score,
            'business_ready': business_ready,
            'production_ready': production_ready,
            'recommendation': self._get_deployment_recommendation(best_score, best_model),
            'risk_assessment': self._assess_deployment_risk(best_score),
            'next_steps': self._get_next_steps(best_score)
        }
        
        return assessment
    
    def _get_deployment_recommendation(self, score: float, model: str) -> str:
        """Get deployment recommendation based on performance."""
        if score >= 75:
            return f"DEPLOY: {model} shows excellent performance ({score:.1f}% within 15%). Ready for production."
        elif score >= 70:
            return f"DEPLOY: {model} meets production criteria ({score:.1f}% within 15%). Monitor closely."
        elif score >= 65:
            return f"PILOT: {model} shows promising results ({score:.1f}% within 15%). Start with limited deployment."
        else:
            return f"DO NOT DEPLOY: {model} performance ({score:.1f}% within 15%) below business requirements."
    
    def _assess_deployment_risk(self, score: float) -> str:
        """Assess deployment risk based on performance."""
        if score >= 75:
            return "LOW: Model performance exceeds business requirements"
        elif score >= 70:
            return "MEDIUM: Model meets minimum requirements but monitor performance"
        elif score >= 65:
            return "HIGH: Model below optimal performance, requires careful monitoring"
        else:
            return "CRITICAL: Model performance insufficient for business use"
    
    def _get_next_steps(self, score: float) -> List[str]:
        """Get recommended next steps based on performance."""
        if score >= 70:
            return [
                "Deploy model to production environment",
                "Set up monitoring and alerting for prediction quality",
                "Collect feedback from business users",
                "Plan for model retraining schedule"
            ]
        else:
            return [
                "Improve feature engineering with domain expertise",
                "Explore additional data sources",
                "Consider ensemble methods",
                "Retrain with more recent data",
                "Collaborate with business experts for better features"
            ]
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete production model training pipeline.
        
        Returns:
            Dictionary with all results and artifacts
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and prepare data
            df = self.load_and_prepare_data()
            
            # Step 2: Create temporal splits
            train_df, val_df, test_df = self.create_temporal_splits(df)
            
            # Step 3: Train Random Forest baseline
            rf_results = self.train_baseline_random_forest(train_df, val_df)
            
            # Step 4: Train CatBoost Blue Book model
            cb_results = self.train_catboost_blue_book(train_df, val_df)
            
            # Step 5: Evaluate on test set
            test_results = self.evaluate_on_test_set(test_df)
            
            # Step 6: Save artifacts
            performance_data = self.save_model_artifacts(rf_results, cb_results, test_results)
            
            total_time = time.time() - start_time
            
            # Final summary
            self.print_final_summary(performance_data, total_time)
            
            return performance_data
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            raise
    
    def print_final_summary(self, performance_data: Dict, total_time: float):
        """
        Print final summary of training results.
        
        Args:
            performance_data: Complete performance data
            total_time: Total pipeline runtime
        """
        print("\n" + "="*80)
        print("PRODUCTION MODEL TRAINING COMPLETE")
        print("="*80)
        
        print(f"Total Runtime: {total_time:.1f} seconds")
        print(f"Strategy: Blue Book Temporal Validation (ZERO data leakage)")
        print(f"Temporal Splits: Train <=2009, Val 2010-2011, Test >=2012")
        
        # Business assessment
        assessment = performance_data['business_assessment']
        print(f"\n[BUSINESS ASSESSMENT]")
        print(f"Best Model: {assessment['best_model']}")
        print(f"Best Performance: {assessment['best_within_15_pct']:.1f}% within 15% tolerance")
        print(f"Business Ready: {assessment['business_ready']}")
        print(f"Production Ready: {assessment['production_ready']}")
        print(f"Risk Level: {assessment['risk_assessment']}")
        
        print(f"\n[RECOMMENDATION]")
        print(f"{assessment['recommendation']}")
        
        print(f"\n[ARTIFACTS SAVED]")
        for model_name, model_data in performance_data['models'].items():
            print(f"   {model_name}: {model_data['model_path']}")
        
        print("="*80)


def main():
    """Main execution function."""
    print("Starting Production Model Training Pipeline...")
    
    # Initialize trainer
    trainer = ProductionModelTrainer()
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    print("\nProduction model training completed successfully!")
    return results


if __name__ == "__main__":
    # Execute the production training pipeline
    results = main()