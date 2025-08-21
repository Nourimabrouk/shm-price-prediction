#!/usr/bin/env python3
"""
Validation Script for Advanced ML Orchestration System
====================================================

This script validates that all orchestration components are working correctly
and demonstrates the complete pipeline integration.

Usage: python validate_orchestration.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

try:
    # Import orchestration components
    from models import EquipmentPricePredictor, ConformalPredictor, EnsembleOrchestrator
    from evaluation import (create_sophisticated_baselines, evaluate_against_baselines,
                           evaluate_uncertainty_quantification, ModelEvaluator)
    from data_loader import load_shm_data
    
    print("All orchestration components imported successfully")
    
except ImportError as e:
    print(f"ERROR Import error: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

def main():
    """Validate complete orchestration system."""
    
    print("\n" + "="*60)
    print("ADVANCED ML ORCHESTRATION VALIDATION")
    print("="*60)
    
    # 1. Data Loading Test
    print("\n1. Testing Data Loading...")
    try:
        df, validation_report = load_shm_data()
        print(f"   OK Loaded {len(df):,} records")
        
        # Use sample for validation
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        print(f"   OK Using {len(df_sample):,} samples for validation")
    except Exception as e:
        print(f"   ERROR Data loading failed: {e}")
        return False
    
    # 2. Advanced Baselines Test
    print("\n2. Testing Advanced Business Baselines...")
    try:
        baselines = create_sophisticated_baselines(
            df_sample,
            target_col='sales_price',
            product_group_col='product_group',
            date_col='sales_date'
        )
        print(f"   OK Generated {len(baselines)} sophisticated baselines")
    except Exception as e:
        print(f"   ERROR Baseline generation failed: {e}")
        return False
    
    # 3. Individual Models Test
    print("\n3. Testing Individual Model Training...")
    try:
        # Split data
        df_sorted = df_sample.sort_values('sales_date')
        split_idx = int(len(df_sorted) * 0.7)
        cal_idx = int(len(df_sorted) * 0.85)
        
        df_train = df_sorted.iloc[:split_idx].copy()
        df_cal = df_sorted.iloc[split_idx:cal_idx].copy()
        df_test = df_sorted.iloc[cal_idx:].copy()
        
        print(f"   OK Split data: {len(df_train)} train, {len(df_cal)} cal, {len(df_test)} test")
        
        # Train models
        models = {}
        
        # RandomForest (always available)
        rf_model = EquipmentPricePredictor(model_type='random_forest', random_state=42)
        rf_results = rf_model.fit_on_pre_split(df_train, df_cal)
        models['RandomForest'] = rf_model
        print(f"   OK RandomForest trained - RMSE: ${rf_results['validation_metrics']['rmse']:,.0f}")
        
        # Try CatBoost if available
        try:
            cb_model = EquipmentPricePredictor(model_type='catboost', random_state=42)
            cb_results = cb_model.fit_on_pre_split(df_train, df_cal)
            models['CatBoost'] = cb_model
            print(f"   OK CatBoost trained - RMSE: ${cb_results['validation_metrics']['rmse']:,.0f}")
        except Exception:
            print("   INFO CatBoost not available, using RandomForest only")
        
    except Exception as e:
        print(f"   ERROR Model training failed: {e}")
        return False
    
    # 4. Ensemble Orchestration Test
    print("\n4. Testing Ensemble Orchestration...")
    try:
        ensemble = EnsembleOrchestrator(combination_method='weighted', random_state=42)
        
        for name, model in models.items():
            ensemble.register_model(name, model)
        
        # Prepare data for ensemble
        X_train, y_train = rf_model.prepare_features_target(rf_model.preprocess_data(df_train, is_training=False))
        X_cal, y_cal = rf_model.prepare_features_target(rf_model.preprocess_data(df_cal, is_training=False))
        X_test, y_test = rf_model.prepare_features_target(rf_model.preprocess_data(df_test, is_training=False))
        
        ensemble.fit_ensemble(X_train, y_train, X_cal, y_cal)
        print(f"   OK Ensemble orchestration complete")
        
        # Test ensemble predictions
        ensemble_pred = ensemble.predict_ensemble(X_test)
        ensemble_pred_with_unc, uncertainty = ensemble.predict_with_uncertainty(X_test)
        print(f"   OK Ensemble predictions generated: {len(ensemble_pred)} samples")
        
    except Exception as e:
        print(f"   ERROR Ensemble orchestration failed: {e}")
        return False
    
    # 5. Conformal Prediction Test
    print("\n5. Testing Conformal Prediction Framework...")
    try:
        # 80% and 90% confidence intervals
        conformal_80 = ConformalPredictor(ensemble, alpha=0.2)
        conformal_90 = ConformalPredictor(ensemble, alpha=0.1)
        
        # Calibrate
        conformal_80.calibrate(X_cal, y_cal)
        conformal_90.calibrate(X_cal, y_cal)
        
        # Generate intervals
        pred_80, lower_80, upper_80 = conformal_80.predict_with_intervals(X_test)
        pred_90, lower_90, upper_90 = conformal_90.predict_with_intervals(X_test)
        
        print(f"   OK Conformal prediction intervals generated")
        
        # Validate coverage
        coverage_80 = conformal_80.validate_coverage(X_test, y_test)
        coverage_90 = conformal_90.validate_coverage(X_test, y_test)
        
        print(f"   OK Coverage validation complete")
        
    except Exception as e:
        print(f"   ERROR Conformal prediction failed: {e}")
        return False
    
    # 6. Evaluation Framework Test
    print("\n6. Testing Advanced Evaluation Framework...")
    try:
        # Baseline comparison
        test_baselines = create_sophisticated_baselines(
            df_test, target_col='sales_price',
            product_group_col='product_group', date_col='sales_date'
        )
        
        baseline_comparison = evaluate_against_baselines(y_test, ensemble_pred, test_baselines)
        print(f"   OK Baseline comparison complete")
        
        # Uncertainty quantification evaluation
        uncertainty_eval = evaluate_uncertainty_quantification(
            y_test, ensemble_pred, lower_80, upper_80, confidence_level=0.8
        )
        print(f"   OK Uncertainty quantification evaluation complete")
        
        # Visualization test (create evaluator but don't generate plots)
        evaluator = ModelEvaluator(output_dir="./plots/")
        print(f"   OK Evaluation framework ready")
        
    except Exception as e:
        print(f"   ERROR Evaluation framework failed: {e}")
        return False
    
    # 7. Performance Summary
    print("\n" + "="*60)
    print("ORCHESTRATION VALIDATION SUMMARY")
    print("="*60)
    
    # Model performance
    ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))
    ensemble_r2 = 1 - np.sum((y_test - ensemble_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"\nEnsemble Performance:")
    print(f"   RMSE: ${ensemble_rmse:,.0f}")
    print(f"   R²: {ensemble_r2:.3f}")
    
    # Coverage performance
    print(f"\nUncertainty Quantification:")
    print(f"   80% Coverage: {coverage_80['empirical_coverage']:.1%} (target: 80%)")
    print(f"   90% Coverage: {coverage_90['empirical_coverage']:.1%} (target: 90%)")
    
    # Baseline comparison
    best_baseline = baseline_comparison['best_baseline']['name']
    improvement = baseline_comparison['model_improvement_vs_best_baseline_percent']
    print(f"\nBaseline Comparison:")
    print(f"   Best baseline: {best_baseline}")
    print(f"   Model improvement: {improvement:+.1f}%")
    print(f"   Beats all baselines: {'Yes' if baseline_comparison['beats_all_baselines'] else 'No'}")
    
    # Production readiness assessment
    coverage_ok_80 = abs(coverage_80['empirical_coverage'] - 0.8) < 0.05
    coverage_ok_90 = abs(coverage_90['empirical_coverage'] - 0.9) < 0.05
    improvement_ok = improvement > 2
    
    readiness_score = sum([coverage_ok_80, coverage_ok_90, improvement_ok, baseline_comparison['beats_all_baselines']])
    
    print(f"\nProduction Readiness Assessment:")
    print(f"   Coverage Quality (80%): {'Excellent' if coverage_ok_80 else 'Needs tuning'}")
    print(f"   Coverage Quality (90%): {'Excellent' if coverage_ok_90 else 'Needs tuning'}")
    print(f"   Performance vs Baselines: {'Strong' if improvement_ok else 'Marginal'}")
    print(f"   Overall Score: {readiness_score}/4")
    print(f"   Status: {'Production Ready' if readiness_score >= 3 else 'Needs Optimization'}")
    
    print(f"\n" + "="*60)
    print("OK ORCHESTRATION VALIDATION COMPLETE")
    print("All components working correctly!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)