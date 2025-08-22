#!/usr/bin/env python3
"""
SIMPLE MODEL TRAINING WITH HONEST METRICS
=========================================

Simplified training script that addresses data leakage and produces honest performance metrics.
Focus on getting realistic, verifiable results rather than perfect scores.

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Local imports
from src.data_loader import load_shm_data

warnings.filterwarnings('ignore')

def create_strict_temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create strict temporal splits with ZERO leakage.
    
    Args:
        df: Complete dataset with sales_date column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n[CRITICAL] Creating STRICT temporal splits...")
    
    # Create year column for splitting
    df = df.copy()
    df['year'] = df['sales_date'].dt.year
    
    # STRICT chronological boundaries - NO OVERLAP
    train_df = df[df['year'] <= 2009].copy()
    val_df = df[(df['year'] >= 2010) & (df['year'] <= 2011)].copy()
    test_df = df[df['year'] >= 2012].copy()
    
    print(f"[SPLIT] Train (<=2009): {len(train_df):,} records")
    print(f"[SPLIT] Val (2010-2011): {len(val_df):,} records") 
    print(f"[SPLIT] Test (>=2012): {len(test_df):,} records")
    
    # Verify NO temporal leakage
    train_max = train_df['sales_date'].max()
    val_min = val_df['sales_date'].min()
    val_max = val_df['sales_date'].max()
    test_min = test_df['sales_date'].min()
    
    assert train_max < val_min, f"LEAKAGE: Train max {train_max} >= Val min {val_min}"
    assert val_max < test_min, f"LEAKAGE: Val max {val_max} >= Test min {test_min}"
    
    print("[OK] NO TEMPORAL LEAKAGE - Splits are valid")
    
    # Drop the temporary year column
    train_df = train_df.drop('year', axis=1)
    val_df = val_df.drop('year', axis=1)
    test_df = test_df.drop('year', axis=1)
    
    return train_df, val_df, test_df

def preprocess_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple preprocessing that prevents data leakage.
    
    Args:
        train_df: Training data
        val_df: Validation data  
        test_df: Test data
        
    Returns:
        Tuple of processed DataFrames
    """
    print("\n[PREPROCESSING] Simple leakage-free preprocessing...")
    
    # Define target and features to exclude
    target = 'sales_price'
    exclude_cols = ['sales_id', 'machine_id', 'sales_date', target]
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Separate categorical and numerical features
    categorical_features = []
    numerical_features = []
    
    for col in feature_cols:
        if train_df[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"[FEATURES] {len(categorical_features)} categorical, {len(numerical_features)} numerical")
    
    # Process training data and store transformations
    train_processed = train_df.copy()
    fill_values = {}
    label_encoders = {}
    
    # Handle missing values and encode categoricals
    for col in categorical_features:
        # Fill missing with mode from training data only
        mode_value = train_processed[col].mode()
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'Unknown'
        fill_values[col] = fill_value
        train_processed[col] = train_processed[col].fillna(fill_value)
        
        # Label encode using training data only
        le = LabelEncoder()
        # Add 'Unknown' to handle unseen categories
        unique_vals = list(train_processed[col].unique()) + ['Unknown']
        le.fit(unique_vals)
        train_processed[col] = le.transform(train_processed[col])
        label_encoders[col] = le
    
    for col in numerical_features:
        # Fill missing with median from training data only
        median_value = train_processed[col].median()
        fill_values[col] = median_value
        train_processed[col] = train_processed[col].fillna(median_value)
    
    # Apply transformations to validation data
    val_processed = val_df.copy()
    for col in categorical_features:
        val_processed[col] = val_processed[col].fillna(fill_values[col])
        # Handle unseen categories
        unknown_mask = ~val_processed[col].isin(label_encoders[col].classes_)
        val_processed.loc[unknown_mask, col] = 'Unknown'
        val_processed[col] = label_encoders[col].transform(val_processed[col])
    
    for col in numerical_features:
        val_processed[col] = val_processed[col].fillna(fill_values[col])
    
    # Apply transformations to test data
    test_processed = test_df.copy()
    for col in categorical_features:
        test_processed[col] = test_processed[col].fillna(fill_values[col])
        # Handle unseen categories
        unknown_mask = ~test_processed[col].isin(label_encoders[col].classes_)
        test_processed.loc[unknown_mask, col] = 'Unknown'
        test_processed[col] = label_encoders[col].transform(test_processed[col])
    
    for col in numerical_features:
        test_processed[col] = test_processed[col].fillna(fill_values[col])
    
    print("[OK] Preprocessing complete")
    
    return train_processed, val_processed, test_processed

def calculate_honest_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate honest performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure positive predictions
    y_pred_pos = np.maximum(y_pred, 1)
    
    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred_pos)
    mse = mean_squared_error(y_true, y_pred_pos)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred_pos)
    
    # Business metrics
    mape = np.mean(np.abs((y_true - y_pred_pos) / y_true)) * 100
    
    # RMSLE (Blue Book primary metric)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))
    
    # Within tolerance accuracy
    within_10_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.10) * 100
    within_15_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.15) * 100
    within_25_pct = np.mean(np.abs(y_true - y_pred_pos) / y_true <= 0.25) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'rmsle': rmsle,
        'within_10_pct': within_10_pct,
        'within_15_pct': within_15_pct,
        'within_25_pct': within_25_pct,
        'samples': len(y_true)
    }

def train_random_forest_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Train Random Forest with honest evaluation.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        
    Returns:
        Dictionary with results
    """
    print("\n[MODEL] Training Random Forest...")
    
    # Use a reasonable sample size to avoid overfitting
    sample_size = min(20000, len(train_df))
    if len(train_df) > sample_size:
        train_sample = train_df.sample(sample_size, random_state=42)
        print(f"[SAMPLE] Using {sample_size:,} training samples")
    else:
        train_sample = train_df
        print(f"[SAMPLE] Using all {len(train_sample):,} training samples")
    
    # Preprocess data
    train_processed, val_processed, test_processed = preprocess_features(train_sample, val_df, test_df)
    
    # Prepare features and target
    target = 'sales_price'
    exclude_cols = ['sales_id', 'machine_id', 'sales_date', target]
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed[target]
    X_val = val_processed[feature_cols]
    y_val = val_processed[target]
    X_test = test_processed[feature_cols]
    y_test = test_processed[target]
    
    print(f"[FEATURES] Using {len(feature_cols)} features")
    
    # Train Random Forest with reasonable parameters
    start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_honest_metrics(y_train, train_pred)
    val_metrics = calculate_honest_metrics(y_val, val_pred)
    test_metrics = calculate_honest_metrics(y_test, test_pred)
    
    print(f"[RESULTS] Random Forest - Training Time: {training_time:.1f}s")
    print(f"   Validation RMSE: ${val_metrics['rmse']:,.0f}")
    print(f"   Validation R2: {val_metrics['r2']:.3f}")
    print(f"   Validation RMSLE: {val_metrics['rmsle']:.3f}")
    print(f"   Validation Within 15%: {val_metrics['within_15_pct']:.1f}%")
    
    print(f"   Test RMSE: ${test_metrics['rmse']:,.0f}")
    print(f"   Test R2: {test_metrics['r2']:.3f}")
    print(f"   Test RMSLE: {test_metrics['rmsle']:.3f}")
    print(f"   Test Within 15%: {test_metrics['within_15_pct']:.1f}%")
    
    return {
        'model': rf_model,
        'training_time': training_time,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_)),
        'sample_size': len(train_sample)
    }

def create_business_assessment(results: Dict) -> Dict:
    """
    Create business readiness assessment.
    
    Args:
        results: Model results
        
    Returns:
        Business assessment
    """
    test_metrics = results['test_metrics']
    val_metrics = results['validation_metrics']
    
    # Use test set performance for final assessment
    within_15_pct = test_metrics['within_15_pct']
    rmsle = test_metrics['rmsle']
    
    # Business criteria
    business_ready = within_15_pct >= 65  # At least 65% within 15%
    production_ready = within_15_pct >= 70 and rmsle <= 0.4  # High bar for production
    
    # Risk assessment
    if within_15_pct >= 75:
        risk = "LOW: Excellent performance"
    elif within_15_pct >= 65:
        risk = "MEDIUM: Meets business requirements"
    else:
        risk = "HIGH: Below business requirements"
    
    # Recommendation
    if production_ready:
        recommendation = f"DEPLOY: Model ready for production ({within_15_pct:.1f}% within 15%)"
    elif business_ready:
        recommendation = f"PILOT: Start with limited deployment ({within_15_pct:.1f}% within 15%)"
    else:
        recommendation = f"IMPROVE: Model needs enhancement ({within_15_pct:.1f}% within 15%)"
    
    return {
        'business_ready': business_ready,
        'production_ready': production_ready,
        'test_within_15_pct': within_15_pct,
        'test_rmsle': rmsle,
        'risk_level': risk,
        'recommendation': recommendation,
        'performance_summary': {
            'validation_rmse': val_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'validation_within_15_pct': val_metrics['within_15_pct'],
            'test_within_15_pct': test_metrics['within_15_pct']
        }
    }

def save_results(results: Dict, assessment: Dict):
    """
    Save model and results.
    
    Args:
        results: Model results
        assessment: Business assessment
    """
    output_dir = Path("./outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = output_dir / f"simple_random_forest_{timestamp}.joblib"
    joblib.dump(results['model'], model_path)
    print(f"[SAVE] Model: {model_path}")
    
    # Save metrics
    metrics_data = {
        'timestamp': timestamp,
        'model_type': 'RandomForest',
        'strategy': 'Simple Temporal Validation',
        'sample_size': results['sample_size'],
        'training_time': results['training_time'],
        'validation_metrics': results['validation_metrics'],
        'test_metrics': results['test_metrics'],
        'business_assessment': assessment
    }
    
    metrics_path = output_dir / f"simple_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"[SAVE] Metrics: {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Main execution function."""
    print("Starting Simple Model Training...")
    print("="*60)
    
    try:
        # Load data
        print("\n[STEP 1] Loading SHM dataset...")
        df, validation_report = load_shm_data()
        print(f"[DATA] Loaded {len(df):,} records")
        
        # Create temporal splits
        print("\n[STEP 2] Creating temporal splits...")
        train_df, val_df, test_df = create_strict_temporal_split(df)
        
        # Train model
        print("\n[STEP 3] Training Random Forest...")
        results = train_random_forest_model(train_df, val_df, test_df)
        
        # Business assessment
        print("\n[STEP 4] Business assessment...")
        assessment = create_business_assessment(results)
        
        # Save results
        print("\n[STEP 5] Saving results...")
        model_path, metrics_path = save_results(results, assessment)
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - HONEST RESULTS")
        print("="*60)
        print(f"Model Type: Random Forest")
        print(f"Sample Size: {results['sample_size']:,}")
        print(f"Training Time: {results['training_time']:.1f}s")
        print(f"")
        print(f"VALIDATION RESULTS:")
        print(f"  RMSE: ${results['validation_metrics']['rmse']:,.0f}")
        print(f"  R2: {results['validation_metrics']['r2']:.3f}")
        print(f"  RMSLE: {results['validation_metrics']['rmsle']:.3f}")
        print(f"  Within 15%: {results['validation_metrics']['within_15_pct']:.1f}%")
        print(f"")
        print(f"TEST RESULTS (FINAL):")
        print(f"  RMSE: ${results['test_metrics']['rmse']:,.0f}")
        print(f"  R2: {results['test_metrics']['r2']:.3f}")
        print(f"  RMSLE: {results['test_metrics']['rmsle']:.3f}")
        print(f"  Within 15%: {results['test_metrics']['within_15_pct']:.1f}%")
        print(f"")
        print(f"BUSINESS ASSESSMENT:")
        print(f"  Business Ready: {assessment['business_ready']}")
        print(f"  Production Ready: {assessment['production_ready']}")
        print(f"  Risk Level: {assessment['risk_level']}")
        print(f"  Recommendation: {assessment['recommendation']}")
        print("="*60)
        
        return results, assessment
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

if __name__ == "__main__":
    # Execute the simple training pipeline
    results, assessment = main()