#!/usr/bin/env python3
"""
HONEST MODEL TRAINING - DATA LEAKAGE FIXED
===========================================

Fixed training script that removes data leakage and produces honest performance metrics.
Key fix: Remove log1p_price feature which was derived from the target variable.

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

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

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

def remove_data_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that cause data leakage.
    
    Args:
        df: DataFrame with potential leakage features
        
    Returns:
        DataFrame with leakage features removed
    """
    # Features that are derived from the target or cause leakage
    leakage_features = [
        'log1p_price',  # Direct transformation of target
        # Add any other features found to be causing leakage
    ]
    
    df_clean = df.copy()
    removed_features = []
    
    for feature in leakage_features:
        if feature in df_clean.columns:
            df_clean = df_clean.drop(feature, axis=1)
            removed_features.append(feature)
    
    if removed_features:
        print(f"[CRITICAL] Removed data leakage features: {removed_features}")
    
    return df_clean

def preprocess_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Honest preprocessing that prevents data leakage.
    
    Args:
        train_df: Training data
        val_df: Validation data  
        test_df: Test data
        
    Returns:
        Tuple of processed DataFrames
    """
    print("\n[PREPROCESSING] Honest leakage-free preprocessing...")
    
    # Remove data leakage features first
    train_df = remove_data_leakage_features(train_df)
    val_df = remove_data_leakage_features(val_df)
    test_df = remove_data_leakage_features(test_df)
    
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
    
    # Use substantial sample size (Blue Book requirement)
    sample_size = min(50000, len(train_df))
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
    
    # Train Random Forest with reasonable parameters to prevent overfitting
    start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,  # Reasonable depth
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
        'model_type': 'RandomForest',
        'training_time': training_time,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_)),
        'sample_size': len(train_sample)
    }

def train_catboost_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Train CatBoost with Blue Book configuration.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        
    Returns:
        Dictionary with results
    """
    if not CATBOOST_AVAILABLE:
        print("[SKIP] CatBoost not available")
        return None
    
    print("\n[MODEL] Training CatBoost with Blue Book Configuration...")
    
    # Use substantial sample size
    sample_size = min(50000, len(train_df))
    if len(train_df) > sample_size:
        train_sample = train_df.sample(sample_size, random_state=42)
        print(f"[SAMPLE] Using {sample_size:,} training samples")
    else:
        train_sample = train_df
        print(f"[SAMPLE] Using all {len(train_sample):,} training samples")
    
    # Remove leakage features
    train_sample = remove_data_leakage_features(train_sample)
    val_df = remove_data_leakage_features(val_df)
    test_df = remove_data_leakage_features(test_df)
    
    # Prepare features (CatBoost handles categoricals natively)
    target = 'sales_price'
    exclude_cols = ['sales_id', 'machine_id', 'sales_date', target]
    feature_cols = [col for col in train_sample.columns if col not in exclude_cols]
    
    X_train = train_sample[feature_cols]
    y_train = train_sample[target]
    X_val = val_df[feature_cols]
    y_val = val_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]
    
    # Fill missing values
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            fill_value = X_train[col].mode().iloc[0] if len(X_train[col].mode()) > 0 else 'Unknown'
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
        else:
            fill_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
    
    # Identify categorical features
    categorical_features = [i for i, col in enumerate(feature_cols) if X_train[col].dtype == 'object']
    
    print(f"[FEATURES] Using {len(feature_cols)} features, {len(categorical_features)} categorical")
    print("[CONFIG] Blue Book CatBoost: RMSLE optimization")
    
    # Blue Book optimized CatBoost for log-price (RMSLE optimization)
    start_time = time.time()
    
    # Apply log transformation to target for RMSLE optimization
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    cb_model = CatBoostRegressor(
        iterations=500,  # Reduced to prevent overfitting
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',  # RMSE on log-price = RMSLE on original price
        eval_metric='RMSE',
        bootstrap_type='Bayesian',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    
    cb_model.fit(
        X_train, y_train_log,
        cat_features=categorical_features,
        eval_set=(X_val, y_val_log),
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    # Make predictions and transform back from log space
    train_pred_log = cb_model.predict(X_train)
    val_pred_log = cb_model.predict(X_val)
    test_pred_log = cb_model.predict(X_test)
    
    train_pred = np.expm1(train_pred_log)  # Inverse of log1p
    val_pred = np.expm1(val_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    # Calculate metrics
    train_metrics = calculate_honest_metrics(y_train, train_pred)
    val_metrics = calculate_honest_metrics(y_val, val_pred)
    test_metrics = calculate_honest_metrics(y_test, test_pred)
    
    print(f"[RESULTS] CatBoost - Training Time: {training_time:.1f}s")
    print(f"   Validation RMSE: ${val_metrics['rmse']:,.0f}")
    print(f"   Validation R2: {val_metrics['r2']:.3f}")
    print(f"   Validation RMSLE: {val_metrics['rmsle']:.3f}")
    print(f"   Validation Within 15%: {val_metrics['within_15_pct']:.1f}%")
    
    print(f"   Test RMSE: ${test_metrics['rmse']:,.0f}")
    print(f"   Test R2: {test_metrics['r2']:.3f}")
    print(f"   Test RMSLE: {test_metrics['rmsle']:.3f}")
    print(f"   Test Within 15%: {test_metrics['within_15_pct']:.1f}%")
    
    return {
        'model': cb_model,
        'model_type': 'CatBoost',
        'training_time': training_time,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': dict(zip(feature_cols, cb_model.feature_importances_)),
        'sample_size': len(train_sample),
        'log_target_used': True
    }

def create_business_assessment(results_dict: Dict) -> Dict:
    """
    Create comprehensive business readiness assessment.
    
    Args:
        results_dict: Dictionary of model results
        
    Returns:
        Business assessment
    """
    assessments = {}
    
    for model_name, results in results_dict.items():
        if results is None:
            continue
            
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
        
        assessments[model_name] = {
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
    
    # Find best model
    best_model = None
    best_score = 0
    for model_name, assessment in assessments.items():
        if assessment['test_within_15_pct'] > best_score:
            best_score = assessment['test_within_15_pct']
            best_model = model_name
    
    return {
        'individual_assessments': assessments,
        'best_model': best_model,
        'best_score': best_score
    }

def save_results(results_dict: Dict, assessment: Dict):
    """
    Save models and results.
    
    Args:
        results_dict: Dictionary of model results
        assessment: Business assessment
    """
    output_dir = Path("./outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_models = {}
    
    # Save models
    for model_name, results in results_dict.items():
        if results is None:
            continue
            
        model_path = output_dir / f"honest_{model_name.lower()}_{timestamp}.joblib"
        joblib.dump(results['model'], model_path)
        saved_models[model_name] = str(model_path)
        print(f"[SAVE] {model_name}: {model_path}")
    
    # Save comprehensive metrics
    metrics_data = {
        'timestamp': timestamp,
        'strategy': 'Honest Temporal Validation - Data Leakage Fixed',
        'leakage_features_removed': ['log1p_price'],
        'models': {
            name: {
                'model_type': results['model_type'],
                'sample_size': results['sample_size'],
                'training_time': results['training_time'],
                'validation_metrics': results['validation_metrics'],
                'test_metrics': results['test_metrics'],
                'model_path': saved_models.get(name)
            } for name, results in results_dict.items() if results is not None
        },
        'business_assessment': assessment
    }
    
    metrics_path = output_dir / f"honest_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"[SAVE] Metrics: {metrics_path}")
    
    return saved_models, metrics_path

def main():
    """Main execution function."""
    print("Starting HONEST Model Training - Data Leakage Fixed")
    print("="*60)
    
    try:
        # Load data
        print("\n[STEP 1] Loading SHM dataset...")
        df, validation_report = load_shm_data()
        print(f"[DATA] Loaded {len(df):,} records")
        
        # Create temporal splits
        print("\n[STEP 2] Creating temporal splits...")
        train_df, val_df, test_df = create_strict_temporal_split(df)
        
        # Train models
        results_dict = {}
        
        print("\n[STEP 3] Training Random Forest...")
        rf_results = train_random_forest_model(train_df, val_df, test_df)
        results_dict['RandomForest'] = rf_results
        
        print("\n[STEP 4] Training CatBoost...")
        cb_results = train_catboost_model(train_df, val_df, test_df)
        if cb_results:
            results_dict['CatBoost'] = cb_results
        
        # Business assessment
        print("\n[STEP 5] Business assessment...")
        assessment = create_business_assessment(results_dict)
        
        # Save results
        print("\n[STEP 6] Saving results...")
        saved_models, metrics_path = save_results(results_dict, assessment)
        
        # Final summary
        print("\n" + "="*60)
        print("HONEST TRAINING COMPLETE - DATA LEAKAGE FIXED")
        print("="*60)
        print(f"Key Fix: Removed log1p_price feature (derived from target)")
        print(f"Strategy: Strict temporal validation")
        print(f"")
        
        for model_name, results in results_dict.items():
            if results is None:
                continue
                
            print(f"{model_name.upper()} RESULTS:")
            print(f"  Sample Size: {results['sample_size']:,}")
            print(f"  Training Time: {results['training_time']:.1f}s")
            print(f"  Validation RMSE: ${results['validation_metrics']['rmse']:,.0f}")
            print(f"  Validation RMSLE: {results['validation_metrics']['rmsle']:.3f}")
            print(f"  Validation Within 15%: {results['validation_metrics']['within_15_pct']:.1f}%")
            print(f"  Test RMSE: ${results['test_metrics']['rmse']:,.0f}")
            print(f"  Test RMSLE: {results['test_metrics']['rmsle']:.3f}")
            print(f"  Test Within 15%: {results['test_metrics']['within_15_pct']:.1f}%")
            print(f"")
        
        print(f"BUSINESS ASSESSMENT:")
        print(f"  Best Model: {assessment['best_model']}")
        print(f"  Best Score: {assessment['best_score']:.1f}% within 15%")
        
        best_assessment = assessment['individual_assessments'][assessment['best_model']]
        print(f"  Business Ready: {best_assessment['business_ready']}")
        print(f"  Production Ready: {best_assessment['production_ready']}")
        print(f"  Risk Level: {best_assessment['risk_level']}")
        print(f"  Recommendation: {best_assessment['recommendation']}")
        print("="*60)
        
        return results_dict, assessment
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

if __name__ == "__main__":
    # Execute the honest training pipeline
    results_dict, assessment = main()