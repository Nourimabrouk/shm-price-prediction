"""Quick training script for testing the pipeline with a smaller sample."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import time
import warnings
from datetime import datetime
from pathlib import Path
import logging
import json
import sys

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import CatBoost with fallback
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import our modules
from src.data import load_and_validate_data
from src.features import LeakProofFeatureEngineer
from src.metrics import RegressionMetrics, save_metrics

warnings.filterwarnings('ignore')

# Fixed random state
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def quick_train(sample_size: int = 50000):
    """Quick training with reduced sample size."""
    logger.info(f"Starting quick training with sample size: {sample_size}")
    
    # Load and validate data
    df, validation_results = load_and_validate_data("data/raw/Bit_SHM_data.csv", "artifacts", RANDOM_STATE)
    
    # Load temporal splits
    splits_dir = Path("artifacts") / "splits"
    train_idx = pd.read_csv(splits_dir / "train_idx.csv")['index'].values
    val_idx = pd.read_csv(splits_dir / "val_idx.csv")['index'].values
    test_idx = pd.read_csv(splits_dir / "test_idx.csv")['index'].values
    
    # Sample for faster training
    if sample_size and sample_size < len(train_idx):
        np.random.seed(RANDOM_STATE)
        sampled_train_idx = np.random.choice(train_idx, size=sample_size, replace=False)
        train_idx = sampled_train_idx
        logger.info(f"Sampled {sample_size} training samples")
    
    # Prepare data
    target_column = 'Sales Price'
    target = df[target_column]
    
    # Initialize feature engineer and metrics
    feature_engineer = LeakProofFeatureEngineer(random_state=RANDOM_STATE)
    metrics_calculator = RegressionMetrics()
    
    # Engineer features
    logger.info("Engineering features...")
    train_features = feature_engineer.fit_transform(df.iloc[train_idx], target.iloc[train_idx])
    val_features = feature_engineer.transform(df.iloc[val_idx])
    test_features = feature_engineer.transform(df.iloc[test_idx])
    
    # Targets
    y_train = target.iloc[train_idx].reset_index(drop=True)
    y_val = target.iloc[val_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)
    
    # Reset indices
    train_features = train_features.reset_index(drop=True)
    val_features = val_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    
    logger.info(f"Training set: {len(train_features)} samples, {train_features.shape[1]} features")
    logger.info(f"Validation set: {len(val_features)} samples")
    logger.info(f"Test set: {len(test_features)} samples")
    
    # Train models
    models = {}
    results = {}
    
    # 1. Linear Regression baseline
    logger.info("Training Linear Regression...")
    lr = LinearRegression()
    start_time = time.time()
    lr.fit(train_features, y_train)
    lr_time = time.time() - start_time
    
    lr_val_pred = lr.predict(val_features)
    lr_test_pred = lr.predict(test_features)
    
    lr_val_metrics = metrics_calculator.evaluate_model(y_val, lr_val_pred, "LinearRegression_Val")
    lr_test_metrics = metrics_calculator.evaluate_model(y_test, lr_test_pred, "LinearRegression_Test")
    
    models['LinearRegression'] = lr
    results['LinearRegression'] = {
        'training_time': lr_time,
        'val_metrics': lr_val_metrics,
        'test_metrics': lr_test_metrics
    }
    
    logger.info(f"LinearRegression - Val R²: {lr_val_metrics['basic_metrics']['r2']:.4f}, "
               f"Test R²: {lr_test_metrics['basic_metrics']['r2']:.4f}")
    
    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start_time = time.time()
    rf.fit(train_features, y_train)
    rf_time = time.time() - start_time
    
    rf_val_pred = rf.predict(val_features)
    rf_test_pred = rf.predict(test_features)
    
    rf_val_metrics = metrics_calculator.evaluate_model(y_val, rf_val_pred, "RandomForest_Val")
    rf_test_metrics = metrics_calculator.evaluate_model(y_test, rf_test_pred, "RandomForest_Test")
    
    models['RandomForest'] = rf
    results['RandomForest'] = {
        'training_time': rf_time,
        'val_metrics': rf_val_metrics,
        'test_metrics': rf_test_metrics
    }
    
    logger.info(f"RandomForest - Val R²: {rf_val_metrics['basic_metrics']['r2']:.4f}, "
               f"Test R²: {rf_test_metrics['basic_metrics']['r2']:.4f}")
    
    # 3. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        logger.info("Training CatBoost...")
        cb = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=8,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False
        )
        start_time = time.time()
        cb.fit(train_features, y_train, eval_set=(val_features, y_val), early_stopping_rounds=50, verbose=False)
        cb_time = time.time() - start_time
        
        cb_val_pred = cb.predict(val_features)
        cb_test_pred = cb.predict(test_features)
        
        cb_val_metrics = metrics_calculator.evaluate_model(y_val, cb_val_pred, "CatBoost_Val")
        cb_test_metrics = metrics_calculator.evaluate_model(y_test, cb_test_pred, "CatBoost_Test")
        
        models['CatBoost'] = cb
        results['CatBoost'] = {
            'training_time': cb_time,
            'val_metrics': cb_val_metrics,
            'test_metrics': cb_test_metrics
        }
        
        logger.info(f"CatBoost - Val R²: {cb_val_metrics['basic_metrics']['r2']:.4f}, "
                   f"Test R²: {cb_test_metrics['basic_metrics']['r2']:.4f}")
    
    # Select best model
    val_r2_scores = {name: res['val_metrics']['basic_metrics']['r2'] for name, res in results.items()}
    best_model_name = max(val_r2_scores, key=val_r2_scores.get)
    best_r2 = val_r2_scores[best_model_name]
    
    logger.info(f"Best model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Save artifacts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    
    # Save models
    for name, model in models.items():
        model_path = models_dir / f"{name.lower()}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} to {model_path}")
    
    # Save best model
    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(models[best_model_name], best_model_path)
    logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    # Save feature engineering artifacts
    feature_engineer.save_feature_artifacts("artifacts")
    
    # Prepare comprehensive results
    comprehensive_results = {
        'timestamp': timestamp,
        'sample_size': sample_size,
        'best_model': best_model_name,
        'best_val_r2': best_r2,
        'model_results': results,
        'feature_count': train_features.shape[1],
        'training_summary': {
            'total_models': len(results),
            'best_model': best_model_name,
            'best_val_r2': best_r2,
            'all_val_r2': val_r2_scores
        }
    }
    
    # Save results
    results_path = metrics_dir / f"quick_training_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_for_json = convert_for_json(comprehensive_results)
    save_metrics(results_for_json, results_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("QUICK TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Sample Size: {sample_size:,}")
    logger.info(f"Features: {train_features.shape[1]}")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best Validation R²: {best_r2:.4f}")
    
    # Show all model performance
    logger.info("\nAll Model Performance:")
    for name, r2 in val_r2_scores.items():
        test_r2 = results[name]['test_metrics']['basic_metrics']['r2']
        within_15 = results[name]['test_metrics']['business_metrics']['within_15_pct']
        logger.info(f"  {name}: Val R² = {r2:.4f}, Test R² = {test_r2:.4f}, Within 15% = {within_15:.1f}%")
    
    # Show best model test performance
    best_test_metrics = results[best_model_name]['test_metrics']
    logger.info(f"\nBest Model Test Performance ({best_model_name}):")
    logger.info(f"  R²: {best_test_metrics['basic_metrics']['r2']:.4f}")
    logger.info(f"  RMSE: ${best_test_metrics['basic_metrics']['rmse']:,.2f}")
    logger.info(f"  Within 15%: {best_test_metrics['business_metrics']['within_15_pct']:.1f}%")
    logger.info(f"  RMSLE: {best_test_metrics['basic_metrics']['rmsle']:.4f}")
    
    # Business assessment
    business_assessment = metrics_calculator.create_business_assessment(best_test_metrics)
    logger.info(f"\nBusiness Assessment:")
    logger.info(f"  Business Ready: {business_assessment['overall_business_ready']}")
    logger.info(f"  Risk Level: {business_assessment['risk_level']}")
    logger.info(f"  Recommendation: {business_assessment['recommendation']}")
    
    logger.info(f"\nArtifacts saved to: outputs/")
    
    return comprehensive_results


if __name__ == "__main__":
    results = quick_train(sample_size=50000)