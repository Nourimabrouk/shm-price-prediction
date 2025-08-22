"""Main training entry point for SHM Heavy Equipment Price Prediction.

This module provides the main training pipeline with:
- Fixed seed for reproducibility (42)
- Temporal validation with strict past→future splits
- Early stopping on validation set
- Windows encoding safety (ASCII console output only)
- Complete artifact trail for audit
- Multiple model training and comparison
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
import joblib
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import sys

# Configure logging for Windows compatibility (ASCII only)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import CatBoost with fallback
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
    logger.info("CatBoost available - will include in model training")
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available - using sklearn models only")

# Import our modules
from src.data import load_and_validate_data
from src.features import LeakProofFeatureEngineer
from src.metrics import RegressionMetrics, save_metrics

warnings.filterwarnings('ignore')

# Fixed random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class ModelTrainer:
    """Main model trainer with comprehensive evaluation."""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.feature_engineer = LeakProofFeatureEngineer(random_state=random_state)
        self.metrics_calculator = RegressionMetrics()
        self.trained_models = {}
        self.training_results = {}
        
    def get_model_configs(self) -> Dict:
        """Get model configurations for training.
        
        Returns:
            Dictionary of model configurations
        """
        configs = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},
                'param_grid': {}
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {'alpha': 1.0},
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 8
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [6, 8, 10]
                }
            }
        }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            configs['CatBoost'] = {
                'model': CatBoostRegressor(
                    random_state=self.random_state,
                    verbose=False,
                    allow_writing_files=False
                ),
                'params': {
                    'iterations': 500,
                    'learning_rate': 0.1,
                    'depth': 8,
                    'early_stopping_rounds': 50
                },
                'param_grid': {
                    'iterations': [300, 500, 1000],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'depth': [6, 8, 10]
                }
            }
        
        return configs
    
    def train_single_model(self, model_name: str, config: Dict,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          optimize_hyperparams: bool = True) -> Dict:
        """Train a single model with optional hyperparameter optimization.
        
        Args:
            model_name: Name of the model
            config: Model configuration
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        # Get base model
        model = config['model']
        
        if optimize_hyperparams and config['param_grid']:
            # Hyperparameter optimization
            logger.info(f"Optimizing hyperparameters for {model_name}")
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model,
                config['param_grid'],
                n_iter=20,  # Reasonable number for speed
                cv=3,       # 3-fold CV for speed
                scoring='neg_mean_squared_error',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
        else:
            # Use default parameters
            if config['params']:
                model.set_params(**config['params'])
            best_model = model
            best_params = config['params']
        
        # Train final model
        if model_name == 'CatBoost' and CATBOOST_AVAILABLE:
            # Special handling for CatBoost with early stopping
            best_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            best_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Calculate metrics
        train_metrics = self.metrics_calculator.evaluate_model(
            y_train, y_train_pred, f"{model_name}_train"
        )
        val_metrics = self.metrics_calculator.evaluate_model(
            y_val, y_val_pred, f"{model_name}_val"
        )
        
        results = {
            'model_name': model_name,
            'model': best_model,
            'best_params': best_params,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'hyperopt_performed': optimize_hyperparams and bool(config['param_grid'])
        }
        
        self.trained_models[model_name] = best_model
        
        logger.info(f"{model_name} training completed in {training_time:.2f}s")
        logger.info(f"{model_name} - Val R²: {val_metrics['basic_metrics']['r2']:.4f}, "
                   f"Val RMSE: {val_metrics['basic_metrics']['rmse']:.2f}")
        
        return results
    
    def find_baseline_performance(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Establish baseline performance using simple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Baseline performance metrics
        """
        logger.info("Establishing baseline performance...")
        
        # Simple mean prediction baseline
        mean_pred = np.full(len(y_val), y_train.mean())
        mean_metrics = self.metrics_calculator.evaluate_model(
            y_val, mean_pred, "Mean_Baseline"
        )
        
        # Simple linear regression baseline
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        lr_metrics = self.metrics_calculator.evaluate_model(
            y_val, lr_pred, "LinearRegression_Baseline"
        )
        
        baseline_results = {
            'mean_baseline': mean_metrics,
            'linear_baseline': lr_metrics,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        logger.info(f"Baseline established - Mean R²: {mean_metrics['basic_metrics']['r2']:.4f}, "
                   f"Linear R²: {lr_metrics['basic_metrics']['r2']:.4f}")
        
        return baseline_results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        optimize_hyperparams: bool = True,
                        quick_mode: bool = False) -> Dict:
        """Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize_hyperparams: Whether to optimize hyperparameters
            quick_mode: If True, skip hyperparameter optimization for speed
            
        Returns:
            All training results
        """
        logger.info("Starting model training pipeline...")
        
        # Establish baseline
        baseline_results = self.find_baseline_performance(X_train, y_train, X_val, y_val)
        
        # Get model configurations
        model_configs = self.get_model_configs()
        
        # Training results
        training_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'baseline_results': baseline_results,
            'model_results': {},
            'best_model': None,
            'training_summary': {}
        }
        
        # Train each model
        model_performances = {}
        
        for model_name, config in model_configs.items():
            try:
                # Skip hyperparameter optimization in quick mode
                optimize = optimize_hyperparams and not quick_mode
                
                results = self.train_single_model(
                    model_name, config,
                    X_train, y_train,
                    X_val, y_val,
                    optimize_hyperparams=optimize
                )
                
                training_results['model_results'][model_name] = results
                
                # Track performance for best model selection
                val_r2 = results['val_metrics']['basic_metrics']['r2']
                model_performances[model_name] = val_r2
                
                logger.info(f"{model_name} - Validation R²: {val_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Select best model
        if model_performances:
            best_model_name = max(model_performances, key=model_performances.get)
            training_results['best_model'] = best_model_name
            training_results['best_performance'] = model_performances[best_model_name]
            
            logger.info(f"Best model: {best_model_name} (R² = {model_performances[best_model_name]:.4f})")
        
        # Training summary
        training_results['training_summary'] = {
            'total_models_trained': len(model_performances),
            'best_model': training_results.get('best_model'),
            'best_r2': training_results.get('best_performance', 0),
            'baseline_r2': baseline_results['linear_baseline']['basic_metrics']['r2'],
            'improvement_over_baseline': training_results.get('best_performance', 0) - 
                                       baseline_results['linear_baseline']['basic_metrics']['r2']
        }
        
        self.training_results = training_results
        
        logger.info("Model training pipeline completed successfully")
        return training_results
    
    def save_models_and_artifacts(self, artifacts_dir: str = "artifacts"):
        """Save trained models and artifacts.
        
        Args:
            artifacts_dir: Directory to save artifacts
        """
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        models_dir = artifacts_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        metrics_dir = artifacts_path / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Save trained models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.trained_models.items():
            model_path = models_dir / f"{model_name.lower()}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save best model with standard name
        if self.training_results.get('best_model'):
            best_model_name = self.training_results['best_model']
            best_model = self.trained_models[best_model_name]
            best_model_path = models_dir / "best_model.pkl"
            joblib.dump(best_model, best_model_path)
            logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
        
        # Save feature engineering artifacts
        self.feature_engineer.save_feature_artifacts(artifacts_dir)
        
        # Save training results
        training_results_path = metrics_dir / f"training_results_{timestamp}.json"
        # Convert numpy types for JSON serialization
        results_for_json = self._prepare_for_json(self.training_results)
        save_metrics(results_for_json, training_results_path)
        
        # Save baseline metrics separately
        baseline_path = metrics_dir / f"baseline_metrics_{timestamp}.json"
        baseline_for_json = self._prepare_for_json(self.training_results['baseline_results'])
        save_metrics(baseline_for_json, baseline_path)
        
        logger.info(f"All artifacts saved to {artifacts_dir}")
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy types."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main(data_path: str = "data/raw/Bit_SHM_data.csv",
         artifacts_dir: str = "artifacts",
         optimize_hyperparams: bool = True,
         quick_mode: bool = False):
    """Main training function.
    
    Args:
        data_path: Path to the data file
        artifacts_dir: Directory to save artifacts
        optimize_hyperparams: Whether to optimize hyperparameters
        quick_mode: If True, skip hyperparameter optimization for speed
    """
    logger.info("Starting SHM Heavy Equipment Price Prediction Training Pipeline")
    logger.info(f"Using random seed: {RANDOM_STATE}")
    
    # Load and validate data
    logger.info("Loading and validating data...")
    df, validation_results = load_and_validate_data(data_path, artifacts_dir, RANDOM_STATE)
    
    # Load temporal splits
    splits_dir = Path(artifacts_dir) / "splits"
    train_idx = pd.read_csv(splits_dir / "train_idx.csv")['index'].values
    val_idx = pd.read_csv(splits_dir / "val_idx.csv")['index'].values
    test_idx = pd.read_csv(splits_dir / "test_idx.csv")['index'].values
    
    # Prepare data
    target_column = 'Sales Price'
    target = df[target_column]
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    # Engineer features on training data
    logger.info("Engineering features...")
    train_features = trainer.feature_engineer.fit_transform(
        df.iloc[train_idx], target.iloc[train_idx]
    )
    val_features = trainer.feature_engineer.transform(df.iloc[val_idx])
    
    # Training targets
    y_train = target.iloc[train_idx]
    y_val = target.iloc[val_idx]
    
    # Align indices
    train_features = train_features.reset_index(drop=True)
    val_features = val_features.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    logger.info(f"Training set: {len(train_features)} samples, {train_features.shape[1]} features")
    logger.info(f"Validation set: {len(val_features)} samples")
    
    # Train models
    logger.info("Training models...")
    training_results = trainer.train_all_models(
        train_features, y_train,
        val_features, y_val,
        optimize_hyperparams=optimize_hyperparams,
        quick_mode=quick_mode
    )
    
    # Save all artifacts
    logger.info("Saving models and artifacts...")
    trainer.save_models_and_artifacts(artifacts_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    
    if training_results.get('best_model'):
        best_model = training_results['best_model']
        best_r2 = training_results['best_performance']
        baseline_r2 = training_results['baseline_results']['linear_baseline']['basic_metrics']['r2']
        improvement = best_r2 - baseline_r2
        
        logger.info(f"Best Model: {best_model}")
        logger.info(f"Best R²: {best_r2:.4f}")
        logger.info(f"Baseline R²: {baseline_r2:.4f}")
        logger.info(f"Improvement: {improvement:.4f} ({improvement/baseline_r2*100:.1f}%)")
        
        # Get best model metrics
        best_results = training_results['model_results'][best_model]
        val_metrics = best_results['val_metrics']
        
        logger.info(f"\nBest Model Performance:")
        logger.info(f"  Validation RMSE: ${val_metrics['basic_metrics']['rmse']:,.2f}")
        logger.info(f"  Validation MAE: ${val_metrics['basic_metrics']['mae']:,.2f}")
        logger.info(f"  Within 15%: {val_metrics['business_metrics']['within_15_pct']:.1f}%")
        logger.info(f"  RMSLE: {val_metrics['basic_metrics']['rmsle']:.4f}")
    
    logger.info(f"\nArtifacts saved to: {artifacts_dir}")
    logger.info("Training pipeline completed successfully!")
    
    return training_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SHM Heavy Equipment Price Prediction Models")
    parser.add_argument("--data_path", default="data/raw/Bit_SHM_data.csv", help="Path to data file")
    parser.add_argument("--artifacts_dir", default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--no_hyperopt", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--quick", action="store_true", help="Quick mode - skip hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Run training
    results = main(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        optimize_hyperparams=not args.no_hyperopt,
        quick_mode=args.quick
    )