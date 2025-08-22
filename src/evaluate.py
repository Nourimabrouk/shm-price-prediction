"""Evaluation and artifact generation module for SHM Heavy Equipment Price Prediction.

This module provides:
- Comprehensive model evaluation on test set
- Artifact generation including plots and reports
- Performance comparison with baselines
- Business assessment and recommendations
- Production readiness evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import warnings

# Configure matplotlib for Windows compatibility
plt.switch_backend('Agg')  # Use non-interactive backend
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use safe font
plt.rcParams['figure.max_open_warning'] = 50

# Configure logging for Windows compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.data import load_and_validate_data
from src.features import LeakProofFeatureEngineer
from src.metrics import RegressionMetrics, save_metrics, load_metrics

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluator with artifact generation."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.metrics_calculator = RegressionMetrics()
        self.feature_engineer = None
        self.models = {}
        self.evaluation_results = {}
        
    def load_artifacts(self):
        """Load saved models and feature engineering artifacts."""
        logger.info("Loading saved artifacts...")
        
        models_dir = self.artifacts_dir / "models"
        
        # Load feature engineering artifacts
        try:
            self.feature_engineer = LeakProofFeatureEngineer()
            self.feature_engineer.label_encoders = joblib.load(models_dir / "label_encoders.pkl")
            self.feature_engineer.scaler = joblib.load(models_dir / "scaler.pkl")
            if (models_dir / "feature_selector.pkl").exists():
                self.feature_engineer.feature_selector = joblib.load(models_dir / "feature_selector.pkl")
            
            # Load feature names
            with open(models_dir / "feature_list.json", 'r') as f:
                feature_info = json.load(f)
                self.feature_engineer.feature_names = feature_info['selected_features']
            
            logger.info("Feature engineering artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load feature engineering artifacts: {e}")
            raise
        
        # Load best model
        try:
            best_model_path = models_dir / "best_model.pkl"
            if best_model_path.exists():
                self.models['best'] = joblib.load(best_model_path)
                logger.info("Best model loaded successfully")
            else:
                logger.warning("Best model not found, will try to load individual models")
                
                # Try to load individual models
                for model_file in models_dir.glob("*.pkl"):
                    if model_file.name not in ['label_encoders.pkl', 'scaler.pkl', 'feature_selector.pkl']:
                        model_name = model_file.stem.split('_')[0]
                        self.models[model_name] = joblib.load(model_file)
                        logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def load_test_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Load and prepare test data.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Tuple of (test_features, test_target, test_indices)
        """
        logger.info("Loading test data...")
        
        # Load data
        df, _ = load_and_validate_data(data_path, self.artifacts_dir)
        
        # Load test indices
        splits_dir = self.artifacts_dir / "splits"
        test_idx = pd.read_csv(splits_dir / "test_idx.csv")['index'].values
        
        # Prepare test data
        target_column = 'Sales Price'
        test_target = df[target_column].iloc[test_idx].reset_index(drop=True)
        
        # Transform features using saved feature engineer
        test_features = self.feature_engineer.transform(df.iloc[test_idx])
        
        logger.info(f"Test data prepared: {len(test_features)} samples, {test_features.shape[1]} features")
        
        return test_features, test_target, test_idx
    
    def evaluate_model_on_test(self, model_name: str, model,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate a single model on test set.
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        evaluation = self.metrics_calculator.evaluate_model(y_test, y_pred, model_name)
        
        # Add business assessment
        business_assessment = self.metrics_calculator.create_business_assessment(evaluation)
        evaluation['business_assessment'] = business_assessment
        
        # Add prediction details for analysis
        evaluation['prediction_details'] = {
            'min_prediction': float(np.min(y_pred)),
            'max_prediction': float(np.max(y_pred)),
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred)),
            'min_actual': float(np.min(y_test)),
            'max_actual': float(np.max(y_test)),
            'mean_actual': float(np.mean(y_test)),
            'std_actual': float(np.std(y_test))
        }
        
        return evaluation
    
    def create_performance_plots(self, y_test: pd.Series, y_pred: np.ndarray,
                               model_name: str, save_dir: Optional[str] = None):
        """Create comprehensive performance plots.
        
        Args:
            y_test: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_dir: Directory to save plots (optional)
        """
        if save_dir:
            plots_dir = Path(save_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
        else:
            plots_dir = self.artifacts_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Predictions vs Actual scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Subplot 1: Predictions vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Subplot 2: Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Subplot 3: Residuals histogram
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Q-Q plot for residuals normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / f"{model_name.lower()}_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plot saved to {plot_path}")
        
        # 2. Feature importance plot (if available)
        try:
            if hasattr(self.models.get('best'), 'feature_importances_'):
                feature_importance = self.models['best'].feature_importances_
                feature_names = self.feature_engineer.feature_names
                
                # Create feature importance plot
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Get top 20 features
                indices = np.argsort(feature_importance)[-20:]
                
                ax.barh(range(len(indices)), feature_importance[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name} - Top 20 Feature Importances')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                importance_path = plots_dir / f"{model_name.lower()}_feature_importance.png"
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Feature importance plot saved to {importance_path}")
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")
        
        # 3. Business metrics visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Percentage error distribution
        percent_errors = np.abs((y_test - y_pred) / y_test) * 100
        axes[0].hist(percent_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(x=15, color='r', linestyle='--', linewidth=2, label='15% threshold')
        axes[0].axvline(x=25, color='orange', linestyle='--', linewidth=2, label='25% threshold')
        axes[0].set_xlabel('Absolute Percentage Error (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative accuracy
        thresholds = np.arange(5, 51, 5)
        accuracies = [np.mean(percent_errors <= t) * 100 for t in thresholds]
        
        axes[1].plot(thresholds, accuracies, 'o-', linewidth=2, markersize=6)
        axes[1].axhline(y=60, color='r', linestyle='--', alpha=0.7, label='60% target')
        axes[1].set_xlabel('Error Threshold (%)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Cumulative Accuracy vs Error Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        business_path = plots_dir / f"{model_name.lower()}_business_metrics.png"
        plt.savefig(business_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Business metrics plot saved to {business_path}")
    
    def evaluate_all_models(self, data_path: str) -> Dict:
        """Evaluate all loaded models on test set.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Load test data
        X_test, y_test, test_idx = self.load_test_data(data_path)
        
        # Evaluation results
        evaluation_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'test_set_size': len(X_test),
            'test_indices': test_idx.tolist(),
            'model_evaluations': {},
            'best_model_analysis': {},
            'summary': {}
        }
        
        # Evaluate each model
        model_performances = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                evaluation = self.evaluate_model_on_test(model_name, model, X_test, y_test)
                evaluation_results['model_evaluations'][model_name] = evaluation
                
                # Track performance
                r2_score = evaluation['basic_metrics']['r2']
                within_15_pct = evaluation['business_metrics']['within_15_pct']
                model_performances[model_name] = (r2_score, within_15_pct)
                
                # Create plots for this model
                y_pred = model.predict(X_test)
                self.create_performance_plots(y_test, y_pred, model_name)
                
                logger.info(f"{model_name} - Test R²: {r2_score:.4f}, Within 15%: {within_15_pct:.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Select best model based on business criteria
        if model_performances:
            # Prioritize within_15_pct, then R²
            best_model_name = max(model_performances, 
                                key=lambda x: (model_performances[x][1], model_performances[x][0]))
            
            evaluation_results['best_model_analysis'] = {
                'best_model_name': best_model_name,
                'best_model_r2': model_performances[best_model_name][0],
                'best_model_within_15_pct': model_performances[best_model_name][1],
                'selection_criteria': 'Prioritized within_15_pct accuracy, then R²'
            }
        
        # Create summary
        if evaluation_results['model_evaluations']:
            summary_stats = []
            for model_name, eval_result in evaluation_results['model_evaluations'].items():
                summary_stats.append({
                    'model': model_name,
                    'r2': eval_result['basic_metrics']['r2'],
                    'rmse': eval_result['basic_metrics']['rmse'],
                    'within_15_pct': eval_result['business_metrics']['within_15_pct'],
                    'rmsle': eval_result['basic_metrics']['rmsle'],
                    'business_ready': eval_result['business_assessment']['overall_business_ready']
                })
            
            evaluation_results['summary'] = {
                'total_models_evaluated': len(summary_stats),
                'models_business_ready': sum(1 for s in summary_stats if s['business_ready']),
                'best_r2': max(s['r2'] for s in summary_stats),
                'best_within_15_pct': max(s['within_15_pct'] for s in summary_stats),
                'model_summary': summary_stats
            }
        
        self.evaluation_results = evaluation_results
        
        logger.info("Model evaluation completed successfully")
        return evaluation_results
    
    def generate_production_report(self) -> str:
        """Generate production model training report.
        
        Returns:
            Report content as markdown string
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Production Model Training Report
Generated: {timestamp}

## Executive Summary

This report presents the results of the production-grade SHM Heavy Equipment Price Prediction model training pipeline.

### Key Results
"""
        
        if self.evaluation_results.get('summary'):
            summary = self.evaluation_results['summary']
            best_model = self.evaluation_results.get('best_model_analysis', {})
            
            report += f"""
- **Models Trained**: {summary['total_models_evaluated']}
- **Best Model**: {best_model.get('best_model_name', 'N/A')}
- **Best R²**: {summary['best_r2']:.4f}
- **Best Within 15% Accuracy**: {summary['best_within_15_pct']:.1f}%
- **Business Ready Models**: {summary['models_business_ready']}/{summary['total_models_evaluated']}

### Performance Summary

| Model | R² | RMSE | Within 15% | RMSLE | Business Ready |
|-------|----|----- |------------|-------|----------------|
"""
            for model_stats in summary['model_summary']:
                ready_status = 'YES' if model_stats['business_ready'] else 'NO'
                report += f"| {model_stats['model']} | {model_stats['r2']:.4f} | ${model_stats['rmse']:,.0f} | {model_stats['within_15_pct']:.1f}% | {model_stats['rmsle']:.4f} | {ready_status} |\n"
        
        # Add detailed analysis for best model
        if self.evaluation_results.get('best_model_analysis'):
            best_model_name = self.evaluation_results['best_model_analysis']['best_model_name']
            if best_model_name in self.evaluation_results['model_evaluations']:
                best_eval = self.evaluation_results['model_evaluations'][best_model_name]
                
                report += f"""
## Best Model Analysis: {best_model_name}

### Performance Metrics
- **R² Score**: {best_eval['basic_metrics']['r2']:.4f}
- **RMSE**: ${best_eval['basic_metrics']['rmse']:,.2f}
- **MAE**: ${best_eval['basic_metrics']['mae']:,.2f}
- **MAPE**: {best_eval['basic_metrics']['mape']:.2f}%
- **RMSLE**: {best_eval['basic_metrics']['rmsle']:.4f}

### Business Metrics
- **Within 5%**: {best_eval['business_metrics']['within_5_pct']:.1f}%
- **Within 10%**: {best_eval['business_metrics']['within_10_pct']:.1f}%
- **Within 15%**: {best_eval['business_metrics']['within_15_pct']:.1f}%
- **Within 25%**: {best_eval['business_metrics']['within_25_pct']:.1f}%

### Business Assessment
- **Business Ready**: {'Yes' if best_eval['business_assessment']['overall_business_ready'] else 'No'}
- **Risk Level**: {best_eval['business_assessment']['risk_level']}
- **Recommendation**: {best_eval['business_assessment']['recommendation']}
"""
                
                # Add confidence intervals if available
                if best_eval.get('confidence_intervals'):
                    ci = best_eval['confidence_intervals']
                    report += f"""
### Confidence Intervals (95%)
- **R² CI**: [{ci['r2_ci'][0]:.4f}, {ci['r2_ci'][1]:.4f}]
- **RMSE CI**: [${ci['rmse_ci'][0]:,.0f}, ${ci['rmse_ci'][1]:,.0f}]
- **MAE CI**: [${ci['mae_ci'][0]:,.0f}, ${ci['mae_ci'][1]:,.0f}]
"""
        
        report += """
## Methodology

### Data Splitting
- **Temporal Validation**: Strict past→future splits to prevent data leakage
- **Split Ratios**: 70% train, 15% validation, 15% test
- **Fixed Seed**: 42 for reproducibility

### Feature Engineering
- **Leakage Prevention**: Removed all price-derived features
- **Temporal Features**: Sale year, quarter, season indicators
- **Equipment Features**: Age, usage intensity, power metrics
- **Categorical Encoding**: Label encoding with unseen category handling
- **Feature Selection**: Statistical significance-based selection

### Model Training
- **Hyperparameter Optimization**: RandomizedSearchCV with 3-fold CV
- **Early Stopping**: Validation-based stopping for gradient boosting
- **Cross-Validation**: Temporal-aware validation strategy

### Evaluation
- **Test Set**: Completely held-out test set (temporal split)
- **Metrics**: Standard regression + business-specific metrics
- **Uncertainty**: Bootstrap confidence intervals
- **Residual Analysis**: Normality testing and distribution analysis

## Artifacts Generated

The following artifacts have been generated and saved:

### Models
- `best_model.pkl`: Best performing model
- Individual model files with timestamps
- Feature engineering pipeline artifacts

### Metrics
- `training_results_[timestamp].json`: Complete training results
- `baseline_metrics_[timestamp].json`: Baseline performance
- Model evaluation results with confidence intervals

### Plots
- Performance analysis plots (predictions vs actual, residuals)
- Feature importance plots
- Business metrics visualizations

### Reports
- This production model training report
- Data validation results
- Feature engineering documentation

## Recommendations

Based on the evaluation results:

1. **Model Selection**: Use the best performing model based on business criteria
2. **Monitoring**: Implement prediction monitoring in production
3. **Retraining**: Consider retraining when performance degrades
4. **Feature Updates**: Monitor for new features that could improve performance

## Technical Notes

- **Windows Compatibility**: All artifacts use ASCII encoding for Windows compatibility
- **Reproducibility**: Fixed random seed (42) ensures reproducible results
- **Production Ready**: Artifacts include all necessary components for deployment

---
*Report generated by SHM Price Prediction Pipeline*
"""
        
        return report
    
    def save_evaluation_artifacts(self):
        """Save all evaluation artifacts."""
        logger.info("Saving evaluation artifacts...")
        
        # Save evaluation results
        metrics_dir = self.artifacts_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete evaluation results
        eval_path = metrics_dir / f"test_evaluation_{timestamp}.json"
        with open(eval_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Generate and save production report
        report_content = self.generate_production_report()
        reports_dir = self.artifacts_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"PRODUCTION_MODEL_TRAINING_REPORT_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Production report saved to {report_path}")
        
        # Also save as standard name for easy access
        standard_report_path = reports_dir / "PRODUCTION_MODEL_TRAINING_REPORT.md"
        with open(standard_report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info("All evaluation artifacts saved successfully")


def main(data_path: str = "data/raw/Bit_SHM_data.csv",
         artifacts_dir: str = "artifacts"):
    """Main evaluation function.
    
    Args:
        data_path: Path to the data file
        artifacts_dir: Directory containing artifacts
    """
    logger.info("Starting SHM Heavy Equipment Price Prediction Model Evaluation")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(artifacts_dir)
    
    # Load artifacts
    evaluator.load_artifacts()
    
    # Evaluate models
    evaluation_results = evaluator.evaluate_all_models(data_path)
    
    # Save evaluation artifacts
    evaluator.save_evaluation_artifacts()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE - SUMMARY")
    logger.info("="*60)
    
    if evaluation_results.get('summary'):
        summary = evaluation_results['summary']
        best_model = evaluation_results.get('best_model_analysis', {})
        
        logger.info(f"Models Evaluated: {summary['total_models_evaluated']}")
        logger.info(f"Best Model: {best_model.get('best_model_name', 'N/A')}")
        logger.info(f"Best R²: {summary['best_r2']:.4f}")
        logger.info(f"Best Within 15%: {summary['best_within_15_pct']:.1f}%")
        logger.info(f"Business Ready: {summary['models_business_ready']}/{summary['total_models_evaluated']}")
    
    logger.info(f"\nArtifacts saved to: {artifacts_dir}")
    logger.info("Evaluation completed successfully!")
    
    return evaluation_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SHM Heavy Equipment Price Prediction Models")
    parser.add_argument("--data_path", default="data/raw/Bit_SHM_data.csv", help="Path to data file")
    parser.add_argument("--artifacts_dir", default="artifacts", help="Directory containing artifacts")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = main(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir
    )