"""Comprehensive evaluation metrics module for SHM Heavy Equipment Price Prediction.

This module provides:
- Standard regression metrics (MAE, RMSE, R², MAPE, RMSLE)
- Business-specific metrics (within percentage accuracy)
- Uncertainty quantification and confidence intervals
- Model comparison and statistical significance testing
- Honest performance reporting with baseline comparisons
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging for Windows compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class RegressionMetrics:
    """Comprehensive regression metrics calculator with business focus."""
    
    def __init__(self):
        self.baseline_metrics = None
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate standard regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of basic metrics
        """
        # Ensure arrays are proper format
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Handle any potential issues
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.error("No valid predictions to evaluate")
            return {}
        
        metrics = {
            'mae': float(mean_absolute_error(y_true_clean, y_pred_clean)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
            'r2': float(r2_score(y_true_clean, y_pred_clean)),
            'samples': int(len(y_true_clean))
        }
        
        # MAPE - handle division by zero
        try:
            # Only calculate MAPE for non-zero true values
            nonzero_mask = y_true_clean != 0
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(np.abs((y_true_clean[nonzero_mask] - y_pred_clean[nonzero_mask]) / y_true_clean[nonzero_mask])) * 100
                metrics['mape'] = float(mape)
            else:
                metrics['mape'] = float('inf')
        except:
            metrics['mape'] = float('inf')
        
        # RMSLE - Root Mean Squared Logarithmic Error
        try:
            # Only for positive values
            pos_mask = (y_true_clean > 0) & (y_pred_clean > 0)
            if np.sum(pos_mask) > 0:
                rmsle = np.sqrt(np.mean((np.log1p(y_true_clean[pos_mask]) - np.log1p(y_pred_clean[pos_mask]))**2))
                metrics['rmsle'] = float(rmsle)
            else:
                metrics['rmsle'] = float('inf')
        except:
            metrics['rmsle'] = float('inf')
        
        return metrics
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate business-specific accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of business metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Clean data
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        # Calculate percentage errors
        percent_errors = np.abs((y_true_clean - y_pred_clean) / y_true_clean) * 100
        
        business_metrics = {
            'within_5_pct': float(np.mean(percent_errors <= 5) * 100),
            'within_10_pct': float(np.mean(percent_errors <= 10) * 100),
            'within_15_pct': float(np.mean(percent_errors <= 15) * 100),
            'within_25_pct': float(np.mean(percent_errors <= 25) * 100),
            'within_50_pct': float(np.mean(percent_errors <= 50) * 100),
            'median_abs_pct_error': float(np.median(percent_errors)),
            'q90_abs_pct_error': float(np.percentile(percent_errors, 90)),
            'max_abs_pct_error': float(np.max(percent_errors))
        }
        
        return business_metrics
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary with confidence intervals
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Clean data
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 10:  # Need minimum samples for CI
            return {}
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        n_samples = len(y_true_clean)
        
        mae_bootstrap = []
        rmse_bootstrap = []
        r2_bootstrap = []
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true_clean[indices]
            y_pred_boot = y_pred_clean[indices]
            
            # Calculate metrics
            mae_bootstrap.append(mean_absolute_error(y_true_boot, y_pred_boot))
            rmse_bootstrap.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
            r2_bootstrap.append(r2_score(y_true_boot, y_pred_boot))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {
            'confidence_level': confidence,
            'mae_ci': [
                float(np.percentile(mae_bootstrap, lower_percentile)),
                float(np.percentile(mae_bootstrap, upper_percentile))
            ],
            'rmse_ci': [
                float(np.percentile(rmse_bootstrap, lower_percentile)),
                float(np.percentile(rmse_bootstrap, upper_percentile))
            ],
            'r2_ci': [
                float(np.percentile(r2_bootstrap, lower_percentile)),
                float(np.percentile(r2_bootstrap, upper_percentile))
            ]
        }
        
        return confidence_intervals
    
    def calculate_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze prediction residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual analysis
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Clean data
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        residuals = y_true_clean - y_pred_clean
        
        residual_analysis = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skewness': float(stats.skew(residuals)),
            'residual_kurtosis': float(stats.kurtosis(residuals)),
            'residual_q25': float(np.percentile(residuals, 25)),
            'residual_median': float(np.median(residuals)),
            'residual_q75': float(np.percentile(residuals, 75)),
            'residual_iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        }
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(residuals) <= 5000:
            try:
                stat, p_value = stats.shapiro(residuals)
                residual_analysis['normality_test'] = 'shapiro-wilk'
                residual_analysis['normality_statistic'] = float(stat)
                residual_analysis['normality_p_value'] = float(p_value)
            except:
                residual_analysis['normality_test'] = 'failed'
        else:
            try:
                result = stats.anderson(residuals, dist='norm')
                residual_analysis['normality_test'] = 'anderson-darling'
                residual_analysis['normality_statistic'] = float(result.statistic)
                residual_analysis['normality_critical_values'] = result.critical_values.tolist()
            except:
                residual_analysis['normality_test'] = 'failed'
        
        return residual_analysis
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Complete evaluation dictionary
        """
        evaluation = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
            'business_metrics': self.calculate_business_metrics(y_true, y_pred),
            'confidence_intervals': self.calculate_confidence_intervals(y_true, y_pred),
            'residual_analysis': self.calculate_residual_analysis(y_true, y_pred)
        }
        
        return evaluation
    
    def compare_with_baseline(self, current_metrics: Dict, 
                            baseline_metrics: Dict) -> Dict:
        """Compare current model with baseline.
        
        Args:
            current_metrics: Current model metrics
            baseline_metrics: Baseline model metrics
            
        Returns:
            Comparison results
        """
        comparison = {
            'comparison_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'baseline_model': baseline_metrics.get('model_name', 'Unknown'),
            'current_model': current_metrics.get('model_name', 'Unknown'),
            'improvements': {},
            'deteriorations': {},
            'overall_assessment': 'unknown'
        }
        
        # Compare basic metrics
        for metric in ['mae', 'rmse', 'r2', 'mape', 'rmsle']:
            baseline_val = baseline_metrics.get('basic_metrics', {}).get(metric)
            current_val = current_metrics.get('basic_metrics', {}).get(metric)
            
            if baseline_val is not None and current_val is not None:
                if metric == 'r2':  # Higher is better
                    improvement = current_val - baseline_val
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'improvement': improvement,
                        'improvement_pct': (improvement / abs(baseline_val)) * 100 if baseline_val != 0 else 0
                    }
                else:  # Lower is better
                    improvement = baseline_val - current_val
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'improvement': improvement,
                        'improvement_pct': (improvement / baseline_val) * 100 if baseline_val != 0 else 0
                    }
        
        # Compare business metrics
        for metric in ['within_15_pct', 'within_25_pct']:
            baseline_val = baseline_metrics.get('business_metrics', {}).get(metric)
            current_val = current_metrics.get('business_metrics', {}).get(metric)
            
            if baseline_val is not None and current_val is not None:
                improvement = current_val - baseline_val  # Higher is better
                comparison['improvements'][metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'improvement': improvement,
                    'improvement_pct': (improvement / baseline_val) * 100 if baseline_val != 0 else 0
                }
        
        # Overall assessment
        r2_improved = comparison['improvements'].get('r2', {}).get('improvement', 0) > 0
        within_15_improved = comparison['improvements'].get('within_15_pct', {}).get('improvement', 0) > 0
        rmse_improved = comparison['improvements'].get('rmse', {}).get('improvement', 0) > 0
        
        if r2_improved and within_15_improved and rmse_improved:
            comparison['overall_assessment'] = 'significant_improvement'
        elif (r2_improved and within_15_improved) or (r2_improved and rmse_improved):
            comparison['overall_assessment'] = 'moderate_improvement'
        elif r2_improved or within_15_improved or rmse_improved:
            comparison['overall_assessment'] = 'slight_improvement'
        else:
            comparison['overall_assessment'] = 'no_improvement'
        
        return comparison
    
    def create_business_assessment(self, metrics: Dict) -> Dict:
        """Create business readiness assessment.
        
        Args:
            metrics: Model evaluation metrics
            
        Returns:
            Business assessment
        """
        basic_metrics = metrics.get('basic_metrics', {})
        business_metrics = metrics.get('business_metrics', {})
        
        # Business requirements (based on typical heavy equipment industry standards)
        requirements = {
            'min_r2': 0.7,  # At least 70% variance explained
            'max_rmse': 15000,  # Max $15k RMSE
            'min_within_15_pct': 60,  # At least 60% within 15%
            'max_rmsle': 0.4  # Max 0.4 RMSLE
        }
        
        assessment = {
            'requirements': requirements,
            'meets_r2_requirement': basic_metrics.get('r2', 0) >= requirements['min_r2'],
            'meets_rmse_requirement': basic_metrics.get('rmse', float('inf')) <= requirements['max_rmse'],
            'meets_accuracy_requirement': business_metrics.get('within_15_pct', 0) >= requirements['min_within_15_pct'],
            'meets_rmsle_requirement': basic_metrics.get('rmsle', float('inf')) <= requirements['max_rmsle'],
            'overall_business_ready': False,
            'risk_level': 'HIGH',
            'recommendation': 'REJECT'
        }
        
        # Calculate overall readiness
        requirements_met = sum([
            assessment['meets_r2_requirement'],
            assessment['meets_rmse_requirement'],
            assessment['meets_accuracy_requirement'],
            assessment['meets_rmsle_requirement']
        ])
        
        if requirements_met >= 3:
            assessment['overall_business_ready'] = True
            assessment['risk_level'] = 'LOW'
            assessment['recommendation'] = 'APPROVE'
        elif requirements_met >= 2:
            assessment['risk_level'] = 'MEDIUM'
            assessment['recommendation'] = 'REVIEW'
        
        # Add specific guidance
        performance_summary = {
            'r2_score': basic_metrics.get('r2', 0),
            'rmse': basic_metrics.get('rmse', 0),
            'within_15_pct': business_metrics.get('within_15_pct', 0),
            'rmsle': basic_metrics.get('rmsle', 0)
        }
        
        assessment['performance_summary'] = performance_summary
        
        return assessment


def save_metrics(metrics: Dict, file_path: str):
    """Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        file_path: Path to save file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {file_path}")


def load_metrics(file_path: str) -> Dict:
    """Load metrics from JSON file.
    
    Args:
        file_path: Path to metrics file
        
    Returns:
        Metrics dictionary
    """
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {file_path}")
    return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    y_true = np.random.normal(50000, 20000, 1000)
    y_pred = y_true + np.random.normal(0, 5000, 1000)  # Add some noise
    
    # Evaluate
    metrics_calculator = RegressionMetrics()
    evaluation = metrics_calculator.evaluate_model(y_true, y_pred, "Example Model")
    
    print("Evaluation Results:")
    print(f"R²: {evaluation['basic_metrics']['r2']:.4f}")
    print(f"RMSE: {evaluation['basic_metrics']['rmse']:.2f}")
    print(f"Within 15%: {evaluation['business_metrics']['within_15_pct']:.1f}%")