# flake8: noqa
"""Integrated leak-proof machine learning pipeline.

This module integrates all temporal leakage elimination components into a
comprehensive ML pipeline that ensures zero temporal leakage while maintaining
high model performance.

CRITICAL MISSION: End-to-end leak-proof model training pipeline.

Key Features:
- Integrated temporal validation and splitting
- Safe feature engineering with past-only data
- Time-aware target encoding
- Comprehensive leakage testing
- Full audit trail generation
- Pipeline orchestration and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import warnings

# Import our leak-proof components
from .temporal_validation import (
    TemporalSplitConfig, TemporalSplitter
)
from .leak_proof_encoding import (
    TargetEncodingConfig, TimeAwareTargetEncoder
)
from .safe_features import (
    SafeFeatureConfig, SafeFeatureEngineer
)
from .leakage_tests import (
    TemporalLeakageTestSuite, run_leakage_tests
)

# Import existing components
from .models import EquipmentPricePredictor
from .data_loader import SHMDataLoader


class LeakProofPipeline:
    """Complete leak-proof machine learning pipeline.
    
    This pipeline orchestrates all components to ensure temporal integrity
    while maximizing model performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize leak-proof pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config or self._get_default_config()
        # Apply environment-driven no-save profile for sandbox runs
        # SHM_NO_SAVE=true will disable saving artifacts/models/predictions
        import os
        if os.getenv('SHM_NO_SAVE', 'false').lower() in ('1', 'true', 'yes'):
            if 'output' in self.config:
                self.config['output']['save_artifacts'] = False
                self.config['output']['save_models'] = False
                self.config['output']['save_predictions'] = False
        self.audit_trail = {}
        self.models = {}
        self.is_fitted = False
        self.output_dir = None
        
        # Initialize components
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            'temporal_split': {
                'train_end_date': '2009-12-31',
                'val_end_date': '2011-12-31',
                'test_start_date': '2012-01-01',
                'date_column': 'sales_date'
            },
            'safe_features': {
                'min_history_days': 30,
                'max_lookback_days': 365,
                'lag_periods': [7, 14, 30, 90],
                'rolling_windows': [7, 30, 90, 180]
            },
            'target_encoding': {
                'min_samples_leaf': 10,
                'smoothing_factor': 0.0,
                'add_noise': True,
                'noise_level': 0.01,
                'shift_periods': 1,
                'cv_folds': 5
            },
            'models': {
                'types': ['random_forest', 'catboost'],
                'catboost_params': {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 8,
                    'l2_leaf_reg': 3
                }
            },
            'validation': {
                'strict_mode': True,
                'enable_comprehensive_tests': True
            },
            'output': {
                'save_artifacts': True,
                'save_models': True,
                'save_predictions': True
            }
        }
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Temporal splitting configuration
        self.temporal_config = TemporalSplitConfig(
            **self.config['temporal_split']
        )
        
        # Safe features configuration
        self.features_config = SafeFeatureConfig(
            **self.config['safe_features']
        )
        
        # Target encoding configuration
        self.encoding_config = TargetEncodingConfig(
            **self.config['target_encoding']
        )
        
        # Initialize splitter
        self.temporal_splitter = TemporalSplitter(
            self.temporal_config, 
            enable_auditing=True
        )
        
        # Initialize feature engineer
        self.feature_engineer = SafeFeatureEngineer(self.features_config)
        
        # Initialize target encoder
        self.target_encoder = TimeAwareTargetEncoder(self.encoding_config)
        
        # Initialize test suite
        self.test_suite = TemporalLeakageTestSuite(
            strict_mode=self.config['validation']['strict_mode']
        )
    
    def fit_predict(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Complete leak-proof training and prediction pipeline.
        
        Args:
            df: Input DataFrame with temporal data
            output_dir: Directory to save artifacts and results
            
        Returns:
            Dictionary with comprehensive results and audit trail
        """
        print("[LEAK-PROOF-PIPELINE] Starting comprehensive leak-proof ML pipeline...")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Data preparation and temporal splitting
        print("\n[STAGE 1] Temporal data splitting with leakage prevention...")
        train_df, val_df, test_df = self._stage_1_temporal_split(df)
        
        # Stage 2: Safe feature engineering
        print("\n[STAGE 2] Safe feature engineering (past-only data)...")
        train_df, val_df, test_df = self._stage_2_safe_features(train_df, val_df, test_df)
        
        # Stage 3: Time-aware target encoding
        print("\n[STAGE 3] Time-aware target encoding...")
        train_df, val_df, test_df = self._stage_3_target_encoding(train_df, val_df, test_df)
        
        # Stage 4: Comprehensive leakage testing
        print("\n[STAGE 4] Comprehensive leakage validation...")
        leakage_results = self._stage_4_leakage_testing(train_df, val_df, test_df)
        
        # Stage 5: Model training with validated data
        print("\n[STAGE 5] Leak-proof model training...")
        model_results = self._stage_5_model_training(train_df, val_df, test_df)
        
        # Stage 6: Final evaluation and artifact generation
        print("\n[STAGE 6] Final evaluation and audit artifact generation...")
        final_results = self._stage_6_final_evaluation(
            train_df, val_df, test_df, model_results, leakage_results
        )
        
        self.is_fitted = True
        
        print("\n[LEAK-PROOF-PIPELINE] Pipeline completed successfully!")
        self._print_pipeline_summary(final_results)
        
        return final_results
    
    def _stage_1_temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stage 1: Perform temporal split with comprehensive auditing."""
        # Perform temporal split
        train_df, val_df, test_df = self.temporal_splitter.split_temporal_data(df)
        
        # Save audit artifacts
        if self.output_dir:
            audit_artifacts = self.temporal_splitter.save_audit_artifacts(
                self.output_dir / "stage_1_temporal_split"
            )
            self.audit_trail['stage_1'] = {
                'artifacts': audit_artifacts,
                'audit': self.temporal_splitter.audit_trail
            }
        
        return train_df, val_df, test_df
    
    def _stage_2_safe_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                             test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stage 2: Engineer safe features using only past data."""
        # Engineer features on training data (fits transformers)
        train_df_features = self.feature_engineer.engineer_safe_features(
            train_df, fit_scalers=True
        )
        
        # Apply to validation and test data (no fitting)
        val_df_features = self.feature_engineer.engineer_safe_features(
            val_df, fit_scalers=False
        )
        test_df_features = self.feature_engineer.engineer_safe_features(
            test_df, fit_scalers=False
        )
        
        # Save feature engineering report
        if self.output_dir:
            feature_report_path = self.output_dir / "stage_2_safe_features"
            feature_report_path.mkdir(parents=True, exist_ok=True)
            
            # Save feature metadata
            with open(feature_report_path / "feature_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.feature_engineer.feature_metadata, f, indent=2)
            
            # Save validation results
            with open(feature_report_path / "feature_validation.json", 'w', encoding='utf-8') as f:
                json.dump(self.feature_engineer.validation_results, f, indent=2, default=str)
            
            self.audit_trail['stage_2'] = {
                'feature_metadata': self.feature_engineer.feature_metadata,
                'validation_results': self.feature_engineer.validation_results
            }
        
        return train_df_features, val_df_features, test_df_features
    
    def _stage_3_target_encoding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stage 3: Apply time-aware target encoding."""
        # Identify categorical columns for encoding
        categorical_columns = [
            col for col in train_df.columns 
            if train_df[col].dtype in ['object', 'category'] and
            col not in [self.config['temporal_split']['date_column'], 'sales_price']
        ]
        
        # Limit to high-cardinality categoricals to avoid over-encoding
        high_cardinality_cats = [
            col for col in categorical_columns 
            if train_df[col].nunique() > 5 and train_df[col].nunique() < 1000
        ]
        # Cardinality monitoring/logging to detect potential explosions
        if high_cardinality_cats:
            card_stats = {c: int(train_df[c].nunique()) for c in high_cardinality_cats}
            print(f"[TARGET-ENCODING] Cardinalities: {card_stats}")
        
        if high_cardinality_cats:
            print(f"[TARGET-ENCODING] Encoding {len(high_cardinality_cats)} categorical columns")
            
            # Fit and transform training data
            train_df_encoded = self.target_encoder.fit_transform_temporal(
                train_df, high_cardinality_cats
            )
            
            # Transform validation and test data
            val_df_encoded = self.target_encoder.transform_temporal(
                val_df, high_cardinality_cats
            )
            test_df_encoded = self.target_encoder.transform_temporal(
                test_df, high_cardinality_cats
            )
            
            # Save encoding validation report
            if self.output_dir:
                encoding_report_path = self.output_dir / "stage_3_target_encoding"
                encoding_report_path.mkdir(parents=True, exist_ok=True)
                
                with open(encoding_report_path / "encoding_validation.json", 'w', encoding='utf-8') as f:
                    json.dump(self.target_encoder.validation_results, f, indent=2, default=str)
                
                self.audit_trail['stage_3'] = {
                    'categorical_columns': high_cardinality_cats,
                    'validation_results': self.target_encoder.validation_results
                }
        else:
            print("[TARGET-ENCODING] No suitable categorical columns found for encoding")
            train_df_encoded, val_df_encoded, test_df_encoded = train_df, val_df, test_df
            self.audit_trail['stage_3'] = {'categorical_columns': []}
        
        return train_df_encoded, val_df_encoded, test_df_encoded
    
    def _stage_4_leakage_testing(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame) -> Dict[str, Any]:
        """Stage 4: Comprehensive leakage testing."""
        leakage_results = run_leakage_tests(
            train_df, val_df, test_df,
            date_column=self.config['temporal_split']['date_column'],
            target_column='sales_price',
            output_dir=self.output_dir / "stage_4_leakage_tests" if self.output_dir else None
        )
        
        # Store in audit trail
        self.audit_trail['stage_4'] = leakage_results
        
        # Handle critical leakage detection
        if leakage_results['overall_status'] == "CRITICAL_LEAKAGE_DETECTED":
            error_msg = f"CRITICAL TEMPORAL LEAKAGE DETECTED: {leakage_results['summary']['critical_issues']}"
            if self.config['validation']['strict_mode']:
                raise ValueError(error_msg)
            else:
                warnings.warn(error_msg)
        
        return leakage_results
    
    def _stage_5_model_training(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Dict[str, Any]:
        """Stage 5: Train models with leak-proof data."""
        model_results = {}
        
        for model_type in self.config['models']['types']:
            print(f"[MODEL-TRAINING] Training {model_type} model...")
            
            # Initialize model
            model = EquipmentPricePredictor(
                model_type=model_type,
                random_state=42
            )
            
            # Train using pre-split data to avoid re-splitting
            results = model.fit_on_pre_split(train_df, val_df)
            
            # Store model and results
            self.models[model_type] = model
            model_results[model_type] = results
            
            # Save model if configured
            if self.output_dir and self.config['output']['save_models']:
                model_path = self.output_dir / f"models/leak_proof_{model_type}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_path))
        
        self.audit_trail['stage_5'] = model_results
        return model_results
    
    def _stage_6_final_evaluation(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                test_df: pd.DataFrame, model_results: Dict[str, Any],
                                leakage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: Final evaluation and comprehensive audit generation."""
        # Test set evaluation
        test_predictions = {}
        test_metrics = {}
        
        if len(test_df) > 0:
            for model_type, model in self.models.items():
                print(f"[FINAL-EVAL] Evaluating {model_type} on test set...")
                
                # Make predictions
                test_pred = model.predict(test_df)
                test_predictions[model_type] = test_pred
                
                # Calculate test metrics
                if 'sales_price' in test_df.columns:
                    test_actual = test_df['sales_price']
                    test_metrics[model_type] = model._calculate_metrics(test_actual, test_pred)
        
        # Compile comprehensive results
        final_results = {
            'pipeline_config': self.config,
            'data_splits': {
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'total_features': len(train_df.columns) - 1  # Exclude target
            },
            'leakage_validation': leakage_results,
            'model_results': model_results,
            'test_predictions': test_predictions,
            'test_metrics': test_metrics,
            'audit_trail': self.audit_trail,
            'pipeline_status': self._determine_pipeline_status(leakage_results, model_results),
            'execution_timestamp': datetime.now().isoformat()
        }
        
        # Save comprehensive audit report
        if self.output_dir and self.config['output']['save_artifacts']:
            self._save_comprehensive_audit(final_results)
        
        return final_results
    
    def _determine_pipeline_status(self, leakage_results: Dict[str, Any], 
                                 model_results: Dict[str, Any]) -> str:
        """Determine overall pipeline status."""
        if leakage_results['overall_status'] == "CRITICAL_LEAKAGE_DETECTED":
            return "FAILED_CRITICAL_LEAKAGE"
        elif leakage_results['overall_status'] == "HIGH_LEAKAGE_DETECTED":
            return "WARNING_HIGH_LEAKAGE"
        elif leakage_results['overall_status'] == "MEDIUM_LEAKAGE_DETECTED":
            return "WARNING_MEDIUM_LEAKAGE"
        elif not model_results:
            return "FAILED_NO_MODELS"
        else:
            return "SUCCESS_LEAK_PROOF"
    
    def _save_comprehensive_audit(self, results: Dict[str, Any]) -> None:
        """Save comprehensive audit report."""
        audit_path = self.output_dir / "comprehensive_audit_report.json"
        
        # Create serializable version
        serializable_results = self._make_serializable(results)
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"[AUDIT] Comprehensive audit report saved to: {audit_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / "pipeline_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_human_readable_summary(results))
        
        print(f"[AUDIT] Human-readable summary saved to: {summary_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_human_readable_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable pipeline summary."""
        summary = []
        summary.append("="*80)
        summary.append("LEAK-PROOF PIPELINE EXECUTION SUMMARY")
        summary.append("="*80)
        summary.append(f"Execution Time: {results['execution_timestamp']}")
        summary.append(f"Pipeline Status: {results['pipeline_status']}")
        summary.append("")
        
        # Data splits
        splits = results['data_splits']
        summary.append("DATA SPLITS:")
        summary.append(f"  Training samples: {splits['train_samples']:,}")
        summary.append(f"  Validation samples: {splits['val_samples']:,}")
        summary.append(f"  Test samples: {splits['test_samples']:,}")
        summary.append(f"  Total features: {splits['total_features']}")
        summary.append("")
        
        # Leakage validation
        leakage = results['leakage_validation']
        summary.append("LEAKAGE VALIDATION:")
        summary.append(f"  Overall Status: {leakage['overall_status']}")
        summary.append(f"  Tests Run: {leakage['total_tests']}")
        summary.append(f"  Tests Passed: {leakage['passed_tests']}")
        summary.append(f"  Tests Failed: {leakage['failed_tests']}")
        
        if leakage['summary']['critical_issues']:
            summary.append(f"  Critical Issues: {', '.join(leakage['summary']['critical_issues'])}")
        if leakage['summary']['high_issues']:
            summary.append(f"  High Issues: {', '.join(leakage['summary']['high_issues'])}")
        summary.append("")
        
        # Model results
        summary.append("MODEL PERFORMANCE:")
        for model_type, model_result in results['model_results'].items():
            val_metrics = model_result['validation_metrics']
            summary.append(f"  {model_type.upper()}:")
            summary.append(f"    Validation RMSE: ${val_metrics['rmse']:,.0f}")
            summary.append(f"    Validation MAE: ${val_metrics['mae']:,.0f}")
            summary.append(f"    Validation R²: {val_metrics['r2']:.3f}")
            summary.append(f"    Within 15%: {val_metrics['within_15_pct']:.1f}%")
        
        # Test set performance
        if results['test_metrics']:
            summary.append("")
            summary.append("TEST SET PERFORMANCE:")
            for model_type, test_metric in results['test_metrics'].items():
                summary.append(f"  {model_type.upper()}:")
                summary.append(f"    Test RMSE: ${test_metric['rmse']:,.0f}")
                summary.append(f"    Test MAE: ${test_metric['mae']:,.0f}")
                summary.append(f"    Test R²: {test_metric['r2']:.3f}")
        
        summary.append("")
        summary.append("="*80)
        
        return "\n".join(summary)
    
    def _print_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Print pipeline execution summary."""
        print("\n" + "="*80)
        print("LEAK-PROOF PIPELINE SUMMARY")
        print("="*80)
        print(f"Status: {results['pipeline_status']}")
        print(f"Leakage Validation: {results['leakage_validation']['overall_status']}")
        print(f"Models Trained: {len(results['model_results'])}")
        print(f"Data Splits: {results['data_splits']['train_samples']} train / {results['data_splits']['val_samples']} val / {results['data_splits']['test_samples']} test")
        print(f"Total Features: {results['data_splits']['total_features']}")
        
        if results['test_metrics']:
            print("\nBest Test Performance:")
            best_model = min(results['test_metrics'].items(), key=lambda x: x[1]['rmse'])
            print(f"  Model: {best_model[0]}")
            print(f"  RMSE: ${best_model[1]['rmse']:,.0f}")
            print(f"  R²: {best_model[1]['r2']:.3f}")
        
        if self.output_dir:
            print(f"\nArtifacts saved to: {self.output_dir}")
        
        print("="*80)


def run_leak_proof_pipeline(df: pd.DataFrame, output_dir: Optional[Path] = None,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the complete leak-proof ML pipeline.
    
    Args:
        df: Input DataFrame with temporal data
        output_dir: Directory to save all artifacts
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        Comprehensive results dictionary
    """
    pipeline = LeakProofPipeline(config)
    return pipeline.fit_predict(df, output_dir)


if __name__ == "__main__":
    # Test the integrated pipeline
    print("Testing integrated leak-proof pipeline...")
    
    # Load sample data
    from .data_loader import load_shm_data
    
    try:
        df, validation_report = load_shm_data()
        
        # Run leak-proof pipeline
        results = run_leak_proof_pipeline(
            df.sample(min(1000, len(df)), random_state=42),  # Sample for testing
            output_dir=Path("test_pipeline_output"),
            config={
                'models': {'types': ['random_forest']},  # Single model for testing
                'validation': {'strict_mode': False}  # Allow warnings for testing
            }
        )
        
        print(f"Pipeline test completed with status: {results['pipeline_status']}")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        # Create synthetic test data
        print("Using synthetic data for testing...")
        
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range("2008-01-01", "2012-12-31", periods=n_samples)
        
        test_df = pd.DataFrame({
            'sales_date': dates,
            'sales_price': np.random.lognormal(10, 0.5, n_samples),
            'machine_id': np.random.choice(range(1000, 1200), n_samples),
            'year_made': np.random.choice(range(2000, 2012), n_samples),
            'equipment_type': np.random.choice(['excavator', 'dozer'], n_samples)
        })
        
        results = run_leak_proof_pipeline(
            test_df,
            output_dir=Path("test_pipeline_output_synthetic"),
            config={'models': {'types': ['random_forest']}}
        )
        
        print(f"Synthetic pipeline test completed with status: {results['pipeline_status']}")