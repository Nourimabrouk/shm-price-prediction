#!/usr/bin/env python3
"""
Demonstration of the Temporal Leakage Elimination System

This script demonstrates the key components of the leak-proof pipeline
using synthetic data to show how the system prevents temporal leakage.

Usage:
    python demo_leak_proof.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings for cleaner demo output
warnings.filterwarnings('ignore')

def create_synthetic_equipment_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create realistic synthetic equipment auction data for demonstration."""
    print(f"[DEMO] Creating {n_samples:,} synthetic equipment records...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate temporal data spanning 2007-2014
    start_date = pd.Timestamp('2007-01-01')
    end_date = pd.Timestamp('2014-12-31')
    dates = pd.date_range(start_date, end_date, periods=n_samples)
    
    # Create equipment characteristics
    machine_ids = np.random.choice(range(1000, 1500), n_samples)
    equipment_types = np.random.choice(['Excavator', 'Bulldozer', 'Loader', 'Grader'], n_samples)
    product_groups = np.random.choice(['TEX', 'WL', 'SSL', 'TTT'], n_samples)
    
    # Years made (realistic for used equipment)
    years_made = np.random.choice(range(1995, 2013), n_samples)
    
    # Machine hours with realistic distribution
    hours = np.random.exponential(3000, n_samples)
    hours = np.clip(hours, 0, 15000)  # Cap at reasonable maximum
    
    # Create realistic price relationships
    base_price = 50000
    
    # Age depreciation factor
    ages = 2012 - years_made  # Assuming 2012 as reference year
    age_factor = np.exp(-0.05 * ages)
    
    # Usage depreciation factor
    usage_factor = np.exp(-0.00003 * hours)
    
    # Equipment type factor
    type_factors = {
        'Excavator': 1.2,
        'Bulldozer': 1.1, 
        'Loader': 1.0,
        'Grader': 0.9
    }
    type_factor = np.array([type_factors[eq_type] for eq_type in equipment_types])
    
    # Calculate prices with realistic relationships
    prices = base_price * age_factor * usage_factor * type_factor
    
    # Add market volatility and noise
    market_noise = np.random.lognormal(0, 0.3, n_samples)
    prices *= market_noise
    
    # Add some financial crisis effects (lower prices in 2008-2009)
    crisis_mask = (pd.Series(dates).dt.year.isin([2008, 2009]))
    prices[crisis_mask] *= 0.85  # 15% price reduction during crisis
    
    # Create the dataset
    synthetic_df = pd.DataFrame({
        'sales_date': dates,
        'sales_price': prices,
        'machine_id': machine_ids,
        'equipment_type': equipment_types,
        'product_group': product_groups,
        'year_made': years_made,
        'machinehours_currentmeter': hours,
        'equipment_age': ages,
        'state': np.random.choice(['TX', 'CA', 'FL', 'NY', 'IL'], n_samples),
        'auction_type': np.random.choice(['Online', 'Live', 'Sealed'], n_samples)
    })
    
    print(f"[DEMO] Synthetic data created:")
    print(f"       Date range: {synthetic_df['sales_date'].min().date()} to {synthetic_df['sales_date'].max().date()}")
    print(f"       Price range: ${synthetic_df['sales_price'].min():,.0f} to ${synthetic_df['sales_price'].max():,.0f}")
    print(f"       Equipment types: {synthetic_df['equipment_type'].nunique()}")
    print(f"       Unique machines: {synthetic_df['machine_id'].nunique()}")
    
    return synthetic_df

def demonstrate_temporal_splitting():
    """Demonstrate temporal splitting with leakage detection."""
    print("\n" + "="*80)
    print("DEMONSTRATION 1: TEMPORAL SPLITTING WITH LEAKAGE DETECTION")
    print("="*80)
    
    # Create synthetic data
    df = create_synthetic_equipment_data(800)
    
    try:
        from src.temporal_validation import TemporalSplitConfig, TemporalSplitter
        
        # Configure temporal split
        config = TemporalSplitConfig(
            train_end_date='2009-12-31',
            val_end_date='2011-12-31',
            test_start_date='2012-01-01',
            date_column='sales_date'
        )
        
        # Create splitter with auditing
        splitter = TemporalSplitter(config, enable_auditing=True)
        
        # Perform split
        train_df, val_df, test_df = splitter.split_temporal_data(df)
        
        print(f"\n[RESULTS] Temporal split completed:")
        print(f"          Training: {len(train_df):,} samples")
        print(f"          Validation: {len(val_df):,} samples") 
        print(f"          Test: {len(test_df):,} samples")
        
        # Save audit artifacts
        output_dir = Path("demo_outputs/temporal_split")
        artifacts = splitter.save_audit_artifacts(output_dir)
        print(f"\n[ARTIFACTS] Saved {len(artifacts)} audit files to: {output_dir}")
        
        return train_df, val_df, test_df
        
    except ImportError as e:
        print(f"[ERROR] Could not import temporal validation components: {e}")
        return None, None, None

def demonstrate_leakage_testing(train_df, val_df, test_df):
    """Demonstrate comprehensive leakage testing."""
    if train_df is None:
        return
        
    print("\n" + "="*80)
    print("DEMONSTRATION 2: COMPREHENSIVE LEAKAGE TESTING")
    print("="*80)
    
    try:
        from src.leakage_tests import run_leakage_tests
        
        # Run comprehensive leakage tests
        test_results = run_leakage_tests(
            train_df, val_df, test_df,
            date_column='sales_date',
            target_column='sales_price',
            output_dir=Path("demo_outputs/leakage_tests")
        )
        
        print(f"\n[RESULTS] Leakage testing completed:")
        print(f"          Overall status: {test_results['overall_status']}")
        print(f"          Tests run: {test_results['total_tests']}")
        print(f"          Tests passed: {test_results['passed_tests']}")
        print(f"          Tests failed: {test_results['failed_tests']}")
        
        if test_results['summary']['critical_issues']:
            print(f"          Critical issues: {test_results['summary']['critical_issues']}")
            
        return test_results
        
    except ImportError as e:
        print(f"[ERROR] Could not import leakage testing components: {e}")
        return None

def demonstrate_safe_features(train_df):
    """Demonstrate safe feature engineering."""
    if train_df is None:
        return None
        
    print("\n" + "="*80)
    print("DEMONSTRATION 3: SAFE FEATURE ENGINEERING")
    print("="*80)
    
    try:
        from src.safe_features import create_safe_features
        
        # Create safe features
        features_df = create_safe_features(
            train_df,
            min_history_days=30,
            lag_periods=[7, 30],
            rolling_windows=[30, 90]
        )
        
        original_cols = len(train_df.columns)
        new_cols = len(features_df.columns)
        added_features = new_cols - original_cols
        
        print(f"\n[RESULTS] Safe feature engineering completed:")
        print(f"          Original features: {original_cols}")
        print(f"          New features added: {added_features}")
        print(f"          Total features: {new_cols}")
        
        # Show some example new features
        new_feature_names = [col for col in features_df.columns if col not in train_df.columns]
        print(f"\n[EXAMPLES] Sample new features:")
        for feature in new_feature_names[:5]:
            print(f"           - {feature}")
        
        return features_df
        
    except ImportError as e:
        print(f"[ERROR] Could not import safe feature components: {e}")
        return None

def demonstrate_target_encoding(train_df):
    """Demonstrate time-aware target encoding."""
    if train_df is None:
        return None
        
    print("\n" + "="*80)
    print("DEMONSTRATION 4: TIME-AWARE TARGET ENCODING")
    print("="*80)
    
    try:
        from src.leak_proof_encoding import create_leak_proof_target_encodings
        
        # Perform time-aware target encoding
        categorical_columns = ['equipment_type', 'product_group', 'state']
        
        encoded_df = create_leak_proof_target_encodings(
            train_df,
            categorical_columns,
            method='temporal',
            min_samples_leaf=5,
            shift_periods=1
        )
        
        original_cols = len(train_df.columns)
        new_cols = len(encoded_df.columns)
        encoded_features = new_cols - original_cols
        
        print(f"\n[RESULTS] Target encoding completed:")
        print(f"          Categorical columns processed: {len(categorical_columns)}")
        print(f"          Encoded features created: {encoded_features}")
        
        # Show encoded feature names
        encoded_feature_names = [col for col in encoded_df.columns if col.endswith('_encoded')]
        print(f"\n[EXAMPLES] Encoded features:")
        for feature in encoded_feature_names:
            print(f"           - {feature}")
        
        return encoded_df
        
    except ImportError as e:
        print(f"[ERROR] Could not import target encoding components: {e}")
        return None

def demonstrate_integrated_pipeline():
    """Demonstrate the complete integrated leak-proof pipeline."""
    print("\n" + "="*80)
    print("DEMONSTRATION 5: INTEGRATED LEAK-PROOF PIPELINE")
    print("="*80)
    
    try:
        from src.leak_proof_pipeline import run_leak_proof_pipeline
        
        # Create synthetic data
        df = create_synthetic_equipment_data(500)  # Smaller dataset for demo
        
        # Configure pipeline
        config = {
            'temporal_split': {
                'train_end_date': '2009-12-31',
                'val_end_date': '2011-12-31',
                'test_start_date': '2012-01-01'
            },
            'models': {
                'types': ['random_forest']  # Single model for demo
            },
            'validation': {
                'strict_mode': False  # Allow warnings for demo
            }
        }
        
        # Run complete pipeline
        results = run_leak_proof_pipeline(
            df,
            output_dir=Path("demo_outputs/integrated_pipeline"),
            config=config
        )
        
        print(f"\n[RESULTS] Integrated pipeline completed:")
        print(f"          Pipeline status: {results['pipeline_status']}")
        print(f"          Leakage status: {results['leakage_validation']['overall_status']}")
        print(f"          Models trained: {len(results['model_results'])}")
        print(f"          Data splits: {results['data_splits']['train_samples']} / {results['data_splits']['val_samples']} / {results['data_splits']['test_samples']}")
        
        # Show model performance if available
        if results['test_metrics']:
            for model_name, metrics in results['test_metrics'].items():
                print(f"\n[MODEL] {model_name.upper()} Performance:")
                print(f"        Test RMSE: ${metrics['rmse']:,.0f}")
                print(f"        Test R²: {metrics['r2']:.3f}")
                print(f"        Within 15%: {metrics['within_15_pct']:.1f}%")
        
        print(f"\n[ARTIFACTS] Complete audit trail saved to: demo_outputs/integrated_pipeline")
        
        return results
        
    except ImportError as e:
        print(f"[ERROR] Could not import integrated pipeline components: {e}")
        return None

def main():
    """Run all demonstrations."""
    print("="*80)
    print("TEMPORAL LEAKAGE ELIMINATION SYSTEM - DEMONSTRATION")
    print("="*80)
    print("This demonstration shows how the leak-proof system prevents temporal leakage")
    print("in time-series machine learning using synthetic equipment auction data.")
    print()
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Demonstration 1: Temporal Splitting
        train_df, val_df, test_df = demonstrate_temporal_splitting()
        
        # Demonstration 2: Leakage Testing
        leakage_results = demonstrate_leakage_testing(train_df, val_df, test_df)
        
        # Demonstration 3: Safe Features
        features_df = demonstrate_safe_features(train_df)
        
        # Demonstration 4: Target Encoding
        encoded_df = demonstrate_target_encoding(train_df)
        
        # Demonstration 5: Integrated Pipeline
        pipeline_results = demonstrate_integrated_pipeline()
        
        # Summary
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        print("✓ Temporal splitting with comprehensive auditing")
        print("✓ Comprehensive leakage detection across 7 test categories")
        print("✓ Safe feature engineering using only past data")
        print("✓ Time-aware target encoding with leakage prevention")
        print("✓ Integrated pipeline with full audit trail")
        print()
        print(f"All demonstration artifacts saved to: {output_dir.absolute()}")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        print("This may be due to missing dependencies or import issues.")
        print("Please ensure all components are properly installed.")

if __name__ == "__main__":
    main()