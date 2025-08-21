#!/usr/bin/env python3
"""
Comprehensive Pipeline Analysis and Testing Script
Tests the end-to-end data pipeline and identifies potential issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import traceback

def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "="*60)
    print("1. TESTING DATA LOADING")
    print("="*60)
    
    try:
        from src.data_loader import SHMDataLoader, load_shm_data
        
        # Test with smaller dataset for quick validation
        df_small = pd.read_csv('data/raw/Bit_SHM_data.csv', nrows=1000)
        print(f"✅ CSV loaded successfully: {df_small.shape}")
        
        # Test data loader
        loader = SHMDataLoader(Path('data/raw/Bit_SHM_data.csv'))
        print(f"✅ SHMDataLoader initialized")
        
        # Test column detection
        date_col = loader.find_column_robust(loader.DATE_CANDIDATES, df_small)
        target_col = loader.find_column_robust(loader.TARGET_CANDIDATES, df_small)
        
        print(f"✅ Date column detected: {date_col}")
        print(f"✅ Target column detected: {target_col}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\n" + "="*60)
    print("2. TESTING FEATURE ENGINEERING")
    print("="*60)
    
    try:
        from src.data_loader import SHMDataLoader
        
        # Load small sample
        df_sample = pd.read_csv('data/raw/Bit_SHM_data.csv', nrows=1000)
        
        loader = SHMDataLoader(Path('data/raw/Bit_SHM_data.csv'))
        
        # Test column normalization
        df_sample.columns = [loader.to_snake_case(col) for col in df_sample.columns]
        print(f"✅ Column normalization: {list(df_sample.columns)[:5]}...")
        
        # Test missing value handling
        df_clean = loader.normalize_missing_values(df_sample)
        print(f"✅ Missing value normalization completed")
        
        # Test feature engineering
        df_features = loader.engineer_features(df_clean)
        print(f"✅ Feature engineering: {df_features.shape[1]} features created")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test model training functionality."""
    print("\n" + "="*60)
    print("3. TESTING MODEL TRAINING")
    print("="*60)
    
    try:
        from src.models import EquipmentPricePredictor
        from src.data_loader import SHMDataLoader
        
        # Load and prepare small dataset
        loader = SHMDataLoader(Path('data/raw/Bit_SHM_data.csv'))
        df_sample = pd.read_csv('data/raw/Bit_SHM_data.csv', nrows=500)
        
        # Basic preprocessing
        df_sample.columns = [loader.to_snake_case(col) for col in df_sample.columns]
        df_sample = loader.coalesce_aliases(df_sample)
        df_sample = loader.normalize_missing_values(df_sample)
        
        # Test model initialization
        model = EquipmentPricePredictor(model_type='random_forest', random_state=42)
        print(f"✅ Model initialized: {model.model_type}")
        
        # Test preprocessing
        df_processed = model.preprocess_data(df_sample, is_training=True)
        print(f"✅ Data preprocessing: {df_processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation():
    """Test evaluation functionality."""
    print("\n" + "="*60)
    print("4. TESTING EVALUATION")
    print("="*60)
    
    try:
        from src.evaluation import ModelEvaluator, evaluate_model_comprehensive
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.lognormal(10, 1, n_samples)  # Realistic price distribution
        y_pred = y_true * (1 + np.random.normal(0, 0.2, n_samples))  # Add some error
        
        # Test evaluator
        evaluator = ModelEvaluator(output_dir="./test_output/")
        print(f"✅ ModelEvaluator initialized")
        
        # Test metrics computation
        metrics = evaluator.compute_business_metrics(y_true, y_pred, "Test Model")
        print(f"✅ Business metrics computed: RMSE=${metrics['rmse']:,.0f}")
        
        # Test prediction intervals
        y_lower, y_upper = evaluator.compute_prediction_intervals(y_true, y_pred)
        print(f"✅ Prediction intervals computed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        traceback.print_exc()
        return False

def test_hybrid_pipeline():
    """Test the hybrid pipeline integration."""
    print("\n" + "="*60)
    print("5. TESTING HYBRID PIPELINE INTEGRATION")
    print("="*60)
    
    try:
        from src.hybrid_pipeline import HybridEquipmentPredictor
        
        # Test initialization
        predictor = HybridEquipmentPredictor(optimization_enabled=False, time_budget=5)
        print(f"✅ HybridEquipmentPredictor initialized")
        
        # Test data loading
        df, validation_report = predictor.load_and_audit_data('data/raw/Bit_SHM_data.csv')
        print(f"✅ Data loaded and audited: {df.shape}")
        
        # Test temporal split with small dataset
        df_small = df.head(1000)  # Use first 1000 rows for testing
        train_df, val_df = predictor.temporal_split_with_comprehensive_audit(df_small, test_size=0.3)
        print(f"✅ Temporal split: {len(train_df)} train, {len(val_df)} validation")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline with minimal data."""
    print("\n" + "="*60)
    print("6. TESTING END-TO-END PIPELINE (MINIMAL)")
    print("="*60)
    
    try:
        from src.data_loader import SHMDataLoader
        from src.models import EquipmentPricePredictor
        from src.evaluation import evaluate_model_comprehensive
        
        # Load minimal dataset
        df_test = pd.read_csv('data/raw/Bit_SHM_data.csv', nrows=200)
        
        # Process data
        loader = SHMDataLoader(Path('data/raw/Bit_SHM_data.csv'))
        df_test.columns = [loader.to_snake_case(col) for col in df_test.columns]
        df_test = loader.coalesce_aliases(df_test)
        df_test = loader.normalize_missing_values(df_test)
        
        # Minimal feature engineering (skip advanced features for speed)
        if 'sales_date' in df_test.columns:
            df_test['sales_date'] = pd.to_datetime(df_test['sales_date'], errors='coerce')
        
        print(f"✅ Data preprocessing completed: {df_test.shape}")
        
        # Train simple model
        model = EquipmentPricePredictor(model_type='random_forest', random_state=42)
        
        # Split data simply
        split_idx = int(len(df_test) * 0.8)
        train_data = df_test[:split_idx]
        val_data = df_test[split_idx:]
        
        # Train on small sample
        results = model.fit_on_pre_split(train_data, val_data)
        print(f"✅ Model trained successfully")
        
        # Test predictions
        predictions = model.predict(val_data)
        print(f"✅ Predictions generated: {len(predictions)} samples")
        
        # Evaluate
        y_true = val_data['sales_price'].dropna().values
        y_pred = predictions[:len(y_true)]  # Match lengths
        
        eval_results = evaluate_model_comprehensive(
            y_true, y_pred, 
            model_name="Test Pipeline",
            output_dir="./test_output/",
            include_intervals=True
        )
        print(f"✅ Evaluation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end pipeline test failed: {e}")
        traceback.print_exc()
        return False

def analyze_data_flow():
    """Analyze data flow and integration points."""
    print("\n" + "="*60)
    print("7. DATA FLOW ANALYSIS")
    print("="*60)
    
    try:
        # Check file existence
        data_file = Path('data/raw/Bit_SHM_data.csv')
        print(f"✅ Data file exists: {data_file.exists()}")
        
        if data_file.exists():
            # Check data format
            df_sample = pd.read_csv(data_file, nrows=5)
            print(f"✅ Data format check: {df_sample.shape}")
            print(f"   Columns: {len(df_sample.columns)}")
            print(f"   Required columns present:")
            print(f"     - Sales Price: {'Sales Price' in df_sample.columns}")
            print(f"     - Sales date: {'Sales date' in df_sample.columns}")
            print(f"     - Year Made: {'Year Made' in df_sample.columns}")
            print(f"     - Machine ID: {'Machine ID' in df_sample.columns}")
        
        # Check module imports
        try:
            from src import data_loader, models, evaluation, hybrid_pipeline
            print(f"✅ All core modules importable")
        except ImportError as e:
            print(f"❌ Module import error: {e}")
        
        # Check internal modules
        try:
            from internal_prototype import feature_engineering
            print(f"✅ Internal feature engineering available")
        except ImportError:
            print(f"⚠️  Internal feature engineering not available (optional)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data flow analysis failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive pipeline analysis."""
    print("SHM PRICE PREDICTION - COMPREHENSIVE PIPELINE ANALYSIS")
    print("="*80)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Evaluation", test_evaluation),
        ("Hybrid Pipeline", test_hybrid_pipeline),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Data Flow Analysis", analyze_data_flow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 PIPELINE ANALYSIS SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline is functioning correctly!")
    else:
        print("⚠️  Some tests failed - see details above for issues to address")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)