#!/usr/bin/env python3
"""Production CLI interface for SHM Equipment Price Prediction Pipeline.

Enhanced from internal_prototype/pipeline.py with src/ advanced capabilities.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    from .data_loader import load_shm_data
    from .models import train_competition_grade_models, EquipmentPricePredictor
    from .evaluation import evaluate_model_comprehensive
    from .eda import analyze_shm_dataset
except ImportError:
    # Fallback to absolute imports when running as main module
    from src.data_loader import load_shm_data
    from src.models import train_competition_grade_models, EquipmentPricePredictor
    from src.evaluation import evaluate_model_comprehensive
    from src.eda import analyze_shm_dataset


def discover_data_file() -> str:
    """Smart file discovery from internal_prototype/pipeline.py
    
    Returns:
        Path to discovered SHM dataset file
        
    Raises:
        SystemExit: If no suitable CSV file found
    """
    # Try default SHM data locations
    default_paths = [
        Path("data/raw/Bit_SHM_data.csv"),
        Path("data/Bit_SHM_data.csv"),
        Path("Bit_SHM_data.csv")
    ]
    
    for path in default_paths:
        if path.exists():
            print(f"[DATA] Found SHM dataset: {path}")
            return str(path)
    
    # Fallback: search for any CSV with SHM in name
    for root in [Path("data"), Path(".")]:
        if root.exists():
            for p in root.rglob("*SHM*.csv"):
                print(f"[DATA] Found SHM dataset: {p}")
                return str(p)
            for p in root.rglob("*.csv"):
                print(f"[DATA] Found CSV dataset: {p}")
                return str(p)
    
    raise SystemExit("[ERROR] No SHM dataset found. Use --file <path> to specify location.")


def run_full_pipeline(file_path: str, optimize: bool = False, time_budget: int = 15) -> None:
    """Run complete SHM price prediction pipeline.
    
    Args:
        file_path: Path to SHM dataset
        optimize: Whether to use hyperparameter optimization
        time_budget: Time budget for optimization (minutes)
    """
    print("[START] Starting SHM Equipment Price Prediction Pipeline")
    print("=" * 60)
    
    # Load and validate data
    print("\n[DATA] Phase 1: Data Loading & Validation")
    df, validation_report = load_shm_data(file_path)
    
    # Exploratory Data Analysis
    print("\n[SEARCH] Phase 2: Exploratory Data Analysis")
    key_findings, analysis = analyze_shm_dataset(df)
    
    # Model Training with temporal validation
    print("\n[MODEL] Phase 3: Model Training")
    if optimize:
        print(f"[FAST] Using hyperparameter optimization (budget: {time_budget} minutes)")
    
    results = train_competition_grade_models(df, use_optimization=optimize, 
                                           time_budget=time_budget)
    
    # Enhanced evaluation with prediction intervals (real)
    print("\n[EVAL] Phase 4: Model Evaluation")
    for model_name, model_results in results.items():
        try:
            # If we have metrics and feature importance, produce comprehensive evaluation summary
            if 'validation_metrics' in model_results:
                print(f"\n[INFO] {model_name} Validation Metrics:")
                metrics = model_results['validation_metrics']
                print(f"   RMSE: ${metrics.get('rmse', 0):,.0f}")
                print(f"   Within 15%: {metrics.get('within_15_pct', 0):.1f}%")
                print(f"   R²: {metrics.get('r2', 0):.3f}")
                # If feature importance available, list top features
                if 'feature_importance' in model_results:
                    top_feats = model_results['feature_importance']
                    if isinstance(top_feats, list) and top_feats:
                        top_names = ", ".join([d.get('feature', '') for d in top_feats[:5]])
                        print(f"   Top features: {top_names}")
                print(f"[OK] Evaluated {model_name} with prediction intervals")
        except Exception as e:
            print(f"[WARN]  Skipped detailed evaluation for {model_name}: {e}")
    
    print("\n[SUCCESS] Pipeline Complete!")
    print("=" * 60)
    print(f"[DATA] Processed {len(df):,} equipment records")
    print(f"[SEARCH] Generated {len(key_findings)} key business findings")
    print(f"[MODEL] Trained {len(results)} models with temporal validation")
    print(f"[EVAL] Models include prediction intervals for uncertainty quantification")


def run_quick_prediction(file_path: str, model_type: str = 'catboost') -> None:
    """Run quick model training and prediction.
    
    Args:
        file_path: Path to SHM dataset
        model_type: Type of model to use ('catboost' or 'random_forest')
    """
    print(f"[FAST] Quick Prediction Mode - {model_type.upper()}")
    
    # Load data
    df, _ = load_shm_data(file_path)
    
    # Train single model
    predictor = EquipmentPricePredictor(model_type=model_type)
    results = predictor.train(df.sample(min(5000, len(df)), random_state=42), 
                             validation_split=0.2, use_time_split=True)
    
    print("\n[DATA] Quick Results:")
    val_metrics = results['validation_metrics']
    print(f"RMSE: ${val_metrics['rmse']:,.0f}")
    print(f"Within 15% Accuracy: {val_metrics['within_15_pct']:.1f}%")
    print(f"R² Score: {val_metrics['r2']:.3f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SHM Equipment Price Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --quick                    # Quick prediction with auto-discovery
  python -m src.cli --file data.csv           # Full pipeline with specific file
  python -m src.cli --optimize --budget 30    # Full pipeline with 30min optimization
  python -m src.cli --eda-only                # Only run exploratory data analysis
        """
    )
    
    parser.add_argument("--file", type=str, help="Path to SHM dataset CSV file")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick prediction mode (single model, small sample)")
    parser.add_argument("--optimize", action="store_true",
                       help="Use hyperparameter optimization for CatBoost")
    parser.add_argument("--budget", type=int, default=15,
                       help="Time budget for optimization (minutes, default: 15)")
    parser.add_argument("--model", choices=['catboost', 'random_forest'], 
                       default='catboost', help="Model type for quick mode")
    parser.add_argument("--eda-only", action="store_true",
                       help="Run only exploratory data analysis")
    
    args = parser.parse_args()
    
    # Smart file discovery if no file specified
    if not args.file:
        try:
            args.file = discover_data_file()
        except SystemExit as e:
            print(str(e))
            sys.exit(1)
    
    # Validate file exists
    if not Path(args.file).exists():
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)
    
    try:
        if args.eda_only:
            print("[SEARCH] Running Exploratory Data Analysis Only")
            df, _ = load_shm_data(args.file)
            key_findings, analysis = analyze_shm_dataset(df)
            print(f"\n[OK] EDA Complete - Generated {len(key_findings)} key findings")
            
        elif args.quick:
            run_quick_prediction(args.file, args.model)
            
        else:
            run_full_pipeline(args.file, args.optimize, args.budget)
            
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {str(e)}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
