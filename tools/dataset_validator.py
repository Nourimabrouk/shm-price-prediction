#!/usr/bin/env python3
"""Dataset validation tool for SHM repository setup verification."""

import sys
from pathlib import Path
import pandas as pd

def validate_dataset():
    """Validate that the SHM dataset is present and accessible."""
    
    # Check potential dataset locations
    dataset_paths = [
        Path("data/raw/Bit_SHM_data.csv"),
        Path("data/Bit_SHM_data.csv"), 
        Path("Bit_SHM_data.csv")
    ]
    
    print("🔍 DATASET VALIDATION")
    print("=" * 50)
    
    dataset_found = False
    working_path = None
    
    for path in dataset_paths:
        print(f"Checking: {path}")
        if path.exists():
            print(f"  ✅ Found dataset at: {path}")
            working_path = path
            dataset_found = True
            break
        else:
            print(f"  ❌ Not found: {path}")
    
    if not dataset_found:
        print("\n🚨 CRITICAL ISSUE: SHM Dataset Not Found!")
        print("\nThe SHM Heavy Equipment dataset (Bit_SHM_data.csv) is required but missing.")
        print("\nThis dataset should contain heavy equipment auction records.")
        print("Expected size: ~400K records, ~400MB file")
        print("\nPlease ensure the dataset is placed in one of these locations:")
        for path in dataset_paths:
            print(f"  • {path}")
        print("\n💡 For WeAreBit assessment: Contact assessor for dataset access")
        return False
    
    # Validate dataset content
    try:
        print(f"\n📊 DATASET CONTENT VALIDATION")
        print("-" * 30)
        
        df = pd.read_csv(working_path, nrows=1000)  # Sample for validation
        
        print(f"✅ Successfully loaded dataset sample")
        print(f"   Shape: {df.shape[0]} sample rows x {df.shape[1]} columns")
        print(f"   Required columns check:")
        
        # Check for critical columns
        required_columns = ['SalePrice', 'saledate', 'YearMade', 'SalesID']
        missing_required = []
        
        for col in required_columns:
            if col in df.columns:
                print(f"     ✅ {col}")
            else:
                print(f"     ❌ {col} (MISSING)")
                missing_required.append(col)
        
        if missing_required:
            print(f"\n⚠️  WARNING: Missing required columns: {missing_required}")
            print("   This may affect demo functionality")
        else:
            print(f"\n✅ All required columns present")
        
        print(f"\n📈 DATASET SUMMARY")
        print(f"   • Columns: {df.shape[1]}")
        print(f"   • Data types: {df.dtypes.value_counts().to_dict()}")
        
        if 'SalePrice' in df.columns:
            prices = df['SalePrice'].dropna()
            if len(prices) > 0:
                print(f"   • Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
        
        return True
        
    except Exception as e:
        print(f"\n🚨 DATASET LOADING ERROR: {e}")
        print("   Dataset file may be corrupted or invalid format")
        return False

def main():
    """Main validation function."""
    print("SHM Heavy Equipment Price Prediction - Dataset Validator")
    print("Ensuring repository is ready for demonstration\n")
    
    if validate_dataset():
        print("\n🎉 VALIDATION PASSED")
        print("Repository is ready for SHM price prediction demonstration")
        return 0
    else:
        print("\n❌ VALIDATION FAILED") 
        print("Repository requires dataset setup before demonstration")
        return 1

if __name__ == "__main__":
    sys.exit(main())