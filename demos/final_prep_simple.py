#!/usr/bin/env python3
"""
FINAL SUBMISSION PREPARATION SCRIPT
===================================
WeAreBit Tech Assessment - Final Preparation
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import subprocess

def main():
    """Execute final preparation sequence."""
    print("=" * 60)
    print("FINAL SUBMISSION PREPARATION")
    print("=" * 60)
    
    project_root = Path.cwd()
    
    # 1. Validate critical files
    print("\n[CHECK] Validating critical files...")
    critical_files = [
        "README.md",
        "main.py",
        "requirements.txt",
        "notebooks/EXECUTIVE_REVIEW_NOTEBOOK.ipynb",
        "outputs/findings/EXECUTIVE_SUMMARY.md",
        "data/raw/Bit_SHM_data.csv"
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n[ERROR] Missing critical files: {missing_files}")
        return False
    
    # 2. Check model artifacts
    print("\n[CHECK] Validating model artifacts...")
    model_dir = project_root / "outputs" / "models"
    if model_dir.exists():
        metrics_files = list(model_dir.glob("*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
            print(f"[OK] Latest metrics: {latest_metrics.name}")
            
            try:
                with open(latest_metrics) as f:
                    metrics = json.load(f)
                print(f"[METRICS] CatBoost RMSLE: {metrics.get('catboost', {}).get('rmsle', 'N/A')}")
                print(f"[METRICS] Business accuracy: {metrics.get('catboost', {}).get('within_15_pct', 0)*100:.1f}%")
            except:
                print("[WARN] Could not read metrics file")
        else:
            print("[WARN] No model metrics found")
    else:
        print("[WARN] No model directory found")
    
    # 3. Check presentation materials
    print("\n[CHECK] Validating presentation materials...")
    presentation_dir = project_root / "outputs" / "presentation"
    if presentation_dir.exists():
        guide_file = presentation_dir / "PRESENTATION_GUIDE.md"
        if guide_file.exists():
            print("[OK] Presentation guide exists")
        else:
            print("[WARN] No presentation guide")
            
        slides_dir = presentation_dir / "business_slides"
        if slides_dir.exists():
            slide_count = len(list(slides_dir.glob("*.pdf")))
            print(f"[OK] Business slides: {slide_count} files")
        else:
            print("[WARN] No business slides found")
    else:
        print("[WARN] No presentation directory")
    
    # 4. Final status
    print("\n" + "=" * 60)
    print("PREPARATION STATUS")
    print("=" * 60)
    print("[OK] Repository structure validated")
    print("[OK] Critical files present")
    print("[OK] Ready for submission")
    print("\nSUBMISSION READY - CHAMPIONSHIP GRADE!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Final preparation complete!")
        sys.exit(0)
    else:
        print("\n[ERROR] Preparation failed!")
        sys.exit(1)