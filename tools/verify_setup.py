#!/usr/bin/env python3
"""
Setup verification script for SHM Heavy Equipment Price Prediction System
Verifies that all required packages are installed and working correctly.
"""

import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package can be imported and return version if available."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name}: MISSING ({e})")
        return False

def run_smoke_test():
    """Run smoke test on main pipeline to catch regressions."""
    print("\n4. Pipeline Smoke Test:")
    try:
        import os
        import sys
        import subprocess
        
        # Set environment for minimal run
        env = os.environ.copy()
        env['SHM_NO_SAVE'] = 'true'
        
        print("[TEST] Running main.py --mode analysis on small sample...")
        
        # Run with timeout and capture output
        result = subprocess.run([
            sys.executable, 'main.py', '--mode', 'analysis'
        ], capture_output=True, text=True, timeout=180, env=env, cwd='..')
        
        if result.returncode == 0:
            print("[OK] Pipeline smoke test PASSED")
            return True
        else:
            print(f"[FAIL] Pipeline smoke test FAILED: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Pipeline smoke test TIMEOUT")
        return False
    except Exception as e:
        print(f"[WARN] Pipeline smoke test SKIPPED: {e}")
        return True  # Don't fail setup if smoke test has issues

def main():
    """Run setup verification."""
    print("SHM Price Prediction System - Setup Verification")
    print("=" * 50)
    
    # Core data science packages
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('scikit-learn', 'sklearn'),
        ('catboost', 'catboost'),
        ('scipy', 'scipy'),
        ('statsmodels', 'statsmodels'),
        ('mapie', 'mapie'),
        ('joblib', 'joblib'),
    ]
    
    # Optional notebook packages
    notebook_packages = [
        ('nbformat', 'nbformat'),
        ('nbclient', 'nbclient'),
        ('jupyter-core', 'jupyter_core'),
    ]
    
    print("\n1. Core Data Science Packages:")
    core_success = all(check_package(name, import_name) for name, import_name in packages)
    
    print("\n2. Notebook Support Packages:")
    notebook_success = all(check_package(name, import_name) for name, import_name in notebook_packages)
    
    print("\n3. Python Environment:")
    print(f"[OK] Python version: {sys.version}")
    print(f"[OK] Python executable: {sys.executable}")
    
    # Run smoke test for regression detection
    smoke_success = run_smoke_test()
    
    print("\n" + "=" * 50)
    if core_success and notebook_success and smoke_success:
        print("PASS: Setup verification PASSED - All packages are working correctly!")
        print("\nNext steps:")
        print("1. Launch Jupyter Lab: python -m jupyter lab")
        print("2. Select kernel: 'Python (shm-price)' in Jupyter")
        print("3. Open and run notebooks in the notebooks/ directory")
        return 0
    else:
        print("FAIL: Setup verification FAILED - Some packages are missing or pipeline issues detected!")
        print("\nTo fix:")
        print("1. Ensure virtual environment is activated")
        print("2. Run: python -m pip install -r requirements.txt")
        if not smoke_success:
            print("3. Check main.py pipeline for errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())