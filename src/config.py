"""Configuration settings for SHM price prediction pipeline.

This module centralizes all configuration settings, paths, and hyperparameters
to avoid hardcoding throughout the codebase.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths - use environment variables with sensible defaults
BASE_DIR = Path(os.getenv('SHM_BASE_DIR', Path(__file__).parent.parent))
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
PLOT_DIR = BASE_DIR / 'plots'

# Data paths
RAW_DATA_PATH = Path(os.getenv('SHM_DATA_PATH', DATA_DIR / 'raw' / 'Bit_SHM_data.csv'))
PROCESSED_DATA_PATH = DATA_DIR / 'processed'

# Output paths
FIGURES_DIR = OUTPUT_DIR / 'figures'
RESULTS_DIR = OUTPUT_DIR / 'results'
MODELS_DIR = OUTPUT_DIR / 'models'

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, PLOT_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PROCESSED_DATA_PATH]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# CatBoost default parameters
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'early_stopping_rounds': 50,
    'use_best_model': True
}

# RandomForest default parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Hyperparameter optimization settings
OPTIMIZATION_TIME_BUDGET = 15  # minutes
OPTIMIZATION_N_TRIALS = 100
OPTIMIZATION_CV_FOLDS = 3

# =============================================================================
# BUSINESS CONFIGURATION
# =============================================================================

# Price tolerance levels for business metrics
TOLERANCE_LEVELS = [0.05, 0.10, 0.15, 0.25]  # 5%, 10%, 15%, 25%

# Target column name
TARGET_COLUMN = 'sales_price'

# Categorical features (known from domain knowledge)
CATEGORICAL_FEATURES = [
    'manufacturer', 
    'model', 
    'serial_number', 
    'country',
    'product_group',
    'product_type',
    'currency'
]

# Numerical features to engineer
NUMERICAL_FEATURES = [
    'year_made',
    'sales_price',
    'nr_of_pictures'
]

# Date columns for parsing
DATE_COLUMNS = ['sales_date']

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Evaluation metrics to compute
METRICS = [
    'rmse',
    'mae',
    'mape',
    'r2',
    'tolerance_accuracy_5',
    'tolerance_accuracy_10',
    'tolerance_accuracy_15',
    'tolerance_accuracy_25'
]

# Cross-validation settings
CV_STRATEGY = 'temporal'  # 'temporal' or 'random'
N_SPLITS = 5

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 8)
DPI = 100

# Color palettes
COLOR_PALETTE = 'viridis'
CATEGORICAL_PALETTE = 'Set2'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging settings
LOG_LEVEL = os.getenv('SHM_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

# Performance settings
N_JOBS = -1  # Use all available cores
CHUNK_SIZE = 10000  # For processing large datasets
MEMORY_EFFICIENT = True  # Enable memory-efficient operations

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path() -> Path:
    """Get the path to the main dataset.
    
    Returns:
        Path to SHM dataset
    """
    if not RAW_DATA_PATH.exists():
        # Try alternative paths
        alternatives = [
            BASE_DIR / 'Bit_SHM_data.csv',
            Path.cwd() / 'data' / 'raw' / 'Bit_SHM_data.csv',
            Path.cwd() / 'Bit_SHM_data.csv'
        ]
        
        for alt_path in alternatives:
            if alt_path.exists():
                return alt_path
        
        raise FileNotFoundError(f"Dataset not found. Tried: {RAW_DATA_PATH} and alternatives")
    
    return RAW_DATA_PATH

def get_output_path(filename: str, subdir: str = 'results') -> Path:
    """Get path for output files.
    
    Args:
        filename: Name of the output file
        subdir: Subdirectory within outputs ('results', 'figures', 'models')
    
    Returns:
        Full path to output file
    """
    output_map = {
        'results': RESULTS_DIR,
        'figures': FIGURES_DIR,
        'models': MODELS_DIR,
        'plots': PLOT_DIR
    }
    
    base_path = output_map.get(subdir, OUTPUT_DIR)
    return base_path / filename

def load_config_from_env():
    """Load configuration from environment variables if available."""
    # Override settings from environment
    global OPTIMIZATION_TIME_BUDGET, RANDOM_STATE
    
    if 'SHM_TIME_BUDGET' in os.environ:
        OPTIMIZATION_TIME_BUDGET = int(os.getenv('SHM_TIME_BUDGET'))
    
    if 'SHM_RANDOM_STATE' in os.environ:
        RANDOM_STATE = int(os.getenv('SHM_RANDOM_STATE'))

# Load environment overrides on module import
load_config_from_env()