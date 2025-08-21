"""Advanced Feature Engineering for Equipment Price Prediction.

This module implements sophisticated econometric feature engineering techniques
for heavy equipment price modeling, migrated and enhanced from bit-tech-case
with production-grade improvements.

Feature Categories:
1. Non-linear Depreciation Curves: age_squared, log1p_age
2. Cyclic Seasonality: sin/cos transformations for temporal patterns
3. Market Crisis Indicators: Financial crisis period flags
4. Interaction Features: Usage-age, horsepower-age cross-effects  
5. Group Normalization: Z-scores by equipment type for relative pricing
6. Intelligent Binning: Econometrically-justified categorical buckets
7. Missingness Indicators: Data quality as informative features
8. Information Completeness: Listing quality proxy

Author: Migrated from bit-tech-case and enhanced for production
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Union
import warnings
import logging

import numpy as np
import pandas as pd

# Configure logging for feature engineering diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


def add_basic_features(
    df: pd.DataFrame, *, date_col: Optional[str] = None, yearmade_col: Optional[str] = None
) -> pd.DataFrame:
    """Add basic temporal and age features (legacy compatibility).
    
    Args:
        df: Input DataFrame
        date_col: Name of date column for temporal features
        yearmade_col: Name of year made column for age calculation
        
    Returns:
        DataFrame with basic features added
    """
    df = df.copy()
    
    if date_col and date_col in df:
        # Ensure datetime
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["sale_year"] = df[date_col].dt.year
        df["sale_month"] = df[date_col].dt.month

    if yearmade_col and yearmade_col in df:
        if "sale_year" in df:
            df["equipment_age"] = (df["sale_year"] - df[yearmade_col]).clip(lower=0)
        else:
            # Use current year as coarse proxy if sale year not available
            current_year = pd.Timestamp.today().year
            df["equipment_age"] = (current_year - df[yearmade_col]).clip(lower=0)

    return df


def add_econometric_features(df: pd.DataFrame, 
                           target_col: str = 'sales_price',
                           validate_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Add sophisticated econometric features for heavy equipment price modeling.
    
    This function implements PhD-level feature engineering techniques optimized
    for heavy equipment auction data, migrated from bit-tech-case with enhancements.
    
    Feature Engineering Strategy:
    
    1. **Non-linear Depreciation Modeling**:
       - age_squared: Captures accelerating depreciation
       - log1p_age: Stabilizes curvature for linear model compatibility
       
    2. **Market Seasonality & Cycles**:
       - sale_month_sin/cos: Captures seasonal purchasing patterns
       - Financial crisis indicators: 2008-2009 market volatility flags
       
    3. **Usage Intensity & Interactions**:
       - usage_age_interaction: Cross-effects of wear and depreciation
       - age_x_hp: Horsepower-age interaction for performance-based depreciation
       
    4. **Relative Pricing via Group Normalization**:
       - Z-scores by ModelID/ProductGroup for within-category comparisons
       - Controls for equipment type heterogeneity
       
    5. **Intelligent Categorical Binning**:
       - Age buckets: 0-3, 3-7, 7-15, 15+ years (econometrically justified)
       - Usage tertiles: Robust to outliers via log-space binning
       
    6. **Data Quality as Features**:
       - Missingness indicators: NA patterns carry information
       - Information completeness: Listing quality proxy
    
    Args:
        df: Input DataFrame with equipment data
        target_col: Name of target price column (for logging only)
        validate_features: Whether to perform feature validation (default: True)
        
    Returns:
        Tuple of (enhanced DataFrame, list of new feature names)
        
    Raises:
        ValueError: If required base columns are missing
        Warning: If feature validation detects issues
    """
    logger.info(f"Starting econometric feature engineering for {len(df):,} records")
    
    df = df.copy()
    engineered_cols: List[str] = []
    
    try:
        # ==================== NON-LINEAR DEPRECIATION CURVES ====================
        if 'age_years' in df.columns:
            # Quadratic depreciation curve
            df['age_squared'] = (df['age_years'] ** 2).astype(float)
            engineered_cols.append('age_squared')
            
            # Log-age for curvature stabilization (handles zero age gracefully)
            df['log1p_age'] = np.log1p(df['age_years'].clip(lower=0))
            engineered_cols.append('log1p_age')
            
            logger.info(f"✅ Added non-linear depreciation features: age_squared, log1p_age")
        
        # Cross-interaction: Usage intensity with age
        if 'hours_per_year' in df.columns and 'age_years' in df.columns:
            df['usage_age_interaction'] = df['hours_per_year'] * df['age_years']
            engineered_cols.append('usage_age_interaction')
            logger.info(f"✅ Added usage-age interaction feature")
        
        # ==================== MARKET SEASONALITY & CYCLES ====================
        
        # Ensure temporal columns are numeric
        for temporal_col in ['sale_year', 'sale_month', 'sale_quarter', 'sale_dow']:
            if temporal_col in df.columns:
                df[temporal_col] = pd.to_numeric(df[temporal_col], errors='coerce')
        
        # Cyclic seasonality encoding for month
        if 'sale_month' in df.columns:
            two_pi = 2.0 * np.pi
            df['sale_month_sin'] = np.sin(two_pi * (df['sale_month'] / 12.0))
            df['sale_month_cos'] = np.cos(two_pi * (df['sale_month'] / 12.0))
            engineered_cols.extend(['sale_month_sin', 'sale_month_cos'])
            logger.info(f"✅ Added cyclic seasonality features: month sin/cos")
        
        # Time trend and financial crisis indicators
        if 'sale_year' in df.columns:
            # Linear time trend (normalized to start at 0)
            df['year_trend'] = df['sale_year'] - float(df['sale_year'].min())
            engineered_cols.append('year_trend')
            
            # Financial crisis period indicator (2008-2009)
            df['is_2008_2009'] = df['sale_year'].isin([2008.0, 2009.0]).astype(float)
            engineered_cols.append('is_2008_2009')
            
            crisis_count = df['is_2008_2009'].sum()
            logger.info(f"✅ Added time trend and crisis indicator ({crisis_count:,} crisis period records)")
        
        # ==================== INTELLIGENT CATEGORICAL BINNING ====================
        
        # Age buckets with econometric justification
        if 'age_years' in df.columns:
            # Bins: 0-3 (new), 3-7 (early), 7-15 (mature), 15+ (old)
            age_bins = [0, 3, 7, 15, np.inf]
            age_labels = [0.0, 1.0, 2.0, 3.0]
            age_binned = pd.cut(df['age_years'].clip(lower=0), bins=age_bins, 
                              labels=age_labels, include_lowest=True)
            df['age_bucket'] = age_binned.astype(float)
            engineered_cols.append('age_bucket')
            
            # Audit age bucket distribution
            bucket_dist = df['age_bucket'].value_counts().sort_index()
            logger.info(f"✅ Added age buckets: {dict(bucket_dist)}")
        
        # Usage buckets via tertiles (robust to outliers)
        if 'log1p_hours' in df.columns:
            valid_hours = df['log1p_hours'].dropna()
            if len(valid_hours) >= 3:
                q1, q2 = np.quantile(valid_hours, [1/3, 2/3])
                df['hours_bucket'] = (
                    (df['log1p_hours'] <= q1).astype(int) * 0
                    + ((df['log1p_hours'] > q1) & (df['log1p_hours'] <= q2)).astype(int) * 1
                    + (df['log1p_hours'] > q2).astype(int) * 2
                ).astype(float)
                engineered_cols.append('hours_bucket')
                logger.info(f"✅ Added usage tertile buckets (thresholds: {q1:.2f}, {q2:.2f})")
        
        # ==================== GROUP NORMALIZATION (Z-SCORES) ====================
        
        def _safe_group_zscore(col_name: str, group_col: str, output_name: str) -> None:
            """Compute group-wise z-scores with robust error handling."""
            try:
                grp_stats = df.groupby(group_col)[col_name].agg(['mean', 'std'])
                # Handle groups with zero variance
                grp_stats['std'] = grp_stats['std'].replace(0, np.nan)
                
                # Map statistics back to original DataFrame
                group_mean = df[group_col].map(grp_stats['mean'])
                group_std = df[group_col].map(grp_stats['std'])
                
                # Compute z-scores
                z_scores = (df[col_name] - group_mean) / group_std
                df[output_name] = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                engineered_cols.append(output_name)
                
                # Audit z-score quality
                valid_zscores = df[output_name].replace(0.0, np.nan).dropna()
                logger.info(f"✅ Added {output_name}: {len(valid_zscores):,} valid z-scores")
                
            except Exception as e:
                logger.warning(f"Failed to create z-score feature {output_name}: {e}")
        
        # Identify grouping columns with flexible naming
        group_mapping = {
            'model_group': ['Model ID', 'ModelID', 'model_id', 'model_group'],
            'product_group': ['Product Group', 'product_group', 'ProductGroup']
        }
        
        active_groups = []
        for group_type, candidates in group_mapping.items():
            for candidate in candidates:
                if candidate in df.columns and candidate not in active_groups:
                    active_groups.append(candidate)
                    break
        
        # Create group-normalized usage features
        if 'hours_per_year' in df.columns:
            for group_col in active_groups:
                safe_col_name = group_col.lower().replace(' ', '_')
                _safe_group_zscore('hours_per_year', group_col, f'hours_per_year_z_by_{safe_col_name}')
        
        # ==================== HORSEPOWER INTERACTIONS ====================
        
        # Find horsepower column with flexible naming
        hp_candidates = ['Engine Horsepower', 'EngineHorsepower', 'engine_horsepower', 'horsepower']
        hp_col = next((col for col in hp_candidates if col in df.columns), None)
        
        if hp_col is not None and 'age_years' in df.columns:
            # Age-Horsepower interaction (performance depreciation)
            age_numeric = df['age_years'].astype(float)
            hp_numeric = pd.to_numeric(df[hp_col], errors='coerce').astype(float)
            df['age_x_hp'] = age_numeric * hp_numeric
            engineered_cols.append('age_x_hp')
            
            valid_interactions = df['age_x_hp'].dropna().shape[0]
            logger.info(f"✅ Added age-horsepower interaction ({valid_interactions:,} valid values)")
        
        # ==================== DATA QUALITY FEATURES ====================
        
        # Missingness indicators for key columns
        key_cols = ['age_years', 'hours_per_year', 'log1p_hours']
        if hp_col:
            key_cols.append(hp_col)
            
        for col in key_cols:
            if col in df.columns:
                na_feature = f'{col}_na'
                df[na_feature] = df[col].isna().astype(float)
                engineered_cols.append(na_feature)
        
        # Information completeness score (0-1)
        non_missing_ratio = df.notna().sum(axis=1) / max(1, df.shape[1])
        df['info_completeness'] = non_missing_ratio.clip(0, 1)
        engineered_cols.append('info_completeness')
        
        completeness_mean = df['info_completeness'].mean()
        logger.info(f"✅ Added data quality features: missingness indicators + completeness (avg: {completeness_mean:.2f})")
        
        # ==================== FEATURE VALIDATION ====================
        
        if validate_features:
            _validate_engineered_features(df, engineered_cols)
        
        logger.info(f"🎯 Econometric feature engineering complete: {len(engineered_cols)} new features")
        logger.info(f"📊 Enhanced dataset shape: {df.shape}")
        
        return df, engineered_cols
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        raise ValueError(f"Econometric feature engineering failed: {e}")


def _validate_engineered_features(df: pd.DataFrame, feature_list: List[str]) -> None:
    """Validate quality and completeness of engineered features.
    
    Args:
        df: DataFrame with engineered features
        feature_list: List of newly created feature names
        
    Raises:
        Warning: If validation detects potential issues
    """
    validation_issues = []
    
    for feature in feature_list:
        if feature not in df.columns:
            validation_issues.append(f"Missing feature: {feature}")
            continue
            
        # Check for excessive missing values
        missing_pct = df[feature].isna().sum() / len(df) * 100
        if missing_pct > 50:
            validation_issues.append(f"High missingness in {feature}: {missing_pct:.1f}%")
        
        # Check for constant features (no variance)
        if df[feature].nunique(dropna=True) <= 1:
            validation_issues.append(f"Constant feature detected: {feature}")
        
        # Check for infinite values
        if np.isinf(df[feature]).any():
            validation_issues.append(f"Infinite values in {feature}")
    
    if validation_issues:
        warning_msg = "Feature validation warnings:\n" + "\n".join(f"  - {issue}" for issue in validation_issues)
        logger.warning(warning_msg)
    else:
        logger.info("✅ All engineered features passed validation")


def get_feature_engineering_summary(df: pd.DataFrame, engineered_features: List[str]) -> Dict[str, any]:
    """Generate comprehensive summary of feature engineering results.
    
    Args:
        df: DataFrame with engineered features
        engineered_features: List of feature names created by engineering
        
    Returns:
        Dictionary with feature engineering summary statistics
    """
    summary = {
        'total_features': len(engineered_features),
        'dataset_shape': df.shape,
        'feature_categories': {},
        'quality_metrics': {},
        'feature_details': {}
    }
    
    # Categorize features by type
    categories = {
        'depreciation': ['age_squared', 'log1p_age'],
        'seasonality': ['sale_month_sin', 'sale_month_cos', 'year_trend', 'is_2008_2009'],
        'interactions': ['usage_age_interaction', 'age_x_hp'],
        'binning': ['age_bucket', 'hours_bucket'],
        'normalization': [f for f in engineered_features if '_z_by_' in f],
        'data_quality': ['info_completeness'] + [f for f in engineered_features if f.endswith('_na')]
    }
    
    for category, feature_patterns in categories.items():
        matching_features = [f for f in engineered_features if any(pattern in f for pattern in feature_patterns)]
        if matching_features:
            summary['feature_categories'][category] = {
                'count': len(matching_features),
                'features': matching_features
            }
    
    # Quality metrics
    for feature in engineered_features:
        if feature in df.columns:
            summary['feature_details'][feature] = {
                'missing_pct': df[feature].isna().sum() / len(df) * 100,
                'unique_values': df[feature].nunique(),
                'data_type': str(df[feature].dtype)
            }
    
    # Overall quality
    all_missing = [summary['feature_details'][f]['missing_pct'] for f in engineered_features if f in df.columns]
    summary['quality_metrics'] = {
        'avg_missing_pct': np.mean(all_missing) if all_missing else 0,
        'features_with_high_missing': sum(1 for pct in all_missing if pct > 50),
        'total_engineered': len(engineered_features)
    }
    
    return summary