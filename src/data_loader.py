"""Data loading and validation for SHM equipment dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
import re

try:
    from .config import (
        RAW_DATA_PATH, TARGET_COLUMN, DATE_COLUMNS, 
        CATEGORICAL_FEATURES, NUMERICAL_FEATURES, get_data_path
    )
except ImportError:
    # Fallback when running as script
    from src.config import (
        RAW_DATA_PATH, TARGET_COLUMN, DATE_COLUMNS,
        CATEGORICAL_FEATURES, NUMERICAL_FEATURES, get_data_path
    )

warnings.filterwarnings('ignore')

class SHMDataLoader:
    """Load and validate heavy equipment auction data."""
    
    def __init__(self, data_path: Path):
        """Initialize data loader with path to SHM dataset.
        
        Args:
            data_path: Path to the SHM heavy equipment dataset CSV file
        """
        self.data_path = Path(data_path)
        
        # Enhanced column detection candidates from internal/preprocessing.py
        self.DATE_CANDIDATES = [
            "saledate", "sale_date", "SaleDate", "sales_date", "Sales date", "date", "Date"
        ]
        self.TARGET_CANDIDATES = [
            "SalePrice", "saleprice", "sales_price", "Sales Price", "Price", "price", "target"
        ]
        self.YEAR_CANDIDATES = [
            "YearMade", "yearmade", "year_made", "year", "Year"
        ]
        self.HOURS_CANDIDATES = [
            "MachineHoursCurrentMeter", "machinehours_currentmeter", "machine_hours",
            "hours", "machine_hours_current_meter"
        ]
        self.ID_CANDIDATES = [
            "SalesID", "sales_id", "salesid", "id", "ID", "machine_id", "MachineID"
        ]

        # Canonical alias map for consistent downstream usage
        self.ALIASES = {
            # dates
            "saledate": "sales_date",
            "sale_date": "sales_date",
            "date": "sales_date",
            # target
            "saleprice": "sales_price",
            "price": "sales_price",
            "target": "sales_price",
            # hours
            "machinehourscurrentmeter": "machinehours_currentmeter",
            "machine_hours": "machinehours_currentmeter",
            "machine_hours_current_meter": "machinehours_currentmeter",
            "hours": "machinehours_currentmeter",
            # year
            "yearmade": "year_made",
            "year": "year_made",
            # ids
            "salesid": "sales_id",
            "machineid": "machine_id",
        }
        
        if not self.data_path.exists():
            # Try fallback locations
            fallback_paths = [
                Path("data/raw/Bit_SHM_data.csv"),
                Path("data/Bit_SHM_data.csv"),
                Path("Bit_SHM_data.csv"),
                Path("../data/Bit_SHM_data.csv"),
                Path("../Bit_SHM_data.csv")
            ]
            for alt_path in fallback_paths:
                if alt_path.exists():
                    self.data_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Data file not found. Tried: {data_path} and {fallback_paths}")
    
    def find_column_robust(self, candidates: List[str], df: pd.DataFrame) -> Optional[str]:
        """Case-insensitive column matching from internal/utils.py
        
        Args:
            candidates: List of candidate column names (in priority order)
            df: DataFrame to search
            
        Returns:
            First matching column name or None if no match found
        """
        lower_map = {c.lower(): c for c in df.columns}
        for name in candidates:
            key = name.lower()
            if key in lower_map:
                return lower_map[key]
        return None
    
    def to_snake_case(self, name: str) -> str:
        """Convert column names to snake_case format with CamelCase awareness."""
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', str(name).strip())
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()

    def coalesce_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename known variant columns to canonical names using alias map."""
        rename_map = {c: self.ALIASES[c] for c in df.columns if c in self.ALIASES}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    
    def normalize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize missing value representations with business-aware zero preservation.
        
        Enhanced from internal/preprocessing.py for domain-specific data quality.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with standardized missing values
        """
        # Define common missing value representations (case-insensitive)
        lower_missing_strings = {"none or unspecified", "unknown", "n/a", "na", "", "none", "unspecified"}
        
        # Replace common missing markers with np.nan (case-insensitive)
        for col in df.columns:
            if df[col].dtype == object:
                ser = df[col].astype(str)
                df[col] = ser.where(~ser.str.strip().str.lower().isin(lower_missing_strings), np.nan)
        
        # Business-aware zero preservation for machine hours (enhanced from internal/)
        hours_col = self.find_column_robust(self.HOURS_CANDIDATES, df)
        if hours_col and hours_col in df.columns:
            # Count zeros before processing for audit
            zero_hours_before = (df[hours_col] == 0).sum()
            
            print(f"[TOOL] [DATA AUDIT] Found {zero_hours_before} zero machine hours records")
            print(f"[TOOL] [DATA AUDIT] Business logic: Zero hours often indicate missing data, not actual usage")
            
            # Convert zeros to NaN for machine hours (they're likely data entry errors)
            df.loc[df[hours_col] == 0, hours_col] = np.nan
            
            # Audit: Verify zeros were handled
            zero_hours_after = (df[hours_col] == 0).sum()
            print(f"[TOOL] [DATA AUDIT] Converted {zero_hours_before - zero_hours_after} zero hours to missing values")
            
            # If there are still zeros, they're legitimate (equipment with no usage)
            if zero_hours_after > 0:
                print(f"[TOOL] [DATA AUDIT] Preserved {zero_hours_after} legitimate zero hours records")
            
        return df
    
    def find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the primary date column using enhanced robust candidate matching.
        
        Args:
            df: DataFrame to search for date columns
            
        Returns:
            Name of the primary date column, or None if not found
        """
        # Use enhanced robust column detection
        date_col = self.find_column_robust(self.DATE_CANDIDATES, df)
        if date_col:
            return date_col
        
        # Fallback: first column containing 'date'
        for c in df.columns:
            if "date" in c.lower():
                return c
        return None
    
    def parse_feet_inches(self, x) -> float:
        """Convert strings like "11' 0\"" or "15' 9\"" to inches (float).
        
        Args:
            x: String representation of feet and inches
            
        Returns:
            Value in inches as float, or NaN if unparseable
        """
        if pd.isna(x): 
            return np.nan
        s = str(x).strip()
        m = re.match(r"^\s*(\d+)\s*'\s*(\d+)?\s*\"?\s*$", s)
        if m:
            feet = float(m.group(1))
            inches = float(m.group(2) or 0.0)
            return feet * 12.0 + inches
        # Try to coerce plain numeric strings
        try:
            return float(s)
        except:
            return np.nan
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features combining basic and advanced econometric techniques.
        
        This method implements a two-stage feature engineering pipeline:
        1. Basic feature engineering: Age, usage, temporal, and dimensional features
        2. Advanced econometric features: Non-linear depreciation, seasonality, interactions, etc.
        
        Args:
            df: DataFrame with basic columns
            
        Returns:
            DataFrame with comprehensive engineered features
        """
        print("[TOOL] Stage 1: Basic Feature Engineering...")
        
        # ==================== BASIC FEATURE ENGINEERING ====================
        
        # Date feature engineering
        date_col = self.find_date_column(df)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
            df['sale_year'] = df[date_col].dt.year
            df['sale_month'] = df[date_col].dt.month
            df['sale_quarter'] = df[date_col].dt.quarter
            df['sale_dow'] = df[date_col].dt.dayofweek
        
        # Year sanity check - set implausible YearMade to NaN
        if 'year_made' in df.columns:
            df.loc[df['year_made'] < 1930, 'year_made'] = np.nan
        
        # Age calculation
        if 'year_made' in df.columns and 'sale_year' in df.columns:
            df['age_years'] = df['sale_year'] - df['year_made']
            df.loc[df['age_years'] < 0, 'age_years'] = np.nan
        
        # Usage features
        if 'machinehours_currentmeter' in df.columns:
            df['log1p_hours'] = np.log1p(df['machinehours_currentmeter'])
        
        # Hours per year (usage intensity)
        if 'age_years' in df.columns and 'machinehours_currentmeter' in df.columns:
            df['hours_per_year'] = df['machinehours_currentmeter'] / df['age_years'].clip(lower=0.5)
        
        # Unit parsing for dimensional features
        dimensional_cols = ['machine_width', 'stick_length', 'blade_width', 'screen_size', 'screen_size_1']
        for col in dimensional_cols:
            snake_col = self.to_snake_case(col)
            if snake_col in df.columns and df[snake_col].dtype == object:
                # Try numeric coercion first
                tmp = pd.to_numeric(df[snake_col], errors='coerce')
                if tmp.isna().mean() > 0.5:
                    # High failure rate, try feet/inches parsing
                    df[snake_col + '_in'] = df[snake_col].apply(self.parse_feet_inches)
                else:
                    df[snake_col + '_num'] = tmp
        
        # Target transformation
        target_candidates = ['sales_price', 'saleprice', 'price']
        target = next((c for c in target_candidates if c in df.columns), None)
        if target:
            df['log1p_price'] = np.log1p(df[target])
        
        print(f"[OK] Basic features complete. Dataset shape: {df.shape}")
        
        # ==================== ADVANCED ECONOMETRIC FEATURES ====================
        
        print("[BRAIN] Stage 2: Advanced Econometric Features...")
        
        try:
            # Import the sophisticated feature engineering
            from internal.feature_engineering import add_econometric_features, get_feature_engineering_summary
            
            # Apply econometric feature engineering
            df_enhanced, new_features = add_econometric_features(
                df, 
                target_col=target if target else 'sales_price',
                validate_features=True
            )
            
            # Generate and display summary
            summary = get_feature_engineering_summary(df_enhanced, new_features)
            
            print(f"[TARGET] Econometric enhancement complete!")
            print(f"[DATA] Added {summary['total_features']} sophisticated features")
            print(f"[EVAL] Final dataset shape: {summary['dataset_shape']}")
            
            # Display feature categories
            if summary['feature_categories']:
                print("[SEARCH] Feature Categories Added:")
                for category, details in summary['feature_categories'].items():
                    print(f"  - {category.title()}: {details['count']} features")
            
            return df_enhanced
            
        except ImportError as e:
            print(f"[WARN]  Advanced features unavailable: {e}")
            print("[NOTE] Falling back to basic feature engineering only")
            return df
        
        except Exception as e:
            print(f"[WARN]  Econometric feature engineering failed: {e}")
            print("[NOTE] Falling back to basic feature engineering only")
            return df
    
    def load_data(self) -> pd.DataFrame:
        """Load data with column normalization and basic cleaning.
        
        Returns:
            Cleaned DataFrame with normalized column names
        """
        print(f"Loading data from: {self.data_path}")
        
        # Load the CSV file
        df = pd.read_csv(self.data_path, low_memory=False)
        
        print(f"Original data shape: {df.shape}")
        
        # Drop the first unnamed index column if it exists
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
            df = df.drop(df.columns[0], axis=1)
        
        # Normalize column names to snake_case
        df.columns = [self.to_snake_case(col) for col in df.columns]
        
        # Apply canonical aliasing immediately after normalization
        df = self.coalesce_aliases(df)

        # Normalize missing values
        df = self.normalize_missing_values(df)
        
        # Engineer features (includes date parsing and derived features)
        print("Engineering features...")
        df = self.engineer_features(df)
        
        print(f"Final data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Perform basic data quality checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation metrics and insights
        """
        validation_report = {}
        
        # Basic statistics
        validation_report['shape'] = df.shape
        validation_report['columns'] = len(df.columns)
        validation_report['rows'] = len(df)
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        validation_report['missing_data'] = {
            'total_missing_values': missing_data.sum(),
            'columns_with_missing': (missing_data > 0).sum(),
            'highest_missing_column': missing_percent.idxmax(),
            'highest_missing_percent': missing_percent.max()
        }
        
        # Target variable analysis (Sales Price)
        if 'sales_price' in df.columns:
            target = df['sales_price'].dropna()
            validation_report['target_variable'] = {
                'name': 'sales_price',
                'count': len(target),
                'missing_count': df['sales_price'].isnull().sum(),
                'min_price': target.min(),
                'max_price': target.max(),
                'mean_price': target.mean(),
                'median_price': target.median()
            }
        
        # Categorical variable analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        validation_report['categorical_features'] = {
            'count': len(categorical_cols),
            'high_cardinality': [col for col in categorical_cols 
                               if df[col].nunique() > 100]
        }
        
        # Numerical variable analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        validation_report['numerical_features'] = {
            'count': len(numerical_cols),
            'features': list(numerical_cols)
        }
        
        # Date analysis
        date_col = self.find_date_column(df)
        if date_col and df[date_col].dtype == 'datetime64[ns]':
            valid_dates = df[date_col].dropna()
            validation_report['temporal_analysis'] = {
                'date_column': date_col,
                'date_range': (valid_dates.min(), valid_dates.max()),
                'missing_dates': df[date_col].isnull().sum()
            }
        
        return validation_report
    
    def print_validation_summary(self, validation_report: Dict[str, any]) -> None:
        """Print a formatted validation summary.
        
        Args:
            validation_report: Output from validate_data method
        """
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Dataset Shape: {validation_report['shape'][0]:,} rows x {validation_report['shape'][1]} columns")
        
        # Missing data
        missing = validation_report['missing_data']
        print(f"\nMissing Data:")
        print(f"  Total missing values: {missing['total_missing_values']:,}")
        print(f"  Columns with missing data: {missing['columns_with_missing']}")
        print(f"  Highest missing: {missing['highest_missing_column']} ({missing['highest_missing_percent']:.1f}%)")
        
        # Target variable
        if 'target_variable' in validation_report:
            target = validation_report['target_variable']
            print(f"\nTarget Variable (Sales Price):")
            print(f"  Valid records: {target['count']:,}")
            print(f"  Price range: ${target['min_price']:,.0f} - ${target['max_price']:,.0f}")
            print(f"  Mean price: ${target['mean_price']:,.0f}")
        
        # Features
        print(f"\nFeature Types:")
        print(f"  Numerical features: {validation_report['numerical_features']['count']}")
        print(f"  Categorical features: {validation_report['categorical_features']['count']}")
        
        high_card = validation_report['categorical_features']['high_cardinality']
        if high_card:
            print(f"  High-cardinality categoricals: {len(high_card)} columns")
        
        # Temporal
        if 'temporal_analysis' in validation_report:
            temp = validation_report['temporal_analysis']
            print(f"\nTemporal Data:")
            print(f"  Date column: {temp['date_column']}")
            print(f"  Date range: {temp['date_range'][0].strftime('%Y-%m-%d')} to {temp['date_range'][1].strftime('%Y-%m-%d')}")
        
        print("="*60)


def load_shm_data(data_path: str = "./data/raw/Bit_SHM_data.csv") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Convenience function to load and validate SHM data.
    
    Args:
        data_path: Path to the SHM dataset CSV file
        
    Returns:
        Tuple of (cleaned DataFrame, validation report)
    """
    loader = SHMDataLoader(data_path)
    df = loader.load_data()
    validation_report = loader.validate_data(df)
    loader.print_validation_summary(validation_report)
    
    return df, validation_report


if __name__ == "__main__":
    # Test the data loader
    df, report = load_shm_data()
    print(f"\nLoaded {len(df):,} records successfully!")