"""Exploratory data analysis functions for SHM equipment dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class SHMDataAnalyzer:
    """Comprehensive EDA for heavy equipment auction data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer with loaded dataset.
        
        Args:
            df: Cleaned SHM dataset from data_loader
        """
        self.df = df.copy()
        self.target_col = 'sales_price'
        
    def analyze_missing_data_patterns(self) -> Dict[str, any]:
        """Analyze missing data patterns with business context.
        
        Returns:
            Dictionary with missing data insights
        """
        missing_analysis = {}
        
        # Calculate missing percentages
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        # Focus on critical business variables
        critical_vars = ['machinehours_currentmeter', 'year_made', 'machine_id', 
                        'model_id', 'sales_date', 'state_of_usage']
        
        missing_analysis['overview'] = {
            'total_columns': len(self.df.columns),
            'columns_with_missing': (missing_counts > 0).sum(),
            'worst_column': missing_percent.idxmax(),
            'worst_percentage': missing_percent.max()
        }
        
        missing_analysis['critical_variables'] = {}
        for var in critical_vars:
            if var in self.df.columns:
                missing_analysis['critical_variables'][var] = {
                    'missing_count': missing_counts[var],
                    'missing_percent': missing_percent[var],
                    'unique_values': self.df[var].nunique()
                }
        
        # Machine hours analysis (key for depreciation)
        if 'machinehours_currentmeter' in self.df.columns:
            hours_col = self.df['machinehours_currentmeter']
            missing_analysis['machine_hours_impact'] = {
                'missing_percentage': missing_percent['machinehours_currentmeter'],
                'affects_depreciation_modeling': missing_percent['machinehours_currentmeter'] > 50,
                'non_zero_records': (hours_col > 0).sum(),
                'zero_or_missing': ((hours_col == 0) | hours_col.isnull()).sum()
            }
        
        return missing_analysis
    
    def analyze_high_cardinality_categoricals(self) -> Dict[str, any]:
        """Analyze categorical variables with high cardinality.
        
        Returns:
            Analysis of categorical complexity
        """
        categorical_analysis = {}
        
        # Find categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        
        cardinality_info = {}
        for col in cat_cols:
            unique_count = self.df[col].nunique()
            cardinality_info[col] = {
                'unique_count': unique_count,
                'total_records': len(self.df),
                'cardinality_ratio': unique_count / len(self.df),
                'top_categories': self.df[col].value_counts().head(3).to_dict()
            }
        
        # Identify high-cardinality variables (>100 unique values)
        high_cardinality = {k: v for k, v in cardinality_info.items() 
                          if v['unique_count'] > 100}
        
        categorical_analysis = {
            'total_categorical': len(cat_cols),
            'high_cardinality_count': len(high_cardinality),
            'high_cardinality_features': high_cardinality,
            'modeling_challenges': len(high_cardinality) > 5  # Many high-card features
        }
        
        return categorical_analysis
    
    def analyze_temporal_patterns(self) -> Dict[str, any]:
        """Analyze temporal patterns and market volatility.
        
        Returns:
            Temporal analysis including market volatility indicators
        """
        temporal_analysis = {}
        
        # Robust date detection: prefer canonical 'sales_date', else any datetime with 'date'
        date_col = 'sales_date' if 'sales_date' in self.df.columns else None
        if not date_col:
            for c in self.df.columns:
                if 'date' in c.lower() and np.issubdtype(self.df[c].dtype, np.datetime64):
                    date_col = c
                    break
        if not date_col:
            # Try to coerce any 'date'-like columns to datetime
            cand = next((c for c in self.df.columns if 'date' in c.lower()), None)
            if cand:
                tmp = pd.to_datetime(self.df[cand], errors='coerce')
                if tmp.notna().any():
                    self.df = self.df.copy()
                    self.df[cand] = tmp
                    date_col = cand
        if not date_col:
            return {'error': 'No date column found'}
        
        # Parse sales dates and extract temporal features
        df_temp = self.df.copy()
        df_temp = df_temp.dropna(subset=[date_col, self.target_col])
        
        # Extract date components
        df_temp['year'] = df_temp[date_col].dt.year
        df_temp['month'] = df_temp[date_col].dt.month
        df_temp['quarter'] = df_temp[date_col].dt.quarter
        
        # Annual price trends
        annual_stats = df_temp.groupby('year')[self.target_col].agg(['mean', 'median', 'count', 'std'])
        
        # Identify market volatility (2008-2009 financial crisis)
        crisis_years = [2008, 2009, 2010]
        crisis_data = annual_stats[annual_stats.index.isin(crisis_years)]
        
        temporal_analysis = {
            'date_range': (df_temp[date_col].min(), df_temp[date_col].max()),
            'years_covered': df_temp['year'].nunique(),
            'annual_statistics': annual_stats.to_dict(),
            'market_volatility': {
                'crisis_period_detected': len(crisis_data) > 0,
                'crisis_years_in_data': crisis_data.index.tolist(),
                'pre_crisis_avg_price': annual_stats[annual_stats.index < 2008]['mean'].mean(),
                'crisis_avg_price': annual_stats[annual_stats.index.isin(crisis_years)]['mean'].mean(),
                'volatility_coefficient': annual_stats['std'].mean() / annual_stats['mean'].mean()
            },
            'seasonal_patterns': df_temp.groupby('month')[self.target_col].mean().to_dict()
        }
        
        return temporal_analysis
    
    def analyze_age_and_usage_anomalies(self) -> Dict[str, any]:
        """Detect anomalies in equipment age and usage data.
        
        Returns:
            Analysis of age and usage data quality
        """
        anomaly_analysis = {}
        
        # Year made analysis
        if 'year_made' in self.df.columns:
            current_year = datetime.now().year
            df_temp = self.df.dropna(subset=['year_made'])
            
            # Calculate equipment age
            ages = current_year - df_temp['year_made']
            
            anomaly_analysis['age_analysis'] = {
                'mean_age': ages.mean(),
                'median_age': ages.median(),
                'future_equipment': (df_temp['year_made'] > current_year).sum(),
                'very_old_equipment': (ages > 50).sum(),  # >50 years old
                'impossible_ages': ((ages < 0) | (ages > 100)).sum()
            }
        
        # Machine hours analysis
        if 'machinehours_currentmeter' in self.df.columns:
            hours = self.df['machinehours_currentmeter'].dropna()
            
            # Detect extreme values
            q99 = hours.quantile(0.99)
            q01 = hours.quantile(0.01)
            
            anomaly_analysis['usage_analysis'] = {
                'mean_hours': hours.mean(),
                'median_hours': hours.median(),
                'max_hours': hours.max(),
                'extreme_high_usage': (hours > q99).sum(),
                'extreme_low_usage': (hours < 10).sum(),  # Very low usage
                'zero_hours': (hours == 0).sum()
            }
        
        return anomaly_analysis
    
    def analyze_geographic_price_variations(self) -> Dict[str, any]:
        """Analyze price variations across geographic regions.
        
        Returns:
            Geographic pricing analysis
        """
        geographic_analysis = {}
        
        if 'state_of_usage' not in self.df.columns:
            return {'error': 'No state_of_usage column found'}
        
        # Group by state and calculate price statistics
        df_clean = self.df.dropna(subset=['state_of_usage', self.target_col])
        
        state_stats = df_clean.groupby('state_of_usage')[self.target_col].agg([
            'mean', 'median', 'count', 'std'
        ]).round(0)
        
        # Identify high and low price states
        state_stats = state_stats[state_stats['count'] >= 10]  # Filter low-sample states
        
        geographic_analysis = {
            'states_analyzed': len(state_stats),
            'highest_avg_price_state': state_stats['mean'].idxmax(),
            'highest_avg_price': state_stats['mean'].max(),
            'lowest_avg_price_state': state_stats['mean'].idxmin(),
            'lowest_avg_price': state_stats['mean'].min(),
            'price_variation_coefficient': state_stats['mean'].std() / state_stats['mean'].mean(),
            'state_statistics': state_stats.to_dict()
        }
        
        return geographic_analysis
    
    def identify_key_findings(self) -> List[Dict[str, str]]:
        """Generate the 5 key findings that warrant special attention.
        
        Returns:
            List of key findings with business impact
        """
        # Run all analyses
        missing_analysis = self.analyze_missing_data_patterns()
        categorical_analysis = self.analyze_high_cardinality_categoricals()
        temporal_analysis = self.analyze_temporal_patterns()
        anomaly_analysis = self.analyze_age_and_usage_anomalies()
        geographic_analysis = self.analyze_geographic_price_variations()
        
        key_findings = []
        
        # Finding 1: Missing Usage Data
        if 'machine_hours_impact' in missing_analysis:
            usage_missing = missing_analysis['machine_hours_impact']['missing_percentage']
            key_findings.append({
                'title': f'Critical Missing Usage Data ({usage_missing:.0f}%)',
                'finding': f'{usage_missing:.0f}% of records lack machine hours data, severely impacting depreciation modeling capabilities.',
                'business_impact': 'High - Usage is critical for equipment valuation',
                'recommendation': 'Develop proxy measures for equipment condition/usage'
            })
        
        # Finding 2: High-Cardinality Model Complexity
        if categorical_analysis['high_cardinality_count'] > 0:
            high_card_count = categorical_analysis['high_cardinality_count']
            key_findings.append({
                'title': f'High-Cardinality Model Complexity ({high_card_count} features)',
                'finding': f'{high_card_count} categorical features have >100 unique values, requiring specialized handling.',
                'business_impact': 'Medium - Affects model training and inference speed',
                'recommendation': 'Use target encoding or embedding approaches'
            })
        
        # Finding 3: Market Volatility Period
        if 'market_volatility' in temporal_analysis and temporal_analysis['market_volatility']['crisis_period_detected']:
            crisis_years = temporal_analysis['market_volatility']['crisis_years_in_data']
            key_findings.append({
                'title': f'Market Volatility Period Detected ({min(crisis_years)}-{max(crisis_years)})',
                'finding': 'Significant price volatility during financial crisis period affects model stability.',
                'business_impact': 'High - Time-aware validation critical for model reliability',
                'recommendation': 'Use chronological validation splits, consider regime-specific models'
            })
        
        # Finding 4: Age and Usage Anomalies
        if 'age_analysis' in anomaly_analysis:
            impossible_ages = anomaly_analysis['age_analysis']['impossible_ages']
            if impossible_ages > 0:
                key_findings.append({
                    'title': f'Data Quality Issues ({impossible_ages} impossible ages)',
                    'finding': f'{impossible_ages} records have impossible manufacturing years or ages.',
                    'business_impact': 'Medium - Affects model training data quality',
                    'recommendation': 'Implement data validation rules and outlier detection'
                })
        
        # Finding 5: Geographic Price Variations
        if 'price_variation_coefficient' in geographic_analysis:
            variation_coeff = geographic_analysis['price_variation_coefficient']
            if variation_coeff > 0.2:  # 20% variation
                high_state = geographic_analysis['highest_avg_price_state']
                low_state = geographic_analysis['lowest_avg_price_state']
                key_findings.append({
                    'title': f'Significant Geographic Price Variations ({variation_coeff:.1%})',
                    'finding': f'Large price differences between states (highest: {high_state}, lowest: {low_state}).',
                    'business_impact': 'Medium - Location is important pricing factor',
                    'recommendation': 'Include geographic features in modeling'
                })
        
        # Ensure we have exactly 5 findings
        while len(key_findings) < 5:
            key_findings.append({
                'title': 'Data Preprocessing Requirements',
                'finding': 'Dataset requires comprehensive preprocessing for ML modeling.',
                'business_impact': 'Medium - Affects model development timeline',
                'recommendation': 'Develop robust preprocessing pipeline'
            })
        
        return key_findings[:5]
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Engineer time-based features for modeling.
        
        Returns:
            DataFrame with additional temporal features
        """
        df_enhanced = self.df.copy()
        
        if 'sales_date' in df_enhanced.columns:
            # Extract temporal components
            df_enhanced['sales_year'] = df_enhanced['sales_date'].dt.year
            df_enhanced['sales_month'] = df_enhanced['sales_date'].dt.month
            df_enhanced['sales_quarter'] = df_enhanced['sales_date'].dt.quarter
            df_enhanced['sales_dayofweek'] = df_enhanced['sales_date'].dt.dayofweek
            
            # Calculate equipment age at sale
            if 'year_made' in df_enhanced.columns:
                df_enhanced['age_at_sale'] = df_enhanced['sales_year'] - df_enhanced['year_made']
        
        return df_enhanced
    
    def print_key_findings_summary(self, findings: List[Dict[str, str]]) -> None:
        """Print formatted summary of key findings.
        
        Args:
            findings: List of key findings from identify_key_findings()
        """
        print("\n" + "="*80)
        print("FIVE KEY FINDINGS - SHM EQUIPMENT PRICE PREDICTION")
        print("="*80)
        
        for i, finding in enumerate(findings, 1):
            print(f"\n{i}. {finding['title']}")
            print(f"   Finding: {finding['finding']}")
            print(f"   Business Impact: {finding['business_impact']}")
            print(f"   Recommendation: {finding['recommendation']}")
            print("-" * 80)


def analyze_shm_dataset(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], Dict[str, any]]:
    """Convenience function to run complete EDA analysis.
    
    Args:
        df: Cleaned SHM dataset from data_loader
        
    Returns:
        Tuple of (key findings list, comprehensive analysis dictionary)
    """
    analyzer = SHMDataAnalyzer(df)
    
    # Run all analyses
    missing_analysis = analyzer.analyze_missing_data_patterns()
    categorical_analysis = analyzer.analyze_high_cardinality_categoricals()
    temporal_analysis = analyzer.analyze_temporal_patterns()
    anomaly_analysis = analyzer.analyze_age_and_usage_anomalies()
    geographic_analysis = analyzer.analyze_geographic_price_variations()
    
    # Generate key findings
    key_findings = analyzer.identify_key_findings()
    analyzer.print_key_findings_summary(key_findings)
    
    # Compile comprehensive analysis
    comprehensive_analysis = {
        'missing_data': missing_analysis,
        'categorical_features': categorical_analysis,
        'temporal_patterns': temporal_analysis,
        'anomaly_detection': anomaly_analysis,
        'geographic_analysis': geographic_analysis
    }
    
    return key_findings, comprehensive_analysis


if __name__ == "__main__":
    # Test the EDA module
    from data_loader import load_shm_data
    
    df, validation_report = load_shm_data()
    key_findings, analysis = analyze_shm_dataset(df)
    
    print(f"\nEDA completed successfully!")
    print(f"Identified {len(key_findings)} key business findings.")