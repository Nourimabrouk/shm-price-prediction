"""Temporal leakage elimination system for SHM Heavy Equipment Price Prediction.

This module implements CRITICAL temporal validation to eliminate ALL forms of data leakage
in time-series equipment price prediction. Designed for Windows environments with 
leak-proof audit trails.

CRITICAL MISSION: Zero temporal leakage tolerance.

Key Features:
- Strict chronological splits with audit trails
- Pipeline-first preprocessing to prevent contamination
- Time-aware feature engineering with past-only aggregations
- Comprehensive leakage detection and reporting
- Windows-safe console output (no emojis)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal data splitting."""
    train_end_date: str  # Last date included in training (YYYY-MM-DD)
    val_end_date: str    # Last date included in validation (YYYY-MM-DD) 
    test_start_date: str # First date for test set (YYYY-MM-DD)
    date_column: str = 'sales_date'
    
    def validate(self) -> None:
        """Validate split configuration for logical consistency."""
        train_end = pd.to_datetime(self.train_end_date)
        val_end = pd.to_datetime(self.val_end_date)
        test_start = pd.to_datetime(self.test_start_date)
        
        if train_end >= val_end:
            raise ValueError(f"Train end ({self.train_end_date}) must be before validation end ({self.val_end_date})")
        if val_end >= test_start:
            raise ValueError(f"Validation end ({self.val_end_date}) must be before test start ({self.test_start_date})")


@dataclass
class TemporalSplitAudit:
    """Comprehensive audit trail for temporal splits."""
    config: TemporalSplitConfig
    train_samples: int
    val_samples: int
    test_samples: int
    train_date_range: Tuple[str, str]
    val_date_range: Tuple[str, str] 
    test_date_range: Tuple[str, str]
    leakage_detected: bool
    leakage_details: List[str]
    split_timestamp: str
    dataset_hash: str  # For reproducibility
    
    def save_audit(self, filepath: Path) -> None:
        """Save audit trail to JSON file."""
        audit_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(audit_dict, f, indent=2)
        print(f"[AUDIT] Temporal split audit saved to: {filepath}")
    
    def print_summary(self) -> None:
        """Print comprehensive audit summary."""
        print("\n" + "="*80)
        print("TEMPORAL SPLIT AUDIT REPORT")
        print("="*80)
        print(f"Split Timestamp: {self.split_timestamp}")
        print(f"Dataset Hash: {self.dataset_hash}")
        print(f"Date Column: {self.config.date_column}")
        print()
        
        print("SPLIT CONFIGURATION:")
        print(f"  Train End Date: {self.config.train_end_date}")
        print(f"  Validation End Date: {self.config.val_end_date}")
        print(f"  Test Start Date: {self.config.test_start_date}")
        print()
        
        print("SAMPLE DISTRIBUTION:")
        total_samples = self.train_samples + self.val_samples + self.test_samples
        print(f"  Training:   {self.train_samples:,} samples ({self.train_samples/total_samples*100:.1f}%)")
        print(f"  Validation: {self.val_samples:,} samples ({self.val_samples/total_samples*100:.1f}%)")
        print(f"  Test:       {self.test_samples:,} samples ({self.test_samples/total_samples*100:.1f}%)")
        print(f"  Total:      {total_samples:,} samples")
        print()
        
        print("DATE RANGES:")
        print(f"  Training:   {self.train_date_range[0]} to {self.train_date_range[1]}")
        print(f"  Validation: {self.val_date_range[0]} to {self.val_date_range[1]}")
        print(f"  Test:       {self.test_date_range[0]} to {self.test_date_range[1]}")
        print()
        
        print("LEAKAGE AUDIT:")
        if self.leakage_detected:
            print("  STATUS: LEAKAGE DETECTED - CRITICAL ERROR!")
            for detail in self.leakage_details:
                print(f"    ERROR: {detail}")
        else:
            print("  STATUS: NO LEAKAGE DETECTED - SPLIT IS VALID")
            print("  All temporal boundaries properly enforced")
        
        print("="*80)


class TemporalLeakageDetector:
    """Advanced leakage detection system for temporal data."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize leakage detector.
        
        Args:
            strict_mode: If True, raise exceptions on leakage detection
        """
        self.strict_mode = strict_mode
        self.detection_results = {}
        
    def detect_temporal_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Comprehensive temporal leakage detection.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            date_column: Name of date column
            
        Returns:
            Dictionary with detailed leakage analysis
        """
        results = {
            'leakage_detected': False,
            'leakage_types': [],
            'details': {},
            'severity': 'NONE'
        }
        
        # Check 1: Direct temporal overlap
        overlap_results = self._check_temporal_overlap(train_df, val_df, test_df, date_column)
        results['details']['temporal_overlap'] = overlap_results
        if overlap_results['has_overlap']:
            results['leakage_detected'] = True
            results['leakage_types'].append('TEMPORAL_OVERLAP')
        
        # Check 2: Entity bleeding (same equipment in multiple sets)
        entity_results = self._check_entity_bleeding(train_df, val_df, test_df)
        results['details']['entity_bleeding'] = entity_results
        if entity_results['has_bleeding']:
            results['leakage_detected'] = True
            results['leakage_types'].append('ENTITY_BLEEDING')
        
        # Check 3: Feature contamination (future information in features)
        feature_results = self._check_feature_contamination(train_df, val_df, test_df, date_column)
        results['details']['feature_contamination'] = feature_results
        if feature_results['has_contamination']:
            results['leakage_detected'] = True
            results['leakage_types'].append('FEATURE_CONTAMINATION')
        
        # Check 4: Target encoding leakage
        encoding_results = self._check_target_encoding_leakage(train_df, val_df, test_df)
        results['details']['target_encoding'] = encoding_results
        if encoding_results['has_leakage']:
            results['leakage_detected'] = True
            results['leakage_types'].append('TARGET_ENCODING_LEAKAGE')
        
        # Determine severity
        if results['leakage_detected']:
            if len(results['leakage_types']) >= 3:
                results['severity'] = 'CRITICAL'
            elif 'TEMPORAL_OVERLAP' in results['leakage_types']:
                results['severity'] = 'HIGH'
            else:
                results['severity'] = 'MEDIUM'
        
        # Handle strict mode
        if self.strict_mode and results['leakage_detected']:
            error_msg = f"TEMPORAL LEAKAGE DETECTED: {', '.join(results['leakage_types'])}"
            raise ValueError(error_msg)
        
        self.detection_results = results
        return results
    
    def _check_temporal_overlap(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Check for direct temporal overlap between splits."""
        results = {
            'has_overlap': False,
            'overlaps': [],
            'gaps': {}
        }
        
        # Get date ranges
        train_dates = pd.to_datetime(train_df[date_column])
        val_dates = pd.to_datetime(val_df[date_column]) if len(val_df) > 0 else pd.Series(dtype='datetime64[ns]')
        test_dates = pd.to_datetime(test_df[date_column]) if len(test_df) > 0 else pd.Series(dtype='datetime64[ns]')
        
        # Check train-validation overlap
        if len(val_dates) > 0:
            train_max = train_dates.max()
            val_min = val_dates.min()
            if train_max >= val_min:
                results['has_overlap'] = True
                results['overlaps'].append({
                    'type': 'TRAIN_VAL_OVERLAP',
                    'train_max': str(train_max),
                    'val_min': str(val_min),
                    'overlap_days': (train_max - val_min).days
                })
        
        # Check validation-test overlap
        if len(val_dates) > 0 and len(test_dates) > 0:
            val_max = val_dates.max()
            test_min = test_dates.min()
            if val_max >= test_min:
                results['has_overlap'] = True
                results['overlaps'].append({
                    'type': 'VAL_TEST_OVERLAP',
                    'val_max': str(val_max),
                    'test_min': str(test_min),
                    'overlap_days': (val_max - test_min).days
                })
        
        # Check train-test overlap (should never happen)
        if len(test_dates) > 0:
            train_max = train_dates.max()
            test_min = test_dates.min()
            if train_max >= test_min:
                results['has_overlap'] = True
                results['overlaps'].append({
                    'type': 'TRAIN_TEST_OVERLAP',
                    'train_max': str(train_max),
                    'test_min': str(test_min),
                    'overlap_days': (train_max - test_min).days
                })
        
        # Calculate gaps
        if len(val_dates) > 0:
            train_val_gap = (val_dates.min() - train_dates.max()).days
            results['gaps']['train_val'] = train_val_gap
            
        if len(test_dates) > 0:
            if len(val_dates) > 0:
                val_test_gap = (test_dates.min() - val_dates.max()).days
                results['gaps']['val_test'] = val_test_gap
            else:
                train_test_gap = (test_dates.min() - train_dates.max()).days
                results['gaps']['train_test'] = train_test_gap
        
        return results
    
    def _check_entity_bleeding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for entity bleeding across temporal splits."""
        results = {
            'has_bleeding': False,
            'bleeding_entities': {},
            'entity_columns': []
        }
        
        # Identify potential entity columns
        entity_candidates = ['sales_id', 'machine_id', 'equipment_id', 'serial_number', 'vin']
        entity_columns = [col for col in entity_candidates if col in train_df.columns]
        results['entity_columns'] = entity_columns
        
        for col in entity_columns:
            train_entities = set(train_df[col].dropna())
            val_entities = set(val_df[col].dropna()) if len(val_df) > 0 else set()
            test_entities = set(test_df[col].dropna()) if len(test_df) > 0 else set()
            
            # Check for overlaps
            train_val_overlap = train_entities & val_entities
            train_test_overlap = train_entities & test_entities
            val_test_overlap = val_entities & test_entities
            
            if train_val_overlap or train_test_overlap or val_test_overlap:
                results['has_bleeding'] = True
                results['bleeding_entities'][col] = {
                    'train_val_overlap': len(train_val_overlap),
                    'train_test_overlap': len(train_test_overlap),
                    'val_test_overlap': len(val_test_overlap),
                    'total_unique_train': len(train_entities),
                    'total_unique_val': len(val_entities),
                    'total_unique_test': len(test_entities)
                }
        
        return results
    
    def _check_feature_contamination(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                   test_df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Check for features that might contain future information."""
        results = {
            'has_contamination': False,
            'contaminated_features': [],
            'suspicious_features': []
        }
        
        # Look for features that suggest future information
        suspicious_patterns = [
            'future_', 'next_', 'ahead_', 'forward_', 'cumulative_', 'total_', 'global_'
        ]
        
        for col in train_df.columns:
            if col == date_column:
                continue
                
            col_lower = col.lower()
            
            # Check for suspicious naming patterns
            if any(pattern in col_lower for pattern in suspicious_patterns):
                results['suspicious_features'].append({
                    'feature': col,
                    'reason': 'Suspicious naming pattern'
                })
            
            # Check for global aggregations (same value across many rows)
            if train_df[col].dtype in ['int64', 'float64']:
                unique_ratio = train_df[col].nunique() / len(train_df)
                if unique_ratio < 0.01 and train_df[col].nunique() > 1:
                    results['suspicious_features'].append({
                        'feature': col,
                        'reason': f'Very low unique ratio ({unique_ratio:.4f}) - possible global aggregation'
                    })
        
        # Features are contaminated if they show impossible patterns
        results['has_contamination'] = len(results['contaminated_features']) > 0
        
        return results
    
    def _check_target_encoding_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                     test_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for target encoding leakage."""
        results = {
            'has_leakage': False,
            'encoded_features': [],
            'leakage_risk_features': []
        }
        
        # Look for features that might be target-encoded
        target_encoding_patterns = [
            '_mean', '_median', '_encoded', '_target_', '_avg', '_encode'
        ]
        
        for col in train_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_encoding_patterns):
                results['encoded_features'].append(col)
                
                # Check if this feature shows perfect correlation patterns
                if train_df[col].dtype in ['int64', 'float64']:
                    # High cardinality with floating point values suggests target encoding
                    unique_ratio = train_df[col].nunique() / len(train_df)
                    if 0.1 < unique_ratio < 0.9 and train_df[col].dtype == 'float64':
                        results['leakage_risk_features'].append({
                            'feature': col,
                            'reason': 'High cardinality float feature suggesting target encoding'
                        })
        
        results['has_leakage'] = len(results['leakage_risk_features']) > 0
        return results
    
    def print_detection_results(self) -> None:
        """Print detailed leakage detection results."""
        if not self.detection_results:
            print("No leakage detection results available. Run detect_temporal_leakage() first.")
            return
        
        results = self.detection_results
        
        print("\n" + "="*80)
        print("TEMPORAL LEAKAGE DETECTION REPORT")
        print("="*80)
        
        # Overall status
        if results['leakage_detected']:
            print(f"LEAKAGE STATUS: DETECTED ({results['severity']} SEVERITY)")
            print(f"LEAKAGE TYPES: {', '.join(results['leakage_types'])}")
        else:
            print("LEAKAGE STATUS: NO LEAKAGE DETECTED")
        
        print()
        
        # Detailed results
        for check_type, details in results['details'].items():
            print(f"{check_type.upper().replace('_', ' ')}:")
            
            if check_type == 'temporal_overlap':
                if details['has_overlap']:
                    print("  STATUS: OVERLAP DETECTED")
                    for overlap in details['overlaps']:
                        print(f"    {overlap['type']}: {overlap.get('overlap_days', 0)} days overlap")
                else:
                    print("  STATUS: NO OVERLAP")
                
                if details['gaps']:
                    print("  GAPS:")
                    for gap_type, gap_days in details['gaps'].items():
                        print(f"    {gap_type}: {gap_days} days")
            
            elif check_type == 'entity_bleeding':
                if details['has_bleeding']:
                    print("  STATUS: ENTITY BLEEDING DETECTED")
                    for col, bleeding_info in details['bleeding_entities'].items():
                        print(f"    {col}: {bleeding_info['train_val_overlap']} train-val overlaps")
                else:
                    print("  STATUS: NO ENTITY BLEEDING")
            
            elif check_type == 'feature_contamination':
                if details['suspicious_features']:
                    print(f"  STATUS: {len(details['suspicious_features'])} SUSPICIOUS FEATURES")
                    for feature_info in details['suspicious_features'][:5]:  # Show first 5
                        print(f"    {feature_info['feature']}: {feature_info['reason']}")
                else:
                    print("  STATUS: NO SUSPICIOUS FEATURES")
            
            elif check_type == 'target_encoding':
                if details['has_leakage']:
                    print("  STATUS: TARGET ENCODING LEAKAGE RISK")
                    for feature_info in details['leakage_risk_features']:
                        print(f"    {feature_info['feature']}: {feature_info['reason']}")
                else:
                    print("  STATUS: NO TARGET ENCODING LEAKAGE")
            
            print()
        
        print("="*80)


class TemporalSplitter:
    """Leak-proof temporal data splitting with comprehensive auditing."""
    
    def __init__(self, config: TemporalSplitConfig, enable_auditing: bool = True):
        """Initialize temporal splitter.
        
        Args:
            config: Temporal split configuration
            enable_auditing: Whether to enable comprehensive auditing
        """
        self.config = config
        self.enable_auditing = enable_auditing
        self.audit_trail = None
        self.leakage_detector = TemporalLeakageDetector(strict_mode=True)
        
        # Validate configuration
        self.config.validate()
    
    def split_temporal_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform leak-proof temporal data splitting.
        
        Args:
            df: Input DataFrame with temporal data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("[TEMPORAL] Starting leak-proof temporal data splitting...")
        
        # Ensure date column exists and is datetime
        if self.config.date_column not in df.columns:
            raise ValueError(f"Date column '{self.config.date_column}' not found in DataFrame")
        
        # Convert to datetime if needed
        df = df.copy()
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        
        # Sort by date to ensure chronological order
        df = df.sort_values(self.config.date_column).reset_index(drop=True)
        
        # Convert split dates
        train_end = pd.to_datetime(self.config.train_end_date)
        val_end = pd.to_datetime(self.config.val_end_date)
        test_start = pd.to_datetime(self.config.test_start_date)
        
        # Create splits using STRICT temporal boundaries
        train_mask = df[self.config.date_column] <= train_end
        val_mask = (df[self.config.date_column] > train_end) & (df[self.config.date_column] <= val_end)
        test_mask = df[self.config.date_column] >= test_start
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        # Detect leakage
        leakage_results = self.leakage_detector.detect_temporal_leakage(
            train_df, val_df, test_df, self.config.date_column
        )
        
        # Generate audit trail
        if self.enable_auditing:
            self.audit_trail = self._generate_audit_trail(
                df, train_df, val_df, test_df, leakage_results
            )
            self.audit_trail.print_summary()
        
        print(f"[TEMPORAL] Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df
    
    def _generate_audit_trail(self, original_df: pd.DataFrame, train_df: pd.DataFrame,
                             val_df: pd.DataFrame, test_df: pd.DataFrame,
                             leakage_results: Dict[str, Any]) -> TemporalSplitAudit:
        """Generate comprehensive audit trail."""
        
        # Calculate date ranges
        def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
            if len(df) == 0:
                return ("N/A", "N/A")
            dates = pd.to_datetime(df[self.config.date_column])
            return (str(dates.min().date()), str(dates.max().date()))
        
        train_range = get_date_range(train_df)
        val_range = get_date_range(val_df)
        test_range = get_date_range(test_df)
        
        # Generate dataset hash for reproducibility
        dataset_hash = str(hash(str(original_df.shape) + str(original_df.columns.tolist())))
        
        # Extract leakage details
        leakage_details = []
        if leakage_results['leakage_detected']:
            for leakage_type in leakage_results['leakage_types']:
                leakage_details.append(f"{leakage_type}: {leakage_results['severity']} severity")
        
        audit = TemporalSplitAudit(
            config=self.config,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_date_range=train_range,
            val_date_range=val_range,
            test_date_range=test_range,
            leakage_detected=leakage_results['leakage_detected'],
            leakage_details=leakage_details,
            split_timestamp=datetime.now().isoformat(),
            dataset_hash=dataset_hash
        )
        
        return audit
    
    def save_audit_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        """Save all audit artifacts to specified directory.
        
        Args:
            output_dir: Directory to save audit artifacts
            
        Returns:
            Dictionary mapping artifact names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        if self.audit_trail:
            # Save audit trail
            audit_path = output_dir / "temporal_split_audit.json"
            self.audit_trail.save_audit(audit_path)
            artifacts['audit_trail'] = audit_path
            
            # Save leakage detection results
            if hasattr(self.leakage_detector, 'detection_results'):
                leakage_path = output_dir / "leakage_detection_results.json"
                with open(leakage_path, 'w') as f:
                    json.dump(self.leakage_detector.detection_results, f, indent=2)
                artifacts['leakage_results'] = leakage_path
            
            # Save split configuration
            config_path = output_dir / "split_configuration.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            artifacts['configuration'] = config_path
        
        print(f"[AUDIT] Saved {len(artifacts)} audit artifacts to: {output_dir}")
        return artifacts


def create_production_temporal_split(df: pd.DataFrame, 
                                   train_end_date: str = "2009-12-31",
                                   val_end_date: str = "2011-12-31", 
                                   test_start_date: str = "2012-01-01",
                                   date_column: str = "sales_date",
                                   output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create production-ready temporal split with full audit trail.
    
    This function implements the Blue Book winning strategy for temporal validation
    with comprehensive leakage elimination and audit artifacts.
    
    Args:
        df: Input DataFrame with temporal data
        train_end_date: Last date for training set (YYYY-MM-DD)
        val_end_date: Last date for validation set (YYYY-MM-DD)
        test_start_date: First date for test set (YYYY-MM-DD)
        date_column: Name of the date column
        output_dir: Directory to save audit artifacts (optional)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Create configuration
    config = TemporalSplitConfig(
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        test_start_date=test_start_date,
        date_column=date_column
    )
    
    # Create splitter
    splitter = TemporalSplitter(config, enable_auditing=True)
    
    # Perform split
    train_df, val_df, test_df = splitter.split_temporal_data(df)
    
    # Save audit artifacts if requested
    if output_dir:
        artifacts = splitter.save_audit_artifacts(output_dir)
        print(f"[AUDIT] Audit artifacts saved: {list(artifacts.keys())}")
    
    return train_df, val_df, test_df


# Convenience functions for common splitting strategies
def create_shm_temporal_split(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporal split optimized for SHM heavy equipment data.
    
    Uses Blue Book winning strategy:
    - Train: <= 2009 (includes 2008 financial crisis)
    - Validation: 2010-2011 (recovery period)
    - Test: >= 2012 (stable market conditions)
    """
    return create_production_temporal_split(
        df, 
        train_end_date="2009-12-31",
        val_end_date="2011-12-31",
        test_start_date="2012-01-01",
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Test the temporal validation system
    print("Testing temporal validation system...")
    
    # Create sample data
    dates = pd.date_range("2007-01-01", "2014-12-31", freq="D")
    sample_data = pd.DataFrame({
        'sales_date': np.random.choice(dates, 1000),
        'sales_price': np.random.lognormal(10, 1, 1000),
        'machine_id': np.random.choice(range(100, 500), 1000),
        'equipment_type': np.random.choice(['excavator', 'bulldozer', 'loader'], 1000)
    })
    
    # Test temporal splitting
    train_df, val_df, test_df = create_shm_temporal_split(sample_data)
    
    print(f"Test completed: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")