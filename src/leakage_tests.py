"""Automated temporal leakage detection tests.

This module provides comprehensive automated testing for temporal leakage detection
in machine learning pipelines. It includes various test scenarios and validation
methods to ensure data integrity and temporal validity.

CRITICAL MISSION: Automated detection of ALL leakage types.

Key Features:
- Comprehensive test suite for temporal leakage detection
- Automated validation of data splits and preprocessing
- Feature contamination detection
- Target encoding leakage tests
- Entity bleeding detection
- Pipeline integrity validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import unittest
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class LeakageTestResult:
    """Result of a leakage test."""
    test_name: str
    passed: bool
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    details: str
    recommendations: List[str]
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'severity': self.severity,
            'details': self.details,
            'recommendations': self.recommendations,
            'evidence': self.evidence
        }


class TemporalLeakageTestSuite:
    """Comprehensive test suite for temporal leakage detection."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize test suite.
        
        Args:
            strict_mode: If True, treat any leakage as critical failure
        """
        self.strict_mode = strict_mode
        self.test_results = []
        self.overall_status = "UNKNOWN"
        
    def run_all_tests(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, date_column: str = 'sales_date',
                     target_column: str = 'sales_price') -> Dict[str, Any]:
        """Run all temporal leakage tests.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            date_column: Name of date column
            target_column: Name of target column
            
        Returns:
            Dictionary with comprehensive test results
        """
        print("[LEAKAGE-TESTS] Running comprehensive temporal leakage test suite...")
        
        self.test_results = []
        
        # Test 1: Temporal boundary validation
        self.test_results.append(self._test_temporal_boundaries(
            train_df, val_df, test_df, date_column
        ))
        
        # Test 2: Entity bleeding detection
        self.test_results.append(self._test_entity_bleeding(
            train_df, val_df, test_df
        ))
        
        # Test 3: Feature contamination detection
        self.test_results.append(self._test_feature_contamination(
            train_df, val_df, test_df, date_column, target_column
        ))
        
        # Test 4: Target encoding leakage
        self.test_results.append(self._test_target_encoding_leakage(
            train_df, val_df, test_df, target_column
        ))
        
        # Test 5: Preprocessing leakage
        self.test_results.append(self._test_preprocessing_leakage(
            train_df, val_df, test_df
        ))
        
        # Test 6: Distribution shift validation
        self.test_results.append(self._test_distribution_shifts(
            train_df, val_df, test_df, date_column
        ))
        
        # Test 7: Future information detection
        self.test_results.append(self._test_future_information(
            train_df, val_df, test_df, date_column
        ))
        
        # Determine overall status
        self._determine_overall_status()
        
        return self._compile_test_report()
    
    def _test_temporal_boundaries(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                test_df: pd.DataFrame, date_column: str) -> LeakageTestResult:
        """Test temporal boundary integrity."""
        try:
            # Ensure date columns are datetime
            train_dates = pd.to_datetime(train_df[date_column])
            val_dates = pd.to_datetime(val_df[date_column]) if len(val_df) > 0 else pd.Series(dtype='datetime64[ns]')
            test_dates = pd.to_datetime(test_df[date_column]) if len(test_df) > 0 else pd.Series(dtype='datetime64[ns]')
            
            overlaps = []
            
            # Check train-validation overlap
            if len(val_dates) > 0:
                train_max = train_dates.max()
                val_min = val_dates.min()
                if train_max >= val_min:
                    overlaps.append(f"Train-Val overlap: {(train_max - val_min).days} days")
            
            # Check validation-test overlap
            if len(val_dates) > 0 and len(test_dates) > 0:
                val_max = val_dates.max()
                test_min = test_dates.min()
                if val_max >= test_min:
                    overlaps.append(f"Val-Test overlap: {(val_max - test_min).days} days")
            
            # Check train-test overlap (should never happen)
            if len(test_dates) > 0:
                train_max = train_dates.max()
                test_min = test_dates.min()
                if train_max >= test_min:
                    overlaps.append(f"Train-Test overlap: {(train_max - test_min).days} days")
            
            if overlaps:
                return LeakageTestResult(
                    test_name="Temporal Boundary Validation",
                    passed=False,
                    severity="CRITICAL",
                    details=f"Temporal overlaps detected: {'; '.join(overlaps)}",
                    recommendations=[
                        "Fix temporal split boundaries",
                        "Ensure strict chronological separation",
                        "Add gap between splits if necessary"
                    ],
                    evidence={'overlaps': overlaps}
                )
            else:
                # Calculate gaps for evidence
                gaps = {}
                if len(val_dates) > 0:
                    gaps['train_val_gap'] = (val_dates.min() - train_dates.max()).days
                if len(test_dates) > 0:
                    if len(val_dates) > 0:
                        gaps['val_test_gap'] = (test_dates.min() - val_dates.max()).days
                    else:
                        gaps['train_test_gap'] = (test_dates.min() - train_dates.max()).days
                
                return LeakageTestResult(
                    test_name="Temporal Boundary Validation",
                    passed=True,
                    severity="NONE",
                    details="No temporal overlaps detected. All splits maintain chronological order.",
                    recommendations=[],
                    evidence={'gaps': gaps}
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Temporal Boundary Validation",
                passed=False,
                severity="HIGH",
                details=f"Error during temporal boundary validation: {str(e)}",
                recommendations=["Check date column format and data quality"],
                evidence={'error': str(e)}
            )
    
    def _test_entity_bleeding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> LeakageTestResult:
        """Test for entity bleeding across splits."""
        try:
            # Identify entity columns
            entity_candidates = ['sales_id', 'machine_id', 'equipment_id', 'serial_number', 'vin']
            entity_columns = [col for col in entity_candidates if col in train_df.columns]
            
            bleeding_detected = False
            bleeding_details = []
            evidence = {}
            
            for col in entity_columns:
                train_entities = set(train_df[col].dropna().astype(str))
                val_entities = set(val_df[col].dropna().astype(str)) if len(val_df) > 0 else set()
                test_entities = set(test_df[col].dropna().astype(str)) if len(test_df) > 0 else set()
                
                # Check overlaps
                train_val_overlap = len(train_entities & val_entities)
                train_test_overlap = len(train_entities & test_entities)
                val_test_overlap = len(val_entities & test_entities)
                
                if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
                    bleeding_detected = True
                    bleeding_details.append(
                        f"{col}: train-val={train_val_overlap}, train-test={train_test_overlap}, val-test={val_test_overlap}"
                    )
                
                evidence[col] = {
                    'train_unique': len(train_entities),
                    'val_unique': len(val_entities),
                    'test_unique': len(test_entities),
                    'train_val_overlap': train_val_overlap,
                    'train_test_overlap': train_test_overlap,
                    'val_test_overlap': val_test_overlap
                }
            
            if bleeding_detected:
                return LeakageTestResult(
                    test_name="Entity Bleeding Detection",
                    passed=False,
                    severity="HIGH",
                    details=f"Entity bleeding detected in {len([d for d in bleeding_details])} columns: {'; '.join(bleeding_details)}",
                    recommendations=[
                        "Remove duplicate entities across splits",
                        "Consider entity-based splitting strategy",
                        "Ensure temporal splits respect entity boundaries"
                    ],
                    evidence=evidence
                )
            else:
                return LeakageTestResult(
                    test_name="Entity Bleeding Detection",
                    passed=True,
                    severity="NONE",
                    details=f"No entity bleeding detected across {len(entity_columns)} entity columns.",
                    recommendations=[],
                    evidence=evidence
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Entity Bleeding Detection",
                passed=False,
                severity="MEDIUM",
                details=f"Error during entity bleeding detection: {str(e)}",
                recommendations=["Check entity column data types and quality"],
                evidence={'error': str(e)}
            )
    
    def _test_feature_contamination(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                  test_df: pd.DataFrame, date_column: str, 
                                  target_column: str) -> LeakageTestResult:
        """Test for feature contamination with future information."""
        try:
            contaminated_features = []
            suspicious_features = []
            evidence = {}
            
            # Check for suspicious feature names
            suspicious_patterns = [
                'future_', 'next_', 'ahead_', 'forward_', 'total_', 'global_', 'overall_'
            ]
            
            for col in train_df.columns:
                if col in [date_column, target_column]:
                    continue
                
                col_lower = col.lower()
                
                # Check naming patterns
                is_suspicious = any(pattern in col_lower for pattern in suspicious_patterns)
                if is_suspicious:
                    suspicious_features.append(col)
                
                # Check for global aggregations (same value across many rows)
                if train_df[col].dtype in ['int64', 'float64']:
                    unique_ratio = train_df[col].nunique() / len(train_df)
                    if unique_ratio < 0.01 and train_df[col].nunique() > 1:
                        contaminated_features.append(col)
                        evidence[col] = {
                            'issue': 'low_unique_ratio',
                            'unique_ratio': unique_ratio,
                            'unique_values': train_df[col].nunique()
                        }
                
                # Check for perfect correlations with target
                if target_column in train_df.columns:
                    target_values = train_df[target_column].dropna()
                    feature_values = train_df[col].dropna()
                    
                    if len(target_values) > 10 and len(feature_values) > 10:
                        # Align indices for correlation calculation
                        common_idx = target_values.index.intersection(feature_values.index)
                        if len(common_idx) > 10:
                            corr = np.corrcoef(
                                target_values.loc[common_idx],
                                feature_values.loc[common_idx]
                            )[0, 1]
                            
                            if abs(corr) > 0.98:
                                contaminated_features.append(col)
                                evidence[col] = {
                                    'issue': 'perfect_correlation',
                                    'correlation': corr
                                }
            
            total_issues = len(contaminated_features) + len(suspicious_features)
            
            if total_issues > 0:
                severity = "CRITICAL" if len(contaminated_features) > 0 else "MEDIUM"
                details = f"Feature contamination detected: {len(contaminated_features)} contaminated, {len(suspicious_features)} suspicious features"
                recommendations = [
                    "Remove or re-engineer contaminated features",
                    "Review feature engineering pipeline for future information",
                    "Use only past data for feature calculations"
                ]
            else:
                severity = "NONE"
                details = "No feature contamination detected"
                recommendations = []
            
            return LeakageTestResult(
                test_name="Feature Contamination Detection",
                passed=(total_issues == 0),
                severity=severity,
                details=details,
                recommendations=recommendations,
                evidence={
                    'contaminated_features': contaminated_features,
                    'suspicious_features': suspicious_features,
                    'feature_evidence': evidence
                }
            )
            
        except Exception as e:
            return LeakageTestResult(
                test_name="Feature Contamination Detection",
                passed=False,
                severity="MEDIUM",
                details=f"Error during feature contamination detection: {str(e)}",
                recommendations=["Check feature data types and quality"],
                evidence={'error': str(e)}
            )
    
    def _test_target_encoding_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                    test_df: pd.DataFrame, target_column: str) -> LeakageTestResult:
        """Test for target encoding leakage."""
        try:
            leakage_features = []
            evidence = {}
            
            # Look for target-encoded features
            target_encoding_patterns = [
                '_mean', '_median', '_encoded', '_target_', '_avg', '_encode'
            ]
            
            for col in train_df.columns:
                if col == target_column:
                    continue
                
                col_lower = col.lower()
                is_target_encoded = any(pattern in col_lower for pattern in target_encoding_patterns)
                
                if is_target_encoded and target_column in train_df.columns:
                    # Check for suspicious characteristics
                    feature_values = train_df[col].dropna()
                    target_values = train_df[target_column].dropna()
                    
                    if len(feature_values) > 10 and len(target_values) > 10:
                        # Check correlation
                        common_idx = feature_values.index.intersection(target_values.index)
                        if len(common_idx) > 10:
                            corr = np.corrcoef(
                                feature_values.loc[common_idx],
                                target_values.loc[common_idx]
                            )[0, 1]
                            
                            # Check for high cardinality with float values
                            unique_ratio = feature_values.nunique() / len(feature_values)
                            is_float = feature_values.dtype == 'float64'
                            
                            if (abs(corr) > 0.9 or 
                                (unique_ratio > 0.5 and is_float and abs(corr) > 0.7)):
                                leakage_features.append(col)
                                evidence[col] = {
                                    'correlation': corr,
                                    'unique_ratio': unique_ratio,
                                    'is_float': is_float
                                }
            
            if leakage_features:
                return LeakageTestResult(
                    test_name="Target Encoding Leakage Detection",
                    passed=False,
                    severity="HIGH",
                    details=f"Target encoding leakage detected in {len(leakage_features)} features: {', '.join(leakage_features[:3])}",
                    recommendations=[
                        "Use time-aware target encoding with proper temporal splits",
                        "Apply cross-validation for target encoding within training set only",
                        "Add regularization or smoothing to target encodings"
                    ],
                    evidence=evidence
                )
            else:
                return LeakageTestResult(
                    test_name="Target Encoding Leakage Detection",
                    passed=True,
                    severity="NONE",
                    details="No target encoding leakage detected",
                    recommendations=[],
                    evidence={}
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Target Encoding Leakage Detection",
                passed=False,
                severity="MEDIUM",
                details=f"Error during target encoding leakage detection: {str(e)}",
                recommendations=["Check feature naming and data quality"],
                evidence={'error': str(e)}
            )
    
    def _test_preprocessing_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                  test_df: pd.DataFrame) -> LeakageTestResult:
        """Test for preprocessing leakage (normalization, scaling computed on all data)."""
        try:
            # This test checks for signs that preprocessing was done on the entire dataset
            # Look for identical scaling patterns across splits
            
            suspicious_features = []
            evidence = {}
            
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_columns:
                if col in val_df.columns and col in test_df.columns:
                    train_stats = {
                        'mean': train_df[col].mean(),
                        'std': train_df[col].std(),
                        'min': train_df[col].min(),
                        'max': train_df[col].max()
                    }
                    
                    val_stats = {
                        'mean': val_df[col].mean(),
                        'std': val_df[col].std(),
                        'min': val_df[col].min(),
                        'max': val_df[col].max()
                    }
                    
                    test_stats = {
                        'mean': test_df[col].mean(),
                        'std': test_df[col].std(),
                        'min': test_df[col].min(),
                        'max': test_df[col].max()
                    }
                    
                    # Check for suspiciously similar statistics (might indicate global scaling)
                    # Standard deviations should not be identical across splits
                    if (pd.notna(train_stats['std']) and pd.notna(val_stats['std']) and 
                        pd.notna(test_stats['std'])):
                        
                        std_similarity = (
                            abs(train_stats['std'] - val_stats['std']) < 0.001 and
                            abs(val_stats['std'] - test_stats['std']) < 0.001
                        )
                        
                        # Also check for identical ranges (suspicious for scaled data)
                        range_similarity = (
                            abs((train_stats['max'] - train_stats['min']) - 
                                (val_stats['max'] - val_stats['min'])) < 0.001 and
                            abs((val_stats['max'] - val_stats['min']) - 
                                (test_stats['max'] - test_stats['min'])) < 0.001
                        )
                        
                        if std_similarity and range_similarity:
                            suspicious_features.append(col)
                            evidence[col] = {
                                'train_stats': train_stats,
                                'val_stats': val_stats,
                                'test_stats': test_stats
                            }
            
            if suspicious_features:
                return LeakageTestResult(
                    test_name="Preprocessing Leakage Detection",
                    passed=False,
                    severity="MEDIUM",
                    details=f"Preprocessing leakage suspected in {len(suspicious_features)} features with identical statistics across splits",
                    recommendations=[
                        "Fit preprocessing transformations only on training data",
                        "Apply fitted transformations to validation and test sets",
                        "Use pipeline-based preprocessing to prevent leakage"
                    ],
                    evidence=evidence
                )
            else:
                return LeakageTestResult(
                    test_name="Preprocessing Leakage Detection",
                    passed=True,
                    severity="NONE",
                    details="No obvious preprocessing leakage detected",
                    recommendations=[],
                    evidence={}
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Preprocessing Leakage Detection",
                passed=False,
                severity="LOW",
                details=f"Error during preprocessing leakage detection: {str(e)}",
                recommendations=["Check data quality and preprocessing pipeline"],
                evidence={'error': str(e)}
            )
    
    def _test_distribution_shifts(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                test_df: pd.DataFrame, date_column: str) -> LeakageTestResult:
        """Test for unexpected distribution shifts that might indicate leakage."""
        try:
            # Check for temporal consistency in feature distributions
            shift_warnings = []
            evidence = {}
            
            # Check a few key numerical features
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns
            check_cols = list(numerical_cols)[:5]  # Limit to avoid too much computation
            
            for col in check_cols:
                if col in val_df.columns and col in test_df.columns:
                    train_mean = train_df[col].mean()
                    val_mean = val_df[col].mean() if len(val_df) > 0 else train_mean
                    test_mean = test_df[col].mean() if len(test_df) > 0 else train_mean
                    
                    # Check for dramatic shifts (could indicate leakage or data quality issues)
                    if pd.notna(train_mean) and train_mean != 0:
                        val_shift = abs(val_mean - train_mean) / abs(train_mean)
                        test_shift = abs(test_mean - train_mean) / abs(train_mean)
                        
                        if val_shift > 2.0 or test_shift > 2.0:  # 200% shift
                            shift_warnings.append(f"{col}: val_shift={val_shift:.2f}, test_shift={test_shift:.2f}")
                            evidence[col] = {
                                'train_mean': train_mean,
                                'val_mean': val_mean,
                                'test_mean': test_mean,
                                'val_shift_ratio': val_shift,
                                'test_shift_ratio': test_shift
                            }
            
            if shift_warnings:
                return LeakageTestResult(
                    test_name="Distribution Shift Validation",
                    passed=False,
                    severity="MEDIUM",
                    details=f"Dramatic distribution shifts detected: {'; '.join(shift_warnings)}",
                    recommendations=[
                        "Investigate causes of distribution shifts",
                        "Check for data quality issues",
                        "Consider if shifts indicate temporal leakage",
                        "Validate temporal split boundaries"
                    ],
                    evidence=evidence
                )
            else:
                return LeakageTestResult(
                    test_name="Distribution Shift Validation",
                    passed=True,
                    severity="NONE",
                    details="No dramatic distribution shifts detected",
                    recommendations=[],
                    evidence={}
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Distribution Shift Validation",
                passed=False,
                severity="LOW",
                details=f"Error during distribution shift validation: {str(e)}",
                recommendations=["Check data quality and feature types"],
                evidence={'error': str(e)}
            )
    
    def _test_future_information(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame, date_column: str) -> LeakageTestResult:
        """Test for features that contain future information."""
        try:
            future_info_features = []
            evidence = {}
            
            # Check for dates in the future relative to split dates
            train_max_date = pd.to_datetime(train_df[date_column]).max()
            
            # Look for date-like columns that might contain future information
            date_like_columns = [col for col in train_df.columns 
                               if 'date' in col.lower() or 'time' in col.lower()]
            
            for col in date_like_columns:
                if col != date_column:
                    try:
                        col_dates = pd.to_datetime(train_df[col], errors='coerce')
                        future_dates = col_dates[col_dates > train_max_date]
                        
                        if len(future_dates) > 0:
                            future_info_features.append(col)
                            evidence[col] = {
                                'future_date_count': len(future_dates),
                                'max_future_date': str(future_dates.max()),
                                'train_max_date': str(train_max_date)
                            }
                    except:
                        # Not a date column, skip
                        pass
            
            # Check for suspiciously perfect features
            if len(train_df) > 100:  # Only for reasonably sized datasets
                for col in train_df.columns:
                    if col in [date_column] or train_df[col].dtype not in ['int64', 'float64']:
                        continue
                    
                    # Check if feature values increase monotonically with date
                    # (might indicate future information leak)
                    try:
                        sorted_by_date = train_df.sort_values(date_column)
                        feature_values = sorted_by_date[col].dropna()
                        
                        if len(feature_values) > 50:
                            # Check correlation with row order (time)
                            row_order = np.arange(len(feature_values))
                            correlation = np.corrcoef(feature_values, row_order)[0, 1]
                            
                            if abs(correlation) > 0.95:
                                future_info_features.append(col)
                                evidence[col] = {
                                    'issue': 'perfect_temporal_correlation',
                                    'correlation': correlation
                                }
                    except:
                        # Skip if error in calculation
                        pass
            
            if future_info_features:
                return LeakageTestResult(
                    test_name="Future Information Detection",
                    passed=False,
                    severity="HIGH",
                    details=f"Future information detected in {len(future_info_features)} features: {', '.join(future_info_features[:3])}",
                    recommendations=[
                        "Remove features containing future information",
                        "Validate date columns for temporal consistency",
                        "Review feature engineering for forward-looking calculations"
                    ],
                    evidence=evidence
                )
            else:
                return LeakageTestResult(
                    test_name="Future Information Detection",
                    passed=True,
                    severity="NONE",
                    details="No future information detected in features",
                    recommendations=[],
                    evidence={}
                )
                
        except Exception as e:
            return LeakageTestResult(
                test_name="Future Information Detection",
                passed=False,
                severity="LOW",
                details=f"Error during future information detection: {str(e)}",
                recommendations=["Check date columns and feature formats"],
                evidence={'error': str(e)}
            )
    
    def _determine_overall_status(self) -> None:
        """Determine overall test suite status."""
        critical_failures = [r for r in self.test_results if not r.passed and r.severity == "CRITICAL"]
        high_failures = [r for r in self.test_results if not r.passed and r.severity == "HIGH"]
        medium_failures = [r for r in self.test_results if not r.passed and r.severity == "MEDIUM"]
        
        if critical_failures:
            self.overall_status = "CRITICAL_LEAKAGE_DETECTED"
        elif high_failures:
            self.overall_status = "HIGH_LEAKAGE_DETECTED"
        elif medium_failures:
            self.overall_status = "MEDIUM_LEAKAGE_DETECTED"
        else:
            self.overall_status = "NO_LEAKAGE_DETECTED"
    
    def _compile_test_report(self) -> Dict[str, Any]:
        """Compile comprehensive test report."""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        severity_counts = {}
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
            severity_counts[severity] = len([r for r in self.test_results if r.severity == severity])
        
        return {
            'overall_status': self.overall_status,
            'total_tests': len(self.test_results),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'severity_counts': severity_counts,
            'test_results': [r.to_dict() for r in self.test_results],
            'summary': {
                'leakage_detected': self.overall_status != "NO_LEAKAGE_DETECTED",
                'critical_issues': [r.test_name for r in self.test_results 
                                  if not r.passed and r.severity == "CRITICAL"],
                'high_issues': [r.test_name for r in self.test_results 
                               if not r.passed and r.severity == "HIGH"],
                'recommendations': self._get_top_recommendations()
            }
        }
    
    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations from failed tests."""
        all_recommendations = []
        for result in self.test_results:
            if not result.passed:
                all_recommendations.extend(result.recommendations)
        
        # Return unique recommendations, prioritizing by frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        return sorted(recommendation_counts.keys(), 
                     key=lambda x: recommendation_counts[x], reverse=True)[:5]
    
    def print_test_report(self) -> None:
        """Print comprehensive test report."""
        print("\n" + "="*80)
        print("TEMPORAL LEAKAGE TEST REPORT")
        print("="*80)
        
        print(f"OVERALL STATUS: {self.overall_status}")
        print(f"Tests Run: {len(self.test_results)}")
        print(f"Passed: {len([r for r in self.test_results if r.passed])}")
        print(f"Failed: {len([r for r in self.test_results if not r.passed])}")
        print()
        
        # Show results by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            severity_tests = [r for r in self.test_results if r.severity == severity]
            if severity_tests:
                print(f"{severity} SEVERITY:")
                for test in severity_tests:
                    status = "PASS" if test.passed else "FAIL"
                    print(f"  [{status}] {test.test_name}: {test.details}")
                print()
        
        # Show recommendations if any failures
        failed_tests = [r for r in self.test_results if not r.passed]
        if failed_tests:
            print("TOP RECOMMENDATIONS:")
            recommendations = self._get_top_recommendations()
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*80)
    
    def save_report(self, filepath: Path) -> None:
        """Save test report to JSON file."""
        report = self._compile_test_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[LEAKAGE-TESTS] Test report saved to: {filepath}")


def run_leakage_tests(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                     date_column: str = 'sales_date', target_column: str = 'sales_price',
                     output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run comprehensive leakage tests and return results.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        date_column: Date column name
        target_column: Target column name
        output_dir: Directory to save test report (optional)
        
    Returns:
        Dictionary with test results
    """
    test_suite = TemporalLeakageTestSuite(strict_mode=True)
    results = test_suite.run_all_tests(train_df, val_df, test_df, date_column, target_column)
    
    test_suite.print_test_report()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        test_suite.save_report(output_dir / "leakage_test_report.json")
    
    return results


if __name__ == "__main__":
    # Test the leakage detection system
    print("Testing temporal leakage detection system...")
    
    # Create sample data with known leakage issues
    np.random.seed(42)
    n_samples = 300
    
    # Create temporal data
    dates = pd.date_range("2008-01-01", "2012-12-31", periods=n_samples)
    machine_ids = np.random.choice(range(100, 200), n_samples)
    prices = np.random.lognormal(10, 0.5, n_samples)
    
    # Create intentional leakage for testing
    future_info = np.arange(n_samples)  # Monotonically increasing (suspicious)
    global_mean = [prices.mean()] * n_samples  # Global statistic (leakage)
    
    # Create test dataframes
    full_df = pd.DataFrame({
        'sales_date': dates,
        'sales_price': prices,
        'machine_id': machine_ids,
        'equipment_type': np.random.choice(['A', 'B', 'C'], n_samples),
        'suspicious_feature': future_info,
        'global_mean_feature': global_mean,
        'normal_feature': np.random.normal(0, 1, n_samples)
    })
    
    # Create temporal splits with intentional overlap for testing
    train_df = full_df[full_df['sales_date'] < '2010-06-01'].copy()
    val_df = full_df[(full_df['sales_date'] >= '2010-01-01') & 
                     (full_df['sales_date'] < '2011-01-01')].copy()  # Intentional overlap
    test_df = full_df[full_df['sales_date'] >= '2010-06-01'].copy()  # Intentional overlap
    
    # Run tests
    test_results = run_leakage_tests(train_df, val_df, test_df)
    
    print(f"\nTest completed. Overall status: {test_results['overall_status']}")
    print(f"Failed tests: {test_results['failed_tests']}")