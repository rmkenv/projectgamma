"""
Data utility functions for processing and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import re
from pathlib import Path

from .logger import get_logger

logger = get_logger()


def validate_dataset_columns(data: pd.DataFrame, expected_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate that dataset has expected columns for pipeline data.
    
    Args:
        data: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        Dictionary with validation results
    """
    
    if expected_columns is None:
        expected_columns = [
            'pipeline_name', 'loc_name', 'connecting_pipeline', 'connecting_entity',
            'rec_del_sign', 'category_short', 'country_name', 'state_abb', 
            'county_name', 'latitude', 'longitude', 'eff_gas_day', 'scheduled_quantity'
        ]
    
    results = {
        'is_valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'column_mapping_suggestions': {},
        'data_types': {}
    }
    
    actual_columns = set(data.columns)
    expected_columns_set = set(expected_columns)
    
    # Find missing and extra columns
    results['missing_columns'] = list(expected_columns_set - actual_columns)
    results['extra_columns'] = list(actual_columns - expected_columns_set)
    
    # Suggest column mappings for missing columns
    for missing_col in results['missing_columns']:
        suggestion = find_similar_column(missing_col, actual_columns)
        if suggestion:
            results['column_mapping_suggestions'][missing_col] = suggestion
    
    # Analyze data types
    for col in data.columns:
        col_info = analyze_column_type(data[col])
        results['data_types'][col] = col_info
    
    # Set validity
    if len(results['missing_columns']) > 5:  # Allow some flexibility
        results['is_valid'] = False
    
    return results


def find_similar_column(target: str, available_columns: List[str], threshold: float = 0.6) -> Optional[str]:
    """
    Find the most similar column name using fuzzy matching.
    
    Args:
        target: Target column name to find
        available_columns: List of available column names
        threshold: Similarity threshold (0-1)
        
    Returns:
        Best matching column name or None
    """
    
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity score between two strings."""
        s1, s2 = s1.lower(), s2.lower()
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Substring match
        if s1 in s2 or s2 in s1:
            return 0.8
        
        # Word overlap
        words1 = set(re.split(r'[_\s]+', s1))
        words2 = set(re.split(r'[_\s]+', s2))
        overlap = len(words1 & words2)
        total_words = len(words1 | words2)
        
        if total_words > 0:
            return overlap / total_words
        
        return 0.0
    
    best_match = None
    best_score = 0.0
    
    for col in available_columns:
        score = similarity_score(target, col)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = col
    
    return best_match


def analyze_column_type(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze the data type and characteristics of a pandas Series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Dictionary with column analysis
    """
    
    analysis = {
        'pandas_dtype': str(series.dtype),
        'null_count': series.isnull().sum(),
        'null_percentage': (series.isnull().sum() / len(series)) * 100,
        'unique_count': series.nunique(),
        'inferred_type': 'unknown'
    }
    
    # Remove null values for analysis
    non_null_series = series.dropna()
    
    if len(non_null_series) == 0:
        analysis['inferred_type'] = 'empty'
        return analysis
    
    # Infer data type
    if pd.api.types.is_numeric_dtype(series):
        analysis['inferred_type'] = 'numeric'
        analysis['min_value'] = non_null_series.min()
        analysis['max_value'] = non_null_series.max()
        analysis['mean_value'] = non_null_series.mean()
        
        # Check if it could be an ID or code
        if analysis['unique_count'] == len(non_null_series):
            analysis['inferred_type'] = 'identifier'
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        analysis['inferred_type'] = 'datetime'
        analysis['min_date'] = non_null_series.min()
        analysis['max_date'] = non_null_series.max()
    
    else:
        # String/object analysis
        sample_values = non_null_series.head(100).astype(str)
        
        # Check for date patterns
        if is_date_column(sample_values):
            analysis['inferred_type'] = 'date_string'
        
        # Check for categorical data
        elif analysis['unique_count'] < len(non_null_series) * 0.5:
            analysis['inferred_type'] = 'categorical'
            analysis['top_values'] = non_null_series.value_counts().head(10).to_dict()
        
        # Check for coordinates
        elif series.name and any(coord in series.name.lower() for coord in ['lat', 'lon', 'coord']):
            try:
                numeric_values = pd.to_numeric(non_null_series, errors='coerce')
                if not numeric_values.isnull().all():
                    analysis['inferred_type'] = 'coordinate'
            except:
                pass
        
        else:
            analysis['inferred_type'] = 'text'
    
    return analysis


def is_date_column(sample_values: pd.Series) -> bool:
    """
    Check if a series contains date strings.
    
    Args:
        sample_values: Sample values from the series
        
    Returns:
        True if likely contains dates
    """
    
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
    ]
    
    total_samples = len(sample_values)
    date_matches = 0
    
    for value in sample_values:
        value_str = str(value)
        for pattern in date_patterns:
            if re.match(pattern, value_str):
                date_matches += 1
                break
    
    return (date_matches / total_samples) > 0.5 if total_samples > 0 else False


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.
    
    Args:
        data: DataFrame with columns to clean
        
    Returns:
        DataFrame with cleaned column names
    """
    
    cleaned_data = data.copy()
    
    # Clean column names
    new_columns = []
    for col in cleaned_data.columns:
        # Convert to lowercase and replace spaces/special chars with underscores
        clean_col = re.sub(r'[^\w\s]', '', str(col).lower())
        clean_col = re.sub(r'\s+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)  # Remove multiple underscores
        clean_col = clean_col.strip('_')  # Remove leading/trailing underscores
        
        new_columns.append(clean_col)
    
    cleaned_data.columns = new_columns
    
    return cleaned_data


def detect_data_quality_issues(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect common data quality issues.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary with data quality issues found
    """
    
    issues = {
        'missing_data': {},
        'duplicate_rows': 0,
        'inconsistent_formats': {},
        'outliers': {},
        'suspicious_values': {}
    }
    
    # Missing data analysis
    missing_counts = data.isnull().sum()
    issues['missing_data'] = {
        col: {'count': count, 'percentage': (count / len(data)) * 100}
        for col, count in missing_counts.items() if count > 0
    }
    
    # Duplicate rows
    issues['duplicate_rows'] = data.duplicated().sum()
    
    # Check for inconsistent formats in string columns
    for col in data.select_dtypes(include=['object']).columns:
        col_data = data[col].dropna().astype(str)
        
        # Check for mixed case in what should be standardized fields
        if col.lower() in ['state_abb', 'country_code', 'category_short']:
            unique_values = col_data.unique()
            if len(unique_values) != len([v.upper() for v in unique_values]):
                issues['inconsistent_formats'][col] = 'Mixed case in standardized field'
        
        # Check for leading/trailing spaces
        has_spaces = col_data.str.strip() != col_data
        if has_spaces.any():
            issues['inconsistent_formats'][f'{col}_spaces'] = 'Leading/trailing spaces found'
    
    # Check for outliers in numeric columns
    for col in data.select_dtypes(include=[np.number]).columns:
        col_data = data[col].dropna()
        if len(col_data) > 10:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_count = len(col_data[(col_data < Q1 - 3*IQR) | (col_data > Q3 + 3*IQR)])
            if outlier_count > 0:
                issues['outliers'][col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(col_data)) * 100
                }
    
    return issues


def suggest_data_fixes(quality_issues: Dict[str, Any]) -> List[str]:
    """
    Suggest fixes for detected data quality issues.
    
    Args:
        quality_issues: Output from detect_data_quality_issues
        
    Returns:
        List of suggested fixes
    """
    
    suggestions = []
    
    # Missing data suggestions
    if quality_issues['missing_data']:
        high_missing = [col for col, info in quality_issues['missing_data'].items() 
                       if info['percentage'] > 50]
        if high_missing:
            suggestions.append(f"Consider removing columns with >50% missing data: {high_missing}")
        
        moderate_missing = [col for col, info in quality_issues['missing_data'].items() 
                           if 10 < info['percentage'] <= 50]
        if moderate_missing:
            suggestions.append(f"Consider imputation strategies for columns: {moderate_missing}")
    
    # Duplicate rows
    if quality_issues['duplicate_rows'] > 0:
        suggestions.append(f"Remove {quality_issues['duplicate_rows']} duplicate rows")
    
    # Format inconsistencies
    if quality_issues['inconsistent_formats']:
        suggestions.append("Standardize data formats:")
        for issue, description in quality_issues['inconsistent_formats'].items():
            suggestions.append(f"  - {issue}: {description}")
    
    # Outliers
    if quality_issues['outliers']:
        outlier_cols = list(quality_issues['outliers'].keys())
        suggestions.append(f"Investigate outliers in columns: {outlier_cols}")
    
    return suggestions


def create_data_profile(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive data profile.
    
    Args:
        data: DataFrame to profile
        
    Returns:
        Dictionary with data profile information
    """
    
    profile = {
        'basic_info': {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
        },
        'column_analysis': {},
        'data_quality': detect_data_quality_issues(data),
        'summary_stats': {}
    }
    
    # Analyze each column
    for col in data.columns:
        profile['column_analysis'][col] = analyze_column_type(data[col])
    
    # Summary statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile['summary_stats']['numeric'] = data[numeric_cols].describe().to_dict()
    
    # Summary for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    profile['summary_stats']['categorical'] = {}
    for col in categorical_cols:
        value_counts = data[col].value_counts()
        profile['summary_stats']['categorical'][col] = {
            'unique_count': len(value_counts),
            'top_values': value_counts.head(10).to_dict()
        }
    
    return profile


def export_data_profile_report(profile: Dict[str, Any], output_path: str = "data_profile_report.txt"):
    """
    Export data profile to a text report.
    
    Args:
        profile: Data profile from create_data_profile
        output_path: Path for output report
    """
    
    report_lines = []
    
    # Header
    report_lines.append("DATA PROFILE REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Basic info
    basic = profile['basic_info']
    report_lines.append("BASIC INFORMATION")
    report_lines.append("-" * 20)
    report_lines.append(f"Rows: {basic['rows']:,}")
    report_lines.append(f"Columns: {basic['columns']}")
    report_lines.append(f"Memory Usage: {basic['memory_usage_mb']:.2f} MB")
    report_lines.append("")
    
    # Column analysis
    report_lines.append("COLUMN ANALYSIS")
    report_lines.append("-" * 20)
    for col, analysis in profile['column_analysis'].items():
        report_lines.append(f"{col}:")
        report_lines.append(f"  Type: {analysis['inferred_type']}")
        report_lines.append(f"  Nulls: {analysis['null_count']} ({analysis['null_percentage']:.1f}%)")
        report_lines.append(f"  Unique: {analysis['unique_count']}")
        report_lines.append("")
    
    # Data quality issues
    quality = profile['data_quality']
    if any([quality['missing_data'], quality['duplicate_rows'], 
           quality['inconsistent_formats'], quality['outliers']]):
        
        report_lines.append("DATA QUALITY ISSUES")
        report_lines.append("-" * 20)
        
        if quality['missing_data']:
            report_lines.append("Missing Data:")
            for col, info in quality['missing_data'].items():
                report_lines.append(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        if quality['duplicate_rows']:
            report_lines.append(f"Duplicate Rows: {quality['duplicate_rows']}")
        
        if quality['inconsistent_formats']:
            report_lines.append("Format Issues:")
            for issue, desc in quality['inconsistent_formats'].items():
                report_lines.append(f"  {issue}: {desc}")
        
        if quality['outliers']:
            report_lines.append("Outliers Detected:")
            for col, info in quality['outliers'].items():
                report_lines.append(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Data profile report saved to {output_path}")
