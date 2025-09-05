""
Data processing and loading utilities for pipeline datasets.
""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from ..utils.logger import get_logger

logger = get_logger()


class DataProcessor:
    """
    Handles data loading, preprocessing, and basic analysis for pipeline datasets.
    
    Expected columns:
    - pipeline_name, loc_name, connecting_pipeline, connecting_entity
    - rec_del_sign, category_short, country_name, state_abb, county_name
    - latitude, longitude, eff_gas_day, scheduled_quantity
    """
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]):
        """Initialize data processor with dataset path and configuration."""
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.data = None
        self.original_data = None
        self.column_info = {}
        
    async def load_data(self):
        """Load and preprocess the pipeline dataset."""
        try:
            logger.info(f"Loading data from {self.dataset_path}")
            
            # Auto-detect loader based on extension and content
            self.data = self._read_dataset(self.dataset_path)
            self.original_data = self.data.copy()
            
            logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Preprocess the data
            await self._preprocess_data()
            
            # Analyze column characteristics
            self._analyze_columns()
            
            logger.info("Data preprocessing completed")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _read_dataset(self, path: Path) -> pd.DataFrame:
        """
        Read CSV or Parquet (optionally Excel) based on file extension.
        Also protects against accidentally downloading an HTML page.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()

        # Quick guard: detect HTML masquerading as data files
        with open(path, "rb") as f:
            head = f.read(512).lstrip()
        if head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html"):
            raise ValueError(
                "The dataset appears to be an HTML page (likely a Google Drive interstitial). "
                "Please ensure you downloaded the actual data file."
            )

        if suffix in [".csv", ".txt"]:
            # Handles compressed CSVs too by extension (e.g., .csv.gz recognized by pandas)
            return pd.read_csv(path)
        elif suffix in [".parquet", ".pq"]:
            # Requires pyarrow or fastparquet
            # You can specify engine="pyarrow" if desired
            return pd.read_parquet(path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .csv, .parquet, .xlsx")
    
    async def _preprocess_data(self):
        """Perform basic data preprocessing."""
        
        # Convert date columns
        if 'eff_gas_day' in self.data.columns:
            try:
                self.data['eff_gas_day'] = pd.to_datetime(self.data['eff_gas_day'])
                logger.info("Converted eff_gas_day to datetime")
            except Exception as e:
                logger.warning(f"Could not convert eff_gas_day to datetime: {e}")
        
        # Convert numeric columns
        numeric_columns = ['scheduled_quantity', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        # Handle categorical columns
        categorical_columns = [
            'pipeline_name', 'loc_name', 'connecting_pipeline', 'connecting_entity',
            'rec_del_sign', 'category_short', 'country_name', 'state_abb', 'county_name'
        ]
        
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('category')
        
        # Log data quality issues
        missing_counts = self.data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Create derived features
        await self._create_derived_features()
    
    async def _create_derived_features(self):
        """Create useful derived features from the original data."""
        
        # Extract date components if eff_gas_day exists
        if 'eff_gas_day' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['eff_gas_day']):
            self.data['year'] = self.data['eff_gas_day'].dt.year
            self.data['month'] = self.data['eff_gas_day'].dt.month
            self.data['day_of_week'] = self.data['eff_gas_day'].dt.dayofweek
            self.data['quarter'] = self.data['eff_gas_day'].dt.quarter
            
            logger.info("Created date-based derived features")
        
        # Create quantity categories if scheduled_quantity exists
        if 'scheduled_quantity' in self.data.columns:
            try:
                quantity_col = self.data['scheduled_quantity'].dropna()
                if len(quantity_col) > 0:
                    # Create quartile-based categories
                    quartiles = quantity_col.quantile([0.25, 0.5, 0.75])
                    self.data['quantity_category'] = pd.cut(
                        self.data['scheduled_quantity'],
                        bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                        labels=['Low', 'Medium', 'High', 'Very High']
                    )
                    logger.info("Created quantity categories")
            except Exception as e:
                logger.warning(f"Could not create quantity categories: {e}")
        
        # Create location-based features
        if 'latitude' in self.data.columns and 'longitude' in self.data.columns:
            try:
                # Create geographic regions (simplified)
                lat_median = self.data['latitude'].median()
                lon_median = self.data['longitude'].median()
                
                conditions = [
                    (self.data['latitude'] >= lat_median) & (self.data['longitude'] >= lon_median),
                    (self.data['latitude'] >= lat_median) & (self.data['longitude'] < lon_median),
                    (self.data['latitude'] < lat_median) & (self.data['longitude'] >= lon_median),
                    (self.data['latitude'] < lat_median) & (self.data['longitude'] < lon_median)
                ]
                choices = ['NE', 'NW', 'SE', 'SW']
                
                self.data['geographic_quadrant'] = np.select(conditions, choices, default='Unknown')
                logger.info("Created geographic quadrant feature")
                
            except Exception as e:
                logger.warning(f"Could not create geographic features: {e}")
    
    def _analyze_columns(self):
        """Analyze column characteristics and data types."""
        self.column_info = {}
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            info = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data),
                'is_categorical': pd.api.types.is_categorical_dtype(col_data),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(col_data)
            }
            
            # Add specific stats for numeric columns
            if info['is_numeric']:
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    info.update({
                        'min': col_clean.min(),
                        'max': col_clean.max(),
                        'mean': col_clean.mean(),
                        'median': col_clean.median(),
                        'std': col_clean.std()
                    })
            
            # Add specific stats for categorical columns
            if info['is_categorical'] or col_data.dtype == 'object':
                if info['unique_count'] <= 20:  # Only for reasonable number of categories
                    info['value_counts'] = col_data.value_counts().head(10).to_dict()
            
            self.column_info[col] = info
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the dataset."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        summary = {
            'shape': {
                'rows': len(self.data),
                'columns': len(self.data.columns)
            },
            'columns': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isnull().sum().sum(),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        # Add date range if available
        if 'eff_gas_day' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['eff_gas_day']):
            date_col = self.data['eff_gas_day'].dropna()
            if len(date_col) > 0:
                summary['date_range'] = f"{date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}"
        
        return summary
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get detailed information about each column."""
        return self.column_info
    
    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the data and return filtered DataFrame.
        
        Args:
            filters: Dictionary of column -> condition mappings
        
        Returns:
            Filtered DataFrame
        """
        filtered_data = self.data.copy()
        
        for column, condition in filters.items():
            if column not in filtered_data.columns:
                logger.warning(f"Column {column} not found in data")
                continue
            
            try:
                if isinstance(condition, dict):
                    # Handle complex conditions
                    if 'equals' in condition:
                        filtered_data = filtered_data[filtered_data[column] == condition['equals']]
                    elif 'in' in condition:
                        filtered_data = filtered_data[filtered_data[column].isin(condition['in'])]
                    elif 'greater_than' in condition:
                        filtered_data = filtered_data[filtered_data[column] > condition['greater_than']]
                    elif 'less_than' in condition:
                        filtered_data = filtered_data[filtered_data[column] < condition['less_than']]
                    elif 'between' in condition:
                        lower, upper = condition['between']
                        filtered_data = filtered_data[
                            (filtered_data[column] >= lower) & (filtered_data[column] <= upper)
                        ]
                else:
                    # Simple equality condition
                    filtered_data = filtered_data[filtered_data[column] == condition]
                    
            except Exception as e:
                logger.error(f"Error applying filter {column}={condition}: {e}")
        
        return filtered_data
    
    def get_sample_data(self, n: int = 10) -> pd.DataFrame:
        """Get a sample of the data for display."""
        if self.data is None:
            return pd.DataFrame()
        
        return self.data.head(n)
    
    def get_unique_values(self, column: str, limit: int = 50) -> List[str]:
        """Get unique values for a specific column."""
        if self.data is None or column not in self.data.columns:
            return []
        
        unique_vals = self.data[column].dropna().unique()
        
        # Convert to strings and limit
        unique_vals = [str(val) for val in unique_vals[:limit]]
        
        return sorted(unique_vals)
