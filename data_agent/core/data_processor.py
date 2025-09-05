
#Data processing and loading utilities for pipeline datasets.


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
        self.geo_features_available = False
        
        # Missing value handling config
        self.missing_config = config.get('preprocessing', {
            'drop_geo_if_all_missing': True,
            'drop_sched_qty_missing': True,
            'impute_categories_with_unknown': True,
            'add_missing_indicators': True,
            'convert_text_to_categorical': True,
            'high_missing_threshold': 0.6  # Drop columns with >60% missing
        })
        
    async def load_data(self):
        """Load and preprocess the pipeline dataset."""
        try:
            logger.info(f"Loading data from {self.dataset_path}")
            
            # Auto-detect loader based on extension and content
            self.data = self._read_dataset(self.dataset_path)
            self.original_data = self.data.copy()
            
            logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Handle missing values first
            self._handle_missing_values()
            
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

    def _handle_missing_values(self):
        """
        Comprehensive missing value handling strategy.
        """
        logger.info("Handling missing values...")
        df = self.data
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        # 1) Analyze missing patterns
        missing_summary = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_rate = missing_count / len(df)
            missing_summary[col] = {
                'count': missing_count,
                'rate': missing_rate
            }
        
        # Log significant missing values
        significant_missing = {k: v['count'] for k, v in missing_summary.items() if v['count'] > 0}
        if significant_missing:
            logger.warning(f"Missing values found: {significant_missing}")
        
        # 2) Drop columns with extremely high missing rates
        cols_to_drop = []
        for col, info in missing_summary.items():
            if info['rate'] >= self.missing_config['high_missing_threshold']:
                cols_to_drop.append(col)
                logger.info(f"Dropping column '{col}' due to high missing rate: {info['rate']:.1%}")
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        # 3) Handle geographic columns specifically
        geo_cols = ["latitude", "longitude"]
        geo_missing_rates = {}
        for col in geo_cols:
            if col in df.columns:
                geo_missing_rates[col] = df[col].isna().mean()
        
        if self.missing_config['drop_geo_if_all_missing']:
            geo_unusable = all(rate >= 0.999 for rate in geo_missing_rates.values())
            if geo_unusable and geo_cols[0] in df.columns:
                logger.info("Dropping geographic columns due to complete missingness")
                df.drop(columns=[col for col in geo_cols if col in df.columns], inplace=True)
                self.geo_features_available = False
            else:
                self.geo_features_available = all(col in df.columns for col in geo_cols)
        else:
            self.geo_features_available = all(col in df.columns for col in geo_cols)
        
        # 4) Handle connecting_pipeline (typically very high missing)
        if "connecting_pipeline" in df.columns:
            miss_rate = df["connecting_pipeline"].isna().mean()
            if miss_rate >= self.missing_config['high_missing_threshold']:
                logger.info(f"Dropping 'connecting_pipeline' due to high missing rate: {miss_rate:.1%}")
                df.drop(columns=["connecting_pipeline"], inplace=True)
            else:
                if self.missing_config['add_missing_indicators']:
                    df["connecting_pipeline_missing"] = df["connecting_pipeline"].isna()
                if self.missing_config['impute_categories_with_unknown']:
                    df["connecting_pipeline"] = df["connecting_pipeline"].fillna("None")
        
        # 5) Handle other categorical columns with moderate missing
        categorical_impute_cols = [
            "connecting_entity", "category_short", "state_abb", "county_name"
        ]
        
        for col in categorical_impute_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if self.missing_config['add_missing_indicators']:
                        df[f"{col}_missing"] = df[col].isna()
                    
                    if self.missing_config['impute_categories_with_unknown']:
                        # Use appropriate fill value based on column
                        fill_value = "Unknown"
                        if col == "state_abb":
                            fill_value = "UNK"
                        elif col == "connecting_entity":
                            fill_value = "Unknown"
                        
                        df[col] = df[col].fillna(fill_value)
                        logger.info(f"Imputed {missing_count} missing values in '{col}' with '{fill_value}'")
        
        # 6) Handle scheduled_quantity (critical numeric column)
        if "scheduled_quantity" in df.columns:
            missing_count = df["scheduled_quantity"].isna().sum()
            if missing_count > 0:
                if self.missing_config['drop_sched_qty_missing']:
                    # Drop rows with missing scheduled_quantity
                    before_rows = len(df)
                    df.dropna(subset=["scheduled_quantity"], inplace=True)
                    after_rows = len(df)
                    dropped_rows = before_rows - after_rows
                    logger.info(f"Dropped {dropped_rows} rows with missing scheduled_quantity")
                else:
                    # Advanced imputation: groupwise median
                    if self.missing_config['add_missing_indicators']:
                        df["scheduled_quantity_missing"] = df["scheduled_quantity"].isna()
                    
                    # Try groupwise imputation by pipeline and weekday
                    if "eff_gas_day" in df.columns and "pipeline_name" in df.columns:
                        try:
                            df["eff_gas_day"] = pd.to_datetime(df["eff_gas_day"], errors='coerce')
                            weekday = df["eff_gas_day"].dt.weekday
                            grp_median = df.groupby(["pipeline_name", weekday])["scheduled_quantity"].transform("median")
                            global_median = df["scheduled_quantity"].median()
                            
                            df["scheduled_quantity"] = df["scheduled_quantity"].fillna(grp_median).fillna(global_median)
                            logger.info(f"Imputed {missing_count} missing scheduled_quantity values using groupwise median")
                        except Exception as e:
                            # Fallback to simple median
                            global_median = df["scheduled_quantity"].median()
                            df["scheduled_quantity"] = df["scheduled_quantity"].fillna(global_median)
                            logger.warning(f"Fallback imputation for scheduled_quantity: {e}")
                    else:
                        # Simple median imputation
                        global_median = df["scheduled_quantity"].median()
                        df["scheduled_quantity"] = df["scheduled_quantity"].fillna(global_median)
                        logger.info(f"Imputed {missing_count} missing scheduled_quantity values with median")
        
        # 7) Convert text columns to categorical to save memory
        if self.missing_config['convert_text_to_categorical']:
            cat_candidates = [
                "pipeline_name", "loc_name", "connecting_entity",
                "category_short", "state_abb", "county_name", "rec_del_sign", "country_name"
            ]
            
            for col in cat_candidates:
                if col in df.columns and df[col].dtype == object:
                    unique_count = df[col].nunique()
                    # Only convert if reasonable cardinality (avoid memory issues with too many categories)
                    if unique_count < len(df) * 0.5:  # Less than 50% unique values
                        df[col] = df[col].astype("category")
                        logger.debug(f"Converted '{col}' to categorical ({unique_count} unique values)")
        
        # Update data
        self.data = df
        
        # Log summary of changes
        final_rows = len(df)
        final_cols = len(df.columns)
        logger.info(f"Missing value handling complete: {initial_rows} -> {final_rows} rows, {initial_cols} -> {final_cols} columns")
        
        # Final missing value check
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            remaining_by_col = df.isnull().sum()
            remaining_by_col = remaining_by_col[remaining_by_col > 0]
            logger.info(f"Remaining missing values: {remaining_missing} total, by column: {remaining_by_col.to_dict()}")
        else:
            logger.info("No missing values remaining after preprocessing")
    
    async def _preprocess_data(self):
        """Perform basic data preprocessing."""
        
        # Convert date columns
        if 'eff_gas_day' in self.data.columns:
            try:
                self.data['eff_gas_day'] = pd.to_datetime(self.data['eff_gas_day'])
                logger.info("Converted eff_gas_day to datetime")
            except Exception as e:
                logger.warning(f"Could not convert eff_gas_day to datetime: {e}")
        
        # Convert numeric columns (only if they still exist after missing value handling)
        numeric_columns = ['scheduled_quantity', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
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
        
        # Create quantity categories if scheduled_quantity exists and has valid data
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
        
        # Create location-based features only if geo data is available
        if self.geo_features_available and 'latitude' in self.data.columns and 'longitude' in self.data.columns:
            try:
                # Only create features if we have sufficient non-null geo data
                geo_data = self.data[['latitude', 'longitude']].dropna()
                if len(geo_data) > 0:
                    lat_median = geo_data['latitude'].median()
                    lon_median = geo_data['longitude'].median()
                    
                    conditions = [
                        (self.data['latitude'] >= lat_median) & (self.data['longitude'] >= lon_median),
                        (self.data['latitude'] >= lat_median) & (self.data['longitude'] < lon_median),
                        (self.data['latitude'] < lat_median) & (self.data['longitude'] >= lon_median),
                        (self.data['latitude'] < lat_median) & (self.data['longitude'] < lon_median)
                    ]
                    choices = ['NE', 'NW', 'SE', 'SW']
                    
                    self.data['geographic_quadrant'] = np.select(conditions, choices, default='Unknown')
                    logger.info("Created geographic quadrant feature")
                else:
                    logger.info("Skipped geographic features due to insufficient valid coordinate data")
                    
            except Exception as e:
                logger.warning(f"Could not create geographic features: {e}")
        else:
            logger.info("Skipped geographic features (coordinates not available)")
    
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
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'geo_features_available': self.geo_features_available
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

    def get_missing_value_report(self) -> Dict[str, Any]:
        """Generate a comprehensive missing value report."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        missing_info = {}
        total_rows = len(self.data)
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_rate = missing_count / total_rows
            
            missing_info[col] = {
                'missing_count': int(missing_count),
                'missing_rate': float(missing_rate),
                'non_missing_count': int(total_rows - missing_count),
                'data_type': str(self.data[col].dtype)
            }
        
        # Summary statistics
        total_missing = sum(info['missing_count'] for info in missing_info.values())
        total_cells = total_rows * len(self.data.columns)
        
        report = {
            'column_details': missing_info,
            'summary': {
                'total_rows': total_rows,
                'total_columns': len(self.data.columns),
                'total_cells': total_cells,
                'total_missing_values': total_missing,
                'overall_missing_rate': total_missing / total_cells if total_cells > 0 else 0,
                'columns_with_missing': sum(1 for info in missing_info.values() if info['missing_count'] > 0),
                'geo_features_available': self.geo_features_available
            }
        }
        
        return report
