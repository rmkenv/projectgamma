
# Advanced analysis engine for statistical analysis, anomaly detection, and pattern recognition.

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_logger

logger = get_logger()


class AnalysisEngine:
    """
    Performs various types of data analysis including statistical analysis,
    anomaly detection, pattern recognition, and clustering.
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Initialize analysis engine with data and configuration."""
        self.data = data
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def execute_simple_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simple data queries like filtering, counting, aggregation."""
        
        try:
            results = {}
            
            # Apply filters if specified
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Get specified columns or all columns
            columns = params.get('columns', list(filtered_data.columns))
            if columns:
                filtered_data = filtered_data[columns]
                
            # Apply aggregation
            aggregation = params.get('aggregation')
            if aggregation:
                results['aggregation_results'] = self._apply_aggregation(filtered_data, aggregation, columns)
            
            # Basic info
            results['row_count'] = len(filtered_data)
            results['column_count'] = len(filtered_data.columns)
            
            # Sample data
            max_display = self.config.get('max_display_rows', 20)
            results['sample_data'] = filtered_data.head(max_display).to_dict('records')
            
            # Summary statistics for numeric columns
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results['numeric_summary'] = filtered_data[numeric_cols].describe().to_dict()
            
            # Value counts for categorical columns
            categorical_cols = filtered_data.select_dtypes(include=['object', 'category']).columns
            results['categorical_summary'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = filtered_data[col].value_counts().head(10)
                results['categorical_summary'][col] = value_counts.to_dict()
            
            results['method'] = 'simple_query'
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Simple query execution failed: {e}")
            return {'error': str(e), 'success': False}
    
    def detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using multiple methods."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Get numeric columns for analysis
            columns = params.get('columns', [])
            if not columns:
                numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
                columns = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
            
            if not columns:
                return {'error': 'No numeric columns found for anomaly detection', 'success': False}
            
            # Prepare data for analysis
            analysis_data = filtered_data[columns].dropna()
            
            if len(analysis_data) < 10:
                return {'error': 'Insufficient data for anomaly detection', 'success': False}
            
            # Method 1: Statistical outliers (Z-score)
            statistical_results = self._detect_statistical_anomalies(analysis_data, columns)
            results['statistical_method'] = statistical_results
            
            # Method 2: Isolation Forest
            isolation_results = self._detect_isolation_forest_anomalies(analysis_data, columns)
            results['isolation_forest'] = isolation_results
            
            # Method 3: IQR method
            iqr_results = self._detect_iqr_anomalies(analysis_data, columns)
            results['iqr_method'] = iqr_results
            
            # Summary of all methods
            results['summary'] = self._summarize_anomaly_results(
                statistical_results, isolation_results, iqr_results, filtered_data
            )
            
            results['method'] = 'anomaly_detection'
            results['success'] = True
            results['columns_analyzed'] = columns
            results['total_rows'] = len(filtered_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'error': str(e), 'success': False}
    
    def analyze_patterns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns and trends in the data."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Time series patterns
            if 'eff_gas_day' in filtered_data.columns:
                time_patterns = self._analyze_time_patterns(filtered_data)
                results['time_patterns'] = time_patterns
            
            # Geographic patterns
            if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
                geo_patterns = self._analyze_geographic_patterns(filtered_data)
                results['geographic_patterns'] = geo_patterns
            
            # Categorical patterns
            categorical_cols = filtered_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                categorical_patterns = self._analyze_categorical_patterns(filtered_data, categorical_cols)
                results['categorical_patterns'] = categorical_patterns
            
            # Numeric patterns
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                numeric_patterns = self._analyze_numeric_patterns(filtered_data, numeric_cols)
                results['numeric_patterns'] = numeric_patterns
            
            results['method'] = 'pattern_analysis'
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    def analyze_correlations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Get numeric columns
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'error': 'Need at least 2 numeric columns for correlation analysis', 'success': False}
            
            # Remove coordinate columns from general correlation if not specifically requested
            if 'columns' not in params:
                numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
            
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient numeric columns for meaningful correlation analysis', 'success': False}
            
            # Calculate correlations
            correlation_data = filtered_data[numeric_cols].dropna()
            
            # Pearson correlation
            pearson_corr = correlation_data.corr(method='pearson')
            results['pearson_correlation'] = pearson_corr.to_dict()
            
            # Spearman correlation (rank-based, good for non-linear relationships)
            spearman_corr = correlation_data.corr(method='spearman')
            results['spearman_correlation'] = spearman_corr.to_dict()
            
            # Find strongest correlations
            strong_correlations = self._find_strong_correlations(pearson_corr, spearman_corr)
            results['strong_correlations'] = strong_correlations
            
            # Correlation with categorical variables
            categorical_cols = filtered_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_numeric_corr = self._analyze_categorical_numeric_correlations(
                    filtered_data, categorical_cols, numeric_cols
                )
                results['categorical_numeric_correlations'] = cat_numeric_corr
            
            results['method'] = 'correlation_analysis'
            results['success'] = True
            results['columns_analyzed'] = numeric_cols
            
            return results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    def statistical_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Basic info
            results['basic_info'] = {
                'total_rows': len(filtered_data),
                'total_columns': len(filtered_data.columns),
                'missing_values': filtered_data.isnull().sum().sum(),
                'memory_usage_mb': filtered_data.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Numeric columns summary
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_summary = filtered_data[numeric_cols].describe()
                results['numeric_summary'] = numeric_summary.to_dict()
                
                # Additional statistics
                for col in numeric_cols:
                    col_data = filtered_data[col].dropna()
                    if len(col_data) > 0:
                        results['numeric_summary'][col]['skewness'] = stats.skew(col_data)
                        results['numeric_summary'][col]['kurtosis'] = stats.kurtosis(col_data)
            
            # Categorical columns summary
            categorical_cols = filtered_data.select_dtypes(include=['object', 'category']).columns
            results['categorical_summary'] = {}
            for col in categorical_cols:
                value_counts = filtered_data[col].value_counts()
                results['categorical_summary'][col] = {
                    'unique_count': value_counts.shape[0],
                    'top_values': value_counts.head(10).to_dict(),
                    'missing_count': filtered_data[col].isnull().sum()
                }
            
            # Data quality assessment
            results['data_quality'] = self._assess_data_quality(filtered_data)
            
            results['method'] = 'statistical_summary'
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical summary failed: {e}")
            return {'error': str(e), 'success': False}
    
    def perform_clustering(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Prepare data for clustering
            clustering_data = self._prepare_clustering_data(filtered_data, params.get('columns', []))
            
            if clustering_data.shape[0] < 10:
                return {'error': 'Insufficient data for clustering', 'success': False}
            
            if clustering_data.shape[1] < 2:
                return {'error': 'Need at least 2 features for clustering', 'success': False}
            
            # DBSCAN clustering
            dbscan_results = self._perform_dbscan_clustering(clustering_data)
            results['dbscan'] = dbscan_results
            
            # K-means clustering
            kmeans_results = self._perform_kmeans_clustering(clustering_data)
            results['kmeans'] = kmeans_results
            
            # Clustering analysis
            cluster_analysis = self._analyze_clusters(filtered_data, dbscan_results, kmeans_results)
            results['cluster_analysis'] = cluster_analysis
            
            results['method'] = 'clustering'
            results['success'] = True
            results['total_rows'] = len(filtered_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {'error': str(e), 'success': False}
    
    def analyze_causal_relationships(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential causal relationships (with appropriate caveats)."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # This is exploratory analysis - not true causal inference
            results['disclaimer'] = ("This analysis identifies statistical associations and patterns. "
                                   "Causal relationships require controlled experiments or specialized methods.")
            
            # Analyze relationships between key variables
            causal_analysis = {}
            
            # Quantity relationships
            if 'scheduled_quantity' in filtered_data.columns:
                quantity_analysis = self._analyze_quantity_relationships(filtered_data)
                causal_analysis['quantity_factors'] = quantity_analysis
            
            # Geographic relationships
            if 'state_abb' in filtered_data.columns:
                geographic_analysis = self._analyze_geographic_relationships(filtered_data)
                causal_analysis['geographic_factors'] = geographic_analysis
            
            # Temporal relationships
            if 'eff_gas_day' in filtered_data.columns:
                temporal_analysis = self._analyze_temporal_relationships(filtered_data)
                causal_analysis['temporal_factors'] = temporal_analysis
            
            # Category relationships
            if 'category_short' in filtered_data.columns:
                category_analysis = self._analyze_category_relationships(filtered_data)
                causal_analysis['category_factors'] = category_analysis
            
            results['causal_analysis'] = causal_analysis
            results['method'] = 'causal_analysis'
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    def general_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general exploratory data analysis."""
        
        try:
            results = {}
            
            # Apply filters
            filtered_data = self._apply_filters(params.get('filters', {}))
            
            # Basic overview
            results['overview'] = self.statistical_summary({'filters': params.get('filters', {})})
            
            # Look for interesting patterns
            if len(filtered_data) > 100:  # Only for reasonably sized datasets
                
                # Quick anomaly check
                anomaly_check = self.detect_anomalies({'filters': params.get('filters', {})})
                if anomaly_check.get('success'):
                    results['anomaly_overview'] = anomaly_check.get('summary', {})
                
                # Quick correlation check
                correlation_check = self.analyze_correlations({'filters': params.get('filters', {})})
                if correlation_check.get('success'):
                    results['correlation_overview'] = correlation_check.get('strong_correlations', [])
                
                # Quick pattern check
                pattern_check = self.analyze_patterns({'filters': params.get('filters', {})})
                if pattern_check.get('success'):
                    results['pattern_overview'] = pattern_check
            
            # Data insights
            insights = self._generate_data_insights(filtered_data)
            results['insights'] = insights
            
            results['method'] = 'general_analysis'
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"General analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    # Helper methods
    
    def _apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the data."""
        filtered_data = self.data.copy()
        
        for column, condition in filters.items():
            if column not in filtered_data.columns:
                continue
                
            try:
                if isinstance(condition, dict):
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
                    filtered_data = filtered_data[filtered_data[column] == condition]
                    
            except Exception as e:
                logger.warning(f"Error applying filter {column}={condition}: {e}")
        
        return filtered_data
    
    def _apply_aggregation(self, data: pd.DataFrame, aggregation: str, columns: List[str]) -> Dict[str, Any]:
        """Apply aggregation to the data."""
        results = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if aggregation == 'count':
            results['total_count'] = len(data)
            for col in columns:
                if col in data.columns:
                    results[f'{col}_count'] = data[col].count()
        
        elif aggregation in ['sum', 'mean', 'median', 'max', 'min'] and len(numeric_cols) > 0:
            for col in numeric_cols:
                if col in columns or not columns:
                    if aggregation == 'sum':
                        results[f'{col}_sum'] = data[col].sum()
                    elif aggregation == 'mean':
                        results[f'{col}_mean'] = data[col].mean()
                    elif aggregation == 'median':
                        results[f'{col}_median'] = data[col].median()
                    elif aggregation == 'max':
                        results[f'{col}_max'] = data[col].max()
                    elif aggregation == 'min':
                        results[f'{col}_min'] = data[col].min()
        
        return results
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect anomalies using statistical methods (Z-score)."""
        
        results = {'method': 'z_score', 'threshold': 3.0, 'anomalies': {}}
        
        for col in columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                anomaly_indices = col_data[z_scores > 3.0].index.tolist()
                
                results['anomalies'][col] = {
                    'count': len(anomaly_indices),
                    'percentage': (len(anomaly_indices) / len(col_data)) * 100,
                    'indices': anomaly_indices[:10]  # First 10 anomalies
                }
        
        return results
    
    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        
        try:
            contamination = self.config.get('anomaly_contamination', 0.1)
            
            # Standardize the data
            scaled_data = self.scaler.fit_transform(data[columns])
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(scaled_data)
            scores = iso_forest.score_samples(scaled_data)
            
            # Get anomaly indices
            anomaly_indices = data.index[predictions == -1].tolist()
            
            results = {
                'method': 'isolation_forest',
                'contamination': contamination,
                'total_anomalies': len(anomaly_indices),
                'percentage': (len(anomaly_indices) / len(data)) * 100,
                'anomaly_indices': anomaly_indices[:20],  # First 20 anomalies
                'anomaly_scores': {str(idx): score for idx, score in zip(anomaly_indices[:10], scores[predictions == -1][:10])}
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Isolation Forest failed: {e}")
            return {'error': str(e)}
    
    def _detect_iqr_anomalies(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect anomalies using IQR method."""
        
        results = {'method': 'iqr', 'anomalies': {}}
        
        for col in columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                results['anomalies'][col] = {
                    'count': len(anomalies),
                    'percentage': (len(anomalies) / len(col_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'indices': anomalies.index.tolist()[:10]
                }
        
        return results
    
    def _summarize_anomaly_results(self, statistical: Dict, isolation: Dict, iqr: Dict, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize results from multiple anomaly detection methods."""
        
        summary = {
            'methods_used': ['statistical_zscore', 'isolation_forest', 'iqr'],
            'total_data_points': len(original_data)
        }
        
        # Count total anomalies by each method
        if 'anomalies' in statistical:
            stat_total = sum(result['count'] for result in statistical['anomalies'].values())
            summary['statistical_anomalies'] = stat_total
        
        if 'total_anomalies' in isolation:
            summary['isolation_forest_anomalies'] = isolation['total_anomalies']
        
        if 'anomalies' in iqr:
            iqr_total = sum(result['count'] for result in iqr['anomalies'].values())
            summary['iqr_anomalies'] = iqr_total
        
        return summary
    
    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        
        patterns = {}
        
        if 'eff_gas_day' in data.columns:
            date_col = data['eff_gas_day'].dropna()
            
            if len(date_col) > 0:
                patterns['date_range'] = {
                    'start': date_col.min().strftime('%Y-%m-%d'),
                    'end': date_col.max().strftime('%Y-%m-%d'),
                    'span_days': (date_col.max() - date_col.min()).days
                }
                
                # Monthly patterns
                if 'month' in data.columns:
                    monthly_counts = data['month'].value_counts().sort_index()
                    patterns['monthly_distribution'] = monthly_counts.to_dict()
                
                # Day of week patterns
                if 'day_of_week' in data.columns:
                    dow_counts = data['day_of_week'].value_counts().sort_index()
                    patterns['day_of_week_distribution'] = dow_counts.to_dict()
        
        return patterns
    
    def _analyze_geographic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic patterns in the data."""
        
        patterns = {}
        
        if 'state_abb' in data.columns:
            state_counts = data['state_abb'].value_counts()
            patterns['state_distribution'] = state_counts.head(10).to_dict()
        
        if 'country_name' in data.columns:
            country_counts = data['country_name'].value_counts()
            patterns['country_distribution'] = country_counts.head(10).to_dict()
        
        if 'geographic_quadrant' in data.columns:
            quad_counts = data['geographic_quadrant'].value_counts()
            patterns['quadrant_distribution'] = quad_counts.to_dict()
        
        return patterns
    
    def _analyze_categorical_patterns(self, data: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze patterns in categorical columns."""
        
        patterns = {}
        
        for col in categorical_cols[:5]:  # Limit to first 5 columns
            value_counts = data[col].value_counts()
            patterns[col] = {
                'unique_count': len(value_counts),
                'top_values': value_counts.head(10).to_dict(),
                'diversity_index': len(value_counts) / len(data) if len(data) > 0 else 0
            }
        
        return patterns
    
    def _analyze_numeric_patterns(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze patterns in numeric columns."""
        
        patterns = {}
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            col_data = data[col].dropna()
            if len(col_data) > 0:
                patterns[col] = {
                    'distribution_stats': {
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std(),
                        'skewness': stats.skew(col_data),
                        'kurtosis': stats.kurtosis(col_data)
                    },
                    'outlier_percentage': len(col_data[np.abs(stats.zscore(col_data)) > 2]) / len(col_data) * 100
                }
        
        return patterns
    
    def _find_strong_correlations(self, pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find strong correlations between variables."""
        
        strong_correlations = []
        threshold = self.config.get('correlation_threshold', 0.5)
        
        # Get upper triangle to avoid duplicates
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
        
        for i, col1 in enumerate(pearson_corr.columns):
            for j, col2 in enumerate(pearson_corr.columns):
                if mask[i, j]:
                    pearson_val = pearson_corr.loc[col1, col2]
                    spearman_val = spearman_corr.loc[col1, col2]
                    
                    if abs(pearson_val) >= threshold or abs(spearman_val) >= threshold:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'pearson_correlation': round(pearson_val, 3),
                            'spearman_correlation': round(spearman_val, 3),
                            'strength': 'strong' if max(abs(pearson_val), abs(spearman_val)) >= 0.7 else 'moderate'
                        })
        
        # Sort by strongest correlation
        strong_correlations.sort(key=lambda x: max(abs(x['pearson_correlation']), abs(x['spearman_correlation'])), reverse=True)
        
        return strong_correlations[:10]  # Return top 10
    
    def _analyze_categorical_numeric_correlations(self, data: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze correlations between categorical and numeric variables."""
        
        correlations = {}
        
        for cat_col in categorical_cols[:3]:  # Limit to 3 categorical columns
            correlations[cat_col] = {}
            
            for num_col in numeric_cols[:3]:  # Limit to 3 numeric columns
                try:
                    # Use ANOVA F-statistic as a measure of association
                    categories = data[cat_col].dropna().unique()
                    if len(categories) > 1:
                        groups = [data[data[cat_col] == cat][num_col].dropna() for cat in categories]
                        groups = [group for group in groups if len(group) > 0]
                        
                        if len(groups) > 1:
                            f_stat, p_value = stats.f_oneway(*groups)
                            correlations[cat_col][num_col] = {
                                'f_statistic': round(f_stat, 3),
                                'p_value': round(p_value, 3),
                                'significant': p_value < 0.05
                            }
                except:
                    continue
        
        return correlations
    
    def _prepare_clustering_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Prepare data for clustering analysis."""
        
        if not columns:
            # Auto-select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove coordinate columns unless specifically requested
            columns = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
        
        # Get the data and handle missing values
        clustering_data = data[columns].dropna()
        
        # Scale the data
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(clustering_data),
            columns=clustering_data.columns,
            index=clustering_data.index
        )
        
        return scaled_data
    
    def _perform_dbscan_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform DBSCAN clustering."""
        
        try:
            min_samples = max(5, int(len(data) * 0.01))  # 1% of data or 5, whichever is larger
            eps = 0.5
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            results = {
                'method': 'DBSCAN',
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_labels': clusters.tolist(),
                'parameters': {'eps': eps, 'min_samples': min_samples}
            }
            
            if n_clusters > 1:
                # Calculate silhouette score (excluding noise points)
                valid_data = data[clusters != -1]
                valid_labels = clusters[clusters != -1]
                if len(valid_data) > 1 and len(set(valid_labels)) > 1:
                    results['silhouette_score'] = silhouette_score(valid_data, valid_labels)
            
            return results
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return {'error': str(e)}
    
    def _perform_kmeans_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform K-means clustering."""
        
        try:
            # Determine optimal number of clusters using elbow method
            max_k = min(10, len(data) // 2)
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data)
                inertias.append(kmeans.inertia_)
                
                if len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(data, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
            
            # Choose optimal k (highest silhouette score)
            optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
            
            # Fit final model
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_clusters = final_kmeans.fit_predict(data)
            
            results = {
                'method': 'K-means',
                'optimal_k': optimal_k,
                'silhouette_score': max(silhouette_scores),
                'cluster_labels': final_clusters.tolist(),
                'cluster_centers': final_kmeans.cluster_centers_.tolist(),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores
            }
            
            return results
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return {'error': str(e)}
    
    def _analyze_clusters(self, original_data: pd.DataFrame, dbscan_results: Dict, kmeans_results: Dict) -> Dict[str, Any]:
        """Analyze the characteristics of discovered clusters."""
        
        analysis = {}
        
        # Analyze K-means clusters
        if 'cluster_labels' in kmeans_results and 'error' not in kmeans_results:
            cluster_labels = kmeans_results['cluster_labels']
            analysis['kmeans_analysis'] = {}
            
            for cluster_id in set(cluster_labels):
                cluster_data = original_data.iloc[[i for i, label in enumerate(cluster_labels) if label == cluster_id]]
                
                analysis['kmeans_analysis'][f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(original_data) * 100
                }
                
                # Analyze categorical distributions in clusters
                categorical_cols = cluster_data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols[:3]:
                    value_counts = cluster_data[col].value_counts()
                    analysis['kmeans_analysis'][f'cluster_{cluster_id}'][f'{col}_distribution'] = value_counts.head(5).to_dict()
        
        return analysis
    
    def _analyze_quantity_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factors that might influence scheduled quantities."""
        
        analysis = {}
        
        if 'scheduled_quantity' in data.columns:
            qty_data = data['scheduled_quantity'].dropna()
            
            # State-based analysis
            if 'state_abb' in data.columns:
                state_qty = data.groupby('state_abb')['scheduled_quantity'].agg(['mean', 'median', 'std', 'count'])
                analysis['state_influence'] = state_qty.sort_values('mean', ascending=False).head(10).to_dict()
            
            # Category-based analysis
            if 'category_short' in data.columns:
                cat_qty = data.groupby('category_short')['scheduled_quantity'].agg(['mean', 'median', 'std', 'count'])
                analysis['category_influence'] = cat_qty.sort_values('mean', ascending=False).to_dict()
            
            # Time-based analysis
            if 'month' in data.columns:
                month_qty = data.groupby('month')['scheduled_quantity'].agg(['mean', 'median', 'count'])
                analysis['seasonal_influence'] = month_qty.to_dict()
        
        return analysis
    
    def _analyze_geographic_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic factors and their relationships."""
        
        analysis = {}
        
        if 'state_abb' in data.columns:
            # State distribution
            state_counts = data['state_abb'].value_counts()
            analysis['state_concentration'] = {
                'most_active_states': state_counts.head(10).to_dict(),
                'state_diversity': len(state_counts) / len(data) if len(data) > 0 else 0
            }
            
            # State-category relationships
            if 'category_short' in data.columns:
                state_category = pd.crosstab(data['state_abb'], data['category_short'])
                analysis['state_category_patterns'] = state_category.to_dict()
        
        return analysis
    
    def _analyze_temporal_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal factors and relationships."""
        
        analysis = {}
        
        if 'eff_gas_day' in data.columns:
            # Yearly trends
            if 'year' in data.columns:
                yearly_counts = data['year'].value_counts().sort_index()
                analysis['yearly_trends'] = yearly_counts.to_dict()
            
            # Monthly seasonality
            if 'month' in data.columns:
                monthly_counts = data['month'].value_counts().sort_index()
                analysis['seasonal_patterns'] = monthly_counts.to_dict()
            
            # Day of week patterns
            if 'day_of_week' in data.columns:
                dow_counts = data['day_of_week'].value_counts().sort_index()
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                analysis['weekly_patterns'] = {weekday_names[i]: dow_counts.get(i, 0) for i in range(7)}
        
        return analysis
    
    def _analyze_category_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze category-based relationships."""
        
        analysis = {}
        
        if 'category_short' in data.columns:
            category_counts = data['category_short'].value_counts()
            analysis['category_distribution'] = category_counts.to_dict()
            
            # Category-delivery sign relationships
            if 'rec_del_sign' in data.columns:
                cat_delivery = pd.crosstab(data['category_short'], data['rec_del_sign'])
                analysis['category_delivery_patterns'] = cat_delivery.to_dict()
        
        return analysis
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        
        quality = {}
        
        # Missing value analysis
        missing_summary = data.isnull().sum()
        quality['missing_values'] = {
            'total_missing': missing_summary.sum(),
            'columns_with_missing': missing_summary[missing_summary > 0].to_dict(),
            'completeness_score': 1 - (missing_summary.sum() / (len(data) * len(data.columns)))
        }
        
        # Duplicate analysis
        duplicates = data.duplicated().sum()
        quality['duplicates'] = {
            'duplicate_rows': duplicates,
            'duplicate_percentage': duplicates / len(data) * 100 if len(data) > 0 else 0
        }
        
        # Data type consistency
        quality['data_types'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        return quality
    
    def _generate_data_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate human-readable insights about the data."""
        
        insights = []
        
        # Size insights
        insights.append(f"Dataset contains {len(data):,} records with {len(data.columns)} columns")
        
        # Missing data insights
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 10:
            insights.append(f"Data has {missing_pct:.1f}% missing values - consider data cleaning")
        elif missing_pct < 1:
            insights.append("Data is very complete with minimal missing values")
        
        # Geographic insights
        if 'state_abb' in data.columns:
            n_states = data['state_abb'].nunique()
            top_state = data['state_abb'].value_counts().index[0]
            insights.append(f"Data covers {n_states} states, with {top_state} being most represented")
        
        # Temporal insights
        if 'eff_gas_day' in data.columns:
            date_range = data['eff_gas_day'].max() - data['eff_gas_day'].min() 
            insights.append(f"Data spans {date_range.days} days")
        
        # Quantity insights
        if 'scheduled_quantity' in data.columns:
            qty_data = data['scheduled_quantity'].dropna()
            if len(qty_data) > 0:
                qty_std = qty_data.std()
                qty_mean = qty_data.mean()
                if qty_std / qty_mean > 1:  # High coefficient of variation
                    insights.append("Scheduled quantities show high variability")
        
        return insights
