
#Advanced pipeline analytics with real-time vessel tracking integration.

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

# Optional imports (install if needed)
try:
    from sklearn.neighbors import KDTree
    from shapely.geometry import Point
    import geopandas as gpd
    import websockets
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger()


class PipelineAnalytics:
    """Advanced pipeline analytics with vessel tracking integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streamed_data = []
        self.spatial_available = SPATIAL_AVAILABLE
        
        # Analytics config
        self.analytics_config = config.get('pipeline_analytics', {
            'utilization_threshold': 0.85,
            'bottleneck_window_days': 30,
            'bottleneck_min_days': 10,
            'max_tanker_distance_km': 20,
            'ais_stream_duration': 30
        })
        
        if not self.spatial_available:
            logger.warning("Spatial analysis libraries not available. Install: pip install geopandas shapely scikit-learn websockets")
    
    def infer_capacity(self, df: pd.DataFrame, pipeline_col='pipeline_name', quantity_col='scheduled_quantity') -> pd.Series:
        """Infer pipeline capacity using 95th percentile of scheduled quantities."""
        try:
            capacities = df.groupby(pipeline_col)[quantity_col].quantile(0.95)
            logger.info(f"Inferred capacities for {len(capacities)} pipelines")
            return capacities
        except Exception as e:
            logger.error(f"Failed to infer capacities: {e}")
            return pd.Series()
    
    def calculate_utilization(self, df: pd.DataFrame, capacity_series: pd.Series, 
                            pipeline_col='pipeline_name', quantity_col='scheduled_quantity') -> pd.DataFrame:
        """Calculate pipeline utilization rates."""
        try:
            df = df.copy()
            df['capacity'] = df[pipeline_col].map(capacity_series)
            df['utilization'] = df[quantity_col] / df['capacity']
            
            # Handle infinite/NaN utilization
            df['utilization'] = df['utilization'].replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Calculated utilization for {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to calculate utilization: {e}")
            return df
    
    def detect_bottlenecks(self, df: pd.DataFrame, utilization_col='utilization', 
                          pipeline_col='pipeline_name') -> pd.DataFrame:
        """Detect pipeline bottlenecks based on sustained high utilization."""
        try:
            threshold = self.analytics_config['utilization_threshold']
            window_days = self.analytics_config['bottleneck_window_days']
            min_days = self.analytics_config['bottleneck_min_days']
            
            df = df.copy()
            df['high_util'] = df[utilization_col] > threshold
            df['eff_gas_day'] = pd.to_datetime(df['eff_gas_day'])
            
            def rolling_high_util_count(sub):
                sub = sub.sort_values('eff_gas_day')
                return sub['high_util'].rolling(window=window_days, min_periods=1).sum()
            
            df['rolling_high_count'] = df.groupby(pipeline_col).apply(
                rolling_high_util_count
            ).reset_index(level=0, drop=True)
            
            bottleneck_df = df[df['rolling_high_count'] >= min_days]
            
            if len(bottleneck_df) == 0:
                logger.info("No bottlenecks detected")
                return pd.DataFrame()
            
            summary = bottleneck_df.groupby([pipeline_col, 'loc_name']).agg(
                peak_utilization=(utilization_col, 'max'),
                high_util_days=('high_util', 'sum'),
                first_bottleneck_date=('eff_gas_day', 'min'),
                last_bottleneck_date=('eff_gas_day', 'max'),
            ).reset_index()
            
            logger.info(f"Detected {len(summary)} potential bottlenecks")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to detect bottlenecks: {e}")
            return pd.DataFrame()
    
    def analyze_flow_direction_anomalies(self, df: pd.DataFrame, direction_col='rec_del_sign', 
                                       pipeline_col='pipeline_name', date_col='eff_gas_day') -> pd.DataFrame:
        """Analyze flow direction changes as potential anomalies."""
        try:
            df = df.sort_values([pipeline_col, date_col])
            df['prev_direction'] = df.groupby(pipeline_col)[direction_col].shift(1)
            df['direction_change'] = df[direction_col] != df['prev_direction']
            
            anomalies = df[df['direction_change'] == True]
            anomaly_counts = anomalies.groupby(pipeline_col).size().reset_index(name='direction_changes')
            
            logger.info(f"Found {len(anomaly_counts)} pipelines with direction changes")
            return anomaly_counts
            
        except Exception as e:
            logger.error(f"Failed to analyze flow direction anomalies: {e}")
            return pd.DataFrame()
    
    async def collect_ais_data(self, duration_sec: int = 30, 
                              url: str = "wss://stream.aisstream.io") -> pd.DataFrame:
        """Collect AIS tanker data from stream."""
        if not self.spatial_available:
            logger.warning("Cannot collect AIS data without websockets library")
            return pd.DataFrame()
        
        self.streamed_data = []
        
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"Connected to AISStream.io, collecting for {duration_sec}s")
                
                # Set timeout for data collection
                end_time = asyncio.get_event_loop().time() + duration_sec
                
                while asyncio.get_event_loop().time() < end_time:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        vessel_type = data.get('typeCode')
                        if vessel_type is not None and 70 <= int(vessel_type) <= 79:  # Tanker types
                            record = {
                                'mmsi': data.get('mmsi'),
                                'timestamp': datetime.utcfromtimestamp(data.get('timestamp', 0)),
                                'latitude': data.get('lat'),
                                'longitude': data.get('lon'),
                                'speed': data.get('speed'),
                                'course': data.get('course'),
                                'status': data.get('navStatus'),
                                'vessel_type': vessel_type,
                            }
                            self.streamed_data.append(record)
                            
                    except asyncio.TimeoutError:
                        continue
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"AIS data collection failed: {e}")
        
        df_tanker = pd.DataFrame(self.streamed_data)
        logger.info(f"Collected {len(df_tanker)} tanker records")
        return df_tanker
    
    def assign_tankers_to_pipelines(self, df_pipeline: pd.DataFrame, 
                                   df_tanker: pd.DataFrame) -> pd.DataFrame:
        """Spatially assign tankers to nearby pipeline locations."""
        if not self.spatial_available or len(df_tanker) == 0:
            return pd.DataFrame()
        
        try:
            max_dist_km = self.analytics_config['max_tanker_distance_km']
            
            # Prepare GeoDataFrames
            df_pipe_locs = df_pipeline[['loc_name', 'latitude', 'longitude']].drop_duplicates()
            df_pipe_locs = df_pipe_locs.dropna(subset=['latitude', 'longitude'])
            
            if len(df_pipe_locs) == 0:
                logger.warning("No valid pipeline coordinates for spatial join")
                return pd.DataFrame()
            
            gdf_pipe = gpd.GeoDataFrame(
                df_pipe_locs, 
                geometry=gpd.points_from_xy(df_pipe_locs.longitude, df_pipe_locs.latitude)
            )
            gdf_tanker = gpd.GeoDataFrame(
                df_tanker, 
                geometry=gpd.points_from_xy(df_tanker.longitude, df_tanker.latitude)
            )
            
            # Set CRS and create buffer
            gdf_pipe = gdf_pipe.set_crs(epsg=4326)
            gdf_tanker = gdf_tanker.set_crs(epsg=4326)
            
            # Buffer pipelines by approximate distance
            approx_buffer_deg = max_dist_km / 111  # Rough conversion km to degrees
            gdf_pipe['geometry'] = gdf_pipe.geometry.buffer(approx_buffer_deg)
            
            # Spatial join
            joined = gpd.sjoin(gdf_tanker, gdf_pipe, how='inner', predicate='within')
            
            logger.info(f"Assigned {len(joined)} tanker records to pipeline locations")
            return joined
            
        except Exception as e:
            logger.error(f"Failed to assign tankers to pipelines: {e}")
            return pd.DataFrame()
    
    async def run_comprehensive_analysis(self, df_pipeline: pd.DataFrame, 
                                       collect_ais: bool = False) -> Dict[str, Any]:
        """Run comprehensive pipeline analysis with optional AIS integration."""
        results = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_pipeline_analytics'
        }
        
        try:
            # 1. Basic pipeline analytics
            logger.info("Starting pipeline capacity and utilization analysis")
            capacities = self.infer_capacity(df_pipeline)
            df_util = self.calculate_utilization(df_pipeline, capacities)
            
            results['capacity_analysis'] = {
                'pipelines_analyzed': len(capacities),
                'avg_capacity': float(capacities.mean()) if len(capacities) > 0 else 0,
                'capacity_range': [float(capacities.min()), float(capacities.max())] if len(capacities) > 0 else [0, 0]
            }
            
            # 2. Bottleneck detection
            bottlenecks = self.detect_bottlenecks(df_util)
            results['bottlenecks'] = {
                'count': len(bottlenecks),
                'details': bottlenecks.to_dict('records') if len(bottlenecks) > 0 else []
            }
            
            # 3. Flow direction anomalies
            flow_anomalies = self.analyze_flow_direction_anomalies(df_util)
            results['flow_anomalies'] = {
                'pipelines_with_changes': len(flow_anomalies),
                'details': flow_anomalies.to_dict('records') if len(flow_anomalies) > 0 else []
            }
            
            # 4. Optional AIS integration
            if collect_ais and self.spatial_available:
                logger.info("Collecting AIS tanker data")
                duration = self.analytics_config['ais_stream_duration']
                df_tanker = await self.collect_ais_data(duration)
                
                if len(df_tanker) > 0:
                    assigned_tankers = self.assign_tankers_to_pipelines(df_util, df_tanker)
                    
                    results['ais_integration'] = {
                        'tankers_collected': len(df_tanker),
                        'tankers_assigned': len(assigned_tankers),
                        'locations_with_tankers': assigned_tankers['loc_name'].nunique() if len(assigned_tankers) > 0 else 0
                    }
                else:
                    results['ais_integration'] = {'status': 'no_tanker_data_collected'}
            else:
                results['ais_integration'] = {'status': 'disabled_or_unavailable'}
            
            logger.info("Comprehensive pipeline analysis completed")
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
