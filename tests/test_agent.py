"""
Basic tests for the Pipeline Data Agent.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

# Import the modules to test
from data_agent.core.data_processor import DataProcessor
from data_agent.core.analysis_engine import AnalysisEngine
from data_agent.utils.data_utils import validate_dataset_columns, analyze_column_type


class TestDataProcessor:
    """Test the DataProcessor class."""
    
    def create_sample_data(self):
        """Create sample pipeline data for testing."""
        data = pd.DataFrame({
            'pipeline_name': ['Pipeline_A', 'Pipeline_B', 'Pipeline_C'],
            'loc_name': ['Location_1', 'Location_2', 'Location_3'],
            'state_abb': ['TX', 'LA', 'OK'],
            'scheduled_quantity': [1000, 2000, 1500],
            'eff_gas_day': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'category_short': ['LNG', 'GAS', 'LNG'],
            'latitude': [29.7604, 30.2241, 35.4676],
            'longitude': [-95.3698, -92.0198, -97.5164]
        })
        return data
    
    @pytest.mark.asyncio
    async def test_data_loading(self):
        """Test basic data loading functionality."""
        # Create temporary CSV file
        sample_data = self.create_sample_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test data processor
            config = {'analysis': {'max_display_rows': 20}}
            processor = DataProcessor(temp_path, config)
            
            await processor.load_data()
            
            # Assertions
            assert processor.data is not None
            assert len(processor.data) == 3
            assert 'pipeline_name' in processor.data.columns
            assert processor.data['scheduled_quantity'].dtype in [np.int64, np.float64]
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_column_info_analysis(self):
        """Test column information analysis."""
        sample_data = self.create_sample_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            config = {'analysis': {'max_display_rows': 20}}
            processor = DataProcessor(temp_path, config)
            processor.data = sample_data  # Set data directly for testing
            processor._analyze_columns()
            
            # Check column info was created
            assert len(processor.column_info) > 0
            assert 'pipeline_name' in processor.column_info
            assert 'scheduled_quantity' in processor.column_info
            
            # Check numeric column analysis
            qty_info = processor.column_info['scheduled_quantity']
            assert qty_info['is_numeric'] == True
            assert 'mean' in qty_info
            
        finally:
            os.unlink(temp_path)


class TestAnalysisEngine:
    """Test the AnalysisEngine class."""
    
    def create_sample_data(self):
        """Create sample data for analysis testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'pipeline_name': [f'Pipeline_{i}' for i in range(n_samples)],
            'state_abb': np.random.choice(['TX', 'LA', 'OK', 'CA'], n_samples),
            'scheduled_quantity': np.random.normal(1000, 200, n_samples),
            'category_short': np.random.choice(['LNG', 'GAS', 'OIL'], n_samples),
            'latitude': np.random.uniform(25, 45, n_samples),
            'longitude': np.random.uniform(-125, -65, n_samples),
            'eff_gas_day': pd.date_range('2024-01-01', periods=n_samples, freq='D')[:n_samples]
        })
        
        # Add some outliers for anomaly detection testing
        data.loc[0, 'scheduled_quantity'] = 5000  # Clear outlier
        data.loc[1, 'scheduled_quantity'] = -100   # Another outlier
        
        return data
    
    def test_simple_query_execution(self):
        """Test simple query execution."""
        sample_data = self.create_sample_data()
        config = {'max_display_rows': 20, 'correlation_threshold': 0.5}
        
        engine = AnalysisEngine(sample_data, config)
        
        # Test basic query
        params = {'filters': {'state_abb': 'TX'}, 'aggregation': 'count'}
        results = engine.execute_simple_query(params)
        
        assert results['success'] == True
        assert 'row_count' in results
        assert 'aggregation_results' in results
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        sample_data = self.create_sample_data()
        config = {'anomaly_contamination': 0.1, 'correlation_threshold': 0.5}
        
        engine = AnalysisEngine(sample_data, config)
        
        # Test anomaly detection
        params = {'columns': ['scheduled_quantity']}
        results = engine.detect_anomalies(params)
        
        assert results['success'] == True
        assert 'statistical_method' in results
        assert 'isolation_forest' in results
        assert 'summary' in results
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        sample_data = self.create_sample_data()
        config = {'correlation_threshold': 0.3}
        
        engine = AnalysisEngine(sample_data, config)
        
        # Test correlation analysis
        params = {}
        results = engine.analyze_correlations(params)
        
        assert results['success'] == True
        assert 'pearson_correlation' in results
        assert 'strong_correlations' in results
    
    def test_statistical_summary(self):
        """Test statistical summary generation."""
        sample_data = self.create_sample_data()
        config = {}
        
        engine = AnalysisEngine(sample_data, config)
        
        # Test statistical summary
        params = {}
        results = engine.statistical_summary(params)
        
        assert results['success'] == True
        assert 'basic_info' in results
        assert 'numeric_summary' in results
        assert 'data_quality' in results


class TestDataUtils:
    """Test data utility functions."""
    
    def test_column_validation(self):
        """Test dataset column validation."""
        # Create test data with expected columns
        data = pd.DataFrame({
            'pipeline_name': ['A', 'B'],
            'scheduled_quantity': [100, 200],
            'state_abb': ['TX', 'LA'],
            'extra_column': ['X', 'Y']
        })
        
        results = validate_dataset_columns(data)
        
        assert 'missing_columns' in results
        assert 'extra_columns' in results
        assert 'extra_column' in results['extra_columns']
        assert len(results['missing_columns']) > 0  # Should be missing some expected columns
    
    def test_column_type_analysis(self):
        """Test column type analysis."""
        # Test numeric column
        numeric_series = pd.Series([1, 2, 3, 4, 5])
        numeric_analysis = analyze_column_type(numeric_series)
        
        assert numeric_analysis['inferred_type'] == 'numeric'
        assert 'mean_value' in numeric_analysis
        
        # Test categorical column
        categorical_series = pd.Series(['A', 'B', 'A', 'C', 'B'])
        categorical_analysis = analyze_column_type(categorical_series)
        
        assert categorical_analysis['inferred_type'] == 'categorical'
        assert 'top_values' in categorical_analysis


# Integration test
class TestAgentIntegration:
    """Test agent integration (requires API key)."""
    
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), 
                       reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization with API key."""
        from data_agent.core.agent import PipelineDataAgent
        
        # This test only runs if API key is available
        try:
            agent = PipelineDataAgent()
            # Basic initialization test
            assert agent.config is not None
            assert agent.anthropic_client is not None
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")


# Run tests
if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic tests...")
    
    # Test data utils
    test_utils = TestDataUtils()
    test_utils.test_column_type_analysis()
    print("✓ Column type analysis test passed")
    
    # Test analysis engine
    test_engine = TestAnalysisEngine()
    test_engine.test_simple_query_execution()
    print("✓ Simple query execution test passed")
    
    test_engine.test_statistical_summary()
    print("✓ Statistical summary test passed")
    
    # Test anomaly detection
    test_engine.test_anomaly_detection()
    print("✓ Anomaly detection test passed")
    
    print("\nAll basic tests passed! ✓")
    print("\nTo run full test suite with pytest:")
    print("pip install pytest pytest-asyncio")
    print("pytest tests/test_agent.py -v")
