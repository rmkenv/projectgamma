""
Core agent class that orchestrates data analysis through natural language queries.
""

import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import pandas as pd
from rich.console import Console

from ..utils.anthropic_client import AnthropicClient
from ..core.data_processor import DataProcessor
from ..core.query_parser import QueryParser
from ..core.analysis_engine import AnalysisEngine
from ..utils.logger import get_logger

console = Console()
logger = get_logger()

class PipelineDataAgent:
    """
    Main agent class that coordinates data analysis through natural language queries.
    
    Capabilities:
    - Load and process pipeline datasets
    - Parse natural language queries
    - Execute statistical and ML analyses
    - Detect patterns, anomalies, and correlations
    - Provide evidence-based responses
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the agent with configuration."""
        self.config = self._load_config(config_path)
        self.data_processor = None
        self.analysis_engine = None
        self.query_parser = None
        self.anthropic_client = None
        self.conversation_history = []
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "agent": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "temperature": 0.1
            },
            "data": {
                "cache_directory": "./data",
                "auto_download_url": "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view?usp=sharing"
            },
            "analysis": {
                "anomaly_contamination": 0.1,
                "clustering_min_samples": 5,
                "correlation_threshold": 0.5,
                "max_display_rows": 20
            }
        }
    
    def _initialize_components(self):
        """Initialize all agent components."""
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.anthropic_client = AnthropicClient(
            api_key=api_key,
            model=self.config["agent"]["model"],
            max_tokens=self.config["agent"]["max_tokens"],
            temperature=self.config["agent"]["temperature"]
        )
        
        # Initialize other components (will be set up when data is loaded)
        self.query_parser = QueryParser(self.anthropic_client)
    
    async def setup_dataset(self, dataset_path: Optional[str] = None, auto_download: bool = False):
        """
        Setup the dataset for analysis.
        
        Args:
            dataset_path: Path to local dataset file
            auto_download: Whether to download dataset from configured URL
        """
        try:
            if auto_download:
                dataset_path = await self._download_dataset()
            
            if not dataset_path:
                raise ValueError("No dataset path provided")
            
            # Initialize data processor with the dataset
            self.data_processor = DataProcessor(dataset_path, self.config)
            await self.data_processor.load_data()
            
            # Initialize analysis engine with the loaded data
            self.analysis_engine = AnalysisEngine(
                self.data_processor.data, 
                self.config["analysis"]
            )
            
            logger.info(f"Dataset loaded successfully: {len(self.data_processor.data)} rows")
            
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise
    
    async def _download_dataset(self) -> str:
        """Download dataset from configured URL."""
        import requests
        from urllib.parse import urlparse
        
        cache_dir = Path(self.config["data"]["cache_directory"])
        cache_dir.mkdir(exist_ok=True)
        
        dataset_path = cache_dir / "pipeline_data.parquet"
        
        # Check if already cached
        if dataset_path.exists():
            logger.info("Using cached dataset")
            return str(dataset_path)
        
        # Download the dataset
        url = self.config["data"]["auto_download_url"]
        logger.info(f"Downloading dataset from {url}")
        
        # Handle Google Drive URL format
        if "drive.google.com" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        else:
            download_url = url
        
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(dataset_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Dataset downloaded to {dataset_path}")
            return str(dataset_path)
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    async def get_dataset_info(self) -> str:
        """Get formatted dataset information."""
        if not self.data_processor:
            return "No dataset loaded"
        
        info = self.data_processor.get_data_summary()
        
        # Format the information nicely
        formatted_info = f"""📊 **Dataset Overview**
        
**Shape:** {info['shape']['rows']:,} rows × {info['shape']['columns']} columns
**Columns:** {', '.join(info['columns'])}
**Data Types:**
{chr(10).join([f"• {col}: {dtype}" for col, dtype in info['dtypes'].items()])}
**Missing Values:** {info['missing_values']} total
**Memory Usage:** {info['memory_usage']}
**Date Range:** {info.get('date_range', 'No date column found')}
        """
        
        return formatted_info
    
    async def process_query(self, user_query: str) -> str:
        """
        Process a natural language query and return analysis results.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Formatted response with analysis results
        """
        try:
            if not self.data_processor or not self.analysis_engine:
                return "❌ Please load a dataset first using setup_dataset()"
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            
            # Parse the query to understand intent and extract parameters
            query_plan = await self.query_parser.parse_query(
                user_query, 
                self.data_processor.get_column_info(),
                self.conversation_history[-5:]  # Last 5 messages for context
            )
            
            logger.info(f"Query plan: {query_plan}")
            
            # Execute the analysis based on the query plan
            results = await self._execute_analysis_plan(query_plan)
            
            # Generate natural language response
            response = await self._generate_response(user_query, query_plan, results)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Error processing your query: {str(e)}"
    
    async def _execute_analysis_plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis plan and return results."""
        results = {}
        
        try:
            analysis_type = query_plan.get("analysis_type", "unknown")
            parameters = query_plan.get("parameters", {})
            
            if analysis_type == "simple_query":
                results = await self._execute_simple_query(parameters)
            elif analysis_type == "anomaly_detection":
                results = await self._execute_anomaly_detection(parameters)
            elif analysis_type == "pattern_analysis":
                results = await self._execute_pattern_analysis(parameters)
            elif analysis_type == "correlation_analysis":
                results = await self._execute_correlation_analysis(parameters)
            elif analysis_type == "statistical_summary":
                results = await self._execute_statistical_summary(parameters)
            elif analysis_type == "clustering":
                results = await self._execute_clustering(parameters)
            elif analysis_type == "causal_analysis":
                results = await self._execute_causal_analysis(parameters)
            else:
                # Default to general analysis
                results = await self._execute_general_analysis(parameters)
                
        except Exception as e:
            logger.error(f"Analysis execution error: {e}")
            results = {"error": str(e)}
        
        return results
    
    async def _execute_simple_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simple data queries like filtering, counting, etc."""
        return self.analysis_engine.execute_simple_query(params)
    
    async def _execute_anomaly_detection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection analysis."""
        return self.analysis_engine.detect_anomalies(params)
    
    async def _execute_pattern_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern recognition analysis."""
        return self.analysis_engine.analyze_patterns(params)
    
    async def _execute_correlation_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute correlation analysis."""
        return self.analysis_engine.analyze_correlations(params)
    
    async def _execute_statistical_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical summary analysis."""
        return self.analysis_engine.statistical_summary(params)
    
    async def _execute_clustering(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute clustering analysis."""
        return self.analysis_engine.perform_clustering(params)
    
    async def _execute_causal_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal analysis."""
        return self.analysis_engine.analyze_causal_relationships(params)
    
    async def _execute_general_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general exploratory analysis."""
        return self.analysis_engine.general_analysis(params)
    
    async def _generate_response(self, user_query: str, query_plan: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Generate a natural language response from analysis results."""
        
        # Create context for the response generation
        context = {
            "user_query": user_query,
            "analysis_type": query_plan.get("analysis_type"),
            "parameters_used": query_plan.get("parameters", {}),
            "results": results,
            "dataset_info": {
                "columns": self.data_processor.get_column_info(),
                "shape": self.data_processor.data.shape
            }
        }
        
        # Use Anthropic to generate natural language response
        response = await self.anthropic_client.generate_analysis_response(context)
        
        return response
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
