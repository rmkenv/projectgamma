"""
Pipeline Data Analysis Agent

A sophisticated AI-powered data agent for analyzing pipeline data through natural language queries.
Supports pattern recognition, anomaly detection, and causal analysis.
"""

__version__ = "1.0.0"
__author__ = "Pipeline Data Agent Team"

from .core.agent import PipelineDataAgent
from .core.data_processor import DataProcessor
from .core.analysis_engine import AnalysisEngine
from .core.query_parser import QueryParser

__all__ = [
    "PipelineDataAgent",
    "DataProcessor", 
    "AnalysisEngine",
    "QueryParser"
]
