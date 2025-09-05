"""
Anthropic Claude client for natural language processing and response generation.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import json
import anthropic
from anthropic import Anthropic

from .logger import get_logger

logger = get_logger()


class AnthropicClient:
    """
    Client for Anthropic Claude API interactions.
    Handles query parsing, analysis planning, and response generation.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", 
                 max_tokens: int = 4000, temperature: float = 0.1):
        """Initialize Anthropic client."""
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = Anthropic(api_key=api_key)
        
        logger.info(f"Initialized Anthropic client with model: {model}")
    
    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Complete a conversation with Claude.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters for the API call
        
        Returns:
            Claude's response as a string
        """
        try:
            # Merge kwargs with defaults
            params = {
                'model': self.model,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'messages': messages
            }
            
            # Make the API call
            response = self.client.messages.create(**params)
            
            # Extract the text content
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    async def generate_analysis_response(self, context: Dict[str, Any]) -> str:
        """
        Generate a natural language response from analysis results.
        
        Args:
            context: Dictionary containing query context and analysis results
            
        Returns:
            Formatted natural language response
        """
        
        try:
            # Create the response generation prompt
            prompt = self._create_response_prompt(context)
            
            # Generate response
            messages = [{"role": "user", "content": prompt}]
            response = await self.complete(messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"❌ Error generating response: {str(e)}"
    
    def _create_response_prompt(self, context: Dict[str, Any]) -> str:
        """Create a prompt for generating analysis responses."""
        
        user_query = context.get('user_query', '')
        analysis_type = context.get('analysis_type', 'unknown')
        parameters = context.get('parameters_used', {})
        results = context.get('results', {})
        dataset_info = context.get('dataset_info', {})
        
        prompt = f"""You are a professional data analyst providing insights about pipeline data. Generate a clear, comprehensive response to the user's query based on the analysis results.

**User Query:** "{user_query}"

**Analysis Type:** {analysis_type}

**Dataset Context:**
- Columns: {list(dataset_info.get('columns', {}).keys())}
- Data Shape: {dataset_info.get('shape', 'Unknown')}

**Analysis Parameters Used:** {json.dumps(parameters, indent=2)}

**Analysis Results:** {json.dumps(results, indent=2, default=str)}

**Response Guidelines:**
1. Start with a direct answer to the user's question
2. Present key findings with specific numbers and evidence
3. Use clear formatting with headers, bullet points, and emphasis
4. Include methodology and parameters used for transparency
5. Highlight any limitations or caveats
6. For anomalies/patterns, provide specific examples and explanations
7. For correlations, interpret the strength and significance
8. End with actionable insights or recommendations when relevant

**Formatting:**
- Use **bold** for important findings
- Use bullet points for lists
- Include specific numbers and percentages
- Use sections with clear headers
- Cite the methodology used

Generate a professional, insightful response that demonstrates expertise while being accessible to business users."""
        
        return prompt
    
    async def parse_query_intent(self, user_query: str, column_info: Dict[str, Any], 
                                conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Parse user query to understand intent and extract parameters.
        
        Args:
            user_query: Natural language query from user
            column_info: Information about available columns
            conversation_history: Recent conversation context
            
        Returns:
            Parsed query information
        """
        
        try:
            prompt = self._create_parsing_prompt(user_query, column_info, conversation_history)
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.complete(messages, temperature=0.1)
            
            # Try to parse as JSON
            try:
                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError:
                logger.warning("Failed to parse query intent as JSON")
                return self._create_fallback_intent(user_query)
                
        except Exception as e:
            logger.error(f"Query intent parsing failed: {e}")
            return self._create_fallback_intent(user_query)
    
    def _create_parsing_prompt(self, user_query: str, column_info: Dict[str, Any], 
                              conversation_history: List[Dict[str, str]] = None) -> str:
        """Create prompt for query intent parsing."""
        
        # Summarize column information
        column_summaries = []
        for col_name, info in column_info.items():
            summary = f"{col_name} ({info.get('dtype', 'unknown')})"
            if info.get('unique_count', 0) < 20 and 'value_counts' in info:
                examples = list(info['value_counts'].keys())[:3]
                summary += f" - examples: {examples}"
            column_summaries.append(summary)
        
        prompt = f"""Analyze this data query and extract the analysis intent and parameters.

**Available Dataset Columns:**
{chr(10).join(column_summaries)}

**User Query:** "{user_query}"

**Recent Conversation:**
{json.dumps(conversation_history or [], indent=2)}

Determine the analysis type and extract relevant parameters. Respond with valid JSON:

{{
  "analysis_type": "one of: simple_query, anomaly_detection, pattern_analysis, correlation_analysis, statistical_summary, clustering, causal_analysis, general_analysis",
  "parameters": {{
    "columns": ["list of specific column names mentioned or relevant"],
    "filters": {{"column_name": "value or condition"}},
    "aggregation": "count/sum/mean/median/max/min or null",
    "specific_values": ["any specific values mentioned"],
    "analysis_focus": "brief description of what to analyze",
    "query_type": "specific type like 'count', 'find', 'compare', 'analyze'"
  }},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of the analysis decision"
}}

**Analysis Type Guidelines:**
- simple_query: Basic filtering, counting, showing data
- anomaly_detection: Finding outliers, unusual patterns, anomalies
- pattern_analysis: Identifying trends, relationships, behaviors
- correlation_analysis: Comparing variables, finding associations
- statistical_summary: Descriptive statistics, distributions
- clustering: Grouping similar items, segmentation
- causal_analysis: Explaining why something happens, causes
- general_analysis: Exploratory analysis, broad investigation

**Column Matching:**
- Use exact column names from the list above
- Consider synonyms (e.g., "quantity" → "scheduled_quantity", "state" → "state_abb")
- Include all relevant columns for the analysis

Respond only with valid JSON."""
        
        return prompt
    
    def _create_fallback_intent(self, user_query: str) -> Dict[str, Any]:
        """Create fallback intent when parsing fails."""
        return {
            "analysis_type": "general_analysis",
            "parameters": {
                "query_text": user_query,
                "analysis_focus": "general data exploration"
            },
            "confidence": 0.1,
            "reasoning": "Fallback response due to parsing failure"
        }
    
    async def explain_analysis_results(self, analysis_type: str, results: Dict[str, Any], 
                                     context: Dict[str, Any] = None) -> str:
        """
        Generate explanation for analysis results.
        
        Args:
            analysis_type: Type of analysis performed
            results: Analysis results to explain
            context: Additional context information
            
        Returns:
            Human-readable explanation of the results
        """
        
        try:
            prompt = f"""Explain these {analysis_type} results in clear, professional language:

**Analysis Results:**
{json.dumps(results, indent=2, default=str)}

**Context:**
{json.dumps(context or {}, indent=2, default=str)}

**Instructions:**
1. Summarize the key findings
2. Explain what the numbers mean in business terms
3. Highlight important patterns or insights
4. Include methodology transparency
5. Note any limitations or caveats
6. Use clear formatting and professional tone

Generate a comprehensive explanation suitable for business users."""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.complete(messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return f"Analysis completed. Results: {str(results)[:500]}..."
    
    async def suggest_follow_up_questions(self, user_query: str, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Suggest relevant follow-up questions based on analysis results.
        
        Args:
            user_query: Original user query
            analysis_results: Results from the analysis
            
        Returns:
            List of suggested follow-up questions
        """
        
        try:
            prompt = f"""Based on the user's query and analysis results, suggest 3-5 relevant follow-up questions.

**Original Query:** "{user_query}"

**Analysis Results Summary:**
{json.dumps(analysis_results, indent=2, default=str)[:1000]}...

Generate specific, actionable follow-up questions that would provide additional insights. Return as a JSON list of strings.

Example format:
["Question 1?", "Question 2?", "Question 3?"]

Focus on:
- Deeper analysis of interesting findings
- Related aspects not covered
- Potential business implications
- Data quality or methodology questions
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.complete(messages, temperature=0.3)
            
            try:
                suggestions = json.loads(response)
                return suggestions if isinstance(suggestions, list) else []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"Follow-up suggestion failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Check if the Anthropic API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.complete(test_messages, max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
