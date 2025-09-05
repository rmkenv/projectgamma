# Natural language query parsing and intent recognition.

import re
from typing import Dict, Any, List, Optional
import json
from ..utils.anthropic_client import AnthropicClient
from ..utils.logger import get_logger

logger = get_logger()


class QueryParser:
    """
    Parses natural language queries and converts them into structured analysis plans.
    
    Uses LLM to understand user intent and extract relevant parameters for data analysis.
    """
    
    def __init__(self, anthropic_client: AnthropicClient):
        """Initialize query parser with Anthropic client."""
        self.anthropic_client = anthropic_client
        
        # Keywords for different analysis types (stemmed/partial forms are fine)
        self.analysis_keywords = {
            'anomaly_detection': [
                'anomal', 'outlier', 'unusual', 'strange', 'weird', 'abnormal',
                'suspicious', 'irregular', 'deviation', 'exception'
            ],
            'pattern_analysis': [
                'pattern', 'trend', 'relationship', 'association', 'connection',
                'link', 'behavior', 'sequence', 'series'
            ],
            'correlation_analysis': [
                'correlat', 'relationship', 'association', 'connect', 'relate',
                'compare', ' vs ', 'versus', ' against ', ' between '
            ],
            'clustering': [
                'cluster', 'group', 'segment', 'categor', 'classify',
                'similar', 'alike', 'partition'
            ],
            'statistical_summary': [
                'summary', 'describe', 'statistics', 'stats', 'overview',
                'distribution', 'average', 'mean', 'median', 'count'
            ],
            'simple_query': [
                'show', 'display', 'list', 'get', 'find', 'select',
                'how many', 'count', 'filter', 'where'
            ],
            'causal_analysis': [
                'why', 'cause', 'reason', 'explain', 'because', 'due to',
                'factor', 'influence', 'impact', 'effect'
            ],
            # New: pipeline operations (capacity/utilization/bottlenecks/AIS)
            'pipeline_operations': [
                'bottleneck', 'capacity', 'utilization', 'utilisation',
                'flow direction', 'reverse flow', 'pipeline efficiency',
                'throughput', 'constraint', 'choke point',
                'tanker', 'vessel', 'ais', 'shipping'
            ]
        }

        # Fast lookup set for AIS toggling
        self._ais_keywords = {'tanker', 'vessel', 'ais', 'shipping'}
    
    async def parse_query(self, user_query: str, column_info: Dict[str, Any], 
                         conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Parse a natural language query into a structured analysis plan.
        
        Args:
            user_query: Natural language query from user
            column_info: Information about available columns
            conversation_history: Recent conversation context
            
        Returns:
            Dictionary containing analysis type and parameters
        """
        try:
            # First, try rule-based parsing for simple cases
            rule_based_result = self._rule_based_parsing(user_query, column_info)
            
            # If we detected pipeline_operations, ensure AIS flag is surfaced
            if rule_based_result and rule_based_result.get('analysis_type') == 'pipeline_operations':
                include_ais = any(k in user_query.lower() for k in self._ais_keywords)
                rule_based_result['parameters']['include_ais_data'] = include_ais

            if rule_based_result and rule_based_result.get('confidence', 0) > 0.8:
                logger.info("Using rule-based parsing result")
                return rule_based_result
            
            # For complex queries, use LLM-based parsing
            llm_result = await self._llm_based_parsing(user_query, column_info, conversation_history)

            # If LLM chose pipeline_operations, add AIS inference too
            if llm_result and llm_result.get('analysis_type') == 'pipeline_operations':
                params = llm_result.setdefault('parameters', {})
                include_ais = any(k in user_query.lower() for k in self._ais_keywords)
                params['include_ais_data'] = include_ais
            
            # Combine results if needed
            if rule_based_result and llm_result:
                # Use LLM result but incorporate high-confidence rule-based insights
                return self._merge_parsing_results(rule_based_result, llm_result)
            
            return llm_result or rule_based_result or self._default_analysis_plan(user_query)
            
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return self._default_analysis_plan(user_query)
    
    def _rule_based_parsing(self, query: str, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use rule-based pattern matching for simple query parsing."""
        
        query_lower = query.lower()
        
        # Detect analysis type based on keywords
        analysis_type = self._detect_analysis_type(query_lower)
        
        # Extract column references
        mentioned_columns = self._extract_column_references(query_lower, column_info)
        
        # Extract filters and conditions
        filters = self._extract_filters(query_lower, column_info)
        
        # Extract aggregation type
        aggregation = self._extract_aggregation_type(query_lower)
        
        # Calculate confidence based on matches
        confidence = self._calculate_confidence(query_lower, analysis_type, mentioned_columns, filters, aggregation)
        
        params: Dict[str, Any] = {
            'columns': mentioned_columns,
            'filters': filters,
            'aggregation': aggregation,
            'query_text': query
        }

        # If pipeline ops, add AIS inference here as well
        if analysis_type == 'pipeline_operations':
            params['include_ais_data'] = any(k in query_lower for k in self._ais_keywords)
        
        return {
            'analysis_type': analysis_type,
            'parameters': params,
            'confidence': confidence,
            'method': 'rule_based'
        }
    
    def _detect_analysis_type(self, query_lower: str) -> str:
        """Detect the type of analysis based on keywords."""
        type_scores = {}
        for analysis_type, keywords in self.analysis_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[analysis_type] = score
        
        if not type_scores:
            return 'general_analysis'
        
        # Return the analysis type with highest score
        return max(type_scores, key=type_scores.get)
    
    def _extract_column_references(self, query_lower: str, column_info: Dict[str, Any]) -> List[str]:
        """Extract column names mentioned in the query."""
        mentioned_columns = []
        
        # Check for exact column name matches
        for col_name in column_info.keys():
            col_name_lower = col_name.lower()
            
            # Direct mention
            if col_name_lower in query_lower:
                mentioned_columns.append(col_name)
                continue
            
            # Check for partial matches or aliases
            aliases = self._get_column_aliases(col_name)
            for alias in aliases:
                if alias.lower() in query_lower:
                    mentioned_columns.append(col_name)
                    break
        
        return list(set(mentioned_columns))  # Remove duplicates
    
    def _get_column_aliases(self, column_name: str) -> List[str]:
        """Get alternative names/aliases for a column."""
        aliases = [column_name]
        alias_map = {
            'scheduled_quantity': ['quantity', 'amount', 'volume', 'scheduled'],
            'pipeline_name': ['pipeline', 'name'],
            'state_abb': ['state', 'states'],
            'country_name': ['country', 'countries'],
            'eff_gas_day': ['date', 'day', 'time', 'gas day'],
            'latitude': ['lat'],
            'longitude': ['lon', 'lng'],
            'rec_del_sign': ['delivery', 'receipt', 'sign'],
            'category_short': ['category', 'type', 'cat'],
            'loc_name': ['location', 'place']
        }
        if column_name in alias_map:
            aliases.extend(alias_map[column_name])
        return aliases
    
    def _extract_filters(self, query_lower: str, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract filter conditions from the query."""
        filters: Dict[str, Any] = {}

        # State: capture two-letter codes after 'in', 'from', or before 'state'
        # e.g., "in TX", "from PA", "TX state pipelines"
        state_matches = re.findall(r'(?:in|from)\s+([a-z]{2})\b|([a-z]{2})\s+state', query_lower)
        for sm in state_matches:
            state = next((s for s in sm if s), None)
            if state and len(state) == 2:
                filters['state_abb'] = state.upper()

        # Year: "in 2023", "since 2022", or bare 20xx/19xx
        ym = re.findall(r'(?:in|since|for)\s+(19|20)\d{2}|\b(19|20)\d{2}\b', query_lower)
        if ym:
            # ym is list of tuples; flatten to the first full 4-digit token found in query
            year_match = re.search(r'\b(19|20)\d{2}\b', query_lower)
            if year_match:
                filters['year'] = int(year_match.group(0))

        # Categories: "category X" or "type Y"
        category_matches = re.findall(r'(?:category|type)\s+([a-z0-9_]+)', query_lower)
        if category_matches:
            filters['category_short'] = category_matches[0].upper()

        # Simple direction hints -> flow direction analysis (optional flag)
        if 'reverse flow' in query_lower or 'flow direction' in query_lower:
            filters['flow_direction_focus'] = True

        return filters
    
    def _extract_aggregation_type(self, query_lower: str) -> Optional[str]:
        """Extract aggregation type from query."""
        if any(word in query_lower for word in ['count', 'how many', 'number of']):
            return 'count'
        elif any(word in query_lower for word in ['average', 'mean']):
            return 'mean'
        elif 'sum' in query_lower or 'total' in query_lower:
            return 'sum'
        elif any(word in query_lower for word in ['max', 'maximum', 'highest']):
            return 'max'
        elif any(word in query_lower for word in ['min', 'minimum', 'lowest']):
            return 'min'
        elif 'median' in query_lower:
            return 'median'
        return None
    
    def _calculate_confidence(self, query_lower: str, analysis_type: str, columns: List[str],
                              filters: Dict[str, Any], aggregation: Optional[str]) -> float:
        """Calculate confidence score for rule-based parsing."""
        confidence = 0.0
        
        # Base confidence for analysis type detection
        if analysis_type != 'general_analysis':
            confidence += 0.35
        
        # Confidence boost for column detection
        if columns:
            confidence += 0.35

        # Confidence boost for filters/aggregation presence
        if filters:
            confidence += 0.15
        if aggregation:
            confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    async def _llm_based_parsing(self, user_query: str, column_info: Dict[str, Any], 
                                conversation_history: List[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to parse complex queries."""
        context = self._build_parsing_context(user_query, column_info, conversation_history)
        parsing_prompt = self._create_parsing_prompt(context)
        
        try:
            # Get LLM response. The client may return a string or an object with .content
            response = await self.anthropic_client.complete(
                messages=[{"role": "user", "content": parsing_prompt}],
                max_tokens=800,
                temperature=0.1
            )

            # Normalize response to string
            if isinstance(response, str):
                raw = response
            elif isinstance(response, dict):
                # Some clients wrap under 'content' or 'completion'
                raw = response.get('content') or response.get('completion') or json.dumps(response)
            else:
                # Try attribute access
                raw = getattr(response, 'content', None) or getattr(response, 'completion', None)
                if raw is None:
                    raw = str(response)

            # If Anthropic returns a list of content blocks, join their text
            if isinstance(raw, list):
                raw = ''.join([blk.get('text', '') if isinstance(blk, dict) else str(blk) for blk in raw])

            result = json.loads(raw)

            # Validate the result
            validated_result = self._validate_parsing_result(result, column_info)
            return validated_result
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return None
    
    def _build_parsing_context(self, user_query: str, column_info: Dict[str, Any], 
                              conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Build context information for LLM parsing."""
        columns_summary = []
        for col_name, info in column_info.items():
            dtype = info.get('dtype', 'unknown')
            col_desc = f"{col_name} ({dtype})"
            if info.get('unique_count', 0) < 20 and 'value_counts' in info:
                examples = list(info['value_counts'].keys())[:5]
                col_desc += f" - examples: {examples}"
            columns_summary.append(col_desc)
        
        context = {
            'user_query': user_query,
            'available_columns': columns_summary,
            'conversation_history': conversation_history or []
        }
        return context
    
    def _create_parsing_prompt(self, context: Dict[str, Any]) -> str:
        """Create a prompt for LLM-based query parsing."""
        prompt = f"""You are a data analysis query parser. Parse the user's natural language query into a structured analysis plan.

Available dataset columns:
{chr(10).join(context['available_columns'])}

User query: "{context['user_query']}"

Recent conversation context:
{json.dumps(context['conversation_history'][-3:], indent=2) if context['conversation_history'] else 'None'}

Analyze the query and respond with a JSON object containing:
{{
  "analysis_type": "one of: simple_query, anomaly_detection, pattern_analysis, correlation_analysis, statistical_summary, clustering, causal_analysis, pipeline_operations, general_analysis",
  "parameters": {{
    "columns": ["list of relevant column names"],
    "filters": {{"column_name": "filter_value or condition"}},
    "aggregation": "count/sum/mean/median/max/min or null",
    "specific_values": ["any specific values mentioned"],
    "analysis_focus": "brief description of what to analyze"
  }},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of the parsing decision"
}}

Focus on:
1. Identifying the main intent (what type of analysis is requested)
2. Extracting relevant column names (use exact names from the list above)
3. Identifying any filters or conditions
4. Understanding the scope of analysis

Respond only with valid JSON."""
        return prompt
    
    def _validate_parsing_result(self, result: Dict[str, Any], column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the parsing result from LLM."""
        if not isinstance(result, dict):
            return self._default_analysis_plan(str(result))
        
        # Ensure required fields exist
        result.setdefault('analysis_type', 'general_analysis')
        params = result.setdefault('parameters', {})
        
        # Validate column names
        cols = params.get('columns', [])
        if cols and isinstance(cols, list):
            valid_columns = []
            for col in cols:
                if col in column_info:
                    valid_columns.append(col)
                else:
                    closest = self._find_closest_column(col, column_info)
                    if closest:
                        valid_columns.append(closest)
            params['columns'] = list(dict.fromkeys(valid_columns))  # dedupe while preserving order
        
        # Ensure filters exist as dict
        if 'filters' in params and not isinstance(params['filters'], dict):
            params['filters'] = {}
        
        # If LLM detected pipeline ops, infer AIS inclusion
        if result.get('analysis_type') == 'pipeline_operations':
            include_ais = any(k in params.get('analysis_focus', '').lower() for k in self._ais_keywords)
            # Also consider the original query text if present
            qtext = params.get('query_text', '')
            if not include_ais and isinstance(qtext, str):
                include_ais = any(k in qtext.lower() for k in self._ais_keywords)
            params['include_ais_data'] = params.get('include_ais_data', include_ais)

        result['parameters'] = params
        result['method'] = 'llm_based'
        return result
    
    def _find_closest_column(self, target: str, column_info: Dict[str, Any]) -> Optional[str]:
        """Find the closest matching column name."""
        if not target:
            return None

        target_lower = target.lower()
        
        # Direct match
        for col_name in column_info.keys():
            if col_name.lower() == target_lower:
                return col_name
        
        # Partial match
        for col_name in column_info.keys():
            if target_lower in col_name.lower() or col_name.lower() in target_lower:
                return col_name
        
        # Check aliases
        for col_name in column_info.keys():
            aliases = self._get_column_aliases(col_name)
            if any(alias.lower() == target_lower for alias in aliases):
                return col_name
        
        return None
    
    def _merge_parsing_results(self, rule_based: Dict[str, Any], llm_based: Dict[str, Any]) -> Dict[str, Any]:
        """Merge rule-based and LLM-based parsing results."""
        merged = llm_based.copy()
        
        # Override with high-confidence rule-based insights
        if rule_based.get('confidence', 0) > 0.8:
            if rule_based.get('analysis_type') and rule_based['analysis_type'] != 'general_analysis':
                merged['analysis_type'] = rule_based['analysis_type']
            
            # Merge parameters
            merged_params = merged.setdefault('parameters', {})
            rule_params = rule_based.get('parameters', {})
            
            if rule_params.get('columns'):
                merged_params['columns'] = list(set(
                    merged_params.get('columns', []) + rule_params['columns']
                ))
            if rule_params.get('filters'):
                merged_params['filters'] = {
                    **merged_params.get('filters', {}),
                    **rule_params['filters']
                }
            if 'include_ais_data' in rule_params:
                merged_params['include_ais_data'] = rule_params['include_ais_data']
        
        merged['method'] = 'hybrid'
        return merged
    
    def _default_analysis_plan(self, user_query: str) -> Dict[str, Any]:
        """Create a default analysis plan when parsing fails."""
        return {
            'analysis_type': 'general_analysis',
            'parameters': {
                'query_text': user_query,
                'columns': [],
                'analysis_focus': 'general data exploration'
            },
            'confidence': 0.1,
            'method': 'fallback'
        }
