
# Example queries and expected outputs for the Pipeline Data Agent. Use these as templates for testing and demonstration.


# Simple data queries
SIMPLE_QUERIES = [
    "How many pipeline records are there?",
    "Show me pipelines in Texas",
    "What are the unique categories in the dataset?",
    "Count pipelines by state",
    "List all pipeline names",
    "Show me data from 2024",
    "What states have the most pipelines?",
    "Display the first 10 records"
]

# Statistical analysis queries
STATISTICAL_QUERIES = [
    "What's the average scheduled quantity?",
    "Show me statistics for scheduled quantities",
    "What's the distribution of pipelines by category?",
    "Calculate median scheduled quantity by state",
    "Show me the data summary",
    "What's the range of scheduled quantities?",
    "Which category has the highest average quantity?"
]

# Anomaly detection queries
ANOMALY_QUERIES = [
    "Find anomalies in scheduled quantities",
    "Detect outliers in the data",
    "Show me unusual pipeline configurations",
    "Find pipelines with suspicious delivery patterns",
    "Identify strange scheduled quantities",
    "What data points look abnormal?",
    "Find pipelines that don't fit the normal pattern"
]

# Pattern recognition queries
PATTERN_QUERIES = [
    "What patterns do you see in the data?",
    "Show trends in scheduled quantities over time",
    "Find patterns in pipeline locations",
    "Analyze seasonal patterns in gas deliveries",
    "What relationships exist between variables?",
    "Show me geographic patterns",
    "Find trends by category"
]

# Correlation analysis queries
CORRELATION_QUERIES = [
    "Find correlations between scheduled quantity and location",
    "Show relationships between state and category",
    "Compare scheduled quantities across different states",
    "How does category relate to delivery sign?",
    "Analyze connection between geographic location and quantities",
    "What variables are correlated?",
    "Show me state vs quantity relationships"
]

# Clustering queries
CLUSTERING_QUERIES = [
    "Cluster pipelines by their characteristics",
    "Group similar pipelines together",
    "Segment pipelines by location and quantity",
    "Find groups in the pipeline data",
    "Classify pipelines into categories",
    "Identify pipeline segments",
    "Group pipelines by similar properties"
]

# Causal analysis queries
CAUSAL_QUERIES = [
    "Why do certain pipelines have higher scheduled quantities?",
    "What factors explain the geographic distribution of pipelines?",
    "What causes variation in scheduled quantities?",
    "Explain the relationship between category and delivery patterns",
    "Why are there more pipelines in some states?",
    "What influences pipeline scheduling?",
    "What drives the differences between pipeline categories?"
]

# Complex multi-step queries
COMPLEX_QUERIES = [
    "Analyze Texas pipelines for anomalies and patterns",
    "Find correlations in the data and explain potential causes",
    "Cluster pipelines and analyze each cluster for anomalies",
    "Compare pipeline patterns between different states and categories",
    "Identify unusual pipelines and suggest reasons for the anomalies",
    "Analyze seasonal trends and their relationship to categories",
    "Find the most important factors that influence scheduled quantities"
]

# Expected outputs templates
EXPECTED_OUTPUTS = {
    "simple_query": {
        "format": "Direct answer with supporting data",
        "example": "Found 1,234 pipeline records in the dataset.\nTop 5 states by pipeline count:\n- TX: 456 pipelines\n- LA: 234 pipelines\n- OK: 189 pipelines\n..."
    },
    
    "anomaly_detection": {
        "format": "Method description, anomaly count, examples, evidence",
        "example": "**Anomaly Detection Results**\n\n**Method: Isolation Forest**\n- Detected 45 anomalies (4.2% of data)\n- Top anomalies: Pipeline_XYZ, Pipeline_ABC\n\n**Evidence:**\n- Columns analyzed: ['scheduled_quantity']\n- Algorithm parameters: contamination=0.1\n..."
    },
    
    "correlation_analysis": {
        "format": "Correlation matrix, strong relationships, interpretation",
        "example": "**Correlation Analysis Results**\n\n**Strong Correlations Found:**\n- State vs Quantity: r=0.73 (strong positive)\n- Category vs Delivery Sign: χ²=45.2, p<0.001\n\n**Interpretation:**\n- Pipeline location strongly influences scheduled quantities\n..."
    }
}

# Test scenarios for development
TEST_SCENARIOS = [
    {
        "name": "Basic functionality test",
        "queries": SIMPLE_QUERIES[:3],
        "expected_features": ["data filtering", "counting", "basic statistics"]
    },
    
    {
        "name": "Anomaly detection test",
        "queries": ANOMALY_QUERIES[:2],
        "expected_features": ["isolation forest", "statistical outliers", "evidence reporting"]
    },
    
    {
        "name": "Pattern analysis test", 
        "queries": PATTERN_QUERIES[:2],
        "expected_features": ["temporal patterns", "geographic patterns", "trend analysis"]
    },
    
    {
        "name": "Complex analysis test",
        "queries": COMPLEX_QUERIES[:2],
        "expected_features": ["multi-step reasoning", "integrated analysis", "causal hypotheses"]
    }
]

if __name__ == "__main__":
    print("Pipeline Data Agent - Example Queries")
    print("=" * 50)
    
    categories = [
        ("Simple Queries", SIMPLE_QUERIES),
        ("Statistical Queries", STATISTICAL_QUERIES),
        ("Anomaly Detection", ANOMALY_QUERIES),
        ("Pattern Recognition", PATTERN_QUERIES),
        ("Correlation Analysis", CORRELATION_QUERIES),
        ("Clustering", CLUSTERING_QUERIES),
        ("Causal Analysis", CAUSAL_QUERIES),
        ("Complex Multi-step", COMPLEX_QUERIES)
    ]
    
    for category_name, queries in categories:
        print(f"\n{category_name}:")
        print("-" * len(category_name))
        for i, query in enumerate(queries[:5], 1):  # Show first 5 of each
            print(f"{i}. {query}")
    
    print(f"\nTotal example queries: {sum(len(queries) for _, queries in categories)}")
