### Pipeline Data Analysis Agent

A sophisticated AI-powered data agent that can analyze pipeline data through natural language queries, detect anomalies, identify patterns, and provide insights with supporting evidence.

#### Features

- 🤖 Natural Language Interface: Ask questions in plain English about your pipeline data
- 📊 Pattern Recognition: Automatically detect trends, correlations, and clustering in the data
- 🚨 Anomaly Detection: Identify outliers and unusual patterns using multiple detection algorithms
- 🔍 Causal Analysis: Propose plausible explanations for observed relationships with evidence
- 📈 Advanced Analytics: Statistical analysis, time series analysis, and predictive modeling
- 💬 Conversational Interface: CLI-based chat interface with rich formatting

### Quick Start

#### Installation

1. Clone this repository:
```bash
git clone https://github.com/rmkenv/projectgamma
cd projectgamma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Note: For Parquet datasets, pandas uses pyarrow (recommended). Ensure pyarrow is installed via requirements.txt.

3. Set up your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### Dataset Setup

This agent supports CSV and Parquet files. You’ll be prompted for the dataset path at runtime.

- Example (Parquet on Colab):
```bash
python main.py
# When prompted, enter: /content/pipeline_data.parquet
```

- Example (CSV locally):
```bash
python main.py
# When prompted, enter: /path/to/your/pipeline_data.csv
```

Tips:
- Use absolute paths for reliability.
- For Parquet, pyarrow must be installed (already included in requirements).

If you need to fetch from Google Drive, download the actual file (not the HTML share page). Using gdown is recommended:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=<FILE_ID>
```

#### Usage

Start the agent:
```bash
python main.py
```

Example queries you can try:

- Simple Retrieval:
  - "How many pipeline records are there in 2024?"
  - "Show me all pipelines in Texas"
  - "What are the unique categories in the dataset?"
- Pattern Recognition:
  - "Find correlations between scheduled quantity and location"
  - "Cluster pipelines by their characteristics"
  - "Show trends in scheduled quantities over time"
- Anomaly Detection:
  - "Find outliers in scheduled quantities"
  - "Detect unusual pipeline configurations"
  - "Identify pipelines with suspicious delivery patterns"
- Causal Analysis:
  - "Why might certain pipelines have higher scheduled quantities?"
  - "What factors could explain the geographic distribution of pipelines?"
  - "Analyze the relationship between pipeline categories and delivery signs"

### Dataset Information

The pipeline dataset may include:
- pipeline_name: Name of the pipeline
- loc_name: Location name
- connecting_pipeline: Connected pipeline information
- connecting_entity: Connected entity details
- rec_del_sign: Receive/delivery sign indicator
- category_short: Short category code
- country_name: Country name
- state_abb: State abbreviation
- county_name: County name
- latitude, longitude: Geographic coordinates
- eff_gas_day: Effective gas day (date)
- scheduled_quantity: Scheduled gas quantity

### Configuration

You can customize behavior in config/config.yaml:

```yaml
agent:
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 4000
  temperature: 0.1

data:
  cache_directory: "./data"
  # If you implement auto-download in your main script, place the URL here:
  # auto_download_url: "your-dataset-url"

analysis:
  anomaly_contamination: 0.1
  clustering_min_samples: 5
  correlation_threshold: 0.5
  max_display_rows: 20
```

Note: The default main.py prompts for a local dataset path. If you re-enable auto-download, ensure the URL points directly to the file (or use gdown for Drive).

### Architecture

```
data_agent/
├── core/
│   ├── agent.py              # Main agent orchestration
│   ├── data_processor.py     # Data loading and preprocessing (CSV/Parquet auto-detect)
│   ├── query_parser.py       # Natural language query parsing
│   └── analysis_engine.py    # Statistical and ML analysis
├── models/
│   ├── query_models.py       # Pydantic models for queries
│   └── response_models.py    # Pydantic models for responses
└── utils/
    ├── anthropic_client.py   # Anthropic API client
    ├── data_utils.py         # Data utility functions
    └── logger.py             # Logging configuration
```

### Example Interactions

```
🤖 Pipeline Data Agent Ready! Ask me anything about your pipeline data.

You: Find anomalies in scheduled quantities

Agent: I'll analyze the scheduled quantities to identify anomalies using multiple detection methods.

📊 Anomaly Detection Results

Method: Isolation Forest
- Detected 45 anomalies (4.2% of data)
- Anomaly scores range: -0.52 to 0.71
- Top anomalies: Pipeline_XYZ (score: 0.71), Pipeline_ABC (score: 0.68)

Method: Statistical (Z-score > 3)
- Detected 23 statistical outliers
- Mean scheduled quantity: 1,234,567
- Standard deviation: 456,789

Evidence:
- Columns analyzed: ['scheduled_quantity']
- Filter applied: None
- Algorithm parameters: contamination=0.1, random_state=42

Key Findings:
- Pipelines in Texas show 3x higher variance in scheduled quantities
- Friday deliveries have 23% more anomalies than other weekdays
- Category 'LNG' shows the highest anomaly rate (8.7%)

Would you like me to investigate any specific anomalies further?
```

### Advanced Features

- Multi-step Analysis:
  - The agent chains steps (detect anomalies, cluster geography, test spatial correlation) for complex questions.

- Evidence-Based Responses:
  - Methodology, data selection, statistical evidence, and limitations are included in responses.

- Conversation Memory:
  - The agent remembers recent context within a session to refine follow-up queries.

### Limitations

- Data Quality: Results depend on data quality; missing/inconsistent data are surfaced.
- Causality: Causal statements are hypotheses, not definitive proof.
- Model Limitations: Detection accuracy varies across datasets and outlier types.
- API Costs: Complex analyses may require multiple API calls.

### Development

#### Running Tests
```bash
pytest tests/
```

#### Code Quality
```bash
# Format code
black data_agent/

# Type checking
mypy data_agent/

# Linting
flake8 data_agent/
```

#### Adding New Analysis Methods

1. Add your method to analysis_engine.py
2. Update the query parser to recognize relevant keywords
3. Add appropriate response models
4. Write tests

### Troubleshooting

- "No API key found"
  - Ensure ANTHROPIC_API_KEY is set in your environment

- "Dataset not found"
  - Check the file path when prompted; use absolute paths if possible

- "Failed to load data" or codec errors
  - Ensure the file format matches its extension (CSV vs Parquet)
  - For Parquet, ensure pyarrow is installed

- "Loaded 0 rows" when using a Google Drive URL
  - You likely downloaded an HTML interstitial page
  - Use gdown or obtain a direct download link

- Performance Issues
  - For very large datasets, consider sampling or column selection

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code quality checks pass
5. Submit a pull request

### License

MIT License - see LICENSE file for details.
