Pipeline Data Analysis Agent
A sophisticated AI-powered data agent that analyzes pipeline data through natural language queries, detects anomalies, identifies patterns, and provides insights with supporting evidence. Now includes advanced Pipeline Operations analytics (capacity, utilization, bottlenecks) and optional AIS vessel-traffic integration.

Features
🤖 Natural Language Interface: Ask questions in plain English about your pipeline data

📊 Pattern Recognition: Detect trends, correlations, and clustering

🚨 Anomaly Detection: Multiple methods (Z-score, IQR, Isolation Forest)

🔍 Causal Analysis: Plausible explanations with supporting evidence

📈 Advanced Analytics: Statistical summaries, time series patterns, clustering

🛠️ Pipeline Operations: Capacity inference, utilization, bottlenecks, flow-direction anomalies

🚢 AIS Integration (optional): Overlay nearby tanker/vessel activity near terminals

💬 Conversational CLI: Rich, chat-like terminal interface

Quick Start
Installation
Clone this repository:

git clone https://github.com/rmkenv/projectgamma
cd projectgamma
Install dependencies:

pip install -r requirements.txt
Note: For Parquet datasets, pandas uses pyarrow (recommended). Ensure pyarrow is installed via requirements.txt.

Optional (for pipeline operations & geospatial features):

# Geospatial stack (recommended for spatial joins/buffer operations)
pip install geopandas shapely rtree
Set up your Anthropic API key:

export ANTHROPIC_API_KEY="your-api-key-here"
Dataset Setup
This agent supports CSV and Parquet files. You’ll be prompted for the dataset path at runtime.

Example (Parquet on Colab):

python main.py
# When prompted, enter: /content/pipeline_data.parquet
Example (CSV locally):

python main.py
# When prompted, enter: /path/to/your/pipeline_data.csv
Tips:

Use absolute paths for reliability.

For Parquet, pyarrow must be installed (already included in requirements).

If fetching from Google Drive, download the actual file (not the HTML share page). Using gdown is recommended:

pip install gdown
gdown https://drive.google.com/uc?id=<FILE_ID>
Usage
Start the agent:

python main.py
Example queries you can try:

Simple Retrieval:

"How many pipeline records are there in 2024?"

"Show me all pipelines in Texas"

"What are the unique categories in the dataset?"

Pattern Recognition:

"Find correlations between scheduled quantity and location"

"Cluster pipelines by their characteristics"

"Show trends in scheduled quantities over time"

Anomaly Detection:

"Find outliers in scheduled quantities"

"Detect unusual pipeline configurations"

"Identify pipelines with suspicious delivery patterns"

Causal Analysis:

"Why might certain pipelines have higher scheduled quantities?"

"What factors could explain the geographic distribution of pipelines?"

"Analyze the relationship between pipeline categories and delivery signs"

Pipeline Operations (NEW):

"Detect pipeline bottlenecks and capacity issues in TX"

"Find pipelines with utilization > 85% in 2024"

"Analyze flow direction anomalies by region"

"Show capacity constraints near terminals"

Pipeline Ops with AIS (NEW):

"Find pipeline bottlenecks in Texas and show nearby tanker vessels from AIS in the last 30 days"

"Analyze capacity utilization by region and overlay AIS tanker traffic near terminals"

"Detect flow direction anomalies and list any vessels within 20 km of affected nodes (use AIS)"

"Which terminals had recurring bottlenecks last month, and what AIS tanker activity was nearby?"

Tip: Include words like "tanker", "vessel", "AIS", or "shipping" to auto-enable AIS in the query.

Dataset Information
The pipeline dataset may include:

pipeline_name: Name of the pipeline

loc_name: Location name

connecting_pipeline: Connected pipeline information

connecting_entity: Connected entity details

rec_del_sign: Receive/delivery sign indicator

category_short: Short category code

country_name: Country name

state_abb: State abbreviation

county_name: County name

latitude, longitude: Geographic coordinates (may be missing in some datasets)

eff_gas_day: Effective gas day (date)

scheduled_quantity: Scheduled gas quantity

Configuration
You can customize behavior in config/config.yaml:

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

pipeline_analytics:
  utilization_threshold: 0.85
  bottleneck_window_days: 30
  bottleneck_min_days: 10
  max_tanker_distance_km: 20
  ais_stream_duration: 30
  enable_ais_integration: false
Notes:

The default main.py prompts for a local dataset path. If you re-enable auto-download, ensure the URL points directly to the file (or use gdown for Drive).

Set pipeline_analytics.enable_ais_integration: true to allow AIS lookups. Queries mentioning "tanker", "vessel", "AIS", or "shipping" will request AIS data.

Architecture
data_agent/
├── core/
│   ├── agent.py               # Main agent orchestration
│   ├── data_processor.py      # Data loading & preprocessing (CSV/Parquet)
│   ├── query_parser.py        # NL query parsing w/ pipeline ops + AIS intent
│   └── analysis_engine.py     # Statistical/ML analysis + pipeline operations
├── analysis/
│   └── pipeline_analytics.py  # Capacity, utilization, bottlenecks, flow anomalies, AIS
├── models/
│   ├── query_models.py        # Pydantic models for queries
│   └── response_models.py     # Pydantic models for responses
└── utils/
    ├── anthropic_client.py    # Anthropic API client
    ├── data_utils.py          # Data utility functions
    └── logger.py              # Logging configuration
Example Interactions
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
Advanced Features
Multi-step Analysis:

The agent chains steps (e.g., detect anomalies → cluster geography → test spatial correlation) for complex questions.

Evidence-Based Responses:

Methodology, data selection, statistical evidence, and limitations are included in responses.

Conversation Memory:

The agent remembers recent context within a session to refine follow-up queries.

Pipeline Operations Suite:

Capacity inference and utilization scoring

Bottleneck detection over configurable rolling windows

Flow-direction anomaly checks

Optional AIS overlay to enrich context around terminals and high-traffic regions

Prerequisites & Performance
Large Datasets: For 20M+ rows, enable preprocessing optimizations (categoricals, selective columns, chunking if needed). Memory use is surfaced in summaries.

Geospatial: geopandas, shapely, and rtree (optional) improve spatial operations. Without them, AIS/spatial features gracefully degrade to simpler proximity checks.

AIS: Set enable_ais_integration: true and mention "tanker", "vessel", "AIS", or "shipping" in your query.

Limitations
Data Quality: Results depend on data quality; missing/inconsistent data are surfaced.

Causality: Causal statements are hypotheses, not proofs.

Model Limitations: Detection accuracy varies across datasets and outlier types.

API Costs: Complex analyses may require multiple API calls.

Real-time Streams: Continuous AIS can be resource-intensive; set sensible durations.

Troubleshooting
"No API key found"

Ensure ANTHROPIC_API_KEY is set in your environment

"Dataset not found"

Check the file path when prompted; use absolute paths if possible

"Failed to load data" or codec errors

Ensure the file format matches its extension (CSV vs Parquet)

For Parquet, ensure pyarrow is installed

"Loaded 0 rows" when using a Google Drive URL

You likely downloaded an HTML interstitial page

Use gdown or obtain a direct download link

Performance Issues

For very large datasets, consider sampling or selecting columns in your query

Development
Running Tests
pytest tests/
Code Quality
# Format code
black data_agent/

# Type checking
mypy data_agent/

# Linting
flake8 data_agent/
Adding New Analysis Methods
Add your method to analysis_engine.py (or analysis/pipeline_analytics.py for pipeline ops)

Update query_parser.py to recognize relevant keywords/intent

Add appropriate response models

Write tests

Contributing
Fork the repository

Create a feature branch

Add tests for new functionality

Ensure code quality checks pass

Submit a pull request

License
MIT License - see LICENSE file for details.
