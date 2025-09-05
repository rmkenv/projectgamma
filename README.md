### Pipeline Data Analysis Agent
A sophisticated AI-powered data agent that analyzes pipeline data through natural language queries, detects anomalies, identifies patterns, and provides insights with supporting evidence. Now includes advanced Pipeline Operations analytics (capacity, utilization, bottlenecks) and optional AIS vessel-traffic integration.

🚀 Features
🤖 Natural Language Interface: Ask questions in plain English about your pipeline data

📊 Pattern Recognition: Detect trends, correlations, and clustering

🚨 Anomaly Detection: Multiple methods (Z-score, IQR, Isolation Forest)

🔍 Causal Analysis: Plausible explanations with supporting evidence

📈 Advanced Analytics: Statistical summaries, time series patterns, clustering

🛠️ Pipeline Operations: Capacity inference, utilization, bottlenecks, flow-direction anomalies

🚢 AIS Integration (optional): Overlay nearby tanker/vessel activity near terminals

💬 Conversational CLI: Rich, chat-like terminal interface

⚡ Quick Start
Installation
Clone this repository:

git clone https://github.com/rmkenv/projectgamma
cd projectgamma
Install dependencies:

pip install -r requirements.txt
Note: For Parquet datasets, pandas uses pyarrow (recommended). Ensure it’s installed via requirements.txt.

Optional (for pipeline operations & geospatial features):

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

For Parquet, ensure pyarrow is installed.

If fetching from Google Drive:

pip install gdown
gdown https://drive.google.com/uc?id=<FILE_ID>
🎯 Usage
Start the agent:

python main.py
Example queries you can try:

🔹 Simple Retrieval
"How many pipeline records are there in 2024?"

"Show me all pipelines in Texas"

"What are the unique categories in the dataset?"

🔹 Pattern Recognition
"Find correlations between scheduled quantity and location"

"Cluster pipelines by their characteristics"

"Show trends in scheduled quantities over time"

🔹 Anomaly Detection
"Find outliers in scheduled quantities"

"Detect unusual pipeline configurations"

"Identify pipelines with suspicious delivery patterns"

🔹 Causal Analysis
"Why might certain pipelines have higher scheduled quantities?"

"Analyze the relationship between pipeline categories and delivery signs"

🔹 Pipeline Operations (NEW)
"Detect pipeline bottlenecks and capacity issues in TX"

"Find pipelines with utilization > 85% in 2024"

"Analyze flow direction anomalies by region"

🔹 Pipeline Ops with AIS (NEW)
"Find pipeline bottlenecks in Texas and show nearby tanker vessels from AIS in the last 30 days"

"Which terminals had recurring bottlenecks last month, and what AIS tanker activity was nearby?"

💡 Tip: Include words like "tanker", "vessel", "AIS", or "shipping" to auto-enable AIS.

📂 Dataset Information
The pipeline dataset may include:

pipeline_name: Name of the pipeline

loc_name: Location name

connecting_pipeline: Connected pipeline information

rec_del_sign: Receive/delivery indicator

scheduled_quantity: Scheduled gas quantity

latitude, longitude: Geographic coordinates

eff_gas_day: Effective gas day (date)

(and more...)

⚙️ Configuration
config/config.yaml:

agent:
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 4000
  temperature: 0.1

analysis:
  anomaly_contamination: 0.1
  clustering_min_samples: 5
  correlation_threshold: 0.5

pipeline_analytics:
  utilization_threshold: 0.85
  bottleneck_window_days: 30
  bottleneck_min_days: 10
  max_tanker_distance_km: 20
  enable_ais_integration: false
🏗️ Architecture
data_agent/
├── core/
│   ├── agent.py
│   ├── data_processor.py
│   ├── query_parser.py
│   └── analysis_engine.py
├── analysis/
│   └── pipeline_analytics.py
├── models/
│   ├── query_models.py
│   └── response_models.py
└── utils/
    ├── anthropic_client.py
    ├── data_utils.py
    └── logger.py
📊 Example Interaction
You: Find anomalies in scheduled quantities
Agent:

Method: Isolation Forest → 45 anomalies (4.2%)

Method: Z-score (>3) → 23 anomalies

Key Findings: TX pipelines show 3x higher variance, LNG has 8.7% anomaly rate

🔧 Development
Running Tests
pytest tests/
Code Quality
black data_agent/
mypy data_agent/
flake8 data_agent/
Adding New Methods
Add your method to analysis_engine.py

Update query_parser.py

Add response models + tests

🤝 Contributing
Fork the repo

Create a feature branch

Add tests + ensure code quality

Submit a pull request

📜 License
MIT License — see LICENSE file for details.
