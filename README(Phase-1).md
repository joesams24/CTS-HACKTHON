Member Risk Stratification — Phase 1: Data Collection and Preprocessing

This document describes the setup, preprocessing workflow, and outputs for Phase 1 of the Member Risk Stratification and Care Management project.
The primary objective of this phase is to collect, clean, and transform synthetic healthcare data (from Synthea and other sources) into structured features suitable for model training and risk prediction.

1. Project Setup
1.1 Directory Structure
backend/
├─ data/
│  ├─ raw/
│  │  ├─ synthea_csv/
│  │  │  ├─ patients.csv
│  │  │  ├─ encounters.csv
│  │  │  ├─ observations.csv
│  │  │  └─ ...
│  └─ processed/
│     ├─ sequences_5.csv
│     ├─ windowed_features_30_60_90.csv
│     └─ static_features.csv
│
├─ scripts/
│  ├─ preprocessing/
│  │  └─ preprocess_data.py
│  ├─ data_collection/
│  │  └─ collect_data.py
│  └─ utils/
│     └─ helpers.py
│
├─ logs/
│  ├─ preprocessing.log
│  └─ data_collection.log
│
├─ venv/                 # Python virtual environment
├─ requirements.txt
└─ app.py or server.js   # Backend entry point (Phase 2)

2. Python Environment Setup
2.1 Create Virtual Environment
python -m venv venv


Activate the Environment

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate

2.2 Install Required Packages
pip install -r requirements.txt


Example requirements.txt:

pandas==2.2.2
numpy==1.26.2
python-dateutil==2.9.4
pytz==2025.7

3. Logging Setup

A centralized logging utility (scripts/utils/helpers.py) is used across all scripts for consistency and traceability.

Preprocessing logs: logs/preprocessing.log

Data collection logs: logs/data_collection.log

Example Usage:

from scripts.utils.helpers import log_message
log_message("Starting preprocessing...", log_file="logs/preprocessing.log")

4. Phase 1: Preprocessing Workflow
4.1 Data Collection Script

Path: scripts/data_collection/collect_data.py

Functions:

Verifies that all required Synthea CSV files are present.

Loads synthetic and real-world member datasets.

Logs all data collection activities for traceability.

4.2 Preprocessing Script

Path: scripts/preprocessing/preprocess_data.py

Key Steps:

Load and clean input CSV files.

Check and ensure temporal consistency of encounters and observations.

Aggregate observations and encounters per patient.

Merge static demographic and socioeconomic features.

Perform feature engineering, including:

Number of encounters per patient.

Average days between encounters.

Generate:

Five most recent lab value sequences (sequences_5.csv).

Windowed features over 30, 60, and 90 days (windowed_features_30_60_90.csv).

Static demographic and clinical features (static_features.csv).

5. Executing the Scripts

Run Data Collection

python -m scripts.data_collection.collect_data


Run Preprocessing

python -m scripts.preprocessing.preprocess_data


Output Files and Descriptions

File Name	Description
sequences_5.csv	Contains the five most recent laboratory value sequences per patient.
windowed_features_30_60_90.csv	Aggregated lab value means for 30, 60, and 90-day periods per patient.
static_features.csv	Contains static demographic and clinical information including age, gender, race, ethnicity, income, number of encounters, and average interval between encounters.
6. Troubleshooting

Missing Columns: The preprocessing script automatically drops unavailable columns.

File Not Found: Ensure all Synthea-generated CSV files are located under data/raw/synthea_csv/.

Encoding Errors: Use UTF-8 encoding when reading or writing files.

Log Review: Check log files in the logs/ directory for detailed error traces.

7. Phase 1 Deliverables

Upon successful completion of Phase 1, the following are available:

Cleaned, structured, and validated healthcare datasets.

Patient-level temporal and static features.

Preprocessed feature files for model development and training.

Centralized logging for reproducibility and auditability.

8. Next Steps (Phase 2)

Phase 2 will focus on:

Developing and training machine learning models (e.g., Random Forest, XGBoost) for member risk stratification.

Implementing risk scoring and visualization dashboards for care management.

Integrating the backend (Flask or Node.js) with a frontend interface (React.js).
