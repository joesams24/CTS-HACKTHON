# Member Risk Stratification — Phase 1: Data Collection & Preprocessing

```
>## This document outlines the setup, preprocessing workflow, and outputs for Phase 1 of the Member Risk Stratification project. 

> The goal of this phase is to collect, clean, and transform synthetic healthcare data (from Synthea and other sources) into structured features ready for model training and risk prediction.
```

---

## 1. Project Setup

### 1.1 Directory Structure
```
backend/
├─ data/
│ ├─ raw/
│ │ ├─ synthea_csv/
│ │ │ ├─ patients.csv
│ │ │ ├─ encounters.csv
│ │ │ ├─ observations.csv
│ │ │ └─ ...
│ └─ processed/
│ ├─ sequences_5.csv
│ ├─ windowed_features_30_60_90.csv
│ └─ static_features.csv
├─ scripts/
│ ├─ preprocessing/
│ │ └─ preprocess_data.py
│ ├─ data_collection/
│ │ └─ collect_data.py
│ └─ utils/
│ └─ helpers.py
├─ logs/
│ ├─ preprocessing.log
│ └─ data_collection.log
├─ venv/
├─ requirements.txt
└─ app.py or server.js

```

---

## 2. Python Environment Setup

### 2.1 Create Virtual Environment

```
python -m venv venv
Activate Environment:

Windows: venv\Scripts\activate

Linux/Mac: source venv/bin/activate
```
### 2.2 Install Required Packages
```
pip install -r requirements.txt
Example requirements.txt:

ini
Copy code
pandas==2.2.2
numpy==1.26.2
python-dateutil==2.9.4
pytz==2025.7
```
### 3. Logging Setup
```
> A centralized logging utility (scripts/utils/helpers.py) is used across all scripts.

> Preprocessing logs: logs/preprocessing.log

> Data collection logs: logs/data_collection.log

Example Usage:

from scripts.utils.helpers import log_message
log_message("Starting preprocessing...", log_file="logs/preprocessing.log")
```
### 4. Phase 1: Preprocessing Pipeline

### 4.1 Data Collection Script
```
Path: scripts/data_collection/collect_data.py

Verifies all required Synthea CSVs exist.

Loads synthetic and real-world member data.

Logs activities to logs/data_collection.log.

```
### 4.2 Preprocessing Script
```
Path: scripts/preprocessing/preprocess_data.py

Steps Performed:

Load and clean CSV files.

Ensure temporal consistency across encounters.

Aggregate observations per patient.

Merge static demographic features.

Compute derived features:

Number of encounters

Average days between encounters

Generate 5 most recent lab sequences.

Create windowed features (30, 60, 90-day averages).

Save processed outputs to data/processed/.

```
### 5. Running the Scripts
```
Data Collection:

python -m scripts.data_collection.collect_data
Preprocessing:

bash
Copy code
python -m scripts.preprocessing.preprocess_data
Output Files:

File	Description
sequences_5.csv	5 most recent LAB_VALUE sequences per patient
windowed_features_30_60_90.csv	Mean LAB_VALUE for 30/60/90-day periods
static_features.csv	Static features: AGE, GENDER, RACE, ETHNICITY, INCOME, NUM_ENCOUNTERS, AVG_DAYS_BETWEEN_ENC

```
### 6. Troubleshooting
```
Missing columns: Automatically dropped by the preprocessing script.

File not found: Ensure all Synthea CSVs are in data/raw/synthea_csv/.

Encoding errors: Use UTF-8 encoding during file read operations.

```
### 7. Phase 1 Deliverables
```
After successful preprocessing, the system provides:

Cleaned and structured healthcare features.

Temporal and sequence-based patient records.

Preprocessed datasets for model training (Phase 2).

Centralized logs for full traceability.
