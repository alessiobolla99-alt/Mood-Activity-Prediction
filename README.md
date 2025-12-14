# Mood & Activity Prediction using ML models

[![CI/CD Pipeline](https://github.com/alessiobolla99-alt/Mood-Activity-Prediction/actions/workflows/main.yml/badge.svg)](https://github.com/alessiobolla99-alt/Mood-Activity-Prediction/actions/workflows/main.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A comprehensive machine learning pipeline to predict daily mood states based on behavioral and lifestyle variables using the **Daylio Mood Tracker** dataset. This project implements a complete data science workflow with temporal validation, ensemble learning methods, SHAP interpretability analysis, and automated CI/CD deployment.

## Project Overview

Understanding the relationship between daily behaviors and emotional well-being is crucial for mental health monitoring and intervention. Traditional psychological assessments rely on periodic clinical evaluations, which may miss day-to-day fluctuations in emotional states.

This project develops an end-to-end predictive system that classifies daily mood into five categories (Awful, Bad, Normal, Good, Amazing) using self-reported behavioral data.

**Key Features:**
- **940 mood diary entries** spanning February 2018 to April 2021
- **Three ensemble ML models:** Random Forest, Gradient Boosting, XGBoost
- **30 engineered features** across temporal dynamics and activity patterns
- **Temporal validation strategy** preventing data leakage
- **SHAP analysis** for model interpretability
- **Reproducible pipeline** with automated CI/CD via GitHub Actions

## Key Results

| Model | Accuracy | Balanced Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|-------------------|------------|---------------|
| Random Forest | 0.926 | 0.732 | 0.728 | 0.912 |
| **Gradient Boosting** ⭐ | **0.984** | **0.956** | **0.943** | **0.985** |
| XGBoost | 0.979 | 0.916 | 0.932 | 0.979 |

**Gradient Boosting achieved 95.6% balanced accuracy**, demonstrating strong performance on minority classes (Awful, Bad) — critical for mental health applications where detecting negative mood states is essential.

## Technical Report

📄 **[Complete project report (PDF)](docs/ADP_AlessioBolla.pdf)**

## Quick Start

### Prerequisites

- **Python 3.11**
- **Conda** (recommended) or pip
- **Git**
- **Kaggle API credentials** (for dataset download)

### Kaggle Credentials Setup

To download the dataset automatically, you need Kaggle API credentials:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to **Account Settings** → **API** → **Create New Token**
3. Download `kaggle.json` and place it in:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

**Alternative:** Set environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

> **Note:** If the dataset is already present in `data/raw/`, Kaggle credentials are not required.

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/alessiobolla99-alt/Mood-Activity-Prediction.git
cd Mood-Activity-Prediction
```

#### 2. Create Conda environment
```bash
conda env create -f environment.yml
conda activate mood-prediction
```

**Alternative (pip):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib kaggle shap
```

#### 3. Run the pipeline
```bash
python main.py
```

## Usage

### Run Full Pipeline
```bash
python main.py
```

This executes the complete workflow:
1. Data preprocessing and cleaning
2. Feature engineering (30 variables)
3. Model training with hyperparameter tuning
4. Model evaluation and visualization

### Run Scripts Directly
```bash
python src/prepare.py       # Step 1: Data preparation
python src/models.py        # Step 2: Model training
python src/evaluate.py      # Step 3: Evaluation & plots
```

## Project Structure

```
Mood-Activity-Prediction/
│
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD pipeline configuration
│
├── data/
│   ├── raw/
│   │   └── Daylio_Abid.csv       # Original dataset
│   └── processed/
│       ├── processed_data.csv    # Cleaned & featured dataset
│       └── mood_mapping.json     # Label encoding mapping
│
├── docs/
│   └── project_report.pdf        # Technical report
│
├── models/                       # Trained models (.pkl)
│   ├── randomforest.pkl
│   ├── gradientboosting.pkl
│   ├── xgboost.pkl
│   ├── best_params.pkl
│   ├── learning_curves.pkl
│   └── test_data.pkl
│
├── results/                      # Evaluation outputs
│   ├── evaluation_results.csv
│   ├── confusion_matrices.png
│   ├── learning_curves.png
│   ├── feature_importance.png
│   └── shap_*.png
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── prepare.py                # Data preprocessing & feature engineering
│   ├── models.py                 # Model training & hyperparameter tuning
│   └── evaluate.py               # Evaluation & visualization
│
├── environment.yml               # Conda dependencies
├── main.py                       # Pipeline entry point
├── PROPOSAL.md                   # Original project proposal
└── README.md                     # This file
```

## CI/CD Pipeline

The project implements automated testing via **GitHub Actions**, executing the complete pipeline on every code push:

1. **Environment setup:** Conda environment creation with all dependencies
2. **Data preparation:** Dataset download (via Kaggle API with GitHub Secrets) and feature engineering
3. **Model training:** Training all three models with hyperparameter tuning
4. **Evaluation:** Metrics computation, visualization generation
5. **Results archival:** Automated storage of trained models and evaluation results

### GitHub Secrets Configuration

For CI/CD to work, configure the following secrets in your repository:
- `KAGGLE_USERNAME` — Your Kaggle username
- `KAGGLE_KEY` — Your Kaggle API key

## Reproducibility

- **Random seed:** `random_state=42` used throughout
- **Temporal validation:** Chronological train/test split
- **Environment:** Dependencies specified in `environment.yml`
- **Versioned outputs:** All models and results saved 

## Dataset

**Source:** [Daylio Mood Tracker Dataset](https://www.kaggle.com/datasets/kingabzpro/daylio-mood-tracker/data) on Kaggle

| Attribute | Value |
|-----------|-------|
| Observations | 940 entries |
| Time Period | February 2018 – April 2021 |
| Features | Date, weekday, time, activities, mood label |
| Target Classes | Awful, Bad, Normal, Good, Amazing |
