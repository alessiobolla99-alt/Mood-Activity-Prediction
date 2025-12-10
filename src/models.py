"""
Model Training Script for Mood & Activity Prediction Project
Author: Alessio Bolla, AI Improved
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    learning_curve,
    train_test_split
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TUNE_HYPERPARAMETERS = True  # Set to False to skip hyperparameter tuning


def load_processed_data(data_path="data/processed/processed_data.csv"):
    """
    Load the processed dataset.

    Args:
        data_path: Path to processed CSV file

    Returns:
        pd.DataFrame: Processed dataset
    """
    print("\nLoading processed data:")

    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Please run prepare.py first")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def temporal_train_test_split(df, test_size=0.2):
    """
    Split data temporally: train on past, test on future.

    Args:
        df: Dataset with date_parsed column
        test_size: Proportion of data for testing

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    print("\nSplitting data temporally:")

    if 'date_parsed' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date_parsed'])
        df = df.sort_values('date_parsed').reset_index(drop=True)

        # 80% train, 20% test based on time
        split_idx = int(len(df) * (1 - test_size))
        split_date = df.iloc[split_idx]['date_parsed']

        print(f"  Split date: {split_date.date()}")
        print(f"  Train: {split_idx:,} samples")
        print(f"  Test: {len(df) - split_idx:,} samples")

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Remove date column from features
        feature_cols = [
            col for col in df.columns if col not in ['target', 'date_parsed']
        ]

        x_train = train_df[feature_cols]
        x_test = test_df[feature_cols]
        y_train = train_df['target']
        y_test = test_df['target']

    else:
        print("  Warning: No date_parsed column, using random split")

        feature_cols = [col for col in df.columns if col != 'target']
        features = df[feature_cols]
        target = df['target']

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=RANDOM_STATE
        )

    print(f"  Features: {len(feature_cols)}")

    return x_train, x_test, y_train, y_test


def get_models():
    """
    Define models with conservative parameters to reduce overfitting.

    Returns:
        dict: Dictionary of model name -> model instance
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }

    return models


def get_param_grids():
    """
    Define hyperparameter search spaces.

    Returns:
        dict: Dictionary of model name -> parameter grid
    """
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_leaf': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 2, 5]
        }
    }

    return param_grids


def tune_hyperparameters(x_train, y_train, models_dir="models", n_iter=20):
    """
    Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Args:
        x_train: Training features
        y_train: Training labels
        models_dir: Directory to save best parameters
        n_iter: Number of parameter combinations to try

    Returns:
        dict: Dictionary of model name -> best model
    """
    print("\nTuning hyperparameters:")

    os.makedirs(models_dir, exist_ok=True)

    # TimeSeriesSplit preserves temporal order
    tscv = TimeSeriesSplit(n_splits=5)

    base_models = {
        'RandomForest': RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'
        )
    }

    param_grids = get_param_grids()
    tuned_models = {}
    best_params = {}

    for name, model in base_models.items():
        print(f"\n  {name}:")

        # RandomizedSearchCV is faster than GridSearchCV
        search = RandomizedSearchCV(
            model,
            param_grids[name],
            n_iter=n_iter,
            cv=tscv,
            scoring='balanced_accuracy',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        search.fit(x_train, y_train)

        tuned_models[name] = search.best_estimator_
        best_params[name] = search.best_params_

        print(f"    Best score: {search.best_score_:.3f}")
        print(f"    Best params: {search.best_params_}")

    params_path = os.path.join(models_dir, "best_params.pkl")
    joblib.dump(best_params, params_path)
    print(f"\nBest parameters saved to {params_path}")

    return tuned_models


def compute_learning_curves(x_train, y_train, models_dir="models"):
    """
    Compute learning curves for all models.

    Args:
        x_train: Training features
        y_train: Training labels
        models_dir: Directory to save learning curves data

    Returns:
        dict: Dictionary of model name -> learning curve data
    """
    print("\nComputing learning curves:")

    os.makedirs(models_dir, exist_ok=True)
    models = get_models()
    learning_curves_data = {}

    # 10 points from 10% to 100% of training data
    train_sizes = np.linspace(0.1, 1.0, 10)

    for name, model in models.items():
        print(f"\n  {name}:")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            x_train,
            y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        learning_curves_data[name] = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }

        final_train = train_scores.mean(axis=1)[-1]
        final_val = val_scores.mean(axis=1)[-1]
        print(f"    Final train score: {final_train:.3f}")
        print(f"    Final val score: {final_val:.3f}")

    lc_path = os.path.join(models_dir, "learning_curves.pkl")
    joblib.dump(learning_curves_data, lc_path)
    print(f"\nLearning curves saved to {lc_path}")

    return learning_curves_data


def train_models(x_train, y_train, x_test, y_test,
                 tuned_models=None, models_dir="models"):
    """
    Train all models and save them.

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        tuned_models: Optional dictionary of pre-tuned models
        models_dir: Directory to save models

    Returns:
        dict: Dictionary of model name -> trained model
    """
    print("\nTraining models:")

    os.makedirs(models_dir, exist_ok=True)

    # Use tuned models if available, otherwise default
    if tuned_models is not None:
        models = tuned_models
        print("  Using tuned hyperparameters")
    else:
        models = get_models()
        print("  Using default hyperparameters")

    trained_models = {}

    for name, model in models.items():
        print(f"\n  {name}:")

        # Skip fitting if already fitted during tuning
        if not hasattr(model, 'n_features_in_'):
            model.fit(x_train, y_train)

        trained_models[name] = model

        model_path = os.path.join(models_dir, f"{name.lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"    Saved to {model_path}")

    # Save test data for evaluate.py
    test_data = {'X_test': x_test, 'y_test': y_test}
    test_path = os.path.join(models_dir, "test_data.pkl")
    joblib.dump(test_data, test_path)
    print(f"\nTest data saved to {test_path}")

    return trained_models


def train(data_path="data/processed/processed_data.csv", models_dir="models"):
    """
    Main function to train all models.

    Args:
        data_path: Path to processed data
        models_dir: Directory to save models

    Returns:
        dict: Dictionary of trained models
    """
    print("MODEL training:")

    df = load_processed_data(data_path)
    x_train, x_test, y_train, y_test = temporal_train_test_split(df)

    # Hyperparameter tuning (optional)
    tuned_models = None
    if TUNE_HYPERPARAMETERS:
        tuned_models = tune_hyperparameters(x_train, y_train, models_dir)

    compute_learning_curves(x_train, y_train, models_dir)
    trained_models = train_models(
        x_train, y_train, x_test, y_test, tuned_models, models_dir
    )

    print("\nModel training complete")

    return trained_models


if __name__ == "__main__":
    train()