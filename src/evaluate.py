"""
Model Evaluation Script for Mood & Activity Prediction Project
Author: Alessio Bolla, AI improved
"""

import json
import os
import sys
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings('ignore')


def load_models(models_dir="models"):
    """
    Load all trained models from disk.

    Args:
        models_dir: Directory containing saved models

    Returns:
        dict: Dictionary of model name -> trained model
    """
    print("\nLoading models:")

    model_files = {
        'RandomForest': 'randomforest.pkl',
        'GradientBoosting': 'gradientboosting.pkl',
        'XGBoost': 'xgboost.pkl'
    }

    models = {}
    for name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"  Loaded {name}")
        else:
            print(f"  Warning: {model_path} not found")

    return models


def load_test_data(models_dir="models"):
    """
    Load test data from disk.

    Args:
        models_dir: Directory containing test data

    Returns:
        tuple: x_test, y_test
    """
    print("\nLoading test data:")

    test_path = os.path.join(models_dir, "test_data.pkl")
    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}")
        print("Please run models.py first")
        sys.exit(1)

    test_data = joblib.load(test_path)
    x_test = test_data['X_test']
    y_test = test_data['y_test']

    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    return x_test, y_test


def load_learning_curves(models_dir="models"):
    """
    Load learning curves data from disk.

    Args:
        models_dir: Directory containing learning curves data

    Returns:
        dict: Learning curves data
    """
    print("\nLoading learning curves:")

    lc_path = os.path.join(models_dir, "learning_curves.pkl")
    if not os.path.exists(lc_path):
        print(f"  Warning: Learning curves not found at {lc_path}")
        return None

    learning_curves = joblib.load(lc_path)
    print(f"  Loaded learning curves for {len(learning_curves)} models")

    return learning_curves


def load_mood_mapping(data_dir="data/processed"):
    """
    Load mood mapping for label names.

    Args:
        data_dir: Directory containing mood mapping

    Returns:
        dict: Mood decoding mapping
    """
    mapping_path = os.path.join(data_dir, "mood_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        return mapping.get('decoding', None)
    return None


def evaluate_models(models, x_test, y_test):
    """
    Evaluate all models and compute metrics.

    Args:
        models: Dictionary of trained models
        x_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary of model name -> metrics and predictions
    """
    print("\nEvaluating models:")

    results = {}

    for name, model in models.items():
        print(f"\n  {name}:")

        # Predict
        y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_pred': y_pred,
            'y_test': y_test
        }

        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Balanced Accuracy: {balanced_acc:.3f}")
        print(f"    F1 (macro): {f1_macro:.3f}")
        print(f"    F1 (weighted): {f1_weighted:.3f}")

    return results


def print_classification_reports(results, mood_mapping=None):
    """
    Print detailed classification reports for all models.

    Args:
        results: Dictionary of model results
        mood_mapping: Optional mapping of class indices to mood names
    """
    print("\nCLASSIFICATION REPORTS:\n")
    # Get target names if mapping available
    if mood_mapping:
        target_names = [
            mood_mapping.get(str(i), str(i))
            for i in sorted(mood_mapping.keys(), key=int)
        ]
    else:
        target_names = None

    for name, metrics in results.items():
        print(f"\n{name}:")
        print("-" * 40)
        print(classification_report(
            metrics['y_test'],
            metrics['y_pred'],
            target_names=target_names,
            zero_division=0
        ))


def print_summary(results):
    """
    Print summary table of model comparison.

    Args:
        results: Dictionary of model results
    """
    print("Models comparison summary")

    header = (f"{'Model':<20} {'Accuracy':>10} {'Bal. Acc.':>10} "
              f"{'F1 (macro)':>12} {'F1 (weighted)':>14}")
    print(f"\n{header}")
    print("-" * 70)

    best_model = None
    best_score = 0

    for name, metrics in results.items():
        row = (f"{name:<20} {metrics['accuracy']:>10.3f} "
               f"{metrics['balanced_accuracy']:>10.3f} "
               f"{metrics['f1_macro']:>12.3f} {metrics['f1_weighted']:>14.3f}")
        print(row)

        # Track best model by balanced accuracy
        if metrics['balanced_accuracy'] > best_score:
            best_score = metrics['balanced_accuracy']
            best_model = name

    print("-" * 70)
    print(f"\nBest model: {best_model} (Balanced Accuracy: {best_score:.3f})")


def plot_confusion_matrices(results, mood_mapping=None, results_dir="results"):
    """
    Plot confusion matrices for all models.

    Args:
        results: Dictionary of model results
        mood_mapping: Optional mapping of class indices to mood names
        results_dir: Directory to save plots
    """
    print("\nPlotting confusion matrices:")

    os.makedirs(results_dir, exist_ok=True)

    # Get class labels
    if mood_mapping:
        labels = [
            mood_mapping.get(str(i), str(i))
            for i in sorted(mood_mapping.keys(), key=int)
        ]
    else:
        labels = None

    n_models = len(results)
    _, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {plot_path}")


def plot_learning_curves(learning_curves, results_dir="results"):
    """
    Plot learning curves for all models.

    Args:
        learning_curves: Dictionary of learning curves data
        results_dir: Directory to save plots
    """
    print("\nPlotting learning curves:")

    if learning_curves is None:
        print("  No learning curves data available")
        return

    os.makedirs(results_dir, exist_ok=True)

    n_models = len(learning_curves)
    _, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, learning_curves.items()):
        train_sizes = data['train_sizes']
        train_mean = data['train_scores_mean']
        train_std = data['train_scores_std']
        val_mean = data['val_scores_mean']
        val_std = data['val_scores_std']

        # Plot training score
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color='blue'
        )

        # Plot validation score
        ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color='orange'
        )

        ax.set_title(f'{name}')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])

    plt.suptitle('Learning Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {plot_path}")


def plot_feature_importance(models, feature_names, results_dir="results", top_n=15):
    """
    Plot feature importance for tree-based models.

    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        results_dir: Directory to save plots
        top_n: Number of top features to display
    """
    print("\nPlotting feature importance:")

    os.makedirs(results_dir, exist_ok=True)

    n_models = len(models)
    _, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print(f"  {name}: No feature_importances_ attribute")
            continue

        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)

        # Plot
        ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        ax.set_title(f'{name}')
        ax.set_xlabel('Importance')

    plt.suptitle(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {plot_path}")


def compute_shap_values(models, x_test, results_dir="results", max_samples=100):
    """
    Compute and plot SHAP values for model interpretability.

    Args:
        models: Dictionary of trained models
        x_test: Test features
        results_dir: Directory to save plots
        max_samples: Maximum number of samples for SHAP computation
    """
    print("\nComputing SHAP values:")

    try:
        import shap  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("  Warning: SHAP not installed, skipping SHAP analysis")
        print("  Install with: pip install shap")
        return

    os.makedirs(results_dir, exist_ok=True)

    # Use subset of data for faster computation
    if len(x_test) > max_samples:
        x_shap = x_test.sample(n=max_samples, random_state=42)
    else:
        x_shap = x_test.copy()

    for name, model in models.items():
        print(f"\n  {name}:")

        try:
            # Select explainer based on model type
            if name == 'GradientBoosting':
                explainer = shap.KernelExplainer(
                    model.predict_proba,
                    shap.sample(x_shap, 50)
                )
                shap_values = explainer.shap_values(x_shap.head(50))
            elif name == 'XGBoost':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_shap.values)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_shap)

            # Summary plot
            plt.figure(figsize=(10, 8))

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_mean = np.abs(np.array(shap_values)).mean(axis=0)
                shap.summary_plot(
                    shap_values_mean,
                    x_shap.head(50) if name == 'GradientBoosting' else x_shap,
                    plot_type="bar",
                    show=False,
                    max_display=15
                )
            else:
                shap.summary_plot(
                    shap_values,
                    x_shap,
                    plot_type="bar",
                    show=False,
                    max_display=15
                )

            plt.title(f'SHAP Feature Importance - {name}')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(results_dir, f"shap_{name.lower()}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    Saved to {plot_path}")

        except (ValueError, RuntimeError) as e:
            print(f"    Error computing SHAP: {e}")


def save_results(results, results_dir="results"):
    """
    Save evaluation results to disk.

    Args:
        results: Dictionary of model results
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics (without predictions) as CSV
    results_summary = {
        name: {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_test']}
        for name, metrics in results.items()
    }

    results_df = pd.DataFrame(results_summary).T
    results_df.index.name = 'model'

    results_path = os.path.join(results_dir, "evaluation_results.csv")
    results_df.to_csv(results_path)
    print(f"\nResults saved to {results_path}")


def evaluate(models_dir="models", results_dir="results"):
    """
    Main function to evaluate all models.
    Can be called from main.py or run standalone.

    Args:
        models_dir: Directory containing trained models
        results_dir: Directory to save evaluation results

    Returns:
        dict: Evaluation results
    """
    print("MODEL evaluation:")

    # Load models and data
    models = load_models(models_dir)
    x_test, y_test = load_test_data(models_dir)
    learning_curves = load_learning_curves(models_dir)
    mood_mapping = load_mood_mapping()

    # Evaluate models
    results = evaluate_models(models, x_test, y_test)

    # Print reports
    print_classification_reports(results, mood_mapping)
    print_summary(results)

    # Generate plots
    plot_confusion_matrices(results, mood_mapping, results_dir)
    plot_learning_curves(learning_curves, results_dir)
    plot_feature_importance(models, x_test.columns.tolist(), results_dir)
    compute_shap_values(models, x_test, results_dir)

    # Save results
    save_results(results, results_dir)

    print("\nModel evaluation complete")

    return results


if __name__ == "__main__":
    evaluate()