"""
Main Script for Mood & Activity Prediction Project
Author: Alessio Bolla

Entry point for the complete ML pipeline:
1. Data preparation (prepare.py)
2. Model training (models.py)
3. Model evaluation (evaluate.py)
"""

import argparse

from src.prepare import prepare_data
from src.models import train
from src.evaluate import evaluate


def main():
    """
    Run the complete ML pipeline.
    """
    print("MOOD & ACTIVITY PREDICTION PROJECT")
    print("_" * 65)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mood & Activity Prediction Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Pipeline step to run (default: all)"
    )
    args = parser.parse_args()
    
    # Run pipeline
    if args.step in ["prepare", "all"]:
        print("\nSTEP 1: Data preparation\n")
        prepare_data()
    
    if args.step in ["train", "all"]:
        print("\nSTEP 2: Models training\n")
        train()
    
    if args.step in ["evaluate", "all"]:
        print("\nSTEP 3: Models evaluation\n")
        evaluate()
    
    print("Ppiline completed!!")

if __name__ == "__main__":
    main()