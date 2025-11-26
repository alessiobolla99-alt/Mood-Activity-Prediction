"""
Data Preparation Script for Mood & Activity Prediction Project
Author: Alessio Bolla
"""

import os
import sys
import pandas as pd

def setup_kaggle_credentials():
    """
    Check if Kaggle credentials are configured.
    
    Returns:
        bool: True if credentials exist, False otherwise
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("Kaggle credentials found")
        return True
    
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        print("Kaggle credentials found in environment")
        return True
    
    print("Error: Kaggle credentials not found")
    print("Please set up Kaggle API credentials first")
    return False

def download_dataset(data_dir="data/raw"):
    """
    Download the Daylio Mood Tracker dataset from Kaggle.
    
    Args:
        data_dir: Directory to save the raw dataset
        
    Returns:
        str: Path to the downloaded CSV file
    """
    try:
        import kaggle
    except ImportError:
        print("Error: Kaggle package not installed")
        sys.exit(1)
    
    os.makedirs(data_dir, exist_ok=True)
    print("\n[1/6] Downloading dataset from Kaggle...")
    
    try:
        kaggle.api.dataset_download_files(
            'kingabzpro/daylio-mood-tracker',
            path=data_dir,
            unzip=True,
            quiet=False
        )
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("Error: No CSV file found")
            sys.exit(1)
        
        csv_path = os.path.join(data_dir, csv_files[0])
        print(f"Dataset downloaded: {csv_files[0]}")
        return csv_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

def load_raw_data(csv_path):
    """
    Load the raw dataset from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Raw dataset
    """
    print("\nLoading raw data.")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

def clean_data(df):
    """
    Clean the dataset: handle missing values, duplicates, and data types.
    
    Args:
        df: Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\nCleaning data:")
    
    initial_rows = len(df)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # Convert full_date column to datetime
    if 'full_date' in df.columns:
        print(f"  Converting dates to datetime format")
        
        date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%Y/%m/%d']
        
        df['full_date_dt'] = None
        for fmt in date_formats:
            if df['full_date_dt'].isna().all():
                df['full_date_dt'] = pd.to_datetime(df['full_date'], format=fmt, errors='coerce')
                if not df['full_date_dt'].isna().all():
                    print(f"    Dates parsed using format: {fmt}")
                    break
        
        if df['full_date_dt'].isna().all():
            print(f"    Using automatic date parser")
            df['full_date_dt'] = pd.to_datetime(df['full_date'], errors='coerce', dayfirst=True)
        
        # Remove rows with invalid dates
        invalid_dates = df['full_date_dt'].isna().sum()
        if invalid_dates > 0:
            print(f"  Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['full_date_dt'])
        
        df['date_parsed'] = df['full_date_dt']
        df = df.sort_values('date_parsed').reset_index(drop=True)
        
        print(f"  Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
    else:
        print(f"  Warning: No 'full_date' column found")
    
    # Handle missing values in activities
    if 'activities' in df.columns:
        missing_activities = df['activities'].isna().sum()
        if missing_activities > 0:
            print(f"  Filling {missing_activities} missing activity values")
            df['activities'] = df['activities'].fillna('')
        df['activities'] = df['activities'].str.strip()
    
    print(f"Cleaned: {initial_rows:,} -> {len(df):,} rows ({initial_rows - len(df)} removed)")
    
    return df
    """
    Main data preparation function.    
    Args:
        data_dir: Directory for raw data
        output_dir: Directory for processed data
        
    Returns:
        str: Path to processed data file
    """
    print("DATA preparation:")
    
    # Check credentials
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    csv_path = download_dataset(data_dir)
    
    # Load raw data
    df = load_raw_data(csv_path)
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = add_temporal_features(df)
    df = add_activity_features(df)
    df = add_mood_features(df)
    
    # Save processed data (with cleanup)
    output_path = save_processed_data(df, output_dir)
    
    # Read back the saved data to display correct info
    df_final = pd.read_csv(output_path)
    
    # Print final data infos
    display_info(df_final)
    
    print("\nData preparation complete")
    
    return output_path

if __name__ == "__main__":
    """Main execution function."""