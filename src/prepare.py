"""
Data Preparation Script for Mood & Activity Prediction Project.
Author: Alessio Bolla
"""

import os
import sys
import pandas as pd
import numpy as np

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


def add_temporal_features(df):
    """
    Create top 15 temporal features based on mood prediction research.
    
    Evidence-based feature selection from academic literature:
    - Lag features: Strongest predictors 
    - Rolling statistics: Capture mood trends
    - Cyclical encoding: Weekend/seasonal patterns
    - Momentum indicators: Direction of change
    
    Args:
        df: Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with temporal features
    """
    print("\nAdd temporal features:")
    
    date_col = 'date_parsed'
    
    if date_col not in df.columns:
        print("  Error: No parsed date column found")
        return df
    
    if df[date_col].dtype != 'datetime64[ns]':
        print(f"  Error: Column '{date_col}' is not datetime type")
        return df
    
    # Extract basic temporal components
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    # Encode mood numerically for temporal features
    if 'mood' in df.columns:
        mood_mapping = {'Awful': 0, 'Bad': 1, 'Normal': 2, 'Good': 3, 'Amazing': 4, 'Excellent': 4}
        df['mood_numeric'] = df['mood'].map(mood_mapping)
        if df['mood_numeric'].isna().any():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['mood_numeric'] = le.fit_transform(df['mood'])
    
    # Lag features
    df['mood_lag_1'] = df['mood_numeric'].shift(1).fillna(df['mood_numeric'].median())
    df['mood_lag_7'] = df['mood_numeric'].shift(7).fillna(df['mood_numeric'].median())
    df['mood_lag_2'] = df['mood_numeric'].shift(2).fillna(df['mood_numeric'].median())
    
    # Rolling statistics
    df['rolling_mean_7d'] = df['mood_numeric'].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7d'] = df['mood_numeric'].rolling(window=7, min_periods=1).std().fillna(0)
    df['rolling_mean_3d'] = df['mood_numeric'].rolling(window=3, min_periods=1).mean()
    df['rolling_min_7d'] = df['mood_numeric'].rolling(window=7, min_periods=1).min()
    
    # MSSD: Mean Squared Successive Difference
    successive_diffs = df['mood_numeric'].diff(1) ** 2
    df['MSSD_7d'] = successive_diffs.rolling(window=7, min_periods=1).mean().fillna(0)
    
    # Cyclical encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Momentum indicators
    df['first_difference'] = df['mood_numeric'].diff(1).fillna(0)
    df['momentum_3d'] = df['first_difference'].rolling(window=3, min_periods=1).mean()
    
    # Trend slope
    def calc_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series, 1)[0]
            return slope
        except:
            return 0
    
    df['trend_slope_7d'] = df['mood_numeric'].rolling(window=7, min_periods=2).apply(calc_slope, raw=True)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    
    print(f"  Created 15 temporal features")
    
    return df


def add_activity_features(df):
    """
    Create top 15 activity/behavioral features.
    Maintains specific activities while adding interpretable aggregates.
    
    Args:
        df: Dataset with activities
        
    Returns:
        pd.DataFrame: Updated dataset
    """
    print("\nAdd activity features:")
    
    if 'activities' not in df.columns:
        print("  Warning: No activities column found, skipping")
        return df
    
    # Split activities by separator
    activities_split = df['activities'].str.split(' | ')
    
    # Get all unique activities
    all_activities = set()
    for activities_list in activities_split:
        if isinstance(activities_list, list):
            all_activities.update(activities_list)
    
    all_activities.discard('')
    all_activities = {a.strip() for a in all_activities if a.strip()}
    all_activities = sorted(all_activities)
    
    print(f"  Found {len(all_activities)} unique activities")
    
    # Count frequency of each activity
    activity_counts = {}
    for activity in all_activities:
        count = df['activities'].str.contains(activity, regex=False, na=False).sum()
        activity_counts[activity] = count
    
    # Keep top 10 most frequent specific activities
    top_10_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"  Keeping top 10 most frequent activities")
    feature_count = 0
    
    # Create binary features for top 10 activities
    for activity, count in top_10_activities:
        col_name = f"activity_{activity.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = df['activities'].str.contains(activity, regex=False, na=False).astype(int)
        feature_count += 1
    
    # Aggregate features
    df['total_activity_count'] = activities_split.apply(lambda x: len(x) if isinstance(x, list) else 0)
    feature_count += 1
    
    # Social activity indicator
    social_keywords = ['friend', 'family', 'social', 'party', 'date', 'people']
    df['any_social_activity'] = df['activities'].apply(
        lambda x: int(any(keyword in str(x).lower() for keyword in social_keywords))
    )
    feature_count += 1
    
    # Physical activity count
    physical_keywords = ['exercise', 'gym', 'sport', 'walk', 'run', 'bike', 'yoga']
    df['physical_activity_count'] = df['activities'].apply(
        lambda x: sum(keyword in str(x).lower() for keyword in physical_keywords)
    )
    feature_count += 1
    
    # Self-care count
    selfcare_keywords = ['meditation', 'relax', 'sleep', 'rest', 'spa', 'massage']
    df['self_care_count'] = df['activities'].apply(
        lambda x: sum(keyword in str(x).lower() for keyword in selfcare_keywords)
    )
    feature_count += 1
    
    # Activity diversity
    df['activity_diversity_7d'] = df['total_activity_count'].rolling(
        window=7, min_periods=1
    ).std().fillna(0)
    feature_count += 1
    
    print(f"  Created {feature_count} activity features")
    
    return df


def add_mood_features(df):
    """
    Encode mood labels and create additional mood-related features.

    Args:
        df: Dataset with mood column
        
    Returns:
        pd.DataFrame: Dataset with encoed mood features
    """
    print("\nAdd mood features:")
    
    mood_col = 'mood'
    
    if mood_col not in df.columns:
        print("  Warning: No mood column found")
        return df
    
    # Display mood distribution
    mood_counts = df[mood_col].value_counts()
    print(f"  Mood distribution:")
    for mood, count in mood_counts.items():
        print(f"    {mood}: {count} ({count/len(df)*100:.1f}%)")
    
    # Define ordinal mapping
    ordinal_mappings = [
        {'Awful': 0, 'Bad': 1, 'Normal': 2, 'Good': 3, 'Amazing': 4, 'Excellent': 4},
        {'rad': 0, 'Normal': 1, 'Good': 2, 'Amazing': 3},
        {'Bad': 0, 'Normal': 1, 'Good': 2}
    ]
    
    # Try to find matching ordinal mapping
    unique_moods = set(df[mood_col].unique())
    mood_mapping = None
    
    for mapping in ordinal_mappings:
        if unique_moods.issubset(set(mapping.keys())):
            mood_mapping = mapping
            break
    
    # Apply encoding
    if mood_mapping is not None:
        df['mood_encoded'] = df[mood_col].map(mood_mapping)
        print(f"  Using ordinal encoding")
        
        # Save reverse mapping for predictions
        reverse_mapping = {v: k for k, v in mood_mapping.items()}
        import json
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/mood_mapping.json', 'w') as f:
            json.dump({
                'encoding': mood_mapping,
                'decoding': reverse_mapping,
                'type': 'ordinal'
            }, f, indent=2)
        print(f"  Mood mapping saved to data/processed/mood_mapping.json")
        
    else:
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['mood_encoded'] = le.fit_transform(df[mood_col])
        
        print(f"  Using label encoding for {df[mood_col].nunique()} unique moods")
        
        # Save encoder
        import joblib
        os.makedirs('data/processed', exist_ok=True)
        joblib.dump(le, 'data/processed/label_encoder.pkl')
        print(f"  Label encoder saved to data/processed/label_encoder.pkl")
    
    # Create target column
    df['target'] = df['mood_encoded']
    
    # Verify no missing values in target
    if df['target'].isna().any():
        print(f"  Warning: {df['target'].isna().sum()} missing values in target")
        df = df.dropna(subset=['target'])
    
    print(f"Mood encoding complete")
    print(f"  Target range: {df['target'].min():.0f} to {df['target'].max():.0f}")
    print(f"  Classes: {df['target'].nunique()}")
    
    return df


def save_processed_data(df, output_dir="data/processed"):
    """
    Save the processed dataset to CSV.
    Removes intermediate columns and keeps only the 31 essential features.
    Ensures no missing values in final dataset.
    
    Args:
        df: Processed dataset
        output_dir: Directory to save processed data
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define columns to keep: 15 temporal + 15 activity + 1 target = 31 total
    temporal_keep = [
        'mood_lag_1', 'mood_lag_7', 'mood_lag_2',
        'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_3d', 'rolling_min_7d', 'MSSD_7d',
        'day_of_week_sin', 'day_of_week_cos', 'is_weekend',
        'first_difference', 'momentum_3d', 'trend_slope_7d', 'month_sin'
    ]
    
    activity_keep = [col for col in df.columns if col.startswith('activity_') or 
                     col in ['total_activity_count', 'any_social_activity', 
                            'physical_activity_count', 'self_care_count', 'activity_diversity_7d']]
    
    # Keep only essential columns (+ date_parsed for temporal split in train.py)
    columns_to_keep = temporal_keep + activity_keep + ['target', 'date_parsed']
    
    # Verify all columns exist
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # Filter dataframe to keep only essential features
    df_clean = df[columns_to_keep].copy()
    
    # Handle any remaining missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Handling {missing_count} remaining missing values")
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"    Filled {col} with median: {median_val:.3f}")
    
    output_path = os.path.join(output_dir, "processed_data.csv")
    df_clean.to_csv(output_path, index=False)
    
    print(f"\nProcessed data saved: {output_path}")
    print(f"  Shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    print(f"  Kept: 15 temporal + {len(activity_keep)} activity + 1 target = {df_clean.shape[1]} features")
    
    return output_path


def display_info(df):
    """
    Display informations about the processed dataset.
    
    Args:
        df: Processed dataset
    """
    print("\nData informations:")
    
    print(f"  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    temporal_features = [col for col in df.columns if any(x in col for x in 
                        ['lag', 'rolling', 'MSSD', 'difference', 'momentum', 'slope', 'sin', 'cos', 'weekend'])]
    activity_features = [col for col in df.columns if col.startswith('activity_') or 
                        col in ['total_activity_count', 'any_social_activity', 
                               'physical_activity_count', 'self_care_count', 'activity_diversity_7d']]
    
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Activity features: {len(activity_features)}")
    print(f"  Target: 1")
    
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")


def prepare_data(data_dir="data/raw", output_dir="data/processed"):
    """
    Main data preparation function.
    Can be called from main.py or run standalone.
    
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
    prepare_data()