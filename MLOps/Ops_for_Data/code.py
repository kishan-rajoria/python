# Ensure required libraries are installed:
# pip install pandas numpy

import pandas as pd
import numpy as np
import os

print("--- Dummy MLOps Data Pipeline Script ---")

# --- Stage 1: Data Ingestion (Simulated) ---
print("\n[Stage 1: Data Ingestion]")
# In a real pipeline, this would read from a database, API, file system (S3, GCS, etc.)
# We simulate reading raw data into a pandas DataFrame.
data = {
    'user_id': [1, 2, 3, 4, 5, 3], # Duplicate ID
    'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:05', '2023-01-01 10:10', 
                              '2023-01-01 10:15', '2023-01-01 10:20', '2023-01-01 10:10']),
    'sensor_A': [10.5, 11.2, None, 10.8, 11.5, 10.1], # Missing value
    'sensor_B': ['20.1', '21.3', '20.5', '21.0', '19.9', '20.5'], # Incorrect type (string)
    'category': ['X', 'Y', 'X', 'Z', 'Y', 'X']
}
raw_df = pd.DataFrame(data)
print(f"Simulated raw data ingested. Shape: {raw_df.shape}")
print("Raw Data Sample:\n", raw_df.head())

# --- Stage 2: Data Validation (Initial - Basic Checks) ---
print("\n[Stage 2: Initial Data Validation]")
# In MLOps, use tools like Great Expectations or Pandera here for comprehensive checks.
# Simple example: Check for required columns.
required_columns = ['user_id', 'timestamp', 'sensor_A', 'sensor_B', 'category']
missing_cols = [col for col in required_columns if col not in raw_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
print(f"Initial validation passed: Required columns {required_columns} present.")
# Add checks for unexpected nulls if needed.

# --- Stage 3: Data Cleaning & Preprocessing ---
print("\n[Stage 3: Data Cleaning & Preprocessing]")
df_processed = raw_df.copy()

# a) Handle Missing Values (Example: fill sensor_A with mean)
mean_sensor_a = df_processed['sensor_A'].mean()
df_processed['sensor_A'].fillna(mean_sensor_a, inplace=True)
print(f"Filled missing values in 'sensor_A' with mean ({mean_sensor_a:.2f}).")

# b) Correct Data Types (Example: convert sensor_B to numeric)
try:
    df_processed['sensor_B'] = pd.to_numeric(df_processed['sensor_B'])
    print("Converted 'sensor_B' to numeric type.")
except ValueError as e:
    print(f"Warning: Could not convert 'sensor_B' to numeric. Error: {e}")
    # Handle conversion errors more robustly in production (e.g., set to NaN, log)

# c) Remove Duplicates (Example: based on user_id and timestamp)
initial_rows = len(df_processed)
df_processed.drop_duplicates(subset=['user_id', 'timestamp'], keep='first', inplace=True)
rows_dropped = initial_rows - len(df_processed)
print(f"Removed {rows_dropped} duplicate row(s) based on 'user_id' and 'timestamp'.")

print(f"Data cleaning/preprocessing done. Shape: {df_processed.shape}")

# --- Stage 4: Feature Engineering ---
print("\n[Stage 4: Feature Engineering]")
# Create simple features. In practice, this could involve complex domain-specific logic.

# a) Example: Calculate ratio between sensors
# Avoid division by zero or issues with non-numeric types
if pd.api.types.is_numeric_dtype(df_processed['sensor_A']) and pd.api.types.is_numeric_dtype(df_processed['sensor_B']):
    # Add small epsilon to avoid potential division by zero if sensor_B can be 0
    df_processed['sensor_ratio_A_B'] = df_processed['sensor_A'] / (df_processed['sensor_B'] + 1e-6)
    print("Created new feature: 'sensor_ratio_A_B'.")
else:
    print("Skipping feature 'sensor_ratio_A_B' due to non-numeric sensor types.")

# b) Example: Extract hour from timestamp
df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
print("Created new feature: 'hour_of_day'.")

print(f"Feature engineering done. Shape: {df_processed.shape}")
print("Processed Data Sample with Features:\n", df_processed.head())

# --- Stage 5: Data Validation (Post-Transformation - Basic Checks) ---
print("\n[Stage 5: Post-Transformation Validation]")
# Again, use tools like Great Expectations for robust checks (distributions, ranges etc.)

# a) Example: Check value range for sensor_A (assuming it should be positive)
if 'sensor_A' in df_processed and pd.api.types.is_numeric_dtype(df_processed['sensor_A']):
    if (df_processed['sensor_A'] < 0).any():
        print("Warning: Found negative values in 'sensor_A' post-processing!")
        # In production: Raise error, quarantine data, or trigger alert
    else:
        print("Post-transform validation passed: 'sensor_A' values are non-negative.")

# b) Example: Check if new feature has expected type
if 'hour_of_day' in df_processed:
    assert pd.api.types.is_integer_dtype(df_processed['hour_of_day']), "Feature 'hour_of_day' is not integer type!"
    print("Post-transform validation passed: 'hour_of_day' has expected integer type.")

# --- Stage 6 & 7: Data Versioning & Loading/Saving (Simulated) ---
print("\n[Stage 6 & 7: Versioning & Saving Output]")
# In MLOps:
# - Use DVC or LakeFS to version the output dataset relative to code/config.
# - Load data into a Feature Store or a versioned table in a data lake/warehouse.

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "processed_sensor_data.csv")

try:
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print("Note: In a real MLOps pipeline, this save operation would be versioned (e.g., using DVC)." )
except Exception as e:
    print(f"Error saving processed data: {e}")

print("\n--- Dummy Data Pipeline Script Finished ---") 