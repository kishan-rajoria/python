# Practical Application Example: Loading Data and Basic EDA

# Import necessary libraries
import pandas as pd
import os # To check if the sample file exists

# --- Define Dataset Path ---
dataset_filename = 'sample_dataset.csv'

# --- Create a Dummy Dataset (if it doesn't exist) ---
# This part is just for demonstration so the code runs without needing an external file initially.
# In a real scenario, you would replace dataset_filename with the path to your actual data.
if not os.path.exists(dataset_filename):
    print(f"Creating a dummy '{dataset_filename}' for demonstration...")
    # Create a sample DataFrame
    dummy_data = {
        'FeatureA': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0],
        'FeatureB': [10, 12, 11, 13, 14, 12, 15, 16, 14, 17],
        'Category': ['X', 'Y', 'X', 'Y', 'Z', 'X', 'Z', 'Y', 'X', 'Z'],
        'Target': [105, 115, 110, 125, 130, 122, 135, 140, 132, 145]
    }
    dummy_df = pd.DataFrame(dummy_data)
    # Save it to a CSV file
    dummy_df.to_csv(dataset_filename, index=False)
    print("Dummy dataset created.")
else:
    print(f"Using existing '{dataset_filename}'.")

# --- Load the Dataset --- 
# Use a try-except block to handle potential file not found errors gracefully.
print(f"\nAttempting to load dataset: {dataset_filename}")
try:
    # Read the data from the CSV file into a Pandas DataFrame
    df = pd.read_csv(dataset_filename)

    # --- Initial Exploration (Basic EDA) ---
    print("\nDataset loaded successfully. Performing basic EDA...")

    # Display the first few rows of the DataFrame
    print("\nFirst 5 rows of the dataset (df.head()):")
    print(df.head())

    # Display information about the DataFrame (column types, non-null counts)
    print("\nDataFrame Info (df.info()):")
    df.info()

    # Display basic descriptive statistics for numerical columns
    # Includes count, mean, std dev, min, max, and quartiles
    print("\nDescriptive Statistics for numerical columns (df.describe()):")
    print(df.describe())

    # Display descriptive statistics for categorical columns (optional)
    # Includes count, unique values, top value, and frequency of top value
    print("\nDescriptive Statistics for categorical columns (df.describe(include='object')):")
    print(df.describe(include='object')) # 'object' usually refers to string columns

except FileNotFoundError:
    print(f"Error: The file '{dataset_filename}' was not found.")
    print("Please make sure the dataset file exists in the correct directory or update the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 