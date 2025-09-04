# Descriptive Statistics Example: Calculation and Visualization

# Import necessary libraries
import pandas as pd               # For data manipulation (DataFrame)
import matplotlib.pyplot as plt  # For plotting
import numpy as np               # For numerical operations (used for std dev calculation)

# --- Sample Data ---
# Representing scores as a list
scores_list = [88, 92, 79, 95, 85, 92, 75, 105] # Added a potential outlier (105)

# Create a Pandas Series (1-dimensional labeled array) for easier calculations
data_series = pd.Series(scores_list)

# --- Calculate Descriptive Statistics ---
print("--- Descriptive Statistics ---")

# Mean
mean_score = data_series.mean()
print(f"Mean Score: {mean_score:.2f}")

# Median
median_score = data_series.median()
print(f"Median Score: {median_score:.2f}")

# Mode
# Note: mode() returns a Series as there can be multiple modes
mode_score = data_series.mode()
print(f"Mode Score(s): {list(mode_score)}")

# Standard Deviation (using Pandas default ddof=1 for sample std dev)
std_dev_score = data_series.std()
print(f"Standard Deviation (Sample): {std_dev_score:.2f}")

# Variance (using Pandas default ddof=1 for sample variance)
variance_score = data_series.var()
print(f"Variance (Sample): {variance_score:.2f}")

# Range
range_score = data_series.max() - data_series.min()
print(f"Range: {range_score}")

# Quartiles (Q1, Q3) and IQR
q1 = data_series.quantile(0.25)
q3 = data_series.quantile(0.75)
iqr = q3 - q1
print(f"Q1 (25th Percentile): {q1:.2f}")
print(f"Q3 (75th Percentile): {q3:.2f}")
print(f"Interquartile Range (IQR): {iqr:.2f}")

# Can also get a summary using describe()
print("\nPandas describe() Summary:")
print(data_series.describe())


# --- Data Visualization: Box Plot ---
print("\nGenerating Box Plot...")
plt.figure(figsize=(6, 8)) # Adjust figure size for better layout

# Create the box plot
# `vert=True` makes it vertical, `patch_artist=True` fills boxes with color
# `showfliers=True` explicitly shows potential outliers
box = plt.boxplot(data_series, vert=True, patch_artist=True, showfliers=True)

# Customize appearance (optional)
colors = ['lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Add titles and labels
plt.title('Box Plot of Scores')
plt.ylabel('Scores')
plt.xticks([1], ['Sample Scores']) # Label the x-axis tick

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show() 