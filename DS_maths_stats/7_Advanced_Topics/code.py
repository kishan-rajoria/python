# Advanced Topics Example: Basic Time Series Plotting

# Import necessary libraries
import pandas as pd               # For creating time series data structures
import matplotlib.pyplot as plt  # For plotting
import numpy as np               # For generating sample data

# --- Generate Sample Time Series Data ---

# Create a date range for the time index (e.g., 30 days starting from 2023-01-01)
dates = pd.date_range(start='2023-01-01', periods=30, freq='D') # 'D' means daily frequency

# Generate some sample data (e.g., simulated daily measurements with some trend and noise)
# Creating a simple upward trend with some random fluctuations
np.random.seed(42) # for reproducibility
trend = np.linspace(start=50, stop=80, num=30) # Linear trend from 50 to 80
noise = np.random.normal(loc=0, scale=5, size=30) # Random noise
data_values = trend + noise

# Create a Pandas Series with the dates as the index
time_series = pd.Series(data_values, index=dates)

print("--- Sample Time Series Data (First 5 rows) ---")
print(time_series.head())

# --- Plotting the Time Series ---
print("\nGenerating Time Series Plot...")
plt.figure(figsize=(10, 5)) # Set the figure size

# Use pandas Series plot() method, which works well with DatetimeIndex
time_series.plot(marker='.', linestyle='-') # Plot with lines and markers

# Add titles and labels
plt.title('Sample Daily Time Series Data')
plt.xlabel('Date')
plt.ylabel('Measurement Value')

# Add grid for better readability
plt.grid(True)

# Improve date formatting on the x-axis (optional)
plt.gcf().autofmt_xdate() # Auto-formats the dates to prevent overlap

# Display the plot
plt.tight_layout() # Adjust layout
plt.show() 