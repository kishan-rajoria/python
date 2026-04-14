# Industrial Data Preprocessing Examples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate # For more advanced interpolation

# --- 1. Simulate Industrial Sensor Data ---
print("--- 1. Simulating Data ---")
# Create a time series index (e.g., every 10 seconds for an hour)
dates = pd.date_range(start='2023-04-15 08:00:00', periods=360, freq='10S')

# Simulate sensor readings
np.random.seed(42)
data = {
    'Temperature_C': 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 360)) + np.random.normal(0, 0.5, 360),
    'Pressure_Pa': 101325 + 100 * np.sin(np.linspace(0, 4 * np.pi, 360)) + np.random.normal(0, 10, 360),
    'Vibration_mm_s2': 0.1 + np.random.rand(360) * 0.2 + 0.05 * np.sin(np.linspace(0, 10 * np.pi, 360))
}
df = pd.DataFrame(data, index=dates)

# Introduce some common issues:
# - Missing values
df.loc['2023-04-15 08:05:00':'2023-04-15 08:05:50', 'Temperature_C'] = np.nan
df.loc['2023-04-15 08:20:10', 'Pressure_Pa'] = np.nan
df.loc['2023-04-15 08:45:30', 'Vibration_mm_s2'] = np.nan
# - Outlier
df.loc['2023-04-15 08:15:00', 'Vibration_mm_s2'] = 1.5 # Inject an outlier

print("Original DataFrame shape:", df.shape)
print("Original DataFrame info:")
df.info()
print("\nOriginal DataFrame head:")
print(df.head())
print("\nMissing values before handling:")
print(df.isnull().sum())

# --- 2. Timestamp Handling & Resampling ---
print("\n--- 2. Timestamp Handling & Resampling ---")
# Assuming timestamps are already parsed correctly by pd.date_range
# Let's resample to 1-minute frequency, aggregating using mean
df_resampled = df.resample('1min').mean()

print("Resampled (1 min avg) DataFrame shape:", df_resampled.shape)
print("Resampled DataFrame head:")
print(df_resampled.head())
print("\nMissing values after resampling (mean ignores NaNs by default):")
print(df_resampled.isnull().sum()) # Mean automatically handles NaNs in aggregation

# --- 3. Missing Data Imputation ---
# We'll work with the 1-minute resampled data
print("\n--- 3. Missing Data Imputation (on 1-min data) ---")

# Option A: Forward Fill (simple for slowly changing values)
df_ffill = df_resampled.copy()
df_ffill.fillna(method='ffill', inplace=True)
print("Missing values after ffill:")
print(df_ffill.isnull().sum())

# Option B: Linear Interpolation (often better for trends)
df_interpolated = df_resampled.copy()
df_interpolated.interpolate(method='linear', inplace=True)
print("\nMissing values after linear interpolation:")
print(df_interpolated.isnull().sum())

# We'll proceed with the interpolated data
df_imputed = df_interpolated.copy()

# --- 4. Outlier Detection/Visualization ---
print("\n--- 4. Outlier Detection/Visualization (on imputed 1-min data) ---")

# Box plot to visually identify outliers, especially in Vibration
plt.figure(figsize=(10, 4))
plt.boxplot(df_imputed['Vibration_mm_s2'], vert=False, patch_artist=True, labels=['Vibration'])
plt.title('Box Plot of Resampled & Imputed Vibration Data')
plt.xlabel('Vibration (mm/s^2)')
plt.grid(True)
print("Generating Box Plot for Vibration - check for outliers visually...")
# plt.show() # Uncomment to display plot if running interactively

# Simple outlier handling: Capping (Winsorizing) based on IQR (example)
Q1 = df_imputed['Vibration_mm_s2'].quantile(0.25)
Q3 = df_imputed['Vibration_mm_s2'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Vibration IQR: {IQR:.4f}, Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}")
df_capped = df_imputed.copy()
# Cap values outside the bounds
df_capped['Vibration_mm_s2'] = np.clip(df_capped['Vibration_mm_s2'], lower_bound, upper_bound)

print("Max Vibration before capping:", df_imputed['Vibration_mm_s2'].max())
print("Max Vibration after capping:", df_capped['Vibration_mm_s2'].max())

# We'll proceed with the capped data
df_processed = df_capped.copy()

# --- 5. Feature Scaling ---
print("\n--- 5. Feature Scaling (on imputed & capped 1-min data) ---")

# Option A: Standardization (Z-score)
scaler_standard = StandardScaler()
df_standardized = scaler_standard.fit_transform(df_processed)
df_standardized = pd.DataFrame(df_standardized, index=df_processed.index, columns=df_processed.columns)

print("Standardized Data (Mean should be ~0, Std Dev ~1):")
print(df_standardized.describe().loc[['mean', 'std']])

# Option B: Normalization (Min-Max Scaling to [0, 1])
scaler_minmax = MinMaxScaler()
df_normalized = scaler_minmax.fit_transform(df_processed)
df_normalized = pd.DataFrame(df_normalized, index=df_processed.index, columns=df_processed.columns)

print("\nNormalized Data (Min should be 0, Max 1):")
print(df_normalized.describe().loc[['min', 'max']])

print("\nPreprocessing demonstration complete.")

# Optional: Plotting to compare original, imputed, scaled data
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(df['Temperature_C'], label='Original Temp (10s)', alpha=0.7)
# plt.title('Original Data')
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(df_imputed['Temperature_C'], label='Imputed Temp (1min avg)', marker='.')
# plt.title('Resampled & Imputed Data')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(df_standardized['Temperature_C'], label='Standardized Temp', marker='.')
# plt.title('Standardized Data')
# plt.legend()
# plt.tight_layout()
# plt.show()
