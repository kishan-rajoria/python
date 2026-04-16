# Unsupervised Learning Example: Anomaly Detection with Isolation Forest & LOF

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Simulate Industrial Sensor Data (Mostly Normal + Some Anomalies) ---
print("--- 1. Simulating Data ---")
np.random.seed(123)
num_samples = 1000

# Simulate normal operation data
dates = pd.to_datetime(np.arange(num_samples) * pd.Timedelta('1 minute') + pd.Timestamp('2023-06-01'))
data = {
    'Temperature_C': np.random.normal(loc=70, scale=5, size=num_samples),
    'Pressure_Bar': np.random.normal(loc=10, scale=0.5, size=num_samples)
}
df = pd.DataFrame(data, index=dates)

# Inject some anomalies
anomaly_indices = np.random.choice(df.index, size=int(num_samples * 0.03), replace=False) # ~3% anomalies
num_anom = len(anomaly_indices)

# Type 1 Anomaly: Sudden spike in temperature
df.loc[anomaly_indices[:num_anom//3], 'Temperature_C'] *= 1.3

# Type 2 Anomaly: Sudden drop in pressure
df.loc[anomaly_indices[num_anom//3 : 2*num_anom//3], 'Pressure_Bar'] *= 0.7

# Type 3 Anomaly: Both temp and pressure deviate moderately
df.loc[anomaly_indices[2*num_anom//3:], 'Temperature_C'] *= 1.15
df.loc[anomaly_indices[2*num_anom//3:], 'Pressure_Bar'] *= 0.85

# Add a real index for plotting convenience later
df['Time_Index'] = np.arange(num_samples)

print(f"Simulated data shape: {df.shape}")
print(f"Number of injected anomalies: {num_anom}")
print("Data head:")
print(df.head())

# --- 2. Feature Scaling ---
# Scaling is generally recommended for LOF
scaler = StandardScaler()
features = ['Temperature_C', 'Pressure_Bar']
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=features)

# --- 3. Anomaly Detection Models ---
contamination_level = 0.03 # Expected proportion of outliers

# Model 1: Isolation Forest
print("\n--- 3a. Training Isolation Forest ---")
isolation_forest = IsolationForest(n_estimators=100, contamination=contamination_level, random_state=42)
isolation_forest.fit(df_scaled)
df['Anomaly_IF'] = isolation_forest.predict(df_scaled)
df['Anomaly_IF'] = df['Anomaly_IF'].map({1: 0, -1: 1}) # Convert to 0=Normal, 1=Anomaly
num_anomalies_if = df['Anomaly_IF'].sum()
print(f"   Isolation Forest detected {num_anomalies_if} anomalies.")

# Model 2: Local Outlier Factor (LOF)
# Note: LOF predicts outliers based on local density. `novelty=False` means it's used for outlier detection on the training data itself.
print("\n--- 3b. Training Local Outlier Factor (LOF) ---")
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_level, novelty=False)
# LOF's predict method directly returns -1 for outliers, 1 for inliers in outlier detection mode
df['Anomaly_LOF'] = lof.fit_predict(df_scaled)
df['Anomaly_LOF'] = df['Anomaly_LOF'].map({1: 0, -1: 1}) # Convert to 0=Normal, 1=Anomaly
num_anomalies_lof = df['Anomaly_LOF'].sum()
print(f"   LOF detected {num_anomalies_lof} anomalies.")


# --- 4. Visualization --- 
# Compare IF and LOF detections
print("\n--- 4. Visualizing Results --- ")

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Plot Isolation Forest Results
axes[0].scatter(df.loc[df['Anomaly_IF'] == 0, 'Time_Index'],
                df.loc[df['Anomaly_IF'] == 0, 'Temperature_C'],
                c='blue', label='Normal (IF)', s=20, alpha=0.6)
axes[0].scatter(df.loc[df['Anomaly_IF'] == 1, 'Time_Index'],
                df.loc[df['Anomaly_IF'] == 1, 'Temperature_C'],
                c='red', label='Anomaly (IF)', s=50, marker='x')
axes[0].set_title('Isolation Forest Anomaly Detection (Temperature)')
axes[0].set_xlabel('Time Index')
axes[0].set_ylabel('Temperature (C)')
axes[0].legend()
axes[0].grid(True)

# Plot LOF Results
axes[1].scatter(df.loc[df['Anomaly_LOF'] == 0, 'Time_Index'],
                df.loc[df['Anomaly_LOF'] == 0, 'Temperature_C'],
                c='green', label='Normal (LOF)', s=20, alpha=0.6)
axes[1].scatter(df.loc[df['Anomaly_LOF'] == 1, 'Time_Index'],
                df.loc[df['Anomaly_LOF'] == 1, 'Temperature_C'],
                c='orange', label='Anomaly (LOF)', s=50, marker='P') # Different marker
axes[1].set_title('Local Outlier Factor (LOF) Anomaly Detection (Temperature)')
axes[1].set_xlabel('Time Index')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Comparison of Anomaly Detection Methods')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
print("Generating plot comparing IF and LOF anomaly detections...")
# plt.show() # Uncomment to display plot

print("\nAnomaly detection example complete.") 