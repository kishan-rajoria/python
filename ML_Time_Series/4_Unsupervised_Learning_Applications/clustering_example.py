# Unsupervised Learning Example: Clustering Industrial Process Data

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score # For evaluating clustering
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate Industrial Process Data for Clustering ---
# Simulate data representing different operational states based on two sensors
print("--- 1. Simulating Process Data for Clustering ---")
np.random.seed(202)
num_samples = 500

# State 1: Normal operation
data1 = np.random.multivariate_normal(mean=[50, 10], cov=[[5, 1], [1, 0.5]], size=num_samples//2)

# State 2: High temp, low pressure state
data2 = np.random.multivariate_normal(mean=[70, 8], cov=[[6, -1.5], [-1.5, 0.8]], size=num_samples//4)

# State 3: Low temp, high pressure state (more scattered)
data3 = np.random.multivariate_normal(mean=[40, 12], cov=[[8, 2], [2, 1.0]], size=num_samples//4)

# Combine data
data = np.vstack((data1, data2, data3))
df = pd.DataFrame(data, columns=['Sensor_A', 'Sensor_B'])

# Add some random noise points (outliers for DBSCAN)
noise_points = np.random.uniform(low=[30, 5], high=[80, 15], size=(20, 2))
df_noise = pd.DataFrame(noise_points, columns=['Sensor_A', 'Sensor_B'])
df = pd.concat([df, df_noise], ignore_index=True)

print(f"Simulated data shape: {df.shape}")
print("Data head:")
print(df.head())

# Visualize the raw data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Sensor_A', y='Sensor_B', alpha=0.6)
plt.title('Simulated Raw Process Data')
plt.xlabel('Sensor A Reading')
plt.ylabel('Sensor B Reading')
plt.grid(True)
# plt.show()

# --- 2. Feature Scaling ---
# Important for distance-based clustering like K-Means and DBSCAN
print("\n--- 2. Scaling Features ---")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# --- 3. Clustering Algorithms ---

# Model 1: K-Means
# We need to specify k (number of clusters). Let's assume we expect 3 operational states.
print("\n--- 3a. Applying K-Means (k=3) ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init to run multiple times
kmeans.fit(df_scaled)
df['KMeans_Cluster'] = kmeans.labels_

# Evaluate K-Means
sil_kmeans = silhouette_score(df_scaled, kmeans.labels_)
print(f"   K-Means Silhouette Score: {sil_kmeans:.3f}")

# Model 2: DBSCAN
# Does not require k, but needs eps (radius) and min_samples
# Finding optimal eps and min_samples often requires experimentation
print("\n--- 3b. Applying DBSCAN --- ")
# Parameters chosen by rough visual inspection / iteration
dbscan = DBSCAN(eps=0.4, min_samples=5)
dbscan.fit(df_scaled)
df['DBSCAN_Cluster'] = dbscan.labels_ # -1 indicates noise/outliers

# Evaluate DBSCAN (excluding noise points for silhouette score)
if len(set(dbscan.labels_)) > 1: # Need at least 2 clusters (excluding noise)
    mask = dbscan.labels_ != -1
    sil_dbscan = silhouette_score(df_scaled[mask], dbscan.labels_[mask])
    print(f"   DBSCAN Silhouette Score (excl. noise): {sil_dbscan:.3f}")
else:
    print("   DBSCAN found fewer than 2 clusters (excluding noise), Silhouette score not applicable.")
num_noise_dbscan = (dbscan.labels_ == -1).sum()
print(f"   DBSCAN identified {num_noise_dbscan} noise points.")


# --- 4. Visualization of Clusters --- 
print("\n--- 4. Visualizing Clustering Results --- ")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# Plot K-Means Results
unique_labels_km = sorted(df['KMeans_Cluster'].unique())
palette_km = sns.color_palette("viridis", len(unique_labels_km))
sns.scatterplot(ax=axes[0], data=df, x='Sensor_A', y='Sensor_B', hue='KMeans_Cluster',
                palette=palette_km, legend='full', alpha=0.7)
axes[0].set_title(f'K-Means Clustering (k=3, Silhouette={sil_kmeans:.3f})')
axes[0].set_xlabel('Sensor A Reading')
axes[0].set_ylabel('Sensor B Reading')
axes[0].grid(True)

# Plot DBSCAN Results
unique_labels_db = sorted(df['DBSCAN_Cluster'].unique())
palette_db = sns.color_palette("icefire", len(unique_labels_db))
# Handle noise points (-1 label) specifically
hue_order_db = [l for l in unique_labels_db if l != -1] + ([-1] if -1 in unique_labels_db else [])
palette_db_dict = {label: col for label, col in zip(hue_order_db, palette_db)}
if -1 in palette_db_dict:
    palette_db_dict[-1] = (0.5, 0.5, 0.5) # Gray for noise

sns.scatterplot(ax=axes[1], data=df, x='Sensor_A', y='Sensor_B', hue='DBSCAN_Cluster',
                hue_order=hue_order_db, palette=palette_db_dict, legend='full', alpha=0.7)
axes[1].set_title(f'DBSCAN Clustering (Noise={num_noise_dbscan}, Sil={sil_dbscan:.3f}*)')
axes[1].set_xlabel('Sensor A Reading')
axes[1].grid(True)

plt.suptitle('Comparison of Clustering Methods')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
print("Generating plot comparing K-Means and DBSCAN clustering results...")
# plt.show() # Uncomment to display plot

print("\nClustering examples complete.") 