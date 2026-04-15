# Supervised Learning Example: Equipment Failure Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate Industrial Sensor Data with Failure Labels ---
print("--- 1. Simulating Data ---")
np.random.seed(42)
num_samples = 1000

# Simulate normal operation data
data = {
    'Timestamp': pd.to_datetime(np.arange(num_samples) * pd.Timedelta('1 minute') + pd.Timestamp('2023-05-01')),
    'Sensor1_Temp': np.random.normal(loc=50, scale=5, size=num_samples),
    'Sensor2_Vibration': np.random.normal(loc=0.1, scale=0.02, size=num_samples),
    'Sensor3_Pressure': np.random.normal(loc=1000, scale=50, size=num_samples)
}
df = pd.DataFrame(data)
df.set_index('Timestamp', inplace=True)

# Simulate failure conditions towards the end
failure_point = int(num_samples * 0.95) # 5% of data represents impending failure
failure_duration = num_samples - failure_point

df.loc[df.index[failure_point]:, 'Sensor1_Temp'] += np.linspace(0, 15, failure_duration) # Temp increases before failure
df.loc[df.index[failure_point]:, 'Sensor2_Vibration'] += np.linspace(0, 0.1, failure_duration) # Vibration increases
df.loc[df.index[failure_point]:, 'Sensor3_Pressure'] -= np.linspace(0, 100, failure_duration) # Pressure might drop

# Add some noise to the failure trend
df.loc[df.index[failure_point]:, 'Sensor1_Temp'] += np.random.normal(0, 2, failure_duration)
df.loc[df.index[failure_point]:, 'Sensor2_Vibration'] += np.random.normal(0, 0.01, failure_duration)
df.loc[df.index[failure_point]:, 'Sensor3_Pressure'] += np.random.normal(0, 10, failure_duration)

# Create Failure Label (1 if failure is imminent, 0 otherwise)
df['Failure'] = 0
df.loc[df.index[failure_point]:, 'Failure'] = 1

print(f"Simulated data shape: {df.shape}")
print(f"Failure data points: {df['Failure'].sum()}")
print("Data head:")
print(df.head())
print("\nData tail (showing failure trend):")
print(df.tail())

# --- 2. Feature Engineering (Simple Example) ---
# Using sensor readings directly. More complex features (lags, rolling stats) could be added (see Topic 7).
features = ['Sensor1_Temp', 'Sensor2_Vibration', 'Sensor3_Pressure']
X = df[features]
y = df['Failure']

# --- 3. Data Splitting (Temporal Split Recommended, but using random for simplicity here) ---
# **IMPORTANT**: For real time series, use a temporal split (train on older data, test on newer).
# Example: train_size = int(len(df) * 0.8); X_train, X_test = X[:train_size], X[train_size:] ...
# Using random split here just for demonstrating the classification workflow.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify helps with imbalanced data

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# --- 4. Feature Scaling ---
# Scaling is important for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Model Training ---
print("\n--- 5. Training Models ---")

# Model 1: Random Forest Classifier
print("   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Handle imbalance
rf_model.fit(X_train_scaled, y_train)

# Model 2: Logistic Regression
print("   Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear') # Handle imbalance
lr_model.fit(X_train_scaled, y_train)

# --- 6. Prediction ---
print("--- 6. Making Predictions --- ")
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1] # Prob for positive class

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1] # Prob for positive class


# --- 7. Evaluation ---
print("\n--- 7. Evaluating Model Performance ---")

print("\n--- Random Forest Results ---")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"ROC AUC Score: {roc_auc_rf:.4f}") # Good for imbalanced classes
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
# plt.show() # Uncomment to display plot

print("\n--- Logistic Regression Results ---")
accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"ROC AUC Score: {roc_auc_lr:.4f}") # Good for imbalanced classes
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens", xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")
# plt.show() # Uncomment to display plot


print("\nNote: Given the simulated nature, results look good. Real-world data requires more complex feature engineering and careful evaluation, especially with imbalanced classes.")
print("Comparing models shows Random Forest may capture non-linearities better here, but Logistic Regression provides a simpler baseline.") 