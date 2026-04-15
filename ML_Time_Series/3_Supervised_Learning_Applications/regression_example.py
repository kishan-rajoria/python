# Supervised Learning Example: RUL Prediction (Regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Simulate Sensor Data with RUL (Simplified from Case Study 1) ---
print("--- 1. Simulating Multi-Unit Sensor Data for Regression ---")
np.random.seed(55)

def generate_unit_data(unit_id, max_cycles, noise_level=0.5, trend_factor=0.01):
    """Simulates sensor data for one unit leading to failure."""
    cycles = np.arange(1, max_cycles + 1)
    # Simulate sensors with trends + noise
    sensor1 = 20 + trend_factor * cycles**1.1 + np.random.normal(0, noise_level*2, max_cycles)
    sensor2 = 100 - trend_factor * 0.5 * cycles**1.2 + np.random.normal(0, noise_level*3, max_cycles)

    df = pd.DataFrame({
        'unit_id': unit_id,
        'cycle': cycles,
        'sensor1': sensor1,
        'sensor2': sensor2,
    })
    # Calculate RUL (Remaining Useful Life)
    df['RUL'] = max_cycles - df['cycle']
    return df

# Generate data for multiple units
num_units = 15
all_unit_data = []
for i in range(1, num_units + 1):
    max_cycles = np.random.randint(120, 250)
    unit_df = generate_unit_data(i, max_cycles)
    all_unit_data.append(unit_df)

df_full = pd.concat(all_unit_data).reset_index(drop=True)

print(f"Generated data for {num_units} units. Total samples: {len(df_full)}")

# --- 2. Feature Engineering (Simple: Use cycle and sensors) ---
print("\n--- 2. Preparing Data for ML ---")
target = 'RUL'
features = ['cycle', 'sensor1', 'sensor2']

X = df_full[features]
y = df_full[target]

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# --- 4. Feature Scaling ---
# Scaling is important for Ridge, Lasso, and SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Model Training ---
print("\n--- 5. Training Regression Models ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1),
    "SVR (kernel='rbf')": SVR(kernel='rbf', C=1.0, epsilon=0.1) # Default params
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"      {name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# --- 6. Evaluation Summary ---
print("\n--- 6. Model Comparison --- ")
results_df = pd.DataFrame(results).T # Transpose for better readability
print(results_df)

# --- 7. Visualization (Actual vs. Predicted for selected models) ---
print("\n--- 7. Visualizing Predictions --- ")
plt.figure(figsize=(12, 6))

# Plot for Ridge Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions["Ridge (alpha=1.0)"], alpha=0.5, s=10, label='Ridge Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Ridge Regression (Actual vs. Predicted RUL)")
plt.legend()
plt.grid(True)

# Plot for SVR
plt.subplot(1, 2, 2)
plt.scatter(y_test, predictions["SVR (kernel='rbf')"], alpha=0.5, s=10, label='SVR Predictions', color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("SVR (Actual vs. Predicted RUL)")
plt.legend()
plt.grid(True)

plt.tight_layout()
print("Generating plots of Actual vs Predicted RUL for Ridge and SVR...")
# plt.show()

print("\nRegression examples complete.") 