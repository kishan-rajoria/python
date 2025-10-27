# Regression Example: Simple Linear Regression using Scikit-learn

# Import necessary libraries
import numpy as np                          # For numerical operations (arrays)
import matplotlib.pyplot as plt             # For plotting
from sklearn.linear_model import LinearRegression # The regression model class
from sklearn.metrics import mean_squared_error, r2_score # For model evaluation

# --- Sample Data ---
# Let's assume we want to predict 'Salary' (Y) based on 'Years of Experience' (X)
# Independent variable (X) - needs to be a 2D array for scikit-learn
years_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# Dependent variable (y) - can be a 1D array
salary = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]) + np.random.normal(0, 5, 10) # Add some noise

# --- Create and Train the Linear Regression Model ---
# Instantiate the model
model = LinearRegression()

# Train the model using the sample data
# .fit(X, y) finds the optimal intercept (β₀) and slope (β₁)
model.fit(years_experience, salary)

# --- Get Model Parameters ---
intercept = model.intercept_
slope = model.coef_[0] # coef_ is an array, get the first element for SLR
print("--- Linear Regression Model ---")
print(f"Intercept (β₀): {intercept:.2f}")
print(f"Slope (β₁ - Salary increase per year of experience): {slope:.2f}")
print(f"Model Equation: Salary ≈ {intercept:.2f} + {slope:.2f} * YearsExperience")

# --- Make Predictions ---
# Use the trained model to predict salaries for the existing experience levels
salary_predictions = model.predict(years_experience)

# --- Evaluate the Model ---
# Calculate metrics to assess model performance
mse = mean_squared_error(salary, salary_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(salary, salary_predictions)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# --- Plotting ---
print("\nGenerating Plot...")
plt.figure(figsize=(8, 6))

# Scatter plot of the actual data points
plt.scatter(years_experience, salary, color='blue', label='Actual Salary Data')

# Plot the regression line (using the predictions)
plt.plot(years_experience, salary_predictions, color='red', linewidth=2, label='Regression Line')

# Add titles and labels
plt.title('Simple Linear Regression: Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)

# Display the plot
plt.show() 