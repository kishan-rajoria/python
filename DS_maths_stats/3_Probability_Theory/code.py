# Probability Example: Simulating Random Variables (Normal Distribution)

# Import necessary libraries
import numpy as np               # For numerical operations and random number generation
import matplotlib.pyplot as plt  # For plotting

# --- Simulation Parameters ---
mean = 0       # Mean (mu) of the normal distribution
std_dev = 1    # Standard deviation (sigma) of the normal distribution
num_samples = 1000 # Number of random samples to generate

# --- Generate Random Samples ---
# Use numpy.random.normal() to draw random samples from a normal (Gaussian) distribution
# Arguments: loc=mean, scale=standard_deviation, size=number_of_samples
data = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

# --- Visualization ---
# Create a histogram to visualize the distribution of the generated samples
# bins: Number of intervals to divide the data range into
# density=True: Normalize the histogram so the area sums to 1 (approximates a PDF)
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Add titles and labels for clarity
plt.title(f'Histogram of {num_samples} Samples from Normal({mean}, {std_dev}^2) Distribution')
plt.xlabel('Value')
plt.ylabel('Density')

# Add a grid for better readability
plt.grid(axis='y', alpha=0.5)

# Display the plot
plt.show()

# --- Basic Statistics (Optional) ---
# Calculate the sample mean and standard deviation
sample_mean = np.mean(data)
sample_std_dev = np.std(data)

print(f"\n--- Simulation Statistics ---")
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {sample_mean:.4f}")
print(f"Theoretical Standard Deviation: {std_dev}")
print(f"Sample Standard Deviation: {sample_std_dev:.4f}")
print("Note: As num_samples increases, sample statistics should converge to theoretical values.") 