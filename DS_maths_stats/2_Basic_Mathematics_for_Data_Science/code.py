# Linear Algebra Example: Matrix Operations using NumPy

# Import the NumPy library, standard for numerical operations in Python
import numpy as np

# --- Matrix Creation ---
# Create matrix A (a 2x2 NumPy array)
print("Creating Matrix A:")
A = np.array([[1, 2],  # First row
              [3, 4]]) # Second row
print(A)

# Create matrix B (a 2x2 NumPy array)
print("\nCreating Matrix B:")
B = np.array([[5, 6],
              [7, 8]])
print(B)

# --- Matrix Addition ---
# Perform element-wise matrix addition.
# NumPy overloads the '+' operator for ndarray objects.
# Note: Matrices must have the same dimensions for element-wise addition.
print("\nPerforming Matrix Addition (A + B):")
C = A + B

# Print the resulting matrix C
print(C)

# --- Matrix Multiplication (Dot Product) ---
# Perform matrix multiplication (dot product).
# We use the '@' operator or np.dot() function.
# Note: For matrix multiplication A @ B, the number of columns in A
# must equal the number of rows in B.
print("\nPerforming Matrix Multiplication (A @ B):")
D = A @ B  # Or use: D = np.dot(A, B)
print(D) 