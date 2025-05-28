# Basic Data Science Example: Loading and Displaying Data

# Import the pandas library, commonly used for data manipulation and analysis
import pandas as pd

# Sample Data represented as a Python dictionary
# Keys represent column names, values represent data in columns
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [24, 30, 22]}

# Create a Pandas DataFrame from the dictionary
# A DataFrame is a 2-dimensional labeled data structure (like a spreadsheet or SQL table)
df = pd.DataFrame(data)

# Print the DataFrame to the console
print("Sample DataFrame:")
print(df) 