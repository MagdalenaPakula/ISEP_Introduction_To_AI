import pandas as pd

dataset = pd.read_csv('COVID19_line_list_data.csv')

# Display the first few rows of the dataset
print(dataset.head())

# Get information about the columns and data types
print(dataset.info())

# Check for missing values
print(dataset.isnull().sum())

# Compute correlations between variables
correlations = dataset.corr()

# Display the correlation matrix
print(correlations)
