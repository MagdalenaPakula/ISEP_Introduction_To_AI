import pandas as pd
import numpy as np

# Loading the dataset
dataset = pd.read_csv('COVID19_line_list_data.csv')

# Computing the correlations
correlations = dataset.corr()

# Find correlations with the target variable (outcome)
target_variable = 'death'
target_correlations = correlations[target_variable].dropna().sort_values(ascending=False)

# Print the correlations with the target variable
print("Correlations with the target variable:")
print(target_correlations)

