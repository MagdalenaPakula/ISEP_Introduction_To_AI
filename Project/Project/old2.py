# Import libraries - EX 1
import pandas as pd
import numpy as np

# Import libraries - EX 2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reading the dataset
data = pd.read_csv('COVID19_line_list_data.csv') 

# Changging types of variables
data['from Wuhan'] = data['from Wuhan'].fillna(0).astype(int)
data['death'] = pd.to_numeric(data['death'], errors='coerce').astype('Int64')
data['recovered'] = pd.to_numeric(data['recovered'], errors='coerce').astype('Int64')
data['symptom_onset'] = pd.to_datetime(data['symptom_onset'])
data['hosp_visit_date'] = pd.to_datetime(data['hosp_visit_date'])
data['exposure_end'] = pd.to_datetime(data['exposure_end'])
data['gender'] = data['gender'].astype(str)
data['location'] = data['location'].astype(str)
data['exposure_start'] = pd.to_datetime(data['exposure_start'])
data['visiting Wuhan'] = data['visiting Wuhan'].astype(int)
data['reporting_date'] = pd.to_datetime(data['reporting date'])
data['from Wuhan'] = data['from Wuhan'].astype(int)

# Checking the info of the columns
print(data.info())

# Exercise 1 -----------------------------------------------------------------------------
print('Exercise 1')
# Computing the correlations
correlations = data.corr()

# Find correlations with the target variable (outcome)
target_variable = 'death'
target_correlations = correlations[target_variable].dropna().sort_values(ascending=False)

# Print the correlations with the death variable
print("Correlations with the 'death' variable:")
print(target_correlations)

# Print the correlations with the age variable
age_corr = correlations['age'].abs().sort_values(ascending=False)
print("Correlations with the 'age' variable:")
print(age_corr)


# Print the correlations with the recovered variable
recovered_corr = correlations['recovered'].abs().sort_values(ascending=False)
print("Correlations with the 'recovered' variable:")
print(recovered_corr)

# Exercise 2 -----------------------------------------------------------------------------
print(' ')
print('Exercise 2')

#Separate the features from the target variable (if applicable):
X = data.drop(['age'], axis=1)  # Replace 'target_variable' with the actual column name if applicable

# Perform PCA on the features:
pca = PCA(n_components=2)  # Specify the number of components to be retained
X_pca = pca.fit_transform(X)

#Create a scatter plot of the projected dataset:
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.show()


