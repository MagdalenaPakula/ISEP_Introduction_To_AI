# Before doing the exercises 
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('COVID19_line_list_data_new.csv') 

# Display the dataset
print(data.head())
print(data.info())
print(data.describe())

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
data['reporting date'] = pd.to_datetime(data['reporting date'])
data['summary'] = data['summary'].astype(str)
data['country'] = data['country'].astype(str)
data['symptom'] = data['symptom'].astype(str)

# Checking the info of the columns
print(data.info())

# Exercise 1 -----------------------------------------------------------------------------
print(' ')
print('Exercise 1A')

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

# Print the correlations with the age variable
age_corr = correlations['age'].abs().sort_values(ascending=False)
print("Correlations with the 'age' variable:")
print(age_corr)


# Print the correlations with the recovered variable
recovered_corr = correlations['recovered'].abs().sort_values(ascending=False)
print("Correlations with the 'recovered' variable:")
print(recovered_corr)

# Exercise 1B -----------------------------------------------------------------------------
print(' ')
print('Exercise 1B')

# importing necessary libraries
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
#data = pd.read_csv('COVID19_line_list_data_new.csv') 

# Extract the required columns from the dataset:
selected_columns = ["age", "visiting Wuhan", "from Wuhan", "death", "recovered"]
df = data[selected_columns]

# Remove rows with missing values
df = df.dropna()  

print("Original Dataset:")
print(df.head())

# Create a PCA object with 2 components
pca = PCA(n_components=2)  
principal_components = pca.fit_transform(df)
principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])


print("\nPrincipal Components:")
print(principal_df.head())

#Plot the dataset using scatter:
plt.scatter(principal_df["PC1"], principal_df["PC2"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot")
plt.show()

