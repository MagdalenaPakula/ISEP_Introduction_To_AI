# Before doing the exercises
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('COVID19_line_list_data.csv')

# Display the dataset
print(data.head())  # displays first few rows
print(data.info())  # displays columns and Dtypes!!
print(data.describe())  # displays count,mean,std,min etc

# Check for missing values
print("Missing values:")
print(data.isnull().sum())  # important!!

# Replace missing values with appropriate strategies
data['case_in_country'].fillna(data['case_in_country'].mean(), inplace=True)
data['reporting date'].fillna(data['reporting date'].mode()[0], inplace=True)
data['gender'].fillna(data['gender'].mode()[0], inplace=True)
data['age'].fillna(data['age'].mean(), inplace=True)
data['symptom_onset'].fillna(data['symptom_onset'].mode()[0], inplace=True)
data['hosp_visit_date'].fillna(data['hosp_visit_date'].mode()[0], inplace=True)
data['exposure_start'].fillna(data['exposure_start'].mode()[0], inplace=True)
data['exposure_end'].fillna(data['exposure_end'].mode()[0], inplace=True)
data['symptom'].fillna(data['symptom'].mode()[0], inplace=True)

# Changing types of variables
# Convert 'reporting date' column to datetime data type
data['reporting date'] = pd.to_datetime(data['reporting date'], format='%d-%m-%Y', errors='coerce')
data['from Wuhan'] = data['from Wuhan'].fillna(0).astype(int)
data['death'] = pd.to_numeric(data['death'], errors='coerce').astype('Int64')
data['recovered'] = pd.to_numeric(data['recovered'], errors='coerce').astype('Int64')
data['symptom_onset'] = pd.to_datetime(data['symptom_onset'], format='%d-%m-%Y', errors='coerce')
data['hosp_visit_date'] = pd.to_datetime(data['hosp_visit_date'], format='%d-%m-%Y', errors='coerce')
data['exposure_end'] = pd.to_datetime(data['exposure_end'], format='%d-%m-%Y', errors='coerce')
data['exposure_start'] = pd.to_datetime(data['exposure_start'], format='%d-%m-%Y', errors='coerce')
data['gender'] = data['gender'].astype(str)
data['location'] = data['location'].astype(str)
data['visiting Wuhan'] = data['visiting Wuhan'].astype(int)
data['from Wuhan'] = data['from Wuhan'].astype(int)
data['reporting date'] = pd.to_datetime(data['reporting date'])
data['summary'] = data['summary'].astype(str)
data['country'] = data['country'].astype(str)
data['symptom'] = data['symptom'].astype(str)

print("Missing values:")
print(data.isnull().sum())

# Encoder only for the object type to string
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column].astype(str))

# Exercise 1 -----------------------------------------------------------------------------
print(' ')
print('Exercise 1A')

# Select only numerical columns
numerical_columns = data.select_dtypes(include=np.number)

# Computing the correlations
correlations = numerical_columns.corr()

# Find correlations with the target variable (outcome)
target_variable = 'death'
target_correlations = correlations[target_variable].dropna().sort_values(ascending=False)

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

# Exercise 1B -----------------------------------------------------------------------------
print(' ')
print('Exercise 1B')

# importing necessary libraries
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
# data = pd.read_csv('COVID19_line_list_data.csv')

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

# Plot the dataset using scatter:
plt.scatter(principal_df["PC1"], principal_df["PC2"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot")
plt.show()  # for showing the plot
