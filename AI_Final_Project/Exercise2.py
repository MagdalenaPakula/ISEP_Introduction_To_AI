# Before doing the exercises
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('COVID19_line_list_data.csv')

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

# Exercise 2A -----------------------------------------------------------------------------
print(' ')
print('Exercise 2A')

# Compute the probability
wuhan_visitors = len(data[data['visiting Wuhan'] == 1])
symptomatic_wuhan_visitors = len(data[(data['symptom_onset'].notna()) & (data['visiting Wuhan'] == 1)])

probability_symptoms_given_wuhan = symptomatic_wuhan_visitors / wuhan_visitors
print("Probability of symptoms given visiting Wuhan:", probability_symptoms_given_wuhan)

# Exercise 2B -----------------------------------------------------------------------------
print(' ')
print('Exercise 2B')

# Compute the probability
symptomatic_cases = len(data[data['symptom_onset'].notna()])
true_positive_cases = len(data[(data['symptom_onset'].notna()) & (data['visiting Wuhan'] == 1)])

probability_true_patient_given_symptoms_and_wuhan = true_positive_cases / symptomatic_cases
print("Probability of being a true patient given symptoms and visiting Wuhan:",
      probability_true_patient_given_symptoms_and_wuhan)
# Exercise 2C -----------------------------------------------------------------------------
print(' ')
print('Exercise 2C')

# Filter the data for patients who visited Wuhan
wuhan_visitors = data[data['visiting Wuhan'] == 1]

# Calculate the total number of deaths among Wuhan visitors
deaths = wuhan_visitors['death'].notnull().sum()

# Calculate the total number of Wuhan visitors
total_visitors = len(wuhan_visitors)

# Calculate the probability of death given visiting Wuhan
probability_death_given_wuhan = deaths / total_visitors
print("Probability of death given visiting Wuhan:", probability_death_given_wuhan)
print("Probability of death given visiting Wuhan in percentage: {:.2%}".format(probability_death_given_wuhan))

# Exercise 2D -----------------------------------------------------------------------------
print(' ')
print('Exercise 2D')

# Filter the data for patients who visited Wuhan and have recovery dates available
wuhan_patients = data.loc[(data['visiting Wuhan'] == 1) & data['recovered'].notnull()].copy()

# Convert recovery dates to datetime
wuhan_patients.loc[:, 'recovery_date'] = pd.to_datetime(wuhan_patients['recovered'])
wuhan_patients.loc[:, 'symptom_onset'] = pd.to_datetime(wuhan_patients['symptom_onset'])

# Calculate the recovery interval for each patient
wuhan_patients.loc[:, 'recovery_interval'] = wuhan_patients['recovery_date'] - wuhan_patients['symptom_onset']

# Remove any outliers or invalid values
wuhan_patients = wuhan_patients[wuhan_patients['recovery_interval'] >= pd.Timedelta(0)]

# Calculate the average recovery interval
total_recovery_interval = wuhan_patients['recovery_interval'].sum()
num_patients = len(wuhan_patients)

# Check if there are patients who visited Wuhan and have recovery dates available
if num_patients > 0:
    average_recovery_interval = total_recovery_interval / num_patients
    print("Average recovery interval for patients who visited Wuhan:", average_recovery_interval)
else:
    print("No patients who visited Wuhan with available recovery dates found.")
