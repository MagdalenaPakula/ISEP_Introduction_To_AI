import pandas as pd

# Load the dataset
data = pd.read_csv('COVID19_line_list_data_new.csv')
# Rename the columns
data = data.rename(columns={'visiting Wuhan': 'visiting_Wuhan', 'from Wuhan': 'from_Wuhan'})

# Print the column names
print(data.columns)
# Exercise 2A -----------------------------------------------------------------------------
print(' ')
print('Exercise 2A')

import pandas as pd

# Compute the probability
total_cases = len(data)
wuhan_visitors = len(data[data['visiting_Wuhan'] == 1])
symptomatic_cases = len(data[data['symptom_onset'].notna()])

probability_symptoms_given_wuhan = symptomatic_cases / wuhan_visitors
print("Probability of symptoms given visiting Wuhan:", probability_symptoms_given_wuhan)

# Exercise 2B -----------------------------------------------------------------------------
print(' ')
print('Exercise 2B')

# Compute the probability
symptomatic_cases = len(data[data['symptom_onset'].notna()])
true_positive_cases = len(data[(data['symptom_onset'].notna()) & (data['visiting_Wuhan'] == 1)])

probability_true_patient_given_symptoms_and_wuhan = true_positive_cases / symptomatic_cases
print("Probability of being a true patient given symptoms and visiting Wuhan:", probability_true_patient_given_symptoms_and_wuhan)
# Exercise 2C -----------------------------------------------------------------------------
print(' ')
print('Exercise 2C')

# Compute the probability
wuhan_visitors = len(data[data['visiting_Wuhan'] == 1])
deaths = len(data[(data['visiting_Wuhan'] == 1) & (data['death'] == 'death')])

probability_death_given_wuhan = deaths / wuhan_visitors
print("Probability of death given visiting Wuhan:", probability_death_given_wuhan)
# Exercise 2D -----------------------------------------------------------------------------
print(' ')
print('Exercise 2D')
# Filter the data for patients who visited Wuhan and have recovery dates available

# Filter the data for patients who visited Wuhan and have recovery dates available
wuhan_patients = data[(data['visiting_Wuhan'] == 1) & (data['recovered'] == 'recovered')]

# Convert recovery dates to datetime
wuhan_patients['recovery_date'] = pd.to_datetime(wuhan_patients['recovered'])
wuhan_patients['symptom_onset'] = pd.to_datetime(wuhan_patients['symptom_onset'])

# Calculate the recovery interval for each patient
wuhan_patients['recovery_interval'] = wuhan_patients['recovery_date'] - wuhan_patients['symptom_onset']

# Calculate the average recovery interval
average_recovery_interval = wuhan_patients['recovery_interval'].mean()

print("Average recovery interval for patients who visited Wuhan:", average_recovery_interval)