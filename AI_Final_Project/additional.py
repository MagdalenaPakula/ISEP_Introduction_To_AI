# Before doing the exercises
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
# data['reporting date'] = pd.to_datetime(data['reporting date'], format='%d-%m-%Y', errors='coerce')
# data['from Wuhan'] = data['from Wuhan'].fillna(0).astype(int)
# data['death'] = pd.to_numeric(data['death'], errors='coerce').astype('Int64')
# data['recovered'] = pd.to_numeric(data['recovered'], errors='coerce').astype('Int64')
# data['symptom_onset'] = pd.to_datetime(data['symptom_onset'], format='%d-%m-%Y', errors='coerce')
# data['hosp_visit_date'] = pd.to_datetime(data['hosp_visit_date'], format='%d-%m-%Y', errors='coerce')
# data['exposure_end'] = pd.to_datetime(data['exposure_end'], format='%d-%m-%Y', errors='coerce')
# data['exposure_start'] = pd.to_datetime(data['exposure_start'], format='%d-%m-%Y', errors='coerce')
# data['gender'] = data['gender'].astype(str)
# data['location'] = data['location'].astype(str)
# data['visiting Wuhan'] = data['visiting Wuhan'].astype(int)
# data['from Wuhan'] = data['from Wuhan'].astype(int)
# data['reporting date'] = pd.to_datetime(data['reporting date'])
# data['summary'] = data['summary'].astype(str)
# data['country'] = data['country'].astype(str)
# data['symptom'] = data['symptom'].astype(str)

# Encoder only for the object type to string
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column].astype(str))

# Exercise 3B -----------------------------------------------------------------------------
print(' ')
print('Exercise 3B')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select the explanatory variables (features)
explanatory_vars = ['case_in_country', 'visiting Wuhan', 'from Wuhan', 'death', 'recovered']

# Drop rows with missing values in the selected variables
data_filtered_regression = data.dropna(subset=explanatory_vars + ['age'])

# Split the data into training and testing sets
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    data_filtered_regression[explanatory_vars], data_filtered_regression['age'],
    test_size=0.2, random_state=42
)

# Create and train the Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train_regression, y_train_regression)

# Make predictions on the testing set
y_pred_regression = regression_model.predict(X_test_regression)

# Compute the Mean Squared Error (MSE) for the predictions
mse = mean_squared_error(y_test_regression, y_pred_regression)

# Create a DataFrame to store the actual and predicted ages
results = pd.DataFrame({'Actual Age': y_test_regression, 'Predicted Age': y_pred_regression})

# Print the MSE and the results
print("Mean Squared Error (MSE):", mse)
print(results)
