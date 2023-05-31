# Before doing the exercises
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

# for column in data.columns:
# label_encoder = LabelEncoder()
# data[column] = label_encoder.fit_transform(data[column].astype(str))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# Select the relevant features and outcome variables
features = ['case_in_country', 'age', 'visiting Wuhan', 'from Wuhan', 'gender', 'symptom_onset']  # Add more features as needed
outcomes = ['death', 'recovered', 'outcome_variable1', 'outcome_variable2']  # Add more outcomes as needed

# Label encode the 'gender' column
label_encoder = LabelEncoder()
data['gender_encoded'] = label_encoder.fit_transform(data['gender'])

for outcome in outcomes:
    print('Outcome:', outcome)
    # Drop rows with missing values in the selected features and outcome
    data_filtered = data.dropna(subset=features + [outcome])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_filtered[features], data_filtered[outcome],
                                                        test_size=0.2, random_state=42)

    # Create and train the K-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
    knn.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print('---------------------------------------')

