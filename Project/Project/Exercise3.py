import pandas as pd

# Load the dataset
data = pd.read_csv('COVID19_line_list_data_new.csv')


# Exercise 3 -----------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Preprocess the data
# Drop unnecessary columns
data = data.drop(["id", "reporting date", "summary", "location", "country", "source", "link"], axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
data["gender"] = label_encoder.fit_transform(data["gender"])

# Convert dates to datetime format
date_columns = ["symptom_onset", "hosp_visit_date", "exposure_start", "exposure_end"]
for column in date_columns:
    data[column] = pd.to_datetime(data[column], errors="coerce")

# Calculate days between dates
data["symptom_duration"] = (data["hosp_visit_date"] - data["symptom_onset"]).dt.days
data["exposure_duration"] = (data["exposure_end"] - data["exposure_start"]).dt.days

# Select features and target variable
X = data[["gender", "age", "symptom_duration", "exposure_duration"]]
y = data["death"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Confusion Matrix:")
print(confusion_matrix)
print("Accuracy:", accuracy)