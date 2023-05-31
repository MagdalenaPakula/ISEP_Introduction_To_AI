# Before doing the exercises
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

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

# Exercise 3 -----------------------------------------------------------------------------
print(' ')
print('Exercise 3 (n_neighbors=3)')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

# Select the relevant features and outcome variable
features = ['case_in_country', 'age', 'visiting Wuhan', 'from Wuhan']  # Replace with the actual feature names
outcome = 'death'  # Replace with the actual outcome variable name

# Drop rows with missing values in the selected features and outcome
data_filtered = data.dropna(subset=features + [outcome])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_filtered[features], data_filtered[outcome], test_size=0.2, random_state=42)

# Create and train the K-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)

# We can analyze the performance of the model and propose solutions for improvement.
#
# Accuracy: The model achieves a high accuracy of approximately 96.24%, indicating that it correctly predicts the outcome for the majority of the cases. However, accuracy alone may not be sufficient for evaluating the model's performance, especially in the presence of imbalanced classes.
#
# Confusion Matrix: The confusion matrix provides a detailed breakdown of the model's predictions. It shows that the model correctly predicts 196 instances of non-death cases (true negatives) and 9 instances of death cases (true positives). However, there are 2 false positives and 6 false negatives.
#
# Precision: The precision score of 0.818 indicates that when the model predicts a death case, it is correct approximately 81.81% of the time. Precision measures the proportion of true positives out of all positive predictions.
#
# Recall: The recall score of 0.6 indicates that the model correctly identifies 60% of the actual death cases. Recall, also known as sensitivity or true positive rate, measures the proportion of true positives out of all actual positive cases.
#
# F1 Score: The F1 score, which combines precision and recall into a single metric, is 0.692. It provides a balance between precision and recall and is useful when the dataset has class imbalance.
#
# To ameliorate the results, considering the imbalance in the classes, you can try the following solutions:
#
# Class Balancing: Since the death cases might be underrepresented in the dataset compared to non-death cases, you can explore techniques for balancing the classes, such as oversampling the minority class (death) or undersampling the majority class (non-death). This can help the model better learn patterns related to death cases.
#
# Feature Engineering: Evaluate the existing features or explore additional features that might provide more discriminatory power between death and non-death cases. Consider domain knowledge or consult with experts to identify relevant features that capture important information.
#
# Hyperparameter Tuning: Experiment with different values of hyperparameters in the K-NN classifier, such as the number of neighbors (k), to find the optimal configuration for improved performance.
#
# Try Different Algorithms: Explore other classification algorithms apart from K-NN, such as decision trees, random forests, or support vector machines. Different algorithms may have different strengths and weaknesses, and one of them might be better suited for this particular problem.
#
# Collect More Data: If possible, collecting more data, especially for the minority class (death), can help improve the model's performance by providing a more balanced representation of the classes.
#
# After implementing and evaluating these solutions, you should re-evaluate the model's performance using the suggested external indexes (Confusion matrix, Accuracy, Recall, Precision, F1 Score) to assess if the results have been ameliorated.

# Exercise 3A -----------------------------------------------------------------------------
print(' ')
print('Exercise 3A')
from imblearn.over_sampling import SMOTE

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

# Create and train the K-NN classifier with oversampled data
knn_oversampled = KNeighborsClassifier(n_neighbors=3)
knn_oversampled.fit(X_train_oversampled, y_train_oversampled)

# Make predictions on the testing set
y_pred_oversampled = knn_oversampled.predict(X_test)

# Evaluate the model with oversampling
confusion_mat_oversampled = confusion_matrix(y_test, y_pred_oversampled)
accuracy_oversampled = accuracy_score(y_test, y_pred_oversampled)
recall_oversampled = recall_score(y_test, y_pred_oversampled)
precision_oversampled = precision_score(y_test, y_pred_oversampled)
f1_oversampled = f1_score(y_test, y_pred_oversampled)

# Print the evaluation metrics with oversampling
print("Confusion Matrix (with oversampling):")
print(confusion_mat_oversampled)
print("Accuracy (with oversampling):", accuracy_oversampled)
print("Recall (with oversampling):", recall_oversampled)
print("Precision (with oversampling):", precision_oversampled)
print("F1 Score (with oversampling):", f1_oversampled)

# Certainly! One of the best solutions to ameliorate the results is to implement class balancing techniques. In this case, since the death cases are the minority class, we can apply oversampling to increase the representation of the death cases in the dataset. This will help the model learn more effectively from the available data.
# Here's an example of how you can implement oversampling using the Synthetic Minority Oversampling Technique (SMOTE) in the code:
# In this code, we first import the SMOTE class from the imblearn library, which provides implementations of various oversampling techniques. We then create an instance of SMOTE and apply it to the training data (X_train and y_train) using the fit_resample method, which performs oversampling.
#
# Next, we create a new K-NN classifier (knn_oversampled) and train it on the oversampled training data. We then make predictions on the testing set (X_test) using the oversampled classifier.
#
# Finally, we evaluate the model's performance using the evaluation metrics (confusion matrix, accuracy, recall, precision, and F1 score) calculated based on the predictions from the oversampled model.
#
# By implementing this oversampling technique, we can address the class imbalance issue and potentially improve the model's performance in predicting death cases.

import warnings
from sklearn.ensemble import RandomForestClassifier

# Method 2: Resampling Techniques - Undersampling the Majority Class
print(' ')
print('Exercise 3A - FINAL TOUCH')

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Oversample the minority class using SMOTE
oversampler = SMOTE()
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7],
    'metric': ['euclidean', 'manhattan']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1')
grid_search.fit(X_train_oversampled, y_train_oversampled)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)


