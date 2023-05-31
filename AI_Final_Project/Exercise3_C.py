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

# Exercise 3C -----------------------------------------------------------------------------
print(' ')
print('Exercise 3C')

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Select the relevant features for clustering
features = ['case_in_country', 'age', 'visiting Wuhan', 'from Wuhan']  # Replace with actual features

# Drop rows with missing values in the selected features
data_cluster = data.dropna(subset=features)

# Standardize the data
scaler = StandardScaler()
data_cluster_scaled = scaler.fit_transform(data_cluster[features])

# Initialize variables to store the Silhouette scores
best_silhouette_score = -1
best_num_clusters = -1

# Try different number of clusters and compute the Silhouette score for each
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data_cluster_scaled)
    silhouette_avg = silhouette_score(data_cluster_scaled, labels)

    # Update the best Silhouette score and number of clusters if necessary
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters

# Perform K-means clustering with the best number of clusters
kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
labels = kmeans.fit_predict(data_cluster_scaled)

# Plot the clustering results
plt.scatter(data_cluster_scaled[:, 0], data_cluster_scaled[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering Results ({} clusters)'.format(best_num_clusters))
plt.show()

# Print the best number of clusters and corresponding Silhouette score
print("Best number of clusters:", best_num_clusters)
print("Silhouette score:", best_silhouette_score)

