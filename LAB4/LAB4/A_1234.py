# Imports
import numpy as np
from sklearn import metrics
# Importing Iris data for testing
from sklearn.datasets import load_iris
# Import 4th - KNN
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
Y = iris.target

# Function TNN 
def TNN(X, Y):
    y_pred = []
    error_count = 0
    for i in range(len(X)):
        distances = metrics.pairwise.euclidean_distances(X[i].reshape(1, -1), np.concatenate([X[:i], X[i+1:]]))
        # Calculating Euclidean distances between X[i] and all other points (we're excluding X[i] itself)
        closest_index = np.argmin(distances)
        y_pred.append(Y[closest_index])
        #  Incrementing error_count if Y[closest_index] and Y[i] are differen
        if Y[closest_index] != Y[i]:
            error_count += 1
    # Prediction error rate 
    error_rate = error_count / len(X)
    # Returning y_pred as a numpy array and error_rate
    return np.array(y_pred), error_rate

#Testing on a data - 3th
y_pred, error_rate = TNN(X, Y)
print("Prediction error rate:", error_rate)

#Testing when K = 1 
#knn = KNeighborsClassifier(n_neighbors=1)
#knn.fit(X, Y)
#y_pred_knn = knn.predict(X)
#error_rate_knn = np.mean(y_pred_knn != Y)
#print("Prediction error rate (KNN), K=1 :", error_rate_knn)
#Testing when K = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, Y)
y_pred_knn = knn.predict(X)
error_rate_knn = np.mean(y_pred_knn != Y)
print("Prediction error rate (KNN, K=3):", error_rate_knn) 

