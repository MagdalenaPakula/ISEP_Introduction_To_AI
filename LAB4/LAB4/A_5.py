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

# Function TNN (modified - 5th BONUS exercise)
def TNN(X, Y, k):
    y_pred = []
    error_count = 0
    for i in range(len(X)):
        distances = metrics.pairwise.euclidean_distances(X[i].reshape(1, -1), X)
        # Calculating Euclidean distances between X[i] and all other points
        closest_indices = np.argsort(distances)[0][:k]
        # Getting the indices of the k closest points
        closest_classes = Y[closest_indices]
        # Getting the classes of the k closest points
        predicted_class = np.argmax(np.bincount(closest_classes))
        # Predicting the class based on the majority vote
        y_pred.append(predicted_class)
        # Incrementing error_count if predicted_class and Y[i] are different
        if predicted_class != Y[i]:
            error_count += 1
    # Prediction error rate 
    error_rate = error_count / len(X)
    # Returning y_pred as a numpy array and error_rate
    return np.array(y_pred), error_rate

# Testing TNN with k=1
y_pred_tnn1, error_rate_tnn1 = TNN(X, Y, k=1)
print("Prediction error rate (TNN with k=1):", error_rate_tnn1)
# Testing TNN with k=3
y_pred_tnn3, error_rate_tnn3 = TNN(X, Y, k=3)
print("Prediction error rate (TNN with k=3):", error_rate_tnn3)
# Testing TNN with k=5
y_pred_tnn5, error_rate_tnn5 = TNN(X, Y, k=5)
print("Prediction error rate (TNN with k=5):", error_rate_tnn5)
