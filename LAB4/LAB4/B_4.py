from sklearn import datasets
import numpy as np

# Importing Iris data for testing
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Y = iris.target

def CBN_gaussian(X, Y):
    # Calculate class barycenters and a priori probabilities
    classes = np.unique(Y)
    barycenters = []
    priors = []
    for c in classes:
        mask = (Y == c)
        class_data = X[mask]
        barycenter = np.mean(class_data, axis=0)
        barycenters.append(barycenter)
        priors.append(len(class_data) / len(X))
    
    # Calculate conditional probabilities using Gaussian distribution
    predictions = []
    num_errors = 0
    for i in range(len(X)):
        data = X[i]
        probs = []
        for j in range(len(classes)):
            barycenter = barycenters[j]
            std_dev = np.std(X, axis=0)
            likelihood = np.exp(-0.5 * ((data - barycenter) ** 2) / (std_dev ** 2))
            probs.append(np.prod(likelihood))
        probs = np.asarray(probs)
        posterior = priors * probs
        prediction = classes[np.argmax(posterior)]
        predictions.append(prediction)
        if prediction != Y[i]:
            num_errors += 1
    
    error_rate = num_errors / len(X)
    print("Error rate: ", error_rate)
    return predictions

predictions_gaussian = CBN_gaussian(X, Y)
print("Bonus Gaussian: ", predictions_gaussian)

