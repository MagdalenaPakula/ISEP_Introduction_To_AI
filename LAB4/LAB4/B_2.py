from sklearn import datasets
import numpy as np

def CBN(X, Y):
    classes = np.unique(Y)
    barycenters = []
    priors = []
    for c in classes:
        mask = (Y == c)
        class_data = X[mask]
        barycenter = np.mean(class_data, axis=0)
        barycenters.append(barycenter)
        priors.append(len(class_data) / len(X))
    
    # Calculate conditional probabilities
    predictions = []
    num_errors = 0
    for i in range(len(X)):
        data = X[i]
        probs = []
        for j in range(len(classes)):
            barycenter = barycenters[j]
            dist = np.linalg.norm(data - barycenter)
            probs.append(dist)
        probs = np.asarray(probs)
        posterior = priors * np.exp(-0.5 * probs ** 2)
        prediction = classes[np.argmax(posterior)]
        predictions.append(prediction)
        if prediction != Y[i]:
            num_errors += 1
    
    error_rate = num_errors / len(X)
    print("Error rate: ", error_rate)
    return predictions

iris = datasets.load_iris()
X = iris.data
Y = iris.target

predictions = CBN(X, Y)
print("Predictions:", predictions)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X, Y)

sklearn_predictions = gnb.predict(X)
print("Gaussian", sklearn_predictions)
