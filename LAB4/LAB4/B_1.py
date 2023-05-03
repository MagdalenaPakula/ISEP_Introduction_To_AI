from sklearn import datasets
import numpy as np

def CBN(X, Y):
    # Calculating the number of classes
    num_classes = len(np.unique(Y))
    
    # Initializing the barycenters 
    barycenters = []
    for i in range(num_classes):
        barycenters.append(np.mean(X[Y == i], axis=0))
    
    # Initializing the prior probabilities
    prior_probs = []
    for i in range(num_classes):
        prior_probs.append(np.sum(Y == i) / len(Y))
    
    # Initializing the predicted labels list
    pred_labels = []
    
    # Calculating the conditional probabilities for each class and each variable
    for data in X:
        # Initializing the probabilities list for each class
        class_probs = []
        
        for i in range(num_classes):
            # Calculating the Gaussian probability density function for each variable
            var_probs = []
            for j in range(len(data)):
                var_prob = (1 / np.sqrt(2 * np.pi * np.var(X[:, j][Y == i]))) * np.exp(-((data[j] - barycenters[i][j]) ** 2) / (2 * np.var(X[:, j][Y == i])))
                var_probs.append(var_prob)
            
            # Calculating the product of the Gaussian probabilities for each variable
            var_probs_product = np.prod(var_probs)
            
            # Calculating the conditional probability for the class
            class_prob = var_probs_product * prior_probs[i]
            class_probs.append(class_prob)
        
        # Assigning the predicted label as the class with the highest conditional probability
        pred_label = np.argmax(class_probs)
        pred_labels.append(pred_label)
    
    error = np.sum(pred_labels != Y) / len(Y)
    return pred_labels, error

# Iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Testing
pred_labels, error = CBN(X, Y)
print("CBN Predicted Labels: ", pred_labels)
print("CBN Prediction Error: ", error)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X, Y)

sklearn_predictions = gnb.predict(X)
print("Gaussian: ", sklearn_predictions)
