import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors

def k_fold_cross_validation(Y, X, C,  k=10):
    """Performs cross-validation using the provided training function."""

    # Size of the data.
    N = Y.shape[0]

    # Counts the number of good predictions - the number of bad predictions.
    correct_prediction = 0

    for i in range(k):
        # Get the training set.
        indices_train = np.where(np.arange(N) % k != i)
        tX_train = X[indices_train]
        y_train = Y[indices_train]

        # Get the validation set.
        indices_test = np.where(np.arange(N) % k == i)
        tX_test = X[indices_test]
        y_test = Y[indices_test]

        # Create the classifier and train it
        n_neighbors = 3
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', p=1)
        clf.fit(tX_train, y_train)

        # Train on the training set.
        # Added : we use a SVM
        #svmClassifier = svm.SVC(C = C, gamma="auto", class_weight="balanced", kernel='rbf')
        #svmClassifier.fit(tX_train, y_train)
        
        # Predict
        Z = clf.predict(tX_test)
        
        correct_prediction += sum(Z == y_test)
        

    # Return the percentage of good predictions.
    return correct_prediction / (1.0 * N)
