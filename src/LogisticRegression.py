import numpy as np
import matplotlib.pyplot as plt

import metrics


class LogisticRegression:
    def __init__(self, learning_rate, max_iter, verbose=False, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.weights = None
        self.bias = None

        self.losses = []
        self.accuracies = []
        self.weights_list = []
        self.bias_list = []
        self.coef_ = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _losses(y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        np.random.seed(self.random_state)  # Set the random seed

        if self.weights is None:
            self.weights = np.random.randn(X.shape[1])
        if self.bias is None:
            self.bias = np.random.randn()

        # Start training
        for i in range(self.max_iter):
            # Forward propagation
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

            # Compute the loss
            self.losses.append(self._losses(y, y_pred))
            # Compute the accuracy
            self.accuracies.append(metrics.accuracy(y, self.predict(X)))

            # Backward propagation
            dw = np.dot(X.T, (y_pred - y)) / y.shape[0]
            db = np.sum(y_pred - y) / y.shape[0]

            # Update the weights and the bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Update coef_
            self.coef_ = np.append(self.bias, self.weights)

            # Print the loss and the accuracy
            if self.verbose:
                print(
                    f'Iteration: {i + 1}/{self.max_iter}, loss: {self.losses[-1]:.4f}, accuracy: {self.accuracies[-1]:.4f}')

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model has not been trained yet")
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(y_pred)

    def predict_proba(self, X):
        if self.weights is None:
            raise Exception("Model has not been trained yet")
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(np.array([1 - y_pred, y_pred]).T, 2)

    def cross_validation(self, X, y, n_splits=10):
        # Split the dataset into n_splits
        X_split = np.array_split(X, n_splits)
        y_split = np.array_split(y, n_splits)
        accuracy_list = []
        # Start the cross validation
        for i in range(n_splits):
            # Get the test set
            X_test = X_split[i]
            y_test = y_split[i]
            # Get the train set
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            # Train the model
            self.fit(X_train, y_train)
            # Get the accuracy
            accuracy = metrics.accuracy(y_test, self.predict(X_test))
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)


def _tuning(X, y, learning_rates, max_iters, k=10, verbose=True):
    assert len(X) == len(y), "Need to have same number of samples for X and y"
    assert k > 0, "k needs to be positive"
    assert k < len(X), "k needs to be less than number of samples"
    assert k == int(k), "k needs to be an integer"
    assert len(learning_rates) > 0, "Need to have at least one learning rate"
    assert len(max_iters) > 0, "Need to have at least one max iter"

    # Tuning hyperparameters using cross validation
    best_accuracy = 0
    best_learning_rate = None
    best_max_iter = None
    for learning_rate in learning_rates:
        for max_iter in max_iters:
            model = LogisticRegression(learning_rate, max_iter, verbose=False, random_state=42)
            accuracy_list = model.cross_validation(X, y, k)
            accuracy = np.mean(accuracy_list)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = learning_rate
                best_max_iter = max_iter
            if verbose:
                print(
                    f'Learning rate: {learning_rate}, max iter: {max_iter}, accuracy: {accuracy:.4f}')
    return best_learning_rate, best_max_iter, best_accuracy
