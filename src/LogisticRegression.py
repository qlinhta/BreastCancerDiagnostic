import numpy as np
import matplotlib.pyplot as plt

from src import metrics


class LogisticRegression:
    def __init__(self, learning_rate, max_iter, f_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.f_intercept = f_intercept
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.intercept = None

        self.losses = []
        self.accuracies = []
        self.weights_list = []
        self.bias_list = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _losses(y, y_pred):
        return (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()

    @staticmethod
    def _accuracy(y, y_pred):
        return np.sum(y == y_pred) / len(y)

    def fit(self, X, y):
        if self.f_intercept:
            X = self._add_intercept(X)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = np.dot(X.T, (y_pred - y)) / y.size
            db = np.sum(y_pred - y) / y.size

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose:
                if i % self.max_iter == 0:
                    print(f'Loss: {self._losses(y, y_pred)} \tAccuracy: {self._accuracy(y, y_pred)}')

            self.losses.append(self._losses(y, y_pred))
            self.accuracies.append(self._accuracy(y, y_pred))
            self.weights_list.append(self.weights)
            self.bias_list.append(self.bias)

    def predict(self, X):
        if self.f_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.weights) + self.bias)


def cross_validation_lr(X, y, learning_rates, max_iters, model, k=5, verbose=False):
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)

    best_accuracy = 0
    best_learning_rate = None
    best_max_iter = None

    for learning_rate in learning_rates:
        for max_iter in max_iters:
            accuracies = []
            for i in range(k):
                # Get the training data
                X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
                y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
                X_val = X_folds[i]
                y_val = y_folds[i]
                model.fit(X_train, y_train, learning_rate, max_iter, verbose)
                y_pred = model.predict(X_val)
                accuracies.append(metrics.accuracy(y_val, y_pred))

            accuracy = np.mean(accuracies)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = learning_rate
                best_max_iter = max_iter

    return best_learning_rate, best_max_iter
