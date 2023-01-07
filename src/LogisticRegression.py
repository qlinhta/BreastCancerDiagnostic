import numpy as np
import matplotlib.pyplot as plt


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
                if i % 100 == 0:
                    print(f'Loss: {self._losses(y, y_pred)} \tAccuracy: {self._accuracy(y, y_pred)}')

            self.losses.append(self._losses(y, y_pred))
            self.accuracies.append(self._accuracy(y, y_pred))
            self.weights_list.append(self.weights)
            self.bias_list.append(self.bias)

    def predict(self, X):
        if self.f_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
