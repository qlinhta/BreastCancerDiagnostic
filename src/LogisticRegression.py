import numpy as np
import matplotlib.pyplot as plt

from src import metrics


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

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _losses(y, y_pred):
        loss = np.zeros(y.shape[0])
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        np.random.seed(self.random_state)  # Set the random seed
        self.bias = np.random.randn()
        self.weights = np.random.randn(X.shape[1])

        # Start training
        for i in range(self.max_iter):
            # Forward propagation
            # TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'
            # This error is because the weights and bias are not initialized
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
            # Print the loss and the accuracy
            if self.verbose:
                print(
                    f'Iteration: {i + 1}/{self.max_iter}, loss: {self.losses[-1]:.4f}, accuracy: {self.accuracies[-1]:.4f}')

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(y_pred)

    def predict_proba(self, X):
        # Make result return as sklearn
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.array([1 - y_pred, y_pred]).T # Return the probability of 0 and 1


def cross_validation_lr(X, y, learning_rates, max_iters, k=10, verbose=True):
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)

    best_accuracy = 0
    best_learning_rate = None
    best_max_iter = None

    for learning_rate in learning_rates:
        for max_iter in max_iters:
            model = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, verbose=verbose)
            accuracies = []
            for i in range(k):
                # Get the training data
                X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
                y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
                X_val = X_folds[i]
                y_val = y_folds[i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                accuracies.append(metrics.accuracy(y_val, y_pred))

            accuracy = np.mean(accuracies)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = learning_rate
                best_max_iter = max_iter

    return best_learning_rate, best_max_iter, best_accuracy
