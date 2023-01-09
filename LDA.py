import numpy as np
from math import *


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None  # store the eigenvector that we compute

    def means(self, X, y):
        classes = np.unique(y)
        means = np.zeros((classes.shape[0], X.shape[1]))
        for i in range(classes.shape[0]):
            class_idx = np.flatnonzero(y == i)
            means[i, :] = np.mean(X[class_idx], axis=0)
        return means

    def pi_k(self, X, y):
        classes = np.unique(y)
        pi_k = np.zeros((len(classes),))
        for c in classes:
            elm = y[y == c]
            pi_k[c] = len(elm) / X.shape[0]
        return pi_k

    def general_cov(self, X, y):
        classes = np.unique(y)
        for c in classes:
            class_idx = np.flatnonzero(y == c)
            sigma = len(X[class_idx]) * np.cov(X[class_idx].T)
        return sigma / X.shape[0]

    def gaussian_proba(self, x, class_x):
        """
         a = np.exp(-1/2*(x-mean).T*np.linalg.inv(cov)*(x-mean))
         b = ((2*np.pi)**(d/2))*(np.absolute(cov)**1/2)
         return a/b"""

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # S_W(),S_B(between classes)
        mean = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            class_idx = np.flatnonzero(y == c)
            X_c = X[class_idx]
            mean_c = np.mean(X_c, axis=0)
            # (4,n_c)*(n_c,4) = (4,4)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]
            # (4,1)*(4,1)T = (4,4)
            mean_diff = (mean_c - mean).reshape(n_features, 1)  # (4,1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        # eigenvalues
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idx = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

    def predict(self, X):
        pass

    def decision_boundary(self, X, y):  # x.Tw + b = 0

        classes = np.unique(y)
        means_overall = self.means(X, y)
        pi_overral = self.pi_k(X, y)
        substract_mean = means_overall[0]
        log = pi_overral[0]
        sigma_inv = np.linalg.inv(self.general_cov(X, y))

        for i in range(1, classes.shape[0]):
            substract_mean = substract_mean - means_overall[i]
            log = log / pi_overral[i]

        # weight
        weight = sigma_inv @ substract_mean

        # b
        p = substract_mean.T @ sigma_inv @ substract_mean
        b = -1 / 2 * p + np.log(log)

        return np.dot(X, weight) + b


