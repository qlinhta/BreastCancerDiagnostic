import numpy as np
import metrics


class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.n_classes = 2
        self.linear_discriminants = None  # store the eigenvector that we compute
        self.coef = np.zeros((self.n_classes,))
        self.intercept = 0.0


    def means(self,X, y):
        classes = np.unique(y)
        means = np.zeros((classes.shape[0], X.shape[1]))
        for i in range(classes.shape[0]):
            means[i, :] = np.mean(X[y == i], axis=0)
        return means


    def prob_k(self, X, y):
        classes = np.unique(y)
        pi_k = np.zeros((len(classes),))
        for c in classes:
            elm = y[y == c]
            pi_k[c] = len(elm) / X.shape[0]
        return pi_k


    def general_cov(self, X, y):
        classes = np.unique(y)
        sigma = np.zeros((X.shape[1], X.shape[1]))
        for c in classes:
            sigma = sigma + len(X[y == c]) * np.cov(X[y == c].T)
        return sigma / X.shape[0]

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            n_c = X_c.shape[0]

            mean_diff = mean_c - mean
            S_B += n_c * mean_diff.dot(mean_diff.T)

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
        scores = self.decision_boundary(X)
        y_predicted = [1 if i > 0 else 0 for i in scores]
        return np.array(y_predicted)

    def set_coef_intercept(self, X, y):
        classes = np.unique(y)
        means_overall = self.means(X, y)
        pi_overall = self.prob_k(X, y)
        sigma_inv = np.linalg.inv(self.general_cov(X, y))

        # coef
        self.coef = sigma_inv @ (means_overall[1] - means_overall[0])

        # intercept
        p = (means_overall[1] - means_overall[0]) @ sigma_inv @ (means_overall[0] + means_overall[1])
        self.intercept = (-1 / 2 * p - np.log(pi_overall[0] / pi_overall[1]))

    def decision_boundary(self, X):  # x.Tw + b = 0
        return X @ self.coef.T + self.intercept

    def cross_validation_lda(self, X, y, k):
        X_folds = np.array_split(X, k)
        y_folds = np.array_split(y, k)
        model = LDA(self.n_components)
        model.set_coef_intercept(X, y)
        accuracies = []
        for i in range(k):
            # Get the training data
            X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
            y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
            X_val = X_folds[i]
            y_val = y_folds[i]
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_val)
            accuracies.append(metrics.accuracy(y_val, y_predicted))
        return np.mean(accuracies)