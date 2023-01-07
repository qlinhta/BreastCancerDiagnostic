import numpy as np
import matplotlib.pyplot as plt


class LDAClassifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.means = np.zeros((self.n_classes, self.n_features))
        self.cov = np.zeros((self.n_features, self.n_features))
        self.priors = np.zeros(self.n_classes)
        self.compute_means()
        self.compute_cov()
        self.compute_priors()

    def compute_means(self):
        for c in self.classes:
            self.means[c] = np.mean(self.X[self.y == c], axis=0)

    def compute_cov(self):
        for c in self.classes:
            self.cov += np.cov(self.X[self.y == c].T)
        self.cov /= self.n_classes

    def compute_priors(self):
        for c in self.classes:
            self.priors[c] = np.sum(self.y == c) / len(self.y)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y_pred[i] = self.predict_one(x)
        return y_pred

    def predict_one(self, x):
        posteriors = np.zeros(self.n_classes)
        for c in self.classes:
            posteriors[c] = self.compute_posterior(x, c)
        return np.argmax(posteriors)

    def compute_posterior(self, x, c):
        prior = np.log(self.priors[c])
        likelihood = self.compute_likelihood(x, c)
        return prior + likelihood

    def compute_likelihood(self, x, c):
        mean = self.means[c]
        cov = self.cov
        x_mu = x - mean
        return -0.5 * (x_mu.T @ np.linalg.inv(cov) @ x_mu + np.log(np.linalg.det(cov)) + self.n_features * np.log(
            2 * np.pi))

    def plot(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()


def main():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, n_features=2, centers=1, cluster_std=1.5, random_state=1)
    lda = LDAClassifier(X, y)
    y_pred = lda.predict(X)
    print('Accuracy: %.2f' % (np.sum(y == y_pred) / len(y)))
    lda.plot()


if __name__ == '__main__':
    main()
