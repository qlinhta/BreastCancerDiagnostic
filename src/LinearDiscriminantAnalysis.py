from scipy.linalg import svd
import dalex as dx
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lime.lime_tabular
import lime.lime_image
import warnings

import shap
from matplotlib.colors import ListedColormap

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)

warnings.filterwarnings('ignore')
from src import LogisticRegression, metrics
from sklearn.model_selection import train_test_split

"""
Linear Discriminant Analysis classifies

Input:
    X: training data
    y: training label
    k: number of classes
    verbose: print the information or not
Output:
    w: weight vector
    b: bias
    mu: mean of each class
    sigma: covariance matrix
"""


class LinearDiscriminantAnalysis(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None

    def fit(self, X_train, y_train):
        self.mu = np.zeros((X_train.shape[1], len(np.unique(y_train))))
        self.sigma = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i, label in enumerate(np.unique(y_train)):
            X_train_label = X_train[y_train == label]
            self.mu[:, i] = np.mean(X_train_label, axis=0)
            self.sigma += np.cov(X_train_label.T)
        self.sigma /= len(np.unique(y_train))
        self.w = np.dot(np.linalg.inv(self.sigma), self.mu)
        self.b = -0.5 * np.sum(np.dot(self.mu.T, np.dot(np.linalg.inv(self.sigma), self.mu)), axis=1)
        if self.verbose:
            print('Weight Vector: {}'.format(self.w))
            print('Bias: {}'.format(self.b))

    def predict(self, X_test):
        return np.argmax(np.dot(X_test, self.w) + self.b, axis=1)

    def score(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / len(y_test)


if __name__ == '__main__':
    df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
    label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
    data = pd.concat([df, label], axis=1)

    # Split the data with stratified sampling
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    lda = LinearDiscriminantAnalysis(verbose=True)
    lda.fit(X_train, y_train)
    print('Training Accuracy: {}'.format(lda.score(X_train, y_train)))
    print('Testing Accuracy: {}'.format(lda.score(X_test, y_test)))

    y_pred = lda.predict(X_test)
    metrics.classification_summary(y_test, y_pred)
    metrics.confusion_matrix(y_test, y_pred)
    metrics.roc_curve(y_test, y_pred)
    metrics.precision_recall_curve(y_test, y_pred)
    print("Accuracy: ", metrics.accuracy(y_test, y_pred))
    print("Precision: ", metrics.precision(y_test, y_pred))
    print("Recall: ", metrics.recall(y_test, y_pred))
    print("F1: ", metrics.f1_score(y_test, y_pred))

