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

    def predict_proba(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def predict_log_proba(self, X_test):
        return np.log(self.predict_proba(X_test))

    def cross_validation(self, X_train, y_train, k=10):
        cross_validation = []
        avg = 0
        # Split the data into k folds
        X_train = np.array_split(X_train, k)
        y_train = np.array_split(y_train, k)
        # Cross validation
        for i in range(k):
            X_train_ = np.concatenate(X_train[:i] + X_train[i + 1:])
            y_train_ = np.concatenate(y_train[:i] + y_train[i + 1:])
            X_test_ = X_train[i]
            y_test_ = y_train[i]
            self.fit(X_train_, y_train_)
            print('Fold {}: {}'.format(i, self.score(X_test_, y_test_)))
            cross_validation.append(self.score(X_test_, y_test_))
            avg += self.score(X_test_, y_test_)
        print('Average: {}'.format(avg / k))
        return cross_validation


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

    plt.subplots(figsize=(8, 8))
    plt.title('Predicted Labels')
    plt.scatter(X_test[y_pred == 0]['smoothness_mean_log'], X_test[y_pred == 0]['texture_mean_log'], marker='o',
                label='Benign', s=100, edgecolors='red', facecolors='white')
    plt.scatter(X_test[y_pred == 1]['smoothness_mean_log'], X_test[y_pred == 1]['texture_mean_log'], marker='v',
                label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
    plt.scatter(X_test[y_pred != y_test]['smoothness_mean_log'], X_test[y_pred != y_test]['texture_mean_log'],
                marker='x',
                label='Misclassified', s=100, edgecolors='black', facecolors='black')
    plt.xlabel('Log Scale of Smoothness Mean')
    plt.ylabel('Log Scale of Texture Mean')
    plt.legend()
    plt.show()

    misclassified = X_test[y_pred != y_test]

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                                       class_names=['Benign', 'Malignant'],
                                                       discretize_continuous=True, verbose=True, mode='classification')
    for i in misclassified.index:
        exp = explainer.explain_instance(X_test.loc[i].values, lda.predict_proba, num_features=10)
        exp.show_in_notebook(show_table=True, show_all=True)
        exp.save_to_file('lda_missed_predict_investigate/' + str(i) + '.html')

    # Cross validation
    lda.cross_validation(X_train, y_train)
