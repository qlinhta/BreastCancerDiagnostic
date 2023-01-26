# Create a pipeline that train multiple classifiers and compare their performance
# path: src/pipeline.py

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
from catboost import CatBoostClassifier
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from tqdm import tqdm

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
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, \
    recall_score, f1_score, auc

df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
# 1. Train multiple classifiers
# 2. Compare their performance
# 3. Save the best model to src/output_models

# 1. Train multiple classifiers
classifiers = {
    "Logistic Regression": LogisticRegression.LogisticRegression(learning_rate=5, max_iter=1000, verbose=True),
    "LinearSVM": LinearSVC(C=1, max_iter=100),
    "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=100),
    "AdaBoost": AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
    "CatBoost": CatBoostClassifier(learning_rate=0.1, depth=13, iterations=500),
}

# Progress bar for training with tqdm

for name, clf in tqdm(classifiers.items()):
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'output_models/{name}.pkl')

# 2. Compare their performance
"""
For each classifier, we will compute the following metrics:
- Accuracy
- Precision
- Recall
- F1 score
- AUC
- ROC curve
"""

# Compute the metrics
metrics_dict = {}

for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    metrics_dict[name] = {
        'accuracy': metrics.accuracy(y_test, y_pred),
        'precision': metrics.precision(y_test, y_pred),
        'recall': metrics.recall(y_test, y_pred),
        'f1_score': metrics.f1_score(y_test, y_pred),
        'auc': metrics.auc(y_test, y_pred_proba),
    }

# Plot the metrics
metrics_df = pd.DataFrame(metrics_dict).T

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_xlabel('Classifier')
ax.set_title('Metrics')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()