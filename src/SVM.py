import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

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
from src import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Linear SVM
from sklearn.svm import LinearSVC

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'max_iter': [100, 1000, 10000]
              }

grid = GridSearchCV(LinearSVC(), param_grid, refit=True, verbose=2)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)

# Get the best hyperparameters
best_C, best_max_iter = grid.best_params_['C'], grid.best_params_['max_iter']

model = LinearSVC(C=best_C, max_iter=best_max_iter)
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
metrics.confusion_matrix(y_test, y_pred)
metrics.roc_curve(y_test, y_pred)
metrics.precision_recall_curve(y_test, y_pred)
print("Accuracy: ", metrics.accuracy(y_test, y_pred))
print("Precision: ", metrics.precision(y_test, y_pred))
print("Recall: ", metrics.recall(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred))
metrics.classification_summary(y_test, y_pred)
metrics.learning_curve_svm(X_train, y_train, X_test, y_test, best_C, best_max_iter)

plt.subplots(figsize=(8, 8))
plt.title('Predicted Labels')
plt.scatter(X_test[y_pred == 0]['smoothness_mean_log'], X_test[y_pred == 0]['texture_mean_log'], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_test[y_pred == 1]['smoothness_mean_log'], X_test[y_pred == 1]['texture_mean_log'], marker='v',
            label='Malignant', s=100, edgecolors='red', facecolors='red')
plt.scatter(X_test[y_pred != y_test]['smoothness_mean_log'], X_test[y_pred != y_test]['texture_mean_log'], marker='x',
            label='Missclassified', s=100, edgecolors='black', facecolors='black')
plt.xlabel('Log Scale of Smoothness Mean')
plt.ylabel('Log Scale of Texture Mean')
plt.legend()
plt.show()
