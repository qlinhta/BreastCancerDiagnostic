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
from src import LogisticRegression, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

plt.subplots(figsize=(8, 8))
plt.title('Dataset')
plt.scatter(X_train[y_train == 0]['smoothness_mean_log'], X_train[y_train == 0]['texture_mean_log'], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_train[y_train == 1]['smoothness_mean_log'], X_train[y_train == 1]['texture_mean_log'], marker='v',
            label='Malignant', s=100, edgecolors='red', facecolors='red')
plt.xlabel('Log Scale of Smoothness Mean')
plt.ylabel('Log Scale of Texture Mean')
plt.legend()
plt.show()

# Tuning the hyperparameters
learning_rates = [0.001, 0.01, 0.1, 1, 5, 10]
max_iters = [100, 200, 400, 500, 1000, 1500]

'''best_learning_rate, best_max_iter, best_accuracy = LogisticRegression.cross_validation_lr(X_train, y_train,
                                                                                          learning_rates, max_iters,
                                                                                          k=10, verbose=True)'''
best_learning_rate, best_max_iter = 5, 1000
model = LogisticRegression.LogisticRegression(learning_rate=best_learning_rate, max_iter=best_max_iter, verbose=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics.classification_summary(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)
metrics.roc_curve(y_test, y_pred)
metrics.precision_recall_curve(y_test, y_pred)
metrics.loss_curve(model.losses)
metrics.accuracy_curve(model.accuracies)
print("Accuracy: ", metrics.accuracy(y_test, y_pred))
print("Precision: ", metrics.precision(y_test, y_pred))
print("Recall: ", metrics.recall(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred))
metrics.learning_curve_lr(X_train, y_train, X_test, y_test, best_learning_rate, best_max_iter)

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

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].set_title('True Labels')
ax[0].scatter(X_test[y_test == 0]['smoothness_mean_log'], X_test[y_test == 0]['texture_mean_log'], marker='o',
              label='Benign', s=100, edgecolors='blue', facecolors='white')
ax[0].scatter(X_test[y_test == 1]['smoothness_mean_log'], X_test[y_test == 1]['texture_mean_log'], marker='v',
              label='Malignant', s=100, edgecolors='green', facecolors='green')
ax[0].set_xlabel('Log Scale of Smoothness Mean')
ax[0].set_ylabel('Log Scale of Texture Mean')
ax[0].legend()

ax[1].set_title('Predicted Labels')
ax[1].scatter(X_test[y_pred == 0]['smoothness_mean_log'], X_test[y_pred == 0]['texture_mean_log'], marker='o',
              label='Benign', s=100, edgecolors='red', facecolors='white')
ax[1].scatter(X_test[y_pred == 1]['smoothness_mean_log'], X_test[y_pred == 1]['texture_mean_log'], marker='v',
              label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
ax[1].set_xlabel('Log Scale of Smoothness Mean')
ax[1].set_ylabel('Log Scale of Texture Mean')
ax[1].legend()
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        ax[1].scatter(X_test.iloc[i]['smoothness_mean_log'], X_test.iloc[i]['texture_mean_log'], marker='x',
                      label='Incorrect', s=100, edgecolors='black', facecolors='black')
plt.show()

# Investigate the misclassified points
misclassified = X_test[y_test != y_pred]
misclassified['diagnosis'] = y_test[y_test != y_pred]
misclassified['predicted_diagnosis'] = y_pred[y_test != y_pred]
print(misclassified)

# Print best hyperparameters
print(f'Best learning rate: {best_learning_rate}')
print(f'Best max iter: {best_max_iter}')
