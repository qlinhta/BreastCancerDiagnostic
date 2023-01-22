import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set style for plots as latex style
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

from src import LogisticRegression, ReduceDimension, metrics

df = pd.read_csv('../dataset/breast-cancer-wisconsin.csv')

# Drop missing values and convert the diagnosis column to 0 and 1
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df = df.dropna()
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split the data with stratified sampling
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Reduce the dimension of the dataset with PCA and t-SNE
X_pca = ReduceDimension.pca(X, number_of_components=10)
X_tsne = ReduceDimension.tsne(X, number_of_components=2)

# Plot the dataset with PCA and t-SNE
plt.subplots(figsize=(8, 8))
plt.title('Dataset with PCA')
plt.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], marker='v',
            label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

plt.subplots(figsize=(8, 8))
plt.title('Dataset with t-SNE')
plt.scatter(X_tsne[y == 0][:, 0], X_tsne[y == 0][:, 1], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_tsne[y == 1][:, 0], X_tsne[y == 1][:, 1], marker='v',
            label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

# Split the data with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

learning_rates = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
max_iters = [100, 400, 500, 1000, 1500, 2000]

best_learning_rate, best_max_iter, best_accuracy = LogisticRegression.cross_validation_lr(X_train, y_train,
                                                                                          learning_rates, max_iters,
                                                                                          k=10, verbose=True)

model = LogisticRegression.LogisticRegression(learning_rate=best_learning_rate, max_iter=best_max_iter, verbose=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics.classification_summary(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)
metrics.roc_curve(y_test, y_pred)
metrics.precision_recall_curve(y_test, y_pred)
metrics.loss_curve(model.losses)
metrics.accuracy_curve(model.accuracies)
metrics.learning_curve_lr(X_train, y_train, X_test, y_test, best_learning_rate, best_max_iter)

# Plot the test data with predicted labels, if misclassified then plot with red cross
plt.subplots(figsize=(8, 8))
plt.title('Test data with predicted labels')
plt.scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], marker='v',
            label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
plt.scatter(X_test[y_pred != y_test][:, 0], X_test[y_pred != y_test][:, 1], marker='x',
            label='Misclassified', s=100, edgecolors='black', facecolors='black')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

# Print best hyperparameters
print(f'Best learning rate: {best_learning_rate}')
print(f'Best max iter: {best_max_iter}')

