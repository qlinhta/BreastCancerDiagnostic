import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LogisticRegressionBinary import _cv_tuning, hyperparameter_tuning, MyLogisticRegression

sns.set_theme(style="darkgrid")
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Read the data
df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Plot
plt.title('Dataset')
plt.scatter(df.iloc[:, 0], df.iloc[:, 4], c=label['diagnosis'], cmap='nipy_spectral')
plt.xlabel('Mean radius')
plt.ylabel('Mean texture')
plt.show()

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Plot percentage of each class in the training set and the test set
fig, ax = plt.subplots(figsize=(15, 10))
plt.title('Percentage of each class in the training set and the test set')
plt.bar(['Training set', 'Test set'],
        [y_train.value_counts()[0] / len(y_train), y_test.value_counts()[0] / len(y_test)], label='Benign')
plt.bar(['Training set', 'Test set'],
        [y_train.value_counts()[1] / len(y_train), y_test.value_counts()[1] / len(y_test)],
        bottom=[y_train.value_counts()[0] / len(y_train), y_test.value_counts()[0] / len(y_test)], label='Malignant')
plt.legend()
plt.show()

'''list_of_learning_rates = [0.001, 0.01, 0.1, 1, 5, 10]
list_of_number_of_iterations = [100, 500, 1000]
best_learning_rate, best_number_of_iterations, best_accuracy = _cv_tuning(X_train, y_train,
                                                                          list_of_learning_rates,
                                                                          list_of_number_of_iterations, 10)'''
best_learning_rate, best_number_of_iterations = 5, 1000
print('Best learning rate:', best_learning_rate)
print('Best number of iterations:', best_number_of_iterations)

# Model
model = MyLogisticRegression(epoch=best_number_of_iterations, learning_rate=best_learning_rate, random_state=42,
                             verbose=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.loss_auc()
model.roc_curve(y_test, y_pred)
model.confusion_matrix(y_test, y_pred)
model.classification_report(y_test, y_pred)
scores, avg_score = model._cv(X_train, y_train, 10)
print('Cross-validation scores:', scores)
print('Average cross-validation score:', avg_score)

# Learning curve
model._cv_learning_curve(X_train, y_train, X_test, y_test, 10)
