import pandas as pd
from src import LogisticRegression, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tuning the hyperparameters
learning_rates = [0.001, 0.01, 0.1, 1, 5, 10]
max_iters = [100, 200, 300, 400, 500, 1000]
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

# Print best hyperparameters
print(f'Best learning rate: {best_learning_rate}')
print(f'Best max iter: {best_max_iter}')
print(f'Best accuracy: {best_accuracy}')
