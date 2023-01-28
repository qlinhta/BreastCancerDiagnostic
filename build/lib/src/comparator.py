import eli5
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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

# Split the data with stratified sampling
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LinearSVM
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, \
    recall_score, f1_score, auc

'''
params_svm = {
    'C': [0.001, 0.01, 0.1, 1, 5, 10],
    'max_iter': [100, 200, 400, 500, 1000, 1500]
}
svm = LinearSVC()
svm_grid = GridSearchCV(svm, params_svm, cv=10, verbose=True)
svm_grid.fit(X_train, y_train)
svm_best_c = svm_grid.best_params_['C']
svm_best_max_iter = svm_grid.best_params_['max_iter']
print(f'Best C: {svm_best_c}')
'''
svm_model = LinearSVC(C=1, max_iter=100)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# XGBoost
from xgboost import XGBClassifier

'''
params_xgb = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
    'max_depth': [3, 5, 7, 9, 11, 13],
    'n_estimators': [100, 200, 400, 500, 1000, 1500]
}
xgb = XGBClassifier()
xgb_grid = GridSearchCV(xgb, params_xgb, cv=10, verbose=True)
xgb_grid.fit(X_train, y_train)

xgb_best_learning_rate = xgb_grid.best_params_['learning_rate']
xgb_best_max_depth = xgb_grid.best_params_['max_depth']
xgb_best_n_estimators = xgb_grid.best_params_['n_estimators']
'''
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

'''
params_ada = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
    'n_estimators': [100, 200, 400, 500, 1000, 1500]
}
ada = AdaBoostClassifier()
ada_grid = GridSearchCV(ada, params_ada, cv=10, verbose=True)
ada_grid.fit(X_train, y_train)

ada_best_learning_rate = ada_grid.best_params_['learning_rate']
ada_best_n_estimators = ada_grid.best_params_['n_estimators']
'''
ada_model = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
ada_model.fit(X_train, y_train)
ada_y_pred = ada_model.predict(X_test)

# CatBoost
from catboost import CatBoostClassifier

'''
params_cb = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
    'depth': [3, 5, 7, 9, 11, 13],
    'iterations': [100, 200, 400, 500, 1000, 1500]
}
cb = CatBoostClassifier()
cb_grid = GridSearchCV(cb, params_cb, cv=10, verbose=True)
cb_grid.fit(X_train, y_train)
cb_best_learning_rate = cb_grid.best_params_['learning_rate']
cb_best_depth = cb_grid.best_params_['depth']
cb_best_iterations = cb_grid.best_params_['iterations']
'''
cat_model = CatBoostClassifier(learning_rate=0.1, depth=13, iterations=500)
cat_model.fit(X_train, y_train)
cat_y_pred = cat_model.predict(X_test)

# Print the results
print(f'LinearSVM Accuracy: {accuracy_score(y_test, svm_y_pred)}')
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_y_pred)}')
print(f'AdaBoost Accuracy: {accuracy_score(y_test, ada_y_pred)}')
print(f'CatBoost Accuracy: {accuracy_score(y_test, cat_y_pred)}')

# Print accuracy, precision, recall, f1-score
print(f'LinearSVM Accuracy: {accuracy_score(y_test, svm_y_pred)}')
print(f'LinearSVM Precision: {precision_score(y_test, svm_y_pred)}')
print(f'LinearSVM Recall: {recall_score(y_test, svm_y_pred)}')
print(f'LinearSVM F1-score: {f1_score(y_test, svm_y_pred)}')

print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_y_pred)}')
print(f'XGBoost Precision: {precision_score(y_test, xgb_y_pred)}')
print(f'XGBoost Recall: {recall_score(y_test, xgb_y_pred)}')
print(f'XGBoost F1-score: {f1_score(y_test, xgb_y_pred)}')

print(f'AdaBoost Accuracy: {accuracy_score(y_test, ada_y_pred)}')
print(f'AdaBoost Precision: {precision_score(y_test, ada_y_pred)}')
print(f'AdaBoost Recall: {recall_score(y_test, ada_y_pred)}')
print(f'AdaBoost F1-score: {f1_score(y_test, ada_y_pred)}')

print(f'CatBoost Accuracy: {accuracy_score(y_test, cat_y_pred)}')
print(f'CatBoost Precision: {precision_score(y_test, cat_y_pred)}')
print(f'CatBoost Recall: {recall_score(y_test, cat_y_pred)}')
print(f'CatBoost F1-score: {f1_score(y_test, cat_y_pred)}')

# For each algorithm, plot predicted labels and true labels
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].set_title('LinearSVM')
ax[0, 0].scatter(X_test[svm_y_pred == 0]['smoothness_mean_log'], X_test[svm_y_pred == 0]['texture_mean_log'],
                 marker='o',
                 label='Benign', s=100, edgecolors='red', facecolors='white')
ax[0, 0].scatter(X_test[svm_y_pred == 1]['smoothness_mean_log'], X_test[svm_y_pred == 1]['texture_mean_log'],
                 marker='v',
                 label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
ax[0, 0].scatter(X_test[y_test != svm_y_pred]['smoothness_mean_log'], X_test[y_test != svm_y_pred]['texture_mean_log'],
                 marker='x',
                 label='Incorrect', s=100, edgecolors='black', facecolors='black')
ax[0, 0].set_xlabel('Log Scale of Smoothness Mean')
ax[0, 0].set_ylabel('Log Scale of Texture Mean')
ax[0, 0].legend()

ax[0, 1].set_title('XGBoost')
ax[0, 1].scatter(X_test[xgb_y_pred == 0]['smoothness_mean_log'], X_test[xgb_y_pred == 0]['texture_mean_log'],
                 marker='o',
                 label='Benign', s=100, edgecolors='red', facecolors='white')
ax[0, 1].scatter(X_test[xgb_y_pred == 1]['smoothness_mean_log'], X_test[xgb_y_pred == 1]['texture_mean_log'],
                 marker='v',
                 label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
ax[0, 1].scatter(X_test[y_test != xgb_y_pred]['smoothness_mean_log'], X_test[y_test != xgb_y_pred]['texture_mean_log'],
                 marker='x',
                 label='Incorrect', s=100, edgecolors='black', facecolors='black')
ax[0, 1].set_xlabel('Log Scale of Smoothness Mean')
ax[0, 1].set_ylabel('Log Scale of Texture Mean')
ax[0, 1].legend()

ax[1, 0].set_title('AdaBoost')
ax[1, 0].scatter(X_test[ada_y_pred == 0]['smoothness_mean_log'], X_test[ada_y_pred == 0]['texture_mean_log'],
                 marker='o',
                 label='Benign', s=100, edgecolors='red', facecolors='white')
ax[1, 0].scatter(X_test[ada_y_pred == 1]['smoothness_mean_log'], X_test[ada_y_pred == 1]['texture_mean_log'],
                 marker='v',
                 label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
ax[1, 0].scatter(X_test[y_test != ada_y_pred]['smoothness_mean_log'], X_test[y_test != ada_y_pred]['texture_mean_log'],
                 marker='x',
                 label='Incorrect', s=100, edgecolors='black', facecolors='black')
ax[1, 0].set_xlabel('Log Scale of Smoothness Mean')
ax[1, 0].set_ylabel('Log Scale of Texture Mean')
ax[1, 0].legend()

ax[1, 1].set_title('CatBoost')
ax[1, 1].scatter(X_test[cat_y_pred == 0]['smoothness_mean_log'], X_test[cat_y_pred == 0]['texture_mean_log'],
                 marker='o',
                 label='Benign', s=100, edgecolors='red', facecolors='white')
ax[1, 1].scatter(X_test[cat_y_pred == 1]['smoothness_mean_log'], X_test[cat_y_pred == 1]['texture_mean_log'],
                 marker='v',
                 label='Malignant', s=100, edgecolors='darkorange', facecolors='darkorange')
ax[1, 1].scatter(X_test[y_test != cat_y_pred]['smoothness_mean_log'], X_test[y_test != cat_y_pred]['texture_mean_log'],
                 marker='x',
                 label='Incorrect', s=100, edgecolors='black', facecolors='black')
ax[1, 1].set_xlabel('Log Scale of Smoothness Mean')
ax[1, 1].set_ylabel('Log Scale of Texture Mean')
ax[1, 1].legend()
# Save plot to src/output_plots
plt.savefig('output_plots/Prediction_others.png')

plt.show()

# For each algorithm, plot ROC curve
'''
plt.plot(fpr, tpr, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    '''
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_y_pred)
roc_auc_svm = auc(fpr_svm, tpr_svm)
ax[0, 0].set_title('LinearSVM')
ax[0, 0].plot(fpr_svm, tpr_svm, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc_svm)
ax[0, 0].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax[0, 0].set_xlabel('False Positive Rate')
ax[0, 0].set_ylabel('True Positive Rate')
ax[0, 0].legend()

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_y_pred)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
ax[0, 1].set_title('XGBoost')
ax[0, 1].plot(fpr_xgb, tpr_xgb, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc_xgb)
ax[0, 1].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax[0, 1].set_xlabel('False Positive Rate')
ax[0, 1].set_ylabel('True Positive Rate')
ax[0, 1].legend()

fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_y_pred)
roc_auc_ada = auc(fpr_ada, tpr_ada)
ax[1, 0].set_title('AdaBoost')
ax[1, 0].plot(fpr_ada, tpr_ada, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc_ada)
ax[1, 0].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax[1, 0].set_xlabel('False Positive Rate')
ax[1, 0].set_ylabel('True Positive Rate')
ax[1, 0].legend()

fpr_cat, tpr_cat, _ = roc_curve(y_test, cat_y_pred)
roc_auc_cat = auc(fpr_cat, tpr_cat)
ax[1, 1].set_title('CatBoost')
ax[1, 1].plot(fpr_cat, tpr_cat, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc_cat)
ax[1, 1].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax[1, 1].set_xlabel('False Positive Rate')
ax[1, 1].set_ylabel('True Positive Rate')
ax[1, 1].legend()
# Save plot to src/output_plots
plt.savefig('output_plots/ROC_others.png')
plt.show()

# For each algorithm, plot confusion matrix
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].set_title('LinearSVM')
ax[0, 0].matshow(confusion_matrix(y_test, svm_y_pred), cmap=plt.cm.Greys, alpha=0.3)
for i in range(confusion_matrix(y_test, svm_y_pred).shape[0]):
    for j in range(confusion_matrix(y_test, svm_y_pred).shape[1]):
        ax[0, 0].text(x=j, y=i, s=confusion_matrix(y_test, svm_y_pred)[i, j], va='center', ha='center')
ax[0, 0].set_xlabel('Predicted label')
ax[0, 0].set_ylabel('True label')
ax[0, 0].tick_params(labelsize=15)

ax[0, 1].set_title('XGBoost')
ax[0, 1].matshow(confusion_matrix(y_test, xgb_y_pred), cmap=plt.cm.Greys, alpha=0.3)
for i in range(confusion_matrix(y_test, xgb_y_pred).shape[0]):
    for j in range(confusion_matrix(y_test, xgb_y_pred).shape[1]):
        ax[0, 1].text(x=j, y=i, s=confusion_matrix(y_test, xgb_y_pred)[i, j], va='center', ha='center')
ax[0, 1].set_xlabel('Predicted label')
ax[0, 1].set_ylabel('True label')
ax[0, 1].tick_params(labelsize=15)

ax[1, 0].set_title('AdaBoost')
ax[1, 0].matshow(confusion_matrix(y_test, ada_y_pred), cmap=plt.cm.Greys, alpha=0.3)
for i in range(confusion_matrix(y_test, ada_y_pred).shape[0]):
    for j in range(confusion_matrix(y_test, ada_y_pred).shape[1]):
        ax[1, 0].text(x=j, y=i, s=confusion_matrix(y_test, ada_y_pred)[i, j], va='center', ha='center')
ax[1, 0].set_xlabel('Predicted label')
ax[1, 0].set_ylabel('True label')
ax[1, 0].tick_params(labelsize=15)

ax[1, 1].set_title('CatBoost')
ax[1, 1].matshow(confusion_matrix(y_test, cat_y_pred), cmap=plt.cm.Greys, alpha=0.3)
for i in range(confusion_matrix(y_test, cat_y_pred).shape[0]):
    for j in range(confusion_matrix(y_test, cat_y_pred).shape[1]):
        ax[1, 1].text(x=j, y=i, s=confusion_matrix(y_test, cat_y_pred)[i, j], va='center', ha='center')
ax[1, 1].set_xlabel('Predicted label')
ax[1, 1].set_ylabel('True label')
ax[1, 1].tick_params(labelsize=15)
# Save plot to src/output_plots
plt.savefig('output_plots/CM_others.png')
plt.show()

# For each algorithm, plot precision-recall curve
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
precision_svm, recall_svm, _ = precision_recall_curve(y_test, svm_y_pred)
ax[0, 0].set_title('LinearSVM')
ax[0, 0].plot(recall_svm, precision_svm, color='black', lw=1, label='Precision-Recall curve')
ax[0, 0].set_xlabel('Recall')
ax[0, 0].set_ylabel('Precision')
ax[0, 0].legend()

precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_y_pred)
ax[0, 1].set_title('XGBoost')
ax[0, 1].plot(recall_xgb, precision_xgb, color='black', lw=1, label='Precision-Recall curve')
ax[0, 1].set_xlabel('Recall')
ax[0, 1].set_ylabel('Precision')
ax[0, 1].legend()

precision_ada, recall_ada, _ = precision_recall_curve(y_test, ada_y_pred)
ax[1, 0].set_title('AdaBoost')
ax[1, 0].plot(recall_ada, precision_ada, color='black', lw=1, label='Precision-Recall curve')
ax[1, 0].set_xlabel('Recall')
ax[1, 0].set_ylabel('Precision')
ax[1, 0].legend()

precision_cat, recall_cat, _ = precision_recall_curve(y_test, cat_y_pred)
ax[1, 1].set_title('CatBoost')
ax[1, 1].plot(recall_cat, precision_cat, color='black', lw=1, label='Precision-Recall curve')
ax[1, 1].set_xlabel('Recall')
ax[1, 1].set_ylabel('Precision')
ax[1, 1].legend()
# Save plot to src/output_plots
plt.savefig('output_plots/PR_others.png')
plt.show()

# For each algorithm, plot learning curve
# Calculation
'''
train_sizes, train_scores_svm, test_scores_svm = learning_curve(svm_model, X_train, y_train, cv=10, scoring='accuracy',
                                                                n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_sizes, train_scores_xgb, test_scores_xgb = learning_curve(xgb_model, X_train, y_train, cv=10, scoring='accuracy',
                                                                n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_sizes, train_scores_nb, test_scores_nb = learning_curve(nb_model, X_train, y_train, cv=10, scoring='accuracy',
                                                              n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_sizes, train_scores_cat, test_scores_cat = learning_curve(cat_model, X_train, y_train, cv=10, scoring='accuracy',
                                                                n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

# Plot
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].set_title('LinearSVM')
ax[0, 0].plot(train_sizes, np.mean(train_scores_svm, axis=1), color='blue', marker='o', markersize=5,
              label='Training accuracy')
ax[0, 0].plot(train_sizes, np.mean(test_scores_svm, axis=1), color='green', linestyle='--', marker='s', markersize=5,
              label='Validation accuracy')
ax[0, 0].set_xlabel('Number of training samples')
ax[0, 0].set_ylabel('Accuracy')
ax[0, 0].legend(loc='lower right')
ax[0, 0].grid()
ax[0, 0].tick_params(labelsize=15)

ax[0, 1].set_title('XGBoost')
ax[0, 1].plot(train_sizes, np.mean(train_scores_xgb, axis=1), color='blue', marker='o', markersize=5,
              label='Training accuracy')
ax[0, 1].plot(train_sizes, np.mean(test_scores_xgb, axis=1), color='green', linestyle='--', marker='s', markersize=5,
              label='Validation accuracy')
ax[0, 1].set_xlabel('Number of training samples')
ax[0, 1].set_ylabel('Accuracy')
ax[0, 1].legend(loc='lower right')
ax[0, 1].grid()
ax[0, 1].tick_params(labelsize=15)

ax[1, 0].set_title('AdaBoost')
ax[1, 0].plot(train_sizes, np.mean(train_scores_nb, axis=1), color='blue', marker='o', markersize=5,
                label='Training accuracy')
ax[1, 0].plot(train_sizes, np.mean(test_scores_nb, axis=1), color='green', linestyle='--', marker='s', markersize=5,
                label='Validation accuracy')
ax[1, 0].set_xlabel('Number of training samples')
ax[1, 0].set_ylabel('Accuracy')
ax[1, 0].legend(loc='lower right')
ax[1, 0].grid()
ax[1, 0].tick_params(labelsize=15)

ax[1, 1].set_title('CatBoost')
ax[1, 1].plot(train_sizes, np.mean(train_scores_cat, axis=1), color='blue', marker='o', markersize=5,
              label='Training accuracy')
ax[1, 1].plot(train_sizes, np.mean(test_scores_cat, axis=1), color='green', linestyle='--', marker='s', markersize=5,
              label='Validation accuracy')
ax[1, 1].set_xlabel('Number of training samples')
ax[1, 1].set_ylabel('Accuracy')
ax[1, 1].legend(loc='lower right')
ax[1, 1].grid()
ax[1, 1].tick_params(labelsize=15)
# Save plot to src/output_plots
plt.savefig('output_plots/LC_others.png')
plt.show()
'''