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
from LDA import LDA
import metrics
from IPython.display import display
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
import LogisticRegression, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/haanh88/ML-PSL/dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('/home/haanh88/ML-PSL/dataset/breast-cancer-wisconsin-processed-label.csv')
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

#Fit a LinearDiscriminantAnalysis model
model = LDA()
model.fit(X_train, y_train)


y_pred = (model.predict(X_test))
print("Accuracy: ", metrics.accuracy(y_test, y_pred))
print("Precision: ", metrics.precision(y_test, y_pred))
print("Recall: ", metrics.recall(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred))
metrics.learning_curve_lda(X_train, y_train, X_test, y_test)
metrics.confusion_matrix(y_test, y_pred)
metrics.precision_recall_curve(y_test, y_pred)
#Predict with normal multivariate probability
"""
model2.fit(X_train, y_train)
mean = model.means(X_train, y_train)
mask0 = y_train == 0
X_y0 = X_train[mask0]
cov0 = model.general_cov(X_y0, y_train[mask0])

mask1 = y_train == 1
X_y1 = X_train[mask1]
cov1 = model.general_cov(X_y1, y_train[mask1])

y_pred2 = model.predict_proba2(X_test, mean,cov0,cov1, model.priors)

print("Accuracy of the 2nd approche: ", metrics.accuracy(y_test, y_pred2))
print("Precision of the 2nd approche: ", metrics.precision(y_test, y_pred2))
print("Recall of the 2nd approche: ", metrics.recall(y_test, y_pred2))
print("F1 of the 2nd approche: ", metrics.f1_score(y_test, y_pred2))
metrics.learning_curve_lda(X_train, y_train, X_test, y_test)

"""

misclassified = X_test[y_pred != y_test]

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                                   class_names=['Benign', 'Malignant'],
                                                   discretize_continuous=True, verbose=True, mode='classification')
for i in misclassified.index:
    exp = explainer.explain_instance(X_test.loc[i].values, model.predict_proba_to_plot, num_features=10)
    exp.show_in_notebook(show_table=True, show_all=True)

#Log scale of smoothness mean1
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].set_title('Probability of being Benign')
ax[0].scatter(X_test[y_pred == 0]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred == 0])[:, 0], marker='o',
              label='Benign', s=100, edgecolors='blue', facecolors='white')
ax[0].scatter(X_test[y_pred == 1]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred == 1])[:, 0], marker='v',
              label='Malignant', s=100, edgecolors='red', facecolors='red')
ax[0].scatter(X_test[y_pred != y_test]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred != y_test])[:, 0],
              marker='x', label='Misclassified', s=100, edgecolors='black', facecolors='black')
ax[0].set_xlabel('Log Scale of Smoothness Mean')
ax[0].set_ylabel('Probability of being Benign')
ax[0].legend()
ax[1].set_title('Probability of being Malignant')

ax[1].scatter(X_test[y_pred == 0]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred == 0])[:, 1], marker='o',
              label='Benign', s=100, edgecolors='blue', facecolors='white')
ax[1].scatter(X_test[y_pred == 1]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred == 1])[:, 1], marker='v',
              label='Malignant', s=100, edgecolors='red', facecolors='red')
ax[1].scatter(X_test[y_pred != y_test]['smoothness_mean_log'], model.predict_proba_to_plot(X_test[y_pred != y_test])[:, 1],
              marker='x', label='Misclassified', s=100, edgecolors='black', facecolors='black')
ax[1].set_xlabel('Log Scale of Smoothness Mean')
ax[1].set_ylabel('Probability of being Malignant')
ax[1].legend()
plt.show()

#Misclassified
plt.subplots(figsize=(8, 8))
plt.title('Predicted Labels')
plt.scatter(X_test[y_pred == 0]['smoothness_mean_log'], X_test[y_pred == 0]['texture_mean_log'], marker='o',
            label='Benign', s=100, edgecolors='blue', facecolors='white')
plt.scatter(X_test[y_pred == 1]['smoothness_mean_log'], X_test[y_pred == 1]['texture_mean_log'], marker='v',
            label='Malignant', s=100, edgecolors='red', facecolors='red')
plt.scatter(X_test[y_pred != y_test]['smoothness_mean_log'], X_test[y_pred != y_test]['texture_mean_log'], marker='x',
            label='Misclassified', s=100, edgecolors='black', facecolors='black')
plt.xlabel('Log Scale of Smoothness Mean')
plt.ylabel('Log Scale of Texture Mean')
plt.legend()
#plt.savefig('../src/output_plots/LR_predicted_labels.png')
plt.show()

plt.show()

#LDA true vs predicted labels
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
#plt.savefig('../src/output_plots/LR_true_vs_predicted_labels.png')
plt.show()