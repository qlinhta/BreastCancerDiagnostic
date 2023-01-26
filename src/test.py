import dalex as dx
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lime.lime_tabular
import lime.lime_image
import warnings
from scipy.stats import multivariate_normal
warnings.filterwarnings('ignore')
import LogisticRegression, metrics
from sklearn.model_selection import train_test_split

import shap
from matplotlib.colors import ListedColormap
from LDA import LDA
import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
df = pd.read_csv('/home/haanh88/ML-PSL/dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('/home/haanh88/ML-PSL/dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modeltest = LinearDiscriminantAnalysis()
modeltest.fit(X_train, y_train)
print("intercep_test ", modeltest.intercept_)
print("coef_test ", modeltest.coef_)
print("cov test," ,modeltest.priors_)
a = modeltest.decision_function(X_train)
print(a)

model = LDA(2)
model.fit(X_train, y_train)
model.transform(X_train)
model.set_coef_intercept(X_train, y_train)
print("---------------------------------------")
print("intercep", model.intercept)
print("coef", model.coef)
print("cov, ", model.prob_k(X_train, y_train))
b= model.decision_boundary(X_train)
print(b)
print("diff," ,a-b)
y_pred = model.predict(X_test)
print(metrics.accuracy(y_test,y_pred))