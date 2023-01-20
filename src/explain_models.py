import joblib
import dalex as dx
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lime.lime_tabular
import lime.lime_image
import warnings

from torch.fx.experimental.unification import variables

sys.path.append('../')

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

# Load model
model = joblib.load('output_models/LR_model.pkl')

# Load data
df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

shape_lr_explainer = shap.KernelExplainer(model.predict_proba, X_train)
shap_values = shape_lr_explainer.shap_values(X_test, nsamples=100)
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.savefig('../src/output_plots/LR_shap_summary_plot.png')
plt.show()

# Plot the shap decision plot
shap.decision_plot(shape_lr_explainer.expected_value[1], shap_values[1], X_test, show=False)
plt.title('Decision process plot')
plt.savefig('../src/output_plots/LR_shap_decision_plot.png')
plt.show()

# Plot dependence plot for 4 most important features
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
shap.dependence_plot('smoothness_mean_log', shap_values[1], X_test, show=False, ax=ax[0, 0])
shap.dependence_plot('texture_mean_log', shap_values[1], X_test, show=False, ax=ax[0, 1])
shap.dependence_plot('area_mean', shap_values[1], X_test, show=False, ax=ax[1, 0])
shap.dependence_plot('concavity_mean', shap_values[1], X_test, show=False, ax=ax[1, 1])
plt.savefig('../src/output_plots/LR_shap_dependence_plot.png')
plt.show()

# Create explainer object dalex
dx_lr_explainer = dx.Explainer(model, X_train, y_train, label='Logistic Regression')
print(dx_lr_explainer.model_performance())
print(dx_lr_explainer.model_parts())
dx_lr_explainer.model_parts().plot()

sample1 = X_test.iloc[14]
sample2 = X_test.iloc[15]
dx_lr_explainer.predict_parts(sample1).plot()
dx_lr_explainer.predict_parts(sample2).plot()

# Get misclassified samples from test set and plot them as prediction parts
misclassified = X_test[y_test != model.predict(X_test)]
misclassified = misclassified.reset_index(drop=True)

for i in range(5):
    dx_lr_explainer.predict_parts(misclassified.iloc[i]).plot()
    plt.show()

mp = dx_lr_explainer.model_performance(model_type='classification')
mp.plot()