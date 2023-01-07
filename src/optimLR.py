# Logistic Regression with Optima

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import missingno as ms
import matplotlib.pyplot as plt
import warnings

import optima as opt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Read the data
df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
data = pd.concat([df, label], axis=1)