# Metrics for evaluating the performance of the model

import numpy as np
import matplotlib.pyplot as plt


def accuracy(y, y_pred):
    return np.sum(y == y_pred) / len(y)


def precision(y, y_pred):
    tp = np.sum(y * y_pred)
    fp = np.sum((1 - y) * y_pred)
    return tp / (tp + fp)


def recall(y, y_pred):
    tp = np.sum(y * y_pred)
    fn = np.sum(y * (1 - y_pred))
    return tp / (tp + fn)


def f1_score(y, y_pred):
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return 2 * p * r / (p + r)


def classification_summary(y, y_pred):
    print(f'Accuracy: {accuracy(y, y_pred)}')
    print(f'Precision: {precision(y, y_pred)}')
    print(f'Recall: {recall(y, y_pred)}')
    print(f'F1 Score: {f1_score(y, y_pred)}')
