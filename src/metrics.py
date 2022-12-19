# Metrics for evaluating the performance of the model

import numpy as np


def accuracy(y_true, y_predicted):
    """
    Function for calculating the accuracy
    Input:
        y_true: True labels
        y_predicted: Predicted labels
    Output:
        accuracy: Accuracy
    """
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    return accuracy


def precision(y_true, y_predicted):
    """
    Function for calculating the precision
    Input:
        y_true: True labels
        y_predicted: Predicted labels
    Output:
        precision: Precision
    """
    true_positives = np.sum(y_true * y_predicted)
    predicted_positives = np.sum(y_predicted)
    precision = true_positives / predicted_positives
    return precision


def recall(y_true, y_predicted):
    """
    Function for calculating the recall
    Input:
        y_true: True labels
        y_predicted: Predicted labels
    Output:
        recall: Recall
    """
    true_positives = np.sum(y_true * y_predicted)
    possible_positives = np.sum(y_true)
    recall = true_positives / possible_positives
    return recall


def f1_score(y_true, y_predicted):
    """
    Function for calculating the F1 score
    Input:
        y_true: True labels
        y_predicted: Predicted labels
    Output:
        f1_score: F1 score
    """
    p = precision(y_true, y_predicted)
    r = recall(y_true, y_predicted)
    f1_score = 2 * p * r / (p + r)
    return f1_score


def confusion_matrix(y_true, y_predicted):
    """
    Function for calculating the confusion matrix
    Input:
        y_true: True labels
        y_predicted: Predicted labels
    Output:
        confusion_matrix: Confusion matrix
    """
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_predicted[i]] += 1
    return confusion_matrix


def auc(false_positive_rate, true_positive_rate):
    """
    Function for calculating the AUC
    Input:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
    Output:
        auc: AUC
    """
    return np.trapz(true_positive_rate, false_positive_rate)


def false_positive_rate(y, y_pred):
    """
    Function for calculating the false positive rate
    Input:
        y: True labels
        y_pred: Predicted labels
    Output:
        false_positive_rate: False positive rate
    """
    false_positive_rate = np.sum((y == 0) & (y_pred == 1)) / np.sum(y == 0)
    return false_positive_rate


def true_positive_rate(y, y_pred):
    """
    Function for calculating the true positive rate
    Input:
        y: True labels
        y_pred: Predicted labels
    Output:
        true_positive_rate: True positive rate
    """
    true_positive_rate = np.sum((y == 1) & (y_pred == 1)) / np.sum(y == 1)
    return true_positive_rate



