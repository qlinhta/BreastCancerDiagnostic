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
    from sklearn.metrics import classification_report
    print('Classification report:')
    print(classification_report(y, y_pred))


def roc_curve(y, y_pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()


def precision_recall_curve(y, y_pred):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y, y_pred)
    average_precision = average_precision_score(y, y_pred)
    plt.figure(figsize=(10, 10))
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()


def confusion_matrix(y, y_pred):
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    tp = np.sum((y == 1) & (y_pred == 1))
    cm = np.array([[tn, fp], [fn, tp]])
    print('Confusion matrix:')
    print(cm)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=20)
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.show()
