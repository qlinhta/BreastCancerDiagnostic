import numpy as np
import matplotlib.pyplot as plt


def _cv_tuning(X, y, list_of_learning_rates, list_of_number_of_iterations, k):
    """
    This function performs cross validation on the data and returns the best learning rate and the best number of
    iterations.
    :param X: data
    :param y: data labels
    :param list_of_learning_rates: list of learning rates
    :param list_of_number_of_iterations: list of number of iterations
    :param k: number of folds
    :return: best learning rate, best number of iterations
    """
    best_learning_rate = 0
    best_number_of_iterations = 0
    best_accuracy = 0
    for learning_rate in list_of_learning_rates:
        for number_of_iterations in list_of_number_of_iterations:
            # Split the data into k folds
            X_folds = np.array_split(X, k)
            y_folds = np.array_split(y, k)
            accuracy = 0
            for i in range(k):
                # Create the training set and the test set
                X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
                y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
                X_test = X_folds[i]
                y_test = y_folds[i]
                # Train the model
                model = MyLogisticRegression(epoch=number_of_iterations, learning_rate=learning_rate, verbose=False)
                model.fit(X_train, y_train)
                # Predict
                y_pred = model.predict(X_test)
                # Compute accuracy
                accuracy += np.sum(y_pred == y_test) / len(y_test)
            accuracy /= k
            if accuracy > best_accuracy:
                best_learning_rate = learning_rate
                best_number_of_iterations = number_of_iterations
                best_accuracy = accuracy
    return best_learning_rate, best_number_of_iterations, best_accuracy


def hyperparameter_tuning(X_train, y_train, X_test, y_test, list_of_learning_rates, list_of_number_of_iterations):
    """
    This function performs hyperparameter tuning on the training set and returns the best learning rate and the best number
    of iterations.
    :param X_train: training set
    :param y_train: training set labels
    :param X_test: test set
    :param y_test: test set labels
    :param list_of_learning_rates: list of learning rates
    :param list_of_number_of_iterations: list of number of iterations
    :return: best learning rate, best number of iterations
    """
    best_learning_rate = 0
    best_number_of_iterations = 0
    best_accuracy = 0
    for learning_rate in list_of_learning_rates:
        for number_of_iterations in list_of_number_of_iterations:
            # Train the model
            model = MyLogisticRegression(epoch=number_of_iterations, learning_rate=learning_rate)
            model.fit(X_train, y_train)
            # Predict
            y_pred = model.predict(X_test)
            # Compute accuracy
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            if accuracy > best_accuracy:
                best_learning_rate = learning_rate
                best_number_of_iterations = number_of_iterations
                best_accuracy = accuracy
    return best_learning_rate, best_number_of_iterations


class MyLogisticRegression:
    """
    Logistic regression classifier.
    Input:
        epoch: int, number of epochs
        learning_rate: float, learning rate
        random_state: int, random state
        verbose: bool, verbose

    Methods:
        fit: fit the model
        predict: predict the label
        sigmoid: sigmoid function
        loss: loss function
        gradient: gradient function
        accuracy: accuracy function
        confusion_matrix: confusion matrix
        plot_loss: plot the loss function
        plot_accuracy: plot the accuracy function

    Attributes:
        weights: array, weights
        bias: float, bias
        loss: array, loss
        accuracy: array, accuracy
    """

    def __init__(self, epoch, learning_rate, random_state=42, verbose=True):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose

        self.weights = None
        self.bias = None
        self.loss = None
        self.accuracy = None

    def fit(self, X, y):
        """
        Fit the model.
        Input:
            X: array, features
            y: array, labels
        """
        # Initialize the weights and the bias
        np.random.seed(self.random_state)
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()

        # Initialize the loss and the accuracy
        self.loss = np.zeros(self.epoch)
        self.accuracy = np.zeros(self.epoch)

        # Start training
        for epoch in range(self.epoch):
            # Forward propagation
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Compute the loss
            self.loss[epoch] = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            # Compute the accuracy
            self.accuracy[epoch] = np.mean(np.where(y_pred >= 0.5, 1, 0) == y)

            # Backward propagation
            dw = np.dot(X.T, (y_pred - y)) / y.shape[0]
            db = np.sum(y_pred - y) / y.shape[0]

            # Update the weights and the bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print the loss and the accuracy
            if self.verbose:
                print(
                    f'Epoch: {epoch + 1}/{self.epoch}, loss: {self.loss[epoch]:.4f}, accuracy: {self.accuracy[epoch]:.4f}')

    def predict(self, X):
        """
        Predict the label.
        Input:
            X: array, features
        Output:
            y_pred: array, predicted labels
        """
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        # Other way to compute the predicted labels
        y_pred = np.where(y_pred >= 0.5, 1, 0)

        return y_pred

    def sigmoid(self, z):
        """
        Sigmoid function.
        Input:
            z: array, input
        Output:
            sigmoid: array, output
        """
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y, y_pred):
        """
        Gradient function.
        Input:
            X: array, features
            y: array, labels
            y_pred: array, predicted labels
        Output:
            dw: array, gradient of the weights
            db: float, gradient of the bias
        """
        dw = np.dot(X.T, (y_pred - y)) / y.shape[0]
        db = np.sum(y_pred - y) / y.shape[0]

        return dw, db

    def loss_auc(self):
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(self.loss, color='red')
        ax1.set_xlabel('Epoch', fontsize=15)
        ax1.set_ylabel('Loss', fontsize=15, color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        ax2.plot(self.accuracy, color='blue')
        ax2.set_ylabel('Accuracy', fontsize=15, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()
        plt.show()

    def roc_curve(self, y, y_pred):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.subplots(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('Receiver operating characteristic example', fontsize=15)
        plt.legend(loc="lower right")
        plt.show()

    def confusion_matrix(self, y, y_pred):
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

    def classification_report(self, y, y_pred):
        from sklearn.metrics import classification_report
        print('Classification report:')
        print(classification_report(y, y_pred))

    def _cv(self, X, y, cv):
        """
        Cross validation.
        Input:
            X: array, features
            y: array, labels
            cv: int, number of folds
        Output:
            scores: array, scores
        """
        score = 0
        scores = []
        for i in range(cv):
            X_train, X_test, y_train, y_test = self._split(X, y, cv, i)
            # Train the model
            self.fit(X_train, y_train)
            # Predict the labels
            y_pred = self.predict(X_test)
            # Compute the accuracy
            score = np.mean(np.where(y_pred >= 0.5, 1, 0) == y_test)
            scores.append(score)

        avg_score = np.mean(scores)

        return scores, avg_score

    def _split(self, X, y, cv, i):
        """
        Split the data into cv folds.
        Input:
            X: array, features
            y: array, labels
            cv: int, number of folds
            i: int, index of the fold
        Output:
            X_train: array, training features
            X_test: array, testing features
            y_train: array, training labels
            y_test: array, testing labels
        """
        # Compute the size of each fold
        fold_size = X.shape[0] // cv
        # Compute the starting and ending index of the fold
        start = i * fold_size
        end = start + fold_size
        # Split the data into training and testing
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        X_test = X[start:end]
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        y_test = y[start:end]

        return X_train, X_test, y_train, y_test

    def _learning_curve(self, X_train, y_train, X_test, y_test):
        # Learning curve on training set, fill_between is the standard deviation
        train_score = []
        test_score = []
        for i in range(1, X_train.shape[0] + 1):
            self.fit(X_train[:i], y_train[:i])
            y_train_pred = self.predict(X_train[:i])
            y_test_pred = self.predict(X_test)
            train_score.append(np.mean(np.where(y_train_pred >= 0.5, 1, 0) == y_train[:i]))
            test_score.append(np.mean(np.where(y_test_pred >= 0.5, 1, 0) == y_test))

        fig, ax = plt.subplots(figsize=(15, 10))
        plt.title('Learning curve on training set', fontsize=20)
        plt.plot(train_score, label='Training set', linewidth=3, color='blue')
        plt.fill_between(range(len(train_score)), np.array(train_score) - np.std(train_score),
                         np.array(train_score) + np.std(train_score), alpha=0.2, color='blue')
        plt.plot(test_score, label='Test set', linewidth=3, color='red')
        plt.fill_between(range(len(test_score)), np.array(test_score) - np.std(test_score),
                         np.array(test_score) + np.std(test_score), alpha=0.2, color='red')
        plt.legend()
        plt.show()

    def _cv_learning_curve(self, X_train, y_train, X_test, y_test, cv):
        # Learning curve on training set, fill_between is the standard deviation
        train_score = []
        test_score = []
        for i in range(1, X_train.shape[0] + 1):
            scores, avg_score = self._cv(X_train[:i], y_train[:i], cv)
            y_test_pred = self.predict(X_test)
            train_score.append(avg_score)
            test_score.append(np.mean(np.where(y_test_pred >= 0.5, 1, 0) == y_test))

        fig, ax = plt.subplots(figsize=(15, 10))
        plt.title('Learning curve on training set', fontsize=20)
        plt.plot(train_score, label='Training set', linewidth=3, color='blue')
        plt.fill_between(range(len(train_score)), np.array(train_score) - np.std(train_score),
                         np.array(train_score) + np.std(train_score), alpha=0.2, color='blue')
        plt.plot(test_score, label='Test set', linewidth=3, color='red')
        plt.fill_between(range(len(test_score)), np.array(test_score) - np.std(test_score),
                         np.array(test_score) + np.std(test_score), alpha=0.2, color='red')
        plt.legend()
        plt.show()

    def _cv2_learning_curve(self, X_train, y_train, X_test, y_test, cv):
        """
        Cross validation learning curve. This function is used to plot the learning curve
        on the training set and the test set. The learning curve is computed by cross validation over
        the training set samples
        Step 1: Initialize the training and test scores, and the number of samples
        Step 2: For each number of samples, compute the cross validation scores
        Step 3: Compute the average score and the standard deviation of the scores
        Step 4: Plot the learning curve
        """
        # Learning curve on training set, fill_between is the standard deviation
        train_score = []
        test_score = []
        for i in range(1, X_train.shape[0] + 1):
            scores, avg_score = self._cv(X_train[:i], y_train[:i], cv)
            y_test_pred = self.predict(X_test)
            train_score.append(avg_score)
            test_score.append(np.mean(np.where(y_test_pred >= 0.5, 1, 0) == y_test))

        fig, ax = plt.subplots(figsize=(15, 10))
        plt.title('Learning curve on training set', fontsize=20)
        plt.plot(train_score, label='Training set', linewidth=3, color='blue')
        plt.fill_between(range(len(train_score)), np.array(train_score) - np.std(train_score),
                         np.array(train_score) + np.std(train_score), alpha=0.2, color='blue')
        plt.plot(test_score, label='Test set', linewidth=3, color='red')
        plt.fill_between(range(len(test_score)), np.array(test_score) - np.std(test_score),
                         np.array(test_score) + np.std(test_score), alpha=0.2, color='red')
        plt.legend()
        plt.show()
