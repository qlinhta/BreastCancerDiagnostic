import numpy as np
import metrics_lda
from scipy.stats import multivariate_normal
from numpy.linalg import det, inv

class LDA:

    def __init__(self):
        self.n_classes = 2
        self.coef = np.zeros((self.n_classes,))
        self.intercept = 0.0
        #2nd method for prediction
        self.priors = []


    def means(self,X, y):
        classes = np.unique(y)
        means = np.zeros((classes.shape[0], X.shape[1]))
        for i in range(classes.shape[0]):
            means[i, :] = np.mean(X[y == i], axis=0)
        return means

    def prob_k(self, X, y):
        classes = np.unique(y)
        pi_k = np.zeros((len(classes),))
        for c in classes:
            pi_k[c] = len(y[y==c]) / len(y)
        return pi_k

    def general_cov(self, X, y,):
        classes = np.unique(y)
        sigma = np.zeros((X.shape[1], X.shape[1]))
        for c in classes:
            sigma = sigma + len(X[y == c]) * np.cov(X[y == c].T)
        sigma = sigma / X.shape[0]
        return sigma
    
    def fit(self,X,y):
        means_overall = self.means(X, y)
        pi_overall = self.prob_k(X,y)
        sigma_inv = np.linalg.inv(self.general_cov(X, y))

        # coef
        self.coef = sigma_inv @ (means_overall[1] - means_overall[0])
        #self.coef = sigma_inv @ means_overall
        

        # intercept
        p = (means_overall[1] - means_overall[0]) @ sigma_inv @ (means_overall[0] + means_overall[1])
        self.intercept = (-0.5 * p - np.log(pi_overall[0] / pi_overall[1]))
    
    def decision_boundary(self, X):  # x.Tw + b = 0
        return X @ self.coef.T + self.intercept

    def predict(self, X):
        #return np.argmax(self.decision_boundary(X),axis = 1)
        scores = self.decision_boundary(X)
        y_predicted = [1 if i > 0 else 0 for i in scores]
        return np.array(y_predicted)
    
    def cross_validation_lda(self, X, y, k):
        X_folds = np.array_split(X, k)
        y_folds = np.array_split(y, k)
        model = LDA(self.n_components)
        model.set_coef_intercept(X, y)
        accuracies = []
        for i in range(k):
            # Get the training data
            X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
            y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
            X_val = X_folds[i]
            y_val = y_folds[i]
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_val)
            accuracies.append(metrics.accuracy(y_val, y_predicted))
        return np.mean(accuracies)

    def predict_proba_to_plot(self,X):
        y_pred = self.predict(X)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = (y_pred == 1).astype(int)
        proba[:, 0] = (y_pred == 0).astype(int)
        return np.round(proba)
    
    #----------------------another approach for prediction-----------------------------------

    def set_priors(self,X,y):
       for i in range(2):
        self.priors.append(len(y[y==i])/len(y))

    def normal_multivariate(self,X,mean,cov):
        n = mean.shape[0]
        cov_inv = inv(cov)
        cov_det = det(cov)
        f = -0.5 * ((X - mean) @ cov_inv) * (X - mean)
        nominator = np.exp(f.sum(axis=1))
        denominator = (2 * np.pi) ** (n / 2) * cov_det ** 0.5
        probas = nominator / denominator
        return probas
         
    def predict_proba(self,X,mean,cov0,cov1,priors):
        probas = np.zeros((X.shape[0],self.n_classes))
        for k in range(self.n_classes):
            mean_k = mean[k]
            if(k==0):
                cov_k = cov0
            else:
                cov_k = cov1
            p_k = priors[k]
            probas[:,k] = p_k*multivariate_normal.pdf(X,mean_k, cov_k)
        y_predicted = [1 if probas[i][1] > probas[i][0] else 0 for i,row in enumerate(probas)]
        return np.array(y_predicted)

