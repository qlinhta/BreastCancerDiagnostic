import numpy as np
from math import *

class LDA:
  def __init__(self,n_components):
    self.n_components = n_components #number of labels
    self.linear_discriminants = None #store the eigenvector that we compute



  def mean(self,X,y):
    classes = np.unique(y)
    means = np.zeros(classes.shape[0],X.shape[1])
    for i in range(classes.shape[0]):
      means[i,:] = np.mean(X[y==i],axis=0)
    return means

  def pi_k(self,X,y,k):
     elm = y[y==k]
     return len(elm)/X.shape[0]

  def general_cov(self,X,y):
    classes = np.unique(y)
    for c in classes:
      class_idx = np.flatnonzero(y == c)
      sigma = len(X[class_idx])*np.cov(X[class_idx].T)
    return sigma/X.shape[0]


  def gaussian_proba(self,x,class_x):
   """
    a = np.exp(-1/2*(x-mean).T*np.linalg.inv(cov)*(x-mean))
    b = ((2*np.pi)**(d/2))*(np.absolute(cov)**1/2)
    return a/b"""


  def fit(self,X,y):
    n_features = X.shape[1]
    class_labels = np.unique(y)

    #within_class_dist(),between_class_dist(between classes)
    mean = np.mean(X,axis=0)
    within_class_dist = np.zeros((n_features,n_features))
    between_class_dist = np.zeros((n_features,n_features))

    for c in class_labels:
      class_idx = np.flatnonzero(y == c)
      X_c = X[class_idx]
      mean_c = np.mean(X_c,axis=0)
      within_class_dist += (X_c - mean_c).T.dot(X_c - mean_c) 


      n_c = X_c.shape[0]
      mean_diff = (mean_c - mean).reshape(n_features,1) 
      between_class_dist += n_c * (mean_diff).dot(mean_diff.T)
    
    #eigenvalues
    A = np.linalg.inv(within_class_dist).dot(between_class_dist)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvectors = eigenvectors.T
    idx = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx]
    self.linear_discriminants = eigenvectors[0:self.n_components]

  def transform(self,X):
    return np.dot(X,self.linear_discriminants.T)

  def predict(self,X):
    pass

  def decision_boundary(self,X,y):
    pass
