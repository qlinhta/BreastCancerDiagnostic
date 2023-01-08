import numpy as np
import matplotlib.pyplot as plt


# PCA and t-SNE

def pca(X, number_of_components):
    from sklearn.decomposition import PCA
    rd = PCA(n_components=number_of_components, random_state=42)
    X_pca = rd.fit_transform(X)
    return X_pca


def tsne(X, number_of_components):
    from sklearn.manifold import TSNE
    rd = TSNE(n_components=number_of_components, random_state=42)
    X_tsne = rd.fit_transform(X)
    return X_tsne
