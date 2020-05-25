import sys
import math as m
import numpy as np
np.random.seed(123)
import scipy.io
from sklearn.decomposition import PCA
from transformations import *


def augment_EEG(data, stdMult, pca=False, n_components=2):
    
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
    
    return augData


def augment_EEG_image(image, std_mult, pca=False, n_components=2):
    
    augData = np.zeros((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))
    
    for c in range(image.shape[1]):
        reshData = np.reshape(data['featMat'][:, c, :, :], (data['featMat'].shape[0], -1))
        if pca:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=True, n_components=n_components)
    
        else:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=False)
    
    return np.reshape(augData, data['featMat'].shape)
