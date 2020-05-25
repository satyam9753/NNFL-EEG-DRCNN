import sys
import numpy as np
from __future__ import print_function

import time
np.random.seed(111)  # fixing seed value as '111'

from functools import reduce
import math as m
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

# importing required functions from their Python files
from utilities.augment import augment_EEG

# function to GENERATE IMAGES
def gen_images(locs, features, n_gridpoints, normalize=True, augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    
    feat_array_temp = []
    nElectrodes = locs.shape[0]    

    n_colors = features.shape[1] / nElectrodes

    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])

    if augment:
        # 2 cases : either PCA or not (in accordance with the paper)
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)

    n_samples = features.shape[0]

    # Interpolate the values accordingly
    grid_x, grid_y = np.mgrid[ min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j, min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j ]
    temp_interp = []

    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generating edgeless images
    if edgeless:
        
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating as required
    for i in range(n_samples):
        
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing as requied
    for c in range(n_colors):

        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        
        
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     

