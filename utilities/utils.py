import sys
import math as m
import numpy as np
np.random.seed(123)
import scipy.io
from sklearn.decomposition import PCA
from augment import *
from transformations import *

def load_data(data_file):
    
    print("Loading data from %s" % (data_file))

    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)

    print("Data loading complete. Shape is %r" % (dataMat['featMat'].shape,))
    
    return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1   # Sequential indices


def reformatInput(data, labels, indices):
    
    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]
    
    if data.ndim == 4:
        return [( data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32) ),
                ( data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32) ),
                ( data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32) ) ]
    
    elif data.ndim == 5:
    
        return [( data[ :, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32) ),
                ( data[ :, validIndices], np.squeeze(labels[validIndices]).astype(np.int32) ),
                ( data[ :, testIndices], np.squeeze(labels[testIndices]).astype(np.int32) ) ]


if __name__ == '__main__':
    
    data = np.random.normal(size=(100, 10))
    print 'Original: {0}'.format(data)
    print 'Augmented: {0}'.format(augment_EEG(data, 0.1, pca=True))