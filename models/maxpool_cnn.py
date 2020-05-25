import sys
import theano
import theano.tensor as T

import numpy as np 
np.seed(111)  # fixing seed value as '111'

import lasagne
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer

# importing required functions from their Python files
from cnn import build_cnn




def build_convpool_max(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=7):

    convnets = []
    w_init = None

    # MODEL STRUCTURE as specified
    for i in range(n_timewin):
    
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
    
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(convnet)
    
    # LAYERS
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum) 
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    
    return convpool



def build_convpool_conv1d(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=7):
    
    convnets = []
    w_init = None
    
    # MODEL STRUCTURE as specified
    for i in range(n_timewin):
    
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
    
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    
    # LAYERS
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    convpool = Conv1DLayer(convpool, 64, 3)
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    
    return convpool
