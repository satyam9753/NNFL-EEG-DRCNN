import sys
import numpy as np
np.seed(111) # fixing seed value as '111'

import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

# importing required functions from their Python files
from cnn import build_cnn
from maxpool_cnn import build_convpool_conv1d, build_convpool_max

def build_convpool_mix(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=7):
    
    w_init = None
    convnets = []
    
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

    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip, nonlinearity=lasagne.nonlinearities.tanh)
    lstm_out = SliceLayer(lstm, -1, 1)
    
    dense_input = ConcatLayer([conv_out, lstm_out])
    
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    convpool = DenseLayer(convpool, num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)

    return convpool