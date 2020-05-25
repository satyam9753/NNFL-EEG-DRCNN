import sys
import numpy as np 
np.seed(111)  # fixing seed value as '111'



import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer

def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3):

    weights = []        
    count = 0

    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
    
    network = InputLayer(shape=(None, n_colors, imsize, imsize), input_var=input_var)
    
    for i, s in enumerate(n_layers):
        for l in range(s):
    
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3), W=w_init[count], pad='same')
            count += 1
            weights.append(network.W)

        # MODEL STRUCTURE as specified
        network = MaxPool2DLayer(network, pool_size=(2, 2))

    return network, weights
