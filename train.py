from __future__ import print_function #similar to that in 'utils.py'
import time
import sys

import numpy as np
np.random.seed(111)
from functools import reduce
import math as m

import scipy.io
import theano
import theano.tensor as T

import lasagne
from lasagne.regularization import regularize_layer_params, regularize_network_params, l1, l2
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer

# importing required functions from their Python files
from models.cnn import build_cnn
from models.maxpool_cnn import build_convpool_conv1d, build_convpool_max
from models.lstm import build_convpool_lstm
from models.mix_lstm import build_convpool_mix
from generate_images import gen_images
from utilities.transformations import azim_proj

# creating mini-batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]



def train(images, labels, fold, model_type, batch_size=32, num_epochs=5):
    
    num_classes = len(np.unique(labels))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)

    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    input_var = T.TensorType('floatX', ((False,) * 5))()
    target_var = T.ivector('targets')
    
    print("Building model and compiling functions...")
    
    if model_type == '1dconv':
        network = build_convpool_conv1d(input_var, num_classes)
    
    elif model_type == 'maxpool':
        network = build_convpool_max(input_var, num_classes)
    
    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100)
    
    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100)
    
    elif model_type == 'cnn':
        input_var = T.tensor4('inputs')
        network, _ = build_cnn(input_var)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']")


    prediction = lasagne.layers.get_output(network)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    reg_factor = 1e-4
    l2_penalty = regularize_network_params(network, l2) * reg_factor
    loss += l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    print("Starting training...")
    best_validation_accu = 0
    

    for epoch in range( num_epochs ):
    
        train_err = 0
        train_batches = 0
        start_time = time.time()
        
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        val_err = 0
        val_acc = 0
        val_batches = 0
        
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        av_train_err = train_err / train_batches
        av_val_err = val_err / val_batches
        av_val_acc = val_acc / val_batches
    
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(av_train_err))
        print("  validation loss:\t\t{:.6f}".format(av_val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))
    
        if av_val_acc > best_validation_accu:
            best_validation_accu = av_val_acc
            test_err = 0
            test_acc = 0
            test_batches = 0
    
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
    
            av_test_err = test_err / test_batches
            av_test_acc = test_acc / test_batches
    
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(av_test_err))
            print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
            np.savez('weights_lasg_{0}'.format(model_type), *lasagne.layers.get_all_param_values(network))

    print('-'*50)
    print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
    
    return av_test_acc


# <----------EXECUTION STARTS---------->
if __name__ == '__main__':
    
    from utilities.utils import reformatInput

    print('Loading data...')
    locs = scipy.io.loadmat('/data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []

    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = scipy.io.loadmat('/data/FeatureMat_timeWin.mat')['features']
    subj_nums = np.squeeze(scipy.io.loadmat('/data/trials_subNums.mat')['subjectNum'])
    fold_pairs = []

    for i in np.unique(subj_nums):
    
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr) 
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))


    # <---------CNN MODEL-------->
    print('Generating images...')

    av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(feats.shape[1] / 192)])
    av_feats = av_feats / (feats.shape[1] / 192)
    images = gen_images(np.array(locs_2d), av_feats, 32, normalize=True)
    
    print('Training the CNN Model...')
    test_acc_cnn = []
    
    for i in range(len(fold_pairs)):
        print('fold {0}/{1}'.format(i + 1, len(fold_pairs)))
        test_acc_cnn.append(train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[i], 'cnn', num_epochs=10))



    # <-------Conv-LSTM MODEL------->
    print('Generating images for all time windows...')

    images_timewin = np.array([gen_images(np.array(locs_2d),feats[:, i * 192:(i + 1) * 192], 32, normalize=True) for i in range(feats.shape[1] / 192)])

    print('Training the LSTM-CONV Model...')
    test_acc_mix = []

    for i in range(len(fold_pairs)):
        print('fold {0}/{1}'.format(i+1, len(fold_pairs)))
        test_acc_mix.append(train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[i], 'mix', num_epochs=10))
    print('*' * 40)
    print('Average MIX test accuracy: {0}'.format(np.mean(test_acc_mix)*100))
    print('Average CNN test accuracy: {0}'.format(np.mean(test_acc_cnn) * 100))
    print('*' * 40)

print('Successfully executed')