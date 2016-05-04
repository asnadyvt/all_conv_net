import lasagne
import numpy as np
import matplotlib.pyplot as plt
import os
import theano
import theano.tensor as T
import cPickle as pickle
import gzip
import glob
import pdb


from utils import *

from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
from lasagne.layers import DenseLayer, NonlinearityLayer, PadLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import GlobalPoolLayer


def hw3_cnn():
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    #net['drop_in'] =  DropoutLayer(net['input'], p=0.2)

    net['conv1_1'] = ConvLayer(net['input'], num_filters=16, filter_size=5, flip_filters=False)

    net['conv2_1'] = PoolLayer(net['conv1_1'], pool_size=2)
    #net['drop2_1'] =  DropoutLayer(net['conv2_1'], p=0.5)


    net['conv3_1'] = ConvLayer(net['conv2_1'], num_filters=512, filter_size=5, flip_filters=False)
    
    net['conv4_1'] = PoolLayer(net['conv3_1'], pool_size=2)
    #net['drop4_1'] =  DropoutLayer(net['conv4_1'], p=0.5)

    net['fc_5'] = DenseLayer(net['conv4_1'], 500)
    net['output'] = NonlinearityLayer(net['fc_5'], softmax)
    return net

