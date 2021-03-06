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


def model_a():
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    #net['drop_in'] =  DropoutLayer(net['input'], p=0.2)
    net['conv1_1'] = ConvLayer(net['input'], num_filters=96, filter_size=5, pad=1, flip_filters=False)

    net['conv2_1'] = PoolLayer(net['conv1_1'], pool_size=3, stride=2)
    #net['drop2_1'] =  DropoutLayer(net['conv2_1'], p=0.5)


    net['conv3_1'] = ConvLayer(net['conv2_1'], num_filters=192, filter_size=5, pad=1, flip_filters=False)

    net['conv4_1'] = PoolLayer(net['conv3_1'], pool_size=3, stride=2)
    #net['drop4_1'] =  DropoutLayer(net['conv4_1'], p=0.5)

    net['conv5_1'] = ConvLayer(net['conv4_1'], num_filters=192, filter_size=3, pad=1, flip_filters=False)

    net['conv6_1'] = ConvLayer(net['conv5_1'], num_filters=192, filter_size=1, flip_filters=False)
    net['conv7_1'] = ConvLayer(net['conv6_1'], num_filters=10, filter_size=1, flip_filters=False)
    net['global_avg'] = GlobalPoolLayer(net['conv7_1'])
    #net['fc10'] = DenseLayer(net['conv7_1'], num_units=10, nonlinearity=None)
    net['output'] = NonlinearityLayer(net['global_avg'], softmax)
    return net

def model_b():
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    net['drop_in'] =  DropoutLayer(net['input'], p=0.2)

    net['conv1_1'] = ConvLayer(net['drop_in'], num_filters=96, pad=1, filter_size=5)
    net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=96, filter_size=1)

    net['conv2_1'] = PoolLayer(net['conv1_2'], pool_size=3, stride=2)
    net['drop2_1'] =  DropoutLayer(net['conv2_1'], p=0.5)

    net['conv3_1'] = ConvLayer(net['drop2_1'], num_filters=192, pad=1, filter_size=5)
    net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=192, filter_size=1)

    net['conv4_1'] = PoolLayer(net['conv3_2'], pool_size=3, stride=2)
    net['drop4_1'] =  DropoutLayer(net['conv4_1'], p=0.5)

    net['conv5_1'] = ConvLayer(net['drop4_1'], num_filters=192, pad=1, filter_size=3)
    net['conv6_1'] = ConvLayer(net['conv5_1'], num_filters=192, filter_size=1)
    net['conv7_1'] = ConvLayer(net['conv6_1'], num_filters=10, filter_size=1)
    net['global_avg'] = GlobalPoolLayer(net['conv7_1'])
    net['output'] = NonlinearityLayer(net['global_avg'], softmax)

    return net

def model_c():
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    net['drop_in'] =  DropoutLayer(net['input'], p=0.2)

    net['conv1_1'] = ConvLayer(net['drop_in'], num_filters=96, pad=1, filter_size=3, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=96, pad=1, filter_size=3, flip_filters=False)

    net['conv2_1'] = PoolLayer(net['conv1_2'], pool_size=3, stride=2)
    net['drop2_1'] =  DropoutLayer(net['conv2_1'], p=0.5)

    net['conv3_1'] = ConvLayer(net['drop2_1'], num_filters=192, pad=1,  filter_size=3, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=192, pad=1, filter_size=3, flip_filters=False)

    net['conv4_1'] = PoolLayer(net['conv3_2'], pool_size=3, stride=2)
    net['drop4_1'] =  DropoutLayer(net['conv4_1'], p=0.5)

    net['conv5_1'] = ConvLayer(net['drop4_1'], num_filters=192, pad=1, filter_size=3, flip_filters=False)
    net['conv6_1'] = ConvLayer(net['conv5_1'], num_filters=192, filter_size=1, flip_filters=False)
    net['conv7_1'] = ConvLayer(net['conv6_1'], num_filters=10, filter_size=1, flip_filters=False)
    net['global_avg'] = GlobalPoolLayer(net['conv7_1'])
    net['output'] = NonlinearityLayer(net['global_avg'], softmax)

    return net
