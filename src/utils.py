import numpy as np
import cPickle as pickle
import os
import gzip
import tarfile

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar(d=10):
    '''
    Load the ATIS dataset

    :type foldnum: int
    :param foldnum: fold number of the ATIS dataset, ranging from 0 to 4.

    '''

    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar100_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    url=cifar10_url
    if d==100:
        url = cifar100_url
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            print('Downloading data from %s' % url)
            urllib.request.urlretrieve(url, new_path)
        f_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        tar = tarfile.open(new_path)
        tar.extractall(f_path)
        tar.close()
        final_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            "cifar-10-batches-py"
        )
        return final_path
    
   
    filename = check_dataset('cifar-10-python.tar.gz')
    train_data = [None]*5
    for i in xrange(5):
        data_path = filename+'/data_batch_'+str(i+1)
        train_data[i] = unpickle(data_path)
    test_data = unpickle(filename+'/test_batch')
    label_names = unpickle(filename+'/batches.meta')
    return train_data,test_data,label_names
    
if __name__ == '__main__':
    load_cifar(10)