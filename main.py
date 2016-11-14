import numpy as np
import os
from model.nn import NN as nn
from model.layer import Layer as lay
from model.activator import sigmoid, softmax
from model.error import cross_entropy

def read_data(fn):
    ml = np.loadtxt(fn, delimiter = ',')
    X, t = np.hsplit(ml, [-1])
    return (X / X.max(), t.astype('int'))

def create_label(t, n_data, n_class):
    T = np.zeros((n_data, n_class))
    T[np.arange(n_data), t[:, 0]] = 1.0
    return T

if __name__ == '__main__':

    print 'read data...'
    fn = '{0}/mldata/mnist_train_data.csv'.format(os.getenv('DPATH')[:-1])
    X, t = read_data(fn)
    n_data, n_input = X.shape
    n_class = np.unique(t).size
    T = create_label(t, n_data, n_class)

    print 'train...'
    sigmoid, softmax = sigmoid(), softmax()
    layers = lay.compose([n_input, 256, 128, n_class], [sigmoid, sigmoid, softmax])
    nn = nn(layers, cross_entropy())
    nn.train(X, T, epsilon = 0.001, lam = 0.0001, s_batch = 16, epochs = 50)

    print 'save figure of loss...'
    nn.save_lossfig()

    print '!!!finish!!!'
