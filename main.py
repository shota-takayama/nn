import numpy
import os
from model.nn import NN
from model.activator import Activator
from model.error import Error

def read_data(fn):
    ml = numpy.loadtxt(fn, delimiter = ',')
    X, t = numpy.hsplit(ml, [-1])
    return (X / X.max(), t.astype('int'))

def create_label(t, n_data, n_class):
    T = numpy.zeros((n_data, n_class))
    T[numpy.arange(n_data), t[:, 0]] = 1.0
    return T

if __name__ == '__main__':

    print 'read data...'
    fn = '{0}/mldata/mnist_train_data.csv'.format(os.getenv('DPATH')[:-1])
    X, t = read_data(fn)
    n_data, n_input = X.shape
    n_class = numpy.unique(t).size
    T = create_label(t, n_data, n_class)

    print 'train...'
    nn = NN(n_input, 128, n_class)
    sigmoid, softmax = Activator.sigmoid(), Activator.softmax()
    error = Error.cross_entropy()
    nn.train(X, T, sigmoid, softmax, error, epsilon = 0.001, lam = 0.0001, s_batch = 16, epoch = 60)

    print 'save weight...'
    nn.save_weight()

    print 'save figure of error...'
    nn.save_errorfig()

    print '!!!finish!!!'
