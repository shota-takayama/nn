import numpy as np
from abc import ABCMeta, abstractmethod

class Activator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivate(self, x):
        pass


class linear(Activator):

    def __init__(self):
        pass

    def activate(self, x):
        return x

    def derivate(self, x):
        return np.ones(x.shape)


class sigmoid(Activator):

    def __init__(self):
        pass

    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivate(self, x):
        return x * (1.0 - x)


class softmax(Activator):

    def __init__(self):
        pass

    def activate(self, x):
        _x = np.exp(x)
        return _x / _x.sum(axis = 0)

    def derivate(self, x):
        return x * (1.0 - x)


class tanh(Activator):

    def __init(self):
        pass

    def activate(self, x):
        p = np.exp(x)
        m = np.exp(-x)
        return (p - m) / (p + m)

    def derivate(self, x):
        return 1.0 - x ** 2
