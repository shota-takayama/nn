import numpy as np
from abc import ABCMeta, abstractmethod

class Error(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, y, t):
        pass

    @abstractmethod
    def derivated_delta(self, y, t):
        pass


class squared(Error):

    def __init__(self):
        pass

    def loss(self, y, t):
        return (y - t).T.dot(y - t).sum() / 2.0

    def derivated_delta(self, y, t):
        return y - t


class cross_entropy(Error):

    def __init__(self):
        pass

    def loss(self, y, t):
        return (-t * np.log(y)).sum()

    def derivated_delta(self, t, y):
        return (y - t) / (y * (1.0 - y))
