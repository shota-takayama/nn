import numpy as np

class Activator:

    class linear:

        def __init__(self):
            'linear class'

        def activate(self, x):
            return x

        def derivate(self, x):
            return np.ones(x.shape)


    class sigmoid:

        def __init__(self):
            'sigmoid class'

        def activate(self, x):
            return 1.0 / (1.0 + np.exp(-x))

        def derivate(self, x):
            return x * (1.0 - x)


    class softmax:

        def __init__(self):
            'softmax class'

        def activate(self, x):
            _x = np.exp(x)
            return _x / _x.sum(axis = 0)

        def derivate(self, x):
            return x * (1.0 - x)


    class tanh:

        def __init(self):
            'tanh class'

        def activate(self, x):
            p = np.exp(x)
            m = np.exp(-x)
            return (p - m) / (p + m)

        def derivate(self, x):
            return 1.0 - x ** 2
