import numpy

class Activator:

    class linear:

        def __init__(self):
            'linear class'

        def activate(self, x):
            return x

        def derivate(self, x):
            return numpy.ones(x.shape)


    class sigmoid:

        def __init__(self):
            'sigmoid class'

        def activate(self, x):
            return 1.0 / (1.0 + numpy.exp(-x))

        def derivate(self, x):
            return x * (1.0 - x)


    class softmax:

        def __init__(self):
            'softmax class'

        def activate(self, x):
            _x = numpy.exp(x)
            return _x / _x.sum(axis = 0)

        def derivate(self, x):
            return x * (1.0 - x)


    class tanh:

        def __init(self):
            'tanh class'

        def activate(self, x):
            p = numpy.exp(x)
            m = numpy.exp(-x)
            return (p - m) / (p + m)

        def derivate(self, x):
            return 1.0 - x ** 2
