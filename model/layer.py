import numpy as np

class Layer(object):

    def __init__(self, n_input, n_output, activator):
        self.n_input = n_input
        self.n_output = n_output
        self.weight = np.random.randn(n_output, n_input + 1) * 0.1
        self.activator = activator

    @classmethod
    def compose(self, units, activators):
        return [Layer(units[i], units[i + 1], activators[i]) for i in range(len(activators))]
