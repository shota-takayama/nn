import numpy
from matplotlib import pyplot

class NN:

    def __init__(self, n_input, n_hidden, n_output):
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.hidden_weight = numpy.random.randn(n_hidden, n_input + 1) * 0.01
        self.output_weight = numpy.random.randn(n_output, n_hidden + 1) * 0.01


    def train(self, X, T, hidden_act, output_act, error, epsilon, lam, s_batch, epoch):
        n_data = X.shape[0]
        Y = numpy.hstack((X, T))
        self.__epoch = epoch
        self.__error = numpy.zeros(epoch)
        for epo in range(epoch):
            for bat in self.__create_batch(Y, n_data, s_batch):
                x, t = map(lambda _m: _m.T, numpy.hsplit(bat, [self.n_input]))
                z, y = self.__forward(x, hidden_act.activate, output_act.activate)

                self.__update_weight(x, z, y, t, epsilon, lam, output_act, hidden_act, error)
                self.__error[epo] += error.delta(y, t)

            print 'epoch: {0}, error: {1}'.format(epo, self.__error[epo])


    def predict(self, X, hidden_act, output_act):
        return self.__forward(X.T, hidden_act.activate, output_act.activate)


    def save_weight(self, fns = ['hidden_weight.npy', 'output_weight.npy']):
        numpy.save(fns[0], self.hidden_weight)
        numpy.save(fns[1], self.output_weight)


    def save_errorfig(self, fn = 'error.png'):
        pyplot.plot(numpy.arange(self.__epoch), self.__error)
        pyplot.savefig(fn)


    def __forward(self, x, hidden_actf, output_actf):
        z = hidden_actf(self.hidden_weight.dot(numpy.vstack((numpy.ones((1, x.shape[1])), x))))
        y = output_actf(self.output_weight.dot(numpy.vstack((numpy.ones((1, z.shape[1])), z))))
        return (z, y)


    def __update_weight(self, x, z, y, t, epsilon, lam, output_act, hidden_act, error):
        s_batch = x.shape[1]
        reg_term = lam * numpy.hstack((numpy.zeros((self.n_output, 1)), self.output_weight[:, 1:]))

        # back-propagate delta
        output_delta = error.derivated_delta(y, t) * output_act.derivate(y)
        hidden_delta = self.output_weight[:, 1:].T.dot(output_delta) * hidden_act.derivate(z)

        # update weight
        self.output_weight -= epsilon * (self.__update(z, output_delta, s_batch) + reg_term)
        self.hidden_weight -= epsilon * self.__update(x, hidden_delta, s_batch)


    def __update(self, u, delta, s_batch):
        return delta.dot(numpy.vstack((numpy.ones((1, u.shape[1])), u)).T) / s_batch


    def __create_batch(self, Y, n_data, s_batch):
        strides = numpy.arange(s_batch, n_data, s_batch)
        return numpy.vsplit(numpy.random.permutation(Y), strides)
