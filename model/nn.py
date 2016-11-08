import numpy as np
from matplotlib import pyplot as plt

class NN:

    def __init__(self, layers, error):
        self.n_layer = len(layers)
        self.layers = layers
        self.error = error


    def train(self, X, T, epsilon, lam, s_batch, epochs):
        n_data = X.shape[0]
        self.__loss = np.zeros(epochs)
        for epo in range(epochs):
            ind = np.random.permutation(n_data)
            for i in range(0, n_data, s_batch):
                x, t = X[ind[i:i+s_batch]].T, T[ind[i:i+s_batch]].T
                zs = self.__forward(x)
                self.__update_weight(zs, t, epsilon, lam, zs[0].shape[1])
                self.__accumulate_loss(zs[-1], t, n_data, epo)

            self.__print_loss(epo)


    def predict(self, X):
        return self.__forward(X.T)


    def save_lossfig(self, fn = 'loss.png'):
        plt.plot(np.arange(self.__loss.size), self.__loss)
        plt.savefig(fn)


    def __forward(self, x):
        zs = [x]
        for l in self.layers:
            u = l.weight.dot(np.vstack((np.ones((1, x.shape[1])), x)))
            z = l.activator.activate(u)
            x = z
            zs += [z]
        return zs


    def __backward(self, zs, t):
        deltas = [self.error.derivated_delta(t, zs[-1]) * self.layers[-1].activator.derivate(zs[-1])]
        for i in range(1, self.n_layer)[::-1]:
            deltas = [self.layers[i].weight[:, 1:].T.dot(deltas[0]) * self.layers[i - 1].activator.derivate(zs[i])] + deltas
        return deltas


    def __grad(self, u, delta, s_batch):
        return delta.dot(np.vstack((np.ones((1, u.shape[1])), u)).T) / s_batch


    def __reg_term(self, lam):
        return lam * np.hstack((np.zeros((self.layers[-1].n_output, 1)), self.layers[-1].weight[:, 1:]))


    def __update_weight(self, zs, t, epsilon, lam, s_batch):
        deltas = self.__backward(zs, t)
        self.layers[-1].weight -= epsilon * (self.__grad(zs[-2], deltas[-1], s_batch) + self.__reg_term(lam))
        for i in range(0, self.n_layer - 1)[::-1]:
            self.layers[i].weight -= epsilon * self.__grad(zs[i], deltas[i], s_batch)


    def __accumulate_loss(self, y, t, n_data, epo):
        self.__loss[epo] += self.error.loss(y, t) / n_data


    def __print_loss(self, epo):
        print 'epoch: {0}, loss: {1}'.format(epo, self.__loss[epo])
