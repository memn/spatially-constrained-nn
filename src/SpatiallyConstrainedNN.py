from copy import deepcopy
from math import tanh, exp
from random import random

import numpy as np

unipolar = True
threshold = 5


def sigmoid(x):
    return 1 / (1 + exp(-x))


def _sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def _tanh(x):
    return 1 - (tanh(x) ** 2)


# activation function
activate = np.vectorize(lambda c: sigmoid(c)) if unipolar else np.vectorize(lambda c: tanh(c))

# derivative of activation function
_activate = np.vectorize(lambda c: _sigmoid(c)) if unipolar else np.vectorize(lambda c: _tanh(c))


class NeuralNetwork:
    def __init__(self, top, train_size):
        self.topology = top
        self.errors = [[] for _ in range(train_size)]
        self.layers = [np.matrix([0 for _ in range(size)]).T for size in top]
        self.weights = list()
        for i, j in zip(top[:-1], top[1:]):
            self.weights.append(np.matrix([[random() * 2 - 1 for _ in range(i + 1)] for _ in range(j)]))
        self.momentum = 0.3
        self.dw = None
        self.deltas = None

    def feedforward(self, a):
        for q, w in zip(self.costs, self.weights):
            a = activate(np.dot(w, a) + q)
        return a

    def backprop(self, expected, index):
        deltas = deepcopy(self.layers)
        self.errors[index].append(np.linalg.norm(expected - self.layers[-1]))
        deltas[-1] = np.dot(expected - self.layers[-1], _activate(self.layers[-1]))
        for i in range(len(deltas) - 1)[::-1]:
            try:
                sc = deltas[i + 1].T @ self.weights[i]
            except ValueError:
                sc = deltas[i + 1][:-1, :].T @ self.weights[i]
            biased = np.concatenate((self.layers[i], [[1]]))  # layer with a bias node of 1 added to it
            deltas[i] = np.multiply(_activate(biased), sc.T)
        self.deltas = deltas[1:]

    def update_weights(self):
        for i in range(len(self.deltas)):
            biased = np.concatenate((self.layers[i], [[1]])).T
            if self.deltas[i].shape[0] is not self.weights[i].shape[0]:
                self.deltas[i] = self.deltas[i][:-1, :]
            assert self.deltas[i].shape[0] == self.weights[i].shape[0]
            assert biased.shape[1] == self.weights[i].shape[1]

            dw = self.deltas[1] @ biased
            if self.dw is not None:
                dw += self.momentum * self.dw
            self.weights[i] += 0.1 * dw
            self.dw = dw

    def test(self, test, index):
        self.backprop(test, index)
        for i in range(len(self.deltas)):
            biased = np.concatenate((self.layers[i], [[1]])).T
            if self.deltas[i].shape[0] is not self.weights[i].shape[0]:
                self.deltas[i] = self.deltas[i][:-1, :]
            assert self.deltas[i].shape[0] == self.weights[i].shape[0]
            assert biased.shape[1] == self.weights[i].shape[1]

            dw = self.deltas[1] @ biased
            if self.dw is not None:
                dw += self.momentum * self.dw
            self.weights[i] += 0.1 * dw
            self.dw = dw
        return self.errors
