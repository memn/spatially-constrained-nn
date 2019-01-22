import getopt
import os
import sys
from copy import deepcopy
from math import exp, tanh
from random import random

import numpy as np
import scipy


def sigmoid(x):
    return 1 / (1 + exp(-x))


def _sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def _tanh(x):
    return 1 - (tanh(x) ** 2)


unipolar = True
threshold = 5

# activation function
activate = np.vectorize(lambda c: sigmoid(c)) if unipolar else np.vectorize(lambda c: tanh(c))

# derivative of activation function
_activate = np.vectorize(lambda c: _sigmoid(c)) if unipolar else np.vectorize(lambda c: _tanh(c))


class NeuralNetwork:
    def __init__(self, top, errors, data):
        self.topology = top
        self.errors = errors
        self.data = data
        self.layers = [np.matrix([0 for _ in range(size)]).T for size in top]
        self.weights = list()
        for i, j in zip(top[:-1], top[1:]):
            self.weights.append(np.matrix([[random() * 2 - 1 for _ in range(i + 1)] for _ in range(j)]))
        self.momentum = 0.3
        self.dw = None
        self.deltas = None

    def feedforward(self, inputs):
        assert len(inputs) == self.topology[0]
        self.layers[0] = np.matrix(inputs[:]).T
        for i in range(len(self.layers) - 1):
            biased = np.concatenate((self.layers[i], np.matrix([1])), axis=0)
            self.layers[i + 1] = activate(self.weights[i] @ biased)

        return self.layers[-1]

    def backpropagate(self, test, index):
        inputs, expected = test
        self.feedforward(inputs)  # load the inputs into the network
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

    def test(self, test, index):
        self.backpropagate(test, index)
        for i in range(len(self.deltas)):
            biased = np.concatenate((self.layers[i], [[1]])).T
            if self.deltas[i].shape[0] is not self.weights[i].shape[0]:
                self.deltas[i] = self.deltas[i][:-1, :]
            assert self.deltas[i].shape[0] == self.weights[i].shape[0]
            assert biased.shape[1] == self.weights[i].shape[1]

            dw = self.deltas[1] @ biased
            if self.dw is not None:
                dw += self.momentum * self.dw
            self.weights[i] += 0.1 * dw[:-1]
            self.dw = dw


def prepare_data(input_size=3):
    if problem == 0:
        data_train = np.random.rand(10) * 10
        data_test = data_train
        if tuning == 1:
            m = scipy.mean(data_train)
            data_train = data_train - m
    else:
        data = scipy.loadtxt(filename, delimiter=',')
        train_size = int(len(data) * 0.8)
        data_train = [[row[:input_size], row[input_size:]] for row in data[:train_size]]
        data_test = [[row[:input_size], row[input_size:]] for row in data[train_size:]]

        if tuning == 1:
            mean_train = scipy.mean(data_train)
            data_train = data_train - np.tile(mean_train, [len(data_train), 1])
            mean_test = scipy.mean(data_test)
            data_test = data_test - np.tile(mean_test, [len(data_test), 1])
    return data_test, data_train


casesDefault = 0
otherOption = 1


def read_opts(argv):
    cost = casesDefault  # static = 0, dynamic = 1
    problem = casesDefault  # regression = 0, classification = 1
    activation = casesDefault  # bipolar (tanh) = 0, unipolar = 1
    update = casesDefault  # without momentum = 0, with momentum = 1
    tuning = casesDefault  # not subtract mean = 0, subtract mean = 1

    try:
        opts, args = getopt.getopt(argv, "hc:p:a:u:t:")
    except getopt.GetoptError:
        print('project.py -c <cost> -p <problem> -a <activation> -u <update> -t <tuning>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('project.py -c <cost> -p <problem> -a <activation> -u <update> -t <tuning>')
            print("cost:  static = 0, dynamic = 1")
            print("problem:  regression = 0, classification = 1")
            print("activation:  bipolar (tanh) = 0, unipolar = 1")
            print("update:  without momentum = 0, with momentum = 1")
            print("tuning:  not subtract mean = 0, subtract mean = 1")
            print("defaults are {:d}".format(casesDefault))
            sys.exit()
        elif opt == "-c":
            cost = int_try_parse(arg)
        elif opt == "-p":
            problem = int_try_parse(arg)
        elif opt == "-a":
            activation = int_try_parse(arg)
        elif opt == "-u":
            update = int_try_parse(arg)
        elif opt == "-t":
            tuning = int_try_parse(arg)
    return activation, cost, problem, tuning, update


def int_try_parse(value, default=casesDefault):
    try:
        if int(value) == otherOption:
            return otherOption
        else:
            # if int(value) != 0:
            #     print("value is not 1, set to default:{:d}".format(default))
            return default
    except ValueError:
        return default


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/dataset.txt')

layers = [3, 3, 4, 1]
activation, cost, problem, tuning, update = read_opts(sys.argv[1:])
data_test, data_train = prepare_data(layers[0])

tr_data = [
    [[5.1, 3.5, 1.4], [0.2]],
    [[4.9, 3.0, 1.4], [0.2]],
    [[5.4, 3.9, 1.7], [0.4]],
    [[4.6, 3.4, 1.4], [0.3]],
    [[5.0, 3.4, 1.5], [0.2]],
    [[4.4, 2.9, 1.4], [0.2]],
    [[4.9, 3.1, 1.5], [0.1]],
    [[4.7, 3.2, 1.3], [0.2]]
]
#
# nn = NeuralNetwork([3, 3, 1])
# nn.test(tr_data[0])
# for i in range(1000):
#     nn.test(tr_data[i % 3])
#
# list(map(print, nn.feedforward([5.1, 3.5, 1.4])))
# list(map(print, nn.feedforward([4.9, 3.0, 1.4])))
# list(map(print, nn.feedforward([4.7, 3.2, 1.3])))

nn = NeuralNetwork(layers, [list() for _ in range(len(data_train))], data_train)
for i in range(1000):
    for j in range(len(tr_data)):
        nn.test(data_train[j], j)

list(map(print, nn.feedforward([5.1, 3.5, 1.4])))
list(map(print, nn.feedforward([4.9, 3.0, 1.4])))
list(map(print, nn.feedforward([4.7, 3.2, 1.3])))
list(map(print, nn.feedforward([4.9, 3.1, 1.5])))
