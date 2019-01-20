import getopt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.SpatiallyConstrainedNN import NeuralNetwork

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/dataset.txt')

casesDefault = 0
otherOption = 1


def main(argv):
    layers = [3, 5, 5, 1]
    activation, cost, problem, tuning, update = read_opts(argv)
    data_test, data_train = prepare_data(problem, tuning, layers[0])
    for i in range(2):
        nn = NeuralNetwork(layers, len(data_train))
        print("weights for test {:d}:".format(i))
        # print(nn.weights)
        print("deltas: ")
        # print(nn.deltas)
        print("testing: ")
        train(nn, data_train)
        list(map(plt.plot, nn.errors))
        plt.show()


# nn.test(data_test[t % len(data_test)])

#
# list(map(print, nn.feedforward([0, 0])))
# list(map(print, nn.feedforward([0, 1])))
# list(map(print, nn.feedforward([1, 0])))
# list(map(print, nn.feedforward([1, 1])))


def train(nn, inputs, dynamic=False, classification=True):
    # if dynamic:
    #     self.init_costs()
    # if classification:
    # input_data = np.random.permutation(data)
    for i in range(100):
        for idx, value in enumerate(inputs):
            nn.feedforward(value[0])
            nn.backprop(value[1], idx)
            nn.update_weights()



def prepare_data(problem, tuning, input_size=3):
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


if __name__ == "__main__":
    main(sys.argv[1:])
