import getopt
import os
import sys

import numpy as np
import pandas as pd
import scipy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/output.csv')

casesDefault = 0
otherOption = 1


def main(argv):
    layers = [30, 1024, 1024, 4]
    activation, cost, problem, tuning, update = read_opts(argv)
    dataset = prepare_data(problem, tuning)

    # split into input (X) and output (Y) variables
    X = dataset.iloc[:, 2:32].values
    y = dataset.iloc[:, 32:].values
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create model
    activation_f = 'tanh' if activation == 0 else 'sigmoid'
    print(activation_f)
    model = Sequential()
    model.add(Dense(layers[1], input_dim=layers[0], activation=activation_f))
    model.add(Dense(layers[2], activation=activation_f, bias_initializer='ones'))
    model.add(Dense(layers[3], activation=activation_f))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=500, batch_size=100)
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]))
    print("end")


def prepare_data(problem, tuning):
    if problem == 1:
        data = np.random.rand(10) * 10
        if tuning == 1:
            m = scipy.mean(data)
            data = data - m
    else:
        data = pd.read_csv(filename, delimiter=',')
        data, salary = data.iloc[:, 0:32], data.iloc[:, 32]
        data = pd.concat([data, salary.apply(lambda x: pd.Series(splitting(x)))], axis=1, ignore_index=True)

        if tuning == 1:
            mean_train = scipy.mean(data)
            data = data - np.tile(mean_train, [len(data), 1])

    return data


def splitting(salary):
    if salary > 7500000:
        return [0, 0, 0, 1]
    elif salary > 5000000:
        return [0, 0, 1, 0]
    elif salary > 2500000:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


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
