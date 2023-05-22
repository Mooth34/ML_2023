import random

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def display_dependency_for_layers(X, y, n_iter): 
    count = 50
    test_count = 10
    nn = np.arange(count) + 1
    mistake_dependence_test = Series(index=nn, dtype=float)
    mistake_dependence_train = Series(index=nn, dtype=float)
    plt.xlabel('Layers')
    plt.ylabel('Misclass')
    for k in ['lbfgs', 'sgd', 'adam']:
        for layer in nn:
            mean_accuracy_test = 0.
            mean_accuracy_train = 0.
            for _ in np.arange(test_count):
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
                clf = MLPClassifier(solver=k, hidden_layer_sizes=(layer, layer, layer), max_iter=n_iter)
                clf.fit(X_train, y_train)
                mean_accuracy_test += clf.score(X_test, y_test) / test_count
                # mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mistake_dependence_test[layer] = 1 - mean_accuracy_test
        # mistake_dependence_train[k] = 1 - mean_accuracy_train
        plt.plot(mistake_dependence_test, label=k, marker='.', markersize=10)
        # plt.plot(mistake_dependence_train, label='train', marker='.', markersize=10)
        plt.legend()
        plt.grid()
        plt.show()


def display_dependency_for_function(X, y, n_iter):
    count = 50

    test_count = 10
    nn = np.arange(count) + 1

    mistake_dependence_test = Series(index=nn, dtype=float)
    mistake_dependence_train = Series(index=nn, dtype=float)
    plt.xlabel('Layers')
    plt.ylabel('Misclass')
    for k in ['identity', 'logistic', 'tanh', 'relu']:
        for layer in nn:
            mean_accuracy_test = 0.
            mean_accuracy_train = 0.
            for _ in np.arange(test_count):
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
                clf = MLPClassifier(solver='lbfgs', activation=k, hidden_layer_sizes=(layer, layer), max_iter=n_iter)
                clf.fit(X_train, y_train)
                mean_accuracy_test += clf.score(X_test, y_test) / test_count
                # mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mistake_dependence_test[layer] = 1 - mean_accuracy_test
        # mistake_dependence_train[k] = 1 - mean_accuracy_train
        plt.plot(mistake_dependence_test, label=k, marker='.', markersize=10)
        # plt.plot(mistake_dependence_train, label='train', marker='.', markersize=10)
        plt.legend()
        plt.grid()
        plt.show()


def display_dependency_for_iter(X, y):
    count = 100

    test_count = 10
    nn = [2 * i + 1 for i in range(count)]

    mistake_dependence_test = Series(index=nn, dtype=float)
    mistake_dependence_train = Series(index=nn, dtype=float)
    plt.xlabel('Max iter')
    plt.ylabel('Misclass')
    for n_iter in nn:
        mean_accuracy_test = 0.
        mean_accuracy_train = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(50, 50), max_iter=n_iter)
            clf.fit(X_train, y_train)
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
        mistake_dependence_test[n_iter] = 1 - mean_accuracy_test
        mistake_dependence_train[n_iter] = 1 - mean_accuracy_train
    plt.plot(mistake_dependence_test, label='test', marker='.', markersize=10)
    plt.plot(mistake_dependence_train, label='train', marker='.', markersize=10)
    plt.legend()
    plt.grid()
    plt.show()