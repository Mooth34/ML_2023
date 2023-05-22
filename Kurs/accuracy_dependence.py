import random

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def display_overtraining_for_k(X, y):
    count = 35
    test_count = 70
    nn = np.arange(count) + 1
    mistake_dependence_test = Series(index=nn, dtype=float)
    mistake_dependence_train = Series(index=nn, dtype=float)
    plt.xlabel('K')
    plt.ylabel('Misclass')
    for k in nn:
        mean_accuracy_test = 0.
        mean_accuracy_train = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=1)
            clf.fit(X_train, y_train)
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
        mistake_dependence_test[k] = 1 - mean_accuracy_test
        mistake_dependence_train[k] = 1 - mean_accuracy_train
    plt.plot(mistake_dependence_test, label='test', marker='.', markersize=10)
    plt.plot(mistake_dependence_train, label='train', marker='.', markersize=10)
    plt.legend()
    plt.grid()
    plt.show()


def display_overtraining_for_p(X, y):
    count = 35
    test_count = 70
    nn = np.arange(count) + 1
    mistake_dependence_test = Series(index=nn, dtype=float)
    mistake_dependence_train = Series(index=nn, dtype=float)
    plt.xlabel('P')
    plt.ylabel('Misclass')
    for p in nn:
        mean_accuracy_test = 0.
        mean_accuracy_train = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=15, weights='uniform', p=p)
            clf.fit(X_train, y_train)
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
        mistake_dependence_test[p] = 1 - mean_accuracy_test
        mistake_dependence_train[p] = 1 - mean_accuracy_train
    plt.plot(mistake_dependence_test, label='test', marker='.', markersize=10)
    plt.plot(mistake_dependence_train, label='train', marker='.', markersize=10)
    plt.legend()
    plt.grid()
    plt.show()


def get_accuracy_dependence_on_the_k(X, y, uniform):
    count = 35
    test_count = 70

    nn = np.arange(count) + 1

    mistake_dependence = Series(index=nn, dtype=float)
    for k in nn:
        mean_accuracy = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=k, weights=('uniform' if uniform else 'distance'), p=1)
            clf.fit(X_train, y_train)
            mean_accuracy += clf.score(X_test, y_test) / test_count
        mistake_dependence[k] = 1 - mean_accuracy
    return mistake_dependence


def get_accuracy_dependence_on_the_metric(X, y, k=3, uniform='uniform'):
    p_max = 35
    test_count = 70
    metrics = np.arange(p_max) + 1

    accuracy_dependence = Series(index=metrics, dtype=float)
    for metric in metrics:
        mean_accuracy = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=k, weights=('uniform' if uniform else 'distance'), p=metric)
            clf.fit(X_train, y_train)
            mean_accuracy += clf.score(X_test, y_test) / test_count
        accuracy_dependence[metric] = 1 - mean_accuracy
    return accuracy_dependence