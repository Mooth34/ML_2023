import random

import pandas as pd
import numpy as np
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def normalize_data(data):
    return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

def get_accuracy_dependence_on_the_sample_size(X, y):
    # Число	разбиений	объема	данных
    count = 25
    # Число	испытаний	для	каждого	объема
    test_count = 30
    # Количество	тестовых	данных
    test_size = int(np.size(y, axis=0) * 0.15)
    # Массив	значений	разбитой	выборки
    sample_sizes = np.linspace(int((X.index.size - test_size) / count), X.index.size - test_size, count)
    # Зависимость	точности	от	объема	выборки
    accuracy_dependence = Series(index=sample_sizes, dtype=float)
    for item in sample_sizes:
        mean_accuracy = 0.
        train_size = round(item * 0.75)
        for i in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size,
            random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_train, y_train)
            mean_accuracy += clf.score(X_test, y_test) / test_count
            accuracy_dependence[item] = mean_accuracy
    return accuracy_dependence


def get_accuracy_dependence_on_the_test_size(X, y, train_size):
    # Число	разбиений	объема	тестовых	данных
    count = 25
    # Число	испытаний	для	каждого	объема
    test_count = 70
    # Массив	значений	разбитой	тестовой	выборки
    test_sizes = np.linspace(int((X.index.size - train_size) / count), X.index.size - train_size, count)
    # Зависимость	точности	от	объема	тестовых	данных
    accuracy_dependence = Series(index=test_sizes, dtype=float)
    for item in test_sizes:
        mean_accuracy = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
            test_size=round(item),
            random_state=random.randint(0, 1000))
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_train, y_train)
            mean_accuracy += clf.score(X_test, y_test) / test_count
        accuracy_dependence[item] = mean_accuracy
    return accuracy_dependence

def get_accuracy_dependence_on_the_k(X, y, uniform):
    # Максимальное	число	ближайших	соседей
    count = 25
    # Число	испытаний	для	каждого	k
    test_count = 70
    # Массив	значений	количества	ближайших	соседей
    nn = np.arange(count) + 1
    # Зависимость	точности	от	k
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


def get_accuracy_dependence_on_the_metric(X, y, k, uniform):
    # Максимальное	число	p
    p_max = 25
    # Число	испытаний	для	каждого	объема
    test_count = 70
    # Массив	значений	p
    metrics = np.arange(p_max) + 1
    # Зависимость	точности	от	объема	выборки
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