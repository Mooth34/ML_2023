import random
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def degree_error_dependence(X, y):
    # Максимальное значение degree
    max_degree = 50
    # Число испытаний для каждого объема
    test_count = 10
    # Массив значений degree
    degree_values = [int(10 * (1 - float(i) / max_degree)) for i in range(max_degree)]  # np.arange(max_degree) + 1
    # Зависимость ошибки от degree
    error_dependence = Series(index=degree_values, dtype=float)
    for degree in degree_values:
        mean_accuracy = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf_polynomial = SVC(kernel='poly', degree=degree)
            clf_polynomial.fit(X_train, y_train)
            mean_accuracy += clf_polynomial.score(X_test, y_test) / test_count
        error_dependence[degree] = 1 - mean_accuracy
    return error_dependence


def gamma_error_dependence(X_train, y_train, X_test, y_test):
    # Максимальное значение gamma
    max_gamma = 35
    # Число испытаний для каждого объема
    test_count = 30
    # Массив значений gamma
    gamma_values = np.arange(max_gamma) + 1
    test_error_dependence = Series(index=gamma_values, dtype=float)
    train_error_dependence = Series(index=gamma_values, dtype=float)
    for kernel in ['poly', 'rbf', 'sigmoid']:
        # Зависимость ошибки от gamma
        for gamma in gamma_values:
            test_mean_accuracy = 0.
            train_mean_accuracy = 0.
            for _ in np.arange(test_count):
                clf = SVC(kernel=kernel)
                clf.fit(X_train, y_train)
                test_mean_accuracy += clf.score(X_test, y_test) / test_count
                train_mean_accuracy += clf.score(X_train, y_train) / test_count
            test_error_dependence[gamma] = 1 - test_mean_accuracy
            train_error_dependence[gamma] = 1 - train_mean_accuracy
        plt.xlabel('Gamma')
        plt.ylabel('Probability mistake')
        plt.plot(train_error_dependence, label='train ' + kernel, marker='.', markersize=10)
        plt.plot(test_error_dependence, label='test ' + kernel, marker='.', markersize=10)
        plt.legend()
        plt.grid(True)
        plt.show()
    # return test_error_dependence