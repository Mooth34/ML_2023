import random

import numpy as np
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def depth_accuracy_dependence(X, y):
    # Максимальное значение depth
    max_depth = 10
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    depth_values = np.arange(max_depth) + 1
    # Зависимость точности
    accuracy_dependence_test = Series(index=depth_values, dtype=float)
    accuracy_dependence_train = Series(index=depth_values, dtype=float)
    for depth in depth_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
        accuracy_dependence_train[depth] = mean_accuracy_train
        accuracy_dependence_test[depth] = mean_accuracy_test
    return accuracy_dependence_train, accuracy_dependence_test


def depth_accuracy_regr_dependence(X, y):
    # Максимальное значение depth
    max_depth = 10
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    depth_values = np.arange(max_depth) + 1
    # Зависимость точности
    accuracy_dependence_test = Series(index=depth_values, dtype=float)
    accuracy_dependence_train = Series(index=depth_values, dtype=float)
    mse_dependence = Series(index=depth_values, dtype=float)
    for depth in depth_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        mean_mse = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            regr = DecisionTreeRegressor(max_depth=depth)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            mean_mse += mean_squared_error(y_test, y_pred) / test_count
            # mean_accuracy_train += regr.score(X_train, y_train) / test_count
            # mean_accuracy_test += regr.score(X_test, y_test) / test_count
        mse_dependence[depth] = mean_mse
        # accuracy_dependence_train[depth] = mean_accuracy_train
        # accuracy_dependence_test[depth] = mean_accuracy_test
    return mse_dependence
    # return accuracy_dependence_train, accuracy_dependence_test


def samples_split_accuracy_dependence(X, y):
    max_samples_split = 15
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    samples_split_values = [1 - float(i) / max_samples_split for i in range(max_samples_split)]
    # Зависимость точности
    accuracy_dependence_test = Series(index=samples_split_values, dtype=float)
    accuracy_dependence_train = Series(index=samples_split_values, dtype=float)
    for samples_split in samples_split_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = DecisionTreeClassifier(min_samples_split=samples_split)
            clf.fit(X_train, y_train)
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
        accuracy_dependence_train[samples_split] = mean_accuracy_train
        accuracy_dependence_test[samples_split] = mean_accuracy_test
    return accuracy_dependence_train, accuracy_dependence_test


def samples_split_accuracy_regr_dependence(X, y):
    max_samples_split = 15
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    samples_split_values = [1 - float(i) / max_samples_split for i in range(max_samples_split)]
    # Зависимость точности
    accuracy_dependence_test = Series(index=samples_split_values, dtype=float)
    accuracy_dependence_train = Series(index=samples_split_values, dtype=float)
    for samples_split in samples_split_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = DecisionTreeRegressor(min_samples_split=samples_split)
            clf.fit(X_train, y_train)
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
        accuracy_dependence_train[samples_split] = mean_accuracy_train
        accuracy_dependence_test[samples_split] = mean_accuracy_test
    return accuracy_dependence_train, accuracy_dependence_test


def impurity_decrease_accuracy_dependence(X, y):
    max_impurity_decrease = 25
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    impurity_decrease_values = [1 - float(i) / max_impurity_decrease for i in range(max_impurity_decrease)]
    # Зависимость точности
    accuracy_dependence_test = Series(index=impurity_decrease_values, dtype=float)
    accuracy_dependence_train = Series(index=impurity_decrease_values, dtype=float)
    for impurity_decrease in impurity_decrease_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = DecisionTreeClassifier(min_impurity_decrease=impurity_decrease)
            clf.fit(X_train, y_train)
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
        accuracy_dependence_train[impurity_decrease] = mean_accuracy_train
        accuracy_dependence_test[impurity_decrease] = mean_accuracy_test
    return accuracy_dependence_train, accuracy_dependence_test


def impurity_accuracy_regression_dependence(X, y):
    max_impurity_decrease = 15
    # Число испытаний для каждого объема
    test_count = 15
    # Массив значений depth
    impurity_decrease_values = [1 - float(i) / max_impurity_decrease for i in range(max_impurity_decrease)]
    # Зависимость точности
    accuracy_dependence_test = Series(index=impurity_decrease_values, dtype=float)
    accuracy_dependence_train = Series(index=impurity_decrease_values, dtype=float)
    for impurity_decrease in impurity_decrease_values:
        mean_accuracy_train = 0.
        mean_accuracy_test = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            clf = DecisionTreeRegressor(min_impurity_decrease=impurity_decrease)
            clf.fit(X_train, y_train)
            mean_accuracy_train += clf.score(X_train, y_train) / test_count
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
        accuracy_dependence_train[impurity_decrease] = mean_accuracy_train
        accuracy_dependence_test[impurity_decrease] = mean_accuracy_test
    return accuracy_dependence_train, accuracy_dependence_test