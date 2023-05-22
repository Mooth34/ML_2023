import random
import pandas as pd
import numpy as np
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def normalize_data(data):
    return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

def get_accuracy_dependence_on_the_sample_size(X, y):
    # Number of partitions of the data volume
    count = 25
    # Number of tests for each volume
    test_count = 70
    # Number of test data
    test_size = int(np.size(y, axis=0) * 0.15)
    # Array of split sample values
    sample_sizes = np.linspace(int((X.index.size - test_size) / count), X.index.size-test_size, count)
    # Dependence of accuracy on sample size
    accuracy_dependence = Series(index=sample_sizes, dtype=float)
    for item in sample_sizes:
        mean_accuracy = 0.
        train_size = round(item)
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size,
                     random_state=random.randint(0, 1000))
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            mean_accuracy += gnb.score(X_test, y_test) / test_count
        accuracy_dependence[item] = mean_accuracy
    return accuracy_dependence


def get_accuracy_dependence_on_the_test_size(X, y, train_size):
    # Number of partitions of the test data volume
    count = 25
    # Number of tests for each volume
    test_count = 70
    # Array of split test set values
    test_sizes = np.linspace(int((X.index.size - train_size) / count), X.index.size - train_size, count)
    # Dependence of accuracy on the amount of test data
    accuracy_dependence = Series(index=test_sizes, dtype=float)
    for item in test_sizes:
        mean_accuracy = 0.
        for _ in np.arange(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=round(item), random_state=random.randint(0, 1000))
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            mean_accuracy += clf.score(X_test, y_test) / test_count
        accuracy_dependence[item] = mean_accuracy
    return accuracy_dependence