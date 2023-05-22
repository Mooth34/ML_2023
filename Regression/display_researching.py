import random

import matplotlib.pyplot as plt
from pandas import Series
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def display_ridge_alpha(X, y):
    test_count = 10
    n_alpha = 50
    alpha_values = [10 * (1 - float(i / n_alpha)) for i in range(n_alpha)]
    # Зависимость точности от объема выборки
    mse_dependence_test = Series(index=alpha_values, dtype=float)
    for alpha in alpha_values:
        mean_mse_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            rlr = Ridge(alpha=alpha)
            rlr.fit(X_train, y_train)
            y_pred = rlr.predict(X_test)
            mean_mse_test += mean_squared_error(y_test, y_pred) / test_count
        mse_dependence_test[alpha] = mean_mse_test
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.plot(mse_dependence_test, label='test', marker='.', markersize=10)
    plt.grid()
    plt.show()


def display_ridge_alpha_train_test(X, y):
    test_count = 10
    n_alpha = 26
    alpha_values = [pow(10, -3 + 0.2 * i) for i in range(n_alpha)]
    # Зависимость точности от объема выборки
    mse_dependence_test = Series(index=alpha_values, dtype=float)
    mse_dependence_train = Series(index=alpha_values, dtype=float)
    for alpha in alpha_values:
        mean_mse_test = 0.
        mean_mse_train = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 1000))
            rlr = Ridge(alpha=alpha)
            rlr.fit(X_train, y_train)
            y_pred = rlr.predict(X_test)
            y_pred_train = rlr.predict(X_train)
            mean_mse_test += mean_squared_error(y_test, y_pred) / test_count
            mean_mse_train += mean_squared_error(y_train, y_pred_train) / test_count
        mse_dependence_test[alpha] = mean_mse_test
        mse_dependence_train[alpha] = mean_mse_train
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.plot(mse_dependence_train, label='train', marker='.', markersize=10)
    plt.plot(mse_dependence_test, label='test', marker='.', markersize=10)
    plt.legend()
    plt.grid()
    plt.show()