import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def display(x, y, model, diagram_name=' ', x_name='', y_name=''):
    X_set, y_set = x, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 0.1, stop=X_set[:, 0].max() + 0.1, step=0.001),
    np.arange(start=X_set[:, 1].min() - 0.1, stop=X_set[:, 1].max() + 0.1, step=0.001))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  # model1
    alpha=0.25, cmap=ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label=j)

    plt.title(diagram_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.grid()
    plt.show()