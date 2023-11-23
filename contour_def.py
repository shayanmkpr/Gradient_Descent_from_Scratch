import numpy as np
import matplotlib.pyplot as plt
from function_def import f1 , f2 , f3


def plot_contour_1(Q, q, p, x_vals, y_vals, levels=None):
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 0.5 * (Q[0, 0] * X**2 + (Q[1, 0] + Q[0, 1]) * X * Y + Q[1, 1] * Y**2) + q[0] * X + q[1] * Y + p

    plt.contour(X, Y, Z, levels=levels)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of f1(x)')

def plot_contour_2(a, b, x_vals, y_vals, levels=None):
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f2(x)

    plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of f2(x)')