import numpy as np
import sympy as sp

def f1(x):
    Q = np.array([[48, 12], [8, 8]])
    q = np.array([[13], [23]])
    p = 4
    f1_x = 0.5 * np.matmul(np.transpose(x), np.matmul(Q, x)) + np.matmul(np.transpose(q), x) + p
    return f1_x

def f2(x):
    n = len(x)
    a = -2
    b = 150
    result = 0
    for i in range(n-1):
        term1 = b * (x[i+1]**2 - x[i])**2
        term2 = (x[i] - a)**2
        result += term1 + term2
    return result


def f3(x):
    f3_x = (x[0] - 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4
    return f3_x
    
