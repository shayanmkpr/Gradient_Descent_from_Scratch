import numpy as np

def hessian(f, x, input_size, reg=1e-6):
    h = 1e-5
    hess = np.zeros((input_size, input_size))
    for i in range(input_size):
        for j in range(input_size):
            x_plus_h1 = x.copy()
            x_plus_h2 = x.copy()
            x_plus_h1[i, 0] += h
            x_plus_h2[j, 0] += h
            hess[i, j] = (f(x_plus_h1) - 2 * f(x) + f(x_plus_h2)) / (h ** 2)
    
    hess_reg = hess + np.eye(hess.shape[0]) * reg
    return hess_reg