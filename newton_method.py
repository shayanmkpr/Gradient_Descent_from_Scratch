import numpy as np
from gradient import gradient
from hessian import hessian


def newton_method(f, x0, max_iter, tol):
    x = x0.reshape(-1, 1)
    input_size = len(x)
    val = np.zeros(max_iter)
    val[0] = f(x)

    x_history = [x.flatten()]
    print("NEWTON'S METHOD WITH HESSIAN")
    print("Iteration\tX Value\t\t\t\tFunction Value")
    print(f"0\t\t{x.flatten()}\t\t{val[0]}")

    for k in range(1, max_iter):
        # Calculate the Newton step
        grad = gradient(f, x, input_size)
        hess = hessian(f, x, input_size, reg=1e-6)

        step_size = 1.0 / np.matmul(grad.T, np.linalg.solve(hess, grad))[0, 0]

        # Update x using the Newton step and the step size
        x = x - 0.1 * step_size * np.linalg.solve(hess, grad)

        val[k] = f(x)
        x_history.append(x.flatten())

        print(f"{k}\t\t{x.flatten()}\t{val[k]}")

        # Check for convergence
        if np.linalg.norm(grad) < tol:
            break
        
        if val[k-1] < val[k] :
            print("NON CONVERGENCE")
            break

    return x, val[:k+1], np.array(x_history)
