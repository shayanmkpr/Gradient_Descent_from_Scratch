import numpy as np
from gradient import gradient

def back_track(f, x0, max_iter, tol, a, b, t):
    x = x0.reshape(-1, 1)
    input_size = len(x)
    val = np.zeros(max_iter)
    val[0] = f(x)
    x_history = [x.flatten()]
    print("backtracking")
    print("Iteration\tX Value\t\t\t\tFunction Value")
    print(f"0\t\t{x.flatten()}\t\t{val[0]}")
    t0 = t
    for k in range(1, max_iter):
        grad = gradient(f, x, input_size)

        # Backtracking line search
        step = t0
        while f(x - step * grad) > f(x) - a * step * np.linalg.norm(grad)**2:
            step = b * step

        # Update x
        x = x - step * grad
        val[k] = f(x)
        t = t0
        x_history.append(x.flatten())
        print(f"{k}\t{x.flatten()}\t\t\t\t{val[k]}")
        # Check for convergence
        if np.linalg.norm(step * grad) < tol:
            break

    return x, val[:k+1], np.array(x_history)