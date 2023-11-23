import numpy as np
from gradient import gradient

def steepset_descent(f , x0,max_iter, tol ,Q):
    x = x0.reshape(-1, 1)
    input_size = len(x)
    val = np.zeros(max_iter)
    val[0] = f(x)

    x_history = [x.flatten()]

    print("Steepset Descent")
    print("Iteration\tX Value\t\t\t\tFunction Value")
    print(f"0\t\t{x.flatten()}\t\t{val[0]}")

    for k in range(1, max_iter):
        # Using a constant step norm (gama)
        grad = gradient(f , x , input_size)
        step = (np.transpose(grad) @ grad)/(np.transpose(grad) @ Q @ grad)
        x = x - step * gradient(f, x, input_size)
        val[k] = f(x)

        x_history.append(x.flatten())

        print(f"{k}\t\t{x.flatten()}\t{val[k]}")

        # Check for convergence
        if np.linalg.norm(gradient(f, x, input_size)) < tol:
            break

        if val[k-1] < val[k] :
            print("NON CONVERGENCE")
            break

    return x, val[:k+1], np.array(x_history)