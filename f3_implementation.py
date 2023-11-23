import numpy as np
import matplotlib.pyplot as plt
from function_def import f3
from gradient_descent_gama import gradient_descent_gama
from gradient_descent_alpha import gradient_descent_alpha
from newton_method import newton_method
from back_track import back_track

x0_3 = np.array([1 , 2 , 2 , 2])
a = 0.2
b = 0.5
gama1 = 0.5
gama2 = 0.05
alpha = 0.0005
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = np.linspace(-0.2, 2.5, 100)
max_iteration = 2000

minimizer_f3_gama, values_f3_gama, x_history_f3_gama = gradient_descent_gama(f3 ,x0_3, max_iteration, 5,gama1)
minimizer_f3_gama2, values_f3_gama2, x_history_f3_gama2 = gradient_descent_gama(f3 ,x0_3, max_iteration, 5,gama2)
minimizer_f3_alpha, values_f3_alpha, x_history_f3_alpha = gradient_descent_alpha(f3, x0_3, max_iteration, 0.1, alpha)
minimizer_f3_new, values_f3_new, x_history_f3_new = newton_method(f3, x0_3, max_iteration, 0.1)
minimizer_f3_back, values_f3_back, x_history_f3_back = back_track(f3, x0_3, max_iteration, 1e-10, a, b, 20)


print("the minimum value with the constant gama method is:")
print(np.min(values_f3_gama))
print("constant gama iteration count" , len(x_history_f3_gama))

print("the minimum value with the constant alpha method is:")
print(np.min(values_f3_alpha))
print("constant alpha iteration count" , len(x_history_f3_alpha))

print("the minimum value with the newton's method is:")
print(np.min(values_f3_new))
print("newton's iteration count"  , len(x_history_f3_new))

print("the minimum value with the backtrack method is:")
print(np.min(values_f3_back))
print("backtrack iteration count"  , len(x_history_f3_back))

def plot_values_and_quiver(algorithm_name, minimizer, values, x_history, color, label):
    plt.scatter(minimizer[0], minimizer[1], s=20, label=f'Minimum({algorithm_name})', edgecolors='black')

plot_values_and_quiver("Gamma1", minimizer_f3_gama, values_f3_gama, x_history_f3_gama, 'blue', 'Gamma Method with step size of 1')
plt.plot(range(len(values_f3_gama)), values_f3_gama, label='Values (Gamma1)')

plot_values_and_quiver("Gamma2", minimizer_f3_gama2, values_f3_gama2, x_history_f3_gama2, 'purple', 'Gamma Method with step size of 0.1')
plt.plot(range(len(values_f3_gama2)), values_f3_gama2, label='Values (Gamma2)')

plot_values_and_quiver("Alpha", minimizer_f3_alpha, values_f3_alpha, x_history_f3_alpha, 'red', 'Alpha Method')
plt.plot(range(len(values_f3_alpha)), values_f3_alpha, label='Values (Alpha)')

# plot_values_and_quiver("Newton", minimizer_f3_new, values_f3_new, x_history_f3_new, 'pink', 'Newton Method')
# plt.plot(range(len(values_f3_new)), values_f3_new, label='Values (Newton)')

plot_values_and_quiver("Backtracking", minimizer_f3_back, values_f3_back, x_history_f3_back, 'black', 'Backtracking Line Search')
plt.plot(range(len(values_f3_back)), values_f3_back, label='Values (Backtracking)')

plt.xlabel('Iteration (k)')
plt.ylabel('Function Value')
plt.xlim(0, 100)
plt.legend()
plt.show()
