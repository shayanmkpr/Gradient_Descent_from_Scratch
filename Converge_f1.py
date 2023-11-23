import numpy as np
import matplotlib.pyplot as plt
from function_def import f1
from gradient_descent_gama import gradient_descent_gama
from gradient_descent_alpha import gradient_descent_alpha
from steepset_descent import steepset_descent
from newton_method import newton_method
from back_track import back_track

# Initial parameters
x0_1 = np.array([23, 37])
x0_1_p = np.array([0.4, -3.4])
Q = np.array([[48, 12], [8, 8]])
gama1=1
gama2 = 0.1
alpha = 0.01

def plot_values_and_quiver(algorithm_name, minimizer, values, x_history, color, label):
    plt.scatter(minimizer[0], minimizer[1], s=20, label=f'Minimum({algorithm_name})')

minimizer_f1_gama, values_f1_gama, x_history_f1_gama = gradient_descent_gama(f1, x0_1, 100, 1e-10, gama1)
plot_values_and_quiver("Gamma1", minimizer_f1_gama, values_f1_gama, x_history_f1_gama, 'blue', 'Gamma Method with step size of 1')
plt.plot(range(len(values_f1_gama)), values_f1_gama, label='Values (Gamma1)')

minimizer_f1_gama2, values_f1_gama2, x_history_f1_gama2 = gradient_descent_gama(f1, x0_1, 100, 1e-10, gama2)
plot_values_and_quiver("Gamma2", minimizer_f1_gama2, values_f1_gama2, x_history_f1_gama2, 'purple', 'Gamma Method with step size of 0.1')
plt.plot(range(len(values_f1_gama2)), values_f1_gama2, label='Values (Gamma2)')

minimizer_f1_alpha, values_f1_alpha, x_history_f1_alpha = gradient_descent_alpha(f1, x0_1, 100, 1e-10, alpha)
plot_values_and_quiver("Alpha", minimizer_f1_alpha, values_f1_alpha, x_history_f1_alpha, 'red', 'Alpha Method')
plt.plot(range(len(values_f1_alpha)), values_f1_alpha, label='Values (Alpha)')

minimizer_f1_steep, values_f1_steep, x_history_f1_steep = steepset_descent(f1, x0_1, 100, 1e-10, Q)
plot_values_and_quiver("Steepset", minimizer_f1_steep, values_f1_steep, x_history_f1_steep, 'green', 'Steepest Descent')
plt.plot(range(len(values_f1_steep)), values_f1_steep, label='Values (Steepest)')

minimizer_f1_new, values_f1_new, x_history_f1_new = newton_method(f1, x0_1, 100, 1e-10)
# Uncomment the following two lines if you want to plot for Newton's Method
# plot_values_and_quiver("Newton", minimizer_f1_new, values_f1_new, x_history_f1_new, 'pink', 'Newton Method')
# plt.plot(range(len(values_f1_new)), values_f1_new, label='Values (Newton)')

minimizer_f1_back, values_f1_back, x_history_f1_back = back_track(f1, x0_1, 100, 1e-10, 0.4, 0.5, 1)
plot_values_and_quiver("Backtracking", minimizer_f1_back, values_f1_back, x_history_f1_back, 'black', 'Backtracking Line Search')
plt.plot(range(len(values_f1_back)), values_f1_back, label='Values (Backtracking)')

plt.xlabel('Iteration (k)')
plt.ylabel('Function Value')
# plt.ylim(-33, -27)
# plt.xlim(0, 100)
plt.legend()
plt.show()