import numpy as np
import matplotlib.pyplot as plt
from function_def import f2
from gradient_descent_gama import gradient_descent_gama
from gradient_descent_alpha import gradient_descent_alpha
from newton_method import newton_method
from back_track import back_track
from contour_def import plot_contour_2

x0_2 = np.array([1, 2])
a_value = 2
b_value = 150
gama1 = 0.5
gama2 = 0.05
alpha = 0.0005
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = np.linspace(-0.2, 2.5, 100)

plot_contour_2(a_value, b_value, x_vals, y_vals, levels=100)
minimizer_f2_gama, values_f2_gama, x_history_f2_gama = gradient_descent_gama(f2 ,x0_2, 2000, 5,gama1)
minimizer_f2_gama2, values_f2_gama2, x_history_f2_gama2 = gradient_descent_gama(f2 ,x0_2, 2000, 5,gama2)


plt.quiver(x_history_f2_gama[:-1, 0], x_history_f2_gama[:-1, 1], x_history_f2_gama[1:, 0] - x_history_f2_gama[:-1, 0], x_history_f2_gama[1:, 1] - x_history_f2_gama[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='blue', width=0.005, headwidth=3, label='Gama Method with the constant step size of 0.5')
plt.scatter(minimizer_f2_gama[0], minimizer_f2_gama[1], s=20, label='Minimum', edgecolors='black')

plt.quiver(x_history_f2_gama2[:-1, 0], x_history_f2_gama2[:-1, 1], x_history_f2_gama2[1:, 0] - x_history_f2_gama2[:-1, 0], x_history_f2_gama2[1:, 1] - x_history_f2_gama2[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='purple', width=0.005, headwidth=2, label='Gama Method with the constant step size pf 0.05')
plt.scatter(minimizer_f2_gama[0], minimizer_f2_gama[1], s=20, label='Minimum', edgecolors='black')

minimizer_f2_alpha, values_f2_alpha, x_history_f2_alpha = gradient_descent_alpha(f2, x0_2, 2000, 0.1, alpha)
plt.quiver(x_history_f2_alpha[:-1, 0], x_history_f2_alpha[:-1, 1], x_history_f2_alpha[1:, 0] - x_history_f2_alpha[:-1, 0], x_history_f2_alpha[1:, 1] - x_history_f2_alpha[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='red', width=0.005, headwidth=1, label='Alpha Method')
plt.scatter(minimizer_f2_alpha[0], minimizer_f2_alpha[1], s=20, label='Minimum', edgecolors='black')

minimizer_f2_new, values_f2_new, x_history_f2_new = newton_method(f2, x0_2, 2000, 0.1)
# plt.quiver(x_history_f2_new[:-1, 0], x_history_f2_new[:-1, 1], x_history_f2_new[1:, 0] - x_history_f2_new[:-1, 0], x_history_f2_new[1:, 1] - x_history_f2_new[:-1, 1],
#            scale_units='xy', angles='xy', scale=1, color='pink', width=0.005, headwidth=1, label='Newton Method')
# plt.scatter(minimizer_f2_alpha[0], minimizer_f2_alpha[1], s=20, label='Minimum', edgecolors='black')

minimizer_f2_back, values_f2_back, x_history_f2_back = back_track(f2, x0_2, 2000, 1e-10, 0.1, 0.5, 20)
plt.quiver(x_history_f2_back[:-1, 0], x_history_f2_back[:-1, 1], x_history_f2_back[1:, 0] - x_history_f2_back[:-1, 0], x_history_f2_back[1:, 1] - x_history_f2_back[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='black', width=0.005, headwidth=3, label='back track')
plt.scatter(minimizer_f2_back[0], minimizer_f2_back[1], s=20, label='Minimum(back)', edgecolors='black')



plt.legend()

plt.show()

print("the minimum value with the constant gama method is:")
print(np.min(values_f2_gama))
print("gama gama iteration count" , len(x_history_f2_gama))

print("the minimum value with the constant alpha method is:")
print(np.min(values_f2_alpha))
print("gama alpha iteration count" , len(x_history_f2_alpha))

print("the minimum value with the newton's method is:")
print(np.min(values_f2_new))
print("newton's iteration count"  , len(x_history_f2_new))