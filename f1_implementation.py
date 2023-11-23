import numpy as np
import matplotlib.pyplot as plt
from function_def import f1
from gradient_descent_gama import gradient_descent_gama
from gradient_descent_alpha import gradient_descent_alpha
from steepset_descent import steepset_descent
from newton_method import newton_method
from back_track import back_track
from contour_def import plot_contour_1

# Initial parameters
x0_1 = np.array([23, 37])
x0_1_p = np.array([0.4, -3.4])
Q = np.array([[48, 12], [8, 8]])
gama1=1
gama2 = 0.1
alpha = 0.01

# Define range for contour plot
x_v = np.linspace(-20, 25, 100)
y_v = np.linspace(-8, 45, 100)


plot_contour_1(Q, np.array([[13], [23]]), 4, x_v, y_v, levels=50)

minimizer_f1_gama, values_f1_gama, x_history_f1_gama = gradient_descent_gama(f1 ,x0_1, 1000, 1e-10,gama1)
minimizer_f1_gama2, values_f1_gama2, x_history_f1_gama2 = gradient_descent_gama(f1 ,x0_1, 1000, 1e-10,gama2)
plt.quiver(x_history_f1_gama[:-1, 0], x_history_f1_gama[:-1, 1], x_history_f1_gama[1:, 0] - x_history_f1_gama[:-1, 0], x_history_f1_gama[1:, 1] - x_history_f1_gama[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='blue', width=0.005, headwidth=4, label='Gamma Method with step size of 1')
plt.scatter(minimizer_f1_gama[0], minimizer_f1_gama[1], s=20, label='Minimum(gama1)', edgecolors='black')


plt.quiver(x_history_f1_gama2[:-1, 0], x_history_f1_gama2[:-1, 1], x_history_f1_gama2[1:, 0] - x_history_f1_gama2[:-1, 0], x_history_f1_gama2[1:, 1] - x_history_f1_gama2[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='purple', width=0.005, headwidth=2, label='Gamma Method with step size of 0.1')
plt.scatter(minimizer_f1_gama[0], minimizer_f1_gama[1], s=20, label='Minimum(gama2)', edgecolors='black')


minimizer_f1_alpha, values_f1_alpha, x_history_f1_alpha = gradient_descent_alpha(f1, x0_1, 1000, 1e-10, alpha)
plt.quiver(x_history_f1_alpha[:-1, 0], x_history_f1_alpha[:-1, 1], x_history_f1_alpha[1:, 0] - x_history_f1_alpha[:-1, 0], x_history_f1_alpha[1:, 1] - x_history_f1_alpha[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='red', width=0.005, headwidth=4, label='Alpha Method')
plt.scatter(minimizer_f1_alpha[0], minimizer_f1_alpha[1], s=20, label='Minimum(alpha)', edgecolors='black')

minimizer_f1_steep, values_f1_steep, x_history_f1_steep = steepset_descent(f1, x0_1, 1000, 1e-10, Q)
plt.quiver(x_history_f1_steep[:-1, 0], x_history_f1_steep[:-1, 1], x_history_f1_steep[1:, 0] - x_history_f1_steep[:-1, 0], x_history_f1_steep[1:, 1] - x_history_f1_steep[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='green', width=0.005, headwidth=3, label='Steepset Descent')
plt.scatter(minimizer_f1_steep[0], minimizer_f1_steep[1], s=20, label='Minimum(steep)', edgecolors='black')

minimizer_f1_new, values_f1_new, x_history_f1_new = newton_method(f1, x0_1_p, 1000, 1e-10)
# plt.quiver(x_history_f1_new[:-1, 0], x_history_f1_new[:-1, 1], x_history_f1_new[1:, 0] - x_history_f1_new[:-1, 0], x_history_f1_new[1:, 1] - x_history_f1_new[:-1, 1],
#            scale_units='xy', angles='xy', scale=1, color='pink', width=0.005, headwidth=3, label='newton method')
# plt.scatter(minimizer_f1_new[0], minimizer_f1_new[1], s=20, label='Minimum(newton)', edgecolors='black')

minimizer_f1_back, values_f1_back, x_history_f1_back = back_track(f1, x0_1, 1000, 1e-10, 0.2 , 0.5 , 1)
plt.quiver(x_history_f1_back[:-1, 0], x_history_f1_back[:-1, 1], x_history_f1_back[1:, 0] - x_history_f1_back[:-1, 0], x_history_f1_back[1:, 1] - x_history_f1_back[:-1, 1],
           scale_units='xy', angles='xy', scale=1, color='black', width=0.005, headwidth=3, label='Back Track')
plt.scatter(minimizer_f1_back[0], minimizer_f1_back[1], s=20, label='Minimum(back)', edgecolors='black')


plt.legend()
plt.show()
print("the minimum value with the gama gama method is:")
print(np.min(values_f1_gama))
print("gama gama iteration count" , len(x_history_f1_gama))

print("the minimum value with the gama alpha method is:")
print(np.min(values_f1_alpha))
print("gama alpha iteration count" , len(x_history_f1_alpha))

print("the minimum value with the Steepset Descent method is:")
print(np.min(values_f1_steep))
print("Steep set descent iteration count"  , len(x_history_f1_steep))

print("the minimum value with the newton's method is:")
print(np.min(values_f1_new))
print("newton's iteration count"  , len(x_history_f1_steep))