#
# plot_Rosenbrock
#
# Plot the Rosenbrock's valley or Rosenbrock's banana function
#
# z = (1 - x)^2 + 100(y - x^2)^2
#
#

# import numpy as np
# import matplotlib.pyplot as plt
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed

ros = lambda x, y: (1 - x)**2 + 100.*(y - x**2)**2

x = linspace(-3., 3., 1000)
y = linspace(-1.5, 10., 1000)
X, Y = meshgrid(x, y)

z = ros(X, Y)

v = logspace(-0.5, 3.5, 10)

fig = plt.figure(figsize=(14,6))
ax2 = fig.add_subplot(121)
#ax2.contour(X, Y, z, 500, cmap=cm.coolwarm)
cs = ax2.contour(X, Y, z, v, cmap=cm.coolwarm)
#ax2.clabel(cs, v[5:], colors='k', fmt='%3.0f')
ax2.plot(1, 1, 'ro')
ax2.set_xlabel(r'$x$', fontsize=20)
ax2.set_ylabel(r'$y$', fontsize=20)
ax2.set_title(r'', fontsize=20)
plot(x, x**2, 'g')
grid(1);

x = linspace(-3., 3., 40)
y = linspace(-1.5, 10., 40)
X, Y = meshgrid(x, y)

z = ros(X, Y)

ax3 = fig.add_subplot(122, projection='3d')
ax3.plot_surface(X, Y, z, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=0.8)
ax3.plot(x, x**2, (1-x)**2, color='b', lw=2)
ax3.plot([1], [1], [0], markerfacecolor='r', marker='o') #, markersize=10)
ax3.set_xlabel(r'$x$', fontsize=20)
ax3.set_ylabel(r'$y$', fontsize=20)
ax3.set_zlabel(r'$z$', fontsize=20)
fig.text(.43, 0.95, r'Rosenbrock Valley', fontsize=20, family='serif')


show()



