#
# search_Rosenbrock.py
#
# Using MCMC-RE on GPU finds the global minimum of the Rosenbrock function
# (aka "Rosenbrock valley" or "Rosenbrock banana function").
#
# The global minimum, z = 0, is at (1,1) inside a long, narrow, parabolic
# shaped flat valley. To find the valley is trivial. To converge to the
# global minimum, however, is a difficult optimization task.
#
# The function is defined by
#    f(x,y)=(1 - x)^2 + 100(y - x^2)^2
#


import imgpu
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort, meshgrid, \
	 argsort, where, logspace
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title, cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed


#
# Only 5 arrays need to be specified:
#

pdescr1 = array((1,  1),  dtype=int32)
pmint1 = array((-10., -10.), dtype=float32)
pmaxt1 = array((10., 10.),   dtype=float32)

coor1 =  array((2, 2, 0.25, 0.5), dtype=float32) # Coeefficients used by model
dat1 =   zeros(1, dtype=float32) # Rosenbrok function will be compared with zero

#
# Create the 'rosen' object of the 'Mcgpu' class.
# At creation, all the variables and arrays needed for the MCMC algorithm
# work are created and initialized. They are the rosen 'attributes'
# and they are accessible as rosen.<attributte-name>.
#

rosen = imgpu.Mcgpu(coor=coor1, dat=dat1, \
				 pdescr=pdescr1, pmint=pmint1, pmaxt=pmaxt1, \
				 nbeta=32, nseq=14, nburn=400, niter=1000)

rosen.burnin_and_search()

rosen.reset_gpu()

chi2 = rosen.chi2
im = chi2.argmin()   # Indices of the chi^2 minimums 
pout = rosen.pout

print 'The minimum chi^2 at chi2[%d] = %f' % (im, chi2[im])
print 'pout[%d] = ' % (im), pout[:,im]

#
# Print the best root pairs found
#
imin = where(chi2 < 0.00001)[0]
chi2_min = chi2[imin]
pout_best = pout[:,imin]
imsrt = argsort(chi2_min)
chi2_min = chi2_min[imsrt]
pout_best = pout_best[:,imsrt]
n_best = len(chi2_min)

print
print 'Exact solution: (x,y) = (%10f,%10f)' % (1, 1)
print
print 'Best minimum found:'
print '     x           y       chi^2'
for im in xrange(n_best):
	print '%10f %10f %10f' % (pout_best[0,im], pout_best[1,im], chi2_min[im])



figure(figsize=(12,6))
subplot(121); hist(pout[0,:], 50, color='b'); grid(1)
xlabel(r'$x$', fontsize=20)
title('MCMC Output Distribution for X')

subplot(122); hist(pout[1,:], 50, color='b'); grid(1)
xlabel(r'$y$', fontsize=20)
title('MCMC Output Distribution for Y')

#
# Plot contours and 3D surface of the Rosenbrock function
#

ros = lambda x, y: (1 - x)**2 + 100.*(y - x**2)**2

x = linspace(-3., 3., 1000)
y = linspace(-1.5, 10., 1000)
X, Y = meshgrid(x, y)

z = ros(X, Y)

v = logspace(-0.5, 3.5, 10)

fig = plt.figure(figsize=(14,6))
ax2 = fig.add_subplot(121)
cs = ax2.contour(X, Y, z, v, cmap=cm.coolwarm)
ax2.plot(1, 1, 'ro')
ax2.set_xlabel(r'$x$', fontsize=20)
ax2.set_ylabel(r'$y$', fontsize=20)
ax2.set_title(r'', fontsize=20)
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

