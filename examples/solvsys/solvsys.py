#
# solvsys.py
#
# Find two pairs of the roots of system of equations
#
#    x^2 + y^2 = 1
# -0.25x + y = 0.5
#
# using MCMC on CUDA GPU through the Python class imgpu.
#

import imgpu
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title, xlim, ylim, legend


pdescr1 = array((1,  1),  dtype=int32)
pmint1 = array((-5., -5.), dtype=float32)
pmaxt1 = array((5., 5.),   dtype=float32)


coor1 =  array((2, 2, 0.25, 0.5), dtype=float32)
dat1 =   array((1, 0.5),          dtype=float32)


#
# Create the 'solver' object of the 'Mcgpu' class.
# At creation, all the variables and arrays needed for the MCMC algorithm
# work are created and initialized. They are the solver 'attributes'
# and they are accessible as solver.<attributte-name>.
#

solver = imgpu.Mcgpu(coor=coor1, dat=dat1, \
				 pdescr=pdescr1, pmint=pmint1, pmaxt=pmaxt1, \
				 nbeta=32, nseq=14, nburn=400, niter=5000)

solver.burnin_and_search()

solver.reset_gpu()

chi2 = solver.chi2
im = chi2.argmin()   # Indices of the chi^2 minimums 
pout = solver.pout

print 'The minimum chi^2 at chi2[%d] = %f' % (im, chi2[im])
print 'pout[:,%d] = ' % (im), pout[:,im]


#
# Exact roots:
#
# (rx1,ry1) = (-0.966, 0.2585) and (rx2,ry2) = (0.7307, 0.6827)
#
rx1 = (-4. - sqrt(832.))/34.
rx2 = (-4. + sqrt(832.))/34.
ry1 = sqrt(1. - rx1**2)
ry2 = sqrt(1. - rx2**2)

											  
figure(figsize=(12,6))
subplot(121); hist(pout[0,:], 50, color='b'); grid(1)
hlo = ylim()[0]
hup = ylim()[1]
plot([rx1,rx1], [hlo, hup], 'r--', lw=2, label='x roots')
plot([rx2,rx2], [hlo, hup], 'r--', lw=2)
xlabel('x')
title(r'MCMC Output Distribution for X')
legend()

subplot(122); hist(pout[1,:], 50, color='b'); grid(1)
hlo = ylim()[0]
hup = ylim()[1]
plot([ry1,ry1], [hlo, hup], 'r--', lw=2, label='y roots')
plot([ry2,ry2], [hlo, hup], 'r--', lw=2)
xlabel('y')
title(r'MCMC Output Distribution for Y')
legend()

x1 = linspace(-1., 1., 1000)
x2 = linspace(-1.25, 1.25, 100)

y1 = sqrt(1 - x1**2)
y2 = .25*x2 + .5



figure(figsize=(8,5))

plot(x1, y1, x2, y2)   # Curve intersection
plot(rx1, ry1, 'or')
plot(rx2, ry2, 'or')
axis('equal'); grid(1)

ylo = ylim()[0]
yup = ylim()[1]
xle = xlim()[0]
xri = xlim()[1]
plot([rx1,rx1], [ylo, ry1], 'r--', lw=1)
plot([rx2,rx2], [ylo, ry2], 'r--', lw=1)
plot([xle,rx1], [ry1, ry1], 'r--', lw=1)
plot([xle,rx2], [ry2, ry2], 'r--', lw=1)

xlabel('x'); ylabel('y'); 
title('Curve intersections are the roots')

#
# Print the best root pairs found
#
imin = where(chi2 < 0.0002)[0]
chi2_min = chi2[imin]
pout_best = pout[:,imin]
imsrt = argsort(chi2_min)
chi2_min = chi2_min[imsrt]
pout_best = pout_best[:,imsrt]
n_best = len(chi2_min)

print
print 'Exact root pairs:'
print '(x1,y1) = (%10f,%10f) and (x2,y2) = (%10f,%10f)' % (rx1, ry1, rx2, ry2)
print
print 'Best root pairs found:'
print '     x           y       chi^2'
for im in xrange(n_best):
	print '%10f %10f %10f' % (pout_best[0,im], pout_best[1,im], chi2_min[im])


show()
