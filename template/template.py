#
# template.py
#
# Using MCMC on CUDA GPU through the Python class imgpu.Mcgpu.
#

import imgpu
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title

#
# The following three arrays MUST be determined and passed to imgpu.Mcgpu.
#

pdescr1 = array((1,  1),  dtype=int32)
pmint1 = array((-5., -5.), dtype=float32)
pmaxt1 = array((5., 5.),   dtype=float32)

#
# Create the 'solver' object of the 'Mcgpu' class.
# At creation, all the variables and arrays needed for the MCMC algorithm
# work are created and initialized. They are the solver 'attributes'
# and they are accessible as solver.<attributte-name>.
#

solver = imgpu.Mcgpu(pdescr=pdescr1, pmint=pmint1, pmaxt=pmaxt1)

#
# The burnin_and_search() method does everything. 
#

solver.burnin_and_search()

solver.reset_gpu()

#
# The best solution is at the lowest chi2
#

chi2 = solver.chi2
im = chi2.argmin()   # Indices of the chi^2 minimums 
pout = solver.pout

print 'The minimum chi^2 at chi2[%d] = %f' % (im, chi2[im])
print 'pout[:,%d] = ' % (im), pout[:,im]


#
# Histograms of the all the parameters "random walk" are sometimes
# really helpful. Below is the case of only two parameters, x and y.
#
											  
figure(figsize=(12,6))
subplot(121); hist(pout[0,:], 50, color='b'); grid(1)
xlabel('x')
title(r'MCMC Output Distribution for X')

subplot(122); hist(pout[1,:], 50, color='b'); grid(1)
xlabel('y')
title(r'MCMC Output Distribution for Y')


show()
