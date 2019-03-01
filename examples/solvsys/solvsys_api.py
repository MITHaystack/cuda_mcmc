#
# solvsys_api.py
#

import mcmc_interf as mi
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title, xlim, ylim

nseq = int32(14)   # Number of processing cores (CUDA blocks)
nbeta = int32(32)  # Number of temperatures
niter = int32(5000)
nburn = int32(400) # Number of initial iters ignored due to the transience
nstates = int32(nseq*nbeta)
nseqit = int32(nseq*niter)
seed = uint64(np.trunc(1e6*time.time()%(10*nstates)))
imodel = 0

pdescr = array((1,  1),  dtype=int32)
ptotal = array((0., 0.), dtype=float32)
nptot = len(ptotal)

#pmint = array((-5., -5.), dtype=float32)
#pmaxt = array((5., 5.),   dtype=float32)
pmint = -5*ones_like(ptotal)
pmaxt =  5*ones_like(ptotal)

# ivar[nprm] maps prm[0:nprm] on ptotal[0:nptot] (not needed here )
ivar = where(pdescr != 0)[0] # Maps prm[0:nprm] on ptotal[0:nptot]
ivar = ivar.astype(int32)
nprm = len(ivar)
pmin = pmint[ivar]
pmax = pmaxt[ivar]
# invar[nptot] maps ptotal[0:nptot] on prm[0:nprm] (not needed here )
invar = zeros(nptot, dtype=int32)
ip = 0
for i in xrange(nptot):
    if pdescr[i] != 0: 
        invar[i] = ip
        ip = ip + 1
    else:
        invar[i] = -1

nadj = int32(100)
beta1 = float32(1.)
betan = float32(0.0001)   # ~exp(chi2/2)

npass = 1

coor =  array((2, 2, 0.25, 0.5), dtype=float32); ncoor = len(coor)
icoor = zeros(1,                 dtype=int32);   nicoor = len(coor)
dat =   array((1, 0.5),          dtype=float32); ndat = len(dat)
#dat =   array((1, 0.5**2),          dtype=float32); ndat = len(dat)
std2r = ones_like(dat,           dtype=float32)
idat =  zeros(1,                 dtype=int32);   nidat = len(idat)
datm =  zeros((nbeta*nseq*ndat), dtype=float32); ndatm = len(datm)
chi2m = zeros((nbeta,nseq,ndat), dtype=float32)
rndst = zeros((nbeta,nseq,48),   dtype=uint8)

beta = zeros((nbeta), dtype=float32)
pstp = zeros((nprm,nbeta,nseq), dtype=float32)  # Adaptive step size
pout = zeros((nprm,nbeta,nseq,niter), dtype=float32)
tout = zeros((nbeta,nseq,niter), dtype=int32)
chi2 = zeros((nseq,niter), dtype=float32)
flag = zeros((nbeta,nseq), dtype=int32)  
tcur = zeros((nbeta,nseq), dtype=int32)  
chi2c = zeros((nbeta,nseq), dtype=float32)  # old chi^2 array
n_cnt =  zeros((nprm,nbeta,nseq), dtype=int32) # number of Metropolis trials
n_acpt = zeros((nprm,nbeta,nseq), dtype=int32)
n_exch = zeros((nbeta,nseq), dtype=int32)
n_hist = zeros((nadj,nprm,nbeta,nseq), dtype=int32) # accept/reject history
ptentn = zeros((nbeta,nseq), dtype=int32)   # Tentative parameter indices
ptent =  zeros((nbeta,nseq), dtype=float32) # Tentative parameters
pcur = zeros((nbeta,nseq,nptot), dtype=float32) # Accepted param. set

# Fill in the working storage for threads calculating model visibilities
for i in xrange(nbeta):
    for j in xrange(nseq):
        pcur[i,j,:] = ptotal
        for k in xrange(nprm):
            pcur[i,j,ivar[k]] = pmin[k] + (pmax[k] - pmin[k])/2.

# Initialize parameter steps
pmid = (pmax - pmin)/50.
for i in xrange(nprm):
    pstp[i,:,:] = pmid[i]

print 'pstp[:,0,0] = ', pstp[:,0,0] 

# Initialize tcur
n0_nbeta = arange(nbeta) # 0..nbeta-1
for iseq in xrange(nseq):
    tcur[:,iseq] = n0_nbeta

# Initialize temperatures
if nbeta == 1:
    bstp = 1.0  # For debugging only
else:
    bstp = (betan/beta1)**(1.0/float(nbeta-1))
for i in xrange(nbeta):
    beta[i] = beta1*bstp**i

# Initialize counter
n_cnt[:] = float32(1.)

print 'std2r = ', std2r, ', idat = ', idat, ', nidat = ', nidat

#sys.exit(0)

mi.mcmcuda(coor, dat, std2r, icoor, idat, datm, chi2m, \
           pdescr, ivar, invar, ptotal, pmint, pmaxt, \
           pout, tout, chi2, chi2c, tcur, flag, \
           n_acpt, n_exch, n_cnt, n_hist, beta, \
           pstp, ptent, ptentn, pcur, rndst, \
           uint64(seed), int32(imodel), \
           int32(ncoor), int32(ndat), int32(nicoor), int32(nidat), \
		   int32(ndatm), int32(nptot), int32(nprm), \
           int32(nadj), int32(npass), int32(nbeta), \
		   int32(nseq), int32(nburn), int32(niter))

mi.reset_gpu()


c2 = chi2.flatten()
po = pout[:,0,:,:].reshape((nprm,nseq*niter))

im = c2.argmin()   # Indices of the chi^2 minimums 


print
print 'The minimum chi^2 at chi2[%d] = %f' % (im, c2[im])
print 'pout[:,%d] = ' % (im), po[:,im]

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
subplot(121); hist(po[0,:], 50, color='b'); grid(1)
hlo = ylim()[0]
hup = ylim()[1]
plot([rx1,rx1], [hlo, hup], 'r--', lw=2, label='x roots')
plot([rx2,rx2], [hlo, hup], 'r--', lw=2)
xlabel('x')
title(r'MCMC Output Distribution for X')

subplot(122); hist(po[1,:], 50, color='b'); grid(1)
hlo = ylim()[0]
hup = ylim()[1]
plot([ry1,ry1], [hlo, hup], 'r--', lw=2, label='y roots')
plot([ry2,ry2], [hlo, hup], 'r--', lw=2)
xlabel('y')
title(r'MCMC Output Distribution for Y')

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
imin = where(c2 < 0.0002)[0]
chi2_min = c2[imin]
pout_best = po[:,imin]
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
