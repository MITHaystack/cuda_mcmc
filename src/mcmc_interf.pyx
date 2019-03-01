#
# Module mcmc
# Contains interface between Python and C code
#
# sizeof(curandState) = 48 B
#

#
# The Mcgpu class is a problem-specific tool for fitting a model to the
# Event Horizon Telescope observation data. It finds an array of vectors
# of the model parameters Mcgpu.pout[] and provides a corresponding array
# of Mcgpu.chi^2[] values. The optimal model fit to the observation data is
# found in Mcgpu.pout[] at the same location as the minimum chi^2 value.
#
#
# The Mcgpu class is a higher-level wrapper for the lower-level classes
# CCalcModel and its inheritor CMcmcFit, defined in mcmcjob.cuh and mcmcjob.cu.
# The instantiation of the latter and calling the method
# CMcmcFit::do_mcmc_on_gpu() that runs the fitting on a CUDA GPU
# is made over the interface Python-C/C++ mcmc_interf.pyx written in
# Cython and compiled into mcmc_interf.c. 
#
#
# The MCMC algorithm is typically used for fitting parameterized models
# to large arrays of data located at many coordinate points. For example,
# it can be a model of celestial radio object observed with a radio
# interferometer. The data here are the visibilities, and the coordinates are
# u and v, the locations of visibilities in the spatial frequency domain.
# The data should be provided to the algorithm in the dat[ndat] array, and
# the coordinates -- in the coor[ncoor] array. Sometimes the model requires
# integer data, passed in the integer array idat[nidat], and integer
# coordinates (like antenna numbers) are passed in the icoor[nicoor] array.
# To compute the chi^2 the algorithm requires the variance std^2.
# The std2r[ndat] array should contain the reciprocals of the data variances.
#
# -----------------------------------------------------------------------------
#
# The MCMC with replica exchange is implemented as a function mcmcuda()
# in the module mcmc_interf. 
#
# Input parameters of the MCMC algorithm
#
# nptot: total number of the parameters to be optimized by MCMC. In other words,
#        the full problem dimensionality. Part of the parameters in the
#        ptotal[nptot] array can be made immutable for a particular MCMC run
#        by setting zeros at the corresponding locations in the descriptor array
#        pdescr[nptot].
#
# nprm: the number of parameters to be optimized by MCMC in the specific run.
#       It is possible to "freeze" values of the part of problem parameters
#       making them immutable constants, and optimize only the rest of them.
#       The frozen (immutable) parameters must be marked by zeros at their
#       positions in the descriptor array pdescr[nptot]. Thus nprm is the
#       number of non-zero elements in pdescr[].
#
# ndat: number of floating point data.
#
# ncoor: number of floating point coordinates.
#
# nidat: number of integer data.
#
# nicoor: number of integer coordinates.
# 
# nseq: number of independent parallel processes of model fitting. In CUDA
#       framework it means the number of thread blocks. In the Nvidia
#       GPU hardware one block of threads is executed on one
#       "Streaming Multiprocessor" (SM). For example, GTX 670 has 7 SMs. 
#
# nbeta: number of "temperatures" in the MCMC algorithm. In CUDA framework
#        it means the number of parallel theads per block. In Nvidia GPU
#        the treads in a SM are executed in units of "warps" of 32 threads
#        each executing the same instruction at a time.
#
# seed: an arbitrary 64-bit unsigned integer (np.uint64) seed for the CUDA
#       random number generator. 
# 
# pdescr[nptot]: array of parameter descriptors with possible values
#                2 - angular parameter (radians), 1 - nonangular parameter.
#                0 - this value excludes the corresponding parameter in
#                    ptotal[nptot] from optimization.
# ptotal[nptot]: array of model parameters. The ptotal[] values at
#                the locations where pdescr[] is nonzero are ignored.
#                The values where pdescr is zero are used in the model.
#                
# pmint[nptot],
# pmaxt[nptot]: minimum and maximum values for the model parameters.
#               The parameters are searched only inside of the
#               nptot-dimensional rectangular parallelepiped determined by
#               pmint[] and pmaxt[]. This parallelepiped determines the "prior".
#
# ivar[nprm]: maps [nprm] optimized parameters on ptotal[nptot].
#             ivar[nprm] has indices of the ptotal[nptot] parameters whose
#             descriptors in pdescr[nptot] are non-zero.
#
# invar[nptot]: maps ptotal[nptot] on the varied parameters [nprm]. If some
#               of the parameters are immutablem invar[] must contain -1 values
#               at the corresponding positions.
#
# beta[nbeta]: "temperature" values in descending order from 1. Recommended are
#              the values between 1 and 0.0001 falling off exponentially.
#
# dat[ndat]: input floating point data (e.g. visibilities, phases etc).
#
# idat[nidat]: input integer data.
#
# coor[ncoor]: input floating point coordinates (e.g. u and v pairs).
#
# icoor[nicoor]: integer coordinates (e.g. antenna numbers)
#              
# std2r[ndat]: the reciprocals of the data variances, 1/std^2.
#
# pcur[nbeta,nseq,nptot]: initial parameter values for all temperatures and
#                         all sequences. Can be zeros or random numbers or
#                         the coordinates of the prior center. During MCMC
#                         run is used for the current parameter values.
#
# tcur[nbeta,nseq]: integer values of the "temperature" indices. On entry it
#                   should contain nseq columns of cardinals from 0 up to
#                   nbeta-1.
#
# n_cnt[nprm,nbeta,nseq]: numbers of Metropolis trials for each mutable
#                         parameter, each temperature, and each sequence.
#                         Sould be initialized to ones.
#
# nadj: number of sets for calculating acceptance rates. Recommended value 100.
#
# npass: number of times the model() function must be called. Sometimes
#        the model computation requires several passes. The data saved in a
#        previous pass are used in the next one. The model() function
#        can determen at which pass it has been called by its parameter ipass
#        taking values from 0 to npass-1.
#
# pstp[nprm,nbeta,nseq]: initial parameter steps for each mutable parameter,
#                        each temperature, and each sequence.
# imodel: an integer parameter passed to the model() function. The usage is
#         arbitrary. For example, it may be a model number to select between
#         several models.
#
# nburn: number of the burn-in iterations. During the burn-in phase the steps
#        for each parameter, each temperature, and each sequence are adjusted,
#        and the transients fall off.
#
# niter: number of optimization operations.
#
# ndatm: must be equal nbeta*nseq*ndat
#
# -----------------------------------------------------------------------------
#
# Workspace arrays of MCMC algorithm
#
# datm[nbeta,nseq,ndat]: computed model data.
#
# chi2m[nbeta,nseq,ndat]: chi^2 terms for each model data element, each
#                         temperature, and each sequence.
#
# rndst[nbeta,nseq,48]: unsigned 8-bit integer (np.uint8) array of states
#                       for the random number generator.
# 
# flag[nbeta,nseq]: 0 means the selected parameter is outside of the prior,
#                   or the parameter did not pass the alpha-test.
#                   1 means the selected parameter is OK.
# chi2c[nbeta,nseq]: old chi^2 for each temperature and each sequence.
#
# ptent[nbeta,nseq]: tentative parameter values for each temperature and each
#                    sequence.
#
# ptentn[nbeta,nseq]: tentative parameter indices for each temperature and each
#                     sequence.
#
# -----------------------------------------------------------------------------
#
# Output parameters of MCMC algorithm
#
# pout[nprm,nbeta,nseq,niter]: mutable parameters found by the MCMC for
#                              all the temperatures, sequences, and iterations.
#                              As a rule, only the coldest temperature is used,
#                              i.e. pout[nprm,0,nseq,niter].
#                              The optimum parameter set is in pout[] at the
#                              location corresponding to the minimum chi^2
#                              in the chi2[nseq,niter] array. Usually these
#                              arrays reshape to 1D form:
#                              po = pout[:,0,:,:].reshape((nprm,nseq*niter))
#                              c2 = chi2.flatten()
#                              and then the best parameter set is found as
#                              po[c2.argmin()]. Also, the histograms of each
#                              parameter can provide important information:
#                              hist(po[0,:], 50, color='b'); grid(1) - for
#                              the 0-th parameter and so on.
#
# chi2[nseq,niter]
#
# n_acpt[nprm,nbeta,nseq]: counting numbers of accepted proposal sets in the
#                          Metropolis-Hastings algorithm.
#
# n_exch[nbeta,nseq]: counting numbers of exchanged adjacent temperature
#                     chanins in the replica exchange algorithm.
#
# n_hist[nadj,nprm,nbeta,nseq]: counting numbers of "accept" of proposal set
#                               in Metropolis-Hastings algorithm
#
#


import numpy as np
cimport numpy as np

cdef extern from "/usr/local/cuda/include/cuda_runtime.h":
    void cudaDeviceReset()

# cdef extern from "mcmcjob.cuh":
#     pass

cdef extern from "mcmc.h":
    void mcmc_cuda(float *coor, float *dat, float *std2r,
            int *icoor, int *idat, float *datm, float *chi2m,    
            int *pdescr, int *ivar, int *invar, float *ptotal, 
            float *pmint, float *pmaxt, 
            float *pout, int *tout, float *chi2,
            float *chi2c, int *tcur, int *flag, 
            int *n_acpt, int *n_exch, int *n_cnt, int *n_hist,  
            float *beta, float *pstp, float *ptent, int *ptentn, 
            float *pcur, unsigned char *rndst,
                       
            unsigned long long seed, int imodel,
            int ncoor, int ndat, int nicoor, int nidat, int ndatm,
            int nptot, int nprm, int nadj, int npass,
            int nbeta, int nseq, int nburn, int niter)


cpdef mcmcuda(
    # Coordinates, Observables and Model Data, Integer Data 
    np.ndarray[np.float32_t,ndim=1] coor, #/* [ncoor]: Coordinates: UV etc */
    np.ndarray[np.float32_t,ndim=1] dat,  #/* [ndat]:  Observation data */
    np.ndarray[np.float32_t,ndim=1] std2r,#/* [ndat]:  Data std^2 recipr. */
    np.ndarray[np.int32_t,ndim=1]   icoor,#/* [nicoor]: Any integer coords */
    np.ndarray[np.int32_t,ndim=1]   idat, #/* [nidat]: Any integer data */
    np.ndarray[np.float32_t,ndim=1] datm, #/* [ndatm]: Model data */
    np.ndarray[np.float32_t,ndim=3] chi2m,#/* [nbeta][nseq][ndc2]: chi^2 terms*/
    # Parameters
    np.ndarray[np.int32_t,ndim=1]   pdescr,  #/* [nptot] */
    np.ndarray[np.int32_t,ndim=1]   ivar,    #/* [nprm] */
    np.ndarray[np.int32_t,ndim=1]   invar,   #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] ptotal,  #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] pmint,   #/* [nptot] */
    np.ndarray[np.float32_t,ndim=1] pmaxt,   #/* [nptot] */
    # MCMC Algorithm 
    np.ndarray[np.float32_t,ndim=4] pout,   #/* [nprm][nbeta][nseq][niter] */
    np.ndarray[np.int32_t,ndim=3]   tout,   #/* [nbeta][nseq][niter] */
    np.ndarray[np.float32_t,ndim=2] chi2,   #/* [nseq][niter] */
    np.ndarray[np.float32_t,ndim=2] chi2c,  #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   tcur,   #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   flag,   #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=3]   n_acpt, #/* [nprm][nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   n_exch, #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=3]   n_cnt,  #/* [nprm][nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=4]   n_hist, #/* [nadj][nprm][nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=1] beta,   #/* [nbeta] */
    np.ndarray[np.float32_t,ndim=3] pstp,   #/* [nprm][nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=2] ptent,  #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   ptentn, #/* [nbeta][nseq] */
    np.ndarray[np.float32_t,ndim=3] pcur,   #/* [nbeta][nseq][nptot] */ 
    # Random Number Generator States 
    np.ndarray[np.uint8_t,ndim=3]   rndst,  #/* [nbeta][nseq][48] */

    seed, imodel,
    ncoor, ndat, nicoor, nidat, ndatm, nptot, nprm,
    nadj, npass, nbeta, nseq, nburn, niter):
    
    mcmc_cuda(<float*>coor.data,   <float*>dat.data,    <float*>std2r.data,
                  <int*>icoor.data,    <int*>idat.data,     <float*>datm.data,
                  <float*>chi2m.data,  <int*>pdescr.data,   <int*>ivar.data,
                  <int*>invar.data,    <float*>ptotal.data, <float*>pmint.data,
                  <float*>pmaxt.data,  <float*>pout.data,   <int*>tout.data,
                  <float*>chi2.data,   <float*>chi2c.data,  <int*>tcur.data,
                  <int*>flag.data,     <int*>n_acpt.data,   <int*>n_exch.data,
                  <int*>n_cnt.data,    <int*>n_hist.data,   <float*>beta.data,
                  <float*>pstp.data,   <float*>ptent.data,  <int*>ptentn.data,
                  <float*>pcur.data,   <unsigned char*>rndst.data,
                  
                  seed, imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nprm,
                  nadj, npass, nbeta, nseq, nburn, niter)



cdef extern from "mcmc.h":
    void calc_modchi2(float *coor, float *dat, float *std2r,
	       int *icoor, int *idat, float *datm, float *chi2m,    
	       float *pcur, float *chi2c, int *flag, int imodel,
	       int ncoor, int ndat, int nicoor, int nidat, int ndatm,
	       int nptot, int nbeta, int nseq)



cpdef calcmodchi2(
    # Coordinates, Observables and Model Data, Integer Data 
    np.ndarray[np.float32_t,ndim=1] coor, #/* [ncoor]: Coordinates: UV etc */
    np.ndarray[np.float32_t,ndim=1] dat,  #/* [ndat]:  Observation data */
    np.ndarray[np.float32_t,ndim=1] std2r,#/* [ndat]:  Data std^2 recipr. */
    np.ndarray[np.int32_t,ndim=1]   icoor,#/* [nicoor]: Any integer coords */
    np.ndarray[np.int32_t,ndim=1]   idat, #/* [nidat]: Any integer data */
    np.ndarray[np.float32_t,ndim=1] datm, #/* [ndatm]: Model data */
    np.ndarray[np.float32_t,ndim=3] chi2m,#/* [nbeta][nseq][ndc2]: chi^2 terms*/
    # Parameters
    np.ndarray[np.float32_t,ndim=3] pcur,   #/* [nbeta][nseq][nptot] */ 
    np.ndarray[np.float32_t,ndim=2] chi2c,  #/* [nbeta][nseq] */
    np.ndarray[np.int32_t,ndim=2]   flag,   #/* [nbeta][nseq] */
    imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nbeta, nseq):
    
    calc_modchi2( <float*>coor.data,   <float*>dat.data,    <float*>std2r.data,
                  <int*>icoor.data,    <int*>idat.data,     <float*>datm.data,
                  <float*>chi2m.data,  <float*>pcur.data,   <float*>chi2c.data,
                  <int*>flag.data,
                  imodel, ncoor, ndat, nicoor, nidat, ndatm, nptot, nbeta, nseq)




cdef extern from "mcmc.h":
    void calc_model(float *coor, int *idat, float *datm, float *pcur, 
                    int imodel, int ncoor, int ndat, int nidat, 
                    int ndatm, int nptot, int nbeta, int nseq)



cpdef calcmodel(
    # Coordinates, Observables and Model Data, Integer Data 
    np.ndarray[np.float32_t,ndim=1] coor, #/* [ncoor]: Coordinates: UV etc */
    np.ndarray[np.int32_t,ndim=1]   idat, #/* [nidat]: Any integer data */
    np.ndarray[np.float32_t,ndim=1] datm, #/* [ndatm]: Model data */
    # Parameters
    np.ndarray[np.float32_t,ndim=3] pcur,   #/* [nbeta][nseq][nptot] */ 
    imodel, ncoor, ndat, nidat, ndatm, nptot, nbeta, nseq):
    
    calc_model( <float*>coor.data,   <int*>idat.data,     <float*>datm.data,
                <float*>pcur.data,
                imodel, ncoor, ndat, nidat, ndatm, nptot, nbeta, nseq)




# cudaDeviceReset();
cpdef reset_gpu():
    cudaDeviceReset()


