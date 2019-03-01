//
// mcmcjob.cu
//
// The classes CCalcModel and its inheritor CMcmcFit contain the data
// transferred from the PC memory (host) to the CUDA GPU memory (device).
// The instantiation of the latter and calling the method
// CMcmcFit::do_mcmc_on_gpu() runs the MCMC fitting on a CUDA GPU.
// is made over the interface Python-C/C++ mcmc_interf.pyx written in
// Cython and compiled into mcmc_interf.c. 
//
//
// The MCMC algorithm is typically used for fitting parameterized models
// to large arrays of data located at many coordinate points. For example,
// it can be a model of celestial radio object observed with a radio
// interferometer. The data here are the visibilities, and the coordinates are
// u and v, the locations of visibilities in the spatial frequency domain.
// The data should be provided to the algorithm in the dat[ndat] array, and
// the coordinates -- in the coor[ncoor] array. Sometimes the model requires
// integer data, passed in the integer array idat[nidat], and integer
// coordinates (like antenna numbers) are passed in the icoor[nicoor] array.
// To compute the chi^2 the algorithm requires the variance std^2.
// The std2r[ndat] array should contain the reciprocals of the data variances.
//
// -----------------------------------------------------------------------------
//
// The MCMC with replica exchange is implemented as a function mcmc_cuda()
// called by the Cython-compiled interface module mcmc_interf. In turn,
// mcmc_cuda() creates mcmcobj as the class CMcmcFit instance. This object
// transfers the arrays to GPU and calls the class method do_mcmc_on_gpu().
// It, in turn, calls the run_mcmc() CUDA C function determined with the rest
// of CUDA MCMC code in gpu_mcmc.cu. Below is the list of parameters accepted
// by mcmc_cuda(). In the C and C++ codes types of the parameters correspond
// to their Python analogs. 
//
//
//
// Input parameters of the MCMC algorithm
//
// nptot: total number of the parameters to be optimized by MCMC, or
//        the full problem dimensionality. Part of the parameters in the
//        ptotal[nptot] array can be made immutable for a particular MCMC run
//        by setting zeros at the corresponding locations in the descriptor
//        array pdescr[nptot].
//
// nprm: the number of parameters to be optimized by MCMC in the specific run.
//       It is possible to "freeze" values of the part of problem parameters
//       making them immutable constants, and optimize only the rest of them.
//       The frozen (immutable) parameters must be marked by zeros at their
//       positions in the descriptor array pdescr[nptot]. Thus nprm is the
//       number of non-zero elements in pdescr[].
//
// ndat: number of floating point data.
//
// ncoor: number of floating point coordinates.
//
// nidat: number of integer data.
//
// nicoor: number of integer coordinates.
// 
// nseq: number of independent parallel processes of model fitting. In CUDA
//       framework it means the number of thread blocks. In the Nvidia
//       GPU hardware one block of threads is executed on one
//       "Streaming Multiprocessor" (SM). For example, GTX 670 has 7 SMs. 
//
// nbeta: number of "temperatures" in the MCMC algorithm. In CUDA framework
//        it means the number of parallel theads per block. In Nvidia GPU
//        the treads in a SM are executed in units of "warps" of 32 threads
//        each executing the same instruction at a time.
//
// seed: an arbitrary 64-bit unsigned integer (np.uint64) seed for the CUDA
//       random number generator. 
// 
// pdescr[nptot]: array of parameter descriptors with possible values
//                2 - angular parameter (radians), 1 - nonangular parameter.
//                0 - this value excludes the corresponding parameter in
//                    ptotal[nptot] from optimization.
// ptotal[nptot]: array of model parameters. The ptotal[] values at
//                the locations where pdescr[] is nonzero are ignored.
//                The values where pdescr is zero are used in the model.
//                
// pmint[nptot],
// pmaxt[nptot]: minimum and maximum values for the model parameters.
//               The parameters are searched only inside of the
//               nptot-dimensional rectangular parallelepiped determined by
//               pmint[] and pmaxt[]. This parallelepiped determines the
//               "prior".
//
// ivar[nprm]: maps [nprm] optimized parameters on ptotal[nptot].
//             ivar[nprm] has indices of the ptotal[nptot] parameters whose
//             descriptors in pdescr[nptot] are non-zero.
//
// invar[nptot]: maps ptotal[nptot] on the varied parameters [nprm]. If some
//               of the parameters are immutablem invar[] must contain -1 values
//               at the corresponding positions.
//
// beta[nbeta]: "temperature" values in descending order from 1. Recommended are
//              the values between 1 and 0.0001 falling off exponentially.
//
// dat[ndat]: input floating point data (e.g. visibilities, phases etc).
//
// idat[nidat]: input integer data.
//
// coor[ncoor]: input floating point coordinates (e.g. u and v pairs).
//
// icoor[nicoor]: integer coordinates (e.g. antenna numbers)
//              
// std2r[ndat]: the reciprocals of the data variances, 1/std^2.
//
// pcur[nbeta,nseq,nptot]: initial parameter values for all temperatures and
//                         all sequences. Can be zeros or random numbers or
//                         the coordinates of the prior center. During MCMC
//                         run is used for the current parameter values.
//
// tcur[nbeta,nseq]: integer values of the "temperature" indices. On entry it
//                   should contain nseq columns of cardinals from 0 up to
//                   nbeta-1.
//
// n_cnt[nprm,nbeta,nseq]: numbers of Metropolis trials for each mutable
//                         parameter, each temperature, and each sequence.
//                         Sould be initialized to ones.
//
// nadj: number of sets for calculating acceptance rates. Recommended value 100.
//
// npass: number of times the model() function must be called. Sometimes
//        the model computation requires several passes. The data saved in a
//        previous pass are used in the next one. The model() function
//        can determen at which pass it has been called by its parameter ipass
//        taking values from 0 to npass-1.
//
// pstp[nprm,nbeta,nseq]: initial parameter steps for each mutable parameter,
//                        each temperature, and each sequence.
// imodel: an integer parameter passed to the model() function. The usage is
//         arbitrary. For example, it may be a model number to select between
//         several models.
//
// nburn: number of the burn-in iterations. During the burn-in phase the steps
//        for each parameter, each temperature, and each sequence are adjusted,
//        and the transients fall off.
//
// niter: number of optimization operations.
//
// ndatm: must be equal nbeta*nseq*ndat
//
// -----------------------------------------------------------------------------
//
// Workspace arrays of MCMC algorithm
//
// datm[nbeta,nseq,ndat]: computed model data.
//
// chi2m[nbeta,nseq,ndat]: chi^2 terms for each model data element, each
//                         temperature, and each sequence.
//
// rndst[nbeta,nseq,48]: unsigned 8-bit integer (np.uint8) array of states
//                       for the random number generator.
// 
// flag[nbeta,nseq]: 0 means the selected parameter is outside of the prior,
//                   or the parameter did not pass the alpha-test.
//                   1 means the selected parameter is OK.
// chi2c[nbeta,nseq]: old chi^2 for each temperature and each sequence.
//
// ptent[nbeta,nseq]: tentative parameter values for each temperature and each
//                    sequence.
//
// ptentn[nbeta,nseq]: tentative parameter indices for each temperature and each
//                     sequence.
//
// -----------------------------------------------------------------------------
//
// Output parameters of MCMC algorithm
//
// pout[nprm,nbeta,nseq,niter]: mutable parameters found by the MCMC for
//                              all the temperatures, sequences, and iterations.
//                              As a rule, only the coldest temperature is used,
//                              i.e. pout[nprm,0,nseq,niter].
//                              The optimum parameter set is in pout[] at the
//                              location corresponding to the minimum chi^2
//                              in the chi2[nseq,niter] array. Usually these
//                              arrays reshape to 1D form:
//                              po = pout[:,0,:,:].reshape((nprm,nseq*niter))
//                              c2 = chi2.flatten()
//                              and then the best parameter set is found as
//                              po[c2.argmin()]. Also, the histograms of each
//                              parameter can provide important information:
//                              hist(po[0,:], 50, color='b'); grid(1) - for
//                              the 0-th parameter and so on.
//
// chi2[nseq,niter]
//
// n_acpt[nprm,nbeta,nseq]: counting numbers of accepted proposal sets in the
//                          Metropolis-Hastings algorithm.
//
// n_exch[nbeta,nseq]: counting numbers of exchanged adjacent temperature
//                     chanins in the replica exchange algorithm.
//
// n_hist[nadj,nprm,nbeta,nseq]: counting numbers of "accept" of proposal set
//                               in Metropolis-Hastings algorithm
//
//



#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "mcmcjob.cuh"
#include "mcmc.h"
//void run_mcmc(CMcmcFit *mc_h, CMcmcFit *mc_d);


//extern "C" 
void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}



/*=========================================================================*/



extern "C" 
void calc_modchi2(float *coor, float *dat, float *std2r,
	          int *icoor, int *idat, float *datm, float *chi2m,    
	          float *pcur, float *chi2c, int *flag, int imodel,
	          int ncoor, int ndat, int nicoor, int nidat, int ndatm,
		  int nptot, int nbeta, int nseq) {
  
  /*
   * Instantiate the CCalcModel class as modobj object:
   */
  CCalcModel modobj(imodel, ncoor, ndat, nicoor, nidat, ndatm, 
		    nptot, nbeta, nseq,
		    /* Coordinates, Observables, Integer Data  */
		    coor,    /* [ncoor] */
		    dat,     /* [ndat] */
		    std2r,   /* [ndat] */
		    icoor,   /* [nicoor] */
		    idat,    /* [nidat] */
		    datm,    /* [ndatm] */
		    chi2m,   /* [nbeta][nseq][ndat] */
		    /* Parameters */
		    pcur,    /* [nbeta][nseq][nptot] */ 
		    flag,     /* [nbeta][nseq] */ 
		    chi2c);  /* [nbeta][nseq] */ 

  modobj.do_model_chi2_gpu();

} /* extern "C" void calc_modchi2() */


extern "C" 
void calc_model(float *coor, int *idat, float *datm, float *pcur, 
		int imodel, int ncoor, int ndat, int nidat, 
		int ndatm, int nptot, int nbeta, int nseq) {

  /* Dummy arrays, not used in just model calculation */
  float *dat = new float [ndat];
  float *std2r = new float [ndat];
  float *chi2m = new float [nbeta*nseq*ndat];
  float *chi2c = new float [nbeta*nseq];
  int nicoor = 1;
  int *icoor = new int [nicoor];
  int *flag = new int [nbeta*nseq];

  /*
   * Instantiate the CCalcModel class as modobj object:
   */
  CCalcModel modobj(imodel, ncoor, ndat, nicoor, nidat, ndatm, 
		    nptot, nbeta, nseq,
		    /* Coordinates, Observables, Integer Data  */
		    coor,    /* [ncoor] */
		    dat,     /* [ndat] */
		    std2r,   /* [ndat] */
		    icoor,   /* [nicoor] */
		    idat,    /* [nidat] */
		    datm,    /* [ndatm] */
		    chi2m,   /* [nbeta][nseq][ndat] */
		    /* Parameters */
		    pcur,    /* [nbeta][nseq][nptot] */ 
		    flag,     /* [nbeta][nseq] */ 
		    chi2c);  /* [nbeta][nseq] */ 

  modobj.do_calc_model_gpu();

  // delete[] dat; 
  // delete[] std2r; 
  // delete[] chi2m;
  // delete[] chi2c;
  // delete[] icoor;
  // delete[] flag;

} /* extern "C" void calc_model() */


/*
 * Constructors
 */
 
CCalcModel::CCalcModel(int imodel, 
	   int ncoor, 
	   int ndat,
	   int nicoor,
	   int nidat,
	   int ndatm,
	   int nptot,  
	   int nbeta,
	   int nseq,
	   /* Coordinates, Observables and Model Data, Integer Data */
	   float *coor,   /* [ncoor]:             Coordinates: UV etc */
	   float *dat,    /* [ndat]:              Observation data */
	   float *std2r,  /* [ndat]:              Data std^2 recipr. */
	   int   *icoor,  /* [nicoor]:            Any integer data */
	   int   *idat,   /* [nidat]:             Any integer data */
	   float *datm,   /* [ndatm]: Model data */
	   float *chi2m,  /* [nbeta][nseq][ndat]: Terms of chi^2 sum */
	   /* Parameters */
	   float *pcur,  /* [nbeta][nseq][nptot] */
	   int *flag,    /* [nbeta][nseq] */ 
	   float *chi2c) /* [nbeta][nseq] */  {

  s_coor = sizeof(float)*ncoor;
  s_dat = sizeof(float)*ndat;
  s_icoor = sizeof(int)*nicoor;
  s_idat = sizeof(int)*nidat;
  s_datm = sizeof(float)*ndatm;
  s_chi2m = sizeof(float)*nbeta*nseq*ndat;
  s_pcur = sizeof(float)*nbeta*nseq*nptot;
  s_bsint = sizeof(int)*nbeta*nseq;
  s_bsflt = sizeof(float)*nbeta*nseq;

  this->imodel = imodel;
  this->ncoor = ncoor;
  this->ndat = ndat;
  this->nicoor = nicoor;
  this->nidat = nidat;
  this->ndatm = ndatm;
  this->nptot = nptot;
  this->nbeta = nbeta;
  this->nseq = nseq;
  /* Coordinates, Observables and Model Data, Integer Data */
  this->coor_h = coor;       /* [ncoor] */
  this->dat_h = dat;         /* [ndat] */
  this->std2r_h = std2r;
  this->icoor_h = icoor;
  this->idat_h = idat;
  this->datm_h = datm;
  this->chi2m_h = chi2m;
  /* Parameters */
  this->pcur_h = pcur;       /* [nbeta][nseq][nptot] */
  this->flag_h = flag;       /* [nbeta][nseq] */
  this->chi2c_h = chi2c;     /* [nbeta][nseq] */

  this->nchi2m = nbeta*nseq*ndat; /* Threads in model data calculation */


  /*
   * Allocate the arrays on the GPU
   */
  CUDA_CALL(cudaMalloc((void **) &this->coor,  s_coor));
  CUDA_CALL(cudaMalloc((void **) &this->dat,   s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->std2r, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->icoor, s_icoor));
  CUDA_CALL(cudaMalloc((void **) &this->idat,  s_idat));
  CUDA_CALL(cudaMalloc((void **) &this->datm,  s_datm));
  CUDA_CALL(cudaMalloc((void **) &this->chi2m, s_chi2m));
  CUDA_CALL(cudaMalloc((void **) &this->pcur,  s_pcur));
  CUDA_CALL(cudaMalloc((void **) &this->flag,  s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->chi2c, s_bsflt));

  /*
   * Copy the host arrays to the GPU
   */
  CUDA_CALL(cudaMemcpy(this->coor,  this->coor_h,  s_coor,  HtoD));
  CUDA_CALL(cudaMemcpy(this->dat,   this->dat_h,   s_dat,   HtoD));
  CUDA_CALL(cudaMemcpy(this->std2r, this->std2r_h, s_dat,   HtoD));
  CUDA_CALL(cudaMemcpy(this->icoor, this->icoor_h, s_icoor, HtoD));
  CUDA_CALL(cudaMemcpy(this->idat,  this->idat_h,  s_idat,  HtoD));
  CUDA_CALL(cudaMemcpy(this->datm,  this->datm_h,  s_datm,  HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2m, this->chi2m_h, s_chi2m, HtoD));
  CUDA_CALL(cudaMemcpy(this->pcur,  this->pcur_h,  s_pcur,  HtoD));
  CUDA_CALL(cudaMemcpy(this->flag,  this->flag_h,  s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2c, this->chi2c_h, s_bsflt, HtoD));

  printf("CONSTRUCTOR of CCalcModel: exit.\n");
}


/*
 * Destructor
 */

CCalcModel::~CCalcModel() {

  CUDA_CALL(cudaMemcpy(this->coor_h,  this->coor,  s_coor,  DtoH));
  CUDA_CALL(cudaMemcpy(this->dat_h,   this->dat,   s_dat,   DtoH));
  CUDA_CALL(cudaMemcpy(this->std2r_h, this->std2r, s_dat,   DtoH));
  CUDA_CALL(cudaMemcpy(this->icoor_h, this->icoor, s_icoor, DtoH));
  CUDA_CALL(cudaMemcpy(this->idat_h,  this->idat,  s_idat,  DtoH));
  CUDA_CALL(cudaMemcpy(this->datm_h,  this->datm,  s_datm,  DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2m_h, this->chi2m, s_chi2m, DtoH));
  CUDA_CALL(cudaMemcpy(this->pcur_h,  this->pcur,  s_pcur,  DtoH));
  CUDA_CALL(cudaMemcpy(this->flag_h,  this->flag,  s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2c_h, this->chi2c, s_bsflt, DtoH));

  CUDA_CALL(cudaFree(this->coor));
  CUDA_CALL(cudaFree(this->dat));
  CUDA_CALL(cudaFree(this->std2r));
  CUDA_CALL(cudaFree(this->icoor));
  CUDA_CALL(cudaFree(this->idat));
  CUDA_CALL(cudaFree(this->datm));
  CUDA_CALL(cudaFree(this->chi2m));
  CUDA_CALL(cudaFree(this->pcur));
  CUDA_CALL(cudaFree(this->flag));

}



void CCalcModel::do_model_chi2_gpu() {
  CCalcModel *dev_mod;

  CUDA_CALL(cudaMalloc((void **) &dev_mod, sizeof(CCalcModel)));
  CUDA_CALL(cudaMemcpy(dev_mod, this, sizeof(CCalcModel), HtoD));

  run_calcmodchi2(this, dev_mod);

  CUDA_CALL(cudaMemcpy(this, dev_mod, sizeof(CCalcModel), DtoH));
  CUDA_CALL(cudaFree(dev_mod));

}



void CCalcModel::do_calc_model_gpu() {
  CCalcModel *dev_mod;

  CUDA_CALL(cudaMalloc((void **) &dev_mod, sizeof(CCalcModel)));
  CUDA_CALL(cudaMemcpy(dev_mod, this, sizeof(CCalcModel), HtoD));

  run_calcmodel(this, dev_mod);

  CUDA_CALL(cudaMemcpy(this, dev_mod, sizeof(CCalcModel), DtoH));
  CUDA_CALL(cudaFree(dev_mod));

}




// void CCalcModel::do_calc_model_gpu() {
//   CCalcModel *dev_mod;

//   CUDA_CALL(cudaMalloc((void **) &dev_mod, sizeof(CCalcModel)));
//   CUDA_CALL(cudaMemcpy(dev_mod, this, sizeof(CCalcModel), HtoD));

//   run_calcmodel(this, dev_mod);

//   CUDA_CALL(cudaMemcpy(this, dev_mod, sizeof(CCalcModel), DtoH));
//   CUDA_CALL(cudaFree(dev_mod));

// }



/*=========================================================================*/



extern "C" 
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
	       int nbeta, int nseq, int nburn, int niter) {

  /*
   * Instantiate the CMcmcFit class as mcmcobj object:
   */
  CMcmcFit mcmcobj(imodel, 
		   ncoor,
		   ndat,
		   nicoor,
		   nidat,
		   ndatm,
		   nptot, 
		   nprm,
		   nadj,
		   npass,
		   nbeta,
		   nseq,
		   nburn,
		   niter,
		   /* Random Number Generators */
		   seed,
		   rndst,   /* [nbeta][nseq][48] */
		   /* Coordinates, Observables, Integer Data  */
		   coor,    /* [ncoor] */
		   dat,     /* [ndat] */
		   std2r,   /* [ndat] */
		   icoor,   /* [nicoor] */
		   idat,    /* [nidat] */
		   datm,    /* [ndatm] */
		   chi2m,   /* [nbeta][nseq][ndat] */
		   /* Parameters */
		   pdescr,  /* [nptot] */
		   ivar,    /* [nprm] */
		   invar,   /* [nptot] */
		   ptotal,  /* [nptot] */
		   pmint,   /* [nptot] */ 
		   pmaxt,   /* [nptot] */
		   /* MCMC Algorithm */
		   pout,    /* [nprm][nbeta][nseq][niter] */
		   tout,    /* [nbeta][nseq][niter] */
		   chi2,    /* [nseq][niter] */
		   chi2c,   /* [nbeta][nseq] */
		   tcur,    /* [nbeta][nseq] */
		   flag,    /* [nbeta][nseq] */
		   n_acpt,  /* [nprm][nbeta][nseq] */
		   n_exch,  /* [nbeta][nseq] */
		   n_cnt,   /* [nprm][nbeta][nseq] */
		   n_hist,  /* [nadj][nprm][nbeta][nseq] */
		   beta,    /* [nbeta] */
		   pstp,    /* [nprm][nbeta][nseq] */
		   ptent,   /* [nbeta][nseq] */
		   ptentn,  /* [nbeta][nseq] */
		   pcur);   /* [nbeta][nseq][nptot] */

  mcmcobj.do_mcmc_on_gpu();

}  /* extern "C" void mcmc_cuda() */


/*
 * Constructor of CMcmcFit
 */
 
CMcmcFit::CMcmcFit(int imodel, 
		   int ncoor,
		   int ndat,
		   int nicoor,
		   int nidat,
		   int ndatm,
		   int nptot,  
		   int nprm, 
		   int nadj,
		   int npass,
		   int nbeta,
		   int nseq,
		   int nburn,
		   int niter,
		   /* Random Number Generators */
		   unsigned long long seed,
		   unsigned char *rndst,
		   /* Coordinates, Observables and Model Data, Integer Data */
		   float *coor,   /* [ncoor]:             Coordinates: UV etc */
		   float *dat,    /* [ndat]:              Observation data */
		   float *std2r,  /* [ndat]:              Data std^2 recipr. */
		   int   *icoor,  /* [nicoor]:            Any integer data */
		   int   *idat,   /* [nidat]:             Any integer data */
		   float *datm,   /* [ndatm]: Model data */
		   float *chi2m,  /* [nbeta][nseq][ndat]: Terms of chi^2 sum */
		   /* Parameters */
		   int   *pdescr, /* [nptot] */
		   int   *ivar,   /* nprm] */
		   int   *invar,  /* [nptot] */
		   float *ptotal, /* [nptot] */
		   float *pmint,  /* [nptot] */ 
		   float *pmaxt,  /* [nptot] */
		   /* MCMC Algorithm */
		   float *pout,   /* [nprm][nbeta][nseq][niter] */
		   int *tout,     /* [nbeta][nseq][niter] */
		   float *chi2,   /* [nseq][niter] */
		   float *chi2c,  /* [nbeta][nseq] */
		   int *tcur,     /* [nbeta][nseq] */
		   int *flag,     /* [nbeta][nseq] */
		   int *n_acpt,   /* [nprm][nbeta][nseq] */
		   int *n_exch,   /* [nbeta][nseq] */
		   int *n_cnt,    /* [nprm][nbeta][nseq] */
		   int *n_hist,   /* [nadj][nprm][nbeta][nseq] */
		   float *beta,   /* [nbeta] */
		   float *pstp,   /* [nprm][nbeta][nseq] */
		   float *ptent,  /* [nbeta][nseq] */
		   int   *ptentn, /* [nbeta][nseq] */
           float *pcur)   /* [nbeta][nseq][nptot] */ 
{

  ntitr = nburn + niter;
  
  s_coor = sizeof(float)*ncoor;
  s_dat = sizeof(float)*ndat;
  s_datm = sizeof(float)*ndatm;
  s_icoor = sizeof(int)*nicoor;
  s_idat = sizeof(int)*nidat;
  s_rndst = sizeof(curandState)*nbeta*nseq;
  s_ptot = sizeof(float)*nptot;
  s_prm = sizeof(float)*nprm;
  s_pout = sizeof(float)*nprm*nbeta*nseq*niter;
  s_tout = sizeof(int)*nbeta*nseq*niter;
  s_chi2 = sizeof(float)*nseq*niter;
  s_beta = sizeof(float)*nbeta;
  s_acpt = sizeof(int)*nprm*nbeta*nseq;
  s_hist = sizeof(float)*nadj*nprm*nbeta*nseq;
  s_bsflt = sizeof(float)*nbeta*nseq;
  s_bsint = sizeof(int)*nbeta*nseq;
  s_chi2m = sizeof(float)*nbeta*nseq*ndat;
  s_pcur = sizeof(float)*nbeta*nseq*nptot;
  s_bsptot = sizeof(float)*nbeta*nseq*nptot;
  s_4bstf = sizeof(float)*nbeta*nseq*4*ntitr;
  s_bstf = sizeof(float)*nbeta*nseq*ntitr;
  s_bsti = sizeof(int)*nbeta*nseq*ntitr;
  s_burn = sizeof(float)*nprm*nbeta*nseq*nburn;

  this->imodel = imodel;
  this->ncoor = ncoor;
  this->ndat = ndat;
  this->nicoor = nicoor;
  this->nidat = nidat;
  this->ndatm = ndatm;
  this->nptot = nptot;
  this->nprm = nprm;
  this->nadj = nadj;
  this->npass = npass;
  this->nbeta = nbeta;
  this->nseq = nseq;
  this->nburn = nburn;
  this->niter = niter;
  /* Random Number Generators */
  this->seed = seed;
  this->rndst_h = (curandState *) rndst;
  //  this->rndst_h = (curandStateMtgp32 *) rndst;
  //  printf("sizeof(curandStateMtgp32) = %ld\n", sizeof(curandStateMtgp32));
  //  printf("sizeof(curandState) = %ld\n", sizeof(curandState));
  /* Coordinates, Observables and Model Data, Integer Data */
  this->coor_h = coor;       /* [ncoor] */
  this->dat_h = dat;         /* [ndat] */
  this->std2r_h = std2r;
  this->icoor_h = icoor;
  this->idat_h = idat;
  this->datm_h = datm;
  this->chi2m_h = chi2m;
  /* Parameters */
  this->pdescr_h = pdescr;
  this->ivar_h = ivar;
  this->invar_h = invar;
  this->ptotal_h = ptotal;
  this->pmint_h = pmint;
  this->pmaxt_h = pmaxt;
  /* MCMC Algorithm */
  this->pout_h = pout;
  this->tout_h = tout;
  this->chi2_h = chi2;       /* [nseq][niter] */
  this->chi2c_h = chi2c;     /* [nbeta][nseq] */
  this->tcur_h = tcur;       /* [nbeta][nseq] */
  this->flag_h = flag;       /* [nbeta][nseq] */
  this->n_acpt_h = n_acpt;   /* [nprm][nbeta][nseq] */
  this->n_exch_h = n_exch;   /* [nbeta][nseq] */
  this->n_cnt_h = n_cnt;     /* [nprm][nbeta][nseq] */
  this->n_hist_h = n_hist;   /* [nadj][nprm][nbeta][nseq] */
  this->beta_h = beta;       /* [nbeta] */
  this->pstp_h = pstp;       /* [nprm][nbeta][nseq] */
  this->ptent_h = ptent;     /* [nbeta][nseq] */
  this->ptentn_h = ptentn;   /* [nbeta][nseq] */  
  this->pcur_h = pcur;       /* [nbeta][nseq][nptot] */

  this->ipm = 0; /* Loop parameter: current model parameter */
  this->itr = 0; /* Loop parameter: current iteration */
  this->nrndst = nbeta*nseq;
  this->nchi2m = nbeta*nseq*ndat; /* Threads in model data calculation */


  /*
   * Allocate the arrays on the GPU
   */
  CUDA_CALL(cudaMalloc((void **) &this->rndst, s_rndst));

  CUDA_CALL(cudaMalloc((void **) &this->coor, s_coor));
  CUDA_CALL(cudaMalloc((void **) &this->dat, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->std2r, s_dat));
  CUDA_CALL(cudaMalloc((void **) &this->icoor, s_icoor));
  CUDA_CALL(cudaMalloc((void **) &this->idat, s_idat));
  CUDA_CALL(cudaMalloc((void **) &this->datm, s_datm));
  CUDA_CALL(cudaMalloc((void **) &this->chi2m, s_chi2m));
  CUDA_CALL(cudaMalloc((void **) &this->pdescr, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->ivar, s_prm));
  CUDA_CALL(cudaMalloc((void **) &this->invar, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->ptotal, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pmint, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pmaxt, s_ptot));
  CUDA_CALL(cudaMalloc((void **) &this->pout, s_pout));
  CUDA_CALL(cudaMalloc((void **) &this->tout, s_tout));
  CUDA_CALL(cudaMalloc((void **) &this->chi2, s_chi2));
  CUDA_CALL(cudaMalloc((void **) &this->chi2c, s_bsflt));
  CUDA_CALL(cudaMalloc((void **) &this->tcur, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->flag, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->n_acpt, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->n_exch, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->n_cnt, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->n_hist, s_hist));
  CUDA_CALL(cudaMalloc((void **) &this->beta, s_beta));
  CUDA_CALL(cudaMalloc((void **) &this->pstp, s_acpt));
  CUDA_CALL(cudaMalloc((void **) &this->ptent, s_bsflt));
  CUDA_CALL(cudaMalloc((void **) &this->ptentn, s_bsint));
  CUDA_CALL(cudaMalloc((void **) &this->pcur, s_bsptot));

  /*
   * Copy the host arrays to the GPU
   */

  CUDA_CALL(cudaMemcpy(this->rndst, this->rndst_h, s_rndst, HtoD));

  CUDA_CALL(cudaMemcpy(this->coor, this->coor_h, s_coor, HtoD));
  CUDA_CALL(cudaMemcpy(this->dat, this->dat_h, s_dat, HtoD));
  CUDA_CALL(cudaMemcpy(this->std2r, this->std2r_h, s_dat, HtoD));
  CUDA_CALL(cudaMemcpy(this->icoor, this->icoor_h, s_icoor, HtoD));
  CUDA_CALL(cudaMemcpy(this->idat, this->idat_h, s_idat, HtoD));
  CUDA_CALL(cudaMemcpy(this->datm, this->datm_h, s_datm, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2m, this->chi2m_h, s_chi2m, HtoD));
  CUDA_CALL(cudaMemcpy(this->pdescr, this->pdescr_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->ivar, this->ivar_h, s_prm, HtoD));
  CUDA_CALL(cudaMemcpy(this->invar, this->invar_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptotal, this->ptotal_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->pmint, this->pmint_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->pmaxt, this->pmaxt_h, s_ptot, HtoD));
  CUDA_CALL(cudaMemcpy(this->tout, this->tout_h, s_tout, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2, this->chi2_h, s_chi2, HtoD));
  CUDA_CALL(cudaMemcpy(this->chi2c, this->chi2c_h, s_bsflt, HtoD));
  CUDA_CALL(cudaMemcpy(this->pout, this->pout_h, s_pout, HtoD));
  CUDA_CALL(cudaMemcpy(this->tcur, this->tcur_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->flag, this->flag_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_acpt, this->n_acpt_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_cnt, this->n_cnt_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_exch, this->n_exch_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->n_hist, this->n_hist_h, s_hist, HtoD));
  CUDA_CALL(cudaMemcpy(this->beta, this->beta_h, s_beta, HtoD));
  CUDA_CALL(cudaMemcpy(this->pstp, this->pstp_h, s_acpt, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptent, this->ptent_h, s_bsflt, HtoD));
  CUDA_CALL(cudaMemcpy(this->ptentn, this->ptentn_h, s_bsint, HtoD));
  CUDA_CALL(cudaMemcpy(this->pcur, this->pcur_h, s_bsptot, HtoD));

}


/*
 * Destructor
 */


CMcmcFit::~CMcmcFit() {

  CUDA_CALL(cudaMemcpy(this->rndst_h, this->rndst, s_rndst, DtoH));

  // CUDA_CALL(cudaMemcpy(this->coor_h, this->coor, s_coor, DtoH));
  // CUDA_CALL(cudaMemcpy(this->dat_h, this->dat, s_dat, DtoH));
  // CUDA_CALL(cudaMemcpy(this->std2r_h, this->std2r, s_dat, DtoH));
  // CUDA_CALL(cudaMemcpy(this->icoor_h, this->icoor, s_icoor, DtoH));
  // CUDA_CALL(cudaMemcpy(this->idat_h, this->idat, s_idat, DtoH));
  // CUDA_CALL(cudaMemcpy(this->datm_h, this->datm, s_datm, DtoH));
  // CUDA_CALL(cudaMemcpy(this->chi2m_h, this->chi2m, s_chi2m, DtoH));

  CUDA_CALL(cudaMemcpy(this->pdescr_h, this->pdescr, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->ivar_h, this->ivar, s_prm, DtoH));
  CUDA_CALL(cudaMemcpy(this->invar_h, this->invar, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptotal_h, this->ptotal, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->pmint_h, this->pmint, s_ptot, DtoH));
  CUDA_CALL(cudaMemcpy(this->pmaxt_h, this->pmaxt, s_ptot, DtoH));

  CUDA_CALL(cudaMemcpy(this->pout_h, this->pout, s_pout, DtoH));
  CUDA_CALL(cudaMemcpy(this->tout_h, this->tout, s_tout, DtoH));
  CUDA_CALL(cudaMemcpy(this->chi2_h, this->chi2, s_chi2, DtoH));
  //CUDA_CALL(cudaMemcpy(this->chi2c_h, this->chi2c, s_bsflt, DtoH));
  CUDA_CALL(cudaMemcpy(this->tcur_h, this->tcur, s_bsint, DtoH));
  // CUDA_CALL(cudaMemcpy(this->flag_h, this->flag, s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_acpt_h, this->n_acpt, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_cnt_h, this->n_cnt, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_exch_h, this->n_exch, s_bsint, DtoH));
  CUDA_CALL(cudaMemcpy(this->n_hist_h, this->n_hist, s_hist, DtoH));
  CUDA_CALL(cudaMemcpy(this->beta_h, this->beta, s_beta, DtoH));
  CUDA_CALL(cudaMemcpy(this->pstp_h, this->pstp, s_acpt, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptent_h, this->ptent, s_bsflt, DtoH));
  CUDA_CALL(cudaMemcpy(this->ptentn_h, this->ptentn, s_bsint, DtoH));
  // CUDA_CALL(cudaMemcpy(this->pcur_h, this->pcur, s_bsptot, DtoH));

  CUDA_CALL(cudaFree(this->rndst));

  // CUDA_CALL(cudaFree(this->coor));
  // CUDA_CALL(cudaFree(this->dat));
  // CUDA_CALL(cudaFree(this->std2r));
  // CUDA_CALL(cudaFree(this->icoor));
  // CUDA_CALL(cudaFree(this->idat));
  // CUDA_CALL(cudaFree(this->datm));
  // CUDA_CALL(cudaFree(this->chi2m));

  CUDA_CALL(cudaFree(this->pdescr));
  CUDA_CALL(cudaFree(this->ivar));
  CUDA_CALL(cudaFree(this->invar));
  CUDA_CALL(cudaFree(this->ptotal));
  CUDA_CALL(cudaFree(this->pmint));
  CUDA_CALL(cudaFree(this->pmaxt));

  CUDA_CALL(cudaFree(this->pout));
  CUDA_CALL(cudaFree(this->tout));
  CUDA_CALL(cudaFree(this->chi2));
  // CUDA_CALL(cudaFree(this->chi2c));

  CUDA_CALL(cudaFree(this->tcur));
  //  CUDA_CALL(cudaFree(this->flag));
  CUDA_CALL(cudaFree(this->n_acpt));
  CUDA_CALL(cudaFree(this->n_cnt));
  CUDA_CALL(cudaFree(this->n_exch));
  CUDA_CALL(cudaFree(this->n_hist));
  CUDA_CALL(cudaFree(this->beta));
  CUDA_CALL(cudaFree(this->pstp));
  CUDA_CALL(cudaFree(this->ptent));
  CUDA_CALL(cudaFree(this->ptentn));
  //  CUDA_CALL(cudaFree(this->pcur));

}


void CMcmcFit::do_mcmc_on_gpu() {
  CMcmcFit *dev_mcmc;

  CUDA_CALL(cudaMalloc((void **) &dev_mcmc, sizeof(CMcmcFit)));
  CUDA_CALL(cudaMemcpy(dev_mcmc, this, sizeof(CMcmcFit), HtoD));

  run_mcmc(this, dev_mcmc);

  printf("sizeof(CMcmcFit) = %d B\n", sizeof(CMcmcFit));

  CUDA_CALL(cudaMemcpy(this, dev_mcmc, sizeof(CMcmcFit), DtoH));
  CUDA_CALL(cudaFree(dev_mcmc));

}

