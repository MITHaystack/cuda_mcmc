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
#include <curand_kernel.h>
#include <stdio.h>
#include "mcmcjob.cuh"
#include "model.cuh"

#define Nbeta (blockDim.x)
#define Nseq (gridDim.x) 
#define Ibeta (threadIdx.x) 
#define Iseq (blockIdx.x) 

//   /* 
//    * The problem-specific function model() should be provided by the user. 
//    *
//    *Arguments:
//    *   id: index into mc->dat. 
//    *   ipt: index of parameters' set: pcur[ipt] ~ pcur[ibeta,iseq,:].
//    *        pcur[ipt] is the first model parameter, and 
//    *        pcur[ipt+mc->nptot-1] is the last parameter.
//    *        is
//    *   imd: "through index" into datm[ibeta,isec,id]. Only used 
//    *        to save the result in mc->datm[imd] before return.
//    *   ipass: pass number, starting from 0. The model() function can be
//    *          called multiple times, or in many passes, so ipass is to know
//    *          which time model() is called now.  
//    *
//    * Result must be saved in
//    *   mc->datm[imd]
//    */

// __device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass);



/*
 * Initialize states one for all the threads
 */
__global__ void init_rng(CMcmcFit *mc) {

  //    int ist = Ibeta*Nseq + Iseq;    /* [ibeta,iseq] */
                
        int ist = threadIdx.x*gridDim.x + blockIdx.x;

        if (ist >= mc->nrndst) return;

	unsigned long long offset = 0;

        curand_init(mc->seed, ist, offset, &mc->rndst[ist]);
}


//==========================================================
/* init_pcur<<<nseq,nbeta>>>(mc_d); */

__global__ void init_pcur(CMcmcFit *mc) {

  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */
  int k, i;
  float p, randu;

  if (ist >= mc->nrndst) return;

  mc->flag[ist] = 1; /* All parameters initially are accepted */
  mc->itr = 0;

  for (i = 0; i < mc->nprm; i++) {
    k = mc->ivar[i];  /* turn i in [0..nprm-1] into k in [0..nptot-1] */

    /* Set p to a urandom value in [pmint[k]..pmaxt[k]] */
    randu = curand_uniform(&mc->rndst[ist]);
    p  = mc->pmint[k] + (mc->pmaxt[k] - mc->pmint[k])*randu;
                                    
    /* If p is angle, correct angle into +-pi */
    if (mc->pdescr[k] == 2) p = atan2(sin(p),cos(p)); 
    mc->pcur[ist*mc->nptot+k] = p; /* pcur[ibeta,iseq,k] = p */
  }
}

//==========================================================

/*     calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);  */

__global__ void calc_chi2_terms(CCalcModel *mc, int ipass) {

  int idatm = blockDim.x*blockIdx.x + threadIdx.x; /* through index */
    
  if (idatm >= mc->nchi2m) return; /* The last block can have extra threads */

  int ist, ipt, idat, iret;
  float dif;  //, dif1;

  /*
   * ndat = nvis + ncph
   *
   * Treat the through index idatm as idatm ~ [ibeta,iseq,idat], or
   * idatm = (ibeta*nseq + iseq)*ndat + idat: 
   */
  idat = idatm%mc->ndat;
  ist =  idatm/mc->ndat;      /* ptent[ist] ~ ptent[ibeta,iseq] */
 
  if (mc->flag[ist] == 0) return; // ==== no calcs for rejected parameters ===>>

  ipt = ist*mc->nptot;     /* pcur[ipt] ~ pcur[ibeta,iseq,:] */

  /*
   * Call model
   */

  iret = model(mc, idat, ipt, idatm, ipass); /* Compute model at idatm */

  if (iret == 0) return; /* No results */
  /*
   * Calculate one term of the chi^2 sum
   */
  dif = mc->dat[idat] - mc->datm[idatm];


  //if (ist == 0) {
  // printf("DEVICE idat=%d, idatm=%d, dat[idat]=%f, datm[idatm]=%f, dif=%f\n",
  //        idat, idatm, mc->dat[idat], mc->datm[idatm], dif);
  //}


  
  if (iret == 2) { /* I.e. dif is closure phase difference */
    dif = atan2(sin(dif), cos(dif));
    // /* In case they are close to +-pi: */
    // dif1 = fabs(mc->dat[idat] + mc->datm[idatm]);
    // dif = fabs(dif);
    // dif = dif < dif1 ? dif : dif1;        /* dif is whichever is smaller */
  }

  mc->chi2m[idatm] = (dif*dif)*mc->std2r[idat];   /* (xi-Xi)^2*(1/std^2) */

}

//========================================================

// calc_chi2<<<nseq,nbeta>>>(mc_d);

__global__ void calc_chi2(CCalcModel *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;     /* [ibeta,iseq] */

  //if (ist >= mc->nrndst) return;

  int ier0 = ist*mc->ndat;         /* [ibeta,iseq,0] */
  int ier1 = ier0 + mc->ndat;      /* [ibeta,iseq,ndat-1] */
  //int ier1 = ier0 + mc->idat[0];      /* [ibeta,iseq,nvis-1] */
  int i;
  float sum = 0.0f;

  for (i = ier0; i < ier1; i++) {
    sum += mc->chi2m[i];
  }
  mc->chi2c[ist] = sum;

  /* Now chi2c[nbeta,nseq] contains the new chi^2 for all temperatures and 
   * all sequences. */
}

//==========================================================

// #define Nbeta (blockDim.x)
// #define Nseq (gridDim.x) 
// #define Ibeta (threadIdx.x) 
// #define Iseq (blockIdx.x) 

/*   gen_proposal<<<nseq,nbeta>>>(mc_d); */

__global__ void gen_proposal(CMcmcFit *mc) {

  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */
  int k, t, pt;
  float r, p, std, randn;

  if (ist >= mc->nrndst) return;

  do { /* Randomly select parameter number k to make step */
    r = curand_uniform(&mc->rndst[ist]);
    k = (int) (mc->nprm*r); /* Random parameter index 0..nprm-1 into ivar[] */
  } while (k >= mc->nprm);
  t = mc->ivar[k]; /* Random parameter index into pcur[ibeta,iseq,:]  */
  mc->ptentn[ist] = t; /* The param # in pcur chosen for variation */
                                    
  /* Select a t'th parameter (t is random!) from pcur[ibeta,iseq,t] */
  pt = ist*mc->nptot + t; /* Index into pcur[ibeta,iseq,t] */
  p = mc->pcur[pt]; /* p = pcur[ibeta,iseq,t] */


  /* Save the current parameter value in ptent[ibeta,iseq] */
  mc->ptent[ist] = p;               /* ptent[ibeta,iseq] = p; */

  /* Add to it a "Gaussian step" - noise with stdev = pstp[k,ibeta,iseq] */
  std = mc->pstp[(Nbeta*k + Ibeta)*Nseq + Iseq]; /* pstp[k,ibeta,iseq] */
  randn = curand_normal(&mc->rndst[ist]);
  p = p + std*randn;

  if (mc->pdescr[t] == 2) 
    p = atan2(sin(p),cos(p)); /* Correct angle into +-pi */

  /* check_p(); */
  if ((p < mc->pmint[t]) || (p > mc->pmaxt[t])) {/* If outside of the prior: */
    mc->flag[ist] = 0;  /* Flag the parameter as 'outside of the prior' */
                        /* and leave pcur[ibeta,iseq,:] intact */
  }
  else {
    mc->flag[ist] = 1;  /* Flag the parameter as 'inside of the prior' */
    /* Save the new parameter in pcur[ibeta,iseq,t]*/
    mc->pcur[pt] = p;  /* pcur[ibeta,iseq,t] = p; - put new parameter value */

  }
  /* Now: 
   * ptentn[nbeta,nseq] has the numbers (0:nptot-1) of the moved parameters.
   * If a parameter is out of the prior, flag[ibeta,iseq] is set to 0,
   * and pcur[ibeta,iseq,t] is left with the old parameter value.
   * Otherwise flag[ibeta,iseq] is set to 1, and
   * pcur[ibeta,iseq,t] has the variated parameter.
   */

}  /* __global__ void gen_proposal(CMcmcFit *mc)  */




//==========================================================
// #define Nbeta (blockDim.x)
// #define Nseq  (gridDim.x) 
// #define Ibeta (threadIdx.x) 
// #define Iseq  (blockIdx.x) 
//
// Single Parameter Metropolis MCMC
//
// spmp_mcmc<<<nseq,nbeta>>>(mc_d);

__global__ void spmp_mcmc(CMcmcFit *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;      /* [ibeta,iseq] */

  if (ist >= mc->nrndst) return;

  if (mc->flag[ist] == 0) return;   /* 0: param was outside of prior */

  int itrm0 = ist*mc->ndat;         /* chi2m[ibeta,iseq,0] */
  int itrm1 = itrm0 + mc->ndat;     /* chi2m[ibeta,iseq,ndat-1] */
  int i, ip;
  float u, a, alpha, chi2 = 0.0f;  //, c2, pp;
  /* Here k is the parameter number randomly chosen in gen_proposal() */
  int k = mc->invar[mc->ptentn[ist]]; /* [0:nprm-1] from [0:nptot-1] */
  int iacpt;   /* iacpt ~ [k,ibeta,iseq] */

  //int ititr = ist*mc->ntitr + mc->itr;  
  //int iti4 = (ititr << 2) + 1; /* 4*ititr + 1 */

  /* Here ptent[ibeta,iseq] parameter is within the prior */
  /* Find chi^2 for model data at (ibeta,iseq) */
  for (i = itrm0; i < itrm1; i++)
    chi2 += mc->chi2m[i];  

  a = exp(-0.5f*mc->beta[Ibeta]*(chi2 - mc->chi2c[ist]));
  alpha = min(a, 1);
  u = curand_uniform(&mc->rndst[ist]);
  // a = -0.5f*mc->beta[Ibeta]*(chi2 - mc->chi2c[ist]);
  // alpha = min(a, 0);
  // u = log(curand_uniform(&mc->rndst[ist]));

  if (u <= alpha) { /* Test OK? */
    mc->chi2c[ist] = chi2; /* chi2c[ibeta,iseq] = chi2; new prm is in pcur */

    /* n_acpt[nprm,nbeta,nseq]:  n_acpt[k,ibeta,inseq]++; */
    iacpt = (Nbeta*k + Ibeta)*Nseq + Iseq; /* iacpt ~ [k,ibeta,iseq] */
    mc->n_acpt[iacpt]++;  /* Accepted! n_acpt[k,ibeta,iseq]++; */
  }
  else {/* Test failed? */
    /* iptot = mc->ivar[ipm]; */
    /* pcur[ibeta,iseq,ptentn[ibeta,iseq]] = ptent[ibeta,iseq] */
    ip = ist*mc->nptot + mc->ptentn[ist];

    //pp = mc->pcur[ip];

    mc->pcur[ip] = mc->ptent[ist]; /* Revert pcur */
    mc->flag[ist] = 0; /* Flag the parameter as 'did not pass chi2 test */
  }
}

//==========================================================

// adjust_steps<<<nseq,nbeta>>>(mc_d);

__global__ void adjust_steps(CMcmcFit *mc) {
  /*  */
  int ist = Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */

  if (ist >= mc->nrndst) return;

  int modn, ihist, sumh;
  int iadj;
  float rate;
  /* Here k is the parameter number randomly chosen in gen_proposal() */
  int k = mc->invar[mc->ptentn[ist]]; /* [0:nprm-1] from [0:nptot-1] */
  int ip = (Nbeta*k + Ibeta)*Nseq + Iseq; /* [k,ibeta,iseq] */

  /* modn - index into n_hist[modn,k,ibeta,iseq] */
  modn = mc->n_cnt[ip] % mc->nadj;  /* modn = n_cnt[k,ibeta,iseq] % nadj; */

  /* n_hist[nadj,nprm,nbeta,nseq]; ihist ~ [modn,k,ibeta,iseq]. */
  ihist = ((mc->nprm*modn + k)*Nbeta + Ibeta)*Nseq + Iseq;


  /*  Memorize the history of accept/reject */
  if (mc->flag[ist] != 0)
    mc->n_hist[ihist] = 1; /* Accepted: n_hist[modn,k,ibeta,iseq] = 1 */
  else 
    mc->n_hist[ihist] = 0; /* Rejected: n_hist[modn,k,ibeta,iseq] = 0 */



  /* Calculate accept rate of the latest nadj trials */
  sumh = 0;
  for (iadj = 0; iadj < mc->nadj; iadj++) {
    /* ihist ~ [iadj,k,ibeta,iseq]. */
    ihist = ((mc->nprm*iadj + k)*Nbeta + Ibeta)*Nseq + Iseq;
    sumh += mc->n_hist[ihist];
  }

  rate = ((float) sumh) / ((float) min(mc->nadj,mc->n_cnt[ip]));
  
  /* Adjust the step size  */
  if (rate < 0.30f) 
    mc->pstp[ip] /= 1.01;
  else
    mc->pstp[ip] *= 1.01;

  mc->n_cnt[ip] += 1;

}


//==========================================================

/* replica_exchange<<<nseq,1>>>(mc_d, burn, itr, st); */

__global__ void replica_exchange(CMcmcFit *mc, int burn, int itr) {
  /*  */
  int ist = Iseq; // Nseq*Ibeta + Iseq;    /* [ibeta,iseq] */
  //int ip = (Nbeta*ipm + Ibeta)*Nseq + Iseq; /* [ipm,ibeta,iseq]
  int nbeta_minus_1, idum, k0, k1, i0, i1;
  float u, alpha;
  float dum, r;
  int ipo, ipc, ico, k, jprm, ibeta, ipm, ito, itc;


  /* Do replica-exchange - can be done only in one thread in each block*/
  if (Ibeta != 0) return;

  nbeta_minus_1 = mc->nbeta - 1;

  for (ibeta = 0; ibeta < mc->nbeta; ibeta++) {  
    do { /* Randomly select adjacent chains */
      r = curand_uniform(&mc->rndst[ist]);
      k = (int) (nbeta_minus_1*r);
    } while (k >= nbeta_minus_1); /* until k in [0..nbeta-2] */


    /* calculate metropolis ratio in selected adjacent temperature chains */ 
    k0 = k*Nseq + Iseq;       /* [k,nsec] in [nbeta,nsec] */
    k1 = (k+1)*Nseq + Iseq;   /* [k+1,nsec] in [nbeta,nsec] */
    r = exp(-0.5f*(mc->beta[k+1] - mc->beta[k]) * \
	    (mc->chi2c[k0] - mc->chi2c[k1]));
            
    /* calculate probability for acceptance */
    alpha = min(r, 1);
    /* exchange adjacent temperature chains with a probability alpha */
    u = curand_uniform(&mc->rndst[ist]);

    if (u <= alpha) {
      /* exchange tempertures */
      idum = mc->tcur[k0];
      mc->tcur[k0] = mc->tcur[k1];
      mc->tcur[k1] = idum;

      /* exchange parameters pcur[k,iseq,:] and pcur[k+1,iseq,:] */
      for(jprm = 0; jprm < mc->nptot; jprm++) {
          i0 = k0*mc->nptot + jprm;
          i1 = k1*mc->nptot + jprm;
          dum = mc->pcur[i0];
          mc->pcur[i0] =  mc->pcur[i1];
          mc->pcur[i1] = dum;
      } /* for(jprm = 0; jprm < mc->nptot; jprm++) */

      /* exchange chisquares */
      dum = mc->chi2c[k0];
      mc->chi2c[k0] = mc->chi2c[k1];
      mc->chi2c[k1] = dum;

      /* At the main stage, n_exch[k,iseq]++; */
      if (burn == 0) mc->n_exch[k*Nseq+Iseq]++;


    } /* if (u <= alpha)  */
  } /* for (i = 0; i < Nbeta; i++)  */

 /* 
  * This is the last iteration point. 
  * Increment the internal total iteration counter: 
  */
  if (Iseq == 0) mc->itr++;

  if (burn) return; /* At the burn-in stage pout[] is not updated */
  
  /*
   * Saving results
   */
  /* Save pcur in pout[itr], tcur in tout[itr], and chi2c in chi2 */
  for (ibeta = 0; ibeta < mc->nbeta; ibeta++) {
    for (ipm = 0; ipm < mc->nptot; ipm++) {
      /* if (invar[ipm] != -1)
       * pout[invar[ipm],ibeta,Iseq,itr] = pcur[ibeta,Iseq,ipm]; */
      if (mc->invar[ipm] != -1) { 
	ipo = ((mc->nbeta*mc->invar[ipm] + ibeta)*Nseq + Iseq)*mc->niter + itr;
	ipc = (Nseq*ibeta + Iseq)*mc->nptot + ipm;
	mc->pout[ipo] = mc->pcur[ipc];
      }
    } /* for (ipm = 0; ipm < mc->nptot; ipm++) */
    /* tout[ibeta,iseq,itr] = tcur[ibeta,iseq] */
    ito = (Nseq*ibeta + Iseq)*mc->niter + itr;
    itc = Nseq*ibeta + Iseq;
    mc->tout[ito] = mc->tcur[itc];
  } /* for (ibeta = 0; ibeta < mc->nbeta; ibeta++) */

  /* chi2[iseq,itr] = chi2c[0,iseq] */
  ico = mc->niter*Iseq + itr;
  /* icc = Nseq*0 + Iseq;  // Save chi2 for ibeta = 0 only */
  mc->chi2[ico] = mc->chi2c[Iseq];

}


//==========================================================

/*     calc_model<<<nblocks,blocksize>>>(mc_d);  */

__global__ void calc_model(CCalcModel *mc) {

  int idatm = blockDim.x*blockIdx.x + threadIdx.x; /* through index */
    
  if (idatm >= mc->nchi2m) return; /* The last block can have extra threads */

  int ist, ipt, idat;  //, iret;

  /*
   * ndat = nvis + ncph
   *
   * Treat the through index idatm as idatm ~ [ibeta,iseq,idat], or
   * idatm = (ibeta*nseq + iseq)*ndat + idat: 
   */
  idat = idatm%mc->ndat;
  ist =  idatm/mc->ndat;      /* ptent[ist] ~ ptent[ibeta,iseq] */
 
  if (mc->flag[ist] == 0) return; // ==== no calcs for rejected parameters ===>>

  ipt = ist*mc->nptot;     /* pcur[ipt] ~ pcur[ibeta,iseq,:] */

  /*
   * Call model with ipass=0 (last parameter)
   */
  model(mc, idat, ipt, idatm, 0);  /* Compute model at idatm */
}

#undef Nbeta
#undef Nseq
#undef Ibeta
#undef Iseq

//==========================================================
//} // extern "C"















void run_mcmc(CMcmcFit *mc_h, CMcmcFit *mc_d) {

  int t;      //, i, j;
  int nseq = mc_h->nseq, nbeta = mc_h->nbeta; 
  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int itr, niter = mc_h->niter, ipm, nprm = mc_h->nprm; 
  int nburn = mc_h->nburn;
  int burn; /* Flag 1: 'burn-in stage'; 0: 'main stage' */
  /* Pass 0: model() calculates amplitudes and phases; 
   * Pass 1: model() calculates closures from precalculated phases */
  int ipass = 0; 


  /* Loop parameters: */
  mc_h->ipm = 0;
  mc_h->itr = 0;

  printf("HOST imodel = %d\n", mc_h->imodel);
    printf("HOST pmint:\n");
    for (t = 0; t < 9; t++) printf("%g  ", mc_h->pmint_h[t]);
    printf("\n");
    printf("HOST pmaxt:\n");
    for (t = 0; t < 9; t++) printf("%g  ", mc_h->pmaxt_h[t]);
    printf("\n");
    printf("HOST ptotal:\n");
    for (t = 0; t < 9; t++) printf("%g  ", mc_h->ptotal_h[t]);
    printf("\n");

  /*
   * Initialization (mc->pstp tested - HEALTHY)
   */
  init_rng<<<nseq,nbeta>>>(mc_d);
  init_pcur<<<nseq,nbeta>>>(mc_d);
  for (ipass = 0; ipass < mc_h->npass; ipass++) {
    calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
  }
  calc_chi2<<<nseq,nbeta>>>(mc_d);

  printf("HOST nburn = %d, nprm = %d\n", nburn, nprm);

  //return;

  /* Burn-in stage */
  burn = 1;
  for (itr = 0; itr < nburn; itr++) {
    for (ipm = 0; ipm < nprm; ipm++) {
      gen_proposal<<<nseq,nbeta>>>(mc_d);
      for (ipass = 0; ipass < mc_h->npass; ipass++) {
          calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
      }
      spmp_mcmc<<<nseq,nbeta>>>(mc_d);
      adjust_steps<<<nseq,nbeta>>>(mc_d);  
    }
    replica_exchange<<<nseq,1>>>(mc_d, burn, itr);
    if (itr%200 == 0 && itr > 0) printf("BURN-IN: itr = %d done\n", itr);
  }
  printf("BURN-IN: total of %d iterations done\n", nburn);

  printf("Burn-in done\n");


  /* Main stage */
  burn = 0;
  for (itr = 0; itr < niter; itr++) {
    for (ipm = 0; ipm < nprm; ipm++) {
      gen_proposal<<<nseq,nbeta>>>(mc_d);
      for (ipass = 0; ipass < mc_h->npass; ipass++) {
          calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, ipass);
      }
      spmp_mcmc<<<nseq,nbeta>>>(mc_d);
    }
    replica_exchange<<<nseq,1>>>(mc_d, burn, itr);

    if (itr%200 == 0 && itr > 0) printf("MCMC: itr = %d done\n", itr);
  }
 
  printf("MCMC:  total of %d iterations done\n", niter);

  printf("GPU: runmcmc() done.\n");
  return;
}


/*======================================================================*/


void run_calcmodchi2(CCalcModel *mc_h, CCalcModel *mc_d) {

  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int nbeta = mc_h->nbeta, nseq = mc_h->nseq;

  printf("nbeta=%d, nseq=%d, nblocks=%d, blocksize=%d, mc_h->nchi2m=%d\n", 
	 mc_h->nbeta, mc_h->nseq, nblocks, blocksize, mc_h->nchi2m);

  calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, 0);
  calc_chi2_terms<<<nblocks,blocksize>>>(mc_d, 1);

  calc_chi2<<<nseq,nbeta>>>(mc_d);
  return;
}



/*======================================================================*/


void run_calcmodel(CCalcModel *mc_h, CCalcModel *mc_d) {

  int blocksize = 128;
  int nblocks= mc_h->nchi2m/blocksize + 1;     /* nchi2m = nbeta*nseq*ndat */
  int nbeta = mc_h->nbeta, nseq = mc_h->nseq;

  printf("nbeta=%d, nseq=%d, nblocks=%d, blocksize=%d, mc_h->nchi2m=%d\n", 
	 nbeta, nseq, nblocks, blocksize, mc_h->nchi2m);

  calc_model<<<nblocks,blocksize>>>(mc_d);

  return;
}




























