//#include <curand_kernel.h>

/* #define CUDA_CALL(x) do { if((x) != cudaSuccess) {	\ */
/*       printf("Error at %s:%d\n",__FILE__,__LINE__);	\ */
/*       exit(EXIT_FAILURE);}} while(0) */
#define CUDA_CALL(call) { cudaAssert(call,__FILE__,__LINE__); }


#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost
#define MAX_NPRM 9
#ifdef MCMC_DOUBLE
#define pi  (3.1415926535897931) 
#define pi4 (12.566370614359172) /* 4*3.1415926535897931 */
#else
#define pi  (3.1415926535897931f) 
#define pi4 (12.566370614359172f) /* 4*3.1415926535897931f */
#endif

#ifdef __cplusplus


/*====================================================================*/
/*                BASE CLASS CCalcModel                               */


class CCalcModel {
 public:
        
  int imodel;
  int ncoor;
  int ndat; 
  int nicoor; 
  int nidat; 
  int ndatm; 
  int nptot; /* Keep 'nptot' because run_calcmodchi2() uses it */
  int nbeta;
  int nseq;
  int nchi2m; /* Number of threads in model calculation */

  /*
   * Array sizes
   */
  int s_coor;   /* = sizeof(float)*ncoor; */
  int s_dat;    /* = sizeof(float)*ndat; */
  int s_icoor;  /* = sizeof(int)*nicoor; */
  int s_idat;   /* = sizeof(int)*nidat; */
  int s_datm;   /* = sizeof(float)*ndatm; */
  int s_chi2m;  /* = sizeof(float)*ndatm; */
  int s_pcur;   /* = sizeof(float)*nbeta*nseq*nptot; */
  int s_bsint;  /* = sizeof(int)*nbeta*nseq; */
  int s_bsflt;  /* = sizeof(float)*nbeta*nseq; */

  /*
   * DEVICE pointers to data. 
   * nseq is the number of streaming multiprocessors.
   */
  /* Coordinates, Observables and Model Data, Integer Data */
  float *coor;   /* [ncoor]:             Coordinates: UV etc */
  float *dat;    /* [ndat]:              Observation data */
  float *std2r;  /* [ndat]:              Data std^2 recipr. */
  int   *icoor;  /* [nicoor]:            Any integer coordinates */
  int   *idat;   /* [nidat]:             Any integer data */
  float *datm;   /* [ndatm]:             Model data */
  float *chi2m;  /* [nbeta][nseq][ndat]: Terms of chi^2 sum in datm*/
  /* Parameters */
  float *pcur;     /* [nbeta][nseq][nptot] */
  int *flag;       /* [nbeta][nseq] */
  float *chi2c;    /* [nbeta][nseq] */

  /*
   * HOST Pointers to data 
   */
  float *coor_h;   /* [ncoor]:             Coordinates: UV etc */
  float *dat_h;    /* [ndat]:              Observation data */
  float *std2r_h;  /* [ndat]:              Data std^2 recipr. */
  int   *icoor_h;  /* [nicoor]:            Any integer data */
  int   *idat_h;   /* [nidat]:             Any integer data */
  float *datm_h;   /* [ndatm]: Model data */
  float *chi2m_h;  /* [nbeta][nseq][ndat]: Terms of chi^2 sum in datm_h*/
  /* Parameters */
  float *pcur_h;     /* [nbeta][nseq][nptot] */
  int *flag_h;       /* [nbeta][nseq] */
  float *chi2c_h;    /* [nbeta][nseq] */

 
  CCalcModel() {}      /* Default constructor */

  CCalcModel(int imodel, 
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
	   float *pcur,   /* [nbeta][nseq][nptot] */ 
	   int *flag,   /* [nbeta][nseq] */ 
	   float *chi2c);  /* [nbeta][nseq] */

  ~CCalcModel();

  void do_model_chi2_gpu();
  void do_calc_model_gpu();
};



/*====================================================================*/
/*                        DERIVED CLASS CMcmcFit                      */




class CMcmcFit: public CCalcModel {
 public:
        
  int nprm;
  int nadj;
  int npass;
  int nburn;
  int niter;
  int nrndst;
  int ipm;  /* Loop parameter: current model parameter */
  int itr;  /* Loop parameter: current iteration */
  unsigned long long seed;
  //int nchi2m; /* Number of threads in model calculation */
  int ntitr;  /* Total number of iterations */

  /*
   * Array sizes
   */
  int s_rndst;  /* = sizeof(curandState)*nrndst); */
  int s_ptot;   /* = sizeof(float)*nptot; */
  int s_prm;    /* = sizeof(float)*nprm; */
  int s_pout;   /* = sizeof(float)*nprm*nbeta*nseq*niter; */
  int s_tout;   /* = sizeof(int)*nbeta*nseq*niter; */
  int s_chi2;   /* = sizeof(float)*nseq*niter; */
  int s_beta;   /* = sizeof(float)*nbeta; */
  int s_acpt;   /* = sizeof(int)*nprm*nbeta*nseq; */
  int s_hist;   /* = sizeof(float)*nadj*nprm*nbeta*nseq; */
  //int s_bsflt;  /* = sizeof(float)*nbeta*nseq; */
  /* int s_bsint;  /\* = sizeof(int)*nbeta*nseq; *\/ */
  int s_bsptot; /* = sizeof(float)*nbeta*nseq*nptot; */
  int s_4bstf;   /* = sizeof(float)*nbeta*nseq*ntitr; */
  int s_bstf;   /* = sizeof(float)*nbeta*nseq*ntitr; */
  int s_bsti;   /* = sizeof(int)*nbeta*nseq*ntitr; */
  int s_burn;   /* = sizeof(float)*nprm*nbeta*nseq*nburn; */

  /*
   * DEVICE pointers to data. 
   * nseq is the number of streaming multiprocessors.
   */
  /* Random Number Generators' States */
  //curandStateMtgp32 *rndst;     /* [nbeta][nseq] */
  curandState *rndst;     /* [nbeta][nseq] */
  /* Parameters */
  int   *pdescr;   /* [nptot] */
  int   *ivar;     /* [nprm] */
  int   *invar;    /* [nptot] */
  float *ptotal;   /* [nptot] */
  float *pmint;    /* [nptot] */ 
  float *pmaxt;    /* [nptot] */
  /* MCMC Algorithm */
  float *pout;     /* [nprm][nbeta][niter,nseq] */
  int *tout;       /* [nbeta][nseq][niter] */
  float *chi2;     /* [nseq][niter] */
  // float *chi2c;    /* [nbeta][nseq] */
  int *tcur;       /* [nbeta][nseq] */
  //  int *flag;       /* [nbeta][nseq] */
  int *n_acpt;     /* [nprm][nbeta][nseq] */
  int *n_exch;     /* [nbeta][nseq] */
  int *n_cnt;      /* [nprm][nbeta][nseq] */
  int *n_hist;     /* [nadj][nprm][nbeta][nseq] */
  float *pstp;     /* [nprm][nbeta][nseq] */
  float *beta;     /* [nbeta] */
  float *ptent;    /* [nbeta][nseq] */
  int *ptentn;     /* [nbeta][nseq] */

  /*
   * HOST Pointers to data 
   */
  /* Parameters */
  int   *pdescr_h;   /* [nptot] */
  int   *ivar_h;     /* [nprm] */
  int   *invar_h;    /* [nptot] */
  float *ptotal_h;   /* [nptot] */
  float *pmint_h;    /* [nptot] */ 
  float *pmaxt_h;    /* [nptot] */
  /* MCMC Algorithm */
  float *pout_h;     /* [nprm][nbeta][niter,nseq] */
  int *tout_h;       /* [nbeta][nseq][niter] */
  float *chi2_h;     /* [nseq][niter] */
  //float *chi2c_h;    /* [nbeta][nseq] */
  int *tcur_h;       /* [nbeta][nseq] */
  //  int *flag_h;       /* [nbeta][nseq] */
  int *n_acpt_h;     /* [nprm][nbeta][nseq] */
  int *n_exch_h;     /* [nbeta][nseq] */
  int *n_cnt_h;      /* [nprm][nbeta][nseq] */
  int *n_hist_h;     /* [nadj][nprm][nbeta][nseq] */
  float *pstp_h;     /* [nprm][nbeta][nseq] */
  float *beta_h;     /* [nbeta] */
  float *ptent_h;    /* [nbeta][nseq] */
  int *ptentn_h;     /* [nbeta][nseq] */
  curandState *rndst_h;     /* [nbeta][nseq] */
  //curandStateMtgp32 *rndst_h;     /* [nbeta][nseq] */

  
/*=========================================================================*/ 

  
  CMcmcFit(int imodel, 
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
       float *pcur);  /* [nbeta][nseq][nptot] */ 


  ~CMcmcFit();

  void do_mcmc_on_gpu();

};


/*====================================================================*/








void run_mcmc(CMcmcFit *mc_h, CMcmcFit *mc_d);
void run_calcmodchi2(CCalcModel *mc_h, CCalcModel *mc_d);
void run_calcmodel(CCalcModel *mc_h, CCalcModel *mc_d);
void cudaAssert(const cudaError err, const char *file, const int line);

#endif

