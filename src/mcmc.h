/* sizeof(curandState) = 48 B */

#ifdef __cplusplus
extern "C" {
#endif

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
            int nbeta, int nseq, int nburn, int niter);


 void calc_modchi2(float *coor, float *dat, float *std2r,
	       int *icoor, int *idat, float *datm, float *chi2m,    
	       float *pcur, float *chi2c, int *flag, int imodel,
	       int ncoor, int ndat, int nicoor, int nidat, int ndatm,
	       int nprm, int nbeta, int nseq);


void calc_model(float *coor, int *idat, float *datm, float *pcur, 
		int imodel, int ncoor, int ndat, int nidat, 
		int ndatm, int nptot, int nbeta, int nseq);


#ifdef __cplusplus
};
#endif


