//
// model.cuh
//
// This is a template model file.
//
// The result - a single number - is stored in datm[imd].
//
// For convenience and brevity, define the variable and coefficient names
// (BE CAREFUL not to mix with the names in src/gpu_mcmc.cu !!!
// To avoid the mess, always undefine the names after use!):
//
// For example:
//
#define V1 (mc->pcur[ipt++])
#define V2 (mc->pcur[ipt])
#define C1 (mc->coor[0])
#define C2 (mc->coor[1])
#define K (mc->idat[0])
#define K1 (mc->idat[1])

__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat. 
   *   ipt: index of the first parameter in the set of model parameters 
   *        for ibeta == threadIdx.x and iseq == blockIdx.x. 
   *        mc->pcur[ipt] is the first model parameter, and 
   *        mc->pcur[ipt+mc->nptot-1] is the last parameter.
   *        If i is a parameter index, from 0 to mc->nptot-1,
   *        then mc->pcur[ipt+i] is the i-th parameter, or
   *        pcur[ibeta,iseq,i] in Python.
   *   imd: "through index" into datm[ibeta,isec,id]. Only used 
   *        to save the result before return.
   *   ipass: pass number, starting from 0. The model() function can be
   *          called multiple times, or in many passes, so pass is to know
   *          which time model() is called now.  
   *
   *
   * All member arrays and parameters of the class CCalcModel are accessible
   * here. Use the expressions like mc->coor, mc->idat, mc->ncoor, 
   * mc->nidat etc. See the class CCalcModel definition in mcmcjob.cuh and
   * mcmcjob.cu.
   * 
   * The result must be saved in
   *   mc->datm[imd]
   */

    //
    // Place your code here. 
    // Remember: it should only calculate one value
    // and save it in mc->datm[imd]. This value will further be compared
    // with its respective value in mc->dat[id] to be used in the chi^2
    // calculation.
    //
    
   return 1;
}

//
// Always undefine the names after use!  
//
#undef V1
#undef V2
#undef C1
#undef C2
#undef K
#undef K1

 
