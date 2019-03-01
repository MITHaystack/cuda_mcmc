//#include <curand_kernel.h>
//#include <stdio.h>

__device__ void sgra_model9(float ul, float vl, float *prm, 
			   float *Vamp, float *Vpha);
__device__ void sgra_model13(float ul, float vl, float *prm, 
			   float *Vamp, float *Vpha);

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
   * The result must be saved in
   *   mc->datm[imd]
   *
   * All member arrays and parameters of the class CCalcModel are accessible
   * here. Use the expressions like mc->coor, mc->idat, mc->ncoor, 
   * mc->nidat etc. See the class CCalcModel definition in mcmcjob.cuh and
   * mcmcjob.cu.
   * 
   */


  /* Data and coordinates */
  int nvis = mc->idat[0];
  int nchi2m = mc->nchi2m; /* Number of chi2 terms & start of mphase arrays */
  int ivis = imd/mc->ndat; /* Throug-index ph[ivis,:] ~ ph[ibeta,iseq,:] */

  //float *phase =  mc->datm + nchi2m + ivis*nvis + id; /* phase[ibeta,iseq,i]*/
  float *phase =  mc->datm + nchi2m + ivis*nvis; /* phase[ibeta,iseq,i]*/

  if (ipass == 0 && id < nvis) { /* Amplitude and phase model calculation */
    int iuv = 2*id;
    float ul = mc->coor[iuv];
    float vl = mc->coor[iuv+1];
    float *prm = &mc->pcur[ipt];
    /*
     *   imodel: a model number. Here  imodel = 1 means sgra_model9(), and
     *                                 imodel = 2 means sgra_model13().
     */
    // if      (mc->imodel == 1) model_ptr = sgra_model9;
    // else if (mc->imodel == 2) model_ptr = sgra_model13;
    // else return -1;                 /* Error: wrong model number */
    if (mc->imodel == 1) {
      sgra_model9(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    }
    else if (mc->imodel == 2) { 
      sgra_model13(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    }
    else return -1;                 /* Error: wrong model number */

    /* (*model_ptr)(ul, vl, prm,  &mc->datm[imd], &phase[id]); */
    //sgra_model9(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    //sgra_model13(ul, vl, prm,  &mc->datm[imd], &phase[id]);
    
    return 1;
  }


  else if (ipass == 1 && id >= nvis) { /* Closures from saved model phases */
    float cph;
    if (mc->idat[2] == 0) { /* cpExt == 0: calc cphs from phases */
      int icph = 3*(id - nvis); /* Phase index triplet number, 0 : 3*(ncph-1) */
      int i = mc->icoor[icph++];
      int j = mc->icoor[icph++];
      int k = mc->icoor[icph];
      //phase = phase - id - nvis; /* Here phase points at phase[0,0,0] */
      phase =  mc->datm + nchi2m + ivis*nvis;
      cph = phase[i] + phase[j] - phase[k];     /* Phase closure */
    }
    else { /* cpExt != 0: cphs and uvcp are provided */
      float *prm = &mc->pcur[ipt];
      float amp, ph1, ph2, ph3;
      float u1, u2, u3, v1, v2, v3;
      int icph = 4*(id - nvis) + 2*nvis; /* Index of [u1,v1,u2,v2] in coor */
      u1 = mc->coor[icph++]; 
      v1 = mc->coor[icph++];
      u2 = mc->coor[icph++];
      v2 = mc->coor[icph];
      u3 = -(u1 + u2);
      v3 = -(v1 + v2);
      if (mc->imodel == 1) {
	sgra_model9(u1, v1, prm,  &amp, &ph1);
	sgra_model9(u2, v2, prm,  &amp, &ph2);
	sgra_model9(u3, v3, prm,  &amp, &ph3);
      }
      else if (mc->imodel == 2) { 
	sgra_model13(u1, v1, prm,  &amp, &ph1);
	sgra_model13(u2, v2, prm,  &amp, &ph2);
	sgra_model13(u3, v3, prm,  &amp, &ph3);
      }
      else return -1;                 /* Error: wrong model number */
      cph = ph1 + ph2 - ph3;
    }
    mc->datm[imd] = atan2(sin(cph), cos(cph));

    return 2;
  }

  else
    return 0;
}

/*========================================================================*/


__device__ void sgra_model13(float ul, float vl, float *prm, 
			     float *Vamp, float *Vpha) {

  /* Model xringaus2 (13-parameter)
   * Arguments:
   * ul, vl: u and v coordinates in Gigalambda
   * prm: model parameters
   * Result:
   * Vamp: visibility amplitude of the model at (ul,vl) points 
   * Vpha: visibility phase (radians) of the model at (ul,vl) points 
   */

    
  /* Locals */
  //int i = blockDim.x*blockIdx.x + threadIdx.x;
  float Zsp, Rex, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th;
  float D, d, gsy, gsx, h, h0, etth;                          /* float d1; */
  float U, V; /* Coordinates rortated by th */
  float rho, arg, ree, ime, rei, imi;
  float Rin, aRin, aRex;
  float gshift, fwhm2std, Gamp, Gph;
  float d_re, Fr_re, Fg_re, Fin_re;
  float d_im, Fr_im, Fg_im, Fin_im;
  float Vim, Vre;

  /* Zsp, Rex, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th = vparam */
  /* Zsp: zero spacing in any units (Jy or such) */
  /* Re: external  ring radius in microarcseconds */ 
  /* rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1 */
  /* ecc: eccentricity of inner circle center from that of outer; in [-1..1] */
  /* fade: [0..1], "noncontrast" of the ring, 1-uniform, 0-zero to maximum */
  /* gr: Rg/Re, Rg - distance to Gaussian center */
  /* gax: Gaussian main axis, expressed in Re */
  /* aq: axes quotient, aq = gsx/gsy. 0 < aq < 1 */
  /* gq: fraction of Gaussian visibility against ring visibility. */
  /*     0-ring only, 1-Gaussian only */
  /* alpha: angle of the Gaussian center orientation in radians */
  /* beta: angle of the Gaussian rotation in radians */
  /* eta: angle of internal circular hole orientation in radians */
  /* th: angle of slope orientation in radians */

  Zsp = prm[0];
  Rex = prm[1];
  rq = prm[2];
  ecc = prm[3];
  fade = prm[4]; 
  gr = prm[5]; 
  gax = prm[6]; 
  aq  = prm[7];
  gq = prm[8]; 
  alpha = prm[9];
  beta = prm[10];
  eta = prm[11];
  th = prm[12];

  /*
   * Convert radii and Gaussian axes from microarcseconds into nanoradians
   * The calculations are performed in the units of Gigalambda which
   * corresponds to the RA-Dec dimensions in nanoradians
   * Rex/3600.: uas -> udeg; radians(Rex/3600.): udeg -> urad
   * radians(Rex/3600.)*1e3: urad -> nrad
   */

  Rex = (pi/180.0f)*(Rex/3600.0f)*1e3f;  /* microarcseconds -> nanoradians */
  Rin = rq*Rex;        /* 0 < rq < 1 is the Rin/Rex quotient */
  /*
   * First calculate the Gaussian
   */
  if (gax != 0. && aq != 0. && gq != 0.) {
    gsy = gax*Rex;   /* nrad. The Gaussian's axis tangential to the ring */
    gsx = aq*gsy;    /* nrad. The Gaussian's radial axis normal to the ring */
    /*
     * Rotate coordinates by beta angle (to rotate Gaussian)
     */
    U =  ul*cos(beta) + vl*sin(beta);
    V = -ul*sin(beta) + vl*cos(beta);
    gshift = gr*Rex*(cos(alpha)*ul + sin(alpha)*vl);   /* Gaussian's shift */
    fwhm2std = 2.0f*pow(pi/(2.0f*sqrt(2.0f*log(2.0f))),2);
    Gamp = exp(-fwhm2std*(pow(U*gsx,2) + pow(V*gsy,2)));
    Gph = -2.0f*pi*gshift;
    /* Displaced Gaussian Fg = Gamp*(cos(Gph) + I*sin(Gph)): */
    Fg_re = Gamp*cos(Gph);
    Fg_im = Gamp*sin(Gph);
  }
  if (gq == 1.) {   /* Gaussian only */
    Vre = Zsp*Fg_re;
    Vim = Zsp*Fg_im;    
    *Vamp = sqrt(Vre*Vre + Vim*Vim);
    *Vpha = atan2(Vim, Vre);
    return;                     //  ==================================>>>
  }

  D = ecc*(Rex - Rin); /* nrad, displacement of inner circle center wrt outer */
  etth = eta - th;
  d =  D*cos(etth);
  /* d1 =  D*sin(etth); */
  // gsy = gax*Rex;   /* nrad. The Gaussian's axis tangential to the ring */
  // gsx = aq*gsy;    /* nrad. The Gaussian's radial axis normal to the ring */
  /* Convert FWHM^2 to stdev^2 */
  h = (2.0f/pi)/((pow(Rex,2) - pow(Rin,2)*(1.0f + d/Rex))*fade
	       + (pow(Rex,2) - pow(Rin,2)*(1.0f - d/Rex)));
  h0 = fade*h;

  /*
   * Rotate coordinates by th angle
   */
  U =  ul*cos(th) + vl*sin(th);
  V = -ul*sin(th) + vl*cos(th);

  /*
   * Eccentric ring
   * h is the ring "height" at which its Zsp = 1.
   */
  rho = sqrt(pow(U,2) + pow(V,2));  /* UV-space baselines in Glam */

  if (rho > 1e-7f) {
    /*
     * At a point where rho > 0 use normal expressions
     */
    arg = 2.0f*pi*rho;   /* in Glambdas */
    aRex = arg*Rex;

    ree = .5f*(h+h0)*Rex*j1(aRex)/rho;
    ime = (h-h0)/pi4*(pi*Rex*(j0(aRex) - jn(2,aRex))/pow(rho,2) 
		      - j1(aRex)/pow(rho,3))*U;
    if (Rin != 0.) {
      aRin = arg*Rin;
      rei = .5f*(h+h0 - d*(h-h0)/Rex)*Rin*j1(aRin)/rho;
      imi = (h-h0)/pi4*rq*(pi*Rin*(j0(aRin) - jn(2,aRin))/pow(rho,2) 
			 - j1(aRin)/pow(rho,3))*U;
    }
  }
  else {
    /*
     * At the singularity point where rho ~ 0,
     * calculate the rho->0 limits for ree, ime, rei, and imi
     */    
    ree = .5f*(h+h0)*pi*pow(Rex,2);
    ime = -.25f*pow(pi,2)*(h-h0)*pow(Rex,3)*U;             /* here U == 0? */
    if (Rin != 0.) {
      rei = .5f*(h+h0 - d*(h-h0)/Rex)*pi*pow(Rin,2);
      imi = -.25f*pow(pi,2)*(h-h0)*(pow(Rin,4)/Rex)*U;       /* here U == 0? */
    }
  }

  /*
   * Complex Fourier image for "slanted" eccentric ring  
   */
  //Fex = ree + I*ime;       /* External slanted pillbox */
  //Fin = Dis*(rei + I*imi);  
  if (Rin != 0.) {
    d_re = cos(2.0f*pi*d*U); /* Displacement factor for inner ring - real */
    d_im = sin(2.0f*pi*d*U); /* Displacement factor for inner ring - imag */
    /* Inner slanted pillbox shifted by d: */
    Fin_re = rei*d_re - imi*d_im;
    Fin_im = rei*d_im + imi*d_re;
    /* Visibility of the slanted ring: Fr = Fex - Fin: */
    Fr_re = ree - Fin_re;
    Fr_im = ime - Fin_im; 
  }
  else {
    /* Visibility of the slanted ring: Fr = Fex - Fin: */
    Fr_re = ree;
    Fr_im = ime; 
  }
  /*
   * Shifted Gaussian, if specified
   */
  if (gax == 0.0f || aq == 0.0f || gq == 0.0f) { /* Gaussian not specified */
    Vre = Zsp*Fr_re;
    Vim = Zsp*Fr_im;
  }
  else {
    // /*
    //  * Rotate coordinates by beta angle (to rotate Gaussian)
    //  */
    // U =  ul*cos(beta) + vl*sin(beta);
    // V = -ul*sin(beta) + vl*cos(beta);

    // /* Gaussian centered at he middle of inner ring */
    // /* xg = .5*(Rex + Rin - d); */ 
    // gshift = gr*Rex*(cos(alpha)*ul + sin(alpha)*vl);   /* Gaussian's shift */
    // Gamp = exp(-fwhm2std*(pow(U*gsx,2) + pow(V*gsy,2)));
    // Gph = -2.0f*pi*gshift;
    // /* Displaced Gaussian Fg = Gamp*(cos(Gph) + I*sin(Gph)): */
    // Fg_re = Gamp*cos(Gph);
    // Fg_im = Gamp*sin(Gph);
    // /* Mix ring and Gaussian in the gq proportion */
    // /* Actually, F = (1-gq)*Fr + gq*Fg; */ 
    Vre = Zsp*((1.0f-gq)*Fr_re + gq*Fg_re);
    Vim = Zsp*((1.0f-gq)*Fr_im + gq*Fg_im);    
  }

  *Vamp = sqrt(Vre*Vre + Vim*Vim);
  *Vpha = atan2(Vim, Vre);
}


 
/*========================================================================*/


__device__ void sgra_model9(float ul, float vl, float *prm, 
			     float *Vamp, float *Vpha) {

  /* Model xringaus (9-parameter)
   * Arguments:
   * ul, vl: u and v coordinates in Gigalambda
   * prm: model parameters
   * Result:
   * Vamp: visibility amplitude of the model at (ul,vl) points 
   * Vpha: visibility phase (radians) of the model at (ul,vl) points 
   */

    
  /* Locals */
  //int i = blockDim.x*blockIdx.x + threadIdx.x;
  float Zsp, Rex, rq, ecc, fade, gax, aq, gq, th;
  float d, gsy, gsx, h, h0;
  float U, V; /* Coordinates rortated by th */
  float rho, arg, ree, ime, rei, imi;
  float Rin, aRin, aRex;
  float gshift, gcoef, Gamp, Gph;
  float d_re, Fr_re, Fg_re, Fin_re;
  float d_im, Fr_im, Fg_im, Fin_im;
  float Vim, Vre;

  /* Zsp, Re, rq, ecc, fade, gax, aq, gq, th = vparam */
  Zsp = prm[0];
  Rex = prm[1];
  rq = prm[2];
  ecc = prm[3];
  fade = prm[4]; 
  gax = prm[5]; 
  aq  = prm[6];
  gq = prm[7]; 
  th = prm[8];

  /*
   * Convert radii and Gaussian axes from microarcseconds into nanoradians
   * The calculations are performed in the units of Gigalambda which
   * corresponds to the RA-Dec dimensions in nanoradians
   * Rex/3600.: uas -> udeg; radians(Rex/3600.): udeg -> urad
   * radians(Rex/3600.)*1e3: urad -> nrad
   */

  Rex = (pi/180.0f)*(Rex/3600.0f)*1e3f;  /* microarcseconds -> nanoradians */
  Rin = rq*Rex;        /* 0 < rq < 1 is the Rin/Rex quotient */
  d = ecc*(Rex - Rin); /* nrad, displacement of inner circle center wrt outer */
  gsy = gax*Rex;       /* nrad. The Gaussian's axis tangential to the ring */
  gsx = aq*gsy;       /* nrad. The Gaussian's radial axis normal to the ring */
  /* Convert FWHM^2 to stdev^2 */
  gcoef = 2.0f*pow(pi/(2.0f*sqrt(2.0f*log(2.0f))),2);
  h = (2.0f/pi)/((pow(Rex,2) - pow(Rin,2)*(1.0f + d/Rex))*fade
	       + (pow(Rex,2) - pow(Rin,2)*(1.0f - d/Rex)));
  h0 = fade*h;

  /*
   * Rotate coordinates by th angle
   */
  U =  ul*cos(th) + vl*sin(th);
  V = -ul*sin(th) + vl*cos(th);

  /*
   * Eccentric ring
   * h is the ring "height" at which its Zsp = 1.
   */
  rho = sqrt(pow(U,2) + pow(V,2));  /* UV-space baselines in Glam */

  if (rho > 1e-7f) {
    /*
     * At a point where rho > 0 use normal expressions
     */
    arg = 2.0f*pi*rho;   /* in Glambdas */
    aRex = arg*Rex;

    ree = .5f*(h+h0)*Rex*j1(aRex)/rho;
    ime = (h-h0)/pi4*(pi*Rex*(j0(aRex) - jn(2,aRex))/pow(rho,2) 
		      - j1(aRex)/pow(rho,3))*U;
    if (Rin != 0.) {
      aRin = arg*Rin;
      rei = .5f*(h+h0 - d*(h-h0)/Rex)*Rin*j1(aRin)/rho;
      imi = (h-h0)/pi4*rq*(pi*Rin*(j0(aRin) - jn(2,aRin))/pow(rho,2) 
			 - j1(aRin)/pow(rho,3))*U;
    }
  }
  else {
    /*
     * At the singularity point where rho ~ 0,
     * calculate the rho->0 limits for ree, ime, rei, and imi
     */    
    ree = .5f*(h+h0)*pi*pow(Rex,2);
    ime = -.25f*pow(pi,2)*(h-h0)*pow(Rex,3)*U;             /* here U == 0? */
    if (Rin != 0.) {
      rei = .5f*(h+h0 - d*(h-h0)/Rex)*pi*pow(Rin,2);
      imi = -.25f*pow(pi,2)*(h-h0)*(pow(Rin,4)/Rex)*U;       /* here U == 0? */
    }
  }

  /*
   * Complex Fourier image for "slanted" eccentric ring  
   */
  //Fex = ree + I*ime;       /* External slanted pillbox */
  //Fin = Dis*(rei + I*imi);  
  if (Rin != 0.) {
    d_re = cos(2.0f*pi*d*U); /* Displacement factor for inner ring - real */
    d_im = sin(2.0f*pi*d*U); /* Displacement factor for inner ring - imag */
    /* Inner slanted pillbox shifted by d: */
    Fin_re = rei*d_re - imi*d_im;
    Fin_im = rei*d_im + imi*d_re;
    /* Visibility of the slanted ring: Fr = Fex - Fin: */
    Fr_re = ree - Fin_re;
    Fr_im = ime - Fin_im; 
  }
  else {
    /* Visibility of the slanted ring: Fr = Fex - Fin: */
    Fr_re = ree;
    Fr_im = ime; 
  }
  /*
   * Shifted Gaussian, if specified
   */
  if (gax == 0.0f || aq == 0.0f || gq == 0.0f) { /* Gaussian not specified */
    Vre = Zsp*Fr_re;
    Vim = Zsp*Fr_im;
  }
  else {
    /* Gaussian centered at he middle of inner ring */
    /* xg = .5*(Rex + Rin - d); */ 
    gshift = Rin - d; /* Gaussian centered at the inner edge of inner ring */
    Gamp = exp(-gcoef*(pow(U*gsx,2) + pow(V*gsy,2)));
    Gph = -2.0f*pi*gshift*U;
    /* Displaced Gaussian Fg = Gamp*(cos(Gph) + I*sin(Gph)): */
    Fg_re = Gamp*cos(Gph);
    Fg_im = Gamp*sin(Gph);
    /* Mix ring and Gaussian in the gq proportion */
    /* Actually, F = (1-gq)*Fr + gq*Fg; */ 
    Vre = Zsp*((1.0f-gq)*Fr_re + gq*Fg_re);
    Vim = Zsp*((1.0f-gq)*Fr_im + gq*Fg_im);    
  }
  *Vamp = sqrt(Vre*Vre + Vim*Vim);
  *Vpha = atan2(Vim, Vre);
}


 
