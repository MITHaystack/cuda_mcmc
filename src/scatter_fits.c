/*
 * scatter_fits.c
 *
 * Author: Vincent Fish, MIT Haystack observatory, 2013
 * 
 * Original name: scatter-better.c
 *
 * COMPILATION:
 *
 * gcc -o scatter_fits scatter_fits.c -lm -lnsl -lcfitsio -lfftw3
 *
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <fitsio.h>
#include <fftw3.h>

#define MAXFILENAMELENGTH 300
#define RETURN_SCATTERED_IMAGE 1
#define RETURN_SCATTERED_UV_AMP 2
#define RETURN_SCATTERED_UV_PHASE 3

#define DEFAULT_BMAJ 22.0
#define DEFAULT_BMIN 11.0
#define DEFAULT_BPA 78.0
#define DEFAULT_RETURN_TYPE 1



struct parameterstruc {
  char inputfilename[MAXFILENAMELENGTH];
  char outputfilename[MAXFILENAMELENGTH];
  double bmaj, bmin, bpa;
  int returntype, nomove;
  int dozero;
  double zeroat;
};

int get_cmdline_options(int argc, char *argv[], struct parameterstruc *parameters);
void printusage();


int main(int argc, char *argv[])
{
  fitsfile *fptr, *fptr_out;
  char keystring[9],comment[8];
  double keyvalue[1];
  int status=0;
  int hdutype,naxis,nulval,anynul,ikeyvalue[1];
  long naxes[2],fpixel[2]={1,1};
  double cdelt1,cdelt2,*data,*backdata;
  double *data_rearrange,*backdata_rearrange;
  long i,row,col,nrows,ncolumns,index,nelements=1;
  fftw_complex *in,*vis,*back,*vis_rearrange,*vis_back;
  fftw_plan plan,inv_plan;
  double uvstep,*ulam,*vlam,*visamp,*visphase;
  double bmaj,bmin,bpa,sigmamaj,sigmamin,uvsigmamaj,uvsigmamin;
  double uvmaj,uvmin,downweight,uvdist;
  struct parameterstruc *parameters;

  parameters = malloc(sizeof(struct parameterstruc));
  
  // Initialize
  parameters->bmaj = (DEFAULT_BMAJ);
  parameters->bmin = (DEFAULT_BMIN);
  parameters->bpa  = (DEFAULT_BPA);
  parameters->returntype = (RETURN_SCATTERED_IMAGE);
  parameters->nomove = 0;
  parameters->dozero = 0;
  
  // Read parameters
  if (get_cmdline_options(argc,argv,parameters)>0){
    printusage();
    return(0);
  }

  // Put into correct units
  bmaj=parameters->bmaj*(1.0e-6)*(1/3600.)*(M_PI)/180; /* uas -> rad */
  bmin=parameters->bmin*(1.0e-6)*(1/3600.)*(M_PI)/180; /* uas -> rad */
  bpa =parameters->bpa*(M_PI)/180;                     /* deg -> rad */
  parameters->zeroat *= 1000000.0;

  // open the data file and get file info
  fits_open_file(&fptr,parameters->inputfilename,READONLY,&status);
  fits_movabs_hdu(fptr,1,&hdutype,&status);
  fits_get_img_dim(fptr,&naxis,&status);
  fits_get_img_size(fptr,2,naxes,&status);
  nrows=naxes[0];
  ncolumns=naxes[1];
  for (i=0;i<naxis;i++){ nelements *= naxes[i];}
  fits_read_key(fptr,TDOUBLE,"CDELT1",&cdelt1,NULL,&status);
  fits_read_key(fptr,TDOUBLE,"CDELT2",&cdelt2,NULL,&status);
  uvstep = 180/(cdelt2*naxes[1]*(M_PI));
  printf("cdelt2=%g\n", cdelt2);
  printf("uvstep=%le nrows=%li ncolumns=%li\n",uvstep,nrows,ncolumns);

  // compute scattering ellipse size in the uv plane
  sigmamaj = bmaj/(2*sqrt(log(4.0)));
  sigmamin = bmin/(2*sqrt(log(4.0)));
  uvsigmamaj = 1.0/(2*sigmamaj*(M_PI));
  uvsigmamin = 1.0/(2*sigmamin*(M_PI));
  printf("%le %le\n",uvsigmamaj,uvsigmamin);

  // set up data arrays
  data=malloc(nelements*sizeof(double));
  data_rearrange=malloc(nelements*sizeof(double));
  ulam = malloc(nelements*sizeof(double));
  vlam = malloc(nelements*sizeof(double));
  visamp=malloc(nelements*sizeof(double));
  visphase=malloc(nelements*sizeof(double));
  backdata=malloc(nelements*sizeof(double));
  backdata_rearrange=malloc(nelements*sizeof(double));

  fits_read_pix(fptr,TDOUBLE,fpixel,nelements,&nulval,data,&anynul,&status);

  // move image from center to corner (fftshift())
  for (i=0;i<nelements;i++){
    row = (long int)(i/ncolumns);
    col = (long int)(i%ncolumns);
    row += (nrows/2);
    col += (ncolumns/2);
    if (row>=nrows) {row -= nrows;}
    if (col>=ncolumns) {col -= ncolumns;}
    index = row*ncolumns+col;
    //if ((i%512)==0) {printf("%li %li %li %li\n",i,row,col,index);}
    if (parameters->nomove){data_rearrange[i] = data[i];}
    else {data_rearrange[index] = data[i];}
  } 


  //printf("pixel 234*512+241: %le\n",data[234*512+241]);
  // 234*512+241 is row 235 (starting at 1), column 242 (starting at 1)

  // set up output file
  fits_create_file(&fptr_out,parameters->outputfilename,&status);
  fits_copy_header(fptr,fptr_out,&status);
  if ((parameters->returntype==(RETURN_SCATTERED_UV_AMP)) || (parameters->returntype==(RETURN_SCATTERED_UV_PHASE))){
    strcpy(keystring,"UU---SIN");
    strcpy(comment," ");
    fits_update_key(fptr_out,TSTRING,"CTYPE1",keystring,comment,&status);
    strcpy(keystring,"VV---SIN");
    fits_update_key(fptr_out,TSTRING,"CTYPE2",keystring,comment,&status);
    keyvalue[0] = -uvstep;
    fits_update_key(fptr_out,TDOUBLE,"CDELT1",keyvalue,comment,&status);
    keyvalue[0] = uvstep;
    fits_update_key(fptr_out,TDOUBLE,"CDELT2",keyvalue,comment,&status);
    keyvalue[0] = 257;
    fits_update_key(fptr_out,TDOUBLE,"CRPIX1",keyvalue,comment,&status);
    fits_update_key(fptr_out,TDOUBLE,"CRPIX2",keyvalue,comment,&status);
  }

  // set up fftw data arrays
  in   = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nelements);
  vis  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nelements);
  vis_rearrange = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nelements);
  vis_back = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nelements);
  back = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*nelements);

  plan = fftw_plan_dft_2d(nrows,ncolumns,in,vis,FFTW_FORWARD,FFTW_ESTIMATE);

  // put data into fftw structure
  for (i=0;i<nelements;i++){
    in[i][0] = data_rearrange[i];
    in[i][1] = 0;
  }

  fftw_execute(plan);


  // rearrange uv data to usable order (peak in center)
  for (i=0;i<nelements;i++){
    row = (long int)(i/ncolumns);
    col = (long int)(i%ncolumns);
    row += (nrows/2);
    col += (ncolumns/2);
    if (row>=nrows) {row -= nrows;}
    if (col>=ncolumns) {col -= ncolumns;}
    index = row*ncolumns+col;
    //if ((i%10000)==0) {printf("%li %li %li %li\n",i,row,col,index);}
    vis_rearrange[index][0] = vis[i][0];    
    vis_rearrange[index][1] = vis[i][1];    
    ulam[index] = -(row-(nrows/2))*uvstep;
    vlam[index] = (col-(ncolumns/2))*uvstep;
  }


  FILE *fd = fopen("uv.txt", "w");
  printf("nelements=%d\n", nelements);

  // apply scattering (Convolution==multiplication in UV domain)
  for (i=0;i<nelements;i++){
    uvmaj = (ulam[i]*cos(-bpa))-(vlam[i]*sin(-bpa));
    uvmin = (ulam[i]*sin(-bpa))+(vlam[i]*cos(-bpa));
    downweight = exp(-(pow(uvmaj,2)/(2.0*pow(uvsigmamaj,2)))-
		     (pow(uvmin,2)/(2.0*pow(uvsigmamin,2))));
    uvdist = sqrt((ulam[i]*ulam[i])+(vlam[i]*vlam[i]));
    //printf("%8.3le %8.3le %2i %2i\n",uvdist,parameters->zeroat,
    //                parameters->dozero,(uvdist>parameters->zeroat));
    if ((parameters->dozero)&&(uvdist>parameters->zeroat)){
      downweight = 0;
      //printf("%i %lf %lf\n",parameters->dozero,uvdist,parameters->zeroat);
    }
    vis_rearrange[i][0] *= downweight;
    vis_rearrange[i][1] *= downweight;
    fprintf(fd, "%15.8e, %15.8e, %15.8e\n", ulam[i], vlam[i], downweight);
  }

  fclose(fd);

  /* FILE *fd = fopen("uv.txt", "w"); */

  /* printf("nelements=%d\n", nelements); */
  /* for (i = 0; i < nelements; i++)  */
  /*   fprintf(fd, "%15.8e, %15.8e, %15.8e\n", ulam[i], vlam[i], ); */
  
  /* fclose(fd); */



  // compute other quantities we may want
  for (i=0;i<nelements;i++){
    visamp[i] = sqrt((vis_rearrange[i][0]*vis_rearrange[i][0])+
		     (vis_rearrange[i][1]*vis_rearrange[i][1]));
    visphase[i] = (180.0/M_PI)*atan2(vis_rearrange[i][1],vis_rearrange[i][0]);
    //if (backdata[i]>2.4){printf("%ld %lf\n",i,backdata[i]);}
  }

  // put uv data back into expected order (peak in corner)
  for (i=0;i<nelements;i++){
    row = (long int)(i/ncolumns);
    col = (long int)(i%ncolumns);
    row -= (nrows/2);
    col -= (ncolumns/2);
    if (row<0) {row += nrows;}
    if (col<0) {col += ncolumns;}
    index = row*ncolumns+col;
    vis_back[index][0] = vis_rearrange[i][0];    
    vis_back[index][1] = vis_rearrange[i][1];    
  }

  // go back to the image domain
  inv_plan = fftw_plan_dft_2d(nrows,ncolumns,vis_back,back,
			      FFTW_BACKWARD,FFTW_ESTIMATE);
  fftw_execute(inv_plan);



  // get the right data to return
  for (i=0;i<nelements;i++){
    if (parameters->returntype==(RETURN_SCATTERED_IMAGE)){
      backdata[i] = back[i][0]/nelements;
    }
    if (parameters->returntype==(RETURN_SCATTERED_UV_AMP)){
      backdata[i] = visamp[i];
    }
    if (parameters->returntype==(RETURN_SCATTERED_UV_PHASE)){
      backdata[i] = visphase[i];
    }
  }
  
  for (i=0;i<nelements;i++){
    row = (long int)(i/ncolumns);
    col = (long int)(i%ncolumns);
    row += (nrows/2);
    col += (ncolumns/2);
    if (row>=nrows) {row -= nrows;}
    if (col>=ncolumns) {col -= ncolumns;}
    index = row*ncolumns+col;
    //if ((i%512)==0) {printf("%li %li %li %li\n",i,row,col,index);}
    if (parameters->nomove) {
      backdata_rearrange[i] = backdata[i];
    }
    else {
      if (parameters->returntype==(RETURN_SCATTERED_IMAGE)){
	backdata_rearrange[index] = backdata[i];
      }
      else {
	backdata_rearrange[i] = backdata[i];
      }
    }
  } 


  // write it
  fits_write_pix(fptr_out,TDOUBLE,fpixel,nelements,backdata_rearrange,&status);

  // clean up
  fftw_free(in);
  fftw_free(vis);
  fftw_free(vis_rearrange);
  fftw_free(vis_back);
  fftw_free(back);
  fftw_destroy_plan(plan);
  fftw_destroy_plan(inv_plan);
  free(data);
  free(backdata);
  free(ulam);
  free(vlam);
  free(visamp);
  free(visphase);
  fits_close_file(fptr,&status);
  fits_close_file(fptr_out,&status);

  // report any FITS errors
  if (status) {fits_report_error(stderr,status);}
  return(status);
}


int get_cmdline_options(int argc, char *argv[], 
			struct parameterstruc *parameters){
  int i, returnval=0;
  int inputspecified=0,outputspecified=0;

  // Read parameters
  for (i=0;i<argc;i++){
    if (strcmp(argv[i],"-i")==0){
      sscanf(argv[i+1],"%s",parameters->inputfilename);
      inputspecified++;
    }
    if (strcmp(argv[i],"-o")==0){
      sscanf(argv[i+1],"%s",parameters->outputfilename);
      outputspecified++;
    }
    if (strcmp(argv[i],"-bmaj")==0){
      sscanf(argv[i+1],"%lf",&parameters->bmaj);
    }
    if (strcmp(argv[i],"-bmin")==0){
      sscanf(argv[i+1],"%lf",&parameters->bmin);
    }
    if (strcmp(argv[i],"-bpa")==0){
      sscanf(argv[i+1],"%lf",&parameters->bpa);
    }
    if (strcmp(argv[i],"-image")==0){
      parameters->returntype=(RETURN_SCATTERED_IMAGE);
    }
    if (strcmp(argv[i],"-visamp")==0){
      parameters->returntype=(RETURN_SCATTERED_UV_AMP);
    }
    if (strcmp(argv[i],"-visphase")==0){
      parameters->returntype=(RETURN_SCATTERED_UV_PHASE);
    }
    if (strcmp(argv[i],"-nomove")==0){
      parameters->nomove=1;
    }
    if (strcmp(argv[i],"-zeroat")==0){
      parameters->dozero=1;
      sscanf(argv[i+1],"%lf",&parameters->zeroat);
    }
  }
  if ((!inputspecified)||(!outputspecified)){
    returnval = 1;
    printf("Input and output files must be specified\n\n");
  }
  return returnval;
}

void printusage(){
  printf("Usage: scatter <options>\n");
  printf("Input/Output (mandatory)\n");
  printf("       -i: input file\n");
  printf("       -o: output file\n");
  printf("Optional parameters\n");
  printf(" -image/-visamp/-visphase: return type [default: image]\n");
  printf("    -bmaj: major axis in microarcseconds [default: %4.1lf]\n",
	 DEFAULT_BMAJ);
  printf("    -bmin: minor axis in microarcseconds [default: %4.1lf]\n",
	 DEFAULT_BMIN);
  printf("     -bpa: major axis position angle in degrees [default: %4.1lf]\n",
	 DEFAULT_BPA);
  printf("  -nomove: don't move image to corner first [default: move image]\n");
  printf("  -zeroat: blank visibilities beyond this radius in Mlambda " \
	 "[default: don't blank]\n");
  printf("Unrecognized options are ignored\n");
  return;
}

