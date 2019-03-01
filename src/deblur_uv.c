/*
 * deblur_uv.c
 *
 * Deconvolve an elliptical Gaussian in the UV-plane 
 * for MAPS-simulated data of Sgr A*.
 * The input model for MAPS simulation has been convolved with this Gaussian.
 * 
 * Author: R.-S. Lu, 2014/Feb/24
 *
 * Build:
 *
 * gcc -o deblur_uv deblur_uv.c -lm -lnsl -lcfitsio -lfftw3
 *
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAXLINES 500000//max num of vis
#define MAXVIS_PER_BASELINE 20000
#define MAJ 0.022 //mas
#define MIN 0.011 //mas
#define PA  78.0 //PA
#define DEBUG 0

double factor=1.0/(180.0/3.14159265*3600000.0);//convert mas to radians
double cont=57.295779579;//convert degree to radians 
double scale=-3.55971;//-pi^2/(4*ln2)

/* data struct */
struct visstruct{
	int year, day, hr, min, ant1, ant2;
        int flag;
	double second, u, v, w, amp, phase, sigma, weight;
};

/*para struct*/
struct parastruct{
	char infilename[300];//maps file
	char outfilename[300];//frame and total flux file
};

/*vis struct*/
struct mod_vis_struct{
        double amp;
        double phase;
};

/*get cmdline_options*/
void printusage(char *argv[]);
int get_cmdline_opts(int argc, char *argv[], struct parastruct *parameters){
	int i;
	int returnvalue=0;
	for(i=0;i<argc;i++){
		if(strcmp(argv[i], "-h")==0){
			printusage(argv);
			returnvalue=1;
		}
		if(strcmp(argv[i], "-i")==0){
			sscanf(argv[i+1],"%s",parameters->infilename);
		}
		if (strcmp(argv[i],"-o")==0){
  			sscanf(argv[i+1],"%s",parameters->outfilename);
		}
	}
	if (returnvalue==0){
		if (strcmp(parameters->infilename, "\0")==0){
       			printf("Must specify input file\n");
			printf("Use -h option to see usage statement\n");
			returnvalue = 1;
		}
		if (strcmp(parameters->outfilename,"\0")==0){
       			printf("Must specify output file\n");
			printf("Use -h option to see usage statement\n");
			returnvalue = 1;
		}
	}
	return(returnvalue);
}
void printusage(char *argv[]){
      	printf("Usage: %s [options]\n",argv[0]);
	printf("     -h: print this help screen\n");
	printf("     -i: specify input file from maps simulation\n");
	printf("     -o: give output file name\n");
	return;
}

/*subroutine to read in the maps simulation data*/
long read_vis(FILE *infile, struct visstruct *data){
	long vis_i=0;
	int ijunk;
	double junk;
	char cjunk;
	char line[300],*fgets();


	while ((fgets(line,300,infile))!=NULL){
	  /* printf("strchr(line,':') = %p\n", strchr(line,':')); */
	  /* printf("line = %s\n", line); */
		if (strchr(line,':')!=NULL){
			if(sscanf(line,"%d:%d:%d:%d:%lf %lf %lf %lf %d-%d %d %c %lf, %lf) %lf %lf",
						&data[vis_i].year,
						&data[vis_i].day,
						&data[vis_i].hr,
						&data[vis_i].min,
						&data[vis_i].second,
						&data[vis_i].u,
						&data[vis_i].v,
						&data[vis_i].w,
						&data[vis_i].ant1,
						&data[vis_i].ant2,
						&ijunk,
						&cjunk,
						&data[vis_i].amp,
						&data[vis_i].phase,
						&data[vis_i].weight,
						&data[vis_i].sigma)>15) {
				
				data[vis_i].flag=0;
				if(DEBUG){
					printf("%li %d %d %d  %6.2lf %6.2lf flag: %d\n",vis_i, data[vis_i].day, data[vis_i].hr, data[vis_i].min,data[vis_i].amp, data[vis_i].sigma,data[vis_i].flag);
				}
				vis_i++;
				if (vis_i==(MAXLINES)){
					printf("Too many visibilities.  Stopping at %li\n",vis_i);
					break;
				}
			}
		}
	}
	if(DEBUG){
		printf("Read in %li visibilities\n",vis_i);
	}
      	return vis_i;
}

/*subroutine to calculate model visibilities given u and v*/
struct mod_vis_struct  model_cal(double u, double v)
{
int i;
double x=0,y=0;//x=r*cos(theta),y=sin(theta)
double rou=0, sprime=0; 
double amp=0, phs=0;

double eta=atan2(u,v)*180.0/M_PI;
double ratio = MIN/MAJ; 	   
	 
struct mod_vis_struct mod_vis;

rou = sqrt(u*u + v*v);//when u v in units of lambda
sprime=rou*sqrt(cos((eta-PA)/cont)*cos((eta-PA)/cont)+ratio*ratio*sin((eta-PA)/cont)*sin((eta-PA)/cont));

	   amp=exp(scale*MAJ*MAJ*sprime*sprime*factor*factor);
	   phs=-2.0*M_PI*(u*x+v*y);

	   mod_vis.amp = amp;
           mod_vis.phase =phs*180.0/M_PI;
	   return(mod_vis);
}	

int main(int argc, char *argv[]){
	struct visstruct *data;
	struct parastruct *parameters;
	struct mod_vis_struct *mod_vis;
	double tmp;
	FILE *infile, *outfile;
	data = calloc(MAXLINES,sizeof(struct visstruct));
	parameters = calloc((MAXLINES),sizeof(struct parastruct));
	mod_vis = calloc((MAXLINES),sizeof(struct mod_vis_struct));
	long numlines = 0, i=0, j=0;
	if (get_cmdline_opts(argc,argv,parameters)>0){
	    	return(0);
	}
	infile = fopen(parameters->infilename,"r");
	outfile = fopen(parameters->outfilename,"w+");
	numlines = read_vis(infile,data);
	for(i=0;i<numlines;i++){
		//calculate the amplitudes of the Gaussian		
		mod_vis[i]=model_cal(data[i].u*1000.0,data[i].v*1000.0);
		fprintf(outfile,"%4d:%03d:%02d:%02d:%05.2lf %13.2lf %13.2lf %13.2lf %02d-%02d    00    (   %10.6lf, %15.6lf) %9.2lf %7.5lf\n",
						data[i].year,
						data[i].day,
						data[i].hr,
						data[i].min,
						data[i].second,
						data[i].u,
						data[i].v,
						data[i].w,
						data[i].ant1,
						data[i].ant2,
						data[i].amp/mod_vis[i].amp,
						data[i].phase,
						data[i].weight,
						data[i].sigma/mod_vis[i].amp);
	
	}
	/*clean up*/	
	free(data);
	free(parameters);
	fclose(infile);
	fclose(outfile);

}


  
