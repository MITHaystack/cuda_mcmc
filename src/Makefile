#
# Makefile to build libmcmcuda.a library with the CUDA portion of mcmcfit:
# mcmcjob.o and gpu_mcmc.o. The library is used by setup_mcmc.py to build
# the Python extension module mcmc_interf.so. 
# Some utility programs are also built.
# The command
# $ make install puts the executables to ~/bin/ and the python modules
# to ~/lib64/python. 
#
# This src/Makefile is called the following way:
#
#    Application_dir/Makefile -> src/setup_mcmc.py -> src/Makefile
#
# The application directory must contain the file model.cuh. Its location
# is exported as the environment variable MODELDIR and is used ti #include
# model.cuh as a header file in gpu_mcmc.cu at its compilation.
#
# To build the extension module mcmc_interf.so use the command(s):
#
# make purge; python2 setup_mcmc.py build_ext --inplace; make clean
#
# To find your GPU architecture and set -arch correctly, run
# $ list_gpus
# If its output includes, say, "Compute Capability 3.0", then set -arch=sm_30 
#

CC=gcc
CXX=nvcc
CXXFLAGS=  -arch=sm_30 -g -Xcompiler -fPIC -Xcompiler -fno-exceptions
OBJECTS= mcmcjob.o gpu_mcmc.o
BASEDIR = $(PWD)

.PHONY: all clean purge install

all:    libmcmcuda.a

libmcmcuda.a:   $(OBJECTS) 
	   $(CXX) -Xcompiler -Wl, -Xcompiler -no-undefined -lib $(LIBS) \
				$(OBJECTS) -o libmcmcuda.a 

mcmcjob.o: mcmcjob.cu
	$(CXX) $(CXXFLAGS) -c mcmcjob.cu -o mcmcjob.o 

gpu_mcmc.o: gpu_mcmc.cu
	$(CXX) $(CXXFLAGS) -I$(MODELDIR) -c gpu_mcmc.cu -o gpu_mcmc.o

scatter_fits: scatter_fits.c
	mkdir -p ~/bin                 # Create the directory if it does not exist
	$(CC) -o scatter_fits scatter_fits.c -lm -lnsl -lcfitsio -lfftw3    && \
	cp scatter_fits ~/bin

deblur_uv: deblur_uv.c
	mkdir -p ~/bin                 # Create the directory if it does not exist
	$(CC) -o deblur_uv deblur_uv.c -lm -lnsl -lcfitsio -lfftw3    && \
	cp deblur_uv ~/bin

list_gpus: list_gpus.cu
	mkdir -p ~/bin                 # Create the directory if it does not exist
	$(CXX) list_gpus.cu -o list_gpus    && \
	cp list_gpus ~/bin

install:
	mkdir -p ~/lib64/python        # Create the directory if it does not exist
	cp mcmc_interf.so ~/lib64/python
	cp imgpu.py  ~/lib64/python
	cp obsdatafit.py ~/lib64/python

clean: 
	rm -fR lib*.a *.o mcmc_interf.c build/ 

purge:  clean
	rm -fR mcmc_interf.so *.pyc scatter_fits deblur_uv list_gpus



