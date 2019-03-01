#
# Compile and build mcmc_interf.so Python extension module
# in the srs/ directory. 
#

TASKDIR = $(PWD)
MODELDIR = $(TASKDIR)
export MODELDIR       # The current dir must have model.cuh to include

.PHONY:	all clean purge 

all:
	cd $(CUDA_MCMC)/src && $(MAKE) purge \
			&& python2 setup_mcmc.py build_ext --inplace \
			&& $(MAKE) clean && $(MAKE) install \
		&& cd $(TASKDIR)
	@echo "  "
	@echo MODEL is included from $(MODELDIR)/model.cuh
	@echo "  "

clean:
	cd $(CUDA_MCMC)/src && $(MAKE) clean && cd $(TASKDIR)

purge:
	cd $(CUDA_MCMC)/src && $(MAKE) purge && cd $(TASKDIR)

