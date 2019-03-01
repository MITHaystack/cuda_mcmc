#
# Makefile to create ~/lib64/python, -p ~/bin, and setup the environment
# for them 
#

CXX=gcc
BASEDIR = $(PWD)

ifneq ($(findstring $(HOME)/bin, $(PATH)), $(HOME)/bin)
	mdfy=1
endif
ifneq ($(findstring $(HOME)/lib64/python, $(PYTHONPATH)), $(HOME)/lib64/python)
	mdfy=1
endif
ifndef CUDA_MCMC
	mdfy=1
endif

.PHONY: all

all:
	mkdir -p ~/lib64/python      # Create the directory if it does not exist
	mkdir -p ~/bin               # Create the directory if it does not exist

ifneq ($(findstring $(HOME)/bin, $(PATH)), $(HOME)/bin)
	@echo 'export PATH=$$PATH:~/bin' >> ~/.bashrc
	@echo Line 'export PATH=$$PATH:~/bin' added to your '~/.bashrc' file.
endif

ifneq ($(findstring $(HOME)/lib64/python, $(PYTHONPATH)), $(HOME)/lib64/python)
	@echo 'export PYTHONPATH=$$PYTHONPATH:~/lib64/python' >> ~/.bashrc
	@echo Line 'export PYTHONPATH=$$PYTHONPATH:~/lib64/python' \
			added to your '~/.bashrc' file.
endif

ifndef CUDA_MCMC
	@echo 'export CUDA_MCMC=$(PWD)' >> ~/.bashrc
	@echo Line 'export CUDA_MCMC=$(PWD)' \
			added to your '~/.bashrc' file.
endif

ifeq ($(mdfy), 1)
	@echo
	@echo To update the environment, please execute the command
	@echo 'source ~/.bashrc'
	@echo
endif
