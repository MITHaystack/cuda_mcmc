# 
# To build and install the extension file mcmc_interf.so use the commands:
#
# $ make purge; python2 setup_mcmc.py build_ext --inplace;  make clean
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

bdir = os.getcwd()
os.system('make')

ext_mods = Extension("mcmc_interf",
    sources = ["mcmc_interf.pyx"],\
    library_dirs = [bdir, '/usr/local/cuda/lib64'],
    libraries = ['mcmcuda', 'cudart', 'm', 'stdc++'])

setup(
  name = 'mcmc_interf',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [ext_mods]
)
