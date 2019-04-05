# cuda_mcmc

The cuda_mcmc software is designed to greatly accelerate solving various optimization problems using the NVIDIA graphics processing units (GPU) with the CUDA computing platform. We use a version of the Markov chain Monte-Carlo (MCMC) algorithm based on the Metropolis-Hastings sampler with replica exchange (MCMC-RE). The detailed description of the algorithm with its mapping on the CUDA GPU architecture including the user instructions to its installation and writing custom optimization/equation solving applications is provided in the `doc/` directory. Several examples of the code usage are given in the `examples/` directory. 

To start using CUDA MCMC software, first enter its containing directory (usually `~/cuda_mcmc`) and issue the
command

    $ make

It will create the directories `~/lib64/python`, and `~/bin`,  and setup the
environment for them by adding to the end of the `~/.bashrc` file the lines

    export PATH=$PATH:~/bin
    export PYTHONPATH=$PYTHONPATH:~/lib64/python

After `make` finishes, update the environment variables PATH and PYTHONPATH issuing the command

    $ source ~/.bashrc

Or simply open a new console window. This will automatically run ~/.bashrc
script and set the environment variables.


