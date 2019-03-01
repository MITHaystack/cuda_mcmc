Application: model_obsfit.py

Fitting 9- and 13-parameter models to simulated observation data

Build:
$ make 


Usage:
$ ipython2 --pylab
  %run model_obsfit.py [9|13] uvdata.txt ng=100 xy=170. \
                                        [nburn=400] [niter=1200]  \
                                        [nbeta=32] [nseq=8]
or (This still needs some more work!)

$ python2 model_obsfit.py [9|13] uvdata.txt ng=100 xy=170. \
                                        [nburn=400] [niter=1200]  \
                                        [nbeta=32] [nseq=8]

Examples:
  %run model_obsfit.py 9 000008_1_1_000snd_uvdata.txt  ng=100 xy=170.
Or, from Linux command line:
$ python model_obsfit.py 9 000508_1_1_000snd_uvdata.txt  ng=100 xy=170.




