#
# model_obsfit.py
#
# Fitting 9- and 13-parameter models to simulated observation data
# Usage:
# %run model_obsfit.py [9|13] uvdata.txt ng=128 xy=160. \
#                                        [nburn=400] [niter=1200]  \
#                                        [nbeta=32] [nseq=8] 
# Example:
# %run model_obsfit.py 9 000_001_002_uvdata.txt ng=128 xy=160.
#
#
#
import imgpu
import numpy as np
from pylab import *
import time
import os, sys, re
import obsdatafit as ob
from obsdatafit import  model_vis2bri_simple, xringaus, xringaus2, cpix2cpid
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix' # Allow math symbols b/w $ and $
mpl.rcParams['text.usetex'] = False     # True


nburn = 400  
niter = 1000
nbeta = 32
nseq = 16


#
# The parameters below determine two different things:
# - dimensions 'UVspan' of the UV domain where the model is calculated,
#   grid size 'ngrid', and corresponding XY domain dimensions 'XYspan';
# - width and height of the central cut of the brightness image 'XYwidth_uas',
#   and the same of the visibility amplitude or phase image 'UVwidth_glam'.
#
# Function model_vis2bri_simple() inputs UVspan and ngrid, and outputs XYspan,
# calculated as
#    XYspan_uas = 3600.*degrees(ngrid/UVspan_glam)*1e-3 # in microarcseconds
# Both finer grid (larger ngrid) and smaller UVspan infer larger XYspan, and
# the brightness image of the central object becomes too small. 
# Therefore, only central part of the brightness image should be plotted,
# and the width and height of it is specified by 'XYwidth_uas'.
# If plotting visibility, the same is made by 'XYwidth_uas'.
#
# Default values:
#
ngrid = 100
XYspan_uas = 170.
delX_uas = -0.7992                  # Micro arcseconds
delX_deg = -4.4437147130151E-10     # Degrees
XYwidth_uas = -170.
UVwidth_glam = 20.
# UVspan_glam is calculated


showPlot = True
png = False
svg = False
eps = False

argc = len(sys.argv)
argv = sys.argv

print 'argv = ', argv

#
# In parsing the command line options of the form <variable>=<value>:
# lv - "left value", the variable name;
# rv - "right value", the value to be assigned to the variable.
#
if argc >= 3:
    modnum = int(argv[1])
    uvdata_file = argv[2]
    print 'modnum = ', modnum, ', uvdata_file = ', uvdata_file
    for i in xrange(3,argc):
        opt = argv[i].split('=')
        if len(opt) == 2:
            lv = opt[0]; rv = opt[1]
            if   lv == 'ng':
                exec('ngrid = int(' + rv + ')')
                print 'ngrid = ', ngrid
            elif lv == 'xy':
                exec('XYspan_uas = float(' + rv + ')')         
                print 'XYspan_uas = ', XYspan_uas
            elif lv == 'dxdeg': exec('delX_deg = float(' + rv + ')')         
            elif lv == 'dxuas': exec('delX_uas = float(' + rv + ')')         
            elif lv == 'xw': exec('XYwidth_uas = float(' + rv + ')')         
            elif lv == 'uw': exec('UVwidth_glam = float(' + rv + ')')         
            elif lv == 'nburn': exec('nburn = int(' + rv + ')') 
            elif lv == 'niter': exec('niter = int(' + rv + ')') 
            elif lv == 'nbeta': exec('nbeta = int(' + rv + ')') 
            elif lv == 'nseq':  exec('nseq =  int(' + rv + ')') 
            else:
                print '\nInvalid variable "%s". Exiting.\n' % argv[i]
                sys.exit(0)
        elif opt == '-noplot' or opt == 'noplot':
            showPlot = False
        elif opt == '-png' or opt == 'png':
            png = True
        elif opt == '-svg' or opt == 'svg':
            svg = True
        elif opt == '-eps' or opt == 'eps':
            eps = True

elif argc < 3 or argc > 7:
      print "\nInvalid command line\n"
      print "Usage: model_obsfit.py {9|13} uvdata_file [nburn=400] " \
            "[niter=1200] [nbeta=32] [nseq=8]\n" 
      sys.exit(0);


#
# uvfile_esc is uvdata_file with escape characters before the underscores
# Otherwise LaTeX interpret them as subscript prefixes
#
#uvfile_esc = uvdata_file.replace('_','\_')
			

if ngrid < 0:
    ngrid = -ngrid
    print '\nGrid size must be set. Example: ng=100. Exiting.\n'
    sys.exit(0)
    
if (XYspan_uas < 0) and (delX_uas < 0) and (delX_deg < 0):
    print '\nBrightness domain size must be set, either as grid increment ' \
          'or as X and Y dimensions.'
    print 'Examples:'
    print '      XYspan_uas=170. (Size in X and Y in micro arcsrconds)'
    print 'or    dxdeg=4.4437147130151E-10  ' \
          '(grid increment in degrees, as in FITS header)'
    print 'or    dxuas=0.7992       (grid increment in mucro arcseconds)'
    print 'Exiting.'
    sys.exit(0)

## if XYspan_uas < 0:
##     print '\nXYspan_uas must be set. Example: XYspan_uas=160. Exiting.\n'
##     sys.exit(0)

if delX_deg > 0: XYspan_uas = ngrid*3600*1e6*delX_deg
if delX_uas > 0: XYspan_uas = ngrid*delX_uas

if XYwidth_uas < 0: XYwidth_uas = XYspan_uas

UVspan_glam = 3600.*(180./pi)*(ngrid/XYspan_uas)*1e-3

#sys.exit(0)

if modnum == 9:
    #
    # For 9-parameter (xringaus) model:
    #
    ## Zsp: zero spacing in any units (Jy or such)
    ## Re: external  ring radius in microarcseconds 
    ## rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ## ecc: eccentricity of inner circle center from that of outer; in [-1..1]
    ## fade: [0..1], "noncontrast" of the ring, 1-uniform, 0-from zero to max.
    ## gax: Gaussian main axis, expressed in Re
    ## aq: axes quotient, aq = gsx/gsy. 0 < aq < 1
    ## gq: fraction of Gaussian visibility against ring visibility.
    ##     0-ring only, 1-Gaussian only
    ## th: angle of circular hole and brightness gradient orientation in radians
    #
    pn = ('Zsp', 'Re', 'rq', 'ecc', 'fade', 'gax', 'aq', 'gq', 'th')
    pnx = (r'$Z_{sp}$', r'$R_e$', r'$r_q$', r'$Ecc$', r'$Fade$', r'$g_{ax}$', \
           r'$a_q$', r'$g_q$', r'$\theta^\circ$')
    ncol = 3    # Number of columns in the histogtam subplots
    #
    #                Zsp, Re, rq, ecc, fade, gax, aq, gq, th
    #

    ## pdescr = array((1,   1,  1,  1,  1,  1,   1,   1,  2), dtype=int32)
    ## ptotal = array((2.4, 0., 0., 0., 0., 0.,  0.,  0., 0.), dtype=float32)
    # Uniform disk: Search for Zsp and Re
    # pdescr = array((1,   1,  0,  0,  0,  0,   0,   0,  0), dtype=int32)
    # ptotal = array((2.4, 0., 0., 0., 1., 0.,  0.,  0., 0.), dtype=float32)
    # NoGauss:
    pdescr = array((1,   1,  1,  1,  1,  1,   1,   1,  2), dtype=int32)
    #pdescr = array((1,   1,  1,  1,  1,  0,   0,   0,  2), dtype=int32)
    ptotal = array((2.4, 0., 0., 0., 1., 0.,  0.,  0., 0.), dtype=float32)
    bnd = array(((0.01, 5.0), (20., 60.), (0., 0.99), (0., 1.), (0., 1.), \
                 (0.1, 2.), (0.1, 1.5), (0., 1.), (-pi, pi)), dtype=float32)

elif modnum == 13:
    #
    # For 13-parameter (xringaus2) model:
    #
    ## Zsp: zero spacing in any units (Jy or such)
    ## Re: external  ring radius in microarcseconds 
    ## rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ## ecc: eccentricity of inner circle center from that of outer; in [-1..1]
    ## fade: [0..1], "noncontrast" of the ring, 1-uniform, 0-from zero to max.
    ## gr: Rg/Re, Rg - distance of Gaussian center from the external ring center
    ## gax: Gaussian main axis, expressed in Re
    ## aq: axes quotient, aq = gsx/gsy. 0 < aq < 1
    ## gq: fraction of Gaussian visibility against ring visibility.
    ##     0-ring only, 1-Gaussian only
    ## alpha: angle of the Gaussian center orientation in radians
    ## beta: angle of the Gaussian rotation in radians
    ## eta: angle of internal circular hole orientation in radians
    ## th: angle of slope orientation in radians
    #
    pn = ('Zsp', 'Re', 'rq', 'ecc', 'fade', 'gr', 'gax', 'aq', 'gq', 'alpha', \
          'beta', 'eta', 'th')
    pnx = (r'$Z_{sp}$', r'$R_e$', r'$r_q$', r'$Ecc$', r'$Fade$', r'$g_{r}$', \
           r'$g_{ax}$', r'$a_q$', r'$g_q$', r'$\alpha^\circ$', \
           r'$\beta^\circ$', r'$\eta^\circ$', r'$\theta^\circ$')
    ncol = 4    # Number of columns in the histogtam subplots
    #
    #         Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th
    #
    pdescr = array((1,   1,   1,  1,  1,  1,  1,   1,  1,  2,  2,  2,  2),
                   dtype=int32)
    #pdescr = array((1,   1,   1,  1,  1,  0,  0,   0,  0,  0,  0,  2,  2),
    #               dtype=int32)
    ptotal = array((2.4, 31., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0), 
                   dtype=float32)
    # Uniform disk: Search for Zsp and Re
    ## pdescr = array((1,   1,  0,  0,  0,  0,   0,   0,  0,  0,  0,  0,  0),
    ##              dtype=int32)
    ## ptotal = array((2.4, 31., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0), 
    ##              dtype=float32)
    #pdescr = array((1,   1,  1,  1,  1,  1,   1,   1,  1,  2,  2,  2,  2),
    # NoGauss:
    #pdescr = array((1,   1,  1,  1,  1,  0,   0,   0,  0,  0,  0,  2,  2)
    #               dtype=int32)
    #ptotal = array((2.4, 31., 0.9, 0., 0., 0.6, 0.,  0., 0.65, 0., pi/2., \
    #                0., 0),
    ## ptotal = array((2.4, 31., 0., 0., 1., 0., 0.,  0., 0., 0., 0., 0., 0),
    ##                dtype=float32)
    #ptotal = array((2.4, 31., 0.9, 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0),
    #               dtype=float32)
    bnd = array(((0.01, 5.0), (20., 60.), (0., 0.99), (0., 1.), (0., 1.), \
                 (0., 1.), (0.1, 2.), (0.1, 1.), (0., 1.), \
             (-pi, pi), (0., pi/2.), (-pi, pi), (-pi, pi)), dtype=float32)
else:
    print "Wrong number of model parameters; you entered ", modnum,
    print " may be either 9 or 13"
    sys.exit(0)
    
pmint = copy(bnd[:,0]); pmaxt= copy(bnd[:,1])

imodel = {9:1, 13:2}[modnum]  # Model code, 1 or 2

nrndst = int32(nseq*nbeta)
seed = uint64(np.trunc(1e6*time.time()%(10*nrndst)))
print 'seed = ', seed


sgra = imgpu.Mcgpu_Sgra(uvfile=uvdata_file, pdescr=pdescr, ptotal=ptotal, \
                pmint=pmint, pmaxt=pmaxt, seed=seed, nseq=nseq, nbeta=nbeta, \
                nburn=nburn, niter=niter, betan=1e-4, imodel=imodel, \
                ecph=1.)
#
# Calculate closure phase errors from visibility SNR
#
# Read the lookup table containing phase errors in degrees as function
# of the visibility SNR, from 0. to 30., step 0.01
#
# In degrees, from snr=0 to snr=30:
#
pherr = loadtxt('pherr_vs_snr.txt')   
snr = sgra.amp/sgra.eamp    # Visibility SNR
#
# Calculate closure phase SNRc using Eq. 31 from the paper by
# Alan Rogers, Sheperd Doeleman, "Fringe detection methods for
# very long baseline arrays", 1995, Astronomical Journal, V. 109, N. 3
#
i1 = sgra.cpix[:,0]
i2 = sgra.cpix[:,1]
i3 = sgra.cpix[:,2]
snrc = snr[i1]*snr[i2]*snr[i3]/           \
       sqrt((snr[i1]*snr[i2])**2 + (snr[i1]*snr[i3])**2 + \
            (snr[i2]*snr[i3])**2)
isnrc_le30 = where(snrc <= 30.)[0]
isnrc_gt30 = where(snrc > 30.)[0]

sgra.ecph[isnrc_gt30] = 1./snrc[isnrc_gt30]
sgra.ecph[isnrc_le30] = np.float32(radians(pherr[100*np.int_(snrc[isnrc_le30])]))
sgra.std2r[sgra.nvis:] = 1./(sgra.ecph**2)

#sys.exit(0)

sgra.burnin_and_search()

chi2 = sgra.chi2_2d.flatten()
pout = sgra.pout_4d[:,0,:,:].reshape((sgra.nprm,sgra.nseq*sgra.niter))

im = chi2.argmin()   # Indices of the chi^2 minimums 

nseqitr = int32(sgra.nseq*sgra.niter)

pout1 = zeros((sgra.nprm,nseqitr), dtype=float32)
for i in xrange(sgra.nprm):
    pout1[i,:] = sgra.pout_4d[i,0,:,:].flatten()

#
# Parse fits file name
#
cnoise = ''

f = uvdata_file.split('_')

fbn = re.findall('00[0-9]_00[0-9]_0[0-9]{2}', uvdata_file)

print 'fbn = re.findall(...)'

if fbn == []: # Like model_9prm_sn_uvdata.txt
    if   uvdata_file.find('_snda') <> -1: cnoise = '_snda'
    elif uvdata_file.find('_snd') <> -1: cnoise = '_snd'
    elif uvdata_file.find('_sn') <> -1: cnoise = '_sn'
    elif uvdata_file.find('_n') <> -1: cnoise = '_n'
    cnam = '_'.join(uvdata_file.split('_')[:-1])   # Cut away 'uvdata.txt'
    czsp = ''
    crot = ''

print 'if fbn == []'

if fbn <> []: # Like 000_001_002sn_uvdata.txt
    f = [fbn[0], '', '', '000']      # temporarily 000 degrees of rotation
    if uvdata_file[11] == 'n':
        cnoise = '_n'
    elif uvdata_file[11:13] == 'sn':
        cnoise = '_sn'
    elif uvdata_file[11:14] == 'snd':
        cnoise = '_snd'
    elif uvdata_file[11:15] == 'snda':
        cnoise = '_snda'

    cnam = f[0]
    czsp = '1'
    crad = '1'
    crot = '000'

elif len(f) == 5:           # Like 000208_1_1_270snd_uvdata.txt

    if len(f[3]) == 4 and f[3][3] == 'n':
        cnoise = '_n'
        f[3] = f[3][:-1]
    else:
        cnoise = ''
        
    if len(f[3]) == 5 and f[3][3:5] == 'sn':
        cnoise = '_sn'
        f[3] = f[3][:-2]
    elif len(f[3]) == 6 and f[3][3:6] == 'snd':
        cnoise = '_snd'
        f[3] = f[3][:-3]
    else:
        cnoise = ''

    cnam = f[0]
    czsp = f[1]
    crad = f[2]
    crot = f[3]

elif len(f) == 2:           # Like 08068snd_uvdata.txt
    cnam = f[0]                # Like 08068snd
    if cnam.find('snd', -3) <> -1:
        cnoise = '_snd'
        cnam = cnam[:-3]
    elif cnam.find('sn', -3) <> -1:
        cnoise = '_sn'
        cnam = cnam[:-2]
    elif cnam.find('n', -3) <> -1:
        cnoise = '_n'
        cnam = cnam[:-1]
    else:
        cnoise = ''

    cnam = f[0]
    czsp = '1'
    crad = '1'
    crot = '000'
    
else:
    czsp = ''
    crot = ''
    crad = ''
    f = f[:-1]
    f.extend(['1', '000'])
    cnoise = ''

#===========================  H I S T O G R A M S ==============================
    
#
# Plot histograms
#

if sgra.nprm%ncol:
    nrow = sgra.nprm/ncol + 1
else:
    nrow = sgra.nprm/ncol 

if nrow >= 3:
    figure(figsize=(19.5,13.5));
elif nrow == 2:
    figure(figsize=(19.5,13.5));
elif nrow == 1:
    ncol = sgra.nprm
    if sgra.nprm == 4:
        figure(figsize=(19.5,7));
    if sgra.nprm == 3:
        figure(figsize=(19.5,7));
    elif sgra.nprm == 2:
        figure(figsize=(14,7));
    elif sgra.nprm == 1:
        figure(figsize=(9,7));

for i in xrange(sgra.nprm):
    j = sgra.ivar[i]
    ax = subplot(nrow, ncol, i+1)
    if pdescr[i] == 2:
        pp = degrees(pout1[i,:])                  # Express angles in degrees
    else:
        pp = np.copy(pout1[i,:])

    title(pnx[j], fontsize=20)
    pat = hist(pp, 50, color='b', label=pnx[j]);
    mu = mean(pp)
    sig = std(pp)
    mumsig = mu - sig
    mupsig = mu + sig
    bf = pp[im]
    ltop = 1.2*pat[0].max()
    lsig = 0.5*ltop
    #bf = pout1[i,im]
    
    plot([mu, mu], [0, ltop], 'k', lw=3, \
              label='mean$=%6.2f$' % mu)
    plot([mumsig, mumsig], [0, ltop], 'g--', lw=2, \
              label='std$=\pm%6.3f$' % sig)
    plot([mupsig, mupsig], [0, ltop], 'g--', lw=2)
    plot([bf, bf], [0, ltop], 'r', lw=3, \
              label='bestfit$=%6.2f$' % bf)
    xticks(fontsize=10)
    yticks(fontsize=10)
    grid(1);
    text(mupsig+0.3*sig, lsig, r'$\sigma=%6.3f$' % sig, fontsize=16)
    legend(loc='best', labelspacing=0.1, prop={'size':10})

if modnum == 9:
    figtext(0.5, 0.95, 'Markov Monte-Carlo (MCMC) Chains for the ' \
            'Model Parameter Estimates', ha='center', fontsize=22)
    figtext(0.08, 0.03, 'File %s' % uvdata_file, fontsize=16)
    figtext(0.4, 0.03, '%d Markov chains x %d iterations = %d  data points' % \
            (nseq, niter, nseq*niter), fontsize=16)
elif modnum == 13:
    figtext(0.5, 0.915, 'Markov Monte-Carlo (MCMC) Chains for the ' \
            'Model Parameter Estimates', ha='center', fontsize=22)
    figtext(0.08, 0.03, 'File %s' % uvdata_file, fontsize=16)
    figtext(0.4, 0.03, '%d Markov chains x %d iterations = %d  data points' % \
            (nseq, niter, nseq*niter), fontsize=16)

if png: savefig("hist_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.png"  %  \
               (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if svg: savefig("hist_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.svg"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if eps: savefig("hist_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.eps"  %  \
        (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))

show() if showPlot else close()

#
# Brightness domain limits for plotting
#
xyext = XYwidth_uas/2.
XYdel = XYspan_uas/ngrid
if ngrid%2 == 0:   # For EVEN ngrid
    # Extended by 1 XYdel in X and Y for beauty
    Xlim = [ xyext,       -xyext-XYdel]
    Ylim = [-xyext-XYdel,  xyext]
    # Correct:
    # Xlim = [ xyext-XYdel, -xyext]
    # Ylim = [-xyext,        xyext-XYdel]
else:               # For ODD  ngrid
    XYlim = [xyext, -xyext, -xyext, xyext]



#
# Visibility domain limits for plotting
#
uvext = UVwidth_glam/2.
UVdel = UVspan_glam/ngrid
if ngrid%2 == 0:   # For EVEN ngrid
    # Extended by 1 UVdel in U and V for beauty
    Ulim = [ uvext,      -uvext-UVdel]
    Vlim = [-uvext-UVdel, uvext]
    # Correct:
    # Ulim = [ uvext-UVdel, -uvext]
    # Vlim = [-uvext,        uvext-UVdel]
else:               # For ODD  ngrid
    UVlim = [uvext, -uvext, -uvext, uvext]


if ngrid%2 == 0:   # For EVEN ngrid
    XYext = [xyext-XYdel, -xyext, -xyext, xyext-XYdel]
else:               # For ODD  ngrid
    XYext = [xyext, -xyext, -xyext, xyext]


## XYspan_urad = radians(XYspan_uas/3600.)
## UVspan_glam = 1e-3*float(ngrid)/XYspan_urad  # Glam

    

#======================  H I S T M E A N  I M A G E  ==========================

#
# Plot the model visibility amplitude for the histogram MEANs of parameters
#



mus = zeros(sgra.nprm)
for i in xrange(sgra.nprm):
    mus[i] = mean(pout1[i,:])

prm_hm = copy(sgra.ptotal)
for i in xrange(sgra.nptot):
    if sgra.invar[i] >= 0: prm_hm[i] = mus[sgra.invar[i]]


if modnum == 9:
    Zsp, Re, rq, ecc, fade, gax, aq, gq, th = prm_hm

    Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext1, XYext1 =  \
        model_vis2bri_simple(xringaus, prm_hm, UVspan_glam, ngrid)
elif modnum == 13:
    Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th = prm_hm

    Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext1, XYext1 =  \
        model_vis2bri_simple(xringaus2, prm_hm, UVspan_glam, ngrid)

    etad = degrees(eta)
    alpd = degrees(alpha)
    betd = degrees(beta)

thd = degrees(th)

absVi = abs(Vi)
absBr = abs(Br)
phVi = arctan2(Vi.imag, Vi.real)



chi2hm, vicpm, pham, chi2m, datm = sgra.calcmodchi2(prm_hm)

chi2hm_r = chi2hm/(sgra.nvis+sgra.ncph-sgra.nprm-1) # Reduced chi^2

print 'Histogram Mean Model Brightnes th=%g, thd=%g' % (th, thd)

if modnum == 9:

    figure(figsize=(6,6))
    imshow(absBr*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
           interpolation='nearest'); grid(1)
    contourf(absBr*1e3, 100, origin='lower', cmap=cm.hot, extent=XYext)
    title('Histogram Mean Model Brightness', fontsize=18)
    colorbar(shrink=0.7)
    xlabel(r'RA ($\mu$as)', fontsize=16)
    ylabel(r'Dec ($\mu$as)', fontsize=16)
    grid(1)
    figtext(0.02, 0.95, r'$Z_{sp} = %4.2f$, $R_e = %4.1f$, $R_i/R_e = %4.2f$, $Ecc = %4.2f$' % (Zsp, Re, rq, ecc), fontsize=16)
    figtext(0.02, 0.89, r'$Fade =  %2.2f$, $g_{ax} = %4.2f$, $a_q = %4.2f$, $g_q = %4.2f$, $\theta = %4.0f^\circ$' % (fade, gax, aq, gq, thd), fontsize=16)
    figtext(0.4, 0.07, r'$\chi^2=%g,\; \rm{reduced:}\, %g$' % (chi2hm, \
            chi2hm_r), ha='center', fontsize=16)
    figtext(0.4,0.02, 'File: '+uvdata_file+'; ', ha='center', fontsize=16)
    figtext(0.80, 0.81, 'mJy/pixel')   #, fontsize=16)

elif modnum == 13:
    
    figure(figsize=(8,8))
    imshow(absBr*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
           interpolation='nearest'); grid(1)
    contourf(absBr*1e3, 100, origin='lower', cmap=cm.hot, extent=XYext) 
    title('Histogram Mean Model Brightness', fontsize=18)
    colorbar(shrink=0.7)
    xlabel(r'RA ($\mu$as)', fontsize=16)
    ylabel(r'Dec ($\mu$as)', fontsize=16)
    grid(1)
    figtext(0.02, 0.95, r'$Z_{sp} = %4.2f$, $R_e = %4.1f$, $r_{q} = %4.2f$, $Ecc = %4.2f$ $Fade =  %2.2f$, $g_{r} = %4.2f$' % (Zsp, Re, rq, ecc, fade, gr), fontsize=16)
    figtext(0.02, 0.89, r'$g_{ax} = %4.2f$, $a_q = %4.2f$, $g_q = %4.2f$, $\alpha = %4.0f^\circ$, $\beta = %4.0f^\circ$, $\eta = %4.0f^\circ$, $\theta = %4.0f^\circ$' % (gax, aq, gq, alpd, betd, etad, thd), fontsize=16)
    figtext(0.4, 0.07, r'$\chi^2=%g,\; \rm{reduced:}\, %g$' % (chi2hm, \
            chi2hm_r), ha='center', fontsize=16)
    figtext(0.4,0.02, 'File: '+uvdata_file+'; ', ha='center', fontsize=16)
    figtext(0.80, 0.81, 'mJy/pixel')   #, fontsize=16)

xticks([80, 60,40,20, 0, -20, -40, -60, -80])    
yticks([80, 60,40,20, 0, -20, -40, -60, -80])    
#show()
        
print 'Histogram mean params = '
for i in xrange(sgra.nptot): print '%g\t' % prm_hm[i],
print
print 'Histmean chi^2 = %g, reduced chi^2 = %g' % (chi2hm, chi2hm_r)


# ftxt = open("hm_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.txt"  %  \
#         (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter), 'w')
# if modnum == 9:
#     ftxt.write('# Zsp, Re, rq, ecc, fade, gax, aq, gq, thd, chi2hm, \
#                 chi2hm_r\n')
#     ftxt.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f ' \
#                '%10.4f %10.2f %10.4f\n' % \
#                (Zsp, Re, rq, ecc, fade, gax, aq, gq, thd, chi2hm, chi2hm_r))
# elif modnum == 13:
#     ftxt.write('# Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, ' \
#                'alpd, betd, etad, thd, chi2hm, chi2hm_r\n')
#     ftxt.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f ' \
#                '%10.4f %10.4f %10.4f %10.4f %10.4f %10.2f %10.4f\n' 
#                % (Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, \
#                   alpd, betd, etad, thd, chi2hm, chi2hm_r))
# else:
#     pass
    
# ftxt.write('\n')
# savetxt(ftxt, absBr*1e3, '%14.6e')
# ftxt.close()

if png: savefig("hm_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.png"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if svg: savefig("hm_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.svg"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if eps: savefig("hm_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.eps"  %  \
        (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))

show() if showPlot else close()
    

#====================  B E S T F I T   I M A G E =============================

#
# Plot the model visibility amplitude for MINIMUM CHI^2 
#

prm_bf = copy(sgra.ptotal)
for i in xrange(sgra.nptot):
    if sgra.invar[i] >= 0: prm_bf[i] = pout[sgra.invar[i],im]

if modnum == 9:
    Zsp, Re, rq, ecc, fade, gax, aq, gq, th = prm_bf

    Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext1, XYext1 =  \
        model_vis2bri_simple(xringaus, prm_bf, UVspan_glam, ngrid)
elif modnum == 13:
    Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th = prm_bf

    Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext1, XYext1 =  \
        model_vis2bri_simple(xringaus2, prm_bf, UVspan_glam, ngrid)

    etad = degrees(eta)
    alpd = degrees(alpha)
    betd = degrees(beta)

thd = degrees(th)


absVi = abs(Vi)
absBr = abs(Br)
phVi = arctan2(Vi.imag, Vi.real)

chi2bf, vicpm, pham, chi2m, datm = sgra.calcmodchi2(prm_bf)

chi2bf_r = chi2bf/(sgra.nvis+sgra.ncph-sgra.nprm-1)  # Reduced chi^2


if modnum == 9:

    figure(figsize=(6,6))
    
    imshow(absBr*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
           interpolation='nearest'); grid(1)
    contourf(absBr*1e3, 100, origin='lower', cmap=cm.hot, extent=XYext) 
    ## show()
    ## xlim(XYext[:2])
    ## ylim(XYext[2:])
    title('Best-Fit Model Brightness', fontsize=18)
    colorbar(shrink=0.7)
    xlabel(r'RA ($\mu$as)', fontsize=16)
    ylabel(r'Dec ($\mu$as)', fontsize=16)
    figtext(0.80, 0.81, 'mJy/pixel')   #, fontsize=16)

    grid(1)
    figtext(0.02, 0.95, r'$Z_{sp} = %4.2f$, $R_e = %4.1f$, $R_i/R_e = %4.2f$, $Ecc = %4.2f$' % (Zsp, Re, rq, ecc), fontsize=16)
    figtext(0.02, 0.89, r'$Fade =  %2.2f$, $g_{ax} = %4.2f$, $a_q = %4.2f$, $g_q = %4.2f$, $\theta = %4.0f^\circ$' % (fade, gax, aq, gq, thd), fontsize=16)
    figtext(0.4, 0.07, r'$\chi^2=%g,\; \rm{reduced:}\, %g$' % (chi2bf, \
            chi2bf_r), ha='center', fontsize=16)
    figtext(0.4,0.02, 'File: '+uvdata_file+'; ', ha='center', fontsize=16)

elif modnum == 13:

    figure(figsize=(8,8))
    imshow(absBr*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
           interpolation='nearest');
    contourf(absBr*1e3, 100, origin='lower', cmap=cm.hot, extent=XYext) 
    ## show()
    ## xlim(XYext[:2])
    ## ylim(XYext[2:])
    grid(1)    # ,vmax=3.2)
    title('Best-Fit Model Brightness', fontsize=18)
    colorbar(shrink=0.7)
    xlabel(r'RA ($\mu$as)', fontsize=16)
    ylabel(r'Dec ($\mu$as)', fontsize=16)
    figtext(0.80, 0.81, 'mJy/pixel')   #, fontsize=16)

    grid(1)
    figtext(0.02, 0.95, r'$Z_{sp} = %4.2f$, $R_e = %4.1f$, $r_{q} = %4.2f$, $Ecc = %4.2f$ $Fade =  %2.2f$, $g_{r} = %4.2f$' % (Zsp, Re, rq, ecc, fade, gr), fontsize=16)
    figtext(0.02, 0.89, r'$g_{ax} = %4.2f$, $a_q = %4.2f$, $g_q = %4.2f$, $\alpha = %4.0f^\circ$, $\beta = %4.0f^\circ$, $\eta = %4.0f^\circ$, $\theta = %4.0f^\circ$' % (gax, aq, gq, alpd, betd, etad, thd), fontsize=16)
    figtext(0.4, 0.08, r'$\chi^2=%g,\; \rm{reduced:}\, %g$' % (chi2bf, \
            chi2bf_r), ha='center', fontsize=16)
    figtext(0.4,0.02, 'File: '+uvdata_file+'; ', ha='center', fontsize=16)
    
xticks([80, 60,40,20, 0, -20, -40, -60, -80])    
yticks([80, 60,40,20, 0, -20, -40, -60, -80])    
#show()
        
print 'Best fit params = '
for i in xrange(sgra.nptot): print '%g\t' % prm_bf[i],
print
print 'Bestfit chi^2 = %g, reduced chi^2 = %g' % (chi2bf, chi2bf_r)

# ftxt = open("bf_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.txt"  %  \
#         (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter), 'w')
# if modnum == 9:
#     ftxt.write('# Zsp, Re, rq, ecc, fade, gax, aq, gq, thd, chi2bf, \
#	chi2bf_r\n')
#     ftxt.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f ' \
#                '%10.4f  %10.2f %10.4f\n' % \
#                (Zsp, Re, rq, ecc, fade, gax, aq, gq, thd, chi2bf, chi2bf_r))
# elif modnum == 13:
#     ftxt.write('# Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, ' \
#                'alpd, betd, etad, thd, chi2bf, chi2bf_r\n')
#     ftxt.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f ' \
#                '%10.4f %10.4f %10.4f %10.4f %10.4f  %10.2f %10.4f\n' 
#                % (Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, \
#                   alpd, betd, etad, thd, chi2bf, chi2bf_r))
# else:
#     pass
# ftxt.write('\n')
# savetxt(ftxt, absBr*1e3, '%14.6e')
# ftxt.close()

if png: savefig("bf_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.png"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if svg: savefig("bf_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.svg"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if eps: savefig("bf_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.eps"  %  \
        (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))

show() if showPlot else close()


#==============   A M P L I T U D E S   AND   P H A S E S   =================

#
# Plot the amplitudes at UV points
#
figure(figsize=(13,6));
subplots_adjust(0.05,0.05,0.95,0.95,0.2,0.2);
subplot2grid((16,16),(2,1), rowspan=12, colspan=6)

if modnum == 9:
    vm = ob.xringaus(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]
elif modnum == 13:
    vm = ob.xringaus2(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]
amm = abs(vm)
phm = arctan2(vm.imag,vm.real)

#errorbar(sgra.base, sgra.amp, yerr=2*sgra.eamp, color='k', linestyle='o', \
#         label='$\pm\sigma$')
plot(sgra.base, sgra.amp, 'b+', label='Observed')  #, markeredgewidth=0, ms=3)
plot(sgra.base, amm, 'r.', label='Best fit')    #, ms=3, markeredgewidth=0);
grid(1)
legend(loc='best')  #, labelspacing=0.1, prop={'size':12})
title('Amplitudes', fontsize=18)
xlabel(r'Baseline (G$\lambda)$', fontsize=16)
ylabel('Amplitude (Jy)', fontsize=16)
figtext(0.1, 0.03, 'File %s' % uvdata_file, fontsize=16)
figtext(0.6, 0.03, r'$\chi^2=%g,\; \rm{reduced:}\, %g$' % (chi2bf, \
        chi2bf_r), fontsize=18)

show() if showPlot else close()


#
# Plot the phases at UV points
#
rtod = 180./np.pi
plotPhases = any(sgra.phase <> 0)

subplot2grid((16,16),(2,9), rowspan=12, colspan=6)

if plotPhases:
    plot(sgra.base, rtod*sgra.phase, 'bs', label='Observed', \
                    markeredgewidth=0, ms=3)
plot(sgra.base, rtod*phm, 'r.', label='Best fit', ms=3); grid(1)
yticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
legend(loc='best')  #, labelspacing=0.1, prop={'size':12})
title('Phases', fontsize=18)
xlabel(r'Baseline (G$\lambda)$', fontsize=16)
ylabel('Phase (degrees)', fontsize=16)
figtext(0.5, 0.92, "Best-Fit %d-parameter Model vs Observations" % \
        modnum, ha='center', fontsize=20)

if png: savefig("amph_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.png"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if svg: savefig("amph_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.svg"  %  \
                (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
if eps: savefig("amph_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.eps"  %  \
        (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))

show() if showPlot else close()


#
# Plot closure phases
#

if  modnum == 9:
    vm = ob.xringaus(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]
elif modnum == 13:
    vm = ob.xringaus2(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]

amm = abs(vm)
phm = arctan2(vm.imag,vm.real)

if sgra.use_cphs:  # Plot closure phases
    if  modnum == 9:
        vm = ob.xringaus(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]
    elif modnum == 13:
        vm = ob.xringaus2(sgra.ulam, sgra.vlam, prm_bf) # Mod vis at [sgra.ulam,sgra.vlam]
    amm = abs(vm)
    phm = arctan2(vm.imag,vm.real)
    cphd = degrees(sgra.cphs)
    mcphd = degrees(vicpm[sgra.nvis:])

    #clpm, cpidm, tridm, tcphsecm = imgpu.calc_closures(phm, sgra.tsec, sgra.blin)
    figure(figsize=(20,6));
    plot(cphd, 'bo', label='Observed')
    plot(mcphd, 'r.', label='Best fit'); grid(1)
    plot(cphd, 'b-', label='Observed')
    plot(mcphd, 'r-', label='Best fit'); grid(1)
    yticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    legend(loc='best', labelspacing=0.1, prop={'size':12})
    title('Closure phases of the source, bestfit and the observed')
    xlabel(r'Time (cond. units)')
    ylabel('Closure phase (degrees)')
    figtext(0.40, 0.12, 'File %s' % uvdata_file, fontsize=20)


    if png: savefig("cp_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.png"  %  \
                    (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
    if svg: savefig("cp_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.svg"  %  \
                    (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
    if eps: savefig("cp_%s_%dprm_%s_%s_%sdeg%s_burn%03d_iter%05d.eps"  %  \
            (cnam, modnum, czsp, crad, crot, cnoise, nburn, niter))
    
show() if showPlot else close()



#show()
