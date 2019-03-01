from pylab import *
import pyfits as pf
#import scipy.ndimage as sn   # Multidimensional image processing library
from obsdatafit import rotzoom
import sys
import re
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix' # Allow math symbols b/w $ and $
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = False    #True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed

pZsp = True   # Print Zsp?

fits_file = sys.argv[1]

annot = True
cann = ''
vmax = None
cvmax = ''

argc = len(sys.argv)
if argc > 2:
    for i in xrange(2,argc):
        op = sys.argv[i]
        if op == '-noannot':
            annot = False
            cann = '_noannot'
        elif op[:4] == 'vmax':
            vmax = float(op[5:])
            cvmax = '_vmax%s' % op[5:]
        else:
            print "Unknown option '%s'", op
            sys.exit(0)
        

#
# Extract original image from the fits file
#
hdulist = pf.open(fits_file)
hdu = hdulist[0]
img = hdu.data   # original image [N x N] pix in Jy/pixel
N = hdu.header['naxis1']
if 'cdelt1' in hdu.header:
    delt = abs(hdu.header['cdelt1'])  # degrees per pixel
    delt_mas = delt*3600e3
elif 'scale' in hdu.header:
    delt_mas = abs(hdu.header['scale'])  # Milli-arcsecs per pixel
    delt = delt_mas*1e-3/3600.           # degrees per pixel
else:
    delt_mas = 0.001                     # Milli-arcsecs per pixel
    delt = delt_mas*1e-3/3600.           # degrees per pixel
    
    print 'WARNING: Neither CDELT nor SCALE keywords are not present.'
    print 'WARNING: Pixel size is assumed 1 micro-arcsecond.'

hdulist.close()

print 'N = %d, delt = %g (deg/pix) = %g (mas/pix)' % (N, delt, delt_mas)

#
# If the image is Quasi-Kerr, parse its name to determine
# parameters a, eps, inclination, Zsp, R, and theta
#
Avery = False
nums = re.findall('\d{6}|\d{0,6}\.\d{1,6}|\d{1,6}', fits_file) # Get all numbers
Zsp = sum(img)    #float(nums[1])
if len(nums) == 4:
    iasp = int(nums[0][0:2])
    iincl = int(nums[0][2:4])
    ieps = int(nums[0][4:6])
    R = float(nums[2])*32.
    thd = float(nums[3])
    Avery = True

fbn = re.findall('00[0-9]_00[0-9]_0[0-9]{2}', fits_file)
if fbn <> []:
    fbn = fbn[0]
    iasp = int(fbn[0:3])
    iincl = int(fbn[4:7])
    ieps = int(fbn[8:11])
    R = 32.
    thd = 0.
    Avery = True
    
if Avery:
    #Zsp = sum(img)    #float(nums[1])
    print 'iasp = %d, iincl = %d, ieps = %d' % (iasp, iincl, ieps)
    # Rulers for the epsilon, a (spin), and incl (inclination) axes
    nasp, ninc, neps = 10, 7, 19
    dela = 0.1
    deli = 10.
    dele = 0.1
    rula = 0.   + dela*arange(nasp) 
    ruli = 90.  - deli*arange(ninc) 
    rule = -0.8 + dele*arange(neps) 
    asp = rula[iasp]
    incl = ruli[iincl]
    eps = rule[ieps]
    print 'asp = %g, incl = %g (deg), eps = %g' % (asp, incl, eps)

    


XYspan_deg = N*delt    # degrees
XYspan = XYspan_deg*3600e6   # microarcseconds 
xyext = XYspan/2.

if N%2 == 0: # For EVEN N
    XYext = (xyext - XYspan/N, -xyext,  -xyext, xyext - XYspan/N)
else:        # For ODD  N
    XYext = (xyext, -xyext, -xyext, xyext)

print 'XYext = ', XYext

#
# A bug in matplotlib.pyplot? A strange dark feature appears at the
# brightness maximum.
#
a = zeros((N+20,N+20), dtype=float)
a[10:N+10,10:N+10] = img
img0 = a[10:N+10,10:N+10]

#img = rotzoom(img0, 1., 1., 135.)
img = img0

figure(figsize=(6,6))
imshow(img*1e3, origin='lower', cmap=cm.hot, extent=XYext, \
       interpolation='nearest', vmax=vmax)       #, vmax=5.2);
contourf(img*1e3, 100, origin='lower', cmap=cm.hot, extent=XYext, vmax=vmax)
#axis('equal')
colorbar(shrink=0.7)

#figtext(0.18, 0.06, 'file %s' % fits_file.replace('_','\_'), fontsize=16)
figtext(0.18, 0.06, 'file %s' % fits_file, fontsize=16)

if annot:
#    grid(1)
    title('Quasi-Kerr Brighrness', fontsize=18)
    #colorbar(shrink=0.7)
    xlabel(r'RA ($\mu$as)', fontsize=16)
    ylabel(r'Dec ($\mu$as)', fontsize=16)
    #figtext(0.18, 0.06, 'file %s' % fits_file.replace('_','\_'), fontsize=16)
    figtext(0.78, 0.81, 'mJy/pixel', fontsize=16)
    if Avery:
        #figtext(0.4, 0.94, r'$a = %g,\, i = %g^{\circ},\, \epsilon = %g$' %
        figtext(0.4, 0.94, r'$a = %g,\, i = %g^\mathrm{o},\, \epsilon = %g$' %
                (asp, incl, eps), ha='center', fontsize=16)

        if pZsp: figtext(0.1, 0.88, r'$Z_{sp} = %4.2f\, \mathrm{Jy},\, ' \
				r'R = %5.1f\, \mu \mathrm{as},\,' \
				r'\theta = %6.1f^\mathrm{o}$' % (Zsp, R, thd), fontsize=16)
                #r'\theta = %6.1f^\circ$' % (Zsp, R, thd), fontsize=20)
    else:
        if pZsp: figtext(0.5, 0.90, r'$Z_{sp} = %4.2f\, Jy$' % Zsp, ha='center',
                         fontsize=16)
else:
    xticks(())
    yticks(())
    

show()

#
# SOMETHING WRONG ABOUT savefig in CentOS!!!
#
#savefig("%s%s%s.png"  % (fits_file.replace('_','\_'), cvmax, cann))
# savefig("%s%s%s.png"  % (fits_file, cvmax, cann))
# savefig("%s%s%s.pdf"  % (fits_file, cvmax, cann))
