#
# qk_rotzoom2fits.py
#
# Create fits file with a Quasi-Kerr black hole image rotated and zoomed
# from a text file provided by Avery Broderick. These quasi-Kerr image
# data can be found? for example, at
#   leonid1.haystack.mit.edu:  \
#        /data/moved_from_home_benkev/quasi-Kerr_images/imlib_ref[0-3]
# or at
#   leonid2.haystack.mit.edu:/data1/imlib_ref0
# The file names have format like pmap__000_000_000_2.
#
# For usage, see Example.
#

from pylab import *
import pyfits as pf
import scipy.ndimage as sn   # Multidimensional image processing library
import obsdatafit
reload(obsdatafit)
from obsdatafit import rotzoom
import os, sys, re


#
# Example:
#
# Create fits file "040208_1_1.4_120.fits" from pmap__004_002_008_2
# with the same Zsp, 1.4 times larger and 120 degrees rotated:
#
# %run qk_rotzoom2fits.py pmap__004_002_008_2 040208_1_1.4_120.fits   \
#                                                           	1.  1.4  120.
#

print 'sys.argv = ', sys.argv
print 'len(sys.argv) = ', len(sys.argv)

# Rulers for spin, inclination, and epsilon
nasp, ninc, neps = 10, 7, 19
dela = 0.1
deli = 10.
dele = 0.1
rula = 0.   + dela*arange(nasp) 
ruli = 90.  - deli*arange(ninc) 
rule = -0.8 + dele*arange(neps) 

Zsp, R, thd = 1., 1., 0.   # Default 
th = radians(thd)

print 

if len(sys.argv) == 6:
    qk_file =  sys.argv[1]
    fits_file =  sys.argv[2]
    Zsp, R, thd = float_(sys.argv[3:])
    th = radians(thd)
elif len(sys.argv) == 3:
    qk_file =  sys.argv[1]
    fits_file =  sys.argv[2]
else:
    print "\nInvalid command line. Usage:"
    print "qk_rotzoom2fits.py   qk_file    fits_file    Zsp,    R,    th"
    print "or"
    print "qk_rotzoom2fits.py    qk_file    fits_file"
    sys.exit(0);

scale_flux = 2.5000;
mass = 4.3;                        # x million solar masses #
distance = 8.0;                    # kpc #
pixincr_in_M = 0.30150808;
npix = 100;            # in each dimension #

if npix%2 == 0:
    RA_refpix = npix//2;            #  #
    Dec_refpix = RA_refpix + 1;     #  #
else: # Odd npix:
    RA_refpix = Dec_refpix = npix//2;              #  #


# pixincr_deg = 4.4583333E-10;
# pixel scale is set explicitly above

M_in_arcsec = 9.87122944e-6;
M_in_arcsec *= mass;
M_in_arcsec /= distance;
pixincr_deg = pixincr_in_M*M_in_arcsec/3600.0;
print "pixel increment in degrees: %g\n" % pixincr_deg

fh = open(qk_file)
fh.readline(); fh.readline(); fh.readline() # Skip header
dat = loadtxt(fh)
fh.close()
x = dat[:,0]
y = dat[:,1]
flx0 = dat[:,2]
flx0 = flx0.reshape((npix,npix))
absf0 = sum(flx0)

#
# Rotate and zoom the original image, if needed
#
if len(sys.argv) == 6:
    flx = rotzoom(flx0, Zsp, R, thd)
else:
    flx = flx0

absf = sum(flx)
flx = Zsp*(absf0/absf)*flx
absZsp = sum(flx)   # Zsp*scale_flux  # Jansky
absR = 32.*R             # microarcsrconds

fnum = re.findall('00[0-9]_00[0-9]_0[0-9]{2}', qk_file)
if fnum <> []:
    fnum = fnum[0]
    iasp = int(fnum[0:3])
    iincl = int(fnum[4:7])
    ieps = int(fnum[8:11])
    asp = rula[iasp]
    incl = ruli[iincl]
    eps = rule[ieps]
else:
    print "Wrong quasi Kerr file name!"
    sys.exit(0)

Rshadow = (4.5 + 0.7*sqrt(1. - asp**2))*5.1  # shadow size (microarcseconds)
Displ = 11.*asp                # shadow center displacement(microarcseconds)

hdu = pf.PrimaryHDU(flx)

h = hdu.header
h.add_comment('FITS (Flexible Image Transport System) '\
              'format is defined in "Astronomy', after=10)
h.add_comment('and Astrophysics", volume 376, page 359; '\
              'bibcode: 2001A&A...376..359H', after=6)
h.add_comment('Parameters:')
h.add_comment('asp = %g: spin (M)' % asp)
h.add_comment('i = %g: inclination (degrees)' % incl)
h.add_comment('eps = %g: epsilon (deviation from Kerr metric)' % eps)
h.add_comment('Zsp = %g: zero spacing flux (Jansky)' % absZsp)
h.add_comment('R = %g: external ring radius (microarcseconds)' % absR)
h.add_comment('th = %g (rad) = %g (deg): angle of rotation' % (th, thd))
h.add_comment('source file: %s' % os.path.basename(qk_file))
h.add_comment('Rshadow = %g: shadow size (microarcseconds)' % Rshadow)
h.add_comment('Displ = %g: shadow center displacement(microarcseconds)' % Displ)
h.update('CTYPE1', 'RA---SIN', 'The sin sky projection')
h.update('CTYPE2', 'DEC--SIN', 'The sin sky projection')
h.update('CTYPE3', 'STOKES')
h.update('CRPIX1', RA_refpix, 'RA coordinate of reference pixel')
h.update('CRPIX2', Dec_refpix, 'Dec coordinate of reference pixel')
h.update('CRVAL1', 266.416837083, 'RA of reference pixel in DEGREES')
h.update('CRVAL2', -29.007810556, 'Dec of reference pixel in DEGREES')
h.update('TELESCOP', 'VLBI    ')
h.update('INSTRUME', 'VLBI    ', '')
h.update('BUNIT', 'Jy/pixel', 'Units of the brightness')
h.update('OBJECT', 'SGRA    ')
h.update('DATE-OBS', '05/10/07')
h.update('CDELT1', -pixincr_deg, 'RA  increment in DEGREES/pixel')
h.update('CDELT2',  pixincr_deg, 'Dec increment in DEGREES/pixel')
h.update('EPOCH', 2000.)
h.update('OBSRA', 266.416837083)
h.update('OBSDEC', -29.007810556)
#h.update('', '', '')

hdu.writeto(fits_file)


