#
# Programs to fit "gaus ring" and "gradient ring" models to amplitude and
# closure phase data.
# Take as input two ascii files from MAPS and closure phase script and fit
# a 5-d parameter space.
#
#from time import mktime, strptime
from time import strptime, strftime, gmtime
from calendar import timegm
from pylab import *
from numpy import *
from bitstring import BitStream
from matplotlib.pyplot import *
from matplotlib.mlab import find
from scipy.special import jv, jn, jvp, j0, j1
import scipy.ndimage as sn   # Multidimensional image processing library
from mpl_toolkits.mplot3d import axes3d
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import pyfits as pf
import os
import re


#
# On 32-bit system with sys.maxint = 2**32-1:
# Maximum 2**10 = 1024 antennas and closure phases
#PK2I = 2**16   # 2 integers in 2x16=32-bit integer (a1, a2 in baseline)
#PK3I = 2**10   # 3 integers in 3x10=30-bit integer (i1, i2, i3 in closure) 
# On 64-bit system with sys.maxint = 2**64-1:
# Maximum 2**21 = 2,097,152 antennas or closure phases
PK2I = 2**21   # 2 integers in 2x21=42-bit integer (a1, a2 in baseline)
PK3I = 2**21   # 3 integers in 3x21=63-bit integer (i1, i2, i3 in closure) 

def to_pi(th):
    """
    Reduce an angle or array of angles in radians th to [-pi..pi] 
    """
    if isscalar(th):
        th = th - 2.*pi*trunc(th/(2.*pi))
        if th < -pi:
            th = th + 2.*pi
        if th >  pi:
            th = th - 2.*pi
        return th
    else: # th is array or sequence
        tha = array(th)
        return arctan2(sin(tha), cos(tha))
        
def to_180(thd):
    """
    Reduce an angle or array of angles in degrees thd to [-180..180] 
    """
    return degrees(to_pi(radians(thd)))


def cbinom(n, m):
    """
    Return value of m-th binomial coefficient of (x+y)^n expansion.
    Or, which is the same, the number of m-subsets of a set with
    n elements
    """
    if n < 0 or m < 0 or m > n:
        return -1
    if m == 0 or m == n:
        return 1
    if m == 1 or m == n-1:
        return n
    num = n
    den = m
    if m > 2:
        for i in xrange(m-1,1,-1):
            den = den*i
    if n > 2:
        for i in xrange(n-1,n-m,-1):
             num = num*i
    return num/den



def combgen(n, m):
    """
    Generates an array C[n,m] of all the subsets of size m
    out of the set of n elements. Returns the array C[csize,m]
    of subsets.
    The C[] lines are indices into a sequence s of arbitrary items,
    pointing at those of them that are chosen for a subset.
    However, the most convenient are sets specified as arrays,
    because arrays allow group indexing.
    For example, say, we have a set s = array((10, 14, 7, 3, 8, 25)).
    After the call
        c = combgen(6,3)
    each line of c has 3 indices into s. All the c lines enumerate
    all the s subsets. For instance:
        c[5,:] = array([0, 2, 4])
    then
        s[c[5,:]] is
           array([10,  7,  8]).
    
    Thus, the array of indices returned by combgen can be used for
    obtaining subsets of sets containing elements of virtually any type. 
    """

    #-------------------------------------- Local function
    
    def cgenrecurs(S, n, m, p, C, ic):
    ## """
    ## Generates the subsets of size m from the set S[n] starting
    ## from the p-th position. Returns the index into array C where
    ## the subsets are stored.
    ## This local recursive function is intended to be only used
    ## in the wrapping combgen() function.
    ## """
        if p < m:
            if p == 0:
                for i in xrange(n):
                    S[0] = i
                    ic = cgenrecurs(S, n, m, 1, C, ic)
            else: # Here p > 0:
                for i in xrange(S[p-1]+1,n):
                    S[p] = i
                    ic = cgenrecurs(S, n, m, p+1, C, ic)
        else: # Here p == m
            C[ic,:] = S[:m]
            ic = ic + 1
        return ic
    #---------------------------------------- End of local function

    csize = cbinom(n, m)
    S = zeros((n), int)         # Set of indices
    C = zeros((csize, m), int)  # Result: all the subsets' indices
    ic = 0                      # Current position in C for writing 
    ic = cgenrecurs(S, n, m, 0, C, ic) 
    return C



def find_triangles(antennas):
    """
    This function takes as its input the array of antennas, and returns
    an array filled with all possible unique triangles and a 1D array
    of the triangle IDs. Call:
    triangles, triids = find_triangles(antennas)
    """
    antarr = array(antennas)        # Convert to an array
    nant = size(antarr)
    itri = combgen(nant, 3)         # Indices of all possible antenna triplets
    ntri = size(itri,0)             # Number of triangles
    tri = empty(itri.shape, int)    # Array of triangles
    trid = empty(ntri, int)         # Array of unique IDs for the triangles 
    for i in xrange(ntri): 
        tri[i,:] = antarr[itri[i,:]]  # Triangle of antennae
        # A triangle ID is made as a radix maxantnum number whose
        # "digits" are the antenna numbers in the triangle.
        # It is unique for each triangle. The ID is used to speed up
        # searches for a specific triangle in the triangle array.
        trid[i] = tri[i,0] + PK3I*(tri[i,1] + PK3I*tri[i,2])
    return tri, trid



def get_time_indices(tsec):
    """
    Find all the equal-time time spans in the tsec array.
    Returns two arrays of the indices into the tsec array (row numbers),
    ibeg[] and iend[], each pointing at the beginning and the end of an
    equal-time data span.
    
    """
    #
    # Get pointers to starts of the equal-time intervals.
    # The last points past the end of tsec array - for consistency.
    #
    ntsec = len(tsec)
    itim = where(diff(tsec))[0] + 1 # 1 needed to fix location error
    itim = hstack((0, itim, ntsec)) # Starts at 0 at the beginning
    #
    # The itim[] now points at *beginnings* of the equal-time data pieces,
    # but we still do not know where the *last* piece (span?) ends.
    # The first difference of itim contains lengths of equal-time spans.
    # Find the last 1 in the itim 1st differences, ditim[]:
    #
    ditim =  diff(itim)     # Lengths of eq-time spans
    nditim = len(ditim)

    #print 'ntsec = ', ntsec, ', nitim = ', len(itim), ', nditim = ', nditim
    #print 'itim = \n', itim
    #print 'ditim = \n', ditim

    #
    # Gather all pointers to the intervals longer than 2 in the
    # lists ibeg and iend
    #
    i = 0
    ibeg = []
    iend = []
    while i < nditim:  # nditim = nitim - 1, so itim[i+1] is valid
        if ditim[i] >= 3:
            ibeg.append(itim[i])
            iend.append(itim[i+1])
        i = i + 1
        
    return ibeg, iend



def calc_closures(phase, tsec, blid):
    """
    Calculate closure phases from the data returned by readamp2():
    ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
    blid must be an int array of baseline indices, bl1 + 0x200000*bl2.
    Returns: 
    cphs: closure phases
    cpid: closure phase indices, dtype=int64
          cpid is calculated as i1 + 0x200000*(i2 + i3 *0x200000),
          where 0x200000 = 2097152 = 2**21,
          and i1, i2, and i3 are indices into ul, vl, amp and phase.
          So, a closure phase can be calculated as
          cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
    trid: triangle indices.
          trid is calculated as a1 + 0x200000*(a2 + a3*0x200000),
          where 0x200000 = 2097152 = 2**21,
          and a1, a2, and a3 are numbers of the stations in a triangle.
    tcphsec: closure phase times in seconds
    If phase has length 0 to 2, returns cphs, cpid, and tcphsec with zero length
    
    calc_closures() is a replacement for readcphs2():
    cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
    where tcphsec is 'tsec for closure phases'.
    """
    if len(phase) >= 3:
        # Get starts and ends of equal-time spans > 3 baselines long
        ibeg, iend = get_time_indices(tsec) 
        ntimes = len(ibeg)  # Number of same-time data segments > 3 bls
        
    if len(phase) < 3 or ntimes == 0: # No triangles
        cphs = array([], dtype=float)
        cpid = array([], dtype=int64)
        trid = array([], dtype=int)
        tcphsec = array([], dtype=float)
        return cphs, cpid, trid, tcphsec  #================================>>>

    times = tsec[ibeg]            # Unique times
    bls = vstack((blid%PK2I, blid/PK2I)).T
    antennas = unique(bls)
    triangles, triids = find_triangles(antennas)
    ntriangles = len(triangles[:,0])
    maxcps = ntimes*ntriangles   # Maximal number of closure phases
    cphs = zeros(maxcps, float)
    cpix = zeros((maxcps,3), int)
    cpid = zeros((maxcps), int64)
    trid = zeros((maxcps), int)
    tcphsec = zeros((maxcps), float) # Closure phase times in seconds

    iclp = 0
    for itim in xrange(ntimes):   # Consider ith equal-time span
        i0 = ibeg[itim]           # Span start in data[]
        i1 = iend[itim]           # Span end   in data[]
        nbl = i1 - i0             # Number of data rows in the span
        bl = int_(bls[i0:i1,:])   # Convert i0:i1 baseline array to int
        ant = unique(bl)          # Leave only sorted unique antenna numbers
        nant = len(ant)           # Number of antenne employed in the i0-i1 span
        # Find all possible triangles in the span
        itri = combgen(nant, 3)   # Indices into ant to get triangles
        ntri = size(itri,0)       # Number of triangles in this span
                
        for j in xrange(ntri): # Over all triangles in this span
            a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
            # Find indices of the three baselines made of the antennae
            ib1 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
            ib2 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
            ib3 = i0 + where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
            #print 'ib1, ib2, ib3 = ', ib1, ib2, ib3
            # Phases in 
            ph1 = phase[ib1]
            ph2 = phase[ib2]
            ph3 = phase[ib3]


            if ph1 != 0. and ph2 != 0. and ph3 != 0.:
                ph = ph1 + ph2 - ph3
                cphs[iclp] = arctan2(sin(ph), cos(ph))
                trid[iclp] = a1 + PK3I*(a2 + PK3I*a3)
                tcphsec[iclp] = times[itim]
                # Get unique closure phase ID
                # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
                cpid[iclp] = ib1 + PK3I*(ib2 + PK3I*ib3)
                iclp = iclp + 1


    cphs = cphs[:iclp]
    cpid = cpid[:iclp]
    trid = trid[:iclp]
    tcphsec = tcphsec[:iclp]

    return cphs, cpid, trid, tcphsec        # End of calc_closures




def calc_closures1(phase, tsec, blid):
    """
    Calculate closure phases from the data returned by readamp2():
    ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
    blid must be an int array of baseline indices, bl1 + 0x200000*bl2.
    Returns: 
    cphs: closure phases
    cpix: closure phase indices [i1, i2, i3].
          cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
    trid: triangle indices.
          trid is calculated as a1 + 0x200000*(a2 + a3*0x200000),
          where 0x200000 = 2097152 = 2**21,
          and a1, a2, and a3 are numbers of the stations in a triangle.
    tcphsec: closure phase times in seconds
    If phase has length 0 to 2, returns cphs, cpid, and tcphsec with zero length
    
    calc_closures1() is a replacement for readcphs2():
    cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
    where tcphsec is 'tsec for closure phases'.
    calc_closures1() differs from calc_closures() in two respects:
    - returned arays are either int32 or float32;
    - instead of the cpid[ncph] codes the cpix[ncph,3] antenna triplets
        are returned.
    """
    if len(phase) >= 3:
        # Get starts and ends of equal-time spans > 3 baselines long
        ibeg, iend = get_time_indices(tsec) 
        ntimes = len(ibeg)  # Number of same-time data segments > 3 bls
        
    if len(phase) < 3 or ntimes == 0: # No triangles
        cphs = np.array([], dtype=np.float32)
        #cpid = np.array([], dtype=np.int64)
        cpix = np.array([], dtype=np.int32)
        trid = np.array([], dtype=np.int32)
        tcphsec = np.array([], dtype=np.float32)
        return cphs, cpid, trid, tcphsec  #================================>>>

    times = tsec[ibeg]            # Unique times
    bls = np.vstack((blid%PK2I, blid/PK2I)).T
    antennas = np.unique(bls)
    triangles, triids = find_triangles(antennas)
    ntriangles = len(triangles[:,0])
    maxcps = ntimes*ntriangles   # Maximal number of closure phases
    cphs = np.zeros(maxcps, np.float32)
    #cpix = np.zeros((maxcps,3), int)
    #cpid = np.zeros((maxcps), np.int64)
    cpix = np.zeros((maxcps,3), np.int32)
    trid = np.zeros((maxcps), np.int32)
    tcphsec = np.zeros((maxcps), float) # Closure phase times in seconds

    iclp = 0
    for itim in xrange(ntimes):   # Consider ith equal-time span
        i0 = ibeg[itim]           # Span start in data[]
        i1 = iend[itim]           # Span end   in data[]
        nbl = i1 - i0             # Number of data rows in the span
        bl = np.int_(bls[i0:i1,:])   # Convert i0:i1 baseline array to int
        ant = np.unique(bl)          # Leave only sorted unique antenna numbers
        nant = len(ant)           # Number of antenne employed in the i0-i1 span
        # Find all possible triangles in the span
        itri = combgen(nant, 3)   # Indices into ant to get triangles
        ntri = np.size(itri,0)       # Number of triangles in this span
                
        for j in xrange(ntri): # Over all triangles in this span
            a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
            # Find indices of the three baselines made of the antennae
            ib1 = i0 + np.where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
            ib2 = i0 + np.where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
            ib3 = i0 + np.where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
            #print 'ib1, ib2, ib3 = ', ib1, ib2, ib3
            # Phases in 
            ph1 = phase[ib1]
            ph2 = phase[ib2]
            ph3 = phase[ib3]


            if ph1 != 0. and ph2 != 0. and ph3 != 0.:
                ph = ph1 + ph2 - ph3
                cphs[iclp] = np.arctan2(np.sin(ph), np.cos(ph))
                trid[iclp] = a1 + PK3I*(a2 + PK3I*a3)
                tcphsec[iclp] = times[itim]
                # Get unique closure phase ID
                # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
                #cpid[iclp] = ib1 + PK3I*(ib2 + PK3I*ib3)
                cpix[iclp,:] = ib1, ib2, ib3
                iclp = iclp + 1


    cphs = cphs[:iclp]
    #cpid = cpid[:iclp]
    cpix = cpix[:iclp,:]
    trid = trid[:iclp]
    tcphsec = tcphsec[:iclp]

    return cphs, cpix, trid, tcphsec        # End of calc_closures




def calc_closuvap(phase, tsec, blid, ulam, vlam, amp):
    """
    Calculate closure phases from the data returned by readamp2():
    ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
    blid must be an int array of baseline indices, bl1 + 0x200000*bl2.
    Returns: 
    cphs: closure phases
    cpid: closure phase indices, dtype=int64
          cpid is calculated as i1 + 0x200000*(i2 + i3 *0x200000),
          where 0x200000 = 2097152 = 2**21,
          and i1, i2, and i3 are indices into ul, vl, amp and phase.
          So, a closure phase can be calculated as
          cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
    trid: triangle indices.
          trid is calculated as a1 + 0x200000*(a2 + a3*0x200000),
          where 0x200000 = 2097152 = 2**21,
          and a1, a2, and a3 are numbers of the stations in a triangle.
    tcphsec: closure phase times in seconds
    uc, vc: u and v triplets for each closure phase.
    
    If phase has length 0 to 2, returns cphs, cpid, tcphsec,
    uc, vc, ac, and pc with zero length.
    
    calc_closures() is a replacement for readcphs2():
    cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
    where tcphsec is 'tsec for closure phases'.
    """
    if len(phase) >= 3:
        # Get starts and ends of equal-time spans > 3 baselines long
        ibeg, iend = get_time_indices(tsec) 
        ntimes = len(ibeg)  # Number of same-time data segments > 3 bls
        
    if len(phase) < 3 or ntimes == 0: # No triangles
        cphs = array([], dtype=float)
        cpid = array([], dtype=int64)
        trid = array([], dtype=int)
        tcphsec = array([], dtype=float)
        uc = array([], dtype=float)
        vc = array([], dtype=float)
        ac = array([], dtype=float)
        pc = array([], dtype=float)
        return cphs, cpid, trid, tcphsec, uc, vc, ac, pc  #================>>>

    times = tsec[ibeg]            # Unique times
    bls = vstack((blid%PK2I, blid/PK2I)).T
    antennas = unique(bls)
    triangles, triids = find_triangles(antennas)
    ntriangles = len(triangles[:,0])
    maxcps = ntimes*ntriangles   # Maximal number of closure phases
    cphs = zeros(maxcps, float)
    cpix = zeros((maxcps,3), int)
    cpid = zeros((maxcps), int64)
    trid = zeros((maxcps), int)
    tcphsec = zeros((maxcps), float) # Closure phase times in seconds
    uc = zeros((maxcps,3), float)
    vc = zeros((maxcps,3), float)
    ac = zeros((maxcps,3), float)
    pc = zeros((maxcps,3), float)
    iclp = 0                      # Closure phase index
    for itim in xrange(ntimes):   # Consider ith equal-time span
        i0 = ibeg[itim]           # Span start in data[]
        i1 = iend[itim]           # Span end   in data[]
        nbl = i1 - i0             # Number of data rows in the span
        bl = int_(bls[i0:i1,:])   # Convert i0:i1 baseline array to int
        ant = unique(bl)          # Leave only sorted unique antenna numbers
        nant = len(ant)           # Number of antenne employed in the i0-i1 span
        # Find all possible triangles in the span
        itri = combgen(nant, 3)   # Indices into ant to get triangles
        ntri = size(itri,0)       # Number of triangles in this span
                
        for j in xrange(ntri): # Over all triangles in this span
            a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
            # Find indices of the three baselines made of the antennae
            ib1 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
            ib2 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
            ib3 = i0 + where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
            #print 'ib1, ib2, ib3 = ', ib1, ib2, ib3
            # Phases in 
            ph1 = phase[ib1]
            ph2 = phase[ib2]
            ph3 = phase[ib3]


            if ph1 != 0. and ph2 != 0. and ph3 != 0.:
                ph = ph1 + ph2 - ph3
                cphs[iclp] = arctan2(sin(ph), cos(ph))
                trid[iclp] = a1 + PK3I*(a2 + PK3I*a3)
                tcphsec[iclp] = times[itim]
                # Get unique closure phase ID
                # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
                cpid[iclp] = ib1 + PK3I*(ib2 + PK3I*ib3)
                uc[iclp,:] = ulam[ib1], ulam[ib2], ulam[ib3] 
                vc[iclp,:] = vlam[ib1], vlam[ib2], vlam[ib3]
                ac[iclp,:] = amp[ib1], amp[ib2], amp[ib3]
                pc[iclp,:] = ph1, ph2, ph3
                iclp = iclp + 1


    cphs = cphs[:iclp]
    cpid = cpid[:iclp]
    trid = trid[:iclp]
    tcphsec = tcphsec[:iclp]
    uc = uc[:iclp,:]
    vc = vc[:iclp,:]
    ac = ac[:iclp,:]
    pc = pc[:iclp,:]

    return cphs, cpid, trid, tcphsec, uc, vc, ac, pc   # End of calc_closuvap




def calc_closures_cuv(ul, vl, amp, phase, tsec, blid):
    """
    TO BE CORRECTED SIMILAR TO calc_closures() !!!!!!!!!!!!!!!!!!!!!!!
    Calculate closure phases from the data returned by readamp2():
    ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
    calc_closures() is a replacement for readcphs2():
    cphs, cuv, tri, tscp  = readcphs2(cphs_file)
    where tscp is 'tsec for closure phases'.
    The same data are returned
    """
    itimes, ibad = get_time_indices(tsec)
    ntimes = len(itimes) - 1        # Number of same-time data segments > 3 bls
    times = tsec[itimes]            # Unique times
    ublid = unique(blid)                      # Unique baseline IDs
    bls = vstack((blid%PK2I, blid/PK2I)).T
    antennas = unique(bls)
    triangles, triids = find_triangles(antennas)
    ntriangles = len(triangles[:,0])
    maxcps = len(ul)  # Maximal number of closure phases is less than uv points
    cphs = zeros(maxcps, float)
    cuv = zeros((maxcps,6), float)
    tri = zeros((maxcps,3), int)
    cpid = zeros((maxcps), int64)
    tcphsec = zeros((maxcps), float) # Closure phase times in seconds

    iclp = 0
    for itim in xrange(ntimes):   # Consider ith equal-time span
        i0 = itimes[itim]         # Span start in data[]
        i1 = itimes[itim+1]       # Span end   in data[]
        if i1 in ibad: continue   #==========>>> Ignore bad bls[i0:i1]
        nbl = i1 - i0            # Number of data rows in the span
        bl = int_(bls[i0:i1,:]) # Convert baseline array to int
        ant = unique(bl)         # Leave only sorted unique antenna numbers
        nant = len(ant)          # Number of antenne employed in the i0-i1 span
        # Find all triangles in the span
        itri = combgen(nant, 3)   # Indices into ant to get triangles
        ntri = size(itri,0)       # Number of triangles in this span

        for j in xrange(ntri): # Over all triangles in this span
            a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
            # Find indices of the three baselines made of the antennae
            ib1 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a2))[0]
            ib2 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a3))[0]
            ib3 = i0 + where((bl[:,0] == a2) & (bl[:,1] == a3))[0]
            # Phases in 
            ph1 = phase[ib1]
            ph2 = phase[ib2]
            ph3 = phase[ib3]


            if ph1 != 0. and ph2 != 0. and ph3 != 0.:
                ph = ph1 + ph2 - ph3 
                cphs[iclp] = arctan2(sin(ph), cos(ph))   
                cuv[iclp,:] = ul[ib1], vl[ib1], ul[ib2], vl[ib2], \
                               ul[ib3], vl[ib3] 
                tri[iclp,:] = a1, a2, a3
                tcphsec[iclp] = times[itim]
                # Get unique closure phase ID
                # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
                cpid[iclp] = ib1 + PK3I*(ib2 + PK3I*ib3)
                iclp = iclp + 1


    cphs = cphs[:iclp]
    cuv = cuv[:iclp,:]
    tri = tri[:iclp,:]
    cpid = cpid[:iclp]
    tcphsec = tcphsec[:iclp]

    return cphs, cuv, tri, tcphsec        # End of calc_closures_cuv


## def calc_closures1(phase, tsec, blid):
##     """
##     Calculate closure phases from the data returned by readamp2():
##     ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
##     blid must be an int array of baseline indices, bl1 + PK2I*bl2.
##     Unlike calc_closures(), calc_closures1() also returns trid 
##     Returns: 
##     cphs: closure phases
##     cpid: closure phase indices.
##           cpid is calculated as i1 + 1024*(i2 + i3 *1024),
##           where i1, i2, and i3 are indices into ul, vl, amp and phase.
##           So, a closure phase can be calculated as
##           cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
##     trid: triangle indices.
##           trid is calculated as a1 + 1024*(a2 + a3 *1024),
##           where a1, a2, and a3 are numbers of the stations in a triangle.
##     tcphsec: closure phase times in seconds
    
##     calc_closures() is a replacement for readcphs2():
##     cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
##     where tcphsec is 'tsec for closure phases'.
##     """
##     if len(phase) < 3:
##         cphs = array([], dtype=float)
##         cpid = array([], dtype=int)
##         trid = array([], dtype=int)
##         tcphsec = array([], dtype=float)
##         return cphs, cpid, trid, tcphsec
    
##     itimes, ibad = get_time_indices(tsec)
##     ntimes = len(itimes) - 1        # Number of same-time data segments > 3 bls
##     times = tsec[itimes]            # Unique times
##     bls = vstack((blid%PK2I, blid/PK2I)).T
##     antennas = unique(bls)
##     triangles, triids = find_triangles(antennas)
##     ntriangles = len(triangles[:,0])
##     # Max number of closure phases is < than uv points #
##     #maxcps = len(phase)
##     maxcps = ntimes*ntriangles   # Maximal number of closure phases
##     cphs = zeros(maxcps, float)
##     #tri = zeros((maxcps,3), int)
##     cpix = zeros((maxcps,3), int)
##     cpid = zeros((maxcps), int)
##     trid = zeros((maxcps), int)
##     tcphsec = zeros((maxcps), float) # Closure phase times in seconds

##     iclp = 0                      # Count total number of closure phases
##     for itim in xrange(ntimes):   # Consider ith equal-time span
##         i0 = itimes[itim]         # Span start in data[]
##         i1 = itimes[itim+1]       # Span end   in data[]
##         if i1 in ibad: continue   #==========>>> Ignore bad bls[i0:i1]
##         nbl = i1 - i0            # Number of data rows in the span
##         bl = int_(bls[i0:i1,:]) # Convert baseline array to int
##         ant = unique(bl)         # Leave only sorted unique antenna numbers
##         nant = len(ant)          # Number of antenne employed in the i0-i1 span
##         # Find all triangles in the span
##         itri = combgen(nant, 3)   # Indices into ant to get triangles
##         ntri = size(itri,0)       # Number of triangles in this span

##         for j in xrange(ntri): # Over all triangles in this span
##             a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
##             # Find indices of the three baselines made of the antennae
##             ib1 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
##             ib2 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
##             ib3 = i0 + where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
##             ## Get unique triangle ID
##             #triid = a1 + maxantnum*(a2 + maxantnum*a3) # Unique triangle ID
##             #itriangle = find(triids == triid)[0]          # Triangle number
##             # Amplitudes
##             #amp1 = amp[ib1]
##             #amp2 = amp[ib2]
##             #amp3 = amp[ib3]
##             # Phases in 
##             ph1 = phase[ib1]
##             ph2 = phase[ib2]
##             ph3 = phase[ib3]


##             if ph1 != 0. and ph2 != 0. and ph3 != 0.:
##                 ph = ph1 + ph2 - ph3 
##                 cphs[iclp] = arctan2(sin(ph), cos(ph))
##                 #tri[iclp,:] = a1, a2, a3
##                 trid[iclp] = a1 + 1024*(a2 + 1024*a3)
##                 tcphsec[iclp] = times[itim]
##                 # Get unique closure phase ID
##                 # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
##                 cpid[iclp] = ib1 + 1024*(ib2 + 1024*ib3)
##                 #print i0, ib1, ib2, ib3
##                 #cpix[iclp,:] = ib1, ib2, ib3
##                 iclp = iclp + 1


##     cphs = cphs[:iclp]
##     #tri = tri[:iclp,:]
##     #cpix = cpix[:iclp,:] # Same as cpid, but not packed into one int32
##     cpid = cpid[:iclp]
##     trid = trid[:iclp]
##     tcphsec = tcphsec[:iclp]

##     return cphs, cpid, trid, tcphsec        # End of calc_closures1




## def calc_closures2(uv, phase, tsec, blid):
##     """
##     Calculate closure phases from the data returned by readamp2():
##     ul, vl, amp, phase, tsec, blid = readamp2(amp_file)
##     blid must be an int array of baseline indices, bl1 + PK2I*bl2.
##     Differs from calc_closures() in that the former needs two separate
##     arrays for ul and vl, while this function needs only one uv[nvis,2] array. 
##     uv unites ul and vl into one 2d array for effective GPU calculations:
##     uv = empty((len(ul),2), dtype=float); uv[:,0] = ul; uv[:,1] = vl 
##     Returns: 
##     cphs: closure phases
##     cpid: closure phase indices.
##           cpid is calculated as i1 + 1024*(i2 + i3 *1024),
##           where i1, i2, and i3 are indices into ul, vl, amp and phase.
##           So, a closure phase can be calculated as
##           cphs = phase[i1] + phase[i2] - phase[i3] for a given cpid.
##     tcphsec: closure phase times in seconds
    
##     calc_closures() is a replacement for readcphs2():
##     cphs, cuv, tri, tcphsec  = readcphs2(cphs_file)
##     where tcphsec is 'tsec for closure phases'.
##     """
##     itimes, ibad = get_time_indices(tsec)
##     ntimes = len(itimes) - 1        # Number of same-time data segments > 3 bls
##     times = tsec[itimes]            # Unique times
##     bls = vstack((blid%PK2I, blid/PK2I)).T
##     antennas = unique(bls)
##     maxantnum = antennas[-1]  # The biggest antenna number (antennas are sorted)
##     triangles, triids = find_triangles(antennas)
##     ntriangles = len(triangles[:,0])
##     maxcps = len(ul[:,0])  # Maximal # of closure phases < uv points #
##     #maxcps = ntimes*ntriangles   # Maximal number of closure phases
##     cphs = zeros(maxcps, float)
##     tri = zeros((maxcps,3), int)
##     #cpix = zeros((maxcps,3), int) # Same as cpid, but not packed into one int
##     cpid = zeros((maxcps), int)
##     tcphsec = zeros((maxcps), float) # Closure phase times in seconds

##     iclp = 0
##     for itim in xrange(ntimes):   # Consider ith equal-time span
##         i0 = itimes[itim]         # Span start in data[]
##         i1 = itimes[itim+1]       # Span end   in data[]
##         if i1 in ibad: continue   #=================>>> Ignore bad bls[i0:i1]
##         nbl = i1 - i0            # Number of data rows in the span
##         bl = int_(bls[i0:i1,:]) # Convert baseline array to int
##         ant = unique(bl)         # Leave only sorted unique antenna numbers
##         nant = len(ant)          # Number of antenne employed in the i0-i1 span
##         # Find all triangles in the span
##         itri = combgen(nant, 3)   # Indices into ant to get triangles
##         ntri = size(itri,0)       # Number of triangles in this span

##         for j in xrange(ntri): # Over all triangles in this span
##             a1, a2, a3 = ant[itri[j,:]] # Triangle of antennae
##             # Find indices of the three baselines made of the antennae
##             ib1 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a2))[0][0]
##             ib2 = i0 + where((bl[:,0] == a1) & (bl[:,1] == a3))[0][0]
##             ib3 = i0 + where((bl[:,0] == a2) & (bl[:,1] == a3))[0][0]
##             # Phases in 
##             ph1 = phase[ib1]
##             ph2 = phase[ib2]
##             ph3 = phase[ib3]


##             if ph1 != 0. and ph2 != 0. and ph3 != 0.:
##                 ph = ph1 + ph2 - ph3 
##                 cphs[iclp] = arctan2(sin(ph), cos(ph))
##                 #tri[iclp,:] = a1, a2, a3
##                 tcphsec[iclp] = times[itim]
##                 # Get unique closure phase ID
##                 # ib1, ib2, ib3 are indices into ul, vl, amp, and phase
##                 cpid[iclp] = ib1 + 1024*(ib2 + 1024*ib3)
##                 #cpix[iclp,:] = ib1, ib2, ib3
##                 iclp = iclp + 1


##     cphs = cphs[:iclp]
##     #cpix = cpix[:iclp,:] # Same as cpid, but not packed into one int32
##     cpid = cpid[:iclp]
##     tcphsec = tcphsec[:iclp]

##     return cphs, cpid, tcphsec        # End of calc_closures2()



def count_lines(filename, numcount=None):
    """
    Returns number of lines in text file. If run under unix/linux
    or any other POSIX-type system, it uses the fastest utility wc:
    wc -l lilename
    Otherwise Python I/O is used.
    The numcount is a number of numbers on the line in integer,
    floating point, or exponential format required for the line
    to be counted as valid. The numbers can be separated by
    arbitrary characters, say
    2009-10-17 21:15:56.21 3.62 03-08 is
    parsed as 9 numbers:
    2009, 10, 17, 21, 15, 56.21 3.62 03, 08
    """
    if os.name == 'posix' and numcount == None: # The fastest way:
        lines = int(os.popen('wc -l ' + filename).read().split()[0])
        return lines
    # If not POSIX system, count using native python means:
    elif os.name != 'posix' and numcount == None:
        n = 1
        f = open(filename, 'r')
        lines = 0
        for line in f:
            lines += 1
        f.close()
        return lines
    elif numcount != None:
        # Regex pattern to match any numeric string, be it integer or float
        p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|'\
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+');
        # Count only the lines with exactly numcount numbers
        f = open(filename, 'r')
        nrows = 0
        while True:
            row = f.readline()
            if row == '': break  #=============== Exit upon EOF ============>>>
            nums = p.findall(row)
            if len(nums) == numcount:
                nrows += 1
        f.close()
        return nrows
    return -1  # In what case?..



def readamp(ulam, vlam, amp, phase, data_file, nlines):
    """
    Read amplitude data file into arrays ulam, vlam, amp, phase
    Returns number of the data rows
    ## here is a snippet of the time, u, v, amp, phase file
    ##
    ## Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    ## Channel         Visibility (amp, phase)      Weight   Sigma
    ##  2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  01-04
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2266654.77   5980230.93   3444330.20  01-05
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    1684941.63   2808918.82   7720608.60  01-06
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    2276757.84   2171092.41   7577581.04  01-07
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    fp = open(data_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = array(nums)     # Array of strings; just to ease indexing
        u, v, amp[i], phase[i], weight = float_(nums[[5,6,11,12,13]])

        if weight != 0.0:
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            phase[i] = radians(phase[i])
        i = i + 1 
    fp.close()
    return i


def readamp1(amp_data_file):
    """
    Read amplitude data file into arrays ulam, vlam, amp, phase
    ## here is a snippet of the time, u, v, amp, phase file
    ##
    ## Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    ## Channel         Visibility (amp, phase)      Weight   Sigma
    ##  2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  01-04
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2266654.77   5980230.93   3444330.20  01-05
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    1684941.63   2808918.82   7720608.60  01-06
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    2276757.84   2171092.41   7577581.04  01-07
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    
    Combines count_lines() and readamp().
    Returns ulam, vlam, amp and phase arrays
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    n = count_lines(amp_data_file)
    
    amp =   zeros(n, float)
    phase = zeros(n, float)
    ulam =  zeros(n, float)
    vlam =  zeros(n, float)

    fp = open(amp_data_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = array(nums)     # Array of strings; just to ease indexing
        u, v, amp[i], phase[i], weight = float_(nums[[5,6,11,12,13]])

        if weight != 0.0:
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            phase[i] = radians(phase[i])
        i = i + 1
        
    fp.close()
    
    if i == 0:
        print "\namp_data_file has no 15-number lines; wrong file"
        sys.exit(1);
      
    ulam.resize(i)
    vlam.resize(i)
    amp.resize(i)
    phase.resize(i)
    
    return ulam, vlam, amp, phase




def readamp2(amp_data_file):
    """
    Read amplitude data file into arrays ulam, vlam, amp, phase, tsec,
    where tsec is time in seconds since beginning of *Epoch*, and
    bl - baseline IDs (to get antenna #s: ant1 = bl%PK2I, ant2 = bl/PK2I)

    ## here is a snippet of the time, u, v, amp, phase file
    ##
    ## Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    ## Channel         Visibility (amp, phase)      Weight   Sigma
    ##  2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  01-04
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2266654.77   5980230.93   3444330.20  01-05
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    1684941.63   2808918.82   7720608.60  01-06
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    2276757.84   2171092.41   7577581.04  01-07
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    
    Combines count_lines() and readamp().
    Returns ulam, vlam, amp, phase, tsec, bl
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    n = count_lines(amp_data_file)
    
    amp =   zeros(n, float)
    phase = zeros(n, float)
    ulam =  zeros(n, float)
    vlam =  zeros(n, float)
    tsec =  zeros(n, float)
    #ant =   zeros((n,2), int)
    bl =    zeros(n, int)
    
    fp = open(amp_data_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = array(nums)     # Array of strings; just to ease indexing
        u, v, amp[i], phase[i], weight = float_(nums[[5,6,11,12,13]])

        if weight != 0.0:
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            phase[i] = radians(phase[i])
            # row[:20] is like 2009:095:02:40:00.00 or so
            ## tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
            ##           float(row[18:20])
            tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) # + \
            if row[18:20].isdigit():
                tsec[i] = tsec[i] + 0.01*np.float32(row[18:20])
            a1 =  int(nums[8])
            a2 = -int(nums[9])
            #print a1, a2
            bl[i] = a1 + PK2I*a2     # Get unique baseline ID
        i = i + 1
        
    fp.close()
    
    if i == 0:
        print "\namp_data_file has no 15-number lines; wrong file"
        sys.exit(1);
      
    ulam.resize(i)
    vlam.resize(i)
    amp.resize(i)
    phase.resize(i)
    tsec.resize(i)
    bl.resize(i)
    #ant.resize((i,2))
    #a = unique(ant)         # Flatten ant and leave only unique baseline IDs
    #maxblnum = len(a)       # Number of unique baseline numbers
    
    return ulam, vlam, amp, phase, tsec, bl



def readamp3(amp_data_file):
    """
    Read amplitude data file into arrays ulam, vlam, amp, phase, tsec,
    where tsec is time in seconds since beginning of *Epoch*, and
    bl - baseline IDs (to get antenna #s: ant1 = bl%PK2I, ant2 = bl/PK2I)

    ## here is a snippet of the time, u, v, amp, phase file
    ##
    ## Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    ## Channel         Visibility (amp, phase)      Weight   Sigma
    ##  2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  01-04
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2266654.77   5980230.93   3444330.20  01-05
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    1684941.63   2808918.82   7720608.60  01-06
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    2276757.84   2171092.41   7577581.04  01-07
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    
    Combines count_lines() and readamp().
    Returns ulam, vlam, amp, phase, sigma, tsec, bl 
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    n = count_lines(amp_data_file)
    
    amp =   zeros(n, float)
    phase = zeros(n, float)
    sigma = zeros(n, float)
    ulam =  zeros(n, float)
    vlam =  zeros(n, float)
    tsec =  zeros(n, float)
    #ant =   zeros((n,2), int)
    bl =    zeros(n, int)
    
    fp = open(amp_data_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = array(nums)     # Array of strings; just to ease indexing
        u, v, amp[i], phase[i], weight, sigma[i] = \
           float_(nums[[5,6,11,12,13,14]])

        if weight != 0.0:
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            phase[i] = radians(phase[i])
            # row[:20] is like 2009:095:02:40:00.00 or so
            #tsec[i] = mktime(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
            ## tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
            ##           float(row[18:20])
            tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) # + \
            if row[18:20].isdigit():
                tsec[i] = tsec[i] + 0.01*np.float32(row[18:20])
            a1 =  int(nums[8])
            a2 = -int(nums[9]) 
            bl[i] = a1 + PK2I*a2     # Get unique baseline ID
        i = i + 1
        
    fp.close()
    
    if i == 0:
        print "\namp_data_file has no 15-number lines; wrong file"
        sys.exit(1);
      
    ulam.resize(i)
    vlam.resize(i)
    amp.resize(i)
    phase.resize(i)
    sigma.resize(i)
    tsec.resize(i)
    bl.resize(i)
    #ant.resize((i,2))
    #a = unique(ant)         # Flatten ant and leave only unique baseline IDs
    #maxblnum = len(a)       # Number of unique baseline numbers
    
    return ulam, vlam, amp, phase, sigma, tsec, bl  # end of def readamp3()




def readamp4(amp_data_file):
    """
    Read amplitude data file into arrays ulam, vlam, amp, phase, tsec,
    where tsec is time in seconds since beginning of *Epoch*, and
    bl - baseline IDs (to get antenna #s: ant1 = bl%PK2I, ant2 = bl/PK2I)

    ## here is a snippet of the time, u, v, amp, phase file
    ##
    ## Scan Start (UT)            U(klam)      V(klam)      W(klam)  Baseline 
    ## Channel         Visibility (amp, phase)      Weight   Sigma
    ##  2008:263:05:50:40.12   -1969263.57    496717.85   2920931.16  01-02     
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -1529174.41      5210.19   2693549.21  01-03
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2687487.50   1809532.77   3124965.72  01-04
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12   -2266654.77   5980230.93   3444330.20  01-05
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    1684941.63   2808918.82   7720608.60  01-06
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    ##  2008:263:05:50:40.12    2276757.84   2171092.41   7577581.04  01-07
    ##    00        (     0.000000,     0.000000)     0.00    0.00000
    
    Combines count_lines() and readamp().
    Returns ulam, vlam, wlam, amp, phase, chan, tsec, weight, sigma, bl
    """
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    
    n = count_lines(amp_data_file)
    
    amp =   zeros(n, float)
    phase = zeros(n, float)
    sigma = zeros(n, float)
    ulam =  zeros(n, float)
    vlam =  zeros(n, float)
    wlam =  zeros(n, float)
    weight = zeros(n, float)
    tsec =  zeros(n, float)
    chan =  zeros(n, int)
    bl =    zeros(n, int)
    
    fp = open(amp_data_file, "r");

    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break
        nums = p.findall(row)  # Fish out all the numbers from row into nums 
        if len(nums) != 15:
            continue #================ non-data line; do nothing =====>>>
      
        nums = array(nums)     # Array of strings; just to ease indexing
        u, v, w, chan[i], amp[i], phase[i], weig, sigma[i] = \
           float_(nums[[5,6,7,10,11,12,13,14]])

        if weig != 0.0:
            weight[i] = weig
            ulam[i] = 1e-3*u    # From kilolambda to Megalambda
            vlam[i] = 1e-3*v    # From kilolambda to Megalambda
            wlam[i] = 1e-3*w    # From kilolambda to Megalambda
            phase[i] = radians(phase[i])
            # row[:20] is like 2009:095:02:40:00.00 or so
            #tsec[i] = mktime(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
            ## tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
            ##           float(row[18:20])
            tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
                      float(row[18:20])
            a1 =  int(nums[8])
            a2 = -int(nums[9]) 
            bl[i] = a1 + PK2I*a2     # Get unique baseline ID
        i = i + 1
        
    fp.close()
    
    if i == 0:
        #print "\namp_data_file has no 15-number lines; wrong file"
        #sys.exit(1);
        #
        # Signal by returning empty arrays
        #
        ulam  = array([], dtype=int)
        vlam  = array([], dtype=int)
        wlam  = array([], dtype=int)
        amp  = array([], dtype=int)
        phase  = array([], dtype=int)
        chan  = array([], dtype=int)
        tsec  = array([], dtype=int)
        weight  = array([], dtype=int)
        sigma  = array([], dtype=int)
        bl  = array([], dtype=int)
        return ulam, vlam, wlam, amp, phase, chan, tsec, weight, sigma, bl
      
    ulam.resize(i)
    vlam.resize(i)
    wlam.resize(i)
    amp.resize(i)
    phase.resize(i)
    sigma.resize(i)
    tsec.resize(i)
    bl.resize(i)
    #ant.resize((i,2))
    #a = unique(ant)         # Flatten ant and leave only unique baseline IDs
    #maxblnum = len(a)       # Number of unique baseline numbers
    
    return ulam, vlam, wlam, amp, phase, chan, tsec, weight, sigma, bl
                        # end of def readamp4()



def readcphs(tcode, cphs, gha, cuv, cphs_data_file, nlines):
    """
    Read phase closure data file into tcode, cphs, gha, and cuv   
    Returns number of the data rows
    Here is a snippet of the closure phase file:
    
    (For triangle 1-2-3, Amp.1 is on baseline 1-2, Amp.2 on 1-3, etc.)
    Time                  Triangle  Cl. Phase        U1            V1
    A1       P1         U2            V2           A2       P2           U3
    V3           A3       P3
    (UT)                 (t1-t2-t3)  (Degrees)    (t1-t2)
    (t1-t3)                                         (t2-t3)
    2009:095:02:40:00.00  06-07-08   -24.7031    -740589.74    -410253.99
    2.160  -24.6219   -2342366.08    8469970.91     0.102  -56.9739
    -1601776.33    8880224.90     0.074  -56.8927
    2009:095:03:00:00.00  06-07-08   -26.9507    -733889.02    -378950.08
    2.176  -24.3681   -2054232.54    8563312.86     0.098  -53.7360
    -1320343.51    8942262.94     0.070  -51.1534
    2009:095:03:20:00.00  06-07-08   -29.0803    -721572.36    -348049.92
    2.193  -23.9052   -1750379.39    8644086.65     0.094  -49.2770
    -1028807.03    8992136.56     0.066  -44.1019

    Data arrays filled:
    tcode[m,3], cphs[m], gha[m], cuv[m,6].
    """
    sgra_ra_hr = 17.761111; # sgra_ra in hours 
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
    fp = open(cphs_data_file, "r");
    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break            #=============== Exit upon EOF ============>>>
        nums = p.findall(row)  # Fish out all the numbers into numstr list 
        if len(nums) != 21:
            continue           #========= non-data line; do nothing =====>>>
        nums = array(nums)     # Array of strings; just to ease indexing
        amp1, amp2, amp3 = float_(nums[[11, 15, 18]])
        if ((amp1 == 0.0) and (amp2 == 0.0) and (amp3 == 0.0)):
            print 'nums = ', nums 
            continue
        year, doy, hr, mn = int_(nums[:4])
        sec = float(nums[4])
        # now use 2008 relation for GMST, GHA = GMST - RA(SgrA*)
        uthr = hr + mn/60.0 + sec/3600.0
        gha[i] = 6.6029168 + 0.0657098244*doy + 1.00273791*uthr - sgra_ra_hr;
        if gha[i] <  0.0: gha[i] += 24
        if gha[i] >= 0.0: gha[i] -= 24
        gha[i] *= pi/12.0  # convert hours to radians
        tcode[i,:] = abs(int_(nums[5:8])) # Triplet-triangle

        cphs[i] = radians(float(nums[8]))
        #u1, v1, u2, v2, u3, v3 = float_(nums[[9, 10, 13, 14, 17, 18]])
        # From kilolambda to Megalambda
        cuv[i,0:2] =  0.001*float_(nums[9:11])
        cuv[i,2:4] =  0.001*float_(nums[13:15]) # Fixed ERROR: 13:15 <-> 17:19
        cuv[i,4:6] = -0.001*float_(nums[17:19]) # Fixed ERROR: 13:15 <-> 17:19

        i = i + 1
          
    fp.close()
    return i



def readcphs1(cphs_data_file):
    """
    Read phase closure data file into tcode, cphs, gha, and cuv   
    Here is a snippet of the closure phase file:
    
    (For triangle 1-2-3, Amp.1 is on baseline 1-2, Amp.2 on 1-3, etc.)
    Time                  Triangle  Cl. Phase        U1            V1
    A1       P1         U2            V2           A2       P2           U3
    V3           A3       P3
    (UT)                 (t1-t2-t3)  (Degrees)    (t1-t2)
    (t1-t3)                                         (t2-t3)
    2009:095:02:40:00.00  06-07-08   -24.7031    -740589.74    -410253.99
    2.160  -24.6219   -2342366.08    8469970.91     0.102  -56.9739
    -1601776.33    8880224.90     0.074  -56.8927
    2009:095:03:00:00.00  06-07-08   -26.9507    -733889.02    -378950.08
    2.176  -24.3681   -2054232.54    8563312.86     0.098  -53.7360
    -1320343.51    8942262.94     0.070  -51.1534
    2009:095:03:20:00.00  06-07-08   -29.0803    -721572.36    -348049.92
    2.193  -23.9052   -1750379.39    8644086.65     0.094  -49.2770
    -1028807.03    8992136.56     0.066  -44.1019

    Data arrays filled:
    tcode[m,3], cphs[m], gha[m], cuv[m,6].
    
    Combines count_lines() and readcphs().
    Returns cphs, gha, cuv, and tcode  arrays
    """
    sgra_ra_hr = 17.761111; # sgra_ra in hours
    
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')

    m = count_lines(cphs_data_file)
    
    cphs =  zeros(m, float)
    cuv =   zeros((m,6), float)
    gha =   zeros(m, float)
    tcode = zeros((m,3), int)
    
    fp = open(cphs_data_file, "r");
    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break              #=============== Exit upon EOF ============>>>
        nums = p.findall(row)  # Fish out all the numbers into numstr list 
        if len(nums) != 21:
            continue           #========= non-data line; do nothing =====>>>
        nums = array(nums)     # Array of strings; just to ease indexing
        amp1, amp2, amp3 = float_(nums[[11, 15, 18]])
        if ((amp1 == 0.0) and (amp2 == 0.0) and (amp3 == 0.0)):
            print 'nums = ', nums 
            continue
        year, doy, hr, mn = int_(nums[:4])
        sec = float(nums[4])
        # now use 2008 relation for GMST, GHA = GMST - RA(SgrA*)
        uthr = hr + mn/60.0 + sec/3600.0
        gha[i] = 6.6029168 + 0.0657098244*doy + 1.00273791*uthr - sgra_ra_hr;
        if gha[i] <  0.0: gha[i] += 24
        if gha[i] >= 0.0: gha[i] -= 24
        gha[i] *= pi/12.0  # convert hours to radians
        tcode[i,:] = abs(int_(nums[5:8])) # Triplet-triangle

        cphs[i] = radians(float(nums[8]))
        #u1, v1, u2, v2, u3, v3 = float_(nums[[9, 10, 13, 14, 17, 18]])
        # From kilolambda to Megalambda
        cuv[i,0:2] =  0.001*float_(nums[9:11])
        cuv[i,2:4] =  0.001*float_(nums[13:15]) # Fixed ERROR: 13:15 <-> 17:19
        cuv[i,4:6] = -0.001*float_(nums[17:19]) # Fixed ERROR: 13:15 <-> 17:19
        i = i + 1
          
    fp.close()

    if m == 0:
        print "\ncphase_data_file has no 21-number lines; wrong file"
        sys.exit(1);

    cphs.resize(i)
    cuv.resize((i,6))
    gha.resize(i)
    tcode.resize((i,3))

    return cphs, cuv, gha, tcode




def readcphs2(cphs_data_file):
    """
    Read phase closure data file into tcode, cphs, gha, and cuv, tsec, and
    where tsec is time in seconds since beginning of *Epoch*.
    
    Here is a snippet of the closure phase file:
    
    (For triangle 1-2-3, Amp.1 is on baseline 1-2, Amp.2 on 1-3, etc.)
    Time                  Triangle  Cl. Phase        U1            V1
    A1       P1         U2            V2           A2       P2           U3
    V3           A3       P3
    (UT)                 (t1-t2-t3)  (Degrees)    (t1-t2)
    (t1-t3)                                         (t2-t3)
    2009:095:02:40:00.00  06-07-08   -24.7031    -740589.74    -410253.99
    2.160  -24.6219   -2342366.08    8469970.91     0.102  -56.9739
    -1601776.33    8880224.90     0.074  -56.8927
    2009:095:03:00:00.00  06-07-08   -26.9507    -733889.02    -378950.08
    2.176  -24.3681   -2054232.54    8563312.86     0.098  -53.7360
    -1320343.51    8942262.94     0.070  -51.1534
    2009:095:03:20:00.00  06-07-08   -29.0803    -721572.36    -348049.92
    2.193  -23.9052   -1750379.39    8644086.65     0.094  -49.2770
    -1028807.03    8992136.56     0.066  -44.1019

    Data arrays filled:
    tcode[m,3], cphs[m], gha[m], cuv[m,6].
    
    Combines count_lines() and readcphs().
    Returns cphs, gha, cuv, and tcode  arrays
    """
    sgra_ra_hr = 17.761111; # sgra_ra in hours
    
    # Regex pattern to match any numeric string, be it integer or float
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                       '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')

    m = count_lines(cphs_data_file)
    
    cphs =  zeros(m, float)
    cuv =   zeros((m,6), float)
    gha =   zeros(m, float)
    tcode = zeros((m,3), int)
    tsec =  zeros(m, float)
    
    fp = open(cphs_data_file, "r");
    i = 0
    while True:
        row = fp.readline()
        if row == '':
            break              #=============== Exit upon EOF ============>>>
        nums = p.findall(row)  # Fish out all the numbers into numstr list 
        if len(nums) != 21:
            continue           #========= non-data line; do nothing =====>>>
        nums = array(nums)     # Array of strings; just to ease indexing
        amp1, amp2, amp3 = float_(nums[[11, 15, 18]])
        if ((amp1 == 0.0) and (amp2 == 0.0) and (amp3 == 0.0)):
            print 'nums = ', nums 
            continue
        year, doy, hr, mn = int_(nums[:4])
        sec = float(nums[4])
        # now use 2008 relation for GMST, GHA = GMST - RA(SgrA*)
        uthr = hr + mn/60.0 + sec/3600.0
        gha[i] = 6.6029168 + 0.0657098244*doy + 1.00273791*uthr - sgra_ra_hr;
        if gha[i] <  0.0: gha[i] += 24
        if gha[i] >= 0.0: gha[i] -= 24
        gha[i] *= pi/12.0  # convert hours to radians
        tcode[i,:] = abs(int_(nums[5:8])) # Triplet-triangle

        cphs[i] = radians(float(nums[8]))
        tsec[i] = timegm(strptime(row[:17], '%Y:%j:%H:%M:%S')) + \
                  float(row[18:20])
        #u1, v1, u2, v2, u3, v3 = float_(nums[[9, 10, 13, 14, 17, 18]])
        # From kilolambda to Megalambda
        cuv[i,0:2] =  0.001*float_(nums[9:11])
        cuv[i,2:4] =  0.001*float_(nums[13:15]) # Fixed ERROR: 13:15 <-> 17:19
        cuv[i,4:6] = -0.001*float_(nums[17:19]) # To make U1+U2+U3=V1+V2+V3=0
        i = i + 1
          
    fp.close()

    if m == 0:
        print "\ncphase_data_file has no 21-number lines; wrong file"
        sys.exit(1);

    cphs.resize(i)
    tsec.resize(i)
    cuv.resize((i,6))
    gha.resize(i)
    tcode.resize((i,3))

    return cphs, cuv, gha, tcode, tsec


def xringaus(ulam, vlam, vparam):
    '''
    This model is a combination of eccentric ring and Gaussian.

    Parameters ulam, vlam may be either elementary variables or arrays
    of any dimensionality. The only restriction is that they must have
    the same structure. 
    For example, sometimes it is convenient to calculate the model on
    a regular grid, so ulam and vlam can be the grids generated by
    meshgrid() function.
    Zsp, Re, Ri, fade, d, gsx, gsy, gq, th are elementary variables.
    They are the model parameters.
    
    Input parameters:
    ulam, vlam: u and v coordinates on the visibility plane in Gigalambdas
    vparam: a sequence of the following parameters:
    Zsp: zero spacing in any units (Jy or such)
    Re: external  ring radius in microarcseconds 
    rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ecc: eccentricity of inner circle center from that of outer; in [-1..1]
    fade: [0..1], "noncontrast" of the ring, 1-uniform, 0-from zero to maximum
    gax: Gaussian main axis (tangential to the ring), expressed in Re
    aq: axes quotient, aq = gsx/gsy. 0 < aq < 1
    gq: quota of Gaussian visibility against ring. 0-ring only, 1-Gaussian only
    th: angle of rotation in radians

    Output:
    Returns a complex array with the model visibility
    values at the (ulam,vlam) points

    Some internal variables:
    gsx, gsy: sizes (ie FWHMs) of the Gaussuan along X and Y axes
    d: displacement of inner circle center wrt outer
    
    '''
    Zsp, Re, rq, ecc, fade, gax, aq, gq, th = vparam

    #
    # Convert radii and Gaussian axes from microarcseconds into nanoradians
    # The calculations are performed in the units of Gigalambda which
    # corresponds to the RA-Dec dimensions in nanoradians
    # Re/3600.: uas -> udeg; radians(Re/3600.): udeg -> urad
    # radians(Re/3600.)*1e3: urad -> nrad
    #
    
    Re = radians(Re/3600.)*1e3  # microarcseconds -> nanoradians
    Ri = rq*Re         # 0 < rq < 1 is radii quotient
    d = ecc*(Re - Ri)  # nrad, displacement of inner circle center wrt outer
    gsy = gax*Re       # nrad. The Gaussian's axis tangential to the ring
    gsx = aq*gsy       # nrad. The Gaussian's radial axis, normal to the ring

    #
    # Rotate coordinates by th angle
    #
    U =  ulam*cos(th) + vlam*sin(th)
    V = -ulam*sin(th) + vlam*cos(th)

    #
    # Eccentric ring
    #
    pi4 = 4.0*pi
    f = fade
    # f = 1. - fade   # I think this was wrong: more "fade" means more uniform
    # h is the ring "height" at which its Zsp = 1.
    h = (2./pi)/((Re**2 - Ri**2*(1. + d/Re))*f + (Re**2 - Ri**2*(1. - d/Re)))  

    #print 'f = ', f, ', d = ', d
    #print 'h = ', h, ', Re = ', Re, ', Ri = ', Ri, 'pi = ', pi
    
    
    h0 = f*h
    rho = sqrt(U**2 + V**2);  # UV-space radius-vectors (baselines) in Glam

    #print 'h = %g, h0 = %g' % (h, h0)

    #
    # Replace rho = 0 by a fake rho = 1 to avoid the divide by zero.
    # At this point the limits at rho -> 0 will be calculated further.
    #
    iz = where(rho < 1e-12)
    rho[iz] = 1.

    arg = 2.*pi*rho           # in Glambdas
    
    ree = .5*(h+h0)*Re*j1(arg*Re)/rho
    ime = (h-h0)/pi4*(pi*Re*(j0(arg*Re) - jn(2,arg*Re))/rho**2 - \
                               j1(arg*Re)/(rho**3))*U
    if Ri <> 0:
        rei = .5*(h+h0 - d*(h-h0)/Re)*Ri*j1(arg*Ri)/rho
        imi = (h-h0)/pi4*(Ri/Re)*(pi*Ri*(j0(arg*Ri) - jn(2,arg*Ri))/rho**2 - \
                               j1(arg*Ri)/(rho**3))*U
    #else:
    #    rei = 0.
    #    imi = 0.
    
    #
    # Calculate the rho -> 0 limits for ree, ime, rei, and imi
    #
    ree[iz] = .5*(h+h0)*pi*Re**2
    ime[iz] = -.25*pi**2*(h-h0)*Re**3*U[iz]
    if Ri <> 0:
        rei[iz] = .5*(h+h0 - d*(h-h0)/Re)*pi*Ri**2
        imi[iz] = -.25*pi**2*(h-h0)*(Ri**4/Re)*U[iz]
    
    Fex = ree + 1j*ime  # External slanted pillbox 
    if Ri <> 0:
        Dis = exp(1j*2.*pi*d*U)    # Displacement factor for inner ring
        Fin = Dis*(rei + 1j*imi) # Inner slanted pillbox shifted by d
        Fr = Fex - Fin
    else:
        Fr = Fex
    
    ## re =  (ree - D*rei) # "minus" rotates the resulting image by 180 degrees
    ## im = -(ime - D*imi) # "minus" rotates the resulting image by 180 degrees
    ## Fr = re + 1j*im
    
    #
    # Shifted gaussian
    #
    #xg = .5*(Re + Ri - d); # Gaussian centered at the middle of the thick part
    if gax == 0. or aq == 0. or gq == 0.:
        return Zsp*Fr
    else:
        gshift = Ri - d; # Gaussian centered at the inner edge of inner ring
        gcoef = 2.*(pi/(2.*sqrt(2.*log(2.))))**2 # Convert FWHM^2 to stdev^2 
        Gamp = exp(-gcoef*((U*gsx)**2 + (V*gsy)**2))
        Gph = -2.*pi*gshift*U

        Fg = Gamp*(cos(Gph) + 1j*sin(Gph)) # Fourier image of displaced Gaussian

        F = (1.-gq)*Fr + gq*Fg # Mix ring and Gaussian in the gq proportion

    return Zsp*F  # End of def xringaus()




def xringaus2(ulam, vlam, vparam):
    '''
    This model is a combination of eccentric ring and Gaussian, where
    Gaussian can take any position, (Rg,alpha), and orientation, beta. The
    direction of the brightness slope, theta, is decoupled from the
    direction of  the inner circle, eta. 

    Parameters ulam, vlam may be either elementary variables or arrays
    of any dimensionality. The only restriction is that they must have
    the same structure. 
    For example, sometimes it is convenient to calculate the model on
    a regular grid, so ulam and vlam can be the grids generated by
    meshgrid() function.
    Zsp, Re, Ri, fade, d, gsx, gsy, gq, th are elementary variables.
    They are the model parameters.
    
    Input parameters:
    ulam, vlam: u and v coordinates on the visibility plane in Gigalambdas
    vparam: a sequence of the following parameters:
    Zsp: zero spacing in any units (Jy or such)
    Re: external  ring radius in microarcseconds 
    rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ecc: eccentricity of inner circle center from that of outer; in [-1..1]
    fade: [0..1], "noncontrast" of the ring, 1-uniform, 0-from zero to maximum
    gr: Rg/Re, Rg - distance of Gaussian center from the external ring center
    gax: Gaussian main axis, expressed in Re
    aq: axes quotient, aq = gsx/gsy. 0 < aq < 1
    gq: fraction of Gaussian visibility against ring.
        0-ring only, 1-Gaussian only
    alpha: angle of the Gaussian center orientation
    beta: angle of the Gaussian rotation
    eta: angle of internal circular hole orientation in radians
    th: angle of slope orientation in radians

    Output:
    Returns a complex array with the model visibility
    values at the (ulam,vlam) points

    Some internal variables:
    gsx, gsy: sizes (ie FWHMs) of the Gaussuan along X and Y axes
    d: displacement of inner circle center wrt outer
    
    '''
    Zsp, Re, rq, ecc, fade, gr, gax, aq, gq, alpha, beta, eta, th = vparam

    #
    # Convert radii and Gaussian axes from microarcseconds into nanoradians
    # The calculations are performed in the units of Gigalambda which
    # corresponds to the RA-Dec dimensions in nanoradians
    # Re/3600.: uas -> udeg; radians(Re/3600.): udeg -> urad
    # radians(Re/3600.)*1e3: urad -> nrad
    #
    Re = radians(Re/3600.)*1e3  # microarcseconds -> nanoradians
    Ri = rq*Re         # 0 < rq < 1 is radii quotient
    #
    # First calculate the Gaussian
    #
    if gax != 0. and aq != 0. and gq != 0.:
        gsy = gax*Re   # nrad. The Gaussian's axis tangential to the ring
        gsx = aq*gsy   # nrad. The Gaussian's radial axis, normal to the ring
        #
        # Rotate coordinates by beta angle (to rotate Gaussian)
        #
        U =  ulam*cos(beta) + vlam*sin(beta)
        V = -ulam*sin(beta) + vlam*cos(beta)
        gshift = gr*Re*(cos(alpha)*ulam + sin(alpha)*vlam)   # Gaussian's shift
        fwhm2std = 2.*(pi/(2.*sqrt(2.*log(2.))))**2 # Convert FWHM^2 to 1/std^2 
        Gamp = exp(-fwhm2std*((U*gsx)**2 + (V*gsy)**2))
        Gph = -2.*pi*gshift
        Fg = Gamp*(cos(Gph) + 1j*sin(Gph)) # Fourier image of displaced Gaussian
        print 'Fg = Gamp*(cos(Gph) + 1j*sin(Gph)) # displaced Gaussian'
        print 'Zsp=%g, Re=%g, gax=%g, aq=%g, gq=%g, alpha=%g, beta=%g' % \
              (Zsp, Re, gax, aq, gq, alpha, beta)
    if gq == 1.:   # Gaussian only
        return Zsp*Fg               #  ================================== >>>
    
    D = ecc*(Re - Ri)  # nrad, displacement of inner circle center wrt outer
    d =  D*cos(eta - th) # D projection on the slope gradient
    d1 = D*sin(eta - th) # D projection on the slope gradient perpendicular
    #gsy = gax*Re   # nrad. The Gaussian's axis tangential to the ring
    #gsx = aq*gsy   # nrad. The Gaussian's radial axis, normal to the ring

    #
    # Rotate coordinates by th angle
    #
    U =  ulam*cos(th) + vlam*sin(th)
    V = -ulam*sin(th) + vlam*cos(th)

    #
    # Eccentric ring
    #
    pi4 = 4.0*pi
    f = fade
    # h is the ring "height" at which its Zsp = 1.
    h = (2./pi)/((Re**2 - Ri**2*(1. + d/Re))*f + (Re**2 - Ri**2*(1. - d/Re)))  

    #print 'f = ', f, ', d = ', d
    #print 'h = ', h, ', Re = ', Re, ', Ri = ', Ri, 'pi = ', pi
    
    
    h0 = f*h
    rho = sqrt(U**2 + V**2);  # UV-space radius-vectors (baselines) in Glam

    #print 'h = %g, h0 = %g' % (h, h0)

    #
    # Replace rho = 0 by a fake rho = 1 to avoid the divide by zero.
    # At this point the limits at rho -> 0 will be calculated further.
    #
    iz = where(rho < 1e-12)
    rho[iz] = 1.

    arg = 2.*pi*rho           # in Glambdas
    
    ree = .5*(h+h0)*Re*j1(arg*Re)/rho
    ime = (h-h0)/pi4*(pi*Re*(j0(arg*Re) - jn(2,arg*Re))/rho**2 - \
                               j1(arg*Re)/(rho**3))*U
    if Ri <> 0:
        rei = .5*(h+h0 - d*(h-h0)/Re)*Ri*j1(arg*Ri)/rho
        imi = (h-h0)/pi4*(Ri/Re)*(pi*Ri*(j0(arg*Ri) - jn(2,arg*Ri))/rho**2 - \
                               j1(arg*Ri)/(rho**3))*U
    #
    # Calculate the rho -> 0 limits for ree, ime, rei, and imi
    #
    ree[iz] = .5*(h+h0)*pi*Re**2
    ime[iz] = -.25*pi**2*(h-h0)*Re**3*U[iz]
    if Ri <> 0:
        rei[iz] = .5*(h+h0 - d*(h-h0)/Re)*pi*Ri**2
        imi[iz] = -.25*pi**2*(h-h0)*(Ri**4/Re)*U[iz]

    #
    # Shift the inner ring
    #
    Fex =      ree + 1j*ime  # External slanted pillbox 
    if Ri <> 0:
        Dis = exp(1j*2.*pi*(d*U + d1*V))    # Displ. factor for inner ring
        Fin = Dis*(rei + 1j*imi) # Inner slanted pillbox shifted by d
        Fr = Fex - Fin
    else:
        Fr = Fex
    
    ## re =  (ree - D*rei) # "minus" rotates the resulting image by 180 degrees
    ## im = -(ime - D*imi) # "minus" rotates the resulting image by 180 degrees
    ## Fr = re + 1j*im
    
    #
    #Rotate coordinates by beta angle (to rotate Gaussian)
    #
    ## U =  ulam*cos(beta) + vlam*sin(beta)
    ## V = -ulam*sin(beta) + vlam*cos(beta)

    #
    # Shifted gaussian
    #
    #xg = .5*(Re + Ri - d); # Gaussian centered at the middle of the thick part
    if gax == 0. or aq == 0. or gq == 0.:
        return Zsp*Fr
    else:
        ## al = alpha
        ## gshift = gr*Re*(cos(al)*ulam + sin(al)*vlam)   # Gaussian's shift
        ## fwhm2std = 2.*(pi/(2.*sqrt(2.*log(2.))))**2 #Conv. FWHM^2 to 1/std^2 
        ## Gamp = exp(-fwhm2std*((U*gsx)**2 + (V*gsy)**2))
        ## Gph = -2.*pi*gshift

        ## Fg = Gamp*(cos(Gph) + 1j*sin(Gph)) # Fourier image of displ. Gaussian

        F = (1.-gq)*Fr + gq*Fg # Mix ring and Gaussian in the gq proportion

    return Zsp*F  # End of def xringaus2()



def uxring(ulam, vlam, vparam):
    '''
    This model is a uniform eccentric ring.

    Parameters ulam, vlam may be either elementary variables or arrays
    of any dimensionality. The only restriction is that they must have
    the same structure. 
    For example, sometimes it is convenient to calculate the model on
    a regular grid, so ulam and vlam can be the grids generated by
    meshgrid() function.
    Zsp, Re, Ri, fade, d, gsx, gsy, gq, th are elementary variables.
    They are the model parameters.
    
    Input parameters:
    ulam, vlam: u and v coordinates on the visibility plane in Gigalambdas
    vparam: a sequence of the following parameters:
    Zsp: zero spacing in any units (Jy or such)
    Re: external  ring radius in microarcseconds 
    rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ecc: eccentricity of inner circle center from that of outer; in [-1..1]
    th: angle of rotation in radians

    Output:
    Returns a complex array with the model visibility
    values at the (ulam,vlam) points

    Some internal variables:
    d: displacement of inner circle center wrt outer
    
    '''
    Zsp, Re, rq, ecc, th = vparam

    #
    # Convert radii and Gaussian axes from microarcseconds into nanoradians
    # The calculations are performed in the units of Gigalambda which
    # corresponds to the RA-Dec dimensions in nanoradians
    # Re/3600.: uas -> udeg; radians(Re/3600.): udeg -> urad
    # radians(Re/3600.)*1e3: urad -> nrad
    #
    
    Re = radians(Re/3600.)*1e3  # microarcseconds -> nanoradians
    Ri = rq*Re         # 0 < rq < 1 is radii quotient
    d = ecc*(Re - Ri)  # nrad, displacement of inner circle center wrt outer

    #
    # Rotate coordinates by th angle
    #
    U =  ulam*cos(th) + vlam*sin(th)
    V = -ulam*sin(th) + vlam*cos(th)

    #
    # Eccentric ring
    #
    pi4 = 4.0*pi
    h = 1./(pi*(Re**2 - Ri**2))

    rho = sqrt(U**2 + V**2);  # UV-space radius-vectors (baselines) in Glam

    #
    # Replace rho = 0 by a fake rho = 1 to avoid the divide by zero.
    # At this point the limits at rho -> 0 will be calculated further.
    #
    iz = where(rho < 1e-12)
    rho[iz] = 1.

    arg = 2.*pi*rho           # in Glambdas
    
    ree = h*Re*j1(arg*Re)/rho
    rei = h*Ri*j1(arg*Ri)/rho
    
    #
    # Calculate the rho -> 0 limits for ree, ime, rei, and imi
    #
    ree[iz] = h*pi*Re**2
    rei[iz] = h*(1. - d/Re)*pi*Ri**2
    
    Dis = exp(-1j*2.*pi*d*U)    # Displacement factor for inner ring
    Fex =     ree   # External pillbox 
    Fin = Dis*rei   # Inner pillbox shifted by d
    F = Fex - Fin
      
    return Zsp*F    # End of def uxring()



def xring_old(ulam, vlam, vparam):
    '''
    Eccentric ring model.

    Parameters ulam, vlam may be either elementary variables or arrays
    of any dimensionality. The only restriction is that they must have
    the same structure. 
    For example, sometimes it is convenient to calculate the model on
    a regular grid, so ulam and vlam can be the grids generated by
    meshgrid() function.
    Zsp, Re, Ri, fade, d, gsx, gsy, gq, th are elementary variables.
    They are the model parameters.
    
    Input parameters:
    ulam, vlam: u and v coordinates on the visibility plane in Gigalambdas
    vparam: a sequence of the following parameters:
    Zsp: zero spacing in any units (Jy or such)
    Re: external  ring radius in microarcseconds 
    th: angle of rotation in radians

    rq = 0.75;  ecc = 0.7
    
    Output:
    Returns a complex array with the model visibility
    values at the (ulam,vlam) points

    Some internal variables:
    d: displacement of inner circle center wrt outer
    rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ecc: eccentricity of inner circle center from that of outer; in [-1..1]
   
    '''
    Zsp, Re, th = vparam

    #
    # Convert radii and Gaussian axes from microarcseconds into nanoradians
    # The calculations are performed in the units of Gigalambda which
    # corresponds to the RA-Dec dimensions in nanoradians
    # Re/3600.: uas -> udeg; radians(Re/3600.): udeg -> urad
    # radians(Re/3600.)*1e3: urad -> nrad
    #
    rq = 0.75
    ecc = 0.7
    
    Re = radians(Re/3600.)*1e3  # microarcseconds -> nanoradians
    Ri = rq*Re         # 0 < rq < 1 is radii quotient
    d = ecc*(Re - Ri)  # nrad, displacement of inner circle center wrt outer

    #
    # Rotate coordinates by th angle
    #
    U =  ulam*cos(th) - vlam*sin(th)
    V =  ulam*sin(th) + vlam*cos(th)

    #
    # Eccentric ring
    #
    pi4 = 4.0*pi
    # h is the ring "height" at which its Zsp = 1.
    h = (2./pi)/(Re**2 - Ri**2*(1. - d/Re))
    rho = sqrt(U**2 + V**2);  # UV-space radius-vectors (baselines) in Glam

    #
    # Replace rho = 0 by a fake rho = 1 to avoid the divide by zero.
    # At this point the limits at rho -> 0 will be calculated further.
    #
    iz = where(rho < 1e-12)
    rho[iz] = 1.

    arg = 2.*pi*rho           # in Glambdas
    
    ree = .5*h*Re*j1(arg*Re)/rho
    ime = h/pi4*(pi*Re*(j0(arg*Re) - jn(2,arg*Re))/rho**2 - \
                               j1(arg*Re)/(rho**3))*U
    rei = .5*h*(1. - d/Re)*Ri*j1(arg*Ri)/rho
    imi = h/pi4*(Ri/Re)*(pi*Ri*(j0(arg*Ri) - jn(2,arg*Ri))/rho**2 - \
                               j1(arg*Ri)/(rho**3))*U
    #
    # Calculate the rho -> 0 limits for ree, ime, rei, and imi
    #
    ree[iz] = .5*h*pi*Re**2
    rei[iz] = .5*h*(1. - d/Re)*pi*Ri**2
    ime[iz] = -.25*pi**2*h*Re**3*U[iz]
    imi[iz] = -.25*pi**2*h*(Ri**4/Re)*U[iz]
    
    D = exp(-1j*2.*pi*d*U)    # Displacement factor for inner ring
    re =  (ree - D*rei) # "minus" rotates the resulting image by 180 degrees
    im = -(ime - D*imi) # "minus" rotates the resulting image by 180 degrees
    F = re + 1j*im

    return Zsp*F     #*exp(1j*2.*pi*V*0.0031075)




def xring(ulam, vlam, vparam):
    '''
    Eccentric ring model.

    Parameters ulam, vlam may be either elementary variables or arrays
    of any dimensionality. The only restriction is that they must have
    the same structure. 
    For example, sometimes it is convenient to calculate the model on
    a regular grid, so ulam and vlam can be the grids generated by
    meshgrid() function.
    Zsp, Re, Ri, fade, d, gsx, gsy, gq, th are elementary variables.
    They are the model parameters.
    
    Input parameters:
    ulam, vlam: u and v coordinates on the visibility plane in Gigalambdas
    vparam: a sequence of the following parameters:
    Zsp: zero spacing in any units (Jy or such)
    Re: external  ring radius in microarcseconds 
    th: angle of rotation in radians

    rq = 0.75;  ecc = 0.7
    
    Output:
    Returns a complex array with the model visibility
    values at the (ulam,vlam) points

    Some internal variables:
    d: displacement of inner circle center wrt outer
    rq: radius quotient, the internal radius, Ri = rq*Re. 0 < rq < 1
    ecc: eccentricity of inner circle center from that of outer; in [-1..1]
   
    '''
    Zsp, Re, th = vparam

    #
    # Convert radii and Gaussian axes from microarcseconds into nanoradians
    # The calculations are performed in the units of Gigalambda which
    # corresponds to the RA-Dec dimensions in nanoradians
    # Re/3600.: uas -> udeg; radians(Re/3600.): udeg -> urad
    # radians(Re/3600.)*1e3: urad -> nrad
    #
    rq = 0.75
    ecc = 0.7
    
    Re = radians(Re/3600.)*1e3  # microarcseconds -> nanoradians
    Ri = rq*Re         # 0 < rq < 1 is radii quotient
    d = ecc*(Re - Ri)  # nrad, displacement of inner circle center wrt outer

    #
    # Rotate coordinates by th angle
    #
    U =   ulam*cos(th) + vlam*sin(th)
    V =  -ulam*sin(th) + vlam*cos(th)

    #
    # Eccentric ring
    #
    pi4 = 4.0*pi
    # h is the ring "height" at which its Zsp = 1.
    h = (2./pi)/(Re**2 - Ri**2*(1. - d/Re))
    rho = sqrt(U**2 + V**2);  # UV-space radius-vectors (baselines) in Glam

    #
    # Replace rho = 0 by a fake rho = 1 to avoid the divide by zero.
    # At this point the limits at rho -> 0 will be calculated further.
    #
    iz = where(rho < 1e-12)
    rho[iz] = 1.

    arg = 2.*pi*rho           # in Glambdas
    
    ree = .5*h*Re*j1(arg*Re)/rho
    ime = h/pi4*(pi*Re*(j0(arg*Re) - jn(2,arg*Re))/rho**2 - \
                               j1(arg*Re)/(rho**3))*U
    rei = .5*h*(1. - d/Re)*Ri*j1(arg*Ri)/rho
    imi = h/pi4*(Ri/Re)*(pi*Ri*(j0(arg*Ri) - jn(2,arg*Ri))/rho**2 - \
                               j1(arg*Ri)/(rho**3))*U
    #
    # Calculate the rho -> 0 limits for ree, ime, rei, and imi
    #
    ree[iz] = .5*h*pi*Re**2
    rei[iz] = .5*h*(1. - d/Re)*pi*Ri**2
    ime[iz] = -.25*pi**2*h*Re**3*U[iz]
    imi[iz] = -.25*pi**2*h*(Ri**4/Re)*U[iz]
    
    Dis = exp(1j*2.*pi*d*U)  # Displacement factor for inner ring
    Fex =      ree + 1j*ime  # External slanted pillbox 
    Fin = Dis*(rei + 1j*imi) # Inner slanted pillbox shifted by d
    F = Fex - Fin

    return Zsp*F   



    
def gradring (ulam, vlam, zsp, gsize, radius, theta, re, im):
    """
    Gradient-ring model in the visibility space
    Parameters: zsp, gsize, radius, theta
    Returns re and im arrays
    Parameters:
    zsp: flux density of the ring
    hamp: half-amplitude
    radius: gradient ring radius 
    """
    
    # Note, factor of 4.848e-6 = 1/206265 converts arcseconds to radians
    as2r = 4.8481e-6;    # Mlambda*microarcsec to lambda*radians conversion
    r2ms = 206264.8062   # Number of arcseconds in 1 rad
    #as2r = 1.;    # Mlambda*microarcsec to lambda*radians conversion
    pi2 = 2.0*pi
    pi4 = 4.0*pi

    hzsp = 0.5*zsp   
    
    Rext = radius*2.4;
    Rint = radius*2;

    rho = sqrt(ulam**2 + vlam**2);  # in Mlambdas
    arg = 2.*pi*rho;                # in Mlambdas
    arge = arg*Rext*as2r
    argi = arg*Rint*as2r
    hr = hzsp/Rext              # Half-amplitude over R_external (bigger R)
    #rrho = rho*as2r          # baselines in lambda*radians

    uvdir = ulam*cos(theta) + vlam*sin(theta)

    ## xg = 1.1*radius*cos(theta);
    ## yg = 1.1*radius*sin(theta);
    rq2 = Rint**2/Rext**2
    k = 2./(1. - rq2)
    
    rho = sqrt(ulam**2 + vlam**2);  # UV-space radius 
    
    #D1 = k*zsp*j1(arge)/arge             #rrho;       # disk1 big
    #D2 = k*rq2*zsp*j1(argi)/argi       #rrho; # disk2 small

  
    ree = k*hzsp*j1(arge)/arge;        # External disk (e), real part
    rei = k*rq2*hzsp*j1(argi)/argi;    # Internal disk (i), imaginary part
    
    ime =     r2ms*(k*hr/pi2)*(0.5*(j0(arge) - jn(2,arge))/rho**2 - \
                               j1(arge)/(arge*rho**2))*uvdir
    imi = r2ms*(k*rq2*hr/pi2)*(0.5*(j0(argi) - jn(2,argi))/rho**2 - \
                               j1(argi)/(argi*rho**2))*uvdir

    
    ## ime =     k*hr/pi2*(pi*Rext*(j0(arge) - jn(2,arge))/arge**2 - \
    ## 				j1(arge)/arge**3)*ulam*as2r;
    ## imi = k*rq2*hr/pi2*(pi*Rint*(j0(argi) - jn(2,argi))/argi**2 - \
    ## 				j1(argi)/argi**3)*ulam*as2r;

    re[:] = (ree - rei)
    im[:] = -(ime - imi) # "minus" rotates the resulting image by 180 degrees



def gausring (ulam, vlam, zsp, gsize, radius, theta, re, im):
    """
    Gaussian-and-ring model in the visibility space
    Parameters: zsp, gsize, radius, theta
    Returns re and im arrays
    """
    
    # Note, factor of 4.8481e-6 converts
    # Mlambda*microarcsec to lambda*radians
    as2rad = 4.8481e-6;
    gcoef = 2.*(4.8481e-6*pi/(2.*sqrt(2.*log(2.))))**2
    
    Rext = radius * 2.4;
    Rint = radius * 2;

    #
    # Gaussian's x and y offsets
    #
    xg = 1.1*radius*cos(theta);  
    yg = 1.1*radius*sin(theta);
    
    rho = sqrt(ulam**2 + vlam**2);
    arg = pi*rho;
  
    # Ring is difference of two bessel functions
    rquo2 = Rint**2/Rext**2
    k = 2./(1. - rquo2)
    
    D1 = k*zsp*j1(arg*Rext*as2rad)/(arg*Rext*as2rad);       # disk1 big
    
    D2 = k*rquo2*zsp*j1(arg*Rint*as2rad)/(arg*Rint*as2rad); # disk2 small

    ## Note, factor of as2rad converts Mlambda*microarcsec to lambda*radians
    
    #
    # Now compute vis of offset circ Gaussian
    # gsize is FWHM of the Gaussian
    # The factor gcoef = 8.36676e-11 = 2*(as2rad*pi/(2*sqrt(2*log(2))))**2
    # FWHM = 2*sqrt(2*log(2))*stdev
    #
    gamp=(2.4-zsp)*exp(-gcoef*(rho*gsize)**2); # 2.4 Jy
    
    gphase =  2.0*pi*as2rad*(xg*ulam + yg*vlam);

    #print 'gphase.shape = ', gphase.shape

    #if False:   # It makes no sense - cos and sine are insensitive to +-2*pi*n
    # Reduce closure phase angle to [-180..+180]
    # First reduce to [-2pi ..+2pi]
    gphase = gphase - 2.*pi*trunc(gphase/(2.*pi)) # Reduce to [-2pi..+2pi]
    # Then to [-pi ..+pi] 
    idx1 = find(gphase < -pi)
    idx2 = find(gphase > pi)
    gphase[idx1] += 2.*pi
    gphase[idx2] -= 2.*pi

    im[:] = gamp*sin(gphase);
    re[:] = gamp*cos(gphase) + D1 - D2;

    # end of gausring()


def calc_chi2(visfun, ulam, vlam, amp, cphs, cpid, vfparam):
    """
    Calculate amplitude chi^2 for a  model given in visfun()
    Input:
       ulam, vlam - UV-plane coopdinates,
       amp, cphs, cpid - amplitudes, closures, and closure indices
    Parameters:
      vfparam = ('Zsp', 'Re', 'rq', 'ecc', 'fade', \
                 'gax', 'aq', 'gq', 'th')
    Returns:
      chi^2
    """
    s2amp = 1.0
    s2cph = 1.0

    chi2 = 0.
    V = visfun(ulam, vlam, vfparam)
    mamp = abs(V)
    mpha = arctan2(V.imag,V.real)
    mcph = cpid2closures(cpid, mpha)
    chi2am = norm((mamp - amp)/s2amp)
    chi2cp = norm((mcph - cphs)/s2cph)

    return chi2am + chi2cp

    

def compute_chi2_amp(visfun, ulam, vlam, amp, phase, \
                     zsp, gsize, radius, theta):
    """
    Compute amplitude chi^2 for a  model given in visfun()
    Parameters: zsp, gsize, radius, theta
    Inputs
      ulam, vlam: visibility data u and v arrays (having read from file)
      amp, phase: data amplitude and phase arrays (having read from file)
    """
    chi_sq_amp = 0.0;

    #
    # First compute chi^2 on amplitude data
    #
    n = size(ulam,0)
    re_vis = zeros(n, float)
    im_vis = zeros(n, float)
    conv=1.0;
    
    visfun(ulam, vlam, zsp, gsize, radius, theta, re_vis, im_vis)

    amp_comp = sqrt(im_vis**2 + re_vis**2)*conv;
    #gphase = arctan2(im_vis, re_vis);
    sig_sq_amp = 0.1;
    
    chi_sq_amp = sum((amp - amp_comp)**2/sig_sq_amp);

    return chi_sq_amp



def compute_chi2_cph(visfun, ulam, vlam, cphs, cuv, \
                  zsp, gsize, radius, theta):
    """
    Compute closure phases' chi^2 for a  model given in visfun()
    Parameters: zsp, gsize, radius, theta
    Inputs
     (having read from file):
      ulam, vlam: visibility data u and v arrays
      cphs, cuv:  data closure phases and closure uv's for each of 3 baselines
    We assume a sigma of 0.5 radians.  Cphases are already in radians.
    """
    chi_sq_cph = 0.0;
    
    m = size(cphs,0)
    re_vis = zeros(m, float)
    im_vis = zeros(m, float)
    sig_sq_cph = 0.5*0.5
    
    # We need to figure out visibility phase for each baseline
    cphs_comp = 0.0;
    for i in xrange(3):
        u = cuv[:,2*i]    # Link to u column 
        v = cuv[:,2*i+1]  # Link to v column 

        visfun(u, v, zsp, gsize, radius, theta, re_vis, im_vis)
        #gausring(u, v, zsp, gsize, radius, theta, re_vis, im_vis)

        cphs_comp += sum(arctan2(im_vis, re_vis))
        
        phase_diff = cphs - cphs_comp
        # Reduce closure phase angle to [-180..+180]
        # First reduce to [-2pi ..+2pi]
        phase_diff = phase_diff - 2.*pi*trunc(phase_diff/(2.*pi))
        # Then to [-pi .. +pi]
        idx = find(phase_diff < -pi)
        phase_diff[idx] += 2.*pi
        idx = find(phase_diff > pi)
        phase_diff[idx] -= 2.*pi
        chi_sq_cph += sum(phase_diff**2)/sig_sq_cph;
        
    return chi_sq_cph




def plot_orig_vs_model(fits_file, visfun, zsp, gsize, radius, theta):
    '''
    Plot side by side original black hole image and its model
    Model parameters: zsp, gsize, radius, theta
    '''
    #
    # Extract original image from the fits file
    #
    hdulist = pf.open(fits_file)
    hdu = hdulist[0]
    orig = hdu.data   # original image [N x N] pix in Jy/pixel
    N = hdu.header['naxis1']
    delt = abs(hdu.header['cdelt1'])  # degrees per pixel
    hdulist.close()

    #
    # Compute model image 
    #
    XYspan_deg = N*delt    # degrees
    XYspan_uas = XYspan_deg*3600e6   # microarcseconds 
    XYspan = radians(XYspan_deg)     # radians

    UVspan = N/XYspan           # in lambdas
    UVspan_Glam = 1e-9*UVspan   # in Gigalambdas
    UVspan_Mlam = 1e-6*UVspan   # in Megalambdas

    print 'XYspan_uas = ', XYspan_uas, ', UVspan_Glam = ', UVspan_Glam

    #sys.exit(0)

    #ngrid = 1024
    #nker = 256  # Size of the middle plotted

    ngrid = int(2**ceil(log2(2*N)))  # Extended grid (with wide zero margins)
    m = ngrid/2
    k = N/2  # Size of the kernel to plot


    #uvext = 100000.       # Mlambda
    uvext = UVspan_Mlam/2. # Mlambda
    urng = linspace(-uvext, uvext, ngrid)
    vrng = urng
    Ugrid, Vgrid = meshgrid(urng, vrng)
    Ug = Ugrid[m-k:m+k,m-k:m+k]
    Vg = Vgrid[m-k:m+k,m-k:m+k]
    U = Ugrid.flatten()
    V = Vgrid.flatten()                        
    re = zeros_like(U)
    im = zeros_like(V)

    visfun(U, V, zsp, gsize, radius, theta, re, im)

    re = reshape(re, (ngrid,ngrid))
    im = reshape(im, (ngrid,ngrid))
    #amp_comp = sqrt(im**2 + re**2)
    F = re + 1j*im
    #F = sqrt(re**2 + im**2)
    Fk = F[m-k:m+k,m-k:m+k]

    #
    # Make a 2D cosine window
    #
    rho = sqrt(Ugrid**2 + Vgrid**2)
    cwin = 0.5*(1. + cos(pi*rho/uvext))
    cwin[where(rho > uvext)] = 0.

    f =  fftshift(ifft2(ifftshift(F*cwin)))

    #xyext = ngrid/(4.*uvext)
    xyext = XYspan_uas/2.


    xrng = linspace(-xyext, xyext, ngrid)
    yrng = xrng
    Xgrid, Ygrid = meshgrid(xrng, yrng)
    Xg = Xgrid[m-k:m+k,m-k:m+k]
    Yg = Ygrid[m-k:m+k,m-k:m+k]
    fk = f[m-k:m+k,m-k:m+k]
    absfk = abs(fk)

    brmax = absfk.max()


    sig_sq = 0.1;
    chi_sq = sum((orig - absfk)**2)/sig_sq;

    figure(figsize=(12,6))
    #figure(figsize=(12,12))
    ax1 = subplot(121)
    imshow(orig, cmap=cm.hot, vmax=brmax) # Use same color scale as for model
    grid(1)
    llim, rlim = xlim()
    tln = linspace(-xyext, xyext, 5)
    tls = ['%.1f' % s for s in tln]
    xticks(linspace(llim, rlim, 5), tls)
    yticks(linspace(llim, rlim, 5), tls)
    xlabel('RA (uas)')
    ylabel('Dec (uas)')
    colorbar(shrink=0.7)

    ax2 = subplot(122)
    imshow(abs(fk), cmap=cm.hot)
    grid(1)
    llim, rlim = xlim()
    tln = linspace(-xyext, xyext, 5)
    tls = ['%.1f' % s for s in tln]
    xticks(linspace(llim, rlim, 5), tls)
    yticks(linspace(llim, rlim, 5), tls)
    xlabel('RA (uas)')
    colorbar(shrink=0.7)
    ## For figsize=(12,12):
    ## figtext(0.18, 0.91, 'Original', fontsize=16)
    ## figtext(0.18, 0.89, 'max B = %f Jy/pixel' % orig.max())
    ## figtext(0.18, 0.87, 'zsp B = %f Jy' % sum(orig))
    ## figtext(0.6, 0.91, 'Model', fontsize=16)
    ## figtext(0.6, 0.89, 'max B = %f Jy/pixel' % absfk.max())
    ## figtext(0.6, 0.87, 'zsp B = %f Jy' % sum(absfk))
    ## For figsize=(12,12):
    figtext(0.18, 0.88, 'Original', fontsize=16)
    figtext(0.18, 0.84, 'max B = %f Jy/pixel' % orig.max())
    figtext(0.18, 0.81, 'sum B = %f Jy' % sum(orig))
    figtext(0.6, 0.88, 'Model', fontsize=16)
    figtext(0.6, 0.84, 'max B = %f Jy/pixel' % absfk.max())
    figtext(0.6, 0.81, 'sum B = %f Jy' % sum(absfk))
    figtext(0.42, 0.12, 'Chi^2 = %f ' % chi_sq)
    figtext(0.25, 0.08, 'zsp = %g,   gsize = %g,   radius = %g,   ' \
            'theta = %g deg' %
            (zsp, gsize, radius, degrees(theta)))

    show()
    

def vis2bri_old(visfun, vfparam, UVspan_total, ntotal, nv, nb):
    """
    Inverse FFT of a cosine-windowed uv-plane image
    Returns only the central square cuts of both the visibility and
    the resulting brightness image, smaller than the whole image range.
    
    Input:
    visfun(): a visibility function returning generally complex,
            2D, ntotal x ntotal, square array of values over the (Ugrid,Vgrid)
    vfparam: a sequence of the visfun() parameters
    UVspan_total: full range of uv-plane in U and V directions (Gigalambdas)
    ntotal: dimensions of the UV plane for initual visibility calculation 
    nv: dimensions of the UV visibility image central cut to return
    nb: dimensions of the XY brightness image central cut to return

    Output:
    Vis: the visibility image computed by visfun(), a generally complex
         2D array, Vis.shape=(ntotal,ntotal)
    Bri: central cut, Bri.shape=(nb,nb), of the IFFT result,
           i.e. XY brightness image
    Ugrid, Vgrid: grids for U and V (returned by meshgrid() standard function)
    Xgrid, Ygrid: grids for X and Y (returned by meshgrid() standard function)
    UVext: + and - extents of the returning visbility image central cut
            in U and V directions. UVext = UVspan/2.
    XYext: + and - extents of the returning brightness image central cut
            in X and Y directions. XYext = XYspan/2.
    ticval, ticnam: values and names of ticks for plotting Vis and Bri
            using imshow()
            
    Some of internal variables:
    XYzoomin: the ratio of total XYspan to the XYspan of the returned XY image
    UVspan: range of the returning visibility image central cut
            in U and V directions.
    XYspan: range of the returning brightness image central cut
            in X and Y directions.
    
    """
    #
    # Make U and V grids
    #
    uvext = UVspan_total/2.                           # Gigalambda
    #
    # HERE WAS AN ERROR!!! Even and odd were confused!!! (Tue, Oct 23, 2012)
    #

    if ntotal%2:   # For EVEN ntotal
        uvrng = linspace(-uvext, uvext - UVspan_total/ntotal, ntotal)
    else:          # For ODD ntotal
        uvrng = linspace(-uvext, uvext, ntotal)
        
    Ugrid_total, Vgrid_total = meshgrid(uvrng, uvrng)
    #
    # Make a 2D cosine window
    #
    rho = sqrt(Ugrid_total**2 + Vgrid_total**2)
    cwin = 0.5*(1. + cos(pi*rho/uvext))
    cwin[where(rho > uvext)] = 0.

    #
    # Calcuulate visibilities over the grid
    #
    Vi = visfun(Ugrid_total, Vgrid_total, vfparam)
    
    ## Ugt = Ugrid_total.flatten()
    ## Vgt = Vgrid_total.flatten()
    ## Vi_flat = mc.model1.xringaus(Ugt, Vgt, vfparam)
    ## Vi = Vi_flat.reshape((ntotal, ntotal))

    
    
    #
    # Conversion from visibility to brightness
    #
    ##########  Br =  fftshift(ifft2(ifftshift(Vi*cwin)))
    Br =  ifftshift(ifft2(Vi*cwin))
    #
    # Cut the central square nb x nb size from f
    #
    XYspan = ntotal/UVspan_total               # in Nanoradians
    XYzoomin = float(nb)/float(ntotal)
    XYspan_nrad = XYzoomin*XYspan             # nanoradians
    XYspan = 3600.*degrees(XYspan_nrad)*1e-3  # microarcseconds
    xyext = XYspan/2.
    xyrng = linspace(-xyext, xyext, nb)
    Xgrid, Ygrid = meshgrid(xyrng, xyrng)
    
    m = ntotal/2
    
    k = nb/2
    if nb%2: k1 = k + 1 # nb is odd
    else:    k1 = k     # nb is even: plot will ve asymmetric

    Bri = Br[m-k:m+k1,m-k:m+k1]

    k = nv/2
    if nv%2: k1 = k + 1 # nv is odd
    else:    k1 = k     # nv is even: plot will ve asymmetric
    
    Ugrid = Ugrid_total[m-k:m+k1,m-k:m+k1]
    Vgrid = Vgrid_total[m-k:m+k1,m-k:m+k1]
    Vis = Vi[m-k:m+k1,m-k:m+k1]
    UVspan = (float(nv)/float(ntotal))*UVspan_total
    
    uvext = UVspan/2.
    if nv%2:
        UVext = (-uvext, uvext, -uvext, uvext)
    else:
        UVext = (-uvext, uvext - UVspan/nv, -uvext, uvext - UVspan/nv)
        
    xyext = XYspan/2.
    #XYext = (-xyext, xyext, -xyext, xyext)
    if nb%2:
        XYext = (-xyext, xyext, -xyext, xyext)
    else:
        XYext = (-xyext, xyext - XYspan/nv, -xyext, xyext - XYspan/nv)
        
    return Vis, Bri, Ugrid, Vgrid, Xgrid, Ygrid, UVext, XYext


    

def model_vis2bri(visfun, vfparam, UVspan_total, ntotal, nv, nb):
    """
    Inverse FFT of a cosine-windowed uv-plane image from a model.
    The UV image from the model visfun() has the U axis pointing
    to the right. In order to conform with the astrophysical standards,
    the returned visibility and brightness images are left-right
    transposed, so that directions of the U and X axes are changed
    to that from right to left. This results in the brightness image
    with the RA growing from right to left. Accordingly, the U and X
    limits in Ugrid, Vgrid, Xgrid, Ygrid, UVext, and XYext are changed
    to the leftward.
    Returns only the central square cuts of both the visibility and
    the resulting brightness image, smaller than the whole image range.
    
    Input:
    visfun(): a visibility function returning generally complex,
            2D, ntotal x ntotal, square array of values over the (Ugrid,Vgrid)
    vfparam: a sequence of the visfun() parameters
    UVspan_total: full range of uv-plane in U and V directions (Gigalambdas)
    ntotal: dimensions of the UV plane for initual visibility calculation 
    nv: dimensions of the UV visibility image central cut to return
    nb: dimensions of the XY brightness image central cut to return

    Output:
    Vis: the visibility image computed by visfun(), a generally complex
         2D array, Vis.shape=(ntotal,ntotal)
    Bri: central cut, Bri.shape=(nb,nb), of the IFFT result,
           i.e. XY brightness image
    Ugrid, Vgrid: grids for U and V (returned by meshgrid() standard function)
    Xgrid, Ygrid: grids for X and Y (returned by meshgrid() standard function)
    UVext: + and - extents of the returning visbility image central cut
            in U and V directions. UVext = UVspan/2.
    XYext: + and - extents of the returning brightness image central cut
            in X and Y directions. XYext = XYspan/2.
    ticval, ticnam: values and names of ticks for plotting Vis and Bri
            using imshow()
            
    Some of internal variables:
    XYzoomin: the ratio of total XYspan to the XYspan of the returned XY image
    UVspan: range of the returning visibility image central cut
            in U and V directions.
    XYspan: range of the returning brightness image central cut
            in X and Y directions.
    
    """
    #
    # Make U and V grids
    #
    uvext = UVspan_total/2.                           # Gigalambda

    #
    # Make grid with standard order for V (for n=4: -2, -1,  0,  1)
    # but LR-flipped order for U          (for n=4:  1,  0, -1, -2)
    #
    if ntotal%2 == 0:   # For EVEN ntotal
        urng = linspace(uvext-UVspan_total/ntotal, -uvext, ntotal)
        vrng = linspace(-uvext, uvext-UVspan_total/ntotal, ntotal)
    else:               # For ODD ntotal
        urng = linspace(uvext, -uvext, ntotal)
        vrng = urng[::-1]                      # Reversed order
        
    Ugrid_total, Vgrid_total = meshgrid(urng, vrng)
    
    #
    # Make a 2D cosine window
    #
    rho = sqrt(Ugrid_total**2 + Vgrid_total**2)
    cwin = 0.5*(1. + cos(pi*rho/uvext))
    cwin[where(rho > uvext)] = 0.
    #figure();
    #imshow(cwin, cmap=cm.jet, origin='lower', interpolation='nearest');
    #grid(1);

    #
    # Calculate visibilities over the grid
    # 
    #Vi = visfun(Ugrid_total, Vgrid_total, vfparam)
    
    Ugt = Ugrid_total.flatten()
    Vgt = Vgrid_total.flatten()
    #Vi_flat = mc.model1.sgra_model(Ugt, Vgt, vfparam)
    Vi_flat = visfun(Ugt, Vgt, vfparam)
    Vi = Vi_flat.reshape((ntotal, ntotal))
    
    #
    # Conversion from visibility to brightness
    #
    Br =  ifftshift(ifft2(Vi*cwin))

    #
    # Find X and Y extents 
    #
    XYspan = ntotal/UVspan_total               # in Nanoradians
    XYspan = 3600.*degrees(XYspan)*1e-3        # microarcseconds
    xyext = XYspan/2. 

    #xyrng_t = linspace(-xyext, xyext-XYspan/ntotal, ntotal)
    #
    # Cut the central square nb x nb size from f
    if ntotal%2 == 0:   # For EVEN ntotal
        xrng_t = linspace(xyext-XYspan/ntotal, -xyext, ntotal)
        yrng_t = linspace(-xyext, xyext-XYspan/ntotal, ntotal)
    else:               # For ODD  ntotal
        yrng_t = linspace(-xyext, xyext, ntotal)
        xrng_t = yrng_t[::-1]                      # Reversed order
    
    m = ntotal/2
    
    k = nb/2
    if nb%2: k1 = k + 1 # nb is odd
    else:    k1 = k     # nb is even: plot will ve asymmetric

    #Bri = fliplr(Br[m-k:m+k1,m-k:m+k1])      # FLIPLR!!!!!!!!!!!!!!!
    Bri = Br[m-k:m+k1,m-k:m+k1]

    k = nv/2
    if nv%2: k1 = k + 1 # nv is odd
    else:    k1 = k     # nv is even: plot will be asymmetric

    Ugrid = Ugrid_total[m-k:m+k1,m-k:m+k1]
    #Ugrid = fliplr(Ugrid)                    # FLIPLR!!!!!!!!!!!!!!!
    Vgrid = Vgrid_total[m-k:m+k1,m-k:m+k1]
    
    Vis = Vi[m-k:m+k1,m-k:m+k1]
    
    uext0 = Ugrid[0,0]; uext1 = Ugrid[0,-1]
    vext0 = Vgrid[0,0]; vext1 = Vgrid[-1,0]
    UVext = (uext0, uext1, vext0, vext1)

    xrng = xrng_t[m-k:m+k1]
    #xrng = xrng[::-1]                     # Reverse order
    yrng = yrng_t[m-k:m+k1]
    Xgrid, Ygrid = meshgrid(xrng, yrng)

    xext0 = Xgrid[0,0]; xext1 = Xgrid[0,-1]
    yext0 = Ygrid[0,0]; yext1 = Ygrid[-1,0]
    XYext = (xext0, xext1, yext0, yext1)

    return Vis, Bri, Ugrid, Vgrid, Xgrid, Ygrid, UVext, XYext
                                                     # end of model_vis2bri()



def model_vis2bri_simple(visfun, vfparam, UVspan, ngrid):
    """
    Inverse FFT of a cosine-windowed uv-plane image from a model.
    The UV image from the model visfun() has the U axis pointing
    to the right. In order to conform with the astrophysical standards,
    the returned visibility and brightness images are left-right
    transposed, so that directions of the U and X axes are changed
    to that from right to left. This results in the brightness image
    with the RA growing from right to left. Accordingly, the U and X
    limits in Ugrid, Vgrid, Xgrid, Ygrid, UVext, and XYext are changed
    to the leftward.
    Returns only the central square cuts of both the visibility and
    the resulting brightness image, smaller than the whole image range.
    
    Input:
    visfun(): a visibility function returning generally complex,
            2D, ngrid x ngrid, square array of values over the (Ugrid,Vgrid)
    vfparam: a sequence of the visfun() parameters
    UVspan: full range of uv-plane in U and V directions (Gigalambdas)
    ngrid: dimensions of the UV plane for initual visibility calculation 

    Output:
    Vis: the visibility image computed by visfun(), a generally complex
         2D array, Vis.shape=(ngrid,ngrid)
    Bri: central cut, Bri.shape=(nb,nb), of the IFFT result,
           i.e. XY brightness image
    Ugrid, Vgrid: grids for U and V (returned by meshgrid() standard function)
    Xgrid, Ygrid: grids for X and Y (returned by meshgrid() standard function)
    UVext: + and - extents of the returning visbility image central cut
            in U and V directions. UVext = UVspan/2.
    XYext: + and - extents of the returning brightness image central cut
            in X and Y directions. XYext = XYspan/2.
    ticval, ticnam: values and names of ticks for plotting Vis and Bri
            using imshow()
            
    Some of internal variables:
    XYzoomin: the ratio of total XYspan to the XYspan of the returned XY image
    UVspan: range of the returning visibility image central cut
            in U and V directions.
    XYspan: range of the returning brightness image central cut
            in X and Y directions.
    
    """
    #
    # Make U and V grids
    #
    uvext = UVspan/2.                           # Gigalambda

    #
    # Make grid with standard order for V (for n=4: -2, -1,  0,  1)
    #    but LR-flipped order for U       (for n=4:  1,  0, -1, -2)#### NO!
    #
    if ngrid%2 == 0:   # For EVEN ngrid
        urng = linspace(uvext-UVspan/ngrid, -uvext, ngrid)
        vrng = linspace(-uvext, uvext-UVspan/ngrid, ngrid)
    else:               # For ODD ngrid
        urng = linspace(-uvext, uvext, ngrid)
        urng = linspace(uvext, -uvext, ngrid)
        
    Ugrid, Vgrid = meshgrid(urng, vrng)  # RA from right to left
    
    #
    # Make a 2D cosine window
    #
    rho = sqrt(Ugrid**2 + Vgrid**2)
    cwin = 0.5*(1. + cos(pi*rho/uvext))
    cwin[where(rho > uvext)] = 0.

    #
    # Calculate visibilities over the grid
    # 
    
    ## Ugt = Ugrid.flatten()
    ## Vgt = Vgrid.flatten()
    ## Vi_flat = visfun(Ugt, Vgt, vfparam)
    ## Vi = Vi_flat.reshape((ngrid, ngrid))

    Vi = visfun(Ugrid, Vgrid, vfparam)


    #mamp, mpha = ig.Mcgpu.calcmodel(Ugrid, Vgrid, vfparam)
    #Vi = mamp*(cos(mpha) + 1j*sin(mpha))
    #print 'mamp, mpha = ig.Mcgpu.calcmodel(Ugrid, Vgrid, vfparam)'
    #print 'Vi = mamp*(cos(mpha) + 1j*sin(mpha))'
    
    #
    # Conversion from visibility to brightness
    #
    #Br = ifftshift(ifft2(ifftshift(Vi))) #*cwin)))

    Br =  ifftshift(ifft2(fftshift(Vi*cwin)))
    
    #wireframe(Br.real)
    #
    # Find X and Y extents 
    #
    XYspan = ngrid/UVspan               # in Nanoradians
    XYspan = 3600.*degrees(XYspan)*1e-3        # microarcseconds
    xyext = XYspan/2. 

    #
    # Brightness domain grid
    #
    if ngrid%2 == 0:   # For EVEN ngrid
        xrng = linspace(xyext-XYspan/ngrid, -xyext, ngrid)
        yrng = linspace(-xyext, xyext-XYspan/ngrid, ngrid)
    else:               # For ODD  ngrid
        xrng = linspace(xyext, -xyext, ngrid)
        yrng = linspace(-xyext, xyext, ngrid)
    

    uext0 = Ugrid[0,0]; uext1 = Ugrid[0,-1]
    vext0 = Vgrid[0,0]; vext1 = Vgrid[-1,0]
    UVext = (uext0, uext1, vext0, vext1)

    Xgrid, Ygrid = meshgrid(xrng, yrng)

    xext0 = Xgrid[0,0]; xext1 = Xgrid[0,-1]
    yext0 = Ygrid[0,0]; yext1 = Ygrid[-1,0]
    XYext = (xext0, xext1, yext0, yext1)

    return Vi, Br, Ugrid, Vgrid, Xgrid, Ygrid, UVext, XYext
                                                # end of model_vis2bri_simple()




def sqcut(A, Aspan, ncut):
    """
    Return a square central cut of dimensions ncut X ncut
    from the 2D array A. The range along X and Y axes,
    Aspan, expressed in arbitrary units, is converted
    into the proportionally shorter Acut
    Also , a four-number Aext tuple is returned to be used
    for plotting Acut with imshow() as:
    >>> imshow(Acut, extent=Aext)
    """
    ngrid = len(A[:,0])
    if ncut > ngrid: 
        return A, Aspan   # Leave A intact
    m = ngrid/2
    k = ncut/2
    Acut = A[m-k:m+k,m-k:m+k]

    Aspan = (float(ncut)/float(ngrid))*Aspan
    aext = Aspan/2.
    Aext = (-aext, aext, -aext, aext)  # Use in imshow(Acut, extent=Aext)
    
    return Acut, Aspan, Aext
   


def read_Visibility_dat(vis_file):
    """
    Read the result of MAPS/visgen program run into a complex 2D
    array. By convention, such files have names like *_Visibility.dat
    """
    
    a = fromfile(vis_file)
    
    #
    # The first 8 bytes in the file contain two 32-bit integers.
    # They are the image dimensions, nrows and ncols. Here the both
    # are packed in one 64-bit float (double).
    #
    
    sz = a[0]      # One float64 variable holding two 32-bit integers
    a = a[1:]      # Remove the 0th element
    
    #
    # Extract nrows and ncols from sz
    #
    b = BitStream(float=sz, length=64)
    b = b.readlist([32, 32])
    nrows = b[0].int
    ncols = b[1].int

    #
    # Extract real and imaginary parts into separate arrays
    # In FITS files data are columnwise, so after reshape we make transpose
    #
    re1 = a[::2]   # Even-indexed numbers are real parts
    im1 = a[1::2]  # Odd-indexed numbers are imaginary parts
    re1 = re1.reshape((nrows,ncols)).transpose()
    im1 = im1.reshape((nrows,ncols)).transpose()

    #
    # Since MAPS_im2uv LR-flips the brightness before fft2,
    # we need to flip it back
    #
    reV = fftshift(re1)                  #[:,::-1]
    imV = fftshift(im1)                  #[:,::-1]
    
    
    Vis = reV + 1j*imV   # Make complex visibility image

    return Vis        #[:,::-1]


def read_fits(fits_file, param=False, xlr=False):
    """
    Extract image from the fits file and return it as 2D array
    along with the RA,Dec extents in microarcseconds.
    If param keyward is True (or 1), the array of model parameters
    from the fits header is returned. If there are no model parameters
    in the fits header, then None value in place of parameter array is
    returned. 
    If xlr keyward is True (or 1), the image is left-right transposed,
    and the X extents are transposed accordingly.
    """
    hdulist = pf.open(fits_file)
    hdu = hdulist[0]
    image = hdu.data   # original image [N x N] pix in Jy/pixel
    N = hdu.header['naxis1']
    delt = abs(hdu.header['cdelt1'])  # degrees per pixel
    #
    # If model parameters requested, extract them:
    #
    if bool(param) == True:
        h = hdu.header
        vl = h.values() # List of values, including COMMENT
        #print 'vl = ', vl
        sprm = ''
        parsep = False  # Not parsing the parameters
        v = vl[0]
        if type(v) == str and 'Model parameters:' in v:
            sprm = v
            parsep = True   # Start parsing the parameters
        else:
            sprm = ''
            parsep = False  # Not parsing the parameters
        for v in vl[1:]:
            #print 'v = ', v
            if type(v) == str and 'Zsp =' in v:
                parsep = False   # Not parsing the parameters any more
            if parsep:
                sprm = sprm +' '+v   # Insert blank not to merge numbers
            if type(v) == str and 'Model parameters:' in v:
                sprm = v
                parsep = True   # Start parsing the parameters
        if sprm != '':
            #print 'sprm=', sprm
            p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE]' \
                           '[+-]?\d+|[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')
            prms = np.float_(p.findall(sprm))
        else:
            prms = None
    hdulist.close()

    pix_uas = delt*1e6*3600.
    lim = (N/2)*pix_uas
        
    #
    # Circumvent an error in matplotlib
    #
    m, n = image.shape
    b = zeros((m+2,n+2), dtype=float) # Container for image with 1-pixel margin
    #
    # If left-right transposition requested
    #
    if bool(xlr) == True:
        b[1:-1,1:-1] = np.fliplr(image)
        XYext = array([-(N/2)*pix_uas, (N/2-1)*pix_uas,
                       -(N/2)*pix_uas, (N/2-1)*pix_uas])
    else:
        b[1:-1,1:-1] = image
        XYext = array([(N/2-1)*pix_uas, -(N/2)*pix_uas,
                       -(N/2)*pix_uas, (N/2-1)*pix_uas])
    
    if bool(param) == True:
        return b[1:-1,1:-1], XYext, prms
    else:
        return b[1:-1,1:-1], XYext



def read_quasikerr(f):
    """
    Read a quasi-Kerr simulated image from the library into a square array.
    f: file name
    """
    fh = open(f)
    fh.readline(); fh.readline(); fh.readline() # Skip header
    dat = loadtxt(fh)
    fh.close()
    x = dat[:,0]
    y = dat[:,1]
    img = dat[:,2]*1e3  # In milliJansky/pix
    img = img.reshape((100,100))
    
    mass = 4.3;               # x million solar masses 
    distance = 8.0;    # kpc
    pixel_increment_in_M = 0.30150808; # Masses
    M_in_arcseconds = 9.87122944e-6;

    M_in_arcseconds *= mass;
    M_in_arcseconds /= distance;
    pixel_increment_in_degrees = pixel_increment_in_M*M_in_arcseconds/3600.0;
    #print "pixel increment in degrees: %g" % pixel_increment_in_degrees
    pixel_increment_in_arcseconds = pixel_increment_in_M*M_in_arcseconds;
    #print "pixel increment in arcseconds: %g" % pixel_increment_in_arcseconds
    pixel_increment_in_uas = pixel_increment_in_M*M_in_arcseconds*1e6;
    #print "pixel increment in uas: %g" % pixel_increment_in_uas

    lim = 50. # pixels!
    XYext = pixel_increment_in_uas*array([lim-1, -lim, -lim, lim-1])

    return img, XYext




def wireframe(z_in):
    """
    Simple interface to plot_wireframe()
    """
    from mpl_toolkits.mplot3d import axes3d
    z = array(z_in)
    m = len(z[:,0])
    n = len(z[0,:])
    ## rstr = m/50 + 1
    ## cstr = n/50 + 1
    rstr = 1
    cstr = 1
    
    xrng = linspace(0., n, n)
    yrng = linspace(0., n, m)
    x, y = meshgrid(xrng, yrng)

    print x.shape, y.shape, z.shape

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=rstr, cstride=cstr,lw=0.5)

    show()


def wireframexy(x_in, y_in, z_in):
    """
    Simple interface to plot_wireframe(x, y, z)
    """
    from mpl_toolkits.mplot3d import axes3d
    x = array(x_in)
    y = array(y_in)
    z = array(z_in)
    m = len(z[:,0])
    n = len(z[0,:])
    ## rstr = m/50 + 1
    ## cstr = n/50 + 1
    rstr = 1
    cstr = 1
    
    print x.shape, y.shape, z.shape

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=rstr, cstride=cstr,lw=0.5)

    show()



    
def rotzoom(img, Zsp, kzoom, thd):
    """
    Rotate the img image by th (rad), zoom by kzoom,
    and change total flux by Zsp 
    """
    npix = len(img[0,:])
    centrpix = npix//2
    absZsp0 = sum(img)
    #
    # The zooming factor MUST be an even percent, like 0.86
    # Here kzoom is rounded towards the closest "even percent"
    # Example: 0.7113 -> 0.7;   0.66852 -> 0.66;   1.532 -> 1.52 etc.
    #
    kz100 = round(100.*kzoom)
    if int(kz100) % 2 == 0:
        kzoom = kz100/100.
    else:
        kzoom = (kz100 - 1.)/100.
    print 'zoom = ', kzoom
    
    if kzoom > 1.:
        img1 = sn.rotate(img, thd, reshape=False)
        img2 = sn.zoom(img1, kzoom)
        nz = len(img2[0,:])
        nh = nz/2
        rng = arange(nz)
        X, Y = meshgrid(rng, rng)
        ix = where((X-nh)**2 + (Y-nh)**2 >= nh**2)
        img2[ix] = 0.
        img3 = img2[nh-50:nh+50,nh-50:nh+50]
    elif kzoom < 1.:
        rng = arange(npix)
        X, Y = meshgrid(rng, rng)
        ix = where((X-centrpix)**2 + (Y-centrpix)**2 >= centrpix**2)
        img1 = sn.rotate(img, thd, reshape=False)
        img1[ix] = 0.
        img2 = sn.zoom(img1, kzoom)
        nz = len(img2[0,:])
        nh = nz/2
        if 2*nh < nz: nh = nh + 1
        img3 = zeros((npix,npix), dtype=float)
        img3[50-nh:50+nh,50-nh:50+nh] = img2
    else:  # i.e. kzoom == 1.
        rng = arange(npix)
        X, Y = meshgrid(rng, rng)
        ix = where((X-centrpix)**2 + (Y-centrpix)**2 >= centrpix**2)
        img3 = sn.rotate(img, thd, reshape=False)
        img3[ix] = 0.

    absZsp = sum(img3)
    img3 = Zsp*(absZsp0/absZsp)*img3

    return img3



def uas2glam(XYspan_uas, N):
    XYspan_nanorad = XYspan_uas*(1e-6/3600.)*(pi/180.)*1e9
    UVspan_Glam = float(N)/XYspan_nanorad
    return UVspan_Glam


def glam2uas(UVspan_Glam, N):
    XYspan_nanorad = float(N)/UVspan_Glam
    XYspan_uas = XYspan_nanorad*1e-9*(180./pi)*3600.*1e6
    return XYspan_uas


def pixeluas2glam(pixel_uas):
    pixel_nanorad = (pixel_uas/3600.)*(pi/180.)*1e3   # 1e-6 * 1e9 = 1e3
    UVspan_Glam = 1./pixel_nanorad
    return UVspan_Glam


def pixelglam2uas(pixel_Glam):
    XYspan_nanorad = 1./pixel_Glam
    XYspan_uas = XYspan_nanorad*(180./pi)*3600.*1e-3
    return XYspan_uas


def cpix2cpid(cpix):
    """
    Compress the cpix[ncph,3] array of phase index triplets
    into the sortable array cpid[ncph], each number in which
    contains three numbers as bit fields.
    """
    ncph = len(cpix[:,0])
    cpid = np.zeros(ncph, dtype=int)

    for i in xrange(ncph):
        cpid[i] = cpix[i,0] + PK3I*(cpix[i,1] + PK3I*cpix[i,2])

    return cpid


def cpid2indx(cpid):
    i1 = cpid%PK3I; i3 = cpid//PK3I; i2 = i3%PK3I; i3 = i3//PK3I
    return i1, i2, i3


def indx2cpid(i1, i2, i3):
    return i1 + PK3I*(i2 + PK3I*i3)


def cpid2closures(cpid, phase):
    """
    Compute closure phases using cpid provided by calc_closures*()
    """
    ncph = len(cpid)
    cphs = zeros(ncph)
    for i in xrange(ncph):
        i1 = cpid[i]%PK3I
        iw = cpid[i]//PK3I
        i2 = iw%PK3I
        i3 = iw//PK3I
        cp = phase[i1] + phase[i2] - phase[i3]
        #print 'phase[i1] = ', phase[i1], ', i1 = ', i1
        #print 'cp = ', cp
        cphs[i] = arctan2(sin(cp),cos(cp))

    return cphs


def doy2date(year, doy, fmt="%Y-%b-%d"):
    """
    Convert day-of-year into the date in arbitrary date format
    (by default YYYY-Mon-dd, fmt).
    
    If specified, fmt must be a string with the format symbols:
    
    %a 	Locale's abbreviated weekday name. 	 
    %A 	Locale's full weekday name. 	 
    %b 	Locale's abbreviated month name. 	 
    %B 	Locale's full month name. 	 
    %c 	Locale's appropriate date and time representation. 	 
    %d 	Day of the month as a decimal number [01,31]. 	 
    %H 	Hour (24-hour clock) as a decimal number [00,23]. 	 
    %I 	Hour (12-hour clock) as a decimal number [01,12]. 	 
    %j 	Day of the year as a decimal number [001,366]. 	 
    %m 	Month as a decimal number [01,12]. 	 
    %M 	Minute as a decimal number [00,59]. 	 
    %p 	Locale's equivalent of either AM or PM.
    %S 	Second as a decimal number [00,61]. 
    %U 	Week number of the year (Sunday as the first day of the week)
        as a decimal number [00,53]. All days in a new year preceding
        the first Sunday are considered to be in week 0. 
    %w 	Weekday as a decimal number [0(Sunday),6]. 	 
    %W 	Week number of the year (Monday as the first day of the week)
        as a decimal number [00,53]. All days in a new year preceding
        the first Monday are considered to be in week 0.
    %x 	Locale's appropriate date representation. 	 
    %X 	Locale's appropriate time representation. 	 
    %y 	Year without century as a decimal number [00,99]. 	 
    %Y 	Year with century as a decimal number. 	 
    %Z 	Time zone name (no characters if no time zone exists). 	 
    %% 	A literal '%' character.
    
    See http://docs.python.org/2/library/time.html
    """
    doy = int(doy)
    if type(year) is str:
        syear = year
    else:
        syear = str(year)
    if fmt == None:
        fmt = "%Y-%b-%d"
    tstruc0 = strptime(syear, '%Y')   # Time structure for year-Jan-01
    tstruc = gmtime(timegm(tstruc0) + (doy-1)*86400) 

    return strftime(fmt, tstruc)



def doy2sec(year, doy):
    """
    Convert day-of-year into the seconds since Epoch
    """
    doy = int(doy)
    if type(year) is str:
        syear = year
    else:
        syear = str(year)
    tstruc0 = strptime(syear, '%Y')   # Time structure for year-Jan-01
    
    return timegm(tstruc0) + (doy-1)*86400



def read_vincent_cphs():
    """
    Read the Sgr A* closure phase data prepared by Vincent Fish.
    Stations (always CARMA-Hawaii-SMT in that order):
    C, D: CARMA, CA; 
    E: Single CARMA L, CA;
    F: Phased CARMA L, CA;
    G: Phased CARMA R, CA;
    J: JCMT, HI;  
    O: CSO L, HI;
    P: SMA L, HI;
    S: ARO/SMT, AZ; 
    T: SMT R, AZ; 
    Numbering:
    01: California (CARMA);     C, D, E, F, G
    02: Hawaii (JCMT);          J, O, P
    03: Arizona (SMT/ARO);      S, T

    basl = {'JC':'02-01', 'JD':'02-01', 'SJ':'03-02', 'SC':'03-01'}
    trisym = {'CJS':(1,2,3), 'CPS':(1,2,3), 'DJS':(1,2,3), 'DPS':(1,2,3), \
              'EJT':(1,2,3), 'FOS':(1,2,3), 'FPS':(1,2,3), 'GJT':(1,2,3)}
    -----------------------------------
    Returns:
      cphs: averaged over equal time spans closure phases
      ecph: averaged over equal time spans closure phase errors (1 sigma)
      cpix: baseline triplets for the closure phases (here only one triplet)
      tcphsec: UT time in seconds from UNIX Epoch
    """
    nhdr = 25
    
    fin = open('vincent_sgra_cphases_2009_2013.txt', 'r')
    for i in xrange(nhdr): print fin.readline(),   # Skip header

    #
    # Count lines
    #
    nlines = 0
    s = fin.readline()
    while s <> "":
        nlines = nlines + 1
        s = fin.readline()
    fin.close()

    #
    # Create data arrays
    #
    yd = np.zeros((nlines,2), np.int32)      # years and days of year
    #cpix = array(((1,2,3),), dtype=np.int32)
    utim = np.zeros((nlines), np.float)
    cphs0 = np.zeros((nlines), np.float32)
    ecph0 = np.zeros((nlines), np.float32)
    utsec = np.zeros((nlines), np.float)

    # Pattern to match a number string
    p = re.compile('[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d*\.?\d+[eE][+-]?\d+|' \
                   '[+-]?\d+\.?\d*|[+-]?\d*\.?\d+|[+-]?\d+')

    fin = open('sgra_cphases_vincent.txt', 'r')
    for i in xrange(nhdr): fin.readline(),   # Skip header

    #
    # Read lines into arrays
    #
    for i in xrange(nlines):
        s = fin.readline()
        dat = p.findall(s)
        yd[i,0] = np.int32(dat[0])  # year
        yd[i,1] = np.int32(dat[1])  # day of year
        dat = p.findall(s)
        utim[i] = float(dat[2])
        cphs0[i] = dat[3]
        ecph0[i] = dat[4]
        utsec[i] = doy2sec(yd[i,0], yd[i,1]) + 3600.*utim[i]
    fin.close()

    ibeg, iend = equal_spans(utim, singles=True)
    ncph = len(ibeg)
    
    cphs = np.zeros((ncph), np.float32)
    ecph = np.zeros((ncph), np.float32)
    tcphsec = np.zeros((ncph), np.float)
    cpix = np.zeros((ncph,3), dtype=np.int32)
    cpix[:,0] = 1; cpix[:,1] = 2; cpix[:,2] = 3;  # set (1,2,3) for all times
    
    #
    # Average the closures over equal time spans
    #
    for i in xrange(ncph):
        j0 = ibeg[i]
        j1 = iend[i]
        if j1 > j0+1:
            cphs[i] = np.average(cphs0[j0:j1])
            ecph[i] = np.average(ecph0[j0:j1])
        else:
            cphs[i] = cphs0[j0]
            ecph[i] = ecph0[j0]
        tcphsec[i] = doy2sec(yd[j0,0], yd[j0,1]) + 3600.*utim[j0]
    
    return cphs, ecph, cpix, tcphsec, cphs0, ecph0, utsec, utim
                                                 # End of read_vincent_cphs()


def equal_spans(dat, start=None, stop=None, singles=False):
    """
    Find spans of equal values in  a section [start:stop] of array dat[].
    If start and/or stop not specified, the entire dat[] is searched:
    if start=None, the beginning of dat[] is assumed.
    if stop=None, the end of dat[] is assumed.
    Otherwise, the dat[start:stop] segment is considered.
    The beginnings and ends of the spans are returned in arrays ibeg and iend.
    If the singles keyward is False, the "spans" consisting of a single
    element are ignored. Otherwise, if singles=True the pointers to single
    elements are included into ibeg[] and iend[].
    Unlike get_time_indices(), this function recognizes spans starting from
    length two and longer.
    """

    if start == None:
        start = 0
    if stop == None:
        stop = len(dat)

    ndat = stop - start
    
    ibeg = zeros(ndat, np.int)
    iend = zeros(ndat, np.int)

    i = start  # into dat[]
    j = 0  # into ibeg[], iend[]
    while i < stop-1:
        if dat[i] == dat[i+1]:
            ibeg[j] = i
            while dat[i] == dat[i+1]:
                i = i + 1
                if i >= stop-1: break
            i = i + 1
            iend[j] = i
            j = j + 1
        else:
            if singles:
                ibeg[j] = i
                iend[j] = i + 1
                j = j + 1
            i = i + 1

    ibeg = ibeg[:j]
    iend = iend[:j]

    return ibeg, iend



def read_vincent_cphs_coord():
    f1 = open('vincent_sgra_cphases_2009_2013.txt', 'r')
    f3 = open('vincent_sgra_coordinates_2009_2013.txt', 'r')

    cphs = []
    snr = []
    #yr = []    # years 
    #doy = []   # days of year
    t_cp = []   # UT in hours for cphs 
    t_uv = []   # UT in hours for uv coordinates 
    usf = []
    vsf = []
    ufp = []
    vfp = []

    s = f1.readline()
    while s <> "":
        if s[0] <> '#' and (s[0] <> '\n'):
            row = s.split()
            t_cp.append(row[2])
            cphs.append(row[6])
            snr.append(row[7])
        s = f1.readline()

    s = f3.readline()
    while s <> "":
        if (s[0] <> '#') and (s[0] <> '\n'):
            row = s.split()
            #yr.append(row[0])
            #doy.append(row[1])
            t_uv.append(row[2])
            usf.append(row[3])
            vsf.append(row[4])
            ufp.append(row[5])
            vfp.append(row[6])
        s = f3.readline()


    cphs = array(cphs, dtype=np.float32)    # Closure phases
    snr =  array(snr, dtype=np.float32)     # Bispectral signal/noise ratio
    #yr =  array(yr,   dtype=np.float32)     # days of year
    #doy = array(doy,  dtype=np.float32)     # years 
    t_cp = array(t_cp, dtype=np.float32)     # UT in hours 
    t_uv = array(t_uv, dtype=np.float32)     # UT in hours 
    usf = array(usf, dtype=np.float32)
    vsf = array(vsf, dtype=np.float32)
    ufp = array(ufp, dtype=np.float32)
    vfp = array(vfp, dtype=np.float32)
    
    #uvcp = vstack((usf,vsf,ufp,vfp)).T

    ## # SF + FP + PS = 0
    ## ups = -(usf + ufp)
    ## vps = -(vsf + vfp)

    f1.close()
    f3.close()

    #
    # Fill in the uv table, uvcp - lines of [usf,vsf,ufp,vfp] of cph length
    #
    ncph = len(cphs)
    nuv = len(usf)
    
    ibeg, iend = equal_spans(t_cp)

    cph_uv = zeros_like(usf)
    snr_uv = zeros_like(usf)
    uvcp = zeros((nuv,4), dtype=float32) # nuv lines of [usf,vsf,ufp,vfp]
    uvcp[:,0] = usf
    uvcp[:,1] = vsf
    uvcp[:,2] = ufp
    uvcp[:,3] = vfp

    i = 0    # Pointer into cphs[] or t_cp
    j = 0    # Pointer into ibeg[] or iend[]
    k = 0    # Pointer into cph_uv[] or uvcp[:,]

    while i < ncph:
        ib = ibeg[j]
        if i == ib:  # Multiple same-time location start encountered
            ie = iend[j]
            cph_uv[k] = mean(cphs[ib:ie])  # Average the cphs equal-time values
            snr_uv[k] = mean(snr[ib:ie])  # Average the cphs equal-time values
            #print 'i,j,k = ', i, j, k, ', ib, ie = ', ib, ie, \
            #      ', cph_uv[k] = ', cph_uv[k], ', cphs[ib:ie] = ', cphs[ib:ie]
            i = ie   # Move pointer into cphs past the equal-time interval
            j = j + 1
        else:        # Single time t_cp[i]
            cph_uv[k] = cphs[i]  # Just copy cphs
            snr_uv[k] = snr[i]  # Just copy snr
            #print 'i,j,k = ', i, j, k, ', cph_uv[k] = ', \
            #      cph_uv[k], ', cphs[i] = ', cphs[i]
            i = i + 1
        k = k + 1

    ecph = abs(1./snr_uv) # Error in closure phase at 1 Sigma in radians

        
    return uvcp, cph_uv, ecph, t_cp




def qk2fits():
    
    pass

    
