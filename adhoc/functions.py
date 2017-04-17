# import functions (produces .pyc file)


def b10tobx(x,b):
# x is an integer in base 10
# b is the base for the desired result
    """Given (DECIMAL_INTEGER,BASE),
    this returns array of base multiples.
    It may lead to unexpected results for BASE greater than 10.
    """
    import scipy as sp
    if x==0: d=1
    else: d=int(sp.log10(x)/sp.log10(b))+1
    z=sp.zeros(d,dtype=int)
    r=range(d)
    for i in r[::-1]:
        c=b**i
        f=x/c
        z[i]=f
        x=x-f*c
    return z[::-1]


def bei(n,x):
    import scipy as sp
    import scipy.special as ss
    z=(1j-1)*sp.sqrt(2)/2.
    return sp.imag(ss.jv(n,z*x))


def ber(n,x):
    import scipy as sp
    import scipy.special as ss
    z=(1j-1)*sp.sqrt(2)/2.
    return sp.real(ss.jv(n,z*x))


def sub_lin_base(y, n):
    """Subtract a linear baseline from a 1D numpy array
    based on the first and last n points.
    """
    import numpy as np
    y1 = np.mean(y[:n])
    y2 = np.mean(y[-n:])
    leny = len(y)
    x = 1.0*np.arange(leny)
    x1 = (n-1)/2.0
    #x2 = len(y) - n + x1
    #a = (y2 - y1) / (x2 - x1)
    a = (y2 - y1) / (leny - n)
    return y - (y1 + a*(x - x1))


def binaries(n):
# n is an integer
    """Given INTEGER,
    this returns a two-dimensional array of all the base-2 digits
    of the integers from zero to INTEGER.
    """
    import scipy as sp
    if n==0: d=1
    else: d=int(sp.log10(n)/sp.log10(2))+1
    out=sp.zeros([n+1,d],dtype=int)
    for i in xrange(n+1):
        dat=b10tobx(i,2)
        out[i,-sp.size(dat):]=dat
    return out


def binomial(n,p):
    """Return pdf of number of successes for n trials of process with
    probability of success p.
    """
    #import math
    import scipy as sp
    import scipy.special as ss
    #f=math.factorial
    g=ss.gamma
    x=sp.arange(n+1)
        #array of number-of-successes outcomes for n trials
    y=sp.zeros(n+1)
    #for i in x: y[i]=f(n)/(f(n-i)*f(i))*p**i*(1.-p)**(n-i)
    for i in x: y[i]=g(n+1)/(g(n-i+1)*g(i+1))*p**i*(1.-p)**(n-i)
    return y


def chances(x):
    """Given CHANCE_FROM_0_TO_1,
    an array of chances of success for a series of trials,
    this returns the chances
    for each total possible number of successes.
    """
    import scipy as sp
    x=sp.array(x)    # if given x is list rather than array
    nt=sp.size(x)    # number of trials
#    if nt > 15:    # may not be needed in python (was for idl version)
#        x=x[:15]
#        nt=15
    ns=nt+1    # number of possible numbers of total successes
    s=sp.zeros(ns)
        # one place for each possible number of total successes
    p=binaries(long(2)**nt)
    p=p[1:,1:]    # all possible permutations of outcomes
    c=p*x+1-p-(1-p)*x    # array of chances for each member of p
    ap=sp.sum(p,axis=1)
    # unordered, unsummed list of possible total number of successes
    ac=sp.sum(c,axis=1)/sp.sum(c)
    # corresponding array of unordered, unsummed chances
    for i in range(ns): s[i]=sp.sum(ac*sp.where(ap==i,1,0))
    # range(ns) is index
    #of labels covering all possible numbers of total successes
    # now generate p an array describing all possible detailed outcomes
    return s


def cmapnames():
    from matplotlib.cm import datad
    return sorted(m for m in datad if not m.endswith('_r'))


def cmap(name):
    from matplotlib.cm import cmap_d
    return cmap_d[name]


def corr(a, b):
    """Return my definition of correlation between two arrays."""
    import numpy as np
    A = a - np.mean(a)
    B = b - np.mean(b)
    return np.mean(A*B) / rms(A) / rms(B)


def cull(a, n):
    """Given a 1D array, return the array with all values removed
    except for every n values starting with the index zero.
    """
    import scipy as sp
    b=a[0].copy()
    for i in range(0,len(a),n): b=sp.append(b,a[i])
    return b[1:]


def delta(t):
    """Given an equally spaced 1D or 2D array of values, return the
    average spacing between those values.
    """
    import numpy as np
    dimt = dimensions(t)
    if dimt == 1:
        return (np.max(t)-np.min(t))/(len(t)-1)
    elif dimt == 2:
        return (np.max(t[0])-np.min(t[0]))/(len(t[0])-1)


def deriv(z, x=None, axis=1, npts=3, cyclic=False):
    """Given an ordinate array z and possibly an unequally spaced
    abscissa array x, output the numerical derivative based on the
    number of points used in a Lagrangian interpolation (see Hildebrand,
    1956).  This works only for data on nonperiodic abscissas.  The
    number of points used npts can be 3 or 5 only. For 2D arrays, axis=0
    means the y derivative and 1 the x.  The default is axis=1, which
    means x derivative.  If cyclic is True, then no endpoint
    calculations are use, but the bulk calculation is used for the
    entire abscissa domain.  Note this will not work properly if both
    endpoints are included in the abscissa (e.g. both 0 and 2pi).  This
    function is based on KM's "derivative.pro" for IDL, which has been
    tested to be practically the same as IDL's "deriv.pro" for
    comparable cases.  I have tested it in python using test_deriv
    below.
    """
    import numpy as np
    if axis == 1:
        # The x-derivative of the 2D z is wanted.
        z = z.T
        if x is not None: # x is an array.
            x = x.T
    #elif axis == 0:
        # z is 1D or the y-derivative of the 2D z is wanted.
    zm1 = np.roll(z, 1, axis=0)
        # 'zminus1' has z values from lower abscissas in particular
        # indices because we are rolling forward by 1.
    zp1 = np.roll(z, -1, axis=0)
        # 'zplus1' has z values from higher abscissas in particular
        # indices because we are rolling forward by -1.
    if x is not None: # x is an array, i.e. could be unevenly spaced.
        xm1 = np.roll(x, 1, axis=0)
        xp1 = np.roll(x, -1, axis=0)
    if npts == 3:
        if x is None: # x is a scalar, i.e. evenly spaced.
            # Bulk calculation
            d = zp1 - zm1
            if not cyclic:
                # Endpoint calculation
                d[0] = -3.0*z[0] + 4.0*z[1] - z[2]
                d[-1] = z[-3] - 4.0*z[-2] + 3.0*z[-1]
            d = d/2.0
        else: # x is an array, i.e. could be unevenly spaced.
            # Bulk calculation
            dx01 = xm1 - x
            dx12 = x - xp1
            dx02 = xm1 - xp1
            d = zm1*dx12/dx01/dx02 - z*(1.0/dx01 - 1.0/dx12) \
                - zp1*dx01/dx02/dx12
            if not cyclic:
                # Endpoint calculation
                dx01 = x[0] - x[1] 
                dx12 = x[1] - x[2] 
                dx02 = x[0] - x[2]
                d[0] = z[0]*(dx01 + dx02)/dx01/dx02 \
                    - z[1]*dx02/dx12/dx01 + z[2]*dx01/dx02/dx12
                dx01 = x[-3] - x[-2] 
                dx12 = x[-2] - x[-1] 
                dx02 = x[-3] - x[-1]
                d[-1] = -z[-3]*dx12/dx01/dx02 + z[-2]*dx02/dx12/dx01 \
                    - z[-1]*(dx02 + dx12)/dx02/dx12		
    elif npts == 5:
        zm2 = np.roll(z, 2, axis=0)
        zp2 = np.roll(z, -2, axis=0)
        if x is None: # x is a scalar, i.e. evenly spaced.
            # Bulk calculation
            d = -zp2 + 8.0*zp1 - 8.0*zm1 + zm2
            if not cyclic:
                # Endpoint calculation
                d[0] = -25.0*z[0] + 48.0*z[1] - 36.0*z[2] + 16.0*z[3] \
                    - 3.0*z[4]
                d[1] = -3.0*z[0] - 10.0*z[1] + 18.0*z[2] - 6.0*z[3] \
                    + z[4]
                d[-2] = 3.0*z[-1] + 10.0*z[-2] - 18.0*z[-3] \
                    + 6.0*z[-4] - z[-5]
                d[-1] = 25.0*z[-1] - 48.0*z[-2] + 36.0*z[-3] \
                    - 16.0*z[-4] + 3.0*z[-5]
            d = d/12.0
        else: # x is an array, i.e. could be unevenly spaced.
            # Bulk calculation
            xm2 = np.roll(x, 2, axis=0)
            xp2 = np.roll(x, -2, axis=0)
            dx01 = xm2 - xm1 
            dx12 = xm1 - x 
            dx23 = x - xp1 
            dx34 = xp1 - xp2
            dx02 = xm2 - x 
            dx13 = xm1 - xp1 
            dx24 = x - xp2 
            dx03 = xm2 - xp1 
            dx14 = xm1 - xp2 
            dx04 = xm2 - xp2
            d = -zm2*dx12*dx23*dx24/dx01/dx02/dx03/dx04 \
                + zm1*dx23*dx24*dx02/dx12/dx13/dx14/dx01 \
                + z*(-1.0/dx12 + 1.0/dx23 - 1.0/dx02 + 1.0/dx24) \
                - zp1*dx12*dx24*dx02/dx13/dx23/dx34/dx03 \
                + zp2*dx12*dx23*dx02/dx14/dx24/dx34/dx04
            if not cyclic:
                # Endpoint calculation
                dx01 = x[0] - x[1] 
                dx12 = x[1] - x[2]
                dx23 = x[2] - x[3] 
                dx34 = x[3] - x[4]
                dx02 = x[0] - x[2] 
                dx13 = x[1] - x[3]
                dx24 = x[2] - x[4] 
                dx03 = x[0] - x[3]
                dx14 = x[1] - x[4] 
                dx04 = x[0] - x[4]
                d[0] = z[0]*(1.0/dx01 + 1.0/dx02 + 1.0/dx03 \
                    + 1.0/dx04) \
                    - z[1]*dx02*dx03*dx04/dx01/dx12/dx13/dx14 \
                    + z[2]*dx01*dx03*dx04/dx02/dx12/dx23/dx24 \
                    - z[3]*dx01*dx02*dx04/dx03/dx13/dx23/dx34 \
                    + z[4]*dx01*dx02*dx03/dx04/dx14/dx24/dx34
                d[1] = z[0]*dx12*dx13*dx14/dx01/dx02/dx03/dx04 \
                    + z[1]*(-1.0/dx01 + 1.0/dx12 + 1.0/dx13 \
                    + 1.0/dx14) \
                    - z[2]*dx01*dx13*dx14/dx02/dx12/dx23/dx24 \
                    + z[3]*dx01*dx12*dx14/dx03/dx13/dx23/dx34 \
                    - z[4]*dx01*dx12*dx13/dx04/dx14/dx24/dx34
                dx01 = x[-5] - x[-4] 
                dx12 = x[-4] - x[-3]
                dx23 = x[-3] - x[-2] 
                dx34 = x[-2] - x[-1]
                dx02 = x[-5] - x[-3] 
                dx13 = x[-4] - x[-2]
                dx24 = x[-3] - x[-1] 
                dx03 = x[-5] - x[-2]
                dx14 = x[-4] - x[-1] 
                dx04 = x[-5] - x[-1]
                d[-2] = z[-5]*dx23*dx34*dx13/dx02/dx03/dx04/dx01 \
                    - z[-4]*dx23*dx34*dx03/dx12/dx13/dx14/dx01 \
                    + z[-3]*dx34*dx03*dx13/dx23/dx24/dx02/dx12 \
                    - z[-2]*(1.0/dx23 - 1.0/dx34 + 1.0/dx03 \
                    + 1.0/dx13) \
                    - z[-1]*dx23*dx03*dx13/dx24/dx34/dx04/dx14
                d[-1] = -z[-5]*dx34*dx14*dx24/dx03/dx04/dx01/dx02 \
                    + z[-4]*dx34*dx04*dx24/dx13/dx14/dx01/dx12 \
                    - z[-3]*dx34*dx04*dx14/dx23/dx24/dx02/dx12 \
                    + z[-2]*dx04*dx14*dx24/dx34/dx03/dx13/dx23 \
                    - z[-1]*(1.0/dx34 + 1.0/dx24 + 1.0/dx14 + 1.0/dx04)
    if axis == 0:
        return d
    elif axis == 1:
        return d.T


def test_deriv(n=100):
    """Test deriv(z, x=None, axis=1, npts=3, cyclic=False) in 1D.
    The answer for n=100 should be about like this:
err3 0.000465412677018 err3x 0.000465412677018 err3_cyc 0.00046516578377
err5 3.72541361994e-07 err5x 3.72541361878e-07 err5_cyc 3.6718007808e-07
    and the plots look right.
    """
    import numpy as np
    import numpy.random as nr
    import matplotlib.pyplot as mp
    x = 2.0*np.pi*np.arange(n)/n
    z = -np.cos(x)
    y = np.sin(x)
    x_ran = 2.0*np.pi*np.sort(nr.rand(n))
    z_ran = -np.cos(x_ran)
    y_ran = np.sin(x_ran)
    y3 = deriv(z, x=None, npts=3, cyclic=False)/delta(x)
    y3x = deriv(z, x=x, npts=3, cyclic=False)
    y3_ran = deriv(z_ran, x=x_ran, npts=3, cyclic=False)
    y3_cyc = deriv(z, x=None, npts=3, cyclic=True)/delta(x)
    y5 = deriv(z, x=None, npts=5, cyclic=False)/delta(x)
    y5x = deriv(z, x=x, npts=5, cyclic=False)
    y5_ran = deriv(z_ran, x=x_ran, npts=5, cyclic=False)
    y5_cyc = deriv(z, x=None, npts=5, cyclic=True)/delta(x)
    err3 = rms(y - y3)
    err3x = rms(y - y3x)
    err3_cyc = rms(y - y3_cyc)
    err5 = rms(y - y5)
    err5x = rms(y - y5x)
    err5_cyc = rms(y - y5_cyc)
    print 'err3', err3, 'err3x', err3x, 'err3_cyc', err3_cyc
    print 'err5', err5, 'err5x', err5x, 'err5_cyc', err5_cyc
    mp.clf()
#    mp.subplot(2, 1, 1)
#    mp.plot(x, z, label='actual')
#    mp.plot(x_ran, z_ran, label='random')
#    mp.subplot(2, 1, 2)
    mp.plot(x, y, label='actual')
    mp.plot(x, y3, label='3')
    mp.plot(x, y3x, label='3x')
    mp.plot(x_ran, y3_ran, 'o-', label='3 random')
    mp.plot(x, y3_cyc, label='3 cyclic')
    mp.plot(x, y5, label='5')
    mp.plot(x, y5x, label='5x')
    mp.plot(x_ran, y5_ran, 'o-', label='5 random')
    mp.plot(x, y5_cyc, label='5 cyclic')
    mp.legend(loc='best')
    mp.xlim(xmax=2.0*np.pi)


# Previous version of deriv worked only for equally spaced abscissas:
#def deriv(z, axis=1, npts=3):
#    """Given an ordinate array z, ouput the numerical derivative based
#    on the number of points used in a Lagrangian interpolation
#    (Hildebrand, 1956).  This works only for data on regularly spaced,
#    nonperiodic abscissas, assumed here to have step size = 1.
#    The number of points used npts can be 3 or 5 only.
#    For 2D arrays, axis=0 means the y derivative and 1 the x.
#    The default is axis=1, which means x derivative, as in python
#    diff().
#    """
#    import numpy as np
#    if axis == 1:
#        # The x-derivative of the 2D z is wanted.
#        #z = np.flipud(np.rot90(z, 1)) # Not sure why I was doing this.
#        z = z.T
#    #elif axis == 0:
#        # z is 1D or the y-derivative of the 2D z is wanted.
#    p1 = np.roll(z, 1, axis=0)
#    m1 = np.roll(z, -1, axis=0)
#    if npts == 5:
#        p2 = np.roll(z, 2, axis=0)
#        m2 = np.roll(z, -2, axis=0)
#        d = -m2 + 8*m1 - 8*p1 + p2   #bulk calculation
#        #endpoint calculation
#        d[0] = -25.*z[0] + 48.*z[1] - 36.*z[2] + 16.*z[3] - 3.*z[4]
#        d[1] = -3.*z[0] - 10.*z[1] + 18.*z[2] - 6.*z[3] + z[4]
#        d[-2] = ( 3.*z[-1] + 10.*z[-2] - 18.*z[-3] + 6.*z[-4]
#            - z[-5] )
#        d[-1] = ( 25.*z[-1] - 48.*z[-2] + 36.*z[-3] - 16.*z[-4]
#            + 3.*z[-5] )
#        d = d/12.
#    elif npts == 3:
#        d = 0.5*(m1 - p1)
#        d[0] = m1[0] - z[0]
#        d[-1] = z[-1] - p1[-1]
#    if axis == 0:
#        return d
#    elif axis == 1:
#        #return np.flipud(np.rot90(d, 3)) # Not sure again.
#        return d.T


def dvars(Dictionary):
    """Transform the items in a dictionary into separate variables.
    Note there must not be a key in your dictionary called Dictionary.
    """
    for i in range(len(Dictionary)):
        #print i, Dictionary.keys()[i]
        exec(Dictionary.keys()[i] + ' = Dictionary.values()[' + 'i]')
    exec('r = ' + ', '.join(Dictionary.keys()))
    return r


def svars(Dictionary):
    """Return an exec string to use dvars.  That is, the user of this
    function should do this:
        import functions # This module
        d = dict(a=1, b=2, c=3)
        exec(functions.svars(d) + 'functions.dvars(d)')
    Note there must not be a key in your dictionary called Dictionary.
    """
    return ', '.join(Dictionary.keys()) + ' = '


def dimensions(x):
    from numpy import shape
    return len(shape(x))


# Differential operators for 2D (z, r) arrays representing
# cylindrically symmetric functions (geom='tor' from NIMROD parlance)
# or functions symmetric in the direction orthogonal to (z, r)
# for Cartesian coordinate systems (geom='lin'):
#    deriv_delta, grad, div, curl, laps, lapv, and delst
# These all assume uniform grid spacing defaulted to 1 with no points
# at r=0.
# Not all of them have the same format for positional inputs.


def deriv_delta(r=None):
    """Determine the abscissa delta to be used in differential
    operators.
    """
    if r is None:
        return 1.0
    dimr = dimensions(r)
    if (dimr == 0) or (dimr == 1 and len(r)) == 1:
        return r
    else:
        return delta(r)


def get_deriv_r_z(f, r=None, z=None):
    """Return the proper r and z values for differential operators
    based on f, the ordinate.
    """
    import numpy as np
    if r is None:
        r = 1.0 + np.arange(len(f[0]))
    if z is None:
        z = np.array([1.0])
    return r, z


def grad(f, r=None, z=None, npts=3):
    """Given f, a 2D array of scalar data, find its gradient.  The
    optional positional variables r and z can be 0D, 1D, or 2D and
    default to 1D and len==1 with a value of 1.
    """
    r, z = get_deriv_r_z(f, r, z)
    dfdr = deriv(f, npts=npts, axis=1)/deriv_delta(r)
    dfdz = deriv(f, npts=npts, axis=0)/deriv_delta(z.T)
    return dfdr, dfdz


def div(Ar, Az, r=None, z=None, npts=3, geom='tor'):
    """Given Ar and Az, two 2D arrays of the (r, z) components of a
    vector A, and r and z, each a 1D or 2D array, find the
    divergence of A.  If r is not input, then it is assumed to be of
    unit spacing and to start at 1.
    """
    r, z = get_deriv_r_z(Ar, r, z)
    if geom == 'tor':
        return deriv(r*Ar, npts=npts, axis=1)/deriv_delta(r)/r \
            + deriv(Az, npts=npts, axis=0)/deriv_delta(z.T)
    elif geom == 'lin':
        return deriv(Ar, npts=npts, axis=1)/deriv_delta(r) \
            + deriv(Az, npts=npts, axis=0)/deriv_delta(z.T)


def curl(Ar, Aphi, Az, r=None, z=None, npts=3, geom='tor'):
    """Given Ar, Aphi, and Az, three 2D arrays of the (r, phi, z)
    components of a vector A, and r and z, each a 1D or 2D array,
    find the curl of A.  If r is not input, then it is assumed
    to be of unit spacing and to start at 1.
    """
    r, z = get_deriv_r_z(Ar, r, z)
    rcomp = -deriv(Aphi, npts=npts, axis=0)/deriv_delta(z.T)
    phicomp = ( deriv(Ar, npts=npts, axis=0)/deriv_delta(z.T)
        - deriv(Az, npts=npts, axis=1)/deriv_delta(r) )
    if geom == 'tor':
        zcomp = deriv(r*Aphi, npts=npts, axis=1)/deriv_delta(r)/r
    elif geom == 'lin':
        zcomp = deriv(Aphi, npts=npts, axis=1)/deriv_delta(r)
    return rcomp, phicomp, zcomp


def laps(f, r=None, z=None, npts=3, geom='tor'):
    """Given f, a 2D array of scalar data, and r and z, each a 1D or 2D
    array, find the Laplacian of f.  If r is not input, then it
    is assumed to be of unit spacing and to start at 1.
    """
    r, z = get_deriv_r_z(f, r, z)
    dfdr, dfdz = grad(f, r, z, npts=npts)
    if geom == 'tor':
        return deriv(r*dfdr, npts=npts, axis=1)/deriv_delta(r)/r \
            + deriv(dfdz, npts=npts, axis=0)/deriv_delta(z.T)
    elif geom == 'lin':
        return deriv(dfdr, npts=npts, axis=1)/deriv_delta(r) \
            + deriv(dfdz, npts=npts, axis=0)/deriv_delta(z.T)


def lapv(Ar, Aphi, Az, r=None, z=None, npts=3, geom='tor'):
    """Given Ar, Aphi, and Az, three 2D arrays of the (r, phi, z)
    components of a vector A, and r and z, each a 1D or 2D
    array, find the Laplacian of A.  If r is not input, then it is
    assumed to be of unit spacing and to start at 1.
    """
    r, z = get_deriv_r_z(Ar, r, z)
    if geom == 'tor':
        rcomp = laps(Ar, r, z, npts=npts, geom=geom) - Ar/r**2
        phicomp = laps(Aphi, r, z, npts=npts) - Aphi/r**2
    elif geom == 'lin':
        rcomp = laps(Ar, r, z, npts=npts, geom=geom)
        phicomp = laps(Aphi, r, z, npts=npts)
    zcomp = laps(Az, r, z, npts=npts)
    return rcomp, phicomp, zcomp


def delst(f, r=None, z=None, npts=3, geom='tor'):
    """Given f, a 2D array of scalar data, and r and z, each a 1D or 2D
    array, find the elliptic operator on f, or del-star f.  If r is not
    input, then it is assumed to be of unit spacing and to start at 1.
    """
    r, z = get_deriv_r_z(f, r, z)
    dfdr, dfdz = grad(f, r, z, npts=npts)
    if geom == 'tor':
        return r*r*div(dfdr/r/r, dfdz/r/r, r, z, npts=npts, geom=geom)
    elif geom == 'lin':
        return laps(f, r, z, npts=npts, geom=geom)


#def lcm(*x):
def lcm(x):
#    """Return the least common multiple of a set of numbers.  Call as
#    'lcm(2, 3, 4, 5)', e.g.
#    """
    """Return the least common multiple of a listoid of numbers.  Call
    as, e.g.,
        a = [2, 3, 4, 5]
        b = lcm(a)
    """
    from fractions import gcd
    z = x[0]
    for y in x[1:]:
        z = z*y//gcd(z, y)
    return z


def remap_cmap(target=None, source=0.5, cmap='RdBu', n=256):
    """Takes cmap, an instance of
    matplotlib.colors.LinearSegmentedColormap, and distorts it, so that
    the color that did appear for a value of source will now appear for
    a value of target, and higher and lower values will be either
    compressed or expanded as needed.  The purpose of this is to make
    it easy to redefine a desired colormap to show maximum contrast at
    some desired value such as target.  The inputs target and source
    must be floats between 0 and 1, inclusive, for minimum and maximum
    value of the data, respectively.  For instance, to get the
    source=midpoint of the color map itself to show up at the
    target=zero of the data, set
    target=-min(data)/(max(data)-min(data)).
    This can be done by sending the data as target=data.
    """
    import numpy as np
    import matplotlib.colors as mc
    import matplotlib.cm as cm
    exec('cmap = cm.' + cmap)
    if np.size(target) > 1:
        h = np.histogram(target, n)
        #target = (np.mean(np.where(h[0] == max(h[0]) ) ) + 0.5)/n
        target = -np.min(target)/(np.max(target)-np.min(target))
    x = 1.0*np.arange(n)/n
    wle = np.where(x <= target)
    wgt = np.where(x > target)
    y = np.zeros(n)
    y[wle] = source/target*x[wle]
    y[wgt] = source + (1.0 - source)/(1.0 - target)*(x[wgt] - target)
    return mc.ListedColormap(cmap(y))


def done(day, hour, minute, value, end=7027., time0=15.+48./60.):
    """Tells when an iterative process started at 'time0' [hours],
    now at some 'value' going until the 'end' value, will be done,
    using which 'day' starting from 0, which 'hour' of the day,
    and which 'minute' of the hour at which the value is read.
    """
    import numpy as np
    time = day * 24. + hour + minute / 60.
    duration = (time - time0) / value * end
    done = time0 + duration
    day = np.fix(done / 24.)
    hour = done - day * 24.
    minute = round((hour - np.fix(hour)) * 60.)
    hour = np.fix(hour)
    print day, hour, minute


#def even(x):
#    """Output True for even x, False for odd."""
#    if x/2. == int(x)/2: return True
#    else: return False


#def odd(x):
#    """Output True for odd x, False for even."""
#    if x/2. != int(x)/2: return True
#    else: return False


def exists_in(x, a):
    """True if x is a member of the listoid a, False if not."""
    from numpy import size, where
    if size(where(a == x)) == 0:
        return False
    else:
        return True


def figure_kwargs(ftype=None, subplot=(None,None), ss=None,
    legendscale=None):
    """Given ftype=None, 'landscape', 'square', or 'portrait',
    return a dictionary containing
    figsize for the figure, called 'figsize',
    fontsize for title, legend, label, and ticks,
    called 'titlesize', 'legendsize', 'labelsize', and 'ticksize',
    and the parameters for pyplot function subplots_adjust,
    which are in their own dictionary called 'subplots_adjust',
    composed of
    'left', 'bottom', 'right', 'top', 'wspace', and 'hspace'.
    Tuple keyword subplot is the number (Nhigh,Nwide) of subplots 
    for which the adjustments are made,
    in the height and width directions.
    Keyword ss is whether axes labels have sub/superscripts,
    in which case all fontsizes would need to be scaled up -- not used.
    Keyword legendscale is whether the legend
    should be scaled up like the title
    -- not(True) means like the labels.
    """
    if subplot == (None,None): subplot=1,1 #Nhigh,Nwide
    if ftype is None:
        figsize=None    #default is 8,6
        titlesize=None  #default is 'large' (=14.5 pt)
        legendsize=None  #default is 'large' (=14.5 pt)
        labelsize=None  #default is 'medium' (=12 pt)
        ticksize=None  #default is 'medium' (=12 pt)
        subplots_adjust={
            #defaults
            'left':0.12, 'bottom':0.1, 'right':0.9,
            'top':0.9, 'wspace':0.2, 'hspace':0.2}
#            'left':None, 'bottom':None, 'right':None,
#            'top':None, 'wspace':None, 'hspace':None}
    else:
        #For 8" wide figures scaled to 3.25" (maximum PoP width, e.g.),
        #fontsize of 20 pt gives a scaled fontsize >= 8 pt,
        #but fontsize of 29 pt is the least such that sub/superscripts
        #also give scaled fontsize >= 8 pt.
        #This code does not allow for sub/superscripts,
        #and simply uses titlesize
        #scaled to 24 to keep about the same proportions
        #to labelsize and ticksize as in the default case,
        #and uses legendsize
        #scaled relatively down compared to the default case
        #to keep the legend small compared to the plotted data.
        #Note mathtext (used with values like r'$latex$')
        #does not seem to look as big as the fontsize value,
        #so avoid using mathtext.
        
        titlesize=24
        if legendscale: legendsize=24
        else: legendsize=20
        labelsize=20
        ticksize=20
        if ftype == 'landscape': figsize=8,6
#            #good values for landscape
#            subplots_adjust={
#                'left': 0.12, 'bottom': 0.12, 'right':0.94,
#                'top':0.92, 'wspace':0.45, 'hspace':0.65}
        elif ftype == 'square': figsize=8,8
        elif ftype == 'portrait': figsize=8,10.6666
        else: figsize=8,8
        Width=figsize[0]    #actual sizes in inches
        Height=figsize[1]
        #Original values
#        Dwidth=1.205    #actual width in inches needed for fontsizes
#        Dheight=1.177    #actual height in inches needed for fontsizes
        #Convenient values
        D=1.125
        Dwidth=D    #actual width in inches needed for fontsizes
        Dheight=D   #actual height in inches needed for fontsizes
        #These next two fractions of whole page
        #don't change for width always = 8".
        left=0.12
        right=0.96
        #These next two fractions of whole page
        #must get bigger (i.e. further from 0 or 1, respectively)
        #as the height of page decreases, and vice versa;
        #i.e. they vary inversely.
        bottom=0.12*6./Height
        top=1.-0.08*6./Height
        #These next two fractions of a single subplot size
        #vary in a complicated way with the other parameters
        #to keep a constant actual spacing available for text.
        Nhigh=subplot[0]
        Nwide=subplot[1]
        wspace=Dwidth*Nwide/(Width*(right-left)-Dwidth*(Nwide-1))
        hspace=Dheight*Nhigh/(Height*(top-bottom)-Dheight*(Nhigh-1))
        subplots_adjust={
            'left':left, 'bottom':bottom, 'right':right,
            'top':top, 'wspace':wspace, 'hspace':hspace}
    return {
        'figsize':figsize, 'titlesize':titlesize,
        'legendsize':legendsize, 'labelsize':labelsize,
        'ticksize':ticksize, 'subplots_adjust':subplots_adjust}


def find_peak(x, y, z, n=500, s=1, test=False):
    """Given 2D arrays of x, y, and z, the number of points to use n,
    and the sign of the peak s, return the x0, y0, and z0 of the
    interpolated summit, assuming the peak is shaped like an elliptic
    paraboloid.

    n=100 is enough for more than about 90% of randomly chosen elliptic
    paraboloids, but it looks like n=300 or above is needed for typical
    Grad-Shafranov equilibrium poloidal flux plots.
    """
    import numpy as np
    z = z.flatten()
    if s == -1:
        z = -z
    indices = np.argsort(z)[-n:]
    x = x.flatten()[indices]
    y = y.flatten()[indices]
    z = z[indices]
    #print 'x, y, z', x, y, z
    abscissa = np.array([x**2, y**2, x*y, x, y, np.ones(len(x))])
    zfit, par, spar, coeff, chi2 = llsfit(abscissa, z)
    #print 'zfit', zfit
    #print 'chi2', chi2
    A, B, C, D, E, F = par
    #print A, B, C, D, E, F
    a, b, t, x0, y0, z0 = fit_pars_to_sol_pars(A, B, C, D, E, F)
    #print 'a, b, t, x0, y0, z0', a, b, t, x0, y0, z0
    if s == -1:
        z = -z
        z0 = -z0
    if not test:
        return x0, y0, z0
    else:
        return x0, y0, z0, x, y, z


def fit_pars_to_sol_pars(A, B, C, D, E, F):
    """Given the fitting parameters for a elliptic paraboloid fitting
    problem, return the solution parameters for the paraboloid itself.
    """
    import numpy as np
    #print 'A, B, C, D, E, F', A, B, C, D, E, F
    R = np.sqrt((A - B)**2 + C**2)
    if A == B and C == 0:
        a = np.sqrt(-A)
        b = np.sqrt(-B)
    elif A < B:
        a = np.sqrt((-A - B + R)/2.0)
        b = np.sqrt((-A - B - R)/2.0)
        if C != 0:
            t = -np.arcsin(C/R)/2.0
    else: # A > B
        a = np.sqrt((-A - B - R)/2.0)
        b = np.sqrt((-A - B + R)/2.0)
        if C != 0:
            t = np.arcsin(C/R)/2.0
    if C == 0:
        t = 0.0
#    x0 = 0.125*(4.0*D*a**2*b**2 + (a**2 - b**2) \
#        *(D*(a**2 - b**2)*np.sin(2.0*t) \
#        - 2.0*E*(a**2*np.cos(t)**2 + b**2*np.sin(t)**2)) \
#        *np.sin(2.0*t)) \
#        /(a**2*b**2*(a**2*np.cos(t)**2 + b**2*np.sin(t)**2))
#    y0 = 0.25*(-D*(a**2 - b**2)*np.sin(2.0*t) \
#        + 2.0*E*(a**2*np.cos(t)**2 + b**2*np.sin(t)**2))/(a**2*b**2) 
#    z0 = F + a**2*x0**2*np.cos(t)**2 + a**2*x0*y0*np.sin(2*t) \
#        + a**2*y0**2*np.sin(t)**2 + b**2*x0**2*np.sin(t)**2 \
#        - b**2*x0*y0*np.sin(2*t) + b**2*y0**2*np.cos(t)**2
    G = 4.0*A*B - C**2
    x0 = (-2.0*B*D + C*E)/G
    y0 = (-2.0*A*E + C*D)/G
    z0 = F - A*x0**2 - B*y0**2 - C*x0*y0
    return a, b, t, x0, y0, z0


def sol_pars_to_fit_pars(a, b, t, x0, y0, z0):
    """Given the solution parameters for an elliptic paraboloid, return
    the fitting parameters for the problem.
    """
    import numpy as np
    A = -a**2*np.cos(t)**2 - b**2*np.sin(t)**2
    B = -b**2*np.cos(t)**2 - a**2*np.sin(t)**2
    C = (b**2 - a**2)*np.sin(2.0*t)
#    D = 2.0*a**2*x0*np.cos(t)**2 + a**2*y0*np.sin(2.0*t) \
#        + 2.0*b**2*x0*np.sin(t)**2 - b**2*y0*np.sin(2.0*t)
#    E = a**2*x0*np.sin(2.0*t) + 2.0*a**2*y0*np.sin(t)**2 \
#        - b**2*x0*np.sin(2.0*t) + 2.0*b**2*y0*np.cos(t)**2
#    F = z0 - a**2*x0**2*np.cos(t)**2 - a**2*x0*y0*np.sin(2.0*t) \
#        - a**2*y0**2*np.sin(t)**2 - b**2*x0**2*np.sin(t)**2 \
#        + b**2*x0*y0*np.sin(2.0*t) - b**2*y0**2*np.cos(t)**2
    D = -2.0*A*x0 - C*y0
    E = -2.0*B*y0 - C*x0
    F = z0 + A*x0**2 + B*y0**2 + C*x0*y0
    return A, B, C, D, E, F


def imshow(*kwargs):
    from matplotlib.pyplot import imshow
    imshow(*kwargs, interpolation='nearest')


def interpnd(new, data, old=None, unflatten=False):
    """Using scipy.ndimage.map_coordinates,
    interpolate a numerical function f(x, y, z, ...)
    of equally spaced cooordinates in N-space,
    where N should be 3 or greater,
    since other simple functionality accomplishes this for N <= 2,
    namely things like scipy.interpolate.interp1d and interp2d.
    
    Arguments:
    new: the abscissa points
        at which ordinate values are to be interpolated,
        given as an array (or list) of shape(new) == (npts, ndims),
        which will be transposed for use in map_coordinates.
        These can be real abscissa units, if old is given to set them,
        or assumed to be in index units, if old is not given.
    data: the ordinate values of the function to be interpolated,
        given as an array ordered according to abscissa values of
        shape(data) == (nxpts, nypts, nzpts, ...).
    old: if given, the abscissa values of the function to be
        interpolated, as a list like [x, y, z, ...]
        with each member an array or list
        of the regular, monotonically changing values
        in that dimension.
        If not given, so that old is None, the abscissa values in new
        are assumed to already be in index units.
    unflatten: if not False, use value of unflatten as the ndims
        to unflatten the input array new.

    Based on "N-D interpolation for equally-spaced data"
    on the Scipy Cookbook/Interpolation site,
    http://scipy.org/Cookbook/Interpolation
        #head-6a59d9e4e9645b568f7877809306bff1130fa34b ,
    the first answer, http://stackoverflow.com/a/6238859 ,
    to the question on
    "Multivariate spline interpolation in python/scipy?" at
    http://stackoverflow.com/questions/6238250/
        multivariate-spline-interpolation-in-python-scipy ,
    and the python help for scipy.ndimage.map_coordinates.
    """
    import numpy as np
    import scipy.ndimage as sn
    if type(new) == list:
        new = np.array(new)
    #print 'new', np.shape(new)
    if unflatten != False:
        new = np.array([[new[i]] for i in range(unflatten)]).T
    #print 'new', np.shape(new)
    if old is not None:
        # If necessary, recast new into index units.
        # First, find the starting value and number
        # of old abscissa values for each dimension.
        xyz0 = []
        dxyz = []
        ndims = np.shape(new.T)[0]
        for i in range(ndims):
            xyz0.append(old[i][0])
            dxyz.append((old[i][-1] - xyz0[i])/(len(old[i]) - 1.0)) 
        # Use implicitly broadcasted operations
        # to get new in index units.
        new = (new - xyz0) / dxyz
    return sn.map_coordinates(data, new.T, mode='nearest', order=3)


def llsfit(X, y, s=None, svd=True, Nweights=None, verbose=False):
    """Linear least-squares fit:

    yfit, a, sa, C, chi2 = llsfit(X, y)

    Return fit parameters 'a', their uncertainties, and chi^2
    for a linear fit of the data to X{i,j}a{j} = y{i},
    where i is the index for the experimental trial,
    j is the index for the type of X abscissa data.
    Note this is not the way X is input (see below).

    The matrix problem to solve is alpha.a = beta,
    where a is the array of parameters sought,
    alpha_kl = Sum_i(X_ki*X_li/sigma^2),
    a_j = a column matrix of j parameters,
    beta_j = column matrix of j values Sum_i(X_ji*y_i).
    This follows the treatment in Bevington, or in Numerical Recipes.

    * X is the array of types of abscissa data from the experimental
        trials, and its data is assumed to be perfect with no
        measurement uncertainty.
    * X should be input either as a standard 1D array for the case of
        one type of abscissa data, or as a 2D array with shape [j, i],
        where j is the index for the type of X abscissa data and i is
        the index for the experimental trial.
    * y is the array of ordinate values from the experimental trials.
    * s can be given as a single float or an array
        of the y-uncertainty(ies), like sigma not sigma^2.
    * Nweights is the maximum number of SVD weights to keep
        (see svdinv nearby).
    * svd is whether to use SVD for the matrix inversion instead of the
        standard matrix inversion.
    * verbose is whether the SVD inversion should show SV information.

    Starting in August 2013, X must be input as an array, which seems to
    be simpler, instead of a list that would be changed into an array
    here. Also the output order was changed from a, sa, chi2, yfit, C to
    yfit, a, sa, C, chi2. 

    Starting in September 2013, X must be input like
    [number of experimental trials, number of types of abscissa values]
    instead of the other way around as it had been before, and SVD is
    used instead of straight matrix inversion.

    Starting in March 2015, X must be input like
    [number of types of abscissa values, number of experimental trials],
    and using SVD is an option True by default.
    """
    import numpy as np
    # Get X into the right shape.
    if len(np.shape(X)) == 1:
        X = X[np.newaxis, :]
    n = np.shape(X)[0]  # Number of abscissa arrays
    m = np.shape(X)[1]  # Number of experimental trials
    # Ensure there is an uncertainty for the ordinate data.
    #if s is None or s == 0.0:
    if s is None:
#        s = 1.
#        s = np.sqrt(2.0)*rms(y)
        s = 0.1*rms(y)
    # Get the alpha array (i.e. various abscissas times abscissas).
    XT = X.T
    alpha = np.dot(X/s**2, XT)
    #print alpha
    # Get the inverse array C (i.e. correlation matrix).
    if svd:
        C = svdinv(alpha, Nweights=Nweights, verbose=verbose)
    else:
        import scipy.linalg as sl
        C = sl.inv(alpha)
    # Get the beta array (i.e. various abscissas times ordinate).
    beta = np.dot(X/s**2, y)
    # Get the fitted parameters.
    a = np.dot(C, beta)
    # Get the (uncorrelated) uncertainties in the parameters.
    sa = np.sqrt(C.diagonal())
    # Get the fitted ordinate array.
    yfit = np.dot(XT, a)
    # Get the reduced chi-squared value.
    error = yfit - y
    chi2 = sum(error*error/s/s) / (m - n)
    #print ( np.shape(yfit), np.shape(a), np.shape(sa), np.shape(C),
    #    np.shape(chi2) )
    return yfit, a, sa, C, chi2


def reduced_chi_squared(yfit, y, sigma, degrees):
    """Given yfit, y, sigma, and (the number of) degrees of freedom,
    return the reduced chi-squared value.
    """
    error = yfit - y
    return sum(error*error/sigma/sigma) / degrees


def match(string1, list2):
    """Return index of list2 that most closely matches string1,
    with ties going to the earliest appearance in list2.
    """
    len1 = len(string1)
    len2 = len(list2)
    s1 = string1.lower()
    l2 = [i.lower() for i in list2]
    scores = []
    for i in range(len2):
        score = 0
        s2 = l2[i]
        for j in range(len1):
            if s2.find(s1[j]) != -1:
                score += 1
                part = s2.partition(s1[j])
                s2 = part[0] + part[2]
        scores.append(score)
    return scores.index(max(scores))


def normalize(x):
    """Given an array, return it normalized from 0 to 1."""
    import numpy as np
    min = np.min(x)
    return (x - min)/(np.max(x) - min)


def freq(n, dt):
    """Given equally spaced time array of length n and step dt,
    return the corresponding frequency array.
    """
    import numpy as np
    return 1.0*np.arange(n)/n/dt


def get_seed():
    import numpy.random as nr
    return nr.get_state()[1][0]


def grayscale(image, formula=(0.299, 0.587, 0.114)):
    """Convert the input ndimage to luminance grayscale using the PIL
    formula (Rec. 601) for luma, L = 0.299*R + 0.587*G + 0.114*B.
    For reference, here is the colorimetric (luminence-preserving)
    method according to https://en.wikipedia.org/wiki/Grayscale:
        1) Gamma expansion:
            if C_srgb <= 0.04045:
                C_linear = C_srgb/12.92
            elif C_srgb > 0.04045:
                C_linear = ((C_srgb + 0.055)/1.055)**2.4
            # where C_srgb is a gamma-compressed R, G, or B in the
            # range [0, 1] and C_linear is the corresponding linear
            # intensity.
        2) Linear luminance:
            Y = 0.2126*R + 0.7152*G + 0.0722*b
        3) Gamma compression:
            if Y <= 0.0031308:
                Y_srgb = 12.92*Y
            elif Y > 0.0031308:
                Y_srgb = 1.055*Y**(1.0/2.4) - 0.055
    Note that Rec. 709 says to apply the colorimetric Y(R, G, B)
    formula above directly to the gamma-compressed values, otherwise
    like the luma method of Rec. 601.
    """
    from numpy import sum
    return sum(formula*image, axis=2).astype(int)


def hilbert(x):
    """return the Hilbert transform of x.
    The canonical transform takes a signal
    and rotates its positive-frequency components by -90 degrees,
    its negative-frequency components by +90 degrees,
    and does nothing to the zero-frequency component.
    For a pure real cosine, this turns it into a sine,
    and for a pure real sine, into a -cosine.
    This function uses this sign convention,
    and can operate on a complex input,
    as can scipy.fftpack.hilbert(),
    but the latter has a sign convention opposite to the canonical.
    Note scipy.signal.hilbert() can only accept a real input,
    but conveniently returns a complex value
    whose real part is this real input
    and whose imaginary part is the hilbert transform of the input
    using the canonical sign convention.
    """
    import numpy as np
    import numpy.fft as nf
    import scipy.fftpack as sf
    n=np.size(x)
    s=-np.ones(n)    # signs
    s[-n/2+1:]=1.0    # use + for the negative frequencies
    # Zero out DC and Nyquist, even though it destroys info,
    # to match the other two Python routines.
    s[0]=0.0
    if n % 2 == 0: s[n/2]=0.0
    return nf.ifft((np.zeros(n)+1j*s)*nf.fft(x))


def integral(y, x=None, initial=0.0, dx=1.0, axis=-1):
    import scipy.integrate as si
    return initial + si.cumtrapz(y, x, dx, axis, initial=0.0)


#def integrate(y, x=None, dx=1.0, axis=-1):
#    import scipy.integrate as si
#    return si.simps(y, x, axis=axis)


def iso_contour(x, y, z, zval, npts=100):
    """Given the 2D arrays x and y for the abscissa and z for the
    ordinate array, return the npts coordinates of a contour with the
    ordinate value zval.
    """
    import numpy as np
    dist = (zabs - zval)**2
    arg = np.argsort(dist,axis=2)
    dist.sort(axis=2)
    w_total = 0.
    z = np.zeros(zabs.shape[:2], dtype=float)
    for i in xrange(int(interp_order)):
        zi = np.take(zs, arg[:,:,i])
        valuei = dist[:,:,i]
        wi = 1/valuei
        np.clip(wi, 0, 1.e6, out=wi) # avoiding overflows
        w_total += wi**power_parameter
        z += zi*wi**power_parameter
    z /= w_total
    return z

def fft_filter(y, x, real=True, fmin=0.0, fmax=1.0):
    """Filter out frequency components as directed.
    Input array y is the signal.
    If provided, x is the abscissa array (e.g. time), and
    if not provided, x is assumed to be an indicial array
    of the same length as y.
    If fmin or fmax is omitted,
        then it is set to the min or max possible.
    If fmin < fmax, then allow only fmin to fmax.
    If fmin >= fmax, then block fmin to fmax.
    Both are given as functions of the sampling frequency.
    real is whether to convert the output to real before returning.
    """
    import scipy.fftpack as sf
    


def ftm(j=None, k=None, verbose=None):
    """Return Fourier transform matrix for two 1D arrays of real values
    which by default are integer sequences.
    j=rows is 1D array of real values for time/space domain,
    k=columns is 1D array of real values
    for frequency/wave-number domain.
    User can input one integer
    meaning number of 1D pts for the default arrays,
    one array meaning the other should be default
    with same number of pts,
    or two arrays which will be j and k
    depending on order if they are not named.
    """
    import scipy as sp
    if verbose is None: verbose=False
    if j is None and k is None:
        #User defines no inputs
        N=8
        j=sp.arange(N)
        k=sp.arange(N)
    elif k is None:
        #User defines one input corresponding to j
        if sp.size(j) == 1:
            #the input is an integer i.e. it means N
            N=j
            j=sp.arange(N)
            k=sp.arange(N)
        if sp.size(j) != 1:
            #the input is not an integer i.e. it means j
            N=sp.size(j)
            k=sp.arange(N)
    elif j is None:
        #User defines one input corresponding to k
        if sp.size(k) == 1:
            #the input is an integer i.e. it means N
            N=k
            j=sp.arange(N)
            k=sp.arange(N)
        if sp.size(k) != 1:
            #the input is not an integer i.e. it means k
            N=sp.size(k)
            j=sp.arange(N)
    else:
        #User defines two inputs which must be two arrays
        #since if one array is given
        #then an integer meaning the size is not needed.
        N=sp.size(j)
    #back to our regularly scheduled programming
    k[k > N/2.]=k[k > N/2.]-N
    J,K=sp.meshgrid(j,k)
    return sp.exp(-1j*2*sp.pi*J*K/N)/N


def iftm(k=None, j=None, verbose=None):
    """Return inverse Fourier transform matrix
    for two 1D arrays of real values
    which by default are integer sequences.
    k=columns is 1D array of real values
    for frequency/wave-number domain,
    j=rows is 1D array of real values for time/space domain.
    User can input one integer
    meaning number of 1D pts for the default arrays,
    one array meaning the other should be default
    with same number of pts,
    or two arrays
    which will be k and j depending on order if they are not named.
    """
    import scipy as sp
    if verbose is None: verbose=False
    if k is None and j is None:
        #User defines no inputs
        Nk=8
        Nj=Nk
        k=sp.arange(Nk)
        j=sp.arange(Nj)
    elif j is None:
        #User defines one input corresponding to k
        if sp.size(k) == 1:
            #the input is an integer i.e. it means N
            Nk=k
            Nj=k
            k=sp.arange(Nk)
            j=sp.arange(Nj)
        if sp.size(k) != 1:
            #the input is not an integer i.e. it means k
            Nk=sp.size(k)
            Nj=Nk
            j=sp.arange(Nj)
    elif k is None:
        #User defines one input corresponding to j
        if sp.size(j) == 1:
            #the input is an integer i.e. it means N
            Nj=j
            Nk=j
            k=sp.arange(Nk)
            j=sp.arange(Nj)
        if sp.size(j) != 1:
            #the input is not an integer i.e. it means j
            Nj=sp.size(j)
            Nk=Nj
            k=sp.arange(Nk)
    else:
        #User defines two inputs which must be two arrays
        #since if one array is given
        #then an integer meaning the size is not needed.
        Nk=sp.size(k)
        Nj=sp.size(j)
    #back to our regularly scheduled programming
    if verbose:
        print 'start'
#        print sp.shape(k),sp.shape(j)
        print k
#        print prod
    Nk=sp.size(k)
    Nj=sp.size(j)
    ku=k.copy()
#    if Nj != Nk: ku[k > Nk/2.]=ku[k > Nk/2.]-Nk
    ku[k > Nk/2.]=ku[k > Nk/2.]-Nk
        #that works
    #Test for aliasing with random abscissa points j
    #and the only w(k)\ne 0 being m=20 for instance with 16 points
    #ku[k > 3*Nk/2.]=ku[k > 3*Nk/2.]-3.*Nk
    K,J=sp.meshgrid(ku,j)
    M=sp.exp(2.j*sp.pi*J*K/Nj)#/Nk
#    if Nk != Nj:
#        w=sp.where(K == Nk/2.)
#        M[w]=sp.cos(2.*sp.pi*J[w]*K[w]/Nj)#/Nk
    w=sp.where(K == Nk/2.)
    #Test for aliasing with random abscissa points j
    #and the only w(k)\ne 0 being m=20 for instance with 16 points
    #Comment next lines out for test to ignore cosine effect
    #w=sp.where(K == 3*Nk/2.)
    #and avoid needing to change this K also
    ##M[w]=sp.cos(2.*sp.pi*J[w]*K[w]/Nj)#/Nk
    #and now back to our show
    #M[w]=sp.cos(2.*sp.pi*J[w]*K[w]/Nj)#/Nk
    M[w]=sp.cos(sp.pi*J[w])#/Nk
    if verbose:
        print 'end'
#        print sp.shape(k),sp.shape(j)
        print k
#        print prod
    return M


def fbessel_coeffs(f, N, order=0):
    """Return the Fourier-Bessel series coefficients for the expansion
    of order 'order' of the function 'f', up to the 'N'th zero.
    The input array f is assumed to begin at x=0 and end at x=1.
    """
    import numpy as np
    import scipy.integrate as si
    import scipy.special as ss
    nx = len(f)
    x = np.linspace(0.0, 1.0, nx)
    zeros = ss.jn_zeros(order, N)
    a = np.zeros(N)
    for i in range(N):
        a[i] = ( 2.0 / ss.jn(order + 1, zeros[i])**2
            * si.simps(x * f * ss.jn(order, zeros[i] * x), x) )
    return a


def fbessel_fit(x, a, order=0):
    """Return the Fourier-Bessel series fit of order 'order', given the
    abscissa array 'x' and the array of coefficients 'a'.
    """
    import numpy as np
    import scipy.special as ss
    na = len(a)
    zeros = ss.jn_zeros(order, na)
    nx = len(x)
    f = np.zeros(nx)
    for i in range(na):
        f = f + a[i] * ss.jn(order, zeros[i] * x)
    return f


def line(list1, list2, plo=False, pri=False, **kwargs):
    """Plots a line on pre-existing graph between points list1, list2,
    returns the parameters of the line as in the form y = ax + b.
    and optionally prints the equation of the line in the form
    y = ax + b.
    """
    import matplotlib.pyplot as mp
    [x1, y1] = list1
    [x2, y2] = list2
    a = (y2 - y1) / (x2 - x1)
    b = (x2*y1 - x1*y2) / (x2 - x1)
    label = str(a) + 'x + ' + str(b)
    if plo:
        mp.plot([x1, x2], [y1, y2], label=label, **kwargs)
    if pri:
        print label
    return a, b


def parabola(list1, list2, list3, plo=False, pri=False, **kwargs):
    """Plots a parabola on pre-existing graph between points
    list1, list2, and list3,
    returns the parameters of the parabola
    as in the form y = ax^2 + bx + c,
    and optionally prints the equation of the parabola
    in the form y = ax^2 + bx + c.
    """
    import matplotlib.pyplot as mp
    import numpy as np
    [x1, y1] = list1
    [x2, y2] = list2
    [x3, y3] = list3
    D = x1**2 * (x2 - x3) + x2**2 * (x3 - x1) + x3**2 * (x1 - x2)
    C = np.array([x2 - x3, x3**2 - x2**2, x2 * x3 * (x2 - x3),
                  x3 - x1, x1**2 - x3**2, x3 * x1 * (x3 - x1),
                  x1 - x2, x2**2 - x1**2, x1 * x2 * (x1 - x2)]
                  ).reshape(3, 3)
    yarr = np.array([y1, y2, y3])
    I = C.T / D
    [a, b, c] = np.dot(I, yarr)
    label = str(a) + 'x^2 + ' + str(b) + 'x + ' + str(c)
    if plo:
        x = np.linspace(x1, x3, 101)
        y = a * x**2 + b * x + c
        mp.plot(x, y, label=label, **kwargs)
    if pri:
        print label
    return a, b, c


def minmax(x):
    import numpy as np
    return [np.min(x),np.max(x)]


def multiplot(x, y, z = None, xlim = None, ylim = None, label = None,
    xlabel = None, ylabel = None, title = None, axvline = None,
    color = None, grid = False, axvlinestyle = 'dotted', focus = None,
    nlevels = 64, int_ticks = False):
    import numpy as np
    import matplotlib.pyplot as pp

    nplots = len(y)
    if z is None:
        contours = False
    else:
        contours = True
    if color is None:
        color = np.repeat('k', nplots)
    xfactor = 0.975 # factors for text boxes
    yfactor = 0.9 # factors for text boxes

    sp = pp.subplot(nplots, 1, 1)
    plot = True
    if contours:
        if len(np.shape(z[0])) == 2:
            plot = False
            pp.contourf(x[0], y[0], z[0], nlevels)
            pp.axis('tight')
    if plot:
        pp.plot(x[0], y[0], color = color[0])
    pp.grid(grid)
    if axvline is not None:
        for j in axvline:
            pp.axvline(j, color = 'k', linestyle = axvlinestyle)
    if focus is not None:
        pp.axvspan(focus[0], focus[1], facecolor = '0.5', alpha = 0.5,
            edgecolor = 'none')
    pp.xlim(xlim)
    sp.xaxis.set_ticklabels('')
    pp.ylabel(ylabel[0])
    if int_ticks:
        sp.yaxis.set_ticks(y[0])
    yvals = sp.yaxis.get_view_interval()
    ylim = [min(yvals), max(yvals)]
    pp.text(xlim[0] + (xlim[1] - xlim[0])*xfactor,
        ylim[0] + (ylim[1] - ylim[0])*yfactor, label[0],
        ha = 'right', va = 'top',# fontsize = 14,
        bbox = dict(boxstyle = 'round', fc = "w"))

    pp.title(title)

    for i in range(1, nplots - 1):
        sp = pp.subplot(nplots, 1, i + 1)
        plot = True
        if contours:
            if len(np.shape(z[i])) == 2:
                plot = False
                pp.contourf(x[i], y[i], z[i], nlevels)
                pp.axis('tight')
        if plot:
            pp.plot(x[i], y[i], color = color[i])
        pp.grid(grid)
        if axvline is not None:
            for j in axvline:
                pp.axvline(j, color = 'k', linestyle = axvlinestyle)
        if focus is not None:
            pp.axvspan(focus[0], focus[1], facecolor = '0.5',
                alpha = 0.5, edgecolor = 'none')
        pp.xlim(xlim)
        sp.xaxis.set_ticklabels('')
        pp.ylabel(ylabel[i])
        if int_ticks:
            sp.yaxis.set_ticks(y[i])
        yvals = sp.yaxis.get_view_interval()
        ylim = [min(yvals), max(yvals)]
        pp.text(xlim[0] + (xlim[1] - xlim[0])*xfactor,
            ylim[0] + (ylim[1] - ylim[0])*yfactor, label[i],
            ha = 'right', va = 'top',# fontsize = 14,
            bbox = dict(boxstyle = 'round', fc = "w"))

    sp = pp.subplot(nplots, 1, nplots)
    plot = True
    if contours:
        if len(np.shape(z[nplots-1])) == 2:
            plot = False
            pp.contourf(x[nplots-1], y[nplots-1], z[nplots-1],
                nlevels)
            pp.axis('tight')
    if plot:
        pp.plot(x[nplots-1], y[nplots-1], color = color[nplots-1])
    pp.grid(grid)
    if axvline is not None:
        for j in axvline:
            pp.axvline(j, color = 'k', linestyle = axvlinestyle)
    if focus is not None:
        pp.axvspan(focus[0], focus[1], facecolor = '0.5', alpha = 0.5,
            edgecolor = 'none')
    pp.xlim(xlim)
    pp.ylabel(ylabel[nplots-1])
    if int_ticks:
        sp.yaxis.set_ticks(y[nplots-1])
    yvals = sp.yaxis.get_view_interval()
    ylim = [min(yvals), max(yvals)]
    pp.text(xlim[0] + (xlim[1] - xlim[0])*xfactor,
        ylim[0] + (ylim[1] - ylim[0])*yfactor, label[nplots-1],
        ha = 'right', va = 'top',
        bbox = dict(boxstyle = 'round', fc = "w"))

    pp.xlabel(xlabel)


def hilbert_amp(x,version='fftpack'):
    """Return the amplitude of the input signal x using hilbert(x)
    from either scipy.signal (default), scipy.fftpack, or functions.py.
    See hilbert() and hilbert_amp() nearby (if not here in
    functions.py) for details on these.
    Aug 2013: Note version='signal' is givin an error
        ValueError: type >f4 is not supported
        when data from an IDL save file is used, so I am changing
        the default to 'fftpack'.
    """
    import numpy as np

    if version == 'signal':
        import scipy.signal as ss
        amp=np.abs(ss.hilbert(x))
    elif version == 'fftpack':
        import scipy.fftpack as sf
        amp=np.abs(x+1j*sf.hilbert(x))
    elif version == 'functions':
        amp=np.abs(x+1j*hilbert(x))
    return amp


def hilbert_corr(y, z):
    """Return the Hilbert correlation of two signals."""
    import numpy.fft as nf
    if len(y) != len(z):
        return 'error'
    F = nf.fft(y)
    G = nf.fft(z)
    FGs = F*G.conj()
    fg = nf.ifft(FGs)/len(y)
    f = rms(y)
    g = rms(z)
    return hilbert_amp(fg)/f/g


def hilbert_phs(x,version='signal',unwrap=True,subtract=False,n=100):
    """return the phase of the input signal x using hilbert(x)
    from either scipy.signal (default), scipy.fftpack, or functions.py.
    See hilbert() nearby (if not here in functions.py)
    for details on these.

    version='signal': scipy.signal.hilbert only takes a real input,
    so this version takes only the real part of the input
    to get the phase.

    version='fftpack': scipy.fftpack.hilbert,
    and therefore this version, can take a complex input.

    version='functions': functions.hilbert,
    and therefore this version, can take a complex input.

    Note a commented-out version of this code used scipy.arctan2,
    and therefore only used the real part of the input
    regardless of type.

    KWARGS:
    unwrap: whether to unwrap
    subtract: whether to subtract linear baseline
    n: num. of pts. at each end for linear baseline subtraction
    """
    import scipy as sp

#    rx=sp.real(x)
    if version == 'signal':
        import scipy.signal as ss
        #phi=sp.angle(ss.hilbert(sp.real(x)))
        phi=sp.angle(ss.hilbert(x))
#        phi=sp.arctan2(sp.imag(ss.hilbert(rx)),rx)
            #Note this hilbert(x) has real(hilbert(x))==real(x)
    elif version == 'fftpack':
        import scipy.fftpack as sf
        phi=-sp.angle(x+1j*sf.hilbert(x))
#        phi=-sp.arctan2(sp.real(sf.hilbert(rx)),rx)
    elif version == 'functions':
        phi=sp.angle(x+1j*hilbert(x))
#        phi=sp.arctan2(sp.real(hilbert(rx)),rx)
    if unwrap == True: phi=sp.unwrap(phi)
    if subtract == True:
        phi = sub_lin_base(phi, n)
    return phi


def phase_space_map(x, y, t, cmap = 'jet',
    title = None, xlabel = None, ylabel = None, tlabel = None):
    import numpy as np
    import matplotlib.pyplot as pp
    from matplotlib.collections import LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap = pp.get_cmap(cmap))
    lc.set_array(t)
    #lc.set_linewidth(3)
    pp.title(title)
    pp.plot(x, y, color = [0, 0, 0, 0])
    pp.gca().add_collection(lc)
    cb = pp.colorbar(lc, fraction = 0.02, aspect = 100, ticks = None)
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    cb.set_label(tlabel)


def polyvals(x0, y0, x, deg=0):
    """Use numpy polyval(p, x) and polyfit(x0, y0, deg) to return the
    polynomial curve fitted to input data (x0, y0) at test abscissa
    value(s) x.
    """
    import numpy as np
    return np.polyval(np.polyfit(x0, y0, deg), x)


def read_csv(filename, flatten=False):
    """Read filename or filename.csv."""
    import csv
    import numpy
    flist=list(csv.reader(open(filename,'r')))
    if flist[-1] == []: flist=flist[:-1]
    a = numpy.array(flist)
    if flatten == True:
        return a.flatten()
    else:
        return a


def write_csv(data,filename='data.csv'):
    import csv
    """Write list or array data into filename or filename.csv."""
    csv_writer = csv.writer(open(filename,'w'))
#    for row in data:
#        csv_writer.writerow(row)
    csv_writer = csv_writer.writerows(data)


def read_txt(filename='filename.txt'):
    """Read a text file into a list of line strings."""
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    return data


def txt_to_csv(infile='input.txt',outfile='output.csv'):
    import csv
    #csv_in = csv.reader(open(infile, 'rb'), delimiter='\t')
    #csv_out = csv.writer(open(outfile, 'w'), delimiter=',')
    #for row in csv_in:
    #    s=str(row)
    #    csv_out.writerow(s[2:-2].split())
    txt_in = open(infile)
    csv_out = csv.writer(open(outfile, 'w'), delimiter=',')
    for line in txt_in.readlines():
        csv_out.writerow(line.split())


def txt_to_array(pathname, shape):
    """Given the path name of a text file and a shape tuple like
    (100, 100), return the data from the file as a 2D numpy array.
    """
    import numpy as np
    f = open(pathname, 'r')
    data = np.array(
        [float(i) for i in f.read().split()]).reshape(shape)
    f.close()
    return data


def write_hdf5(data, filename):
    """Given numpy array or python dictionary data, write it to an hdf5
    file.
    """
    import h5py as hp
    import numpy as np
    hfile = hp.File(filename, 'w')
    typ = type(data)
    if typ == dict:
        for k in data.iterkeys():
            # The straight code gives ustrings, which I don't like.
#            hfile[k] = data[k]
            exec("hfile['" + k + "'] = data['" + k + "']")
    elif typ == np.ndarray:
        hfile['data'] = data
    hfile.close()


def read_hdf5(filename):
    """Given the filename of an hdf5 file, read its data as a numpy
    array or python dictionary.
    """
    import h5py as hp
    hfile = hp.File(filename, 'r')
    lenk = len(hfile.keys())
    if lenk == 1:
        data = hfile[hfile.keys()[0]].value
    else:
        data = {}
        for k in hfile.iterkeys():
            # The straight code gives ustrings, which I don't like.
#            data[k] = hfile[k].value
            exec("data['" + k + "'] = hfile['" + k + "'].value")
    hfile.close()
    return data


def readsav_struct(filename):
    """Given the filename of an IDL save file of a structure, return
    its data as a python dictionary.  Note this need modification for
    save files of non-structure IDL data, such as a set of arrays.
    """
    import scipy.io as sio
    idl = sio.readsav(filename)
    rec = idl[idl.keys()[0]]
    keys = [k.lower() for k in rec.dtype.names]
    data = rec[0]
    return dict(zip(keys, data))


def rms(x):
    """Return the root mean square of x, instead of doing
    matplotlib.mlab.rms_flat.
    """
    import numpy as np
    return np.sqrt(np.mean(x*x))


def rk4(x = None, y0 = 0., f = None):
    """Fourth-order Runge-Kutta solver
    for a first-order ordinary differential equation.
    The inputs are
    x the equally-spaced abscissa values,
    y0 the initial value of the ordinate y to be found,
    f the function to be integrated across x.
    The output is the solved ordinate y.
    Starting with
    dy/dx = f(x, y(x))
    and
    h = x[1] - x[0] = x[2] - x[1] = ... ,
    for each point,
    k0 = h * f(x[n], y[n])
    k1 = h * f(x[n] + (1./2)*h, y[n] + (1./2)*k0)
    k2 = h * f(x[n] + (1./2)*h, y[n] + (1./2)*k1)
    k3 = h * f(x[n] + h, y[n] + k2)
    y[n+1]
      = y[n] + (1./6)*k0 + (1./3)*k1 + (1./3)*k2 + (1./6)*k3 + O(h5).
    An example of how to use this function may be found
    in cyl_eq_ode.py.
    """
    o3 = 1./3
    o6 = 1./6
    h = x[1] - x[0]
    hh = h/2.
    y = x - x
    y[0] = y0
    for i in range(1, len(x)):
        x0 = x[i-1]
        x0hh = x0 + hh
        x0h = x[i]
        y0 = y[i-1]
        k0 = h * f(x0, y0)
        k1 = h * f(x0hh, y0 + 0.5*k0)
        k2 = h * f(x0hh, y0 + 0.5*k1)
        k3 = h * f(x0h, y0 + k2)
        y[i] = y0 + o6*(k0 + k3) + o3*(k1 + k2)
    return y


def rk4_set(x, v, h, vtype='function'):
    """Given x, which is an array, tuple, or list
    of the values of different position-like quantities
    in a set of coupled 1st-order ODEs
    at an initial time point,
    v, either a vtype='array' or a vtype='function'
    returning the corresponding velocities,
    and h, a time step value,
    this returns the next positions as an array
    using the 4th-order Runge-Kutta method.
    For example, if the position arrays in a problem are y, z, and w,
    where positions=scipy.array(scipy.transpose((y,z,w))),
    then input x=positions[index]
    which is an array of the positions at the time of index
    regardless of how many position quantities there are,
    and then, if v is an array, then it should have initial and final
    v values such that v[index] are the initial
    and v[index+1] the final
    (i.e. it should have twice as many members as x),
    and if v is a function, then it should have the same shape as x
    for its input and output.
    Note that this function could probably handle v(y,z,w,...,t)
    if a simple differential equation for t is used,
    but it was not designed for such a case.
    Note also that in future maybe could add vtype='streamline'
    to cover the case where a streamline is desired
    from a map of some quantity's value versus x and t, e.g.,
    which case may be difficult to cover with the vtypes described.
    An example of how to use this function
    may be found in lotka_volterra.py.
    """

    import scipy as sp

    xi = sp.array(x)

    ho2 = h/2.

    if vtype == 'array':
        v0 = v[0]
        v3 = v[1]
        v1 = 0.5*(v0 + v3)
        v2 = v1

    if vtype == 'function':
        v0 = v(xi)
        v1 = v(xi+ho2*v0) 
        v2 = v(xi+ho2*v1)
        v3 = v(xi+h*v2)

    return xi + h/6.*(v0 + 2.*v1 + 2.*v2 + v3)


def shift(y, s, test=False, ndense=10.0):
    """Positive shift ordinate array y to the right by s indices, taking
    fractions into account.  For fractional shifts, this likely only
    works as desired for periodic domains with narrow spectral content,
    because it's using Fourier interpolation to generate the data between
    index points of the input array, while for integer shifts, it works
    the same as numpy.roll().
    Note for n even or odd:
        (n-1)/2 is for the largest frequency < Nyquist,
        (n+1)/2 is for the smallest frequency >= Nyquist,
        n/2+1 is for the smallest frequency > Nyquist.
    """
    import numpy.fft as nf
    import numpy as np
    n = len(y)
    ft = nf.fft(y)
    #freq = 1.0/n*np.arange(n)
    freq = nf.fftfreq(n)
        # Note these have positive and negative numerical values.
    amp = abs(ft)
    phase = np.arctan2(ft.imag, ft.real)
    pshift = s*freq*2.0*np.pi
    z = np.zeros(n, dtype=complex)
    z = amp*np.exp(1j*(phase - pshift))
    if test:
        import matplotlib.pyplot as mp
        for i in range(n):
            print ft[i], z[i]
        x = np.arange(float(n))
        ndense = ndense*n
        xdense = np.arange(float(ndense))/float(ndense)*n
        ydense = np.zeros(xdense.shape, dtype=complex)
        for i in range(len(ft)):
            ydense = ydense + \
                ft[i]/float(n)*np.exp(1j*2.0*np.pi*freq[i]*xdense)
        ydense = ydense.real
        mp.clf()
        mp.grid()
        mp.plot(x, y, 'o-')
        mp.plot(xdense, ydense)
        mp.plot(x, shift(y, s), 'o-')
        mp.plot(xdense, shift(ydense, s*ndense/n),
            linewidth=2)
        m = int(s*ndense/n)
        #for i in range(m):
        #    mp.plot(xdense, np.roll(ydense, i))
        mp.plot(xdense, np.roll(ydense, m))
    return nf.ifft(z).real


def sign_extremum(y):
    """Says whether the sign of the data with largest absolute value
    is positive, negative, or zero.
    """
    import numpy as np
    mxabs = np.max(np.abs(y))
    if mxabs == 0.0:
        return 0
    else:
        mx = np.max(y)
        if mxabs == mx:
            return 1
        else:
            return -1


def smooth(x, n, periodic=False):
    """Midpoint boxcar smooth of size n for input data x.  If an even n
    is given, this adds 1 to it for the smoothing number -- note this
    is different than IDL behavior (which doubles it and then adds 1).
    """
    from numpy import zeros, roll, mean
    n = int(n)
    if n%2 == 0:
        n += 1
    y = zeros(len(x))
    for i in range(-n/2+1, n/2+1):
        y = y + roll(x, i)
    y = y / n
    if periodic == False:
        for j in range(n/2):
            #y[j] = sum(x[0:n/2+1+j]) / (n/2+1+j)
            #y[-j-1] = sum(x[-n/2-j:]) / (n/2+1+j)
                # Not sure why I was doing that,
                # does not work for linear cases
            y[j] = mean(x[:2*j+1])
            y[-j-1] = mean(x[-2*j-1:])
    return y


def dsmooth(x, n1, n2, periodic=False):
    """Double smooth"""
    n1 = int(n1)
    if n1%2 == 0:
        n1 += 1
    n2 = int(n2)
    if n2%2 == 0:
        n2 += 1
    return smooth(smooth(x, n1, periodic=periodic), n2,
        periodic=periodic)


def bin(x, y, n=100, endpoints=False):
    """Conventional binning: given abscissa x, ordinate y, and abscissa
    range divisor n, return xnew, ynew, which are the abscissa and
    ordinate with binned values for each case for which there are
    multiple abscissa values in one of the n regular bins of width
    (max(x)-min(x))/n.  The keyword 'endpoints' is whether to include
    the abscissa endpoints in this process.  There will be at most n
    points in the output, neglecting endpoints.  
    """
    import numpy as np
    delta = 1.0*(max(x) - min(x))/n
    xbin = min(x) + delta*np.arange(n)
    if endpoints:
        xuse = x
        yuse = y
        xnew = []
        ynew = []
    else:
        xuse = x[1:-1]
        yuse = y[1:-1]
        xnew = [x[0]]
        ynew = [y[0]]
    for i in range(n):
        w = np.where((xuse >= xbin[i]) & (xuse < xbin[i] + delta))[0]
        if len(w) >= 1:
            xnew.append(np.mean(xuse[w]))
            ynew.append(np.mean(yuse[w]))
    if not(endpoints):
        xnew.append(x[-1])
        ynew.append(y[-1])
    return np.array(xnew), np.array(ynew)


def rbin(x, y, n=100, endpoints=False):
    """I am not sure this works right.
    Running bin or advanced boxcar smoothing: given abscissa x,
    ordinate y, and abscissa range divisor n, return xnew, ynew, which
    are the abscissa and ordinate with mean values for all values within
    (max(x)-min(x))/n/2 of each input abscissa point.  The keyword
    'endpoints' is whether to include the abscissa endpoints in this
    process.  Note this can give back fewer points than input because
    redundant points of duplicate sets are neglected.
    """
    import numpy as np
    delta = 1.0*(max(x) - min(x))/n
    if endpoints:
        xuse = x
        yuse = y
    else:
        xuse = x[1:-1]
        yuse = y[1:-1]
    for i in range(len(xuse)):
        w = np.where((x >= (xuse[i] - delta/2.0))
            & (x < (xuse[i] + delta/2.0)))[0]
        if len(w) > 1:
            xuse[i] = np.mean(x[w])
            yuse[i] = np.mean(y[w])
    xnew, anew = np.unique(xuse, return_index=True)
    ynew = yuse[anew]
    if not(endpoints):
        xnew[0] = x[0]
        xnew[-1] = x[-1]
        ynew[0] = y[0]
        ynew[-1] = y[-1]
    return xnew, ynew


def func_example(pars, x):
    """Example of function to be used with error() and fmin() below.
    This example is a simple exponential function.
    """
    from numpy import exp
    return pars[0]*exp(pars[1]*t)

def error(pars, func, x, y):
    """Given parameters pars for a function func in the abscissa x and
    the ordinate y, return the error between the function and the data.
    """
    return rms(func(pars, x) - y)


def fmin(pars, func, x, y,
        disp=False, maxfun=None, maxiter=None):
    """Use fmin to find the best nonlinear fit parameters for a
    function func in the abscissa x and the ordinate y.
    """
    import scipy.optimize as so
    pars, fopt, iter, funcalls, warnflag = so.fmin(
        error, pars, args=(func, x, y),
        disp=disp, full_output=True, maxfun=maxfun, maxiter=maxiter)
    #return pars, warnflag, funcalls, iter
    return pars


def msplines(x, t, k=3, verbose=False):
    """Using Ramsay, Stat. Sci. 3, 425-461 (1988), return the order-k
    M-splines defined on the array-like domain x, given internal knots
    at t.
    """
    import numpy as np
    n_points = len(x)
    n_int_knots = len(t)
    n_splines = len(t) + k
    tt = np.append(np.repeat(np.min(x), k), t)
    tt = np.append(tt, np.repeat(np.max(x), k))
    n_tot_knots = len(tt)
    M = np.zeros([n_splines, n_points])
    for i in range(n_splines):
        w = np.where((x>=tt[i]) & (x<tt[i+1]))[0]
        if len(w) > 0:
            if verbose:
                print 1, i, n_splines
            M[i, w] = 1.0/(tt[i+1] - tt[i])
    w = np.where((x>=tt[n_splines-1]) & (x<=tt[n_splines]))[0]
    M[n_splines-1, w] = 1.0/(tt[n_splines] - tt[n_splines-1])
    for j in range(2, k+1):
        if verbose:
            print
        for i in range(n_splines-1):
#        for i in range(n_splines):
            w = np.where((x>=tt[i]) & (x<tt[i+j]))[0]
            if len(w) > 0:
                if verbose:
                    print j, i, n_splines
                M[i, w] = (
                    j*( (x[w] - tt[i])*M[i, w]
                    + (tt[i+j] - x[w])*M[i+1, w])/
                    (j - 1)/(tt[i+j] - tt[i]) )
        w = np.where((x>=tt[n_splines-1]) & (x<=tt[n_splines-1+j]))[0]
        if len(w) > 0:
            if verbose:
                print j, n_splines-1, n_splines
            M[n_splines-1, w] = (
                j*(x[w] - tt[n_splines-1])*M[n_splines-1, w]/
                (j - 1)/(tt[n_splines-1+j] - tt[n_splines-1]) )
    return M


def isplines(x, t, k=3, both=False, verbose=False):
    """Using Ramsay, Stat. Sci. 3, 425-461 (1988), return the order-k
    I-splines defined on the array-like domain x, given internal knots
    at t.  The keyword 'both' says whether to return both i and m.
    """
    import numpy as np
    import scipy.integrate as si
    m = msplines(x, t, k, verbose=verbose)
    i = np.zeros(shape=np.shape(m))
    for j in range(len(m)):
        i[j, :] = integral(m[j, :], x)
    if both:
        return i, m
    else:
        return i


def test_splines(t=[0.3, 0.5, 0.6], verbose=True):
    """Test msplines and isplines using Fig. 1 from Ramsay, Stat. Sci.
    3, 425-461 (1988).
    """
    import numpy as np
    import matplotlib.pyplot as mp
    n = 41 #101
    x = np.arange(n)/(n-1.0)
    I, M = isplines(x, t, both=True, verbose=verbose)
    if len(t) == 3:
        a = np.array([1.2, 2.0, 1.2, 1.2, 3.0, 0.0])
    else:
        a = np.ones(len(t) + 3)
        a[-1] = 0.0
    ym = np.sum(a*M.T, axis=1)
    yi = np.sum(a*I.T, axis=1)/6.0
    mp.clf()
    mp.subplot(2,1,1)
    for i in range(len(I)):
        mp.plot(x, I[i], color='k')
    for v in t:
        mp.axvline(v, linestyle='dotted', color='k')
    mp.plot(x, yi, '.', color='k')
    mp.subplot(2,1,2)
    for i in range(len(M)):
        mp.plot(x, M[i], color='k')
    for v in t:
        mp.axvline(v, linestyle='dotted', color='k')
    mp.plot(x, ym, '.', color='k')


def error_splines(pars, use, x, y, t, k, stype):
    """Given pars=[amp1, amp2, ...] for a model with internal knots at
    t, return the error between a linear combination of M-splines on x
    and the data y.
    """
    from numpy import dot
    if stype == 'm':
        return rms(dot(use*pars, msplines(x, t, k)) - y)
    elif stype == 'i':
        return rms(
            dot(use*pars, isplines(x, t, k)) - y)


def fmin_splines(pars, use, x, y, t, k, stype, disp=False,
        maxfun=None, maxiter=None):
    """Use fmin to find amplitudes for an M-spline or I-spline fit to y
    on x. Default for both maxfun and maxiter appears to be around 2000.
    """
    import scipy.optimize as so
    pars, fopt, iter, funcalls, warnflag = so.fmin(
        error_splines, pars, args=(use, x, y, t, k, stype),
        disp=disp, full_output=True, maxfun=maxfun, maxiter=maxiter)
    return pars, warnflag, funcalls, iter


def yfit_splines(x, y, u, Nt=5, k=3, stype='m',
    left=False, right=False, verbose=False, maxfun=None, maxiter=None):
    """Given x and y, the input data, and u, the output abscissa,
    return an M- or an I-spline fit to y.
    """
    import numpy as np
    t = ( min(u)
        + (max(u) - min(u))/(Nt + 1)*(np.arange(Nt) + 1) )
    pars = np.ones(Nt + k)
    use = pars.copy()
    if left:
        use[0] = 0.0
    if right:
        use[-1] = 0.0
    pars, warnflag, funcalls, iter = fmin_splines(
        pars, use, x, y-y[0], t, k, stype,
        maxfun=maxfun, maxiter=maxiter)
    if stype == 'm':
        yfit = y[0] + np.dot(use*pars, msplines(u, t, k=k))
    elif stype == 'i':
        yfit = y[0] + np.dot(use*pars, isplines(u, t, k=k))
    if verbose:
        print warnflag
        print funcalls
        print iter
        print rms(yfit - y)/rms(y)
        print use*pars
    return yfit


def test_fmin_splines(Nt=5, k=3, Nx=1000, stype='m',
    reverse=True, half=True,
    right=False, maxfun=None, maxiter=None):
    """For a zero ordinate derivative at either edge, use left=True or
    right=True, respectively.  This will encourage convergence at higher
    Nt values and with fewer iterations and function calls in fmin.
    Note higher Nt values tend to give smaller fitting errors.
    Some typical worst cases for k=3, Nx=1000:
        stype='m':
            Nt=5 gives an error < 8E-3.
            Nt=6 gives an error < 5E-3.
            Nt=7 gives an error < 3E-3.
            Nt=8 gives an error < 6E-2, which is faulty;
                need to set maxfun and/or maxiter to about 3000,
                which gives an error < 3E-3.
        stype='i'
            Nt=5 gives an error < 6E-5.
            Nt=6 gives an error < 3E-5.
            Nt=7 gives faulty results; need maxfun and/or maxiter set.
    Generally, one can get better results, i.e. better convergence at
    high Nt, by increasing maxfun and maxiter beyond their scipy fmin
    defaults of about 2000 for each.
    """
    import numpy as np
    import matplotlib.pyplot as mp
    x = 2.0*np.pi*np.arange(Nx + 1)/Nx
    if stype == 'm':
        left = False
        y = 0.5*(1.0 + np.cos(x/2.0))
        y = 0.5*(1.0 + np.cos(x/4.0))
        y = np.cos(x)
        y = np.sin(x)
    elif stype == 'i':
        if reverse:
            left = True
            y = 0.5*(1.0 + np.cos(x/2.0))
            if half:
                y = 0.5*(1.0 + np.cos(x/4.0))
        else:
            left = False
            y = 0.5*(1.0 - np.cos(x/2.0))
            if half:
                y = 0.5*(1.0 - np.cos(x/4.0))
    yfit = yfit_splines(x, y, x, Nt, k, stype, left, right, verbose=True,
        maxfun=maxfun, maxiter=maxiter)
    mp.clf()
    mp.plot(x, y, label='actual')
    mp.plot(x, yfit, label='fit')
    mp.xlim(xmin=min(x), xmax=max(x))
    mp.ylim(ymin=np.min([y, yfit]), ymax=np.max([y, yfit]))
    #mp.legend()


def sorting_indices(a):
    """Given an array, return the array of indices that would sort it.
    """
    import scipy as sp
    return sp.array( [ list(a).index(i) for i in sorted(a) ] )
    # The above is unstable even though the Kool-Aid(R)-drinkers say
    # stable.


def svdinv(X, Nweights=None, factor=1E3, verbose=False):
    """Return the pseudo-inverse of a REAL matrix X using SVD.
    Nweights, if given, is the maximum number of weights to keep,
        for none thrown out due to the factor criterion,
        explained below.
    factor is the minimum ratio between neighboring elements
        that will cause a zeroing of an inverse weight.
        The true inverse is more likely to be returned in the limit
        of a very large factor.
    verbose is whether to return singular value information.
    """
    import scipy as sp
    import scipy.linalg as sl
    U, s, Vh = sl.svd(X)
    # U is mxm, s has length n,
    # and Vh (in general the Hermitian conjugate of V) is nxn.
    m = len(U)
    n = len(s)
    # Create the pseudo-inverse of S,
    # where S is the mxn matrix whose only nonzero elements
    # are the n diagonals populated by the values of s.
    Sp = sp.zeros([n, m])
    Sp[0, 0] = 1.0/s[0]
        # We will always keep at least one weight.
    #print s[0:-1]/s[1:]
    if Nweights is not None:
        if Nweights <= n:
            n = Nweights
            # Don't change n if Nweights is too big.
        #print n
    for i in range(1, n):
        ratio = s[i-1]/s[i]
        if ratio < factor:
            Sp[i, i] = 1.0/s[i]
        else:
            if verbose:
                print i, ratio
            break
    return sp.dot(Vh.T, sp.dot(Sp, U.T))


def text(subplot, text, xfactor = 0.95, yfactor = 0.95, fontsize = 14):
    """Puts text string in a rounded box in a plot."""
    import matplotlib.pyplot as pp
    xvals = subplot.xaxis.get_view_interval()
    xlim = [min(xvals), max(xvals)]
    yvals = subplot.yaxis.get_view_interval()
    ylim = [min(yvals), max(yvals)]
    pp.text(xlim[0] + (xlim[1] - xlim[0])*xfactor,
        ylim[0] + (ylim[1] - ylim[0])*yfactor, text,
        ha = 'center', va = 'center', fontsize = fontsize,
        bbox = dict(boxstyle = 'round', fc = "w"))


def triangle_wave(y0=0., y1=1., l=50, m=25, s=0):
    """Given float y0 the DC value,
    float y1 the AC amplitude,
    integer l the number of pts per cycle,
    integer m the number of cycles in the waveform,
    integer s which when positive shifts the waveform to smaller times,
    return a triangle wave as an array.
    """
    import scipy as sp
    lu=int(l)
    mu=int(m)
    su=int(s)
    t=1.*sp.arange(lu/2)   #abscissa for one half cycle
    c=sp.zeros(lu)   #ordinate for one cycle
    c[0:lu/2]=1.-2.*t/(lu/2.)
    c[lu/2:lu]=-1.+2.*t/(lu/2.)
    w=c
    for i in range(mu-1): w=sp.append(w,c)
    return y0+y1*sp.roll(w,su)


def unroll(x, lim = 3.14159265359):
    """Detect if the diff between neighboring points in x
    is greater than lim, and if it is, shift all future points by jump.
    """
    from numpy import diff
    d = diff(x)
    for i in range(0,len(x)-1):
        if abs(d[i]) > lim: x[i+1:] = x[i+1:] - d[i]
    return x


def weights(x):
    """Given a possibly unequally spaced abscissa array, return an array
    of weights for those values that could be used as weights in
    numpy.average.  For instance, for an input of an unequally spaced
    time array, the output can be used to help compute a proper time
    average.
    """
    import numpy as np
    w = (np.roll(x, -1) - np.roll(x, 1))/2.0
    w[0] = np.roll(x, -1)[0] - x[0]
    w[-1] = x[-1] - np.roll(x, 1)[-1]
    if len(w) == 1:
        w = np.array([1.0])
    return w


def whereval(x, x0):
    """Given an input array x and target value x0, return the index of
    x for which x is closest to x0.  If more than one index satisfies
    the condition, the lowest is returned.
    """
    import numpy as np
    y = abs(x - x0)
    return np.where(y == np.min(y))[0][0]


def ylim_range(vals=None):
    """Find the ylimits for a range of data values."""
    import numpy as np
    import matplotlib.pyplot as mp
    if vals == None:
        valsNone = True
        vals = np.arange(-9.9, 10.0, 0.1)
    else:
        valsNone = False
#    for v in range(len(vals)):
#        print v, vals[v], 
#        if vals[v] < 0:
#            vals[v] -= 0.1
#        elif vals[v] > 0:
#            vals[v] += 0.1
#        print vals[v] 
    ylim = []
    for v in vals:
        mp.clf()
        mp.plot([0, v])
        ylim.append(mp.axis.func_globals['ylim']())
    mp.clf()
    mp.plot(vals, ylim, 'o')
    if valsNone:
        return vals, ylim
    else:
        return ylim


def ylim(y):
    """Given y data, return the ylim_range values (as a list)."""
    mp.figure(99)
    a = mp.subplot()
    a.plot(y)
    #ylim = a.axis.func_globals['ylim']()
    ylim = a.get_ylim()
    mp.close(99)
    return ylim


def leading_digit(y):
    """Given a float, return its leading digit and power of 10."""
    from numpy import floor, log10
#    print 'y', y
#    print 'log10(y)', log10(y)
#    print 'floor(log10(y))', floor(log10(y))
#    print '-floor(log10(y))', -floor(log10(y))
    power = floor(log10(y))
#    print '10**(-power)', 10**(-power)
#    print 'floor(10**(-power)*y)', \
#        floor(10**(-power)*y)
    return floor(10**(-power)*y), power


def test_ylim_subset_plot(x=None, y=None, x1=0.45, x2=0.55):
#def test_ylim_subset_plot():
    """Test plotting with ylim based on the subset of the data where
    x1<=x<x2.
    """
    import matplotlib.pyplot as mp
    import numpy as np
    x = np.arange(1000)/1000.0
    y = -np.cos(2*np.pi*x) - 0.5*np.cos(4*np.pi*x)
    w = np.where((x >= x1) & (x < x2))
    mp.clf()
#    # First way, timeit ~ 10, 11
    a = mp.subplot()
    b, = a.plot(x[w], y[w], color='k')
    ylim = a.get_ylim()
    b.remove()
#    # Second way, don't know if it works
#   a = mp.subplot()
#    mp.plot(x[w], y[w])
#    ylim = a.get_ylim()
#    a.axes.lines[0].remove()
#    # Third way, timeit ~ 20
#    mp.plot(x[w], y[w])
#    ylim = mp.axis.func_globals['ylim']()
#    mp.clf()
    # Back to main routine
    mp.plot(x, y)
    mp.ylim(ylim)


def zero_crossings(x, y):
    """Given x the abscissa and y the ordinate, return the values of x
    for which y crosses zero.
    """
    n = len(x)
    x_zc = []
    for i in range(n-1):
        if y[i] == 0.0:
            x_zc.append(x[i])
        elif ( (y[i] > 0.0 and y[i+1] < 0.0)
            or (y[i] < 0.0 and y[i+1] > 0.0) ):
            x_zc.append(
                (y[i] * x[i+1] - y[i+1] * x[i]) / (y[i] - y[i+1]))
    return x_zc

