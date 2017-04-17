# old16 is to preserve the code that was adding one element to all the
# radial arrays in adhoc.py and cyleq.py.

# Generic cylindrical equilibrium solutions


def zfunc(rho, bz, bq, lam, press):
    return -lam * bq - press / (bz**2 + bq**2) * bz

def qfunc(rho, bz, bq, lam, press):
    if rho == 0.0:
        return (lam * bz)
    else:
        return (lam * bz) - (1.0/rho + press / (bz**2 + bq**2) ) * bq

#def press(rho, beta0):
def press_quadratic(rho, beta0):
    """Pressure function that returns quadratic p, gradp."""
    p = (beta0 / 2.0)*(1 - rho**2)
    gradp = (beta0 / 2.0) * (-2.0 * rho)
    return p, gradp

def press_cubic(rho, beta0):
    """Pressure function that returns matched cubic p, gradp.
    Found that
    p/p0 = 1 - (4/3)rho**3+(1/3)rho**12
    (matched polynomial with dp/dr(rho=1) = 0
    closely matches measured p profile from Biewer's thesis
    I like this type of polynomial since dp/dr = 0 at edge
    is required physically.
    Note the quartic profile case for completeness:
    p = (beta0/2.0)*(-4*rho**3+3*rho**4)
    gradp = ( beta0/2.0 ) * ( -12.0*rho**2 + 12.0*rho**3)
    """
    p = (beta0 / 2.0)*(1.0 - (4.0/3.0)*rho**3+(1.0/3.0)*rho**12)
    gradp = (beta0 / 2.0)*(-4.0*rho**2 + 4.0*rho**11)
    return p, gradp

def lam_to_eq(lam, pars, ip,
    pmodel='quadratic', beta=0.07, n=51, ret='all',
    corr='cyl', d=0.01, a=0.52, Ra=1.50):
    """
    Given 1D lambda-profile function and ip as a scaling quantity,
    return various field quantities from the cylindrical equilibrium
    model.
    Note ip must be in mks for this to work right, while ip comes from
    MDSplus in kA.

    lam: a function that takes in radius x and parameters pars
    and outputs lambda at that x.  Note that lam in this file
    always means lambda * b, the local inverse-scale length
    times the minor radius at which the plasma current vanishes.

    beta: the average pressure over Bpw**2/(2*mu0), i. e. 'poloidal beta'.

    n: number of radial points.

    a and Ra: minor and major radius of measurements in mks

    ret='scalars': returns ip, btw, btave, b0, beta0 as a tuple,
        for use in optimization codes like simplex.

    ret='all' (default): returns things from 'scalars', as well as
        Ra, a, d, Rb, b, rho, bq, bz, jq, jz, p, gradp, q, lam,
        all as named in a dictionary.
    """
    import numpy as np
    import scipy.integrate as sig

    mu0 = np.pi * 4E-7
    # The original JSS value:
    m_max = 4  # beta iterations
    # KJM 2012-02 to use conditional loop with tolerance.
#    m_max = 10  # beta iterations

    h = 1.0 / n
    hh = h / 2.0

    # Normalized radial coordinate
    rho = np.linspace(0.0, 1.0, n)

    # Define B arrays.
    bz = np.zeros(n)
    bq = np.zeros(n)

    # Integrate pressure gradient for profile and average pressure
    # factor.
    if pmodel == 'quadratic':
        press = press_quadratic
    elif pmodel == 'cubic':
        press = press_cubic
    p, gradp = press(rho, 2.0)
    p = p - p[-1]
    avg_p_fac = 0.5 / sig.simps(rho*p, rho)

#    beta0_tol = 1E-3
    # 1E-3 gives same number of iterations as m_max=4 with no condition.
    for m in range(m_max): #loop for different beta
        if m == 0:      #first time, zero beta
            beta0 = 0.0
        else:    #next times, derive beta0 for given beta
            #general pressure profile
            beta0 = avg_p_fac * beta * bq[-1]**2
#            print beta0, abs(beta0 - beta0_old) / beta0
#            if abs(beta0 - beta0_old) / beta0 < beta0_tol:
#                break
#        beta0_old = beta0
#        print beta0

        bz[0] = 1.0         #axis values of
        bq[0] = 0.0         #field components

        for i in range(n-1):
            x = rho[i]
            y = lam(x, *pars)
            p, z = press(x, beta0)

            t1_z = h * zfunc(x, bz[i], bq[i], y, z)
            t1_q = h * qfunc(x, bz[i], bq[i], y, z)

            x = rho[i] + hh
            y = lam(x, *pars)
            p, z = press(x, beta0)
            t2_z = h * zfunc(x, bz[i]+t1_z/2.0, bq[i]+t1_q/2.0, y, z)
            t2_q = h * qfunc(x, bz[i]+t1_z/2.0, bq[i]+t1_q/2.0, y, z)

            t3_z = h * zfunc(x, bz[i]+t2_z/2.0, bq[i]+t2_q/2.0, y, z)
            t3_q = h * qfunc(x, bz[i]+t2_z/2.0, bq[i]+t2_q/2.0, y, z)

            x = rho[i+1]
            y = lam(x, *pars)
            p, z = press(x, beta0)
            t4_z = h * zfunc(x, bz[i]+t3_z, bq[i]+t3_q, y, z)
            t4_q = h * qfunc(x, bz[i]+t3_z, bq[i]+t3_q, y, z)

            bz[i+1] = bz[i] + (t1_z + 2.0*t2_z + 2.0*t3_z + t4_z) / 6.0
            bq[i+1] = bq[i] + (t1_q + 2.0*t2_q + 2.0*t3_q + t4_q) / 6.0
#        print m

    # Calculate corrections to fields.
    #d = 0.01    # outboard gap between LCFS & shell, in meters
    if corr == 'tor':
        b = a - d / (1.0 - a / Ra)  #LCFS plasma radius, in meters
        Rb = Ra + a - b - d      #LCFS plasma major radius, in meters
        # Note b = 0.504694, Rb = 1.50531 for MST.
        # Toroidal geometry factors
        tg_a = Ra * (1.0 - np.sqrt(1.0 - (a / Ra)**2) )
        tg_b = Rb * (1.0 - np.sqrt(1.0 - (b / Rb)**2) )
    elif corr == 'cyl':
        b = a - d   #LCFS plasma radius, in meters
        Rb = Ra + a - b - d      #LCFS plasma major radius, in meters
        # Note b = 0.51, Rb = Ra = 1.5 for MST.
    # Get final field profiles, where bz is done before bq to avoid a bug.
    bpw = mu0 * ip / 2.0 / np.pi / a
    bpw_b = bpw * a / b
    bz = bz * bpw_b / bq[-1]
    bq = bq * bpw_b / bq[-1]
    btave_b = 2.0 * sig.simps(rho * bz, rho)
    # New beta0 value may be slightly inconsistent with fields,
    # so recalculate it.
    beta0 = avg_p_fac * beta * bq[-1]**2
    # Find BTW and BTAVE using values at/inside LCFS
    if corr == 'tor':
        btw = bz[-1] / tg_b * tg_a / (a / b)**2
        btave = ( btave_b + bz[-1] * (tg_a / tg_b - 1.0) ) / (a / b)**2
    elif corr == 'cyl':
        btw = bz[-1]
        btave = ( btave_b * b**2 + btw * (a**2 - b**2) ) / a**2
    if ret == 'scalars':
        return ip, btw, btave, bz[0], beta0
    elif ret == 'all':
        # Get pressure and gradient in MKS.
        p, gradp = press(rho, beta0)
        p = bz[0] * bz[0] / mu0 * p
        gradp = bz[0] * bz[0] / mu0 / b * gradp
        # Safety factor q = r * bt / (Ra * bp)
        #q = deriv(r * bz) / deriv(Ra * bq)
        y = lam(0.0, *pars)
        q = 2.0 * b / Rb / y + np.zeros(n)
        q[1:] = rho[1:] * b * bz[1:] / Rb / bq[1:]
        # Added 2015-10, KM
        q[0] = np.polyval(np.polyfit(rho[1:4], q[1:4], 2), rho[0])
        # Get parallel current in MKS.
        y = lam(rho, *pars)
        jq = y * bq / mu0 / b
        jz = y * bz / mu0 / b
        # Add perpendicular current for ASSUMED pressure profile.
        bb = bz * bz + bq * bq
        jq = jq + bz / bb * gradp
        jz = jz - bq / bb * gradp
        # Get total poloidal and toroidal fluxes (not per radian).
        r = rho*b
        psi = 2.0*np.pi*Ra*sig.cumtrapz(
            np.append(bq, bpw), np.append(r, a), initial=0.0)
        Psi = psi[-1]
        psi = psi[:-1]
        phi = 2.0*np.pi*sig.cumtrapz(
            np.append(r, a)*np.append(bz, btw), np.append(r, a),
            initial=0.0)
        Phi = phi[-1]
        phi = phi[:-1]
        return {
            'ip':ip, 'bpw':bpw, 'btw':btw, 'btave':btave, 'b0':bz[0],
            'beta0':beta0, 'F':btw/btave, 'Theta':bpw/btave,
            'bpw_b':bpw_b, 'btw_b':bz[-1], 'btave_b':btave_b,
            'b0':bz[0], 'beta0':beta0,
            'a':a, 'Ra':Ra, 'd':d, 'b':b, 'Rb':Rb, 'rho':rho, 'r':r,
            'bq':bq, 'bz':bz, 'jq':jq, 'jz':jz,
            'Psi':Psi, 'psi':psi, 'Phi':Phi, 'phi':phi,
            'p':p, 'gradp':gradp,
            'q':q, 'lam':y,
            'pars':pars, 'pmodel':pmodel, 'beta':beta,
            'corr':corr
            }

