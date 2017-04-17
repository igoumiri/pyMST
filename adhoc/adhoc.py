def pars_to_eq(pars, ip,
    lmodel='alpha', pmodel='quadratic', beta=0.0, n=101, ret='all',
    corr='cyl', d=0.01, a=0.52, Ra=1.50):
    """Input lambda model
    pars=parameters, ip=ip in A, lmodel=lambda-model type,
    ret=what to return, pmodel=pressure-model type, beta=beta_poloidal,
    and n=number of radial points
    to output values from cyleq.lam_to_eq()."""
    import functools as ft
    import cyleq as ce
    reload(ce)
    if lmodel == 'alpha':
        import alpha as al
        reload(al)
        lam = al.lam
    if lmodel == 'twopow':
        import twopow as tp
        reload(tp)
        lam = tp.lam
    if lmodel == 'alptanh':
        import alptanh as at
        reload(at)
        lam = at.lam
    if lmodel == 'mbfm':
        import mbfm as mb
        reload(mb)
        lam = mb.lam
    if lmodel == 'pbfm':
        import pbfm as pb
        reload(pb)
        lam = pb.lam
    if lmodel == 'gamma':
        import gamma as gm
        reload(gm)
        lam = gm.lam
    return ce.lam_to_eq(lam, pars, ip, pmodel, beta, n, ret,
        corr, d, a, Ra)

def errlam(pars, ip, btw, btave, b0, lmodel, pmodel, beta, n,
    corr, d, a, Ra):
    """Given pars, ip, btw, btave, lmodel, pmodel, beta, n,
    return error between
    the equilibrium calculation for btw and btave
    and the given btw and btave.
    This should work as intended
    if ip, btw, and btave are in MKS units."""
    import numpy as np
    npars = len(pars)
    ipout, btwout, btaveout, b0out, beta0out = pars_to_eq(pars, ip,
        lmodel, pmodel=pmodel, beta=beta, n=n, ret='scalars',
        corr=corr, d=d, a=a, Ra=Ra)
    if abs(btw) < 1E-4: btw = -1E-4
    if npars == 2:
        return np.sqrt( (btwout - btw)**2 + (btaveout - btave)**2 )
    else:    # npars = 3
        return np.sqrt( (btwout - btw)**2 + (btaveout - btave)**2
            + ((b0out - b0) / 40.0)**2 )
        # That assumes 1% error for btave (1000 G)
        # and 10% for b0 (4000 G).
    #return np.sqrt( (btwout - btw)**2 + (btaveout - btave)**2 )

def getpars(pars=[3.25, 4.0], ip=100E3, btw=0.01, btave=0.025, b0=0.1,
    lmodel='alpha', pmodel='quadratic', beta=0.07, n=[4,101],
    disp=0, corr='cyl', d=0.01, a=0.52, Ra=1.50):
    """Simple wrapper to loop fmin(errlam) to get parameters
    from the input field values."""
    import scipy.optimize as so
    for j in n:
        pars[0], fopt, ite, funcalls, warnflag = so.fmin(errlam, pars[:1],
            args=(ip, btw, btave, b0, lmodel, pmodel, beta, j, corr,
            d, a, Ra),
            disp=disp, full_output=1)
    return pars, warnflag
    # Note in call to fmin, just changing xtol from default 1E-4 to
    # 1E-3 cuts down on iterations by <10%, and just doing the same
    # to ftol makes no difference in the test I did.
    # Note so.fmin_powell doesn't get as close to the accurate result,
    # and both so.fmin_cg and so.fmin_bfgs usually fail badly.

def f_theta_to_eq(Theta, F, ip=250E3,
    lmodel='alpha', pmodel='quadratic', beta=0.07, n=101,
    d=0.01, corr='tor', a=0.52, Ra=1.50):
    """Convenience function to input Theta, F, ip in A,
    and output the dictionary containing the cylindrical equilibrium
    fit.
    """
    import numpy as np
    mu0 = np.pi * 4E-7
    if lmodel == 'alpha':
        pars0 = [3.25, 4.0]
    if lmodel == 'twopow':
        pars0 = [3.25, 8.0]
    if lmodel == 'alptanh':
        pars0 = [3.25, 4.0, 0.825]
    if lmodel == 'mbfm':
        pars0 = [3.25, 0.8]
    if lmodel == 'pbfm':
        pars0 = [3.25, 0.8]
    if lmodel == 'gamma':
        pars0 = [3.25, 6]
    bpw = mu0 * ip / 2.0 / np.pi / a
    btave = bpw / Theta
    btw = F * btave
    pars, warnflag = getpars(pars0, ip, btw, btave=btave,
        #lmodel=lmodel, pmodel=pmodel, beta=beta, n=[4, n / 10, n],
        lmodel=lmodel, pmodel=pmodel, beta=beta,
        corr=corr, d=d, a=a, Ra=Ra)
    return pars_to_eq(pars, ip,
        lmodel, pmodel=pmodel, beta=beta, n=n, ret='all',
        corr=corr, d=d, a=a, Ra=Ra)

def loop_f_theta_to_pars(Theta, F, btave=1E3,
    lmodel='alpha', pmodel='quadratic', beta=0.0, nradii=101, n=[4,101],
    disp=0, save=False, ret=True, d=0.01, corr='cyl', a=0.52, Ra=1.50):
    """Use lists or generally shaped arrays of Theta and F
    to generate values for the model parameters,
    possibly using an optimizing loop."""
    import numpy as np
    if lmodel == 'alpha':
        pars0 = [3.25, 4.0]
        parnames = ['lam0', 'alpha']
    if lmodel == 'twopow':
        pars0 = [4.0, 4.0]
        parnames = ['lam0', 'alpha']
    if lmodel == 'pbfm':
        pars0 = [3.25, 0.8]
        parnames = ['lam0', 'x0']
    if lmodel == 'alptanh':
        pars0 = [3.25, 0.8]
        parnames = ['lam0', 'x0']
    npars = len(parnames)
    Theta = np.array(Theta)
    F = np.array(F)
    nshots = np.size(Theta)
    shape = np.shape(Theta)
    pars = np.zeros([nshots, npars])
    # pars will be arrays of shape nshots, npars.
    warnflag = np.zeros(nshots).astype(int)
    # reserve array for beta0 values.
    beta0 = np.zeros(nshots)
    # Change Theta, F into quantities used by previously written codes.
    mu0_over_2pi = 2E-7
    btave = btave / 1E4     # G to T
    ip = Theta.flatten() * btave * a / mu0_over_2pi
    btw = F.flatten() * btave
    # Get pars for all shots
    for shot in np.arange(nshots):
        pars[shot, :], warnflag[shot] = getpars(
            pars0, ip[shot], btw[shot], btave=btave,
            lmodel=lmodel, pmodel=pmodel, beta=beta, n=n, disp=disp,
            corr=corr, d=d, a=a, Ra=Ra)
        ipout, btwout, btaveout, b0out, beta0out = pars_to_eq(
            pars[shot, :], ip[shot],
            lmodel, pmodel=pmodel, beta=beta, n=nradii, ret='scalars',
            corr=corr, d=d, a=a, Ra=Ra)
        beta0[shot] = beta0out
        print shot, nshots, pars[shot, :], warnflag[shot]
    if save:
        import h5py as hp
        dirname = 'data_python_adhoc/f_theta_pars/'
        filename = ( 'f_theta_pars'
            + '-' + lmodel
            + '_' + pmodel
            + '_beta'
            + str(round(beta, 2))
            + '_delta'
            + str(round(d*100.0, 2))
            + '.hdf5' )
        File = hp.File(dirname + filename, 'w')
        File.attrs['lmodel'] = lmodel
        File.attrs['pmodel'] = pmodel
        File.attrs['beta'] = beta
        File.attrs['delta'] = d*100.0
        File.attrs['delta_Units'] = 'cm'
        File.attrs['nradii'] = nradii
        File.attrs['parnames'] = parnames
        File['Theta'] = Theta
        File['F'] = F
        dset = File.create_dataset('warnflag',
            data=warnflag.reshape(shape))
        dset.attrs['Units'] = '0: OK, 1: Max. eval., 2: Max. iter.'
        # Break apart pars into separate arrays
        for i in range(npars):
            File[parnames[i]] = pars[:, i].reshape(shape)
        File['beta0'] = beta0.reshape(shape)
        File.close()
        print dirname + filename
    if ret:
        return pars, parnames, warnflag, beta0

def make_f_theta_grid(delTheta=0.025, delF=[0.04, 0.01]):
    """
    Using parameters found in f_theta_line.py, make a parallelogram
    grid of points in (Theta, F) space for use in cylindrical
    equilibrium modeling.  In this function x means Theta and y means
    F.
    """
    import matplotlib.mlab as mm
    import numpy as np
    import functions as fu
    reload(fu)
    # Specify the boundaries of the grid.
    # Original, simple case for 2012-02-17 AM or early PM.
    # Parallelogram case
#def make_f_theta_grid(delTheta=0.025, delF=0.025):
#    [x2, y2] = [1.0, 0.0]
#    [x3, y3] = [2.25, 0.0]
#    [x0, y0] = [2.5, -2.0]
#    [x1, y1] = [3.75, -2.0]
#    Nx = int((x1 - x0) / delTheta + 1)
#    Ny = int((y2 - y0) / delF + 1)
#    x = np.linspace(x0, x1, Nx)
#    y = np.linspace(y0, y2, Ny)
#    X, Y = np.meshgrid(x, y)
#    a, b = fu.line([x0, y0], [x2, y2])
#    for i in range(Ny):
#        X[i] = X[i] - (x0 - (Y[i] - b) / a)
#    return X, Y
    # Updated case from 2012-02-17 late PM or later.
    # Trapezoid case until 2012-10-12
    # called larger_theta_f_space
    # with (delTheta=0.025, delF=[0.04, 0.01]):
    #[x2, y2] = [1.25, 0.125]
    #[x3, y3] = [1.625, 0.125]
    #[x0, y0] = [2.375, -2.0]
    #[x1, y1] = [3.75, -2.0]
    # Trapezoid case until 2012-10-14
    # called smaller_theta_f_space
    # with delTheta=0.05, delF=[0.05, 0.01]
    #[x2, y2] = [1.375, 0.0]
    #[x3, y3] = [1.675, 0.0]
    #[x0, y0] = [2.25, -1.75]
    #[x1, y1] = [3.75, -1.75]
    # Trapezoid case after 2012-10-13 (not used yet):
    # called larger_theta_f_space, like pre-2012-10-12,
    # with (delTheta=0.025, delF=[0.04, 0.01]):
    # MST
    [x2, y2] = [1.25, 0.125]
    [x3, y3] = [1.625, 0.125]
    [x0, y0] = [2.375, -2.0]
    [x1, y1] = [3.75, -2.0]
    # RELAX
    [x2, y2] = [0.75, 0.5]
    [x3, y3] = [1.25, 0.5]
    [x0, y0] = [2.75, -1.5]
    [x1, y1] = [4.25, -1.5]
    # Get the desired delTheta resolution at middle altitude:
    Nx = int(((x1 + x3) - (x0 + x2)) / 2.0 / delTheta + 1)
    # Old way is to use regular y, i.e. constant diff(y).
    #Ny = int((y2 - y0) / delF + 1)
    #y = np.array([[i] for i in np.linspace(y0, y2, Ny)])
    #Y = np.tile(y, Nx)
    # New way is to use y with a varying diff(y).
    # Get the desired delF resolution as a function of f.
    # Note the sum delF[0] + delF[1] should be an integer divisor of
    # 2(y2 - y0).
    Ny = int(2.0 * (y2 - y0) / (delF[0] + delF[1])) + 2
    delFuse = np.append(0.0,
        delF[0] + (delF[1] - delF[0]) / (Ny - 2.0) * np.arange(Ny - 1) )
    y = y0 + np.cumsum(delFuse)
    X = np.zeros([Ny, Nx])
    Y = X.copy()
    aleft, bleft = fu.line([x0, y0], [x2, y2])
    aright, bright = fu.line([x1, y1], [x3, y3])
    Y = np.tile(np.array([y]).T, Nx)
    for i in range(Ny):
        X[i] = np.linspace((y[i] - bleft) / aleft,
            (y[i] - bright) / aright, Nx)
    return X, Y

def testgetpars(times=range(5000),
    lmodel='alpha', pmodel='quadratic', beta=0.07, nradii=101,
    n=[4,101], disp=0, optimize=None, save=False, ret=True, test=False,
    corr='tor', d=0.01, a=0.52, Ra=1.50):
    """Use data Ip, Btw, Btave(, and perhaps B0_MSE) values
    for range of indices times to generate values
    for the model parameters using getpars
    for comparison to database values.
    Each comparison is expressed in terms of the log10
    of the absolute value of the relative error."""
    import numpy as np
    import scipy.io as si
    pars0 = [3.25, 4.0]
    parnames = ['lam0', 'alpha']
    npars = len(pars0)
    # Get data
    dire = '../idl_adhoc/'
    rs = si.readsav(dire + 'test_b0-1080219003.sav')
    time = rs['test']['t'][0]
    ip = rs['test']['ip'][0]
    ip = ip * a / 0.52
    btw = rs['test']['btw'][0]
    btave = rs['test']['btave'][0]
    #b0 = rs['test']['b0'][0]
    alpha = rs['test']['alpha'][0]
    lam0 = rs['test']['lam0'][0]
    # Take sub-arrays
    ip = ip[times]
    btw = btw[times]
    btave = btave[times]
    #b0 = b0[times]
    time = time[times]
    alpha = alpha[times]
    lam0 = lam0[times]
    ntimes = len(times)
    pars = np.zeros([ntimes, npars])
    # pars will be arrays of shape ntimes, npars.
    warnflag = np.zeros(ntimes).astype(int)
    # Get pars for all times
    for time in range(0, ntimes):
        pars[time, :], warnflag[time] = getpars(pars0,
            ip=ip[time], btw=btw[time], btave=btave[time],
            lmodel=lmodel, pmodel=pmodel, beta=beta, n=n,
            disp=disp, corr=corr, d=d, a=a, Ra=Ra)
#        print ( time,
#            np.log10(np.abs((pars[time, 0]
#            - lam0[time])/lam0[time])),
#            np.log10(np.abs((pars[time, 1]
#            - alpha[time])/alpha[time])) )
        print ( time,
            pars[time, 0], lam0[time],
            pars[time, 1], alpha[time] )


def loopgetpars(shots=range(7027),
    lmodel='alpha', pmodel='quadratic', beta=0.07, nradii=101,
    n=[4,101], disp=0, optimize=None, save=False, ret=True, test=False,
    N=32):
    """Use data Ip, Btw, Btave(, and perhaps B0_MSE) values
    for range of indices shots (not mdsplus shot numbers)
    to generate values for the model parameters,
    possibly using an optimizing loop."""
    import numpy as np
    if lmodel == 'alpha':
        pars0 = [3.25, 4.0]
        parnames = ['lam0', 'alpha']
    if lmodel == 'alptanh':
        pars0 = [3.25, 4.0, 0.825]
        parnames = ['lam0', 'alpha', 'x0']
    if lmodel == 'twopow':
        pars0 = [4.0, 4.0, 1E-6]
        parnames = ['lam0', 'alpha', 'gamma']
    if lmodel == 'pbfm':
        pars0 = [3.25, 0.8]
        parnames = ['lam0', 'x0']
    if lmodel == 'twopow3':
        pars0 = [4.0, 4.0]
        parnames = ['lam0', 'alpha']
    if lmodel == 'twopow6':
        pars0 = [4.0, 4.0]
        parnames = ['lam0', 'alpha']
    if lmodel == 'tworeg':
        pars0 = [4.0, 4.0, 0.8]
        parnames = ['lam0', 'alpha', 'x0']
    npars = len(pars0)
    # Get data
    dire='data_python_adhoc/mse_ext-125us/'
    ip=np.load(dire+'ip_kA.npy') * 1E3
    btw=np.load(dire+'btw_G.npy') / 1E4
    btave=np.load(dire+'btave_G.npy') / 1E4
    b0=np.load(dire+'b0mse_G.npy') / 1E4
    pulse = np.load(dire+'pulse.npy')
    time = np.load(dire+'time_s.npy')
    if test:
         lam0 = np.load(dire+'lam0_a.npy')
         alpha = np.load(dire+'alpha.npy')
    # Take sub-arrays
    ip = ip[shots]
    btw = btw[shots]
    btave = btave[shots]
    b0 = b0[shots]
    pulse = pulse[shots]
    time = time[shots]
    nshots = len(shots)
    pars = np.zeros([nshots, npars])
    # pars will be arrays of shape nshots, npars.
    warnflag = np.zeros(nshots).astype(int)
    # Get pars for first shot
    if optimize == 'table':
        print optimize
        import h5py as hp
        import scipy.optimize as so
        dirname = 'data_python_adhoc/pars_to_b_table_N~' + str(N) + '/'
        filename = ( 'pars_to_b_table'
            + '-lmodel~' + lmodel
            + '_pmodel~' + pmodel
            + '_beta~' + str(int(beta * 100)).zfill(2)
            + '_n~' + str(nradii) + '.hdf5' )
        print dirname + filename
        File = hp.File(dirname + filename, 'r')
        parsax = []
        for i in range(npars):
            parsax.append(File[File.attrs['parnames'][i]].value)
        b = File['b'].value
        File.close()
        for shot in range(0, nshots):
            print shot
            # Configure input data
            bexp = np.array([btw[shot], btave[shot], b0[shot]]
                )[:npars]/ip[shot]*1E4*1E3
            if abs(bexp[0]) < 1E-4:
                bexp[0] = -1E-4
            chi2 = np.sqrt(np.sum((((b.T - bexp)/bexp)**2).T, axis=0))
            wmin = np.where(chi2 == np.min(chi2))
            parsuse = np.array([parsax[i][wmin[i]]
                for i in range(len(parsax))]).T
            for j in n:
                pars[shot, :], fopt, ite, funcalls, warnflag[shot] = (
                    so.fmin(errlam, parsuse,
                    args=(ip[shot], btw[shot], btave[shot],
                    b0[shot], lmodel, pmodel, beta, j), disp=disp,
                    full_output=1) )
    elif optimize == 'loop':
        print optimize
        inputs = np.transpose([btw/ip, btave/ip, b0/ip]) * 1E4 * 1E3
        # inputs will be arrays of shape nshots, npars.
        pars[0, :], warnflag[0] = getpars(pars0,
            ip=ip[0], btw=btw[0], btave=btave[0], b0=b0[0],
            lmodel=lmodel, pmodel=pmodel, beta=beta, n=n, disp=disp,
            corr='tor')
        print 0, pars[0, :], warnflag[0]
        # Get pars for other shots
        for shot in range(1, nshots):
            deviation = np.sum((inputs[shot] - inputs[:shot])**2,
                axis=1)
            w = np.where(deviation == min(deviation))[0]
            pars[shot, :], warnflag[shot] = getpars(pars[w, :],
                ip=ip[shot], btw=btw[shot],
                btave=btave[shot], b0=b0[shot],
                lmodel=lmodel, pmodel=pmodel, beta=beta, n=n,
                disp=disp, corr='tor')
            print shot, pars[shot, :], warnflag[shot]
    else:
        # Get pars for all shots
        for shot in range(0, nshots):
            pars[shot, :], warnflag[shot] = getpars(pars0,
                ip=ip[shot], btw=btw[shot],
                btave=btave[shot], b0=b0[shot],
                lmodel=lmodel, pmodel=pmodel, beta=beta, n=n,
                disp=disp, d=0.01, corr='tor', a=0.52, Ra=1.50)
            if test:
                print ( shot,
                    np.log10(np.abs((pars[shot, 0]
                    - lam0[shot])/lam0[shot])),
                    np.log10(np.abs((pars[shot, 1]
                    - alpha[shot])/alpha[shot])) )
            else:
                print shot, pars[shot, :], warnflag[shot]
    if save:
        import h5py as hp
        dirname = 'data_python_adhoc/expt_pars/'
        step = shots[1] - shots[0]
        filename = ( 'expt_pars'
            + '-' + lmodel
            + '_' + pmodel
            + '_beta' + str(round(beta, 2))
            + '_nradii' + str(nradii)
            + '_step' + str(step) + '.hdf5' )
        File = hp.File(dirname + filename, 'w')
        File.attrs['lmodel'] = lmodel
        File.attrs['pmodel'] = pmodel
        File.attrs['beta'] = beta
        File.attrs['nradii'] = nradii
        File.attrs['step'] = shots[1] - shots[0]
        File.attrs['parnames'] = parnames
        dset = File.create_dataset('pulse', data=pulse)
        dset.attrs['Units'] = '1'
        dset = File.create_dataset('time', data=time)
        dset.attrs['Units'] = 's'
        dset = File.create_dataset('ip', data=ip/1E3)
        dset.attrs['Units'] = 'kA'
        dset = File.create_dataset('btw', data=btw*1E4)
        dset.attrs['Units'] = 'G'
        dset = File.create_dataset('btave', data=btave*1E4)
        dset.attrs['Units'] = 'G'
        dset = File.create_dataset('b0', data=b0*1E4)
        dset.attrs['Units'] = 'G'
        dset = File.create_dataset('warnflag', data=warnflag)
        dset.attrs['Units'] = '0: OK, 1: Max. eval., 2: Max. iter.'
        # Break apart pars into separate arrays
        for i in range(npars):
            dset = File.create_dataset(parnames[i], data=pars[:, i])
            dset.attrs['Units'] = '1'
        File.close()
        print dirname + filename
    if ret:
        return ( pulse, time, ip/1E3, btw*1E4, btave*1E4, b0*1E4,
            parnames, pars, warnflag )

def read_expt_pars(lmodel='alptanh', pmodel='quadratic', beta=0.07,
    nradii=101, step=1):
    """Given filename parameters, output the experimental parameters
    and results of loopgetpars as an hdf5 dataset, not including some
    attributes."""
    import numpy as np
    import h5py as hp
    dirname = 'data_python_adhoc/expt_pars/'
    filename = ( 'expt_pars'
        + '-' + lmodel
        + '_' + pmodel
        + '_beta' + str(round(beta, 2))
        #+ '_nradii' + str(nradii)
        + '_step' + str(step) + '.hdf5' )
    print(dirname + filename)
    File = hp.File(dirname + filename, 'r')
    parnames = File.attrs['parnames']
    npars=len(parnames)
    pulse = File['pulse'].value
    #npulse = len(pulse)
    #pars = File[parnames[0]].value
    #for p in range(1,npars):
    #    pars = np.append(pars, File[parnames[p]].value)
    #pars = pars.reshape(npulse, npars)
    for p in range(npars):
        exec(parnames[p] + " = File['" + parnames[p] + "'].value")
    keys = [
        'parnames',
        'pulse',
        'time',
        'ip',
        'btw',
        'btave',
        'b0']#,
        #'warnflag']
    for p in parnames:
        keys.append(p)
    items = [
        parnames,
        pulse,
        File['time'].value,
        File['ip'].value,
        File['btw'].value,
        File['btave'].value,
        File['b0'].value]#,
        #File['warnflag'].value]
    for p in range(npars):
        items.append(File[parnames[p]].value)
    D = dict(zip(keys, items))
    File.close()
    return D

def comprofmods(Theta, F, pmodel='quadratic',
    beta=0.07, n=100, d=0.01, corr='cyl', a=0.52, Ra=1.50):
    """Based on Theta, F input,
    compare models' profiles, which can be used for testing purposes.
    """
    import numpy as np
    import matplotlib.pyplot as pp
    import cyleq as ce
    reload(ce)
    import functions as fu
    reload(fu)
    alpha = f_theta_to_eq(Theta, F, lmodel='alpha', beta=beta, n=n,
        d=d, corr='tor', a=a, Ra=Ra)
    twopow = f_theta_to_eq(Theta, F, lmodel='twopow', beta=beta, n=n,
        d=d, corr='tor', a=a, Ra=Ra)
#    mbfm = f_theta_to_eq(Theta, F, lmodel='mbfm', n=n)
#    pbfm = f_theta_to_eq(Theta, F, lmodel='pbfm', n=n)
    #gamma = f_theta_to_eq(Theta, F, lmodel='gamma', n=n)

    zc_alpha = fu.zero_crossings(alpha['rho'], alpha['q'])
    zc_twopow = fu.zero_crossings(twopow['rho'], twopow['q'])
#    zc_mbfm = fu.zero_crossings(mbfm['rho'], mbfm['q'])
#    zc_pbfm = fu.zero_crossings(pbfm['rho'], pbfm['q'])
    #zc_gamma = fu.zero_crossings(gamma['rho'], gamma['q'])

    pp.clf()
    pp.subplot(3, 1, 1)
    pp.title('Theta=' + str(Theta) + ', F=' + str(F))
    pp.plot(alpha['rho'], alpha['lam'], label='alpha')
    pp.plot(twopow['rho'], twopow['lam'], label='twopow')
#    pp.plot(mbfm['rho'], mbfm['lam'], label='mbfm')
#    pp.plot(pbfm['rho'], pbfm['lam'], label='pbfm')
    #pp.plot(gamma['rho'], gamma['lam'], label='gamma')
#    pp.axvline(zc_alpha, color='b', linestyle='solid')
#    pp.axvline(zc_twopow, color='g', linestyle='solid')
#    pp.axvline(zc_mbfm, color='g', linestyle='solid')
#    pp.axvline(zc_pbfm, color='c', linestyle='solid')
    #pp.axvline(zc_gamma, color='b', linestyle='solid')
    pp.axhline(0, color='k')
    pp.xlim([0.0, 1])
    pp.ylim(ymin=0)
    pp.xlabel('rho')
    pp.ylabel('lambda*a')
    pp.legend(shadow=True, loc='best', numpoints=1)

    pp.subplot(3, 1, 2)
    pp.plot(alpha['rho'], alpha['q'], label='alpha')
    pp.plot(twopow['rho'], twopow['q'], label='twopow')
#    pp.plot(mbfm['rho'], mbfm['q'], label='mbfm')
#    pp.plot(pbfm['rho'], pbfm['q'], label='pbfm')
    #pp.plot(gamma['rho'], gamma['q'], label='gamma')
    pp.axhline(0, color='k')
    pp.xlabel('rho')
    pp.ylabel('q')
    pp.xlim([0., 1])

    pp.subplot(3, 1, 3)
    pp.plot(alpha['rho'], alpha['bz'], color='b', label='alpha')
    pp.plot(alpha['rho'], alpha['bq'], color='b', linestyle='dashed')
    pp.plot(twopow['rho'], twopow['bz'], color='g', label='twopow')
    pp.plot(twopow['rho'], twopow['bq'], color='g', linestyle='dashed')
    pp.axhline(0, color='k')
    pp.xlabel('rho')
    pp.ylabel('B')
    pp.xlim([0., 1])

def comprof2mods(Theta=1.56, F=-0.24, qres=1.0/6.0,
    lmodel=['alpha', 'alpha'], pmodel='quadratic',
    #beta=0.07, n=100, d=[0.0, 0.0], corr='cyl'):
    beta=0.07, n=100, d=[0.0, 0.52/50.0], corr='cyl'):
    """Based on Theta, F input,
    compare 2 models' profiles for lambda gradient at resonant surface.
    """
    import numpy as np
    import matplotlib.pyplot as pp
    import cyleq as ce
    reload(ce)
    import functions as fu
    reload(fu)
    D = []
    rres = []
    for delta in range(len(d)):
        E = f_theta_to_eq(Theta, F, lmodel=lmodel[delta], n=n,
            d=d[delta])
        D.append(E)
        rres.append(fu.zero_crossings(E['rho']*E['b']/E['a'],
            E['q']-qres))

    color = ['b', 'g']
    pp.clf()
    pp.subplot(2, 1, 1)
    pp.title('Theta=' + str(Theta) + ', F=' + str(F)
        + ', q_res=' + str(round(qres, 4)))
    for delta in range(len(d)):
        pp.plot(D[delta]['rho']*D[delta]['b']/D[delta]['a'],
            D[delta]['lam']*D[delta]['a']/D[delta]['b'],
            label='d = ' + str(round(d[delta]/D[delta]['a'], 2)) + 'a',
            linewidth=2)
        pp.axvline(rres[delta], color=color[delta], linestyle='solid')
    pp.axhline(0, color='k')
    pp.xlim([0.0, 1])
    pp.ylim(ymin=0)
    #pp.xlabel('rho')
    pp.ylabel('lambda*a')
    pp.legend(shadow=True, loc='best', numpoints=1)

    pp.subplot(2, 1, 2)
    for delta in range(len(d)):
        pp.plot(D[delta]['rho']*D[delta]['b']/D[delta]['a'],
            D[delta]['q'], label='d = ' + str(d[delta]) + ' cm')
        pp.axvline(rres[delta], color=color[delta], linestyle='solid')
    pp.axhline(0, color='k')
    pp.axhline(qres, color='r')
    pp.xlabel('r/a')
    pp.ylabel('q')
    pp.xlim([0., 1])

def read_f_theta_hdf5_for_adhoc(lmodel, pmodel='quadratic', beta=0.0,
    delta=1.0):
    """Read from f_theta_pars-...hdf5 file and return data for use in
    adhoc.
    """
    import h5py as hp
    filename = ( 'data_python_adhoc/f_theta_pars/'
    #filename = ( '../data_python_adhoc/f_theta_pars/'
        #+ 'cylindrical_f_theta_pars/' + 'f_theta_pars-'
        + 'toroidal_f_theta_pars/' + 'f_theta_pars-'
        + lmodel + '_'
        + pmodel + '_'
        + 'beta' + str(round(beta, 2))# + '_'
        #+ 'delta' + str(round(delta, 2))
        + '.hdf5' )
    print filename
    File = hp.File(filename, 'r')
    lam0 = File['lam0'].value
    alpha = File['alpha'].value
    #alpha = File['x0'].value
    Theta = File['Theta'].value
    F = File['F'].value
    File.close()
    return lam0, alpha, Theta, F

def detect_double_res(lmodel='alpha', pmodel='quadratic', beta=0.0,
    delta=1.0):
    """
    Read from f_theta_pars-...hdf5 file and loop over equilibria with
    cyleq to find any examples that have double resonances.
    """
    import numpy as np
    import h5py as hp
    import functions as fu
    reload(fu)
    filename = ( 'data_python_adhoc/f_theta_pars/'
        + 'cylindrical_f_theta_pars/'
        + 'f_theta_pars-'
        + lmodel + '_'
        + pmodel + '_'
        + 'beta' + str(round(beta, 2)) + '_'
        + 'delta' + str(round(delta, 2))
        + '.hdf5' )
    print filename
    File = hp.File(filename, 'r')
    lam0 = File['lam0'].value.flatten()
    alpha = File['alpha'].value.flatten()
    Theta = File['Theta'].value.flatten()
    F = File['F'].value.flatten()
    File.close()
    nshots = np.size(lam0)
    idres = []
    for i in range(nshots):
        eq = pars_to_eq([lam0[i], alpha[i]], 1.0,
            lmodel='alpha', pmodel='quadratic',
            beta=0.0, n=100, ret='all', corr='cyl', d=0.01)
        q = eq['q']
        np.append(q, eq['a']*eq['btw'] / eq['Rb'] / eq['bpw'])
        d = fu.deriv(q)
        md = max(d)
        w = np.where(d == md)[0][0]
        if i/100 == i/100.0:
            print i
        if (md >= 0.0) and (w > 2):
            idres.append(i)
            print i, w, md, Theta[i], F[i]
    return idres

def plot_eq_profiles(Theta, F, ip=250E3,
    lmodel='alpha', pmodel='quadratic', beta=0.07, nradii=101,
    d=0.01, corr='tor', a=0.52, Ra=1.50, ptype='fields',
    surfaces=False, origin=False, label=None, talk=False):
    """Given Theta, F, and an equilibrium model, plot profiles of
    magnetic fields for ptype='fields' or q and lambda otherwise.
    Note the version for introductory RFP plots with disappearing dots
    for mode-resonant surfaces is in adhoc_old18.py.
    """
    import matplotlib.artist as ma
    import matplotlib.pyplot as mp
    import numpy as np
    import functions as fu
    reload(fu)
    eq = f_theta_to_eq(Theta, F, ip,
        lmodel=lmodel, pmodel=pmodel, beta=beta, n=nradii, d=d,
        corr=corr, a=a, Ra=Ra)
    rho = eq['rho']
    print 'pars,', eq['pars']
#    print 'bpw/bpave,', eq['bpw']/eq['btave']
#    print 'btw/btave,', eq['btw']/eq['btave']
    print 'b0/btave,', eq['b0']/eq['btave']
#    import matplotlib as m
#    m.rc('text', usetex=True)
#    mp.rcParams.update({
#        'text.latex.preamble': [r'\usepackage{amsmath}']})
#    title = lmodel + r' model, $\beta_\text{p}$=' + str(int(beta*1E2)) + r'\%, d=' \
    title = lmodel + r' model, $\beta_p$=' + str(int(beta*1E2)) \
        + r'%, $d$=' + str(int(d*100)) + 'cm, ' + r'$F$=' + str(F) \
        + r', $\Theta$=' + str(round(Theta, 2))
    if ptype.lower() == 'fields':
        title = r'MST, $I_p$=' + str(int(ip/1E3)) + 'kA, ' + title
        bq = eq['bq']
        bz = eq['bz']
        btave = eq['btave']
        mp.clf()
        mp.title(title)
        if origin:
            mp.axhline(0.0, color='k')
        mp.plot(rho, bq*1E4, label='Bp')
        mp.plot(rho, bz*1E4, label='Bt')
        mp.axhline(btave*1E4, color='r', linestyle='dashed',
            label='Btave')
        mp.grid()
        mp.legend(loc='best')
        mp.xlabel('r/a')
        mp.ylabel('B[G]')
    else:
        q = eq['q']
        lam = eq['lam']
#        lam = lam / max(lam) * max(q)
        #mp.clf()
        mp.subplot(2, 1, 1)
        mp.plot(rho, q)
        mp.title(title)
        #mp.xlabel('r/a')
        mp.ylabel('q')
        mp.legend(loc='best')
        if origin:
            mp.axhline(0.0, color='k')
        mp.grid()
        mp.subplot(2, 1, 2)
        mp.plot(rho, lam, label=label)
        mp.xlabel('r/a')
        mp.ylabel('lambda')
        if origin:
            mp.axhline(0.0, color='k')
        mp.grid()

def scan_plot_q_profiles(lmodel='twopow', surfaces=False, beta=0.07,
    nradii=101, label=None, talk=False):
    """
    Plot a scan of q profiles using the above code.
    """
    import matplotlib.pyplot as mp

    # Deep F scan, natural Theta
#    F = [0.0, -0.25, -0.5, -0.75, -1.0]
#    Theta = [1.525, 1.7, 1.9, 2.125, 2.4]

    # More resolute deep F scan, natural Theta
#    F = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0]
#    Theta = [1.525, 1.675, 1.8, 2.0, 2.2, 2.4]

    # Typical F scan, natural Theta
#    F = [0.0, -0.25, -0.5]
#    Theta = [1.55, 1.7, 1.9]

    # F scan, fixed Theta
#    F = [-0.1, -0.25, -0.4]
#    Theta = [1.7, 1.7, 1.7]

    # Theta scan, fixed F
#    F = [-0.25, -0.25, -0.25]
#    Theta = [1.55, 1.7, 1.85]

    # Sparse typical F scan, natural Theta, for mhd12 talk
    F = [-0.2, 0.0]
    Theta = [1.675, 1.55]
    label = ['Standard RFP', 'F = 0']

    # Sparse typical F scan, natural Theta, for mhd12 talk, 1st page
    F = [-0.2]
    Theta = [1.675]
    label = ['Standard RFP']

    mp.clf()
    plot_q_profile(Theta[0], F[0],
        lmodel=lmodel, beta=beta, nradii=nradii,
#        origin=True, label='F=' + str(F[0]) + ',
#            Theta=' + str(Theta[0]))
        origin=True, label=label[0], talk=talk)
    for i in range(1, len(F)):
        plot_q_profile(Theta[i], F[i],
            lmodel=lmodel, beta=beta, nradii=nradii,
#            label='F=' + str(F[i]) + ', Theta=' + str(Theta[i]))
            label=label[i], talk=talk)
    mp.grid()

def qvals_loop(lmodel='alpha', pmodel='quadratic', beta=0.0, delta=0.0,
    nradii=101):
    """Scan through f_theta_pars-...hdf5 data and return
    q[0], dqdr[0], min(q), max(q)."""
    import numpy as np
    import functions as fu
    reload(fu)
    lam0, alpha, Theta, F = read_f_theta_hdf5_for_adhoc(
        lmodel, pmodel, beta=beta, delta=delta)
    nshots = np.size(lam0)
    q0 = np.zeros(nshots)
    dq0 = np.zeros(nshots)
    qmin = np.zeros(nshots)
    qmax = np.zeros(nshots)
    for i in range(nshots):
        out = pars_to_eq([lam0.flatten()[i], alpha.flatten()[i]], 1E5,
            lmodel=lmodel, pmodel=pmodel, beta=beta,
            n=nradii, ret='all', corr='cyl', d=delta/100.0)
        q = out['q']
        q0[i] = q[0]
        dq0[i] = fu.deriv(q)[0]/(nradii)
        qmin[i] = min(q)
        qmax[i] = max(q)
        if i/100 == i/100.0:
            print i
    return q0, dq0, qmin, qmax

def alpha_lam0_to_theta_f(alpha, lam0, beta=0.0, delta=0.0, n=102,
        corr='cyl', a=0.52, Ra=1.50):
    """Input array-likes of alpha and lambda_0 and output arrays of
    Theta and F, and optionally save them to a numpy .npy file."""
    import numpy as np
    # First arrange to put into array form.
    # Will return as a list if that is how it is given.
    typ = type(alpha)
    if type(lam0) != typ:
        print 'Type error.'
        return
    alpha = np.array(alpha)
    lam0 = np.array(lam0)
    shape = np.shape(alpha)
    if np.shape(lam0) != shape:
        print 'Shape error.'
        return
    size = np.size(alpha)
    if np.size(lam0) != size:
        print 'Size error.'
        return
    Theta = np.zeros(size)
    F = np.zeros(size)
    alpha = alpha.flatten()
    lam0 = lam0.flatten()
    for i in range(size):
        eq = pars_to_eq([lam0[i], alpha[i]], 100E3, beta=beta,
            n=n, ret='all', corr=corr, d=delta/100.0, a=a, Ra=Ra)
        Theta[i] = eq['Theta']
        F[i] = eq['F']
    Theta = Theta.reshape(shape)
    F = F.reshape(shape)
    if typ == list:
        Theta = list(Theta)
        F = list(F)
    return Theta, F

