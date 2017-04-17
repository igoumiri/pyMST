# -*- coding: utf-8 -*-

# Utils - they should maybe go in pyMST instead
# Imène Goumiri
# April 11, 2017
# MIT License

import json
from hashlib import sha1
import numpy as np


def time(config):
	t0, tf, dt = config["t0"], config["tf"], config["dt"]
	num = int(round((tf - t0) / dt)) + 1
	return np.linspace(t0, tf, num=num), num


def linspace(config):
	num = config["ndiv"] + 1
	return np.linspace(config["min"], config["max"], num=num), num


def loadVec(config):
	if "file" in config:
		column = None
		if "column" in config:
			column = config["column"] - 1
		return np.loadtxt(config["file"], usecols=(column,))
	return np.array(config)


def hashname(config):
	# File hash depends only on config values used for the precomputation
	subconfig = {
		"lambda_0": config["lambda_0"],
		"alpha": config["alpha"],
		"rho": config["rho"],
		"beta_theta": config["beta_theta"],
		"pressure": config["pressure"],
	}
	return "z" + sha1(json.dumps(subconfig, sort_keys=True)).hexdigest()[0:7] + ".h5"


def smoothy(x, win=19, iter=20):
	w = np.ones(win) / win
	y = np.copy(x)
	for i in xrange(iter):
		y[win/2:-win/2+1] = np.convolve(y, w, mode='valid')
	return y


def deriv5(x, t):
	n = len(x)
	y = np.zeros(n)

	y[0] = -25 * x[0] + 48 * x[1] - 36 * x[2] + 16 * x[3] - 3 * x[4]
	y[1] = -3 * x[0] - 10 * x[1] + 18 * x[2] - 6 * x[3] + x[4]
	for i in xrange(2, n-3):
		y[i] = x[i-2] - 8.0 * x[i-1] + 8.0 * x[i+1] - x[i+2]
	y[n-2] = 3 * x[n-1] + 10 * x[n-2] - 18 * x[n-3] + 6 * x[n-4] - x[n-5]
	y[n-1] = 25 * x[n-1] - 48 * x[n-2] + 36 * x[n-3] - 16 * x[n-4] + 3 * x[n-5]

	dt = np.ediff1d(t, to_end=t[-1]-t[-2])
	return y / 12 / dt


def generateShotData(shot, aspect_ratio, flux_ref):
	import MDSplus.connection as mdsplus
	mds = mdsplus.Connection("dave.physics.wisc.edu")
	mds.openTree("mst", shot) # 1100903060 -> 10-40, 14-NOV-2005
	try:
		ip = mds.get(r"\ip").data()
		# vla = mds.get(r"\vloop_alpha").data()
		vlp = mds.get(r"\vloop_pfm").data()
		nel  = mds.get(r"\n_co2").data()
		vpg = mds.get(r"\vpg").data()
		# vtg = mds.get(r"\vtg").data()
		btw = mds.get(r"\btw_b").data()
		btave = mds.get(r"\btave_b").data()
		alpha = mds.get(r"\alpha_par").data()
		minor_b = mds.get(r"\minor_b").data()
		tm_ip = mds.get(r"dim_of(\ip)").data()
		tm_nel = mds.get(r"dim_of(\n_co2)").data()
	finally:
		mds.closeAllTrees()

	a = minor_b
	R0 = aspect_ratio * a

	sub = (tm_nel >= tm_ip[0])
	tn = tm_nel[sub]
	density = nel[sub]

	offset = np.argwhere(ip > 0.1)[0,0] + 160
	# offset = 160

	t = tm_ip[offset:]

	alpha = alpha[offset:]

	tn = tn[offset:]
	density = density[offset:]
	density = np.maximum(density * 1.0e6, 1.0e18)

	flux = btave[offset:] * 1e-4 * np.pi * a**2
	flux_multiplier = flux[0] / flux_ref

	V_phi = vpg[offset:]
	V_theta = smoothy(-deriv5(flux, t))

	I_phi = ip[offset:]
	I_theta = btw[offset:] * 0.97 * R0 / 2.0e-7 * 1.0e-10

	P_ohm_over_I_phi = vlp[offset:]

	nz = btave[offset:] != 0
	theta = t * np.nan
	theta[nz] = 4 * I_phi[nz] / btave[offset:][nz]
	f = t * np.nan
	f = btw[offset:][nz] / btave[offset:][nz]

	data = np.column_stack((t, alpha, V_phi, V_theta, flux, I_theta, I_phi, P_ohm_over_I_phi, theta, f))
	header = "flux_multiplier = {0}".format(flux_multiplier)
	np.savetxt("realdata.dat", data, header=header)
	np.savetxt("density.dat", np.column_stack((tn, density)))


# @autosave(subconfig, prefix="x")
# def f(a, b):
# 	return b, a
#
# class autosave:
# 	def __init__(self, config, prefix=""):
# 		self.config = config
# 		self.prefix = prefix
# 	def __call__(self, f):
# 		self.f = f
# 		return self.wrap
# 	def wrap(self, *args, **kwargs):
# 		filename = hashname(self.config, prefix=self.prefix)
# 		try: # Load pre-computed data set if it exists
# 			with h5py.File(filename, "r") as file:
# 				data = Dict()
# 				for key in file:
# 					data[key] = file["lambda_0"][:]
# 				pass
# 		except: # Otherwise, pre-compute and save the results
# 			with h5py.File(filename, "w") as file:
# 				data = self.f(*args, **kwargs)
# 				file.create_dataset("lambda_0", data=lmbda0)
# 				pass