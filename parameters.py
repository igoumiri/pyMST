# -*- coding: utf-8 -*-

# Lookup table for MST parameters
# Imène Goumiri
# April 11, 2017
# MIT License

import h5py
import numpy as np
import scipy.integrate as si
import scipy.interpolate as sp

from utils import hashname, generateShotData, linspace


def lmbda(r, alpha, lmbda0):
	return lmbda0 * (1 - r**alpha)


def Pohm(r, B, lmbda):
	eta = 1 / (0.95 * (1 - r**8) + 0.05)**1.5
	return si.simps(r * eta * lmbda**2 * (B[:,0]**2 + B[:,1]**2), r)


def Umag(r, B):
	return si.simps(r * (B[:,0]**2 + B[:,1]**2), r)


def gradP(r, beta0, c1, c2): # P(r) = (1 - r**c1)**c2
	return -beta0/2 * c1 * c2 * r**(c1 - 1) * (1 - r**c1)**(c2 - 1)


def gradB(r, B, lmbda0, alpha, beta0, c1, c2):
	B_phi, B_theta = B # toroidal, poloidal
	gradP_norm = gradP(r, beta0, c1, c2) / (B_phi**2 + B_theta**2)
	lmbda1 = lmbda(r, alpha, lmbda0)
	gradB_phi = (-lmbda1 * B_theta) - gradP_norm * B_phi
	gradB_theta = lmbda1 * B_phi - (1 / max(r, 1e-4) + gradP_norm) * B_theta
	return [gradB_phi, gradB_theta]


class LookupTable:
	def __init__(self, config):
		# Shot data is mode-dependent
		mode = config["mode"]
		if mode not in config:
			raise ValueError("parameters for mode '%s' missing from config file" % mode)
		config_mode = config[mode]

		# Generate shot data
		if "shot" in config_mode:
			shot = config_mode["shot"]
			aspect_ratio = config["aspect_ratio"]
			flux_ref = config_mode["flux"]["ref"]
			# TODO save and reload instead of always generating
			generateShotData(shot, aspect_ratio, flux_ref)

		# Load or pre-compute magnetic fields, ohmic power and magnetic energy
		filename = hashname(config)
		try: # Load pre-computed data set if it exists
			with h5py.File(filename, "r") as file:
				alpha = file["alpha"][:]
				lmbda0 = file["lambda_0"][:]
				flux = file["flux"][:]
				B_phi = file["B_phi"][:]
				B_theta = file["B_theta"][:]
				P_ohm = file["P_ohm"][:]
				U_mag = file["U_mag"][:]
				Ip = file["Ip"][:]
				F = file["F"][:]
		except: # Otherwise, pre-compute and save the results
			with h5py.File(filename, "w") as file:
				alpha, lmbda0, flux, B_phi, B_theta, P_ohm, U_mag, Ip, F = self.preCalc(config)
				file.create_dataset("alpha", data=alpha)
				file.create_dataset("lambda_0", data=lmbda0)
				file.create_dataset("flux", data=flux)
				file.create_dataset("B_phi", data=B_phi)
				file.create_dataset("B_theta", data=B_theta)
				file.create_dataset("P_ohm", data=P_ohm)
				file.create_dataset("U_mag", data=U_mag)
				file.create_dataset("Ip", data=Ip)
				file.create_dataset("F", data=F)

		# Make splines for magnetic fields, ohmic power and magnetic energy
		self.B_phi = sp.RectBivariateSpline(alpha, lmbda0, B_phi)
		self.B_theta = sp.RectBivariateSpline(alpha, lmbda0, B_theta)
		self.P_ohm = sp.RectBivariateSpline(alpha, lmbda0, P_ohm)
		self.U_mag = sp.RectBivariateSpline(alpha, lmbda0, U_mag)

		# One more spline
		fixed_alpha = 4.0 # FIXME: using hardcoded fixed alpha for now
		fixed_B_theta = sp.interp1d(alpha, B_theta, axis=0)(fixed_alpha)
		self.B_theta_to_lmbda0 = sp.CubicSpline(fixed_B_theta, lmbda0, bc_type="natural")

		# "translation" interpolators
		index_alpha = np.nonzero(alpha == fixed_alpha)[0][0] # FIXME: at least interpolate
		fixed_F = F[index_alpha,:,:]
		fixed_Ip = Ip[index_alpha,:,:]
		points = np.empty((fixed_F.size, 2))
		lmbda0_values = np.empty(fixed_F.size)
		flux_values = np.empty(fixed_F.size)
		for (i, l0) in enumerate(lmbda0):
			for (j, fl) in enumerate(flux):
				k = i*flux.size+j
				points[k,:] = [fixed_F[i,j], fixed_Ip[i,j]]
				lmbda0_values[k] = l0
				flux_values[k] = fl

		self.lmbda0 = sp.CloughTocher2DInterpolator(points, lmbda0_values, rescale=True)
		self.flux = sp.CloughTocher2DInterpolator(points, flux_values, rescale=True)


	def preCalc(self, config):
		list_alpha, num_alpha = linspace(config["alpha"])
		list_lmbda0, num_lmbda0 = linspace(config["lambda_0"])
		list_flux, num_flux = linspace(config["flux"])

		a = config["a"]
		mu0 = config["mu0"]

		B_phi = np.empty((num_alpha, num_lmbda0))
		B_theta = np.empty((num_alpha, num_lmbda0))
		P_ohm = np.empty((num_alpha, num_lmbda0))
		U_mag = np.empty((num_alpha, num_lmbda0))
		Ip = np.empty((num_alpha, num_lmbda0, num_flux))
		F = np.empty((num_alpha, num_lmbda0, num_flux))

		for (i, alpha) in enumerate(list_alpha):
			for (j, lmbda0) in enumerate(list_lmbda0):
				r, B = self.calcB(lmbda0, alpha, config)
				flux = si.simps(r * B[:,0], r)
				B_phi[i,j] = B[-1,0] / flux
				B_theta[i,j] = B[-1,1] / flux
				P_ohm[i,j] = Pohm(r, B, lmbda(r, alpha, lmbda0)) / flux**2
				U_mag[i,j] = Umag(r, B) / flux**2
				for (k, phi) in enumerate(list_flux):
					Ip[i,j,k] = B[-1,1] * phi / (a * mu0 * flux)
					F[i,j,k] = B[-1,0] / (2 * flux)

		return list_alpha, list_lmbda0, list_flux, B_phi, B_theta, P_ohm, U_mag, Ip, F


	def calcB(self, lmbda0, alpha, config):
		# Parameters
		r, num = linspace(config["rho"])
		beta_theta = config["beta_theta"]
		c1, c2 = config["pressure"]["c1"], config["pressure"]["c2"]
		P = (1 - r**c1)**c2
		P_avg = 2 * si.simps(r * P, r)

		# Magnetic field
		B = np.zeros((num, 2))

		# Iterate to derive poloidal beta and magnetic fields
		max_iter = 10
		for i in xrange(max_iter):
			beta0 = beta_theta * B[-1,1]**2 / P_avg
			B[0,0] = 1.0
			B[0,1] = 0.0

			# Solve ODE
			solver = si.ode(gradB)
			solver.set_integrator("dopri5")
			solver.set_initial_value(B[0,:], r[0])
			solver.set_f_params(lmbda0, alpha, beta0, c1, c2)
			for i in xrange(1, num):
				if not solver.successful():
					print("Warning: Integration failed at ρ={0} (iteration: {1})".format(r[i], i))
					break
				solver.integrate(r[i])
				B[i,:] = solver.y

			# Stop at 0.01% accuracy for poloidal beta
			if abs(1 - beta0 * P_avg / (beta_theta * max(B[-1,1]**2, 1e-8))) <= 1e-4:
				break

		# Return final profile
		return r, B