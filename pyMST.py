#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python implementation of John Sarff's MST code
# Imène Goumiri
# October 11, 2016
# MIT License

import sys
import toml
import json
import h5py
from hashlib import sha1
import numpy as np
import scipy.integrate as si
import scipy.interpolate as sp
import matplotlib.pyplot as plt

def lmbda(r, alpha, lmbda0):
	return lmbda0 * (1 - r**alpha)

def gradP(r, beta0, c1, c2): # P(r) = (1 - r**c1)**c2
	return -beta0/2 * c1 * c2 * r**(c1 - 1) * (1 - r**c1)**(c2 - 1)

def gradB(r, B, lmbda0, alpha, beta0, c1, c2):
	B_phi, B_theta = B # toroidal, poloidal
	gradP_norm = gradP(r, beta0, c1, c2) / (B_phi**2 + B_theta**2)
	lmbda1 = lmbda(r, alpha, lmbda0)
	gradB_phi = (-lmbda1 * B_theta) - gradP_norm * B_phi
	gradB_theta = lmbda1 * B_phi - (1 / max(r, 1e-4) + gradP_norm) * B_theta
	return [gradB_phi, gradB_theta]

def calcB(lmbda0, alpha, config):
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

def lmbdaDot(B_phi, B_theta, V_phi, V_theta, V_phi_DC, P_ohm, U_mag, U_mag_dot, flux, a, mu0, eta0, R0):
	I_phi = B_theta * flux / (a * mu0)
	I_theta = B_phi * flux * R0 / (a**2 * mu0)
	P_diss = P_ohm * flux**2 * eta0 * R0 / (a**4 * mu0**2) # check eta0 and P_ohm
	num = ((V_phi_DC + V_phi) * I_phi - V_theta * I_theta - P_diss) * 2 * mu0 * a**2 / R0
	num += 2 * V_theta * flux * U_mag
	den = flux**2 * U_mag_dot
	return num / den

def lfDot(t, x, time, spl_B_phi, spl_B_theta, V_phi_wave, V_theta_wave, V_phi_DC, spl_P_ohm, spl_U_mag, alpha, a, mu0, eta0, R0):
	lmbda0, flux = x
	B_phi, B_theta = spl_B_phi(lmbda0, alpha), spl_B_theta(lmbda0, alpha)
	V_phi, V_theta = np.interp(t, time, V_phi_wave), np.interp(t, time, V_theta_wave)
	P_ohm = spl_P_ohm(lmbda0, alpha)
	U_mag = spl_U_mag(lmbda0, alpha)
	U_mag_dot = spl_U_mag(lmbda0, alpha, dx=1)
	lmbda_dot = lmbdaDot(B_phi, B_theta, V_phi, V_theta, V_phi_DC, P_ohm, U_mag, U_mag_dot, flux, a, mu0, eta0, R0)
	flux_dot = -V_theta
	return [lmbda_dot, flux_dot]

def Te0(I_phi, density, a):
	# return 2.8 * (2 * a)**0.83 * (I_phi * 1e-3)**0.67 * (I_phi * 1e14 / (np.pi * a**2 * density))**0.51
	return max(2.8 * (2 * a)**0.83 * np.sqrt(I_phi / (density * np.pi * a**2) * 1e14) * (I_phi * 1e-3)**0.67, 20)

def Pohm(r, B, lmbda):
	eta = 1 / (0.95 * (1 - r**8) + 0.05)**1.5
	return si.simps(r * eta * lmbda**2 * (B[:,0]**2 + B[:,1]**2), r)

def Umag(r, B):
	return si.simps(r * (B[:,0]**2 + B[:,1]**2), r)

def time(config):
	t0, tf, dt = config["t0"], config["tf"], config["dt"]
	num = int(round((tf - t0) / dt)) + 1
	return np.linspace(t0, tf, num=num), num

def linspace(config):
	num = config["ndiv"] + 1
	return np.linspace(config["min"], config["max"], num=num), num

def preCalc(config):
	list_lmbda0, num_lmbda0 = linspace(config["lambda_0"])
	list_alpha, num_alpha = linspace(config["alpha"])

	B_phi = np.empty((num_lmbda0, num_alpha))
	B_theta = np.empty((num_lmbda0, num_alpha))
	P_ohm = np.empty((num_lmbda0, num_alpha))
	U_mag = np.empty((num_lmbda0, num_alpha))

	for (i, lmbda0) in enumerate(list_lmbda0):
		for (j, alpha) in enumerate(list_alpha):
			r, B = calcB(lmbda0, alpha, config)
			flux = si.simps(r * B[:,0], r)
			B_phi[i,j] = B[-1,0] / flux
			B_theta[i,j] = B[-1,1] / flux
			P_ohm[i,j] = Pohm(r, B, lmbda(r, alpha, lmbda0)) / flux**2
			U_mag[i,j] = Umag(r, B) / flux**2

	return list_lmbda0, list_alpha, B_phi, B_theta, P_ohm, U_mag

def calc(list_lmbda0, list_alpha, grid_B_phi, grid_B_theta, grid_P_ohm, grid_U_mag, config):
	# Make splines for magnetic fields, ohmic power and magnetic energy
	spl_B_phi = sp.RectBivariateSpline(list_lmbda0, list_alpha, grid_B_phi)
	spl_B_theta = sp.RectBivariateSpline(list_lmbda0, list_alpha, grid_B_theta)
	spl_P_ohm = sp.RectBivariateSpline(list_lmbda0, list_alpha, grid_P_ohm)
	spl_U_mag = sp.RectBivariateSpline(list_lmbda0, list_alpha, grid_U_mag)

	# Parameters
	mu0 = config["mu0"]
	aspect_ratio = config["aspect_ratio"]
	a = config["a"]
	R0 = aspect_ratio * a
	zneo = config["zneo"]
	zeff = config["zeff"]
	zohm = (0.4 + 0.6 * zeff) * zneo

	alpha = 4.0 # FIXME: using hardcoded fixed alpha for now, for comparison with IDL code

	# Time discretization
	t, num_t = time(config["time"])

	# Parameters specific to simulation mode
	mode = config["mode"]
	if mode not in config:
		raise ValueError("parameters for mode '%s' missing from config file" % mode)

	config_mode = config[mode]
	flux0 = config_mode["flux_ref"] * config_mode["flux_multiplier"]
	density = config_mode["density_ref"] * config_mode["density_multiplier"] * np.ones(num_t)

	config_phi = config_mode["toroidal"]
	V_phi_wave = np.interp(t, config_phi["time"], config_phi["voltage"])
	V_phi_DC = config_phi["DC_voltage"]

	config_theta = config_mode["poloidal"]
	V_theta_wave = np.interp(t, config_theta["time"], config_theta["voltage"])

	config_anom = config_mode["anomalous"]
	zanom_wave = np.interp(t, config_anom["time"], config_anom["voltage"])

	if "OPCD" in config_mode:
		config_opcd = config_mode["OPCD"]
		t_start = config_opcd["t_start"]
		freq = config_opcd["freq"]
		domain = t >= t_start & t <= t_start + config_opcd["cycles"] / freq
		V_theta_wave -= config_opcd["max_voltage"] * np.sin(2 * np.pi * freq * t) * domain

	if "OFCD" in config_mode:
		config_ofcd = config_mode["OPCD"]
		t_start = config_ofcd["t_start"]
		freq = config_ofcd["freq"]
		cycles = config_ofcd["cycles"]
		phase = config_ofcd["phase"]
		sin_wave = config_ofcd["max_voltage"] * np.sin(2 * np.pi * freq * t)
		theta_domain = t >= t_start & t <= t_start + cycles / freq
		V_theta_wave -= sin_wave * theta_domain
		phi_domain = t >= t_start + phase / freq & t <= t_start + (phase + cycles) / freq
		V_phi_wave -= sin_wave * phi_domain

	if "buck" in config_mode:
		config_buck = config_mode["buck"]
		V_phi_wave += np.interp(t, config_buck["time"], config_buck["voltage"])

	if "Te" in config_mode:
		config_Te = config_mode["Te"]
		PPCD_Te_mult += np.interp(t, config_Te["time"], config_Te["voltage"])

	# Allocate arrays
	flux = np.zeros(num_t)
	lmbda0 = np.zeros(num_t)
	I = np.zeros((num_t, 2)) # Current
	P_ohm = np.zeros(num_t)

	# Initial conditions
	I[0,0] = config["initial"]["I_phi"]
	flux[0] = I[0,0] / 100.0 * flux0
	list_B_theta = np.squeeze(spl_B_theta(list_lmbda0, alpha))
	lmbda0[0] = sp.CubicSpline(list_B_theta, list_lmbda0, bc_type="natural")(mu0 * a * I[0,0] / flux[0])
	I[0,1] = spl_B_phi(lmbda0[0], alpha) * flux[0] * R0 / (a**2 * mu0)

	# Time integration
	solver = si.ode(lfDot)
	solver.set_integrator("dopri5")
	solver.set_initial_value([lmbda0[0], flux[0]], t[0])
	for i in xrange(1, num_t):
		eta0 = zanom_wave[i-1] * 1.6 * 7.75e-4 * zohm / Te0(I[i-1,0], density[i-1], a)**1.5
		if mode is "PPCD_550KA":
			eta0 /= PPCD_Te_mult[i-1]**1.5
		solver.set_f_params(t, spl_B_phi, spl_B_theta, V_phi_wave, V_theta_wave, V_phi_DC, spl_P_ohm, spl_U_mag, alpha, a, mu0, eta0, R0)
		lmbda0[i], flux[i] = solver.integrate(t[i])
		B_phi, B_theta = spl_B_phi(lmbda0[i], alpha), spl_B_theta(lmbda0[i], alpha)
		I[i,0] = B_theta * flux[i] / (a * mu0)
		I[i,1] = B_phi * flux[i] * R0 / (a**2 * mu0)
		P_ohm[i] = spl_P_ohm(lmbda0[i], alpha) * flux[i]**2 * eta0 * R0 / (a**4 * mu0**2)
		if not solver.successful():
			print("Warning: Integration failed at t={0}".format(t[i]))
			break
		if i > num_t/10 and I[i,0] < 1e4:
			print("Info: Integration stopped at t={0}".format(t[i]))
			# t[i:] = np.nan # strip plots
			break

	return t, I, flux, P_ohm

def plotVphiAndBPCoreFlux(t, V_phi, BP_core_flux):
	plt.plot(t, V_phi)
	plt.plot(t, 100 * BP_core_flux)
	plt.axhline(y=100*1.9/2, linestyle="dashed")
	plt.xlabel("Time (s)")
	plt.ylabel("Voltage (V)")
	plt.title(r"$V_\phi$ & BP core flux ($\times 100$)")
	plt.legend([r"$V_\phi$", r"BP core flux ($\times 100$)"])
	plt.grid()

def plotVtheta(t, V_theta):
	plt.plot(t, V_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Voltage (V)")
	plt.title(r"$V_\theta$")
	plt.grid()

def plotFlux(t, flux):
	plt.plot(t, flux)
	plt.xlabel("Time (s)")
	plt.ylabel("Flux (Wb)")
	plt.title("BT Flux")
	plt.grid()

def plotIphi(t, I_phi, BP_core_flux):
	plt.plot(t, 1e-3 * I_phi)
	plt.plot(t, np.exp(7 * BP_core_flux / 0.58) * 0.67e-3)
	plt.xlabel("Time (s)")
	plt.ylabel("Current (kA)")
	plt.title(r"$I_\phi$ & $I_\mathrm{mag}$")
	plt.legend([r"$I_\phi$", r"$I_\mathrm{mag}$"])
	plt.grid()

def plotItheta(t, I_theta):
	plt.plot(t, 1e-6 * I_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Current (MA)")
	plt.title(r"$I_\theta$")
	plt.grid()

def plotPohmOverIphi(t, P_ohm, I_phi):
	plt.plot(t, P_ohm / np.max(I_phi, 1e-20))
	plt.xlabel("Time (s)")
	plt.ylabel("Voltage (V)")
	plt.title(r"$P_\mathrm{ohm}/I_\phi$")
	plt.grid()

def plotVphiTimesIphi(t, V_phi, I_phi):
	plt.plot(t, 1e-6 * V_phi * I_phi)
	plt.xlabel("Time (s)")
	plt.ylabel("Power (MW)")
	plt.title(r"$V_\phi \cdot I_\phi$")
	plt.grid()

def plotVthetaTimesItheta(t, V_theta, I_theta):
	plt.plot(t, 1e-6 * V_theta * I_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Power (MW)")
	plt.title(r"$V_\theta \cdot I_\theta$")
	plt.grid()

def plotThetaAndF(t, theta, f):
	plt.plot(t, theta)
	plt.plot(t, f)
	plt.plot(t, -f) # Is that what F is?
	plt.xlabel("Time (s)")
	plt.ylabel(r"$\Theta$ & $F$")
	plt.title(r"$\Theta$ & $F$")
	plt.legend([r"$\Theta$", r"$F$"])
	plt.grid()

def plotEnergyPhi(t, ener_phi, abs_ener_phi):
	plt.plot(t, ener_phi)
	plt.plot(t, abs_ener_phi)
	plt.xlabel("Time (s)")
	plt.ylabel("Energy (kJ)")
	plt.title(r"$\int V_\phi I_\phi$ & $\int |V_\phi I_\phi|$")
	plt.legend([r"$\int V_\phi I_\phi$", r"$\int |V_\phi I_\phi|$"])
	plt.grid()

def plotEnergyTheta(t, ener_theta, abs_ener_theta):
	plt.plot(t, ener_theta)
	plt.plot(t, abs_ener_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Energy (kJ)")
	plt.title(r"$\int V_\theta I_\theta$ & $\int |V_\theta I_\theta|$")
	plt.legend([r"$\int V_\theta I_\theta$", r"$\int |V_\theta I_\theta|$"])
	plt.grid()

def plotB(r, B):
	plt.plot(r, B)
	plt.xlabel(r"$r$")
	plt.ylabel("Magnetic field (T)")
	plt.legend([r"$B_\phi$", r"$B_\theta$"])
	plt.grid()

def plotI(t, I):
	plt.plot(t, I)
	plt.xlabel("Time (s)")
	plt.ylabel("Current (A)")
	plt.legend([r"$I_\phi$", r"$I_\theta$"])
	plt.grid()

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

def run():
	# Parse config file
	if len(sys.argv) != 2:
		print("Usage: {0} <config.toml>".format(sys.argv[0]))
		sys.exit(1)
	with open(sys.argv[1]) as config_file:
		config = toml.loads(config_file.read())

	# Load or pre-compute magnetic fields, ohmic power and magnetic energy
	filename = hashname(config)
	try: # Load pre-computed data set if it exists
		with h5py.File(filename, "r") as file:
			lmbda0 = file["lambda_0"][:]
			alpha = file["alpha"][:]
			B_phi = file["B_phi"][:]
			B_theta = file["B_theta"][:]
			P_ohm = file["P_ohm"][:]
			U_mag = file["U_mag"][:]
	except: # Otherwise, pre-compute and save the results
		with h5py.File(filename, "w") as file:
			lmbda0, alpha, B_phi, B_theta, P_ohm, U_mag = preCalc(config)
			file.create_dataset("lambda_0", data=lmbda0)
			file.create_dataset("alpha", data=alpha)
			file.create_dataset("B_phi", data=B_phi)
			file.create_dataset("B_theta", data=B_theta)
			file.create_dataset("P_ohm", data=P_ohm)
			file.create_dataset("U_mag", data=U_mag)

	# Run program
	t, I, flux, P_ohm = calc(lmbda0, alpha, B_phi, B_theta, P_ohm, U_mag, config)

	# Plot results
	V_phi = np.interp(t, config[config["mode"]]["toroidal"]["time"], config[config["mode"]]["toroidal"]["voltage"])
	V_theta = np.interp(t, config[config["mode"]]["poloidal"]["time"], config[config["mode"]]["poloidal"]["voltage"])
	BP_core_flux = si.cumtrapz(V_phi, t, initial=0) - (0.72 + 0.13)
	mu0 = config["mu0"]
	aspect_ratio = config["aspect_ratio"]
	a = config["a"]
	R0 = aspect_ratio * a
	f = mu0 * a**2 * I[:,1] / (2 * R0 * np.max(flux, 1e-20))
	theta = mu0 * a * I[:,0] / (2 * np.max(flux, 1e-20))
	ener_phi = 1e-3 * si.cumtrapz(V_phi * I[:,0], t, initial=0)
	abs_ener_phi = 1e-3 * si.cumtrapz(np.abs(V_phi * I[:,0]), t, initial=0)
	ener_theta = 1e-3 * si.cumtrapz(V_theta * I[:,1], t, initial=0)
	abs_ener_theta = 1e-3 * si.cumtrapz(np.abs(V_theta * I[:,1]), t, initial=0)
	tf = config["time"]["tf"]

	plt.rc("font", family="serif")

	plt.subplot(4, 3, 1)
	plotVphiAndBPCoreFlux(t, V_phi, BP_core_flux)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 2)
	plotVtheta(t, V_theta)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 3)
	plotFlux(t, flux)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 4)
	plotIphi(t, I[:,0], BP_core_flux)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 5)
	plotItheta(t, I[:,1])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 6)
	plotPohmOverIphi(t, P_ohm, I[:,0])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 7)
	plotVphiTimesIphi(t, V_phi, I[:,0])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 8)
	plotVthetaTimesItheta(t, V_theta, I[:,1])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 9)
	plotThetaAndF(t, theta, f)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 10)
	plotEnergyPhi(t, ener_phi, abs_ener_phi)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 11)
	plotEnergyTheta(t, ener_theta, abs_ener_theta)
	plt.xlim(xmax=tf)

	plt.show()

if __name__ == "__main__":
	run()
