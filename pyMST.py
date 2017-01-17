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
import control

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

def lfDot(t, x, time, spl_B_phi, spl_B_theta, V_phi, V_theta, V_phi_DC, spl_P_ohm, spl_U_mag, alpha, a, mu0, eta0, R0):
	lmbda0, flux = x
	B_phi, B_theta = spl_B_phi(lmbda0, alpha), spl_B_theta(lmbda0, alpha)
	# V_phi, V_theta = np.interp(t, time, V_phi_wave), np.interp(t, time, V_theta_wave)
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

def loadVec(config):
	if "file" in config:
		column = None
		if "column" in config:
			column = config["column"] - 1
		return np.loadtxt(config["file"], usecols=(column,))
	return np.array(config)

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

	if "flux" in config_mode:
		config_flux = config_mode["flux"]
		flux0 = config_flux["ref"] * config_flux["multiplier"]
	else:
		flux0 = config_mode["flux_ref"] * config_mode["flux_multiplier"]

	if "density" in config_mode:
		config_density = config_mode["density"]
		density = np.interp(t, loadVec(config_density["time"]), loadVec(config_density["value"]))
	else:
		density = config_mode["density_ref"] * config_mode["density_multiplier"] * np.ones(num_t)

	config_phi = config_mode["toroidal"]
	V_phi_wave = np.interp(t, loadVec(config_phi["time"]), loadVec(config_phi["voltage"]))
	V_phi_DC = config_phi["DC_voltage"]

	config_theta = config_mode["poloidal"]
	V_theta_wave = np.interp(t, loadVec(config_theta["time"]), loadVec(config_theta["voltage"]))

	config_anom = config_mode["anomalous"]
	zanom_wave = np.interp(t, config_anom["time"], config_anom["voltage"])

	if "OPCD" in config_mode:
		config_opcd = config_mode["OPCD"]
		t_start = config_opcd["t_start"]
		freq = config_opcd["freq"]
		domain = (t >= t_start) & (t <= t_start + config_opcd["cycles"] / freq)
		V_theta_wave -= config_opcd["max_voltage"] * np.sin(2 * np.pi * freq * t) * domain

	if "OFCD" in config_mode:
		config_ofcd = config_mode["OFCD"]
		t_start = config_ofcd["t_start"]
		freq = config_ofcd["freq"]
		cycles = config_ofcd["cycles"]
		phase = config_ofcd["phase"]
		sin_wave = config_ofcd["max_voltage"] * np.sin(2 * np.pi * freq * t)
		theta_domain = (t >= t_start) & (t <= t_start + cycles / freq)
		V_theta_wave[theta_domain] -= sin_wave[theta_domain]
		offset = int(phase / (freq * (t[1] - t[0])))
		phi_domain = np.roll(theta_domain, -offset)
		V_phi_wave[phi_domain] = V_phi_wave[theta_domain] - 10 * sin_wave[theta_domain]

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
	V = np.zeros((num_t, 2)) # Voltage
	P_ohm = np.zeros(num_t)
	eta0 = np.zeros(num_t)
	xh = np.zeros((num_t, 3)) # estimated state (quick and dirty)
	xi = np.zeros((num_t, 3)) # integral state (quick and dirty)

	# Initial conditions
	I[0,0] = config["initial"]["I_phi"]
	flux[0] = flux0 # I[0,0] / 100.0 * flux0
	list_B_theta = np.squeeze(spl_B_theta(list_lmbda0, alpha))
	lmbda0[0] = sp.CubicSpline(list_B_theta, list_lmbda0, bc_type="natural")(mu0 * a * I[0,0] / flux[0])
	I[0,1] = spl_B_phi(lmbda0[0], alpha) * flux[0] * R0 / (a**2 * mu0)

	# Time integration
	controller = control.LQG(config["control"])
	solver = si.ode(lfDot)
	solver.set_integrator("dopri5")
	solver.set_initial_value([lmbda0[0], flux[0]], t[0])
	V[0,:] = V_phi_wave[0], V_theta_wave[0]
	for i in xrange(1, num_t):
		if t[i] < config["control"]["t0"]:
			V[i,:] = V_phi_wave[i], V_theta_wave[i]
			controller.observe(V[i-1,:], I[i-1,:], xh, xi, i)
		else:
			V[i,:] = controller.control(I[i-1,:], xh, xi, i)
		eta0[i] = zanom_wave[i-1] * 1.6 * 7.75e-4 * zohm / Te0(I[i-1,0], density[i-1], a)**1.5
		if mode is "PPCD_550KA":
			eta0[i] /= PPCD_Te_mult[i-1]**1.5
		solver.set_f_params(t, spl_B_phi, spl_B_theta, V[i,0], V[i,1], V_phi_DC, spl_P_ohm, spl_U_mag, alpha, a, mu0, eta0[i], R0)
		lmbda0[i], flux[i] = solver.integrate(t[i])
		B_phi, B_theta = spl_B_phi(lmbda0[i], alpha), spl_B_theta(lmbda0[i], alpha)
		I[i,0] = B_theta * flux[i] / (a * mu0)
		I[i,1] = B_phi * flux[i] * R0 / (a**2 * mu0)
		P_ohm[i] = spl_P_ohm(lmbda0[i], alpha) * flux[i]**2 * eta0[i] * R0 / (a**4 * mu0**2)
		if not solver.successful():
			print("Warning: Integration failed at t={0}".format(t[i]))
			break
		if i > num_t/10 and I[i,0] < 1e4:
			print("Info: Integration stopped at t={0}".format(t[i]))
			# t[i:] = np.nan # strip plots
			break

	# plt.subplot(3, 1, 1)
	# plt.plot(t, xh[:,0])
	# plt.subplot(3, 1, 2)
	# plt.plot(t, xh[:,1])
	# plt.subplot(3, 1, 3)
	# plt.plot(t, xh[:,2])
	# plt.show()

	return t, I, flux, P_ohm, eta0, V[:,0], V[:,1]

def plotVphiAndBPCoreFlux(t, V_phi, BP_core_flux, V_phi_shot=None):
	if V_phi_shot is not None:
		plt.plot(t, V_phi_shot, color="red")
	plt.plot(t, V_phi)
	plt.plot(t, 100 * BP_core_flux)
	plt.axhline(y=100*1.9/2, linestyle="dashed")
	plt.xlabel("Time (s)")
	plt.ylabel("Voltage (V)")
	plt.title(r"$V_\phi$ & BP core flux ($\times 100$)")
	#plt.legend([r"$V_\phi$", r"BP core flux ($\times 100$)"])
	plt.grid()

def plotVtheta(t, V_theta, V_theta_shot=None):
	if V_theta_shot is not None:
		plt.plot(t, V_theta_shot, color="red")
	plt.plot(t, V_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Voltage (V)")
	plt.title(r"$V_\theta$")
	plt.grid()

def plotFlux(t, flux, flux_shot=None):
	if flux_shot is not None:
		plt.plot(t, flux_shot, color="red")
	plt.plot(t, flux)
	plt.xlabel("Time (s)")
	plt.ylabel("Flux (Wb)")
	plt.title("BT Flux")
	plt.grid()

def plotIphi(t, I_phi, BP_core_flux, I_phi_shot=None):
	if I_phi_shot is not None:
		plt.plot(t, I_phi_shot, color="red")
	plt.plot(t, 1e-3 * I_phi)
	# plt.plot(t, np.exp(7 * BP_core_flux / 0.58) * 0.67e-3)
	plt.xlabel("Time (s)")
	plt.ylabel("Current (kA)")
	plt.title(r"$I_\phi$ & $I_\mathrm{mag}$")
	#plt.legend([r"$I_\phi$", r"$I_\mathrm{mag}$"])
	plt.grid()

def plotItheta(t, I_theta, I_theta_shot=None):
	if I_theta_shot is not None:
		plt.plot(t, I_theta_shot, color="red")
	plt.plot(t, 1e-6 * I_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Current (MA)")
	plt.title(r"$I_\theta$")
	plt.grid()

def plotPohmOverIphi(t, P_ohm, I_phi, P_ohm_over_I_phi_shot=None):
	if P_ohm_over_I_phi_shot is not None:
		plt.plot(t, P_ohm_over_I_phi_shot, color="red")
	nz = I_phi != 0
	P_ohm_over_I_phi = t * np.nan
	P_ohm_over_I_phi[nz] = P_ohm[nz] / I_phi[nz]
	plt.plot(t, P_ohm_over_I_phi)
	plt.ylim([-20,120])
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

def plotThetaAndF(t, theta, f, theta_shot=None, f_shot=None):
	if theta_shot is not None and f_shot is not None:
		plt.plot(t, theta_shot, color="red")
		plt.plot(t, f_shot, color="red")
	plt.plot(t, theta, label=r"$\Theta$")
	plt.plot(t, f, label=r"$F$")
	plt.ylim([-300,200])
	plt.xlabel("Time (s)")
	plt.ylabel(r"$\Theta$ & $F$")
	plt.title(r"$\Theta$ & $F$")
	# plt.legend()
	plt.grid()

def plotEnergyPhi(t, ener_phi, abs_ener_phi):
	plt.plot(t, ener_phi)
	plt.plot(t, abs_ener_phi)
	plt.xlabel("Time (s)")
	plt.ylabel("Energy (kJ)")
	plt.title(r"$\int V_\phi I_\phi$ & $\int |V_\phi I_\phi|$")
	# plt.legend([r"$\int V_\phi I_\phi$", r"$\int |V_\phi I_\phi|$"])
	plt.grid()

def plotEnergyTheta(t, ener_theta, abs_ener_theta):
	plt.plot(t, ener_theta)
	plt.plot(t, abs_ener_theta)
	plt.xlabel("Time (s)")
	plt.ylabel("Energy (kJ)")
	plt.title(r"$\int V_\theta I_\theta$ & $\int |V_\theta I_\theta|$")
	# plt.legend([r"$\int V_\theta I_\theta$", r"$\int |V_\theta I_\theta|$"])
	plt.grid()

def plotEta0(t, eta0):
	plt.plot(t, eta0)
	plt.xlabel("Time (s)")
	plt.ylabel(r"$\eta_0$ ($\Omega$)")
	plt.title(r"$\eta_0$")
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

	mode = config["mode"]
	if mode not in config:
		raise ValueError("parameters for mode '%s' missing from config file" % mode)
	config_mode = config[mode]

	# Generate shot data
	if "shot" in config_mode:
		shot = config_mode["shot"]
		aspect_ratio = config["aspect_ratio"]
		flux_ref = config_mode["flux"]["ref"]
		generateShotData(shot, aspect_ratio, flux_ref)

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
	t, I, flux, P_ohm, eta0, V_phi, V_theta = calc(lmbda0, alpha, B_phi, B_theta, P_ohm, U_mag, config)

	# Plot results
	BP_core_flux = si.cumtrapz(V_phi, t, initial=0) - (0.72 + 0.13)
	mu0 = config["mu0"]
	aspect_ratio = config["aspect_ratio"]
	a = config["a"]
	R0 = aspect_ratio * a
	f = t * np.nan
	theta = t * np.nan
	nz = flux != 0
	f[nz] = mu0 * a**2 * I[nz,1] / (2 * R0 * flux[nz]) # B_phi / 2
	theta[nz] = mu0 * a * I[nz,0] / (2 * flux[nz]) # B_theta / 2
	ener_phi = 1e-3 * si.cumtrapz(V_phi * I[:,0], t, initial=0)
	abs_ener_phi = 1e-3 * si.cumtrapz(np.abs(V_phi * I[:,0]), t, initial=0)
	ener_theta = 1e-3 * si.cumtrapz(V_theta * I[:,1], t, initial=0)
	abs_ener_theta = 1e-3 * si.cumtrapz(np.abs(V_theta * I[:,1]), t, initial=0)
	tf = config["time"]["tf"]

	mode = config["mode"]
	if mode not in config:
		raise ValueError("parameters for mode '%s' missing from config file" % mode)
	config_mode = config[mode]
	P_ohm_over_I_phi_shot = np.interp(t, loadVec(config_mode["time"]), loadVec(config_mode["P_ohm_over_I_phi"]))
	f_shot = np.interp(t, loadVec(config_mode["time"]), loadVec(config_mode["f"]))
	theta_shot = np.interp(t, loadVec(config_mode["time"]), loadVec(config_mode["theta"]))
	config_phi = config_mode["toroidal"]
	I_phi_shot = np.interp(t, loadVec(config_phi["time"]), loadVec(config_phi["current"]))
	V_phi_shot = np.interp(t, loadVec(config_phi["time"]), loadVec(config_phi["voltage"]))
	config_theta = config_mode["poloidal"]
	I_theta_shot = np.interp(t, loadVec(config_theta["time"]), loadVec(config_theta["current"]))
	V_theta_shot = np.interp(t, loadVec(config_theta["time"]), loadVec(config_theta["voltage"]))
	config_flux = config_mode["flux"]
	flux_shot = np.interp(t, loadVec(config_flux["time"]), loadVec(config_flux["value"]))

	plt.rc("font", family="serif")

	plt.subplot(4, 3, 1)
	plotVphiAndBPCoreFlux(t, V_phi, BP_core_flux, V_phi_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 2)
	plotVtheta(t, V_theta, V_theta_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 3)
	plotFlux(t, flux, flux_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 4)
	plotIphi(t, I[:,0], BP_core_flux, I_phi_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 5)
	plotItheta(t, I[:,1], I_theta_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 6)
	plotPohmOverIphi(t, P_ohm, I[:,0], P_ohm_over_I_phi_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 7)
	plotVphiTimesIphi(t, V_phi, I[:,0])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 8)
	plotVthetaTimesItheta(t, V_theta, I[:,1])
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 9)
	plotThetaAndF(t, theta, f, theta_shot, f_shot)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 10)
	plotEnergyPhi(t, ener_phi, abs_ener_phi)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 11)
	plotEnergyTheta(t, ener_theta, abs_ener_theta)
	plt.xlim(xmax=tf)

	plt.subplot(4, 3, 12)
	plotEta0(t, eta0)
	plt.xlim(xmax=tf)

	plt.show()


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


if __name__ == "__main__":
	run()
