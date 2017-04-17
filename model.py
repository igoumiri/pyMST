# -*- coding: utf-8 -*-

# Dynamical model of MST based on John Sarff's IDL code
# Imène Goumiri
# April 11, 2017
# MIT License

import numpy as np
import scipy.integrate as si
import scipy.interpolate as sp
from utils import time, loadVec


def Te0(I_phi, density, a):
	return max(2.8 * (2 * a)**0.83 * np.sqrt(I_phi / (density * np.pi * a**2) * 1e14) * (I_phi * 1e-3)**0.67, 20)


class MST:
	def __init__(self, params, config):
		# Get splines for magnetic fields, ohmic power and magnetic energy
		self.spl_B_phi = params.B_phi
		self.spl_B_theta = params.B_theta
		self.spl_P_ohm = params.P_ohm
		self.spl_U_mag = params.U_mag

		# Parameters
		self.mu0 = config["mu0"]
		aspect_ratio = config["aspect_ratio"]
		self.a = config["a"]
		self.R0 = aspect_ratio * self.a
		R_star = 0.969 * self.R0 # FIXME: use config file
		zneo = config["zneo"]
		zeff = config["zeff"]
		self.zohm = (0.4 + 0.6 * zeff) * zneo

		self.alpha = 4.0 # FIXME: using hardcoded fixed alpha for now

		# Time discretization
		t, num_t = time(config["time"])
		self.tmin = t[num_t/10]
		self.dt = config["time"]["dt"]

		# Parameters specific to simulation mode
		self.mode = config["mode"]
		if self.mode not in config:
			raise ValueError("parameters for mode '%s' missing from config file" % self.mode)
		config_mode = config[self.mode]

		# if "alpha" in config_mode:
		# 	config_alpha = config_mode["alpha"]
		# 	alpha = sp.interp1d(loadVec(config_alpha["time"]), loadVec(config_alpha["value"]))

		if "flux" in config_mode:
			config_flux = config_mode["flux"]
			flux0 = config_flux["ref"] * config_flux["multiplier"]
		else:
			flux0 = config_mode["flux_ref"] * config_mode["flux_multiplier"]

		if "density" in config_mode:
			config_density = config_mode["density"]
			self.density = sp.interp1d(loadVec(config_density["time"]), loadVec(config_density["value"]))
		else:
			self.density = lambda t: config_mode["density_ref"] * config_mode["density_multiplier"]

		config_phi = config_mode["toroidal"]
		V_phi_wave = np.interp(t, loadVec(config_phi["time"]), loadVec(config_phi["voltage"]))
		self.V_phi_DC = config_phi["DC_voltage"]

		config_theta = config_mode["poloidal"]
		V_theta_wave = np.interp(t, loadVec(config_theta["time"]), loadVec(config_theta["voltage"]))

		config_anom = config_mode["anomalous"]
		self.zanom_wave = sp.interp1d(config_anom["time"], config_anom["voltage"])

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

		self.V_theta_wave = sp.interp1d(t, V_theta_wave)
		self.V_phi_wave = sp.interp1d(t, V_phi_wave)

		I_phi_0 = config["initial"]["I_phi"]

		# Initialize state variables
		self.flux = flux0
		self.lmbda0 = params.lmbda0(self.mu0 * self.a * I_phi_0 / flux0)
		self.I_phi = I_phi_0
		self.I_theta = self.spl_B_phi(self.lmbda0, self.alpha) * flux0 * self.R0 / (self.a**2 * self.mu0)
		self.V_phi = self.V_phi_wave(t[0])
		self.V_theta = self.V_theta_wave(t[0])
		self.P_ohm = 0.0
		self.eta0 = 0.0

		# Initialize ODE solver
		self.solver = si.ode(self.lfDot)
		self.solver.set_integrator("dopri5")
		self.solver.set_initial_value([self.lmbda0, self.flux], t[0])


	def step(self, t, u):
		"""Advance simulation to time t using constant input u."""

		if u is not None:
			self.V_phi = u[0]
			self.V_theta = u[1]
		else:
			self.V_phi = self.V_phi_wave(t)
			self.V_theta = self.V_theta_wave(t)

		tm = t - self.dt
		self.eta0 = self.zanom_wave(tm) * 1.6 * 7.75e-4 * self.zohm / Te0(self.I_phi, self.density(tm), self.a)**1.5
		if self.mode is "PPCD_550KA":
			self.eta0 /= self.PPCD_Te_mult(tm)**1.5

		self.lmbda0, self.flux = self.solver.integrate(t)

		B_phi = self.spl_B_phi(self.lmbda0, self.alpha)
		B_theta = self.spl_B_theta(self.lmbda0, self.alpha)
		self.I_phi = B_theta * self.flux / (self.a * self.mu0)
		self.I_theta = B_phi * self.flux * self.R0 / (self.a**2 * self.mu0)
		self.P_ohm = self.spl_P_ohm(self.lmbda0, self.alpha) * self.flux**2 * self.eta0 * self.R0 / (self.a**4 * self.mu0**2)

		if not self.solver.successful():
			raise UserWarning("Warning: Integration failed at t={0}".format(t))

		if t > self.tmin and self.I_phi < 1e4:
			raise StopIteration("Info: Integration stopped at t={0}".format(t))

		return self.lmbda0, self.flux


	def lmbdaDot(self, lmbda0, flux):
		# Evaluate splines
		B_phi = self.spl_B_phi(lmbda0, self.alpha)
		B_theta = self.spl_B_theta(lmbda0, self.alpha)
		P_ohm = self.spl_P_ohm(lmbda0, self.alpha)
		U_mag = self.spl_U_mag(lmbda0, self.alpha)
		U_mag_dot = self.spl_U_mag(lmbda0, self.alpha, dx=1)

		# Evaluate currents and dissipation power
		I_phi = B_theta * flux / (self.a * self.mu0)
		I_theta = B_phi * flux * self.R0 / (self.a**2 * self.mu0)
		P_diss = P_ohm * flux**2 * self.eta0 * self.R0 / (self.a**4 * self.mu0**2)

		# Compute the time derivative of lambda
		num = ((self.V_phi_DC + self.V_phi) * I_phi - self.V_theta * I_theta - P_diss)
		num *= 2 * self.mu0 * self.a**2 / self.R0
		num += 2 * self.V_theta * flux * U_mag
		den = flux**2 * U_mag_dot
		return num / den


	def lfDot(self, t, x):
		lmbda0, flux = x
		lmbda_dot = self.lmbdaDot(lmbda0, flux)
		flux_dot = -self.V_theta
		return [lmbda_dot, flux_dot]