# -*- coding: utf-8 -*-

# Output generation (plots and files)
# Imène Goumiri
# April 11, 2017
# MIT License

import numpy as np
import scipy.integrate as si
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from utils import time, loadVec


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


class Plotter:
	def __init__(self, config):
		# Parameters
		self.mu0 = config["mu0"]
		aspect_ratio = config["aspect_ratio"]
		self.a = config["a"]
		self.R0 = aspect_ratio * self.a

		# Time discretization
		self.t, num_t = time(config["time"])
		self.tf = config["time"]["tf"]

		# Mode-dependent parameters
		mode = config["mode"]
		if mode not in config:
			raise ValueError("parameters for mode '%s' missing from config file" % mode)
		config_mode = config[mode]
		self.P_ohm_over_I_phi_shot = np.interp(self.t, loadVec(config_mode["time"]), loadVec(config_mode["P_ohm_over_I_phi"]))
		self.f_shot = np.interp(self.t, loadVec(config_mode["time"]), loadVec(config_mode["f"]))
		self.theta_shot = np.interp(self.t, loadVec(config_mode["time"]), loadVec(config_mode["theta"]))
		config_phi = config_mode["toroidal"]
		self.I_phi_shot = np.interp(self.t, loadVec(config_phi["time"]), loadVec(config_phi["current"]))
		self.V_phi_shot = np.interp(self.t, loadVec(config_phi["time"]), loadVec(config_phi["voltage"]))
		config_theta = config_mode["poloidal"]
		self.I_theta_shot = np.interp(self.t, loadVec(config_theta["time"]), loadVec(config_theta["current"]))
		self.V_theta_shot = np.interp(self.t, loadVec(config_theta["time"]), loadVec(config_theta["voltage"]))
		config_flux = config_mode["flux"]
		self.flux_shot = np.interp(self.t, loadVec(config_flux["time"]), loadVec(config_flux["value"]))

		# Init arrays, hopefully with their final size, otherwise expandable
		self.flux = np.zeros(num_t)
		self.lmbda0 = np.zeros(num_t)
		self.I_phi = np.zeros(num_t)
		self.I_theta = np.zeros(num_t)
		self.V_phi = np.zeros(num_t)
		self.V_theta = np.zeros(num_t)
		self.P_ohm = np.zeros(num_t)
		self.eta0 = np.zeros(num_t)
		self.I_phi_primary = np.zeros(num_t)
		self.I_theta_primary = np.zeros(num_t)

		# Current index
		self.index = 1

		# Convert voltages to primary current
		config_primary = config["primary"]
		self.coeff_phi = config_primary["coeff_phi"]
		self.coeff_theta = config_primary["coeff_theta"]
		self.R_phi = config_primary["R_phi"]
		self.R_theta = config_primary["R_theta"]
		self.L_phi = config_primary["L_phi"]
		self.L_theta = config_primary["L_theta"]
		self.solver = si.ode(self.primaryCurrentDot)
		self.solver.set_integrator("dopri5")
		self.solver.set_initial_value([self.V_phi_shot[0], self.V_theta_shot[0]], self.t[0])


	def primaryCurrentDot(self, t, x, u):
		I_phi, I_theta = x
		if u is not None:
			V_phi, V_theta = u
		else:
			V_phi = self.V_phi_shot[self.index]
			V_theta = self.V_theta_shot[self.index]
		I_phi_dot = (self.coeff_phi * V_phi - self.R_phi * I_phi) / self.L_phi
		I_theta_dot = (self.coeff_theta * V_theta - self.R_theta * I_theta) / self.L_theta
		return [I_phi_dot, I_theta_dot]


	def __enter__(self):
		return self


	def __exit__(self, type, value, traceback):
		# Do not plot if there is an exception
		if isinstance(value, Exception) and not isinstance(value, StopIteration):
			return False

		# Fill remaining space with nans
		self.flux[self.index:] = np.nan
		self.lmbda0[self.index:] = np.nan
		self.I_phi[self.index:] = np.nan
		self.I_theta[self.index:] = np.nan
		self.V_phi[self.index:] = np.nan
		self.V_theta[self.index:] = np.nan
		self.P_ohm[self.index:] = np.nan
		self.eta0[self.index:] = np.nan
		self.I_phi_primary[self.index:] = np.nan
		self.I_theta_primary[self.index:] = np.nan

		# Show all plots
		self.plotAll()
		plt.show()


	def save(self, t, r, u, y, mst, ctrl):
		i = self.index

		# Save to arrays
		self.t[i] = t
		self.flux[i] = mst.flux
		self.lmbda0[i] = mst.lmbda0
		self.I_phi[i] = mst.I_phi
		self.I_theta[i] = mst.I_theta
		self.V_phi[i] = mst.V_phi
		self.V_theta[i] = mst.V_theta
		self.P_ohm[i] = mst.P_ohm
		self.eta0[i] = mst.eta0

		# Convert voltages to primary current
		self.solver.set_f_params(u)
		self.solver.integrate(t)
		self.I_phi_primary[i] = self.solver.y[0]
		self.I_theta_primary[i] = self.solver.y[1]

		# Increment current index
		self.index += 1


	def plotAll(self):
		# Compute derived quantities
		BP_core_flux = si.cumtrapz(self.V_phi, self.t, initial=0) - (0.72 + 0.13)
		f = self.t * np.nan
		theta = self.t * np.nan
		nz = self.flux != 0
		f[nz] = self.mu0 * self.a**2 * self.I_theta[nz] / (2 * self.R0 * self.flux[nz]) # B_phi / 2
		theta[nz] = self.mu0 * self.a * self.I_phi[nz] / (2 * self.flux[nz]) # B_theta / 2
		ener_phi = 1e-3 * si.cumtrapz(self.V_phi * self.I_phi, self.t, initial=0)
		abs_ener_phi = 1e-3 * si.cumtrapz(np.abs(self.V_phi * self.I_phi), self.t, initial=0)
		ener_theta = 1e-3 * si.cumtrapz(self.V_theta * self.I_theta, self.t, initial=0)
		abs_ener_theta = 1e-3 * si.cumtrapz(np.abs(self.V_theta * self.I_theta), self.t, initial=0)

		# Plot
		plt.rc("font", family="serif")

		plt.figure()
		plt.subplot(1, 2, 1)
		plt.plot(self.t, 1e-3 * self.I_phi_primary)
		plt.xlabel("Time (s)")
		plt.ylabel("Current (kA)")
		plt.title(r"$I_\phi^primary$")
		plt.grid()
		plt.subplot(1, 2, 2)
		plt.plot(self.t, 1e-6 * self.I_theta_primary)
		plt.xlabel("Time (s)")
		plt.ylabel("Current (MA)")
		plt.title(r"$I_\theta^primary$")
		plt.grid()

		plt.figure()
		plt.subplot(4, 3, 1)
		plotVphiAndBPCoreFlux(self.t, self.V_phi, BP_core_flux, self.V_phi_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 2)
		plotVtheta(self.t, self.V_theta, self.V_theta_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 3)
		plotFlux(self.t, self.flux, self.flux_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 4)
		plotIphi(self.t, self.I_phi, BP_core_flux, self.I_phi_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 5)
		plotItheta(self.t, self.I_theta, self.I_theta_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 6)
		plotPohmOverIphi(self.t, self.P_ohm, self.I_phi, self.P_ohm_over_I_phi_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 7)
		plotVphiTimesIphi(self.t, self.V_phi, self.I_phi)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 8)
		plotVthetaTimesItheta(self.t, self.V_theta, self.I_theta)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 9)
		plotThetaAndF(self.t, theta, f, self.theta_shot, self.f_shot)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 10)
		plotEnergyPhi(self.t, ener_phi, abs_ener_phi)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 11)
		plotEnergyTheta(self.t, ener_theta, abs_ener_theta)
		plt.xlim(xmax=self.tf)

		plt.subplot(4, 3, 12)
		plotEta0(self.t, self.eta0)
		plt.xlim(xmax=self.tf)

