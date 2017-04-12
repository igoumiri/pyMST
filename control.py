# -*- coding: utf-8 -*-

# MST Controller
# Imène Goumiri
# January 4, 2017
# MIT License

import numpy.matlib as np
import scipy.integrate as si

class LQG:
	def __init__(self, config):
		self.A = np.mat(config["A"])
		self.B = np.mat(config["B"])
		self.C = np.mat(config["C"])
		self.D = np.mat(config["D"])
		self.F = np.mat(config["F"])
		self.K = np.mat(config["K"])
		self.Ki = np.mat(config["Ki"])
		self.L = np.mat(config["L"])
		self.ud = np.mat(config["ud"])
		self.xd = np.mat(config["xd"])
		self.r0 = np.mat(config["r0"])
		self.u0 = np.mat(config["u0"])
		self.dt = config["dt"]

		self.AmLC = self.A - self.L * self.C
		self.BmLD = self.B - self.L * self.D

		q, n = self.C.shape
		self.xh = np.zeros((n, 1))
		self.xi = np.zeros((q, 1))

	def observe(self, u, y, xh, xi, i):
		u = np.mat(u).T - self.u0
		y = np.mat(y).T - self.r0
		self.xh = self.AmLC * self.xh + self.BmLD * u + self.L * y
		# self.xi += self.dt * (y - self.r)
		xh[i,:] = self.xh.flat
		xi[i,:] = self.xi.flat

	def control(self, r, y, xh, xi, vh, i):
		y = np.mat(y).T - self.r0
		u = self.F * (r - self.r0) - self.K * self.xh - self.Ki * self.xi

		vh[i,:] = (u + self.u0).flat
		u = np.clip(u + self.u0, [[-250], [-40]], [[250], [40]]) - self.u0

		self.xh = self.AmLC * self.xh + self.BmLD * u + self.L * y
		self.xi += self.dt * (y - (r - self.r0))
		xh[i,:] = self.xh.flat
		xi[i,:] = self.xi.flat
		return (u + self.u0).T
