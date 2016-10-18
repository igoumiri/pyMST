#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python implementation of John Sarff's MST code
# Imène Goumiri
# October 11, 2016
# MIT License

import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import toml
import sys

def lmbda(r, alpha, lmbda0):
	return lmbda0 * (1 - r**alpha)

def gradP(r, beta0, p1, p2): # P(r) = (1 - r**p1)**(p2 - 1)
	return -beta0/2 * p1 * p2 * r**(p1 - 1) * (1.0 - r**p1)**(p2 - 1)

def gradB(r, B, beta0, p1, p2, alpha, lmbda0):
	B_phi, B_theta = B # toroidal, poloidal
	gradPnorm = gradP(r, beta0, p1, p2) / (B_phi**2 + B_theta**2)
	lmbda1 = lmbda(r, alpha, lmbda0)
	gradB_phi = (-lmbda1 * B_theta) - gradPnorm * B_phi
	gradB_theta = (lmbda1 * B_phi) - B_theta * (1 + gradPnorm)
	if r < 1e-20:
		gradB_theta = lmbda1 * B_phi / 2
	return [gradB_phi, gradB_theta]

def run():
	# Parse config file
	if len(sys.argv) != 2:
		print("Usage: pyMST.py <config.toml>")
		sys.exit(1)
	with open(sys.argv[1]) as config_file:
		config = toml.loads(config_file.read())

	# Initalize variables
	N = config["N"]
	r = np.linspace(0, 1, num=N+1)
	B = np.zeros((N+1, 2)) # Magnetic field
	p1, p2 = config["p1"], config["p2"]
	Pr = (1 - r**p1)**p2
	Pavg = 2 * si.simps(r * Pr, r)
	beta_theta = config["beta_theta"]

	# Loop to derive profiles
	for i in xrange(config["m"]):
		beta0 = beta_theta * B[N,1]**2 / Pavg
		B[:,0] = 1.0
		B[:,1] = 0.0

		# Solve ODE
		solver = si.ode(gradB)
		solver.set_integrator("dopri5")
		solver.set_initial_value(B[0,:], r[0])
		solver.set_f_params(beta0, p1, p2, config["alpha"], config["lambda_0"])
		for i in xrange(1, N+1):
			if not solver.successful():
				print("Failed :(")
				break
			solver.integrate(r[i])
			B[i,:] = solver.y

	# Plot results
	plt.plot(r, B)
	plt.xlabel("r")
	plt.ylabel("magnetic field")
	plt.legend(["B_phi", r"B_theta"], loc="center left")
	plt.show()

if __name__ == "__main__":
	run()
