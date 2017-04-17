# -*- coding: utf-8 -*-

# Generates a reference signal for a controller
# Imène Goumiri
# April 11, 2017
# MIT License

class Basic:
	def __init__(self, params, config):
		self.t1 = config["t1"]
		self.t2 = config["t2"]
		self.r1 = config["r1"]
		self.r2 = config["r2"]

	def __call__(self, t):
		if t < self.t1:
			return None
		elif t < self.t2:
			return self.r1
		else:
			return self.r2

class Adhoc:
	def __init__(self, params, config):
		from adhoc import adhoc
		eq1 = adhoc.f_theta_to_eq(Theta=config["Theta1"], F=config["F1"], ip=config["Ip1"])
		eq2 = adhoc.f_theta_to_eq(Theta=config["Theta2"], F=config["F2"], ip=config["Ip2"])

		self.t1 = config["t1"]
		self.t2 = config["t2"]
		self.r1 = [[eq1['pars'][0][0]], [eq1['Phi']]]
		self.r2 = [[eq2['pars'][0][0]], [eq2['Phi']]]

	def __call__(self, t):
		if t < self.t1:
			return None
		elif t < self.t2:
			return self.r1
		else:
			return self.r2