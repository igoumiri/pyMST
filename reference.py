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

class Translator:
	def __init__(self, params, config):
		F1 = config["F1"]
		F2 = config["F2"]
		Ip1 = config["Ip1"]
		Ip2 = config["Ip2"]

		self.t1 = config["t1"]
		self.t2 = config["t2"]
		self.r1 = [[params.lmbda0(F1, Ip1)], [params.flux(F1, Ip1)]]
		self.r2 = [[params.lmbda0(F2, Ip2)], [params.flux(F2, Ip2)]]

	def __call__(self, t):
		if t < self.t1:
			return None
		elif t < self.t2:
			return self.r1
		else:
			return self.r2