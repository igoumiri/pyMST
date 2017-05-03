#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python implementation of John Sarff's MST code
# Imène Goumiri
# October 11, 2016
# MIT License

import sys
import toml

import parameters
import model
import controller
import reference
import output
from utils import time


def run(config):
	"""Run a fixed length simulation based on the inputs provided in the config."""

	# Init lookup table of parameters
	params = parameters.LookupTable(config)

	# Init MST model
	mst = model.MST(params, config)

	# Init LQG controller
	ctrl = controller.LQG(config["control"])

	# Init reference signal
	ref = reference.Translator(params, config["control"])

	# Init output plotter (will plot on exit)
	with output.Plotter(config) as out:
		# Main loop
		y = [mst.lmbda0, mst.flux]
		for t in time(config["time"])[0][1:]:
			r = ref(t)
			u = ctrl.control(r, y)
			y = mst.step(t, u)
			out.save(t, r, u, y, mst, ctrl)


if __name__ == "__main__":
	# Parse config file
	if len(sys.argv) != 2:
		print("Usage: {0} <config.toml>".format(sys.argv[0]))
		sys.exit(1)
	with open(sys.argv[1]) as config_file:
		config = toml.loads(config_file.read())

	# Run fixed length simulation
	try:
		run(config)
	except UserWarning as warning:
		print(warning)
	except StopIteration as info:
		print(info)
