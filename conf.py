# import the necessary packages
from __future__ import print_function
from scipy import io
import numpy as np
import argparse
import glob

from json_minify import json_minify
import json

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# USAGE
# python explore_dims.py --conf conf/cars.json


class Conf:
	def __init__(self, confPath):
		# load and store the configuration and update the object's dictionary
		conf = json.loads(json_minify(open(confPath).read()))
		self.__dict__.update(conf)

	def __getitem__(self, k):
		# return the value associated with the supplied key
		return self.__dict__.get(k, None)


if __name__ == "__main__":
    # load the configuration file and initialize the list of widths and heights
    conf = Conf(args["conf"])
    print(conf["image_dataset"])