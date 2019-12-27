'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/

    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

from __future__ import print_function
from utils import convert_files
import scipy.io as sio
import os
import numpy as np
import argparse

def read_binvoxfile_as_3d_array(*args):
	# This file requires some external library function that reads binvox files
	# and returns the loaded shapes as a numpy 3D array.
	raise NotImplementedError


def convert_binvox(filename):
	""" Converts a single binvox file to Matlab. 
		The resulting mat file stores one variable 'voxel'.
	"""
	print('Converting %s ... ' % filename, end='')
	try:
		with open(filename, 'rb') as f:
			md = read_binvoxfile_as_3d_array(f)
			v = np.array(md.data, dtype='uint8')
			sio.savemat(filename[:-7]+'.vox.mat', {'voxel':v[::-1,::-1,::-1].copy()}, do_compression=True)
			pass
		pass
	except:
		print('failed.')
		return
	print('done.')
	pass


if __name__ == '__main__':

	parser = argparse.ArgumentParser('Converts .binvox files to .mat files.')
	parser.add_argument('directory', type=str, help='Directory with binvox files.', default='.')
	parser.add_argument('-r', '--recursive', action='store_true', help='Recursively traverses the directory and converts all binvox files.')

	args = parser.parse_args()

	convert_files(args.directory, '.binvox', convert_binvox, args.recursive)
	pass
