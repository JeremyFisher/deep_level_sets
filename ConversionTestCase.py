'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

import unittest

import numpy as np
import torch

# for loading voxel grid from MATLAB file
import scipy.io as sio

import voxel2layer as v2lnp
import voxel2layer_torch as v2lt

class ConversionTestCase(unittest.TestCase):

	def setUp(self):
		# load sample shape as voxel representation
		d = sio.loadmat('./data/chair.mat')
		self.voxelnp = d['voxel']
		self.voxelt  = torch.from_numpy(self.voxelnp)
		self.shlnp   = d['shl']
		self.shlt    = torch.from_numpy(self.shlnp.astype(np.int32)).to(torch.int16)
		pass

	def tearDown(self):
		pass

	def test_encode_numpy(self):
		""" Tests converting from voxel to shape layer representation using numpy. """
		shl = v2lnp.encode_shape(self.voxelnp)
		self.assertTrue(np.all(shl == self.shlnp))
		pass

	def test_decode_numpy(self):
		""" Tests converting from shape layer to voxel representation using numpy. """
		voxel = v2lnp.decode_shape(self.shlnp)
		# sio.savemat('decode_numpy.mat', {'voxel':voxel, 'gt':self.voxelnp})
		self.assertTrue(np.all(voxel == self.voxelnp))
		pass

	def test_encode_torch(self):
		""" Tests converting from voxel to shape layer representation using torch. """
		shl = v2lt.encode_shape(self.voxelt)
		# sio.savemat('encode_torch.mat', {'shl':shl.numpy(), 'gt':self.shlt.numpy()})
		self.assertTrue((shl == self.shlt).all())
		pass

	def test_decode_torch(self):
		""" Tests converting from shape layer to voxel representation using torch. """
		voxel = v2lt.decode_shape(self.shlt)
		# sio.savemat('decode_torch.mat', {'voxel':voxel.numpy(), 'gt':self.voxelt.numpy()})
		self.assertTrue((voxel == self.voxelt).all())
		pass

	def test_roundtrip_numpy(self):
		""" Tests converting from voxel to shape layer and back using numpy."""
		voxel = v2lnp.decode_shape(v2lnp.encode_shape(self.voxelnp, 2))
		self.assertTrue(np.all(voxel == self.voxelnp))
		pass

	def test_roundtrip_torch(self):
		""" Tests converting from voxel to shape layer and back using torch."""
		voxel = v2lt.decode_shape(v2lt.encode_shape(self.voxelt, 2))
		self.assertTrue((voxel == self.voxelt).all())
		pass	

	def test_shapelayer_conversion(self):
		""" Tests modifying the shape layer representation for better alignment.
		"""
		shlx = v2lt.shl2shlx(self.shlt.clone().permute(2,0,1).reshape(1,6,128,128))
		shl  = v2lt.shlx2shl(shlx).reshape(6,128,128).permute(1,2,0)
		self.assertTrue((shl == self.shlt).all())
		pass

if __name__ == '__main__':
	unittest.main()
