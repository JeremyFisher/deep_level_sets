'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

from __future__ import print_function
import torch.utils.data as data
from PIL import Image, ImageOps
import os
import torch
import numpy as np
from torchvision import transforms
import scipy.io as sio

from binvox_rw import read_as_3d_array
from ipdb import set_trace
import skfmm

class RandomColorFlip(object):    
    def __call__(self, img):        
        c = np.random.choice(3,3,np.random.random() < 0.5)            
        return img[c,:,:]

class DatasetLoader(data.Dataset):       
    def __init__(self, samples, side, num_comp=1, input_transform=None, no_images=False, no_shapes=False):

        self.input_transform  = input_transform
        self.num_comp = num_comp      
        self.samples  = samples
        self.side = side
        self.no_images = no_images
        self.no_shapes = no_shapes
        pass

    def __getitem__(self, index):
        
        imagepath = self.samples[index][0]
        shapepath = self.samples[index][1]
        example_id = self.samples[index][2]
        
        flipped = False#random.random() > 0.5 if self.flip else False
        
        if self.no_images:
            imgs = None
        else:
            imgs = self.input_transform(self._load_image(Image.open(imagepath),flipped))

        if self.no_shapes:
            shape = None
        else:
            if shapepath.endswith('.shl.mat'): # shape layer
                shape = self._load_shl(shapepath)
            elif shapepath.endswith('.vox.mat'): # voxels in mat format
                shape = self._load_vox2(shapepath)
            elif shapepath.endswith('.binvox'): # voxels in binvox format
                shape = self._load_binvox(shapepath)
            elif ('sdf_' in shapepath)  and (shapepath.endswith('.npy')):
                shape = self._load_sdf(shapepath)
            elif ('vox_' in shapepath)  and (shapepath.endswith('.npy')):
                shape = self._load_vox(shapepath)           
            elif 'dist' in shapepath: # chamfer
                shape = self._load_dist(shapepath)
            else:
                assert False, ('Could not determine shape representation from file name (%s has neither ".shl.mat" nor ".vox.mat").' % shapepath)

        if self.no_images:
            if self.no_shapes:
                return
            else:
                return shape
        else:
            if self.no_shapes:
                return imgs
            else:
                return imgs, shape, example_id


    def __len__(self):
        return len(self.samples)

    def _load_dist(self, path):
        dist = np.load(path)
        assert dist.shape == (self.side**3,)
        return torch.from_numpy(dist.astype('float32'))        

    def _load_vox(self, path):
        vox = np.load(path)
        if len(vox.shape) != 3: # sdf is flattened
            vox = vox.squeeze()
            assert len(vox.shape) == 1
            dim = np.cbrt(vox.shape[0])
            assert dim == int(dim)
            vox = vox.reshape((int(dim),)*3).astype('int')
        return torch.from_numpy(vox.astype('int'))

    def _load_sdf(self, path):
        sdf = np.load(path)
        if len(sdf.shape) != 3: # sdf is flattened
            sdf = sdf.squeeze()
            assert len(sdf.shape) == 1
            dim = np.cbrt(sdf.shape[0])
            assert dim == int(dim)
            sdf = sdf.reshape((int(dim),)*3).astype('float32')
        return torch.from_numpy(sdf.astype('float32'))


    def _load_vox22(self, path):
        d = sio.loadmat(path)
        return torch.from_numpy(d['voxel'])

    def _load_binvox(self, path):
        ''' loads voxels saved in binvox format. see also _load_vox '''
        if not os.path.exists(path):
            raise Exception('path does not exist: '+ path)
        with open(path, 'rb') as fin:
            voxels = read_as_3d_array(fin)
        return torch.from_numpy(voxels.data.astype('uint8'))               

    def _load_shl(self, path):
        d = sio.loadmat(path)
        return torch.from_numpy(np.array(d['shapelayer'], dtype=np.int32)[:,:,:6*self.num_comp]).permute(2,0,1).contiguous().float() 

    def _load_image(self, temp, flipped=False):   
        if temp.mode == 'RGBA':
            alpha = temp.split()[-1]
            bg = Image.new("RGBA", temp.size, (128,128,128) + (255,))
            bg.paste(temp, mask=alpha)
            im = bg.convert('RGB').copy()
            bg.close()
            temp.close()
        else:
            im = temp.copy()
            temp.close()
        return (im.transpose(Image.FLIP_LEFT_RIGHT) if flipped else im)
