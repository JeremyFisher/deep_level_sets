'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import log, floor

from ipdb import set_trace


class ResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, num_penultimate_channels, \
        input_resolution, output_resolution, num_initial_channels=16, num_inner_channels=64, \
        num_downsampling=3, num_blocks=6, bottleneck_dim=1024):

        assert num_blocks >= 0

        super(ResNet, self).__init__()

        relu = nn.ReLU(True)
       
        model = [nn.BatchNorm2d(num_input_channels, True)] # TODO

        # additional down and upsampling blocks to account for difference in input/output resolution
        num_additional_down   = int(log(input_resolution / output_resolution,2)) if output_resolution <= input_resolution else 0
        num_additional_up     = int(log(output_resolution / input_resolution,2)) if output_resolution >  input_resolution else 0

        # number of channels to add during downsampling
        num_channels_down     = int(floor(float(num_inner_channels - num_initial_channels)/(num_downsampling+num_additional_down)))

        # adjust number of initial channels
        num_initial_channels += (num_inner_channels-num_initial_channels) % num_channels_down

        # initial feature block
        model += [nn.ReflectionPad2d(1),
            nn.Conv2d(num_input_channels, num_initial_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_initial_channels),
            relu]
        model += [nn.ReflectionPad2d(1),
            nn.Conv2d(num_initial_channels, num_initial_channels, kernel_size=3, padding=0)]

        # downsampling
        for i in range(num_downsampling+num_additional_down):                        
            model += [ResDownBlock(num_initial_channels, num_channels_down)]
            model += [ResSameBlock(num_initial_channels+num_channels_down)]
            num_initial_channels += num_channels_down
            pass

        # inner blocks at constant resolution
        for i in range(num_blocks):
            model += [ResSameBlock(num_initial_channels)]
            pass
    

        model_encoder = model[:20].copy() # model_encoder is bsx512x1x1
        self.model_enc = nn.Sequential(*model_encoder)

        # BOTTLENECK
        tmp_init_depth = 8 # nr of channels 
        tmp_n_final = 6 # we will first resize to 6x6x6 with 8 channels

        enc_last_dim = num_initial_channels
        model_bottleneck = [nn.Linear(enc_last_dim, bottleneck_dim)]
        model_bottleneck += [nn.Linear(bottleneck_dim, tmp_init_depth*tmp_n_final**3)] 
        self.model_bottleneck = nn.Sequential(*model_bottleneck)
        # DECODER
        model_decoder = []

        kernel_size_list = [3,3, 3, 5, 5, 7, 7]
        channels_list =    [8,8, 16, 32, 16, 8, 4, 4] 
        
        for deconv_iter in range(len(kernel_size_list)):
            model_decoder += [nn.ConvTranspose3d(channels_list[deconv_iter],
                channels_list[deconv_iter+1], kernel_size_list[deconv_iter],
                stride=1, padding=0, output_padding=0, groups=1, bias=True,
                dilation=1),
                nn.BatchNorm3d(channels_list[deconv_iter+1], eps=1e-05, 
                    momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True)]

        model_decoder += [nn.ConvTranspose3d(channels_list[-1],
                1, 1,
                stride=1, padding=0, output_padding=0, groups=1, bias=True,
                dilation=1)]
        self.model_decoder = nn.Sequential(*model_decoder)
        return 
            

        
    def forward(self, input):
        x = self.model_enc(input) # x should be 32,512,1,1
        x = x.view(-1, x.shape[1])        
        # bottleneck FC 1024, FC 6*6*6*8
        x = self.model_bottleneck(x)
        tmp_init_depth = 8 # nr of channels 
        tmp_n_final = 6 # we will first resize to 6x6x6 with 8 channels
        x = x.view((-1,)+(tmp_init_depth,) + (tmp_n_final,)*3 )
        # decoder 
        x = self.model_decoder(x)
        x = x[:,0,:,:,:] # remove the '1' in bsxcxWxDxH, c=1
        return x
    pass


class ResSameBlock(nn.Module):
    """ ResNet block for constant resolution.
    """
    
    def __init__(self, dim):
        super(ResSameBlock, self).__init__()

        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),            
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)])

    def forward(self, x):
        return x + self.model(x)
    pass    


class ResUpBlock(nn.Module):
    """ ResNet block for upsampling.
    """

    def __init__(self, dim, num_up):
        super(ResUpBlock, self).__init__()
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True),\
            nn.ReLU(False),
            nn.ConvTranspose2d(dim, -num_up+dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(-num_up+dim, True),
            nn.ReLU(True),
            nn.Conv2d(-num_up+dim, -num_up+dim, kernel_size=3, padding=1)])

        self.project = nn.Conv2d(dim,dim-num_up,kernel_size=1)
        pass

    def forward(self, x):        
        # xu = F.upsample(x,scale_factor=2,mode='nearest')
        xu = F.interpolate(x, scale_factor=2, mode='nearest')
        bs,_,h,w = xu.size()
        return self.project(xu) + self.model(x)
    pass


class ResDownBlock(nn.Module):
    """ ResNet block for downsampling.
    """
    
    def __init__(self, dim, num_down):
        super(ResDownBlock, self).__init__()
        self.num_down = num_down
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(False),
            nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_down+dim, True),
            nn.ReLU(True),
            nn.Conv2d(num_down+dim, num_down+dim, kernel_size=3, padding=1)])
        pass

    def forward(self, x):
        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()
        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + self.model(x)
    pass
