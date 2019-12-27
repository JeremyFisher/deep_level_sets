'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.io as sio
import functools
import PIL
import logging
import time
#from utils import *
import math
import sys, os

from matr_iccv.ResNet import *
from matr_iccv.DatasetLoader import *
from matr_iccv.DatasetCollector import *

from matr_iccv.grad_approx_fun.finite_diff_torch_faster import get_grad3d_batch_faster_torch,\
    get_norm3d_batch_torch


from ipdb import set_trace

from matr_iccv.vis_objects import vis_view, vis_mesh, vis_sdf, vis_voxels, show, merge_mesh
import trimesh


def vis_slice(voxels, slice_nr=64, axis='y'):
    from matplotlib import pyplot as plt
    if axis == 'x':
        tmp_slice = voxels[slice_nr,:,:]    
    elif axis == 'y':
        tmp_slice = voxels[:,slice_nr,:]    
    elif axis == 'z':
        tmp_slice = voxels[:,:,slice_nr] 
    plt.imshow(tmp_slice);plt.show()



def dirac_delta_torch(phi_pred, eps=0.1):
    small_eps = 10**(-6) # just in case
    mask = (torch.abs(phi_pred) <= eps)
    dirac_delta = (1/(2*eps)) * (1+torch.cos(np.pi * phi_pred/eps))
    if phi_pred.dtype == torch.float32:
        dirac_delta = dirac_delta * mask.float()            
    elif phi_pred.dtype == torch.float64:
        dirac_delta = dirac_delta * mask.double()
    else:
        print('strange dtype in delta')
        exit()
    return dirac_delta

def my_loss(pred, target, loss_type, p_norm=2, eps=0.1, alpha_sdf=0.1):
    if loss_type == 'cross':
        loss = F.binary_cross_entropy(torch.sigmoid(pred), target.float())        
    elif loss_type == 'l2':
        pred.type = torch.float64
        loss = F.mse_loss(pred, target)
    elif loss_type == 'chamfer':      
        delta = dirac_delta_torch(pred.view(target.shape), eps=eps)
        chamf_loss = (delta*target) # BS X side**3
        chamf_loss = chamf_loss.sum(dim=1) # BS
        chamf_loss = chamf_loss**(1/p_norm) # BS, do abs?
        chamf_loss = chamf_loss.sum()
        phi_grad_t = get_grad3d_batch_faster_torch(pred, device)
        phi_grad_norm_t = get_norm3d_batch_torch(phi_grad_t, device)
        sdf_loss = (phi_grad_norm_t-1)**2
        sdf_loss_flatten = sdf_loss.view(sdf_loss.shape[0],-1)
        sdf_loss_final = sdf_loss_flatten.mean(dim=1) # BS
        sdf_loss_final = sdf_loss_final.sum()

        return (chamf_loss, sdf_loss_final)
    else:
        print('unkown loss type')
        exit()
    return loss
    
def pos_loss(pred, target, num_components=6):
    """ Modified L1-loss, which penalizes background pixels
        only if predictions are closer than 1 to being considered foreground.
    """
    fg_loss  = pred.new_zeros(1)
    bg_loss  = pred.new_zeros(1)
    fg_count = 0 # counter for normalization
    bg_count = 0 # counter for normalization

    for i in range(num_components):
        mask     = target[:,i,:,:].gt(0).float().detach()
        target_i = target[:,i,:,:]
        pred_i   = pred[:,i,:,:]
        # L1 between prediction and target only for foreground
        fg_loss  += torch.mean((torch.abs(pred_i-target_i)).mul(mask))
        fg_count += torch.mean(mask)
        # flip mask => background
        mask = 1-mask
        # L1 for background pixels > -1
        bg_loss  += torch.mean(((pred_i + 1)).clamp(min=0).mul(mask))
        bg_count += torch.mean(mask)
        pass

    return fg_loss / max(1, fg_count) + \
           bg_loss / max(1, bg_count)

def my_iou_voxel(pred, voxel):
    """ Computes intersection over union between two shapes.
        Returns iou with the length of a batch
    """
    pred = pred.detach()
    voxel = voxel.detach()
    
    bs,_,h,w = pred.size()
    
    inter = pred.mul(voxel).detach()
    union = pred.add(voxel).detach()
    union = union.sub_(inter) # probably to reduce 2 to 1.
    inter = inter.sum(3).sum(2).sum(1)
    union = union.sum(3).sum(2).sum(1)
    return inter.div(union), bs

def my_iou_sdf(pred, target):
    ''' 
    calculates iou between two sdfs.
    For simplicity, isosurface is extracted through > 0 
    '''
    pred_bin = (pred >= 0).type(torch.float64)
    target_bin = (target >= 0).type(torch.float64)

    pred_bin = pred_bin.detach()
    target_bin = target_bin.detach()
    
    bs,_,h,w = pred_bin.size()
    
    inter = pred_bin.mul(target_bin).detach()
    union = pred_bin.add(target_bin).detach()
    union = union.sub_(inter) # probably to reduce 2 to 1.
    inter = inter.sum(3).sum(2).sum(1)
    union = union.sum(3).sum(2).sum(1)
    return inter.div(union), bs



def iou_voxel(pred, voxel):
    """ Computes intersection over union between two shapes.
        Returns iou summed over batch
    """
    bs,_,h,w = pred.size()
    
    inter = pred.mul(voxel).detach()
    union = pred.add(voxel).detach()
    union = union.sub_(inter)
    inter = inter.sum(3).sum(2).sum(1)
    union = union.sum(3).sum(2).sum(1)
    return inter.div(union).sum(), bs
        

def iou_shapelayer(pred, voxel, id1, id2, id3):
    """ Compares prediction and ground truth shape layers using IoU.
        Returns iou summed over batch and number of samples in batch.
    """
    pred  = pred.detach()
    voxel = voxel.detach()

    bs, _, side, _ = pred.shape
    vp = pred.new_zeros(bs,side,side,side, requires_grad=False)
    vt = pred.new_zeros(bs,side,side,side, requires_grad=False)
    
    for i in range(bs):
        vp[i,:,:,:] = decode_shape(pred[i,:,:,:].short().permute(1,2,0),  id1, id2, id3)
        vt[i,:,:,:] = decode_shape(voxel[i,:,:,:].short().permute(1,2,0), id1, id2, id3)

    return iou_voxel(vp,vt)
    

k_save = 0
def save(c, d, name=None):
    global k_save
    if c:
        k_save += 1
        if name is None:
            name = 'dbg_%d.mat' % k_save
        sio.savemat(name, {k:d[k].detach().cpu().numpy() for k in d.keys()})

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv) # nice to have in log files

    # register networks, datasets, etc.
    name2net        = {'resnet': ResNet}
    net_default     = 'resnet'

    name2dataset    = {\
        'ShapeNet':ShapeNet3DR2N2Collector}
    dataset_default = 'ShapeNet'

    name2optim      = {'adam': optim.Adam}
    optim_default   = 'adam'

    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',     type=str,            default='matryoshka', help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',   action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu',       type=int,            default=0,     help='GPU ID if cuda is available and enabled')
    parser.add_argument('--no_save',   action='store_true', default=False, help='Disables saving of final model')
    parser.add_argument('--no_val',    action='store_true', default=False, help='Disable validation for faster training')
    parser.add_argument('--batchsize', type=int,            default=16,    help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',    type=int,            default=1,    help='number of epochs to train') 
    parser.add_argument('--nthreads',  type=int,            default=4,     help='number of threads for loader') 
    parser.add_argument('--seed',      type=int,            default=1,     help='random seed (default: 1)')
    parser.add_argument('--val_inter', type=int,            default=1,     help='Validation interval in epochs (default: 1)')
    parser.add_argument('--log_inter', type=int,            default=100,   help='Logging interval in batches (default: 100)')
    parser.add_argument('--save_inter', type=int,           default=10,    help='Saving interval in epochs (default: 10)')

    # options for optimizer
    parser.add_argument('--optim', type=str,   default=optim_default, help=('Optimizer [%s]' % ','.join(name2optim.keys())))
    parser.add_argument('--lr',    type=float, default=1e-3,          help='Learning rate (default: 1e-3)')
    parser.add_argument('--decay', type=float, default=0,             help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--drop',  type=int,   default=30)

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--basedir',          type=str,            default='/media/data/',       help='Base directory for dataset.')
    parser.add_argument('--no_shuffle_train', action='store_true', default=False,           help='Disable shuffling of training samples')
    parser.add_argument('--no_shuffle_val',   action='store_true', default=False,           help='Disable shuffling of validation samples')

    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--side',  type=int, default=32, help='Output resolution [if dataset has multiple resolutions.] (default: 128)') 
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')
    parser.add_argument('--ninf',  type=int, default=8,   help='Number of initial feature channels (default: 8)')
    parser.add_argument('--ngf',   type=int, default=512, help='Number of inner channels to train (default: 512)')
    parser.add_argument('--noutf', type=int, default=128, help='Number of penultimate feature channels (default: 128)')
    parser.add_argument('--down',  type=int, default=5,   help='Number of downsampling blocks. (default: 5)')
    parser.add_argument('--block', type=int, default=1,   help='Number of inner blocks at same resolution. (default: 1)')

    # options for visualisation
    parser.add_argument('--vis_inputs', action='store_true', default=False, help='if True, will only print inputs')

    # other options
    parser.add_argument('--label_type',  type=str, default='vox', help='Type of representation: vox(voxels), sdf or chamfer')
    parser.add_argument('--path_to_res',  type=str, default='/media/results', help='path to output results')
    parser.add_argument('--path_to_data',  type=str, default='/media/data', help='path to output results')
    parser.add_argument('--path_to_prep_shapenet',  type=str, default='/media/data/prep_shapenet', help='path to prep shapenet')
 
    parser.add_argument('--subset',  type=str, default='train', help='data subset, can be train or test')

    parser.add_argument('--cat_id',  type=str, default='02958343', help='cat_id, default is cars 02958343') 
    parser.add_argument('--p_norm',  type=int, default=1, help='p_norm for paper loss') 
    parser.add_argument('--eps_delta',  type=float, default=0.1, help='epsilon for dirac delta') 

    
    args = parser.parse_args()
   

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.shuffle_train = not args.no_shuffle_train
    args.shuffle_val   = not args.no_shuffle_val 

    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")

    torch.manual_seed(1)

    if args.vis_inputs == True:
        args.shuffle_train = False

    # load paths of voxels and views
    try:
        logging.info('Initializing dataset "%s"' % args.dataset)
        Collector = ShapeNet3DR2N2Collector(base_dir=args.basedir,cat_id = args.cat_id,
            representation=args.label_type, side=args.side, p_norm=args.p_norm)
    except KeyError:
        logging.error('A dataset named "%s" is not available.' % args.net)
        exit(1)

    set_trace()
#    print('check out samples')

    logging.info('Initializing dataset loader')
    if args.subset == 'train':
        samples = Collector.train()
    elif args.subset == 'test':
        samples = Collector.test()
    else:
        print('unknown subset')
        exit()

    logging.info('Found %d training samples.' % len(samples))

    acoto_dataset= DatasetLoader(samples, args.side, 
        input_transform=transforms.Compose([transforms.ToTensor(), RandomColorFlip()]))

    train_loader = torch.utils.data.DataLoader(acoto_dataset, \
        batch_size=args.batchsize, shuffle=args.shuffle_train, num_workers=args.nthreads, \
        pin_memory=True)
    
    if args.no_val: 
        samples = Collector.val()
        logging.info('Found %d validation samples.' % len(samples))
        val_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor()])), \
        batch_size=args.batchsize, shuffle=args.shuffle_val,   num_workers=args.nthreads, \
        pin_memory=True)
        pass

    samples = []

    # load network
    try:
        logging.info('Initializing "%s" network' % args.net)
        net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=args.ninf,
            num_inner_channels=args.ngf,
            num_penultimate_channels=args.noutf, 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=32,
            bottleneck_dim = 128,
            num_downsampling=args.down, 
            num_blocks=args.block,
            ).to(device)
        logging.info(net)
    except KeyError:
        logging.error('A network named "%s" is not available.' % args.net)
        exit(2)
    if args.file:
        savegame = torch.load(args.file)
        net.load_state_dict(savegame['state_dict'])

    # init optimizer
    try:
        logging.info('Initializing "%s" optimizer with learning rate = %f and weight decay = %f' % (args.optim, args.lr, args.decay))
        optimizer = name2optim[args.optim](net.parameters(), lr=args.lr, weight_decay=args.decay)
    except KeyError:
        logging.error('An optimizer named "%s" is not available.' % args.optim)
        exit(3)
  
    try:
        net.train()

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.drop, gamma=0.5)
        if args.label_type == 'vox':
            loss_type = 'cross'
        elif args.label_type == 'sdf':
            loss_type = 'l2'
        elif args.label_type == 'chamfer':
            loss_type = 'chamfer'
        else:
            print('unknown label_type')
            exit()
        

        agg_loss  = 0.
        count     = 0

        for epoch in range(1, args.epochs + 1):
            scheduler.step()
            # VIS INPUTS BLOCK
            if args.vis_inputs:
                coto_dataiter = iter(train_loader)
                while True:
                    set_trace()
                    print('check out images and labels')
                    images, labels, example_ids = coto_dataiter.next()
                    for example_nr in range(args.batchsize):
                        c_view = images[example_nr].numpy()
                        c_label = labels[example_nr].numpy()
                        vis_view(c_view)
                        if args.label_type == 'vox':
                            vis_voxels(c_label)
                        elif args.label_type == 'sdf':
                            vis_sdf(c_label)
                        elif args.label_type == 'chamfer':
                            # vis gt manifold
                            path_to_man = os.path.join(args.path_to_prep_shapenet,
                                args.cat_id, 'meshes',
                                example_ids[example_nr],
                                'sim_manifold_50000.obj')
                            c_label_man = trimesh.load_mesh(path_to_man)
                            if type(c_label_man) is list:
                                c_label_man = merge_mesh(c_label_man)
                            vis_mesh(c_label_man.vertices, c_label_man.faces)
                            # vis gt SDF
                            path_to_sdf = os.path.join(args.path_to_data,
                                'ShapeNetSDF'+str(args.side), args.cat_id,
                                example_ids[example_nr],
                                'sdf_manifold_50000_grid_size_'+str(args.side)+'.npy')
                            c_label_sdf = np.load(path_to_sdf)\
                                .reshape((args.side,)*3)
                            vis_sdf(c_label_sdf)

                        else:
                            print('unknown label_type')
                            exit()
                        show()
                        print('Should I stop? ')
                        user_response = input()
                        if user_response == 'y':
                            exit()
                        else:
                            break 
            tmp_loss_list = []

            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs  = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                pred    = net(inputs) # this is compressed

                if loss_type == 'chamfer':
                    loss_chamf, loss_sdf = my_loss(pred, targets, loss_type, eps=args.eps_delta,
                        p_norm=args.p_norm)
                    alpha_sdf = 0.1
                    loss = loss_chamf + alpha_sdf * loss_sdf
                else:
                    loss = my_loss(pred, targets, loss_type, eps=args.eps_delta,
                        p_norm=args.p_norm)

    
                loss.backward()        
                optimizer.step()

                agg_loss += loss.detach()
                count    += inputs.shape[0]
                tmp_loss_list.append(loss.item())
                pass
            
                if batch_idx % args.log_inter == 0:
                    logging.info('%d/%d: Train loss: %.10f [%s]' % (epoch, batch_idx, agg_loss.item()/count, args.title))
                
                    agg_loss = 0.
                    count    = 0

            filename = '%s_%s_%s_%d.pth.tar'\
                % (args.title, args.dataset, args.label_type, epoch)
            path_to_out = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'loss',
                filename+'_loss.txt')
            cmd = 'mkdir -p '+os.path.dirname(path_to_out)
            os.system(cmd)
            tmp_loss_list = list(map(str, tmp_loss_list))
        
            with open(path_to_out, 'a') as fout:
                fout.write('\n'.join(tmp_loss_list))
            
            if (epoch % 10 == 0):
                filename = '%s_%s_%s_%d.pth.tar'\
                    % (args.title, args.dataset, args.label_type, epoch)
                logging.info('Saving model to %s.' % filename)
                torch.save({'state_dict': net.state_dict(), 
                     'optimizer' : optimizer.state_dict(),
                     'ninf':args.ninf,
                     'ngf':args.ngf,
                     'noutf':args.noutf,
                     'block':args.block,
                     'side': args.side,
                     'down':args.down,
                     'epoch': epoch,
                     'optim': args.optim,
                     'lr': args.lr,
                 }, os.path.join(args.path_to_res, args.cat_id, args.label_type, filename))
     
            if args.no_val and epoch % args.val_inter == 0: # FIX
                net.eval()
        
                agg_iou = 0.
                count   = 0
                with torch.no_grad():    
                    for batch_idx, (inputs, targets) in enumerate(val_loader):

                        inputs  = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                
                        pred     = net(inputs)                
                        iou, bs  = iou_shapelayer(shlx2shl(pred), targets, id1, id2, id3)
                        agg_iou += float(iou)
                        count   += bs

                        pass
                    pass
            
                net.train()
                
                total_iou = (100 * agg_iou / count) if count > 0 else 0

                logging.info('%d: Val set accuracy, iou: %.1f [%s]' % (epoch, total_iou, args.title))                
                pass
            pass

    except KeyboardInterrupt:
        pass

    if not args.no_save or False:
        filename = '%s_%s_%d.pth.tar' % (args.title, args.dataset, epoch)
        logging.info('Saving model to %s.' % filename)
        torch.save({'state_dict': net.state_dict(), 
                     'optimizer' : optimizer.state_dict(),
                     'ninf':args.ninf,
                     'ngf':args.ngf,
                     'noutf':args.noutf,
                     'block':args.block,
                     'side': args.side,
                     'down':args.down,
                     'epoch': epoch,
                     'optim': args.optim,
                     'lr': args.lr,
                 }, os.path.join(args.path_to_res, args.cat_id, args.label_type, filename)) 
