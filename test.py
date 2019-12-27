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
from utils import *
import math
import sys, os


from matr_iccv.train import iou_voxel, iou_shapelayer, my_iou_voxel, my_iou_sdf
from matr_iccv.ResNet import *
from matr_iccv.DatasetLoader import *
from matr_iccv.DatasetCollector import *
from ipdb import set_trace
from scipy.stats import logistic
from vis_objects import vis_view, vis_mesh, vis_sdf, vis_voxels, show, merge_mesh
import trimesh

from binvox_rw import read_as_3d_array

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer


widgets = [Percentage(),
           ' ', Bar(),
           ' ', ETA(),
           ' ', AdaptiveETA()]


def vis_slice(sdf_slice):
    from mayavi import mlab
    mlab.surf(sdf_slice, warp_scale='auto')

if __name__ == '__main__':

    
    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv) # nice to have in log files

    # register networks, datasets, etc.
    name2net        = {'resnet': ResNet}
    net_default     = 'resnet'

    name2dataset    = {'ShapeNet':ShapeNet3DR2N2Collector}
    dataset_default = 'ShapeNet'

    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',     type=str,            default='matryoshka', help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',   action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu',       type=int,            default=0,     help='GPU ID if cuda is available and enabled')
    parser.add_argument('--batchsize', type=int,            default=16,    help='input batch size for training (default: 128)') 
    parser.add_argument('--nthreads',  type=int,            default=1,     help='number of threads for loader') 
    parser.add_argument('--save_inter', type=int,           default=10,    help='Saving interval in epochs (default: 10)')

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--set',              type=str,            default='test',         help='Validation or test set. (default: val) or train. input samples', choices=['train', 'val', 'test']) 
    parser.add_argument('--basedir',          type=str,            default='/media/data/',       help='Base directory for dataset.')

    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')

    # other options
    parser.add_argument('--label_type',  type=str, default='vox', help='Type of representation: vox(voxels), sdf or mesh')

    parser.add_argument('--vis_pred', action='store_true', default=False, help='if True, will only print predictions')
    parser.add_argument('--gen_report', type=str, default='no', help='if not <no> then generate iou, imgs or all the above')
    parser.add_argument('--test_ids', type=str, default='0 5 10', help='if not <no> then generate iou, loss, imgs or all the above')    
    parser.add_argument('--cat_id',  type=str, default='02958343', help='cat_id, default is cars 02958343') 
    parser.add_argument('--path_to_res',  type=str, default='/media/results/', help='path to output results')
    parser.add_argument('--path_to_prep_shapenet',  type=str, default='/media/prep_shapenet', help='path to prep shapenet')
    parser.add_argument('--path_to_data',  type=str, default='/media/data', help='path to output results')

    parser.add_argument('--side',  type=int, default=32, help='Output resolution [if dataset has multiple resolutions.] (default: 128)') 
    parser.add_argument('--p_norm',  type=int, default=1, help='p_norm for paper loss, default =2') 

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
 
    assert args.gen_report in ['iou', 'loss', 'imgs', 'all', 'no']

    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")

    torch.manual_seed(1)

    if args.file == None:
        args.file = '/home/results/matryoshka_ShapeNet_10.pth.tar'
    else:
        pass
    assert os.path.exists(args.file)

    args.no_cuda = True

    if args.vis_pred:
        args.batchsize=1

    savegame = torch.load(args.file)
    args.side = savegame['side']
   
    # load dataset
    try:
        logging.info('Initializing dataset "%s"' % args.dataset)
        Collector = ShapeNet3DR2N2Collector(base_dir=args.basedir,cat_id=args.cat_id,
            representation=args.label_type, side=args.side, p_norm=args.p_norm)
    except KeyError:
        logging.error('A dataset named "%s" is not available.' % args.dataset)
        exit(1)

    logging.info('Initializing dataset loader')
    set_trace()

    if args.set == 'val':
        samples = Collector.val()
    elif args.set == 'test':
        samples = Collector.test()
    elif args.set == 'train':
        samples = Collector.train()
    
    num_samples = len(samples)
    logging.info('Found %d test samples.' % num_samples)
    test_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.side, args.ncomp, \
        input_transform=transforms.Compose([transforms.ToTensor()])), \
        batch_size=args.batchsize, shuffle=False, num_workers=args.nthreads, \
        pin_memory=True)

    if args.gen_report != 'no':
        test_ids = list(map(int, args.test_ids.split(' ')))
        samples_small = []
        for test_id in test_ids:
            samples_small += [samples[test_id*24]]

        test_loader_small = torch.utils.data.DataLoader(DatasetLoader(samples_small, args.ncomp, \
            input_transform=transforms.Compose([transforms.ToTensor()])), \
            batch_size=len(samples_small), shuffle=False, num_workers=1, \
            pin_memory=True)
            
        # gather np guys from test_ids
        test_ids = list(map(int, args.test_ids.split(' ')))
        test_guys_np = np.zeros((len(test_ids), args.side, args.side, args.side))

        if args.label_type == 'vox':
            for _, test_id in enumerate(test_ids):
                path_to_vox_tmp = samples[test_id*args.batchsize][1]
                with open(path_to_vox_tmp, 'rb') as fin:
                    vox_tmp = read_as_3d_array(fin).data
                test_guys_np[_] = vox_tmp
        elif args.label_type == 'sdf':            
            for _, test_id in enumerate(test_ids):
                path_to_sdf_tmp = samples[test_id*args.batchsize][1]
                sdf_tmp = np.load(path_to_sdf_tmp)\
                    .reshape(args.side, args.side, args.side)
                test_guys_np[_] = sdf_tmp
        else:
            print('strange label type')
            exit()

    samples = []

    net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=savegame['ninf'],
            num_inner_channels=savegame['ngf'],
            num_penultimate_channels=savegame['noutf'], 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=savegame['side'],
            num_downsampling=savegame['down'],
            bottleneck_dim = 128,
            num_blocks=savegame['block'],
            ).to(device)
    logging.info(net)
    net.load_state_dict(savegame['state_dict'])
    
    net.eval()

    agg_iou   = 0.
    count     = 0
    results   = torch.zeros(args.batchsize*100, 6, 128,128).to(device)

    ctr_tej=0
    real_ctr = 0
    vox_threshold = 0.5

    iou_list = []
     

    if args.gen_report == 'imgs' or args.gen_report == 'all':
        with torch.no_grad():
            for batch_idx, (inputs, targets, ex_ids) in enumerate(test_loader_small):
                if args.label_type == 'vox':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred = net(inputs)
                    pred = torch.sigmoid(pred) > vox_threshold
                    targets = targets.float()
                    pred = pred.float()
                    targets = targets.cpu().numpy()
                    pred = pred.cpu().numpy()
                    inputs = inputs.cpu().numpy()
                elif args.label_type == 'sdf':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred    = net(inputs) # this is compressed
                    targets = targets.float()
                    pred = pred.float()
                    targets = targets.cpu().numpy()
                    pred = pred.cpu().numpy()
                    inputs = inputs.cpu().numpy()
#           assert batch_idx == 0 
            # GT
            tmp_str = os.path.basename(args.file)
            path_to_out_gt = os.path.join(os.path.dirname(args.file), 'imgs',
                tmp_str[:-4]+'_gt_imgs')
            cmd = 'mkdir -p '+os.path.dirname(path_to_out_gt)
            os.system(cmd)
            np.save(path_to_out_gt, targets)
            # PRED
            path_to_out_gt = os.path.join(os.path.dirname(args.file), 'imgs',
                tmp_str[:-4]+'_pred_imgs')
            np.save(path_to_out_gt, pred)
            # VIEW
            path_to_out_view = os.path.join(os.path.dirname(args.file), 'imgs',
                tmp_str[:-4]+'_view_imgs')
            np.save(path_to_out_view, inputs)
            # EXAMPLE IDS
            example_ids = []
            for i in range(len(samples_small)):
                example_id = samples_small[i][1]
                tmp_right_idx = example_id.rfind('/')
                tmp_left_idx = example_id.rfind('/', 0, tmp_right_idx)
                example_id = example_id[tmp_left_idx+1:tmp_right_idx]
                example_ids.append(example_id)
            path_to_out_ids = os.path.join(os.path.dirname(args.file), 'imgs',
                tmp_str[:-4]+'_ids_imgs.txt')
            with open(path_to_out_ids, 'w') as fout:
                fout.write('\n'.join(example_ids))

    with torch.no_grad():
        for batch_idx, (inputs, targets, example_ids) in enumerate(test_loader):
            print(batch_idx)
            # VIS INPUTS BLOCK
            if args.vis_pred:              
                if batch_idx % 24 != 10: # there are 24 views, take only 1
                    continue        
                set_trace()
                inputs  = inputs.to(device, non_blocking=True)
                pred = net(inputs)
                inputs = inputs.cpu()
                pred = pred.cpu()                   

                example_nr=0
                c_view = inputs[example_nr].numpy()
                c_label = targets[example_nr].numpy()              
                vis_view(c_view)
                if args.label_type == 'vox':
                    c_pred = logistic.cdf(pred[example_nr].numpy()) > vox_threshold
                    vis_voxels(c_label, color=(0,1,0))
                    vis_voxels(c_pred)
                elif args.label_type == 'sdf':
#                    c_pred = pred[example_nr].numpy()+0.1
                    c_pred = pred[example_nr].numpy()
                    vis_sdf(c_label, color=(0,1,0))
                    vis_sdf(c_pred)                    
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
                    vis_sdf(c_label_sdf, color=(1,0,0))
                    
                    c_pred = pred[example_nr].numpy()
                    vis_sdf(c_pred)                    

                else:
                    print('unknown label type')
                    exit()
                show()
                print('Should I stop? ')
                user_response = input()
                if user_response == 'y':
                    exit()
                else:
                    continue
            else: 
                if args.label_type == 'vox':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred = net(inputs)
                    pred = torch.sigmoid(pred) > vox_threshold
                    targets = targets.float()
                    pred = pred.float()

                    targets = targets.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    inputs = inputs.detach().cpu().numpy()

                    tmp_nr = 7
                    example_id = example_ids[tmp_nr]
                    path_to_out_gt = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_vox_gt_'+str(args.side)+'_pad_1.npy')
                    cmd = 'mkdir -p '+os.path.dirname(path_to_out_gt)
                    os.system(cmd)
                    np.save(path_to_out_gt, targets[tmp_nr])

                    path_to_out_pred = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_vox_pred_'+str(args.side)+'_pad_1.npy')
                    np.save(path_to_out_pred, pred[tmp_nr])

                    path_to_out_view = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_view.npy')
                    np.save(path_to_out_view, inputs[tmp_nr])

                elif args.label_type == 'sdf':
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred = net(inputs)
                    targets = targets.float()
                    pred = pred.float()
                    # Just save to file
                    targets = targets.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    inputs = inputs.detach().cpu().numpy()
                    tmp_nr = 7
                    example_id = example_ids[tmp_nr]
                    path_to_out_gt = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_sdf_gt_'+str(args.side)+'_pad_1.npy')
                    cmd = 'mkdir -p '+os.path.dirname(path_to_out_gt)
                    os.system(cmd)
                    np.save(path_to_out_gt, targets[tmp_nr])

                    path_to_out_pred = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_sdf_pred_'+str(args.side)+'_pad_1.npy')
                    np.save(path_to_out_pred, pred[tmp_nr])

                    path_to_out_view = os.path.join(args.path_to_res, args.cat_id, args.label_type, 'metrics', example_id,
                        example_id+'_view.npy')
                    np.save(path_to_out_view, inputs[tmp_nr])
                   

                else:
                    print('unknown label type')
                    exit()               
                if args.gen_report == 'iou' or args.gen_report == 'all':
                    if args.label_type == 'vox':
                        iou, bs = my_iou_voxel(pred, targets)
                        iou_list += iou.cpu().tolist()
                    elif args.label_type == 'sdf':
                        iou, bs = my_iou_sdf(pred, targets)
                        iou_list += iou.cpu().tolist()                       
                    else:
                        print('unkown label type')
                        exit()
                   



    pass


