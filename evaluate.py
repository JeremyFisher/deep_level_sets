'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

import numpy as np
import scipy.io as sio
import argparse
import os

from voxel2layer import decode_shape, generate_indices
from DatasetCollector import *

def evaluate_sample(file, shape_layer, id1, id2, id3):

    d = sio.loadmat(file)
    voxel = d['voxel']

    decoded = decode_shape(shape_layer, id1, id2, id3)
    
    return np.sum(np.logical_and(voxel, decoded)) / np.sum(np.logical_or(voxel, decoded))
    

if __name__ == '__main__':

    name2dataset    = {'ShapeNetPTN':ShapeNetPTNCollector, \
        'ShapeNetCars':ShapeNetCarsOGNCollector, \
        'ShapeNet':ShapeNet3DR2N2Collector}
    dataset_default = 'ShapeNet'

    parser = argparse.ArgumentParser('Evaluate results')
    parser.add_argument('result_dir', type=str, default='.', help='Directory with result batches.')
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--basedir',          type=str,            default='./data/',       help='Base directory for dataset.')
    parser.add_argument('--set', type=str, choices=['val', 'test'], help='Subset to evaluate. (default: val)', default='val')
    args = parser.parse_args()

    collector = name2dataset[args.dataset](basedir=args.basedir)        

    if args.set == 'val':
        samples = collector.val()
    elif args.set == 'test':
        samples = collector.test()

    batchfiles = [os.path.join(args.result_dir, f) \
        for f in sorted(os.listdir(args.result_dir)) \
            if f.startswith('b_') and f.endswith('.mat')]

    agg_iou = 0
    count   = 0
    num_samples = len(samples)

    with open(os.path.join(args.result_dir, 'results.txt'), 'w') as f:
        for batchfile in batchfiles:
            batch = sio.loadmat(batchfile)['results']
            for i in range(batch.shape[0]):
                if count == 0:
                    side = batch.shape[2]
                    id1, id2, id3 = generate_indices(side)
                    pass

                shape_layer = np.transpose(batch[i,:,:,:], axes=[1,2,0])
                iou      = evaluate_sample(samples[count][1][:-8]+'.vox.mat', shape_layer, id1, id2, id3)
                agg_iou += iou

                f.write('%d,%s,%.1f\n' % (count, samples[count][1][:-8], 100*iou))				

                count   += 1				

                if count % 100 == 0:
                    print('Mean %.1f (%d/%d)' % (100*agg_iou / count, count, num_samples))
                    pass
                pass
            pass
        f.write('Mean,%.1f\n' % (100*agg_iou / count))
        pass
    pass
