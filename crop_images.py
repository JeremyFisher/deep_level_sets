'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

from utils import convert_files
from math import ceil, floor
import PIL.Image
import numpy as np
import argparse
import os
from ipdb import set_trace

def load_image(temp):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if temp.mode == 'RGBA':
        alpha = temp.split()[-1]
        bg = PIL.Image.new("RGBA", temp.size, (255,255,255) + (255,))
        bg.paste(temp, mask=alpha)
        im = bg.convert('RGB').copy()
        bg.close()
        temp.close()
    else:
        im = temp.copy()
        temp.close()
    return im

def make_crop_func(size):
    def croptox(path):
        pad = 100
        img = load_image(PIL.Image.open(path))
        a = np.asarray(img)
        bg_mask = np.min(img, axis=2)==255
        cols = np.flatnonzero(np.logical_not(np.min(bg_mask, axis=0)))
        rows = np.flatnonzero(np.logical_not(np.min(bg_mask, axis=1)))
        p = ceil(np.max([cols[-1]-cols[0], rows[-1]-rows[0]]) / 2)
        imp = np.pad(img, ((pad, pad), (pad, pad), (0,0)), 'constant', constant_values=255)
        row_offset = pad+floor((rows[0]+rows[-1])/2)
        col_offset = pad+floor((cols[0]+cols[-1])/2)
        i2 = imp[row_offset-p:row_offset+p+1, \
                col_offset-p:col_offset+p+1,:]

        i2 = PIL.Image.fromarray(i2).resize((size, size), PIL.Image.LANCZOS)
        i2.save(path[:-4] + '.%d.png' % size, 'PNG')
        pass    
    return croptox


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Crop images to certain size.')
    parser.add_argument('directory', type=str, help='Directory with PNG files.', default='.')
    parser.add_argument('-s', '--size', type=int, default=128, help='Target size of images (width in pixels).')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively traverses the directory.')

    args = parser.parse_args()

    set_trace()

    convert_files(args.directory, '.png', make_crop_func(args.size), args.recursive, '.%s.png' % args.size)
    pass
