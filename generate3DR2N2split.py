'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

import os
import json
from collections import OrderedDict

from ipdb import set_trace


 

def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id and v[1] > id:
            return (k, id - v[0])

 

 

def category_model_id_pair(dataset_portion=[]):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    cats = json.load(open('ShapeNet.json'))

    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    for k, cat in cats.items():  # load by categories
        model_path = os.path.join('/home/matryoshka/matryoshka/data/ShapeNetVox32', cat['id'])

        models = model_names(model_path)
        num_models = len(models)

        portioned_models = models[int(num_models * dataset_portion[0]):int(num_models *
                                                                           dataset_portion[1])]

        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    return category_name_pair

 

with open('3dr2n2-train.txt', 'w') as f:
    for synset, model in category_model_id_pair([0,0.8]):
        f.write('%s/%s\n' % (synset, model))

 

with open('3dr2n2-test.txt', 'w') as f:
    for synset, model in category_model_id_pair([0.8,1]):
        f.write('%s/%s\n' % (synset, model))
