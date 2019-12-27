'''
This code is based on the Matryoshka [1] repository [2] and was modified accordingly:

    [1] https://arxiv.org/abs/1804.10975

    [2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/
    
    Copyright (c) 2018, Visual Inference Lab @TU Darmstadt
'''

import os

def convert_files(dir, in_ext, action, recursive=True, exclude_ext=None):
    """ Traverse directory recursively to convert files.
        If recursive==False, only files in the directory dir are converted.
    """
    files = sorted(os.listdir(dir))
    for file in files:
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            if recursive:
                convert_files(path, in_ext, action, recursive)
                pass
            pass
        elif path.endswith(in_ext):
            if exclude_ext is None or not path.endswith(exclude_ext):
                action(path)
                pass
            pass
        pass
    pass
