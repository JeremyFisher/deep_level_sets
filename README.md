Implemetation of the paper:
**Implicit Surface Representations as Layers in Neural Networks**

Michalkiewicz M, Pontes K, Jack D, Baktashmotlagh M, Eriksson A. In ICCV 2019.


[link] (http://openaccess.thecvf.com/content_ICCV_2019/papers/Michalkiewicz_Implicit_Surface_Representations_As_Layers_in_Neural_Networks_ICCV_2019_paper.pdf)

-----------------------
Dependencies:
-----------------------

To run the code, install the following packages in conda environment:

```
conda create  -n dls python=3.7
source activate dls
conda install scipy pillow Pillow trimesh numpy
conda install -c conda-forge scikit-fmm 
conda install  pytorch torchvision -c pytorch
```


----------------------------------------------------------------------
General notes
----------------------------------------------------------------------
**The code is largely based on Matryoshka [1] repository [2] and was modified accordingly.**

The 2D encoder used is based on Matryoshka paper [1], however using any other encoder
should give similar results.

The very simple 3D decoder used is based on TL paper [3], however using any other
3D decoder should give similar (most likely better) results.

------------
Datasets
------------
We have used 3D models from ShapeNetCore.v1

2D input images are expected to be have a shape of 128x128.

To process standard 3D-R2N2 [4] views, use `crop_images.py`.

3D ground truth should be signed distance functions of watertight manifolds of shape
32x32x32. Watertight manifolds can be obtained with the Manifold code [5]

Datasets are loaded using DatasetCollector.py and DatasetLoader.py.

-------------------------
References
---------------------------
	
[1] https://arxiv.org/abs/1804.10975

[2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/

[3] https://arxiv.org/abs/1603.08637

[4] https://arxiv.org/abs/1604.00449
