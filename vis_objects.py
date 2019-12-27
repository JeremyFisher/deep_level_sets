from time import sleep
from mayavi import mlab
from mayavi.mlab import show
from matplotlib import pyplot as plt
from ipdb import set_trace
from binvox_rw import read_as_3d_array
import numpy as np
from trimesh import load_mesh
import trimesh
from skimage.measure import marching_cubes_lewiner as mc

from os import listdir, system
from os.path import join

color = tuple(np.asarray((38, 139, 210))/255.0)

def merge_mesh(mesh_list):                                                      
    vertices = mesh_list[0].vertices                                            
    faces = mesh_list[0].faces                                                  
    for i in range(len(mesh_list))[1:]:                                         
        nr_of_verts = vertices.shape[0]                                         
        new_verts = mesh_list[i].vertices                                       
        new_faces = mesh_list[i].faces                                          
        vertices = np.append(vertices, new_verts).reshape(-1,3)                 
        new_faces += nr_of_verts                                                
        faces = np.append(faces, new_faces).reshape(-1,3)                       
    new_mesh = trimesh.Trimesh(vertices, faces)                                 
    return new_mesh

def vis_pc(pc, **kwargs):
    ''' vis point cloud of shape X,3
    use scale_factor=0.1 for example as arg
    '''
    assert len(pc.shape) == 2
    assert pc.shape[1] == 3
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], **kwargs)

def vis_mesh(
        vertices, faces, axis_order='zyx', include_wireframe=True,
        color=color, distance=2.,**kwargs):
    if len(faces) == 0:
        print('Warning: no faces')
        return
    fig_data = mlab.figure(size=(800, 600), bgcolor=(1,1,1), fgcolor=None, engine=None)
    x, y, z = permute_xyz(*vertices.T, order=axis_order)
    mlab.triangular_mesh(x, y, z, faces, color=color, **kwargs)
    mlab.view(distance=distance, focalpoint='auto', roll=2)          

def vis_sdf(sdf, distance=56., threshold=0.0, **kwargs):
    ''' extracts levelset and uses vis_mesh'''
    levelset = mc(sdf, threshold)
    vis_mesh(vertices=levelset[0], faces = levelset[1], distance=distance,
        **kwargs)

def vis_voxels(voxels, axis_order='xzy',color=color, **kwargs):
    fig_data = mlab.figure(size=(800, 600), bgcolor=(1,1,1), fgcolor=None, engine=None)
    data = permute_xyz(*np.where(voxels), order=axis_order)
    if len(data[0]) == 0:
        # raise ValueError('No voxels to display')
        Warning('No voxels to display')
    else:
        kwargs.setdefault('mode', 'cube')
        mlab.points3d(*data,color=color, **kwargs)

def vis_view(view_np):
    ''' views view. assumes a np array'''
    if view_np.shape[0] == 3: 
        new_view\
            = np.zeros((view_np.shape[1],view_np.shape[2], view_np.shape[0]))
        new_view[:,:,0] = view_np[0,:]
        new_view[:,:,1] = view_np[1,:]
        new_view[:,:,2] = view_np[2,:]
        view_np = new_view
    plt.imshow(view_np)
    plt.show()    

def show():
    mlab.show()

def permute_xyz(x, y, z, order='xyz'):
    _dim = {'x': 0, 'y': 1, 'z': 2}
    data = (x, y, z)
    return tuple(data[_dim[k]] for k in order)

def print_voxels_to_file(vxls,path_to_save_data):                               
    vis_voxels(vxls)                                                            
    mlab.savefig(path_to_save_data)                                             
    sleep(2)                                                                    
    mlab.close()                                                                
                                                                                
def print_sdf_to_file(sdf, path_to_save_data, threshold=0.0, distance=56.):
    vertices, faces, _, _ = mc(sdf, threshold)
    print_mesh_to_file(vertices, faces, path_to_save_data, distance=distance)
                                                                           
def print_mesh_to_file(verts, faces, path_to_save_data, distance=56.):
    vis_mesh(verts, faces, distance=distance)
    mlab.savefig(path_to_save_data)                                             
    sleep(2)                                                                    
    mlab.close()                                                                

