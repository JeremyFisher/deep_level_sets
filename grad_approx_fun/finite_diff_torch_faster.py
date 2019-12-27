from ipdb import set_trace
import numpy as np
import torch


def get_norm3d_batch(list_of_grad):
    assert type(list_of_grad) is list
    batch_size = len(list_of_grad)
    grid_size = list_of_grad[0][0].shape[0]
    norm_batch = np.zeros((batch_size, grid_size, grid_size, grid_size))
    for _, c_grad in enumerate(list_of_grad):
        c_norm = get_norm3d(c_grad)
        norm_batch[_] = c_norm
    return norm_batch

def get_norm3d_batch_torch(list_of_grad_torch, device):
#    set_trace()
    assert type(list_of_grad_torch) is list
    batch_size = len(list_of_grad_torch)
    grid_size = list_of_grad_torch[0][0].shape[0]
    norm_batch_torch\
        = torch.zeros((batch_size, grid_size, grid_size, grid_size),\
         dtype=list_of_grad_torch[0][0].dtype, device=device)
    for _, c_grad_t in enumerate(list_of_grad_torch):
        c_norm_t = get_norm3d_torch(c_grad_t)
        norm_batch_torch[_] = c_norm_t
    return norm_batch_torch


def get_norm3d(gradients):
    assert type(gradients) is tuple
    assert len(gradients) == 3
    norm = gradients[0]**2 + gradients[1]**2 + gradients[2]**2
    norm = np.sqrt(norm)
    return norm

def get_norm3d_torch(gradients_torch, small_nr = 1e-8):
    assert type(gradients_torch) is tuple
    assert len(gradients_torch) == 3
    norm_torch\
        = gradients_torch[0]**2 + gradients_torch[1]**2 + gradients_torch[2]**2
    norm_torch = torch.sqrt(norm_torch+small_nr)
    return norm_torch


def apply_diff3d_torch(phi_t, i, j, k, direction, type, step=1): 
    assert len(phi_t.shape) == 3 # phi_t is 3d
    assert direction in ['X', 'Y', 'Z']
    assert type in ['forward', 'backward', 'central']
    if type == 'forward':
        if direction == 'X':
            c_value = (phi_t[i+step,j, k]-phi_t[i,j, k])/step
        elif direction == 'Y':
            c_value = (phi_t[i,j+step, k]-phi_t[i,j, k])/step
        elif direction == 'Z':
            c_value = (phi_t[i,j, k+step]-phi_t[i,j, k])/step

    elif type == 'backward':
        if direction == 'X':
            c_value = (phi_t[i,j, k]-phi_t[i-step,j, k])/step
        elif direction == 'Y':
            c_value = (phi_t[i,j, k]-phi_t[i,j-step, k])/step
        elif direction == 'Z':
            c_value = (phi_t[i,j, k]-phi_t[i,j, k-step])/step

    elif type == 'central':
        if direction == 'X':
            c_value = (phi_t[i+step,j, k]-phi_t[i-step,j, k])/(2*step)
        elif direction == 'Y':
            c_value = (phi_t[i,j+step, k]-phi_t[i,j-step, k])/(2*step)
        elif direction == 'Z':
            c_value = (phi_t[i,j, k+step]-phi_t[i,j, k-step])/(2*step)

    return c_value

def apply_diff3d(phi, i, j, k, direction, type, step=1):
    assert len(phi.shape) == 3 # phi is 3d
    assert direction in ['X', 'Y', 'Z']
    assert type in ['forward', 'backward', 'central']
    if type == 'forward':
        if direction == 'X':
            c_value = (phi[i+step,j, k]-phi[i,j, k])/step
        elif direction == 'Y':
            c_value = (phi[i,j+step, k]-phi[i,j, k])/step
        elif direction == 'Z':
            c_value = (phi[i,j, k+step]-phi[i,j, k])/step

    elif type == 'backward':
        if direction == 'X':
            c_value = (phi[i,j, k]-phi[i-step,j, k])/step
        elif direction == 'Y':
            c_value = (phi[i,j, k]-phi[i,j-step, k])/step
        elif direction == 'Z':
            c_value = (phi[i,j, k]-phi[i,j, k-step])/step

    elif type == 'central':
        if direction == 'X':
            c_value = (phi[i+step,j, k]-phi[i-step,j, k])/(2*step)
        elif direction == 'Y':
            c_value = (phi[i,j+step, k]-phi[i,j-step, k])/(2*step)
        elif direction == 'Z':
            c_value = (phi[i,j, k+step]-phi[i,j, k-step])/(2*step)

    return c_value

def get_grad3d_batch_faster(phi):
    assert len(phi.shape) == 4
    batch_size = phi.shape[0]
    grad_list = []
    for i in range(batch_size):
        c_grad = get_grad3d_faster(phi[i])
        grad_list.append(c_grad)
    return grad_list


def get_grad3d_batch(phi):
    assert len(phi.shape) == 4
    batch_size = phi.shape[0]
    grad_list = []
    for i in range(batch_size):
        c_grad = get_grad3d(phi[i])
        grad_list.append(c_grad)
    return grad_list

def get_grad3d_batch_faster_torch(phi_t, device):
    assert len(phi_t.shape) == 4
    batch_size = phi_t.shape[0]
    grad_list = []
    for i in range(batch_size):
        c_grad = get_grad3d_faster_torch(phi_t[i], device)
        grad_list.append(c_grad)
    return grad_list

def get_grad3d_batch_torch(phi_t):
    assert len(phi_t.shape) == 4
    batch_size = phi_t.shape[0]
    grad_list = []
    for i in range(batch_size):
        c_grad = get_grad3d_torch(phi_t[i])
        grad_list.append(c_grad)
    return grad_list

def get_grad3d_torch(phi_t):
    assert len(phi_t.shape) == 3 
    grad_x = torch.zeros(phi_t.shape)
    grad_y = torch.zeros(phi_t.shape)
    grad_z = torch.zeros(phi_t.shape)
    X = phi_t.shape[0]
    Y = phi_t.shape[1]
    Z = phi_t.shape[1]
    assert X == Y and Y == Z 
    grid_size = X

    for i in range(X):
        for j in range(Y):
            for k in range(Z):

            # do gradient with respect to X ; i axis
                if i == 0: # forward difference
                    grad_x[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'X', 'forward')
                elif i == grid_size-1: # backward diff
                    grad_x[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'X', 'backward')
                else: # central diff
                    grad_x[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'X', 'central')

                # do gradient with respect to Y ; j axis
                if j == 0: # forward difference
                    grad_y[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Y', 'forward')
                elif j == grid_size-1: # backward diff
                    grad_y[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Y', 'backward')
                else: # central diff
                    grad_y[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Y', 'central')

                # do gradient with respect to Z ; k axis
                if k == 0: # forward difference
                    grad_z[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Z', 'forward')
                elif k == grid_size-1: # backward diff
                    grad_z[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Z', 'backward')
                else: # central diff
                    grad_z[i,j, k] = apply_diff3d_torch(phi_t, i, j, k, 'Z', 'central')

    return (grad_x, grad_y, grad_z)

def get_grad3d(phi):
    assert len(phi.shape) == 3 # phi is 3d
    grad_x = np.zeros(phi.shape)
    grad_y = np.zeros(phi.shape)
    grad_z = np.zeros(phi.shape)

    X = phi.shape[0]
    Y = phi.shape[1]
    Z = phi.shape[1]

    assert X == Y and Y == Z 

    grid_size = X

    for i in range(X):
        for j in range(Y):
            for k in range(Z):

            # do gradient with respect to X ; i axis
                if i == 0: # forward difference
                    grad_x[i,j, k] = apply_diff3d(phi, i, j, k, 'X', 'forward')
                elif i == grid_size-1: # backward diff
                    grad_x[i,j, k] = apply_diff3d(phi, i, j, k, 'X', 'backward')
                else: # central diff
                    grad_x[i,j, k] = apply_diff3d(phi, i, j, k, 'X', 'central')

                # do gradient with respect to Y ; j axis
                if j == 0: # forward difference
                    grad_y[i,j, k] = apply_diff3d(phi, i, j, k, 'Y', 'forward')
                elif j == grid_size-1: # backward diff
                    grad_y[i,j, k] = apply_diff3d(phi, i, j, k, 'Y', 'backward')
                else: # central diff
                    grad_y[i,j, k] = apply_diff3d(phi, i, j, k, 'Y', 'central')

                # do gradient with respect to Z ; k axis
                if k == 0: # forward difference
                    grad_z[i,j, k] = apply_diff3d(phi, i, j, k, 'Z', 'forward')
                elif k == grid_size-1: # backward diff
                    grad_z[i,j, k] = apply_diff3d(phi, i, j, k, 'Z', 'backward')
                else: # central diff
                    grad_z[i,j, k] = apply_diff3d(phi, i, j, k, 'Z', 'central')

    return (grad_x, grad_y, grad_z)

def apply_diff2d(phi, i, j, direction, type, step=1):
    assert len(phi.shape) == 2 # phi is 2d
    assert direction in ['X', 'Y']
    assert type in ['forward', 'backward', 'central']
    if type == 'forward':
        if direction == 'X':
            c_value = (phi[i+step,j]-phi[i,j])/step
        elif direction == 'Y':
            c_value = (phi[i,j+step]-phi[i,j])/step
    elif type == 'backward':
        if direction == 'X':
            c_value = (phi[i,j]-phi[i-step,j])/step
        elif direction == 'Y':
            c_value = (phi[i,j]-phi[i,j-step])/step
    elif type == 'central':
        if direction == 'X':
            c_value = (phi[i+step,j]-phi[i-step,j])/(2*step)
        elif direction == 'Y':
            c_value = (phi[i,j+step]-phi[i,j-step])/(2*step)
    return c_value

def get_grad3d_faster_torch(phi_t, device):
    assert len(phi_t.shape) == 3

    grad_t_x = torch.zeros(phi_t.shape, dtype=phi_t.dtype, device=device)
    grad_t_y = torch.zeros(phi_t.shape, dtype=phi_t.dtype, device=device)
    grad_t_z = torch.zeros(phi_t.shape, dtype=phi_t.dtype, device=device)

    X = phi_t.shape[0]
    Y = phi_t.shape[1]
    Z = phi_t.shape[2]

    grid_size = X

    # CENTRAL
    # get the slice:
    phi_t_slice_central = phi_t[1:-1, 1:-1, 1:-1]
    # X direction
    phi_t_slice_central_x_pos = phi_t[1+1:, :, :] # should be : -0
    phi_t_slice_central_x_neg = phi_t[1-1:-1-1, :, :]
    grad_t_x[1:-1, :, :] = (phi_t_slice_central_x_pos - phi_t_slice_central_x_neg)/2.0
    # Y direction
    phi_t_slice_central_y_pos = phi_t[:, 1+1:, :]    
    phi_t_slice_central_y_neg = phi_t[:, 1-1:-1-1, :]
    grad_t_y[:, 1:-1, :] = (phi_t_slice_central_y_pos - phi_t_slice_central_y_neg)/2.0    
    # Z direction
    phi_t_slice_central_z_pos = phi_t[:,:, 1+1:]    
    phi_t_slice_central_z_neg = phi_t[:,:, 1-1:-1-1]
    grad_t_z[:,:, 1:-1] = (phi_t_slice_central_z_pos - phi_t_slice_central_z_neg)/2.0    
    # FORWARD
    # X direction
    phi_t_slice_forward_x = phi_t[0,:,:]
    phi_t_slice_forward_x_pos = phi_t[0+1,:,:]
    grad_t_x[0,:,:] = phi_t_slice_forward_x_pos - phi_t_slice_forward_x
    # Y direction
    phi_t_slice_forward_y = phi_t[:,0,:]
    phi_t_slice_forward_y_pos = phi_t[:,0+1,:]
    grad_t_y[:,0,:] = phi_t_slice_forward_y_pos - phi_t_slice_forward_y
    # Z direction
    phi_t_slice_forward_z = phi_t[:,:,0]
    phi_t_slice_forward_z_pos = phi_t[:,:,0+1]
    grad_t_z[:,:,0] = phi_t_slice_forward_z_pos - phi_t_slice_forward_z
    # BACKWARD
    # X direction
    phi_t_slice_backward_x = phi_t[-1,:,:]
    phi_t_slice_backward_x_neg = phi_t[-1-1,:,:]
    grad_t_x[-1,:,:] = phi_t_slice_backward_x - phi_t_slice_backward_x_neg
    # Y direction
    phi_t_slice_backward_y = phi_t[:,-1,:]
    phi_t_slice_backward_y_neg = phi_t[:,-1-1,:]
    grad_t_y[:,-1,:] = phi_t_slice_backward_y - phi_t_slice_backward_y_neg
    # Z direction
    phi_t_slice_backward_z = phi_t[:,:,-1]
    phi_t_slice_backward_z_neg = phi_t[:,:,-1-1]
    grad_t_z[:,:,-1] = phi_t_slice_backward_z - phi_t_slice_backward_z_neg

    return (grad_t_x, grad_t_y, grad_t_z)


def get_grad3d_faster(phi):
    assert len(phi.shape) == 3

    grad_x = np.zeros(phi.shape)
    grad_y = np.zeros(phi.shape)
    grad_z = np.zeros(phi.shape)

    X = phi.shape[0]
    Y = phi.shape[1]
    Z = phi.shape[2]

    grid_size = X

    # CENTRAL
    # get the slice:
    phi_slice_central = phi[1:-1, 1:-1, 1:-1]
    # X direction
    phi_slice_central_x_pos = phi[1+1:, :, :] # should be : -0
    phi_slice_central_x_neg = phi[1-1:-1-1, :, :]
    grad_x[1:-1, :, :] = (phi_slice_central_x_pos - phi_slice_central_x_neg)/2.0
    # Y direction
    phi_slice_central_y_pos = phi[:, 1+1:, :]    
    phi_slice_central_y_neg = phi[:, 1-1:-1-1, :]
    grad_y[:, 1:-1, :] = (phi_slice_central_y_pos - phi_slice_central_y_neg)/2.0    
    # Z direction
    phi_slice_central_z_pos = phi[:,:, 1+1:]    
    phi_slice_central_z_neg = phi[:,:, 1-1:-1-1]
    grad_z[:,:, 1:-1] = (phi_slice_central_z_pos - phi_slice_central_z_neg)/2.0    
    # FORWARD
    # X direction
    phi_slice_forward_x = phi[0,:,:]
    phi_slice_forward_x_pos = phi[0+1,:,:]
    grad_x[0,:,:] = phi_slice_forward_x_pos - phi_slice_forward_x
    # Y direction
    phi_slice_forward_y = phi[:,0,:]
    phi_slice_forward_y_pos = phi[:,0+1,:]
    grad_y[:,0,:] = phi_slice_forward_y_pos - phi_slice_forward_y
    # Z direction
    phi_slice_forward_z = phi[:,:,0]
    phi_slice_forward_z_pos = phi[:,:,0+1]
    grad_z[:,:,0] = phi_slice_forward_z_pos - phi_slice_forward_z
    # BACKWARD
    # X direction
    phi_slice_backward_x = phi[-1,:,:]
    phi_slice_backward_x_neg = phi[-1-1,:,:]
    grad_x[-1,:,:] = phi_slice_backward_x - phi_slice_backward_x_neg
    # Y direction
    phi_slice_backward_y = phi[:,-1,:]
    phi_slice_backward_y_neg = phi[:,-1-1,:]
    grad_y[:,-1,:] = phi_slice_backward_y - phi_slice_backward_y_neg
    # Z direction
    phi_slice_backward_z = phi[:,:,-1]
    phi_slice_backward_z_neg = phi[:,:,-1-1]
    grad_z[:,:,-1] = phi_slice_backward_z - phi_slice_backward_z_neg

    return (grad_x, grad_y, grad_z)


def get_grad2d_fast(phi):
    grad_x = np.zeros(phi.shape)
    grad_y = np.zeros(phi.shape)
    X = phi.shape[0]
    Y = phi.shape[1]
    grid_size = X
    # CENTRAL
    # get the slice:
    phi_slice_central = phi[1:-1, 1:-1]
    # X direction
    phi_slice_central_x_pos = phi[1+1:, :] # should be : -0
    phi_slice_central_x_neg = phi[1-1:-1-1, :]
    grad_x[1:-1, :] = (phi_slice_central_x_pos - phi_slice_central_x_neg)/2.0
    # Y direction
    phi_slice_central_y_pos = phi[:, 1+1:]    
    phi_slice_central_y_neg = phi[:, 1-1:-1-1]
    grad_y[:, 1:-1] = (phi_slice_central_y_pos - phi_slice_central_y_neg)/2.0    
    # FORWARD
    # X direction
    phi_slice_forward_x = phi[0,:]
    phi_slice_forward_x_pos = phi[0+1,:]
    grad_x[0,:] = phi_slice_forward_x_pos - phi_slice_forward_x
    # Y direction
    phi_slice_forward_y = phi[:,0]
    phi_slice_forward_y_pos = phi[:,0+1]
    grad_y[:,0] = phi_slice_forward_y_pos - phi_slice_forward_y
    # BACKWARD
    # X direction
    phi_slice_backward_x = phi[-1,:]
    phi_slice_backward_x_neg = phi[-1-1,:]
    grad_x[-1,:] = phi_slice_backward_x - phi_slice_backward_x_neg
    # Y direction
    phi_slice_backward_y = phi[:,-1]
    phi_slice_backward_y_neg = phi[:,-1-1]
    grad_y[:,-1] = phi_slice_backward_y - phi_slice_backward_y_neg

    return (grad_x, grad_y)

def get_grad2d(phi):
    grad_x = np.zeros(phi.shape)
    grad_y = np.zeros(phi.shape)
    X = phi.shape[0]
    Y = phi.shape[1]
    grid_size = X
    for i in range(X):
        for j in range(Y):
        # do gradient with respect to X ; i axis
            if i == 0: # forward difference
                grad_x[i,j] = apply_diff2d(phi, i, j, 'X', 'forward')
            elif i == grid_size-1: # backward diff
                grad_x[i,j] = apply_diff2d(phi, i, j, 'X', 'backward')
            else: # central diff
                grad_x[i,j] = apply_diff2d(phi, i, j, 'X', 'central')

            # do gradient with respect to Y ; j axis
            if j == 0: # forward difference
                grad_y[i,j] = apply_diff2d(phi, i, j, 'Y', 'forward')
            elif j == grid_size-1: # backward diff
                grad_y[i,j] = apply_diff2d(phi, i, j, 'Y', 'backward')
            else: # central diff
                grad_y[i,j] = apply_diff2d(phi, i, j, 'Y', 'central')
    return (grad_x, grad_y)

def test_grad_2d(grid_size=32):
    set_trace()

    phi = np.random.rand(grid_size, grid_size)

    my_grad = get_grad2d(phi)
    np_grad = np.gradient(phi)
   
    assert (my_grad[0] - np_grad[0]).sum() == 0
    assert (my_grad[1] - np_grad[1]).sum() == 0
    print('2D test successfull.')
   
def test_grad_3d(grid_size=32):
    set_trace()

    phi = np.random.rand(grid_size, grid_size, grid_size)

    my_grad = get_grad3d(phi)
    np_grad = np.gradient(phi)
    
    assert (my_grad[0] - np_grad[0]).sum() == 0
    assert (my_grad[1] - np_grad[1]).sum() == 0
    assert (my_grad[2] - np_grad[2]).sum() == 0

    print('3D test successfull.')

def test_get_grad3d_faster_torch(grid_size=32):

    phi = np.random.rand(grid_size, grid_size, grid_size)
    phi_t = torch.from_numpy(phi)

    np_grad = get_grad3d_faster(phi)
    torch_grad = get_grad3d_faster_torch(phi_t)

    assert (np_grad[0]-torch_grad[0].cpu().detach().numpy()).sum() == 0
    assert (np_grad[1]-torch_grad[1].cpu().detach().numpy()).sum() == 0
    assert (np_grad[2]-torch_grad[2].cpu().detach().numpy()).sum() == 0

    print('get_grad3d faster in torch match those in numpy')

def test_get_grad3d_torch(grid_size=32):

    phi = np.random.rand(grid_size, grid_size, grid_size)
    phi_t = torch.from_numpy(phi)

    np_grad = get_grad3d(phi)
    torch_grad = get_grad3d_torch(phi_t)

    set_trace()
    print('compare the two')

def test_get_norm3d_batch_faster_torch(batch_size=16, grid_size=32):
    phi = np.random.rand(batch_size, grid_size, grid_size, grid_size)
    phi_t = torch.from_numpy(phi)

    grad_list_np = get_grad3d_batch_faster(phi)
    grad_list_torch = get_grad3d_batch_faster_torch(phi_t)


def test_get_norm3d_batch_torch(batch_size=16, grid_size=32):
    phi = np.random.rand(batch_size, grid_size, grid_size, grid_size)
    phi_t = torch.from_numpy(phi)

    grad_list_np = get_grad3d_batch(phi)
    grad_list_torch = get_grad3d_batch_torch(phi_t)

def test_get_grad2d(grid_size=32):
    phi = np.random.rand(grid_size, grid_size)

    grad_np = np.gradient(phi)

    my_grad = get_grad2d_fast(phi)

    assert (my_grad[0]-grad_np[0]).sum() == 0
    assert (my_grad[1]-grad_np[1]).sum() == 0

    print('get_grad 2d, fast: all good')

def test_get_grad3d(grid_size=32):
    phi = np.random.rand(grid_size, grid_size, grid_size)

    grad_np = np.gradient(phi)

    my_grad = get_grad3d_fast(phi)

    assert (my_grad[0]-grad_np[0]).sum() == 0
    assert (my_grad[1]-grad_np[1]).sum() == 0
    assert (my_grad[2]-grad_np[2]).sum() == 0


    print('get_grad 3d, fast: all good')



    


