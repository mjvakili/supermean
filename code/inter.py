from __future__ import division

from numpy import arange, array, linspace, ones, zeros
from scipy.linalg import solve_banded
import interp as _spline

def cal_coefs(a, b, y, c=None, alpha=0, beta=0):
    '''
    Return spline coefficients 

    Parameters
    ----------
    a : float
        lower bound of the grid.
    b : float
        upper bound of the grid.
    y : ndarray
        actual function value at grid points.
    c : (y.shape[0] + 2, ) ndarray, optional
        ndarry to be written
    alpha : float
        Second-order derivative at a. Default is 0.
    beta : float
        Second-order derivative at b. Default is 0.

    Returns
    -------
    out : ndarray
        Array of coefficients.
    '''
    n = y.shape[0] - 1
    h = (b - a)/n

    if c is None:
        c = zeros((n + 3, ))
        ifreturn = True
    else:
        assert(c.shape[0] == n + 3)
        ifreturn = False

    c[1] = 1/6 * (y[0] - (alpha * h**2)/6)
    c[n + 1] = 1/6 * (y[n] - (beta * h**2)/6)

    # ab matrix here is just compressed banded matrix
    ab = ones((3, n - 1))
    ab[0, 0] = 0
    ab[1, :] = 4
    ab[-1, -1] = 0

    B = y[1:-1].copy()
    B[0] -= c[1]
    B[-1] -=  c[n + 1]

    c[2:-2] = solve_banded((1, 1), ab, B)

    c[0]  = alpha * h**2/6 + 2 * c[1]  - c[2]
    c[-1] = beta  * h**2/6 + 2 * c[-2] - c[-3]

    if ifreturn:
        return(c)

# aliases
interpolate = _spline.interpolate
interpolate_2d = _spline.interpolate_2d
import numpy as np

if __name__ == '__main__':

    # 1D interpolation
    f = lambda x: np.cos(x-.3)

    a = -1.
    b = 1.
    n = 10  # there are n + 1 grid points (0,..., n)

    h = (b - a) / n
    grid = arange(n + 1) * h + a

    y = f(grid)
    #print y , y.shape
    alpha = 0
    beta = 0

    c = cal_coefs(a, b, y)
    print c , c.shape

    grid_hat = linspace(-1.+.3, 1.+.3, 10)
    #print interpolate(0 , a , b , c)

    fhat = array([interpolate(x, a, b, c) for x in grid_hat])
    
    
    from matplotlib import pyplot as plt
    line_actual = plt.plot(grid_hat, f(grid_hat)-fhat, '*',label='SR model')
    line_approx = plt.plot(grid_hat, fhat, '.', label='model sampled at LR data grid')
    #plt.setp(line_actual, linewidth=1, linestyle='--')
    #plt.setp(line_approx, linewidth=2, linestyle='-.')
    plt.legend()
    plt.show()

if (2>1): #not "2D interpolation":
    # 2D interpolation
    f2d = lambda x, z: np.exp(-1.*(x-.6)**2. -1.* (z-.5)**2.)#10*(x-.5)**3. + 2*(x-.5)*z + z ** 0.5 - 3 * (z-.5)**4.
    
    a1, a2 = 0., 0.
    b1, b2 = 1., 1.
    n1, n2 = 100, 100  # n + 1 grid points (0,..., n)

    h1, h2 = (b1 - a1)/n1, (b2 - a2)/n2
    grid_x = arange(n1 + 1) * h1 + a1
    grid_z = arange(n2 + 1) * h2 + a2

    y = zeros((n1 + 1, n2 + 1))
    
    for i, x in enumerate(grid_x):
        for j, z in enumerate(grid_z):
            y[i, j] = f2d(x, z)

    alpha = 0
    beta = 0

    c_tmp = zeros((n1 + 3, n2 + 1))
    cal_coefs(a1, b1, y, c_tmp , alpha = 0 , beta = 0)

    c = zeros((n1 + 3, n2 + 3))
    # NOTE: here you have to pass c_tmp.T and c.T
    cal_coefs(a2, b2, c_tmp.T, c.T , alpha = 0 , beta = 0)

    #defining the new grid of points to which we want to interpolate the function to:

    a1 , a2 = 0. , 0.
    b1 , b2 = 1. , 1.
    nn1 , nn2 = 10, 10
    hh1, hh2 = (b1 - a1)/nn1, (b2 - a2)/nn2
    
    newgrid_x = arange(nn1 + 1) * hh1 + a1  
    newgrid_z = arange(nn2 + 1) * hh2 + a2 
   
    #print grid_x , newgrid_x

    fhat = zeros((nn1 + 1, nn2 + 1))
    for i, x in enumerate(newgrid_x):
        for j, z in enumerate(newgrid_z):
            fhat[i, j] = interpolate_2d(x, z, a1, b1, a2, b2, c)

    #fhat = interpolate_2d(np.meshfrid(x,y) , a1 , b1 , a2 , b2 , c)  #function evaluation at test points 
 
    real_val = zeros((nn1 + 1, nn2 + 1))
    for i, x in enumerate(newgrid_x):
        for j, z in enumerate(newgrid_z):
            real_val[i, j] = f2d(x,z)#(x+3.*hh1 , z+1*hh2)
    from scipy import ndimage
    real_val = ndimage.interpolation.shift(y , (-5,-5))
    mi = min(fhat.min(),real_val.min(),y.min())
    ma = max(fhat.max(),real_val.max(),y.max())

    import matplotlib.pyplot as plt
    plt.subplot(4,1,1)
    plt.imshow(y , interpolation="None" , origin="lower" , vmin = mi , vmax = ma)
    plt.colorbar()
    
    plt.subplot(4,1,2)
    plt.imshow(real_val , interpolation = "None" , origin = "lower" , vmin = mi , vmax = ma)
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.imshow(fhat , interpolation =  "None" , origin = "lower" , vmin = mi , vmax = ma)
    plt.colorbar()

    plt.subplot(4,1,4)
    plt.imshow(real_val , interpolation =  "None" , origin = "lower")
    plt.colorbar()

    plt.show()
    """
  from mayavi import mlab

    def draw_3d(grid_x, grid_y, fval, title='pi'):
        mlab.figure()
        mlab.surf(grid_x, grid_y, fval)#, warp_scale="auto")
        mlab.axes(xlabel='x', ylabel='z', zlabel=title)
        mlab.orientation_axes(xlabel='x', ylabel='z', zlabel=title)
        mlab.title(title)

    draw_3d(grid_x, grid_z, fhat, title='interpolated')
    draw_3d(grid_x, grid_z, real_val, title='real')
    mlab.show()"""
