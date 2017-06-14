import numpy as np
import scipy.optimize as op

def fit_single_patch(data, mask , psf, theta, floor, gain):
    """
       Inputs:
       data  = patch,
       mask  = True for healthy pixels, False for flagged pixels
       psf   = psf model rendered at the data grid
       theta = [old_flux, old_bkg],
       where:
               old_flux = current flux estimate for a patch
               old_back = current bkg estimate for the patch
       floor = floor variance of the noise model
       gain  = gain of the noise model
       Outputs:
       (non-regularized) NLL of the patch, and derivative 
       of (non-regularized) NLL w.r.t flux and bkg of the patch.
       Note that the regularization is independent of F, B.
    """

    var = floor + gain * np.abs(theta[1] * psf[mask] + theta[0])
    A = np.ones((var.size, 2))
    A[:, 1] = psf[mask]
    model = np.dot(A, theta) #masked model
    res  = data[mask] - model
    func = 0.5*np.sum(((res)**2.)/var) + 0.5*np.sum(np.log(var))
    Grad = 0.5*gain*(1./var - res*res/(var*var)) - (res*res)/var 
    grad = np.sum(Grad[:, None]*A , axis = 1) 

    return func , grad

# different version of the function defined above. We'll see which one is faster:

def v2_fit_single_patch(theta, masked_data, masked_psf, floor, gain):

    """
       Inputs:

       theta = [old_bkg, old_flux],
       where:
              old_flux = current flux estimate for a patch
              old_back = current bkg estimate for the patch       

       masked_data = patch with flagged pixels masked out
       masked_psf = psf model rendered at the data grid
                    masked out
                    where pixels are flagged
       
       floor = floor variance of the noise model
       gain  = gain of the noise model

       Outputs:

       (non-regularized) NLL of the patch, and derivative 
       of (non-regularized) NLL w.r.t flux and bkg of the patch.
       Note that the regularization is independent of F, B.
    """

    var = floor + gain * np.abs(theta[1]* masked_psf + theta[0])
    A = np.ones((var.size, 2))
    A[:, 1] = masked_psf
    model = np.dot(A , theta) #masked model
    res  = masked_data - model
    func = 0.5*np.sum(((res)**2.)/var) + 0.5*np.sum(np.log(var))
    Grad = 0.5*gain*(1./var - res*res/(var*var)) - (res*res)/var 
    grad = np.sum(Grad[:, None]*A , axis = 0)  #this could unnecessarily slow down the code!! 
    print func.shape , grad.shape
    return func , grad

##### this is probably the fastest version!

def v3_fit_single_patch(theta, masked_data, masked_psf, floor, gain):

    """
       Inputs:
    
       theta = [old_flux, old_bkg],
       where:
              old_flux = current flux estimate for a patch
              old_back = current bkg estimate for the patch

       masked_data = patch with flagged pixels masked out
       masked_psf = psf model rendered at the data grid
                    masked out
                    where pixels are flagged
       
       floor = floor variance of the noise model
       gain  = gain of the noise model

       Outputs:

       (non-regularized) NLL of the patch, and derivative 
       of (non-regularized) NLL w.r.t flux and bkg of the patch.
       Note that the regularization is independent of F, B.
    """
    grad = np.zeros((2))

    var = floor + gain * np.abs(theta[1] * masked_psf + theta[0])
    model = theta[0] + theta[1]*masked_psf #masked model
    res  = masked_data - model
    func = 0.5*np.sum(((res)**2.)/var) + 0.5*np.sum(np.log(var))
    
    gain_term_b = - (gain/2.)*(res**2./var**2.) + (gain/2.)*(var**-1.)
    #gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid   
    grad[0] = -1.*np.sum(res/var) + np.sum(gain_term_b)


    gain_term_f = (gain/2.)*masked_psf*(var**-1. - res**2./var**2.)
    #gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid
    grad[1] = np.sum(-1.*res*masked_psf/var) + np.sum(gain_term_f)    

    return func, grad
def v4_fit_single_patch(theta, data, mask, flux, bkg, \
                        alpha, mean, floor, gain):

    """
       Inputs:
    
       theta = [old_cx, old_cy],
       where:
              old_cx = current cx estimate for a patch
              old_cy = current cy estimate for the patch

       masked_data = patch with flagged pixels masked out
       masked_psf = psf model rendered at the data grid
                    masked out
                    where pixels are flagged
       
       floor = floor variance of the noise model
       gain  = gain of the noise model

       Outputs:

       (non-regularized) NLL of the patch,
       Note that the regularization is independent of F, B.
    """
    Dx , Dy = theta[0], theta[1]
    h = 1./25
    x2 = np.arange(0, 25)*h + 0.5*h - Dx*h
    y2 = np.arange(0, 25)*h + 0.5*h - Dy*h
    x2, y2 = np.meshgrid(x2, y2, indexing = "ij")
    hh = 1./101
    x = np.arange(0, 101)*hh + 0.5*hh
    y = np.arange(0, 101)*hh + 0.5*hh
    x, y = np.meshgrid(x, y, indexing = "ij")
    samples2 = np.vstack((x2.flatten(), y2.flatten())).T #data_grid
    samples = np.vstack((x.flatten(), y.flatten())).T    #sr grid
    Kxsx = self.kernel.value(samples2, samples)
    model = flux*(np.dot(Kxsx, alpha) + mean) + bkg

    data = data[mask!=0]
    model = model[mask!=0]
    res = data - model
    var = floor + gain*np.abs(model) 
    func = np.sum(res**2. / var + np.log(var))

    return func
