import numpy as np
import scipy.optimize as op
import sampler

def objfunction(theta, masked_data, mask , sr_psf, flux, bkg, floor, gain, fl):

    """
       Inputs:
    
       theta = [old_cx, old_cy],
       where:
              old_cx = current sub-pixel shift of the center of star in x at the naitive pixel resolution
              old_cy = current sub-pixel shift of the center of star in y at the naitive pixel resolution

       masked_data = patch with flagged pixels masked out
       mask = flagged pixels
       sr_psf  = super resolution psf model
       floor = floor variance of the noise model
       gain  = gain of the noise model
       fl = floor level of the psf. to be added to the 
            super-resolution psf model while evaluating model
       
       Outputs:

       (non-regularized) NLL of the patch.
       Note that the regularization term is 
       only a function of superresolution psf, 
       so we do not include it here.
    """
    Kp = sampler.imatrix_new(25, H, theta[0], theta[1])[: , mask]      # masked sampler
    model = flux*np.dot(sr_psf + fl , Kp) + bkg                        # maksed model
    var = floor + gain * np.abs(model)
    res  = masked_data - model
    func = 0.5*np.sum(((res)**2.)/var) + 0.5*np.sum(np.log(var))
    return func

def fit(theta , masked_data, mask , sr_psf, flux, bkg, floor, gain, fl):
    
    nll = lambda *args: objfunction(*args)
    results = op.fmin(nll , [theta[0] , theta[1]] , args = (masked_data, mask , sr_psf, flux, bkg, floor, gain, fl), disp = False)

    return results[0] , results[1]
