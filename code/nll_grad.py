import numpy as np
import h5py
import time

def nll_grad_X(X, F, B , f, g, p):

    """
    inputs: 
    lnX : super-resolution PSF shared for all data
    F : list of flux values of the stars in different patches
    B : list of backgrounds of patches
    K : Array of sampling matrices. Each matrix samples
                      the PSF model at the data grid
    #fl: floor of the PSF model
    f : floor variance of the pixel-noise
    g : gain (variance = f + g*model)
            
                
    outputs:
    func = -2.*log(L), scalar
    dfunc/dX         , vector: has the same dimension as X
    """
    MS = h5py.File('masked_sampler.hdf5','r')
    MD = h5py.File('masked_data.hdf5','r')
    masked_samplers = MS["masked_sampler"]
    masked_data = MD["masked_data"]

    grad = np.zeros_like(X)               
    func  = 0.0
    Kp = masked_samplers[str(p)]
    Y = np.array(masked_data[str(p)])
    modelp = F[p]*(np.dot(Kp, X-np.mean(X))+np.mean(x))+B[p]

    resp = Y - modelp
    varp = f + g*np.abs(modelp)
    fctr = (-2.*resp*varp - g*resp*resp + g*varp)*F[p]/(varp*varp)
    dfi = np.einsum('j, ji' , fctr , Kp)[1:]
    df0 = np.sum(fctr)
    df = np.hstack([[df0],dfi])
    grad[0] = np.mean(df)
    grad[1:] = df[1:] - np.mean(df)
    #gainp = (g/2.)*(1./varp - ep*ep/(varp*varp))
    #gradp = Kp*(ep/varp - gainp)[None , :]
    #gradp = X[:,None]*Kp*(ep/varp - gainp)[None,:]
    #Gradp = np.sum(gradp, axis = 1) 
    #grad = -1.*F[p]*Gradp
    func = np.sum(((resp)**2.)/varp) + np.sum(np.log(varp))
    MS.close()
    MD.close()
    
    return func, grad
