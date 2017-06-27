import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import sampler_new

import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
rc('text', usetex=True)
from matplotlib import cm
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#b  = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]

fl= 1e-5
f = .05
g = .01 
H = 3

X = np.exp(np.loadtxt("superb_wfc_mean_iter_5_nfl.txt"))


data = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]
mask = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]

mask[mask == 0] = -1
mask[mask > 0] = False
mask[mask < 0] = True
mask = mask == True

cx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")[0:120]
cy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")[0:120]

F = np.loadtxt("superb_wfc_flux_iter_5_nfl.txt")#.reshape(75,75)
B = np.loadtxt("superb_wfc_bkg_iter_5_nfl.txt")#.reshape(75,75)

MS = h5py.File('masked_samplerx3.hdf5','r')
MD = h5py.File('masked_data.hdf5','r')
masked_samplers = MS["masked_samplerx3"]#[random]
masked_data = MD["masked_data"]#[random]
import sampler
import time

import scipy.optimize as op

def function(theta, masked_data, mask , sr_psf, flux, bkg, floor, gain):

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

       Outputs:

       (non-regularized) NLL of the patch.
       Note that the regularization term is 
       only a function of superresolution psf, 
       so we do not include it here.
    """
    Kp = sampler.imatrix_new(25, H, theta[0], theta[1])[: , mask] # masked sampler
    model = flux*np.dot(sr_psf , Kp) + bkg            # maksed model
    var = floor + gain * np.abs(model)
    res  = masked_data - model
    func = 0.5*np.sum(((res)**2.)/var) + 0.5*np.sum(np.log(var))
    return func


def fit(theta , masked_data, mask , sr_psf, flux, bkg, floor, gain):
    
    #size = I.shape[0]
    #x = np.linspace(0.5, size - .5, size)
    #y = x[:,np.newaxis]
    #x0_true = I.shape[0]/2.
    #y0_true = I.shape[0]/2.
    #flux0_true = 1.
    #size = I.shape[0]
    #x = np.linspace(0.5, size - .5, size)
    #y = x[:,np.newaxis]
    nll = lambda *args: function(*args)
    results = op.fmin(nll , [theta[0] , theta[1]] , args = (masked_data, mask , sr_psf, flux, bkg, floor, gain))# , disp = False)
    #flux0_ml , x0_ml , y0_ml = results["x"]

    return results[0] , results[1]



for p in range(0,2):
	 a = time.time()
         print cx[p] , cy[p]
         x, y = fit((cx[p] , cy[p]) , data[p,mask[p]], mask[p] , X, F[p], B[p], 0.05, 0.01)
         
         Kp = sampler.imatrix_new(25, H, x,y)
	 print time.time() - a
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = F[p]*np.dot(fl+X,Kp) + B[p]
        
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
         
         chi = (data[p,:] - model)/(f+g*np.abs(model))**.5
         chi = chi.reshape(25,25)
         maskp = mask[p,:].reshape(25,25)
	 #res[maskp!=0]  = np.nan
         chi[maskp!=True]  = np.nan
         #resi[maskp!=0] = np.nan
         mi = min(data[p , mask[p]].min(), model[mask[p]].min())
         ma = max(data[p , mask[p]].max(), model[mask[p]].max())
         
         	
         plt.figure(figsize=(10,10))
         plt.subplot(1,3,1)
         plt.set_cmap('RdBu')
      	 ax = plt.gca()
    	 im = ax.imshow(data[p].reshape(25,25) , interpolation="None" , origin="lower" , norm = LogNorm(), vmin = mi , vmax = ma)
         plt.title("Data")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(1,3,2)

	 ax = plt.gca()
    	 im = ax.imshow(model.reshape(25,25) , interpolation = "None" , origin = "lower",norm=LogNorm(), vmin = mi , vmax = ma)
         plt.title("Model")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(1,3,3)
	 ax = plt.gca()
    	 im = ax.imshow(chi.reshape(25,25) , interpolation = "None" , origin = "lower")
         plt.title("Chi")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.show()
         plt.close()
         
