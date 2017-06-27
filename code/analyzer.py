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
#anderson = pf.open("anderson.fits")[0].data
#X = np.exp(np.loadtxt("noregwfc_mean_iter_2.txt")).reshape(100,100)
#X1 = np.exp(np.loadtxt("noregwfc_mean_iter_1.txt")).reshape(100,100)
fl=1e-5
f = .05
g = .01 
H = 3
X = np.exp(np.loadtxt("superb_wfc_mean_iter_5_nfl.txt"))


Z = (X).reshape(25*H,25*H)

print Z.sum()
import c3

x0 , y0 = c3.find_centroid(Z)
print x0, y0
import scipy
from scipy import ndimage
Z2 =  scipy.ndimage.interpolation.shift(Z, [-x0,-y0], output=None, order=3, mode='nearest', prefilter=True)

k = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]])

Z3 = ndimage.convolve(Z2, k, mode='nearest')
Z3 = Z3 * Z.sum() /Z3.sum()

#F = np.loadtxt("noregwfc_flux_iter_22.txt")
#B = np.loadtxt("noregwfc_background_iter_22.txt")
#plt.hist(F , bins = 10)
#plt.show()

data = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]
mask = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]
cx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")[0:120]
cy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")[0:120]

F = np.loadtxt("superb_wfc_flux_iter_5_nfl.txt")#.reshape(75,75)
B = np.loadtxt("superb_wfc_bkg_iter_5_nfl.txt")#.reshape(75,75)

#MS = h5py.File('masked_samplerx3.hdf5','r')
#MD = h5py.File('masked_data.hdf5','r')
#masked_samplers = MS["masked_samplerx3"]#[random]
#masked_data = MD["masked_data"]#[random]

#res = []
#brightness = []
import sampler

def lnlike(theta , p):
         x, y = theta
         Kp = sampler.imatrix_new(25, H, x, y)
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = F[p]*np.dot(fl+X,Kp) + B[p]
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
         chi = (data[p,:] - model)**2./(f+g*np.abs(model)) + np.log(f+g*np.abs(model))
         maskp = mask[p,:]
         chi = chi[maskp==0]
         return np.sum(chi)

import scipy.optimize as op
nll = lambda *args: lnlike(*args)


import sampler
for p in range(1,120):


         result = op.minimize(nll, [cx[p], cy[p]], args=(p))
         x_ml, y_ml, = result["x"] 

         #X = Z2.flatten()
         #Kp = sampler.imatrix_new(25, H, cx[p], cy[p])
         Kp = sampler.imatrix_new(25, H, x_ml, y_ml)
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = F[p]*np.dot(fl+X,Kp) + B[p]
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
         chi = (data[p,:] - model)/(f+g*np.abs(model))**.5
         chi = chi.reshape(25,25)
         maskp = mask[p,:].reshape(25,25)
	 res[maskp!=0]=np.nan
         chi[maskp!=0]=np.nan
         resi[maskp!=0]=np.nan
         mi = min(data[p].min(), model.min())
         ma = max(data[p].max(), model.max())
         
         	
         plt.figure(figsize=(10,10))
         
         plt.subplot(2,3,1)
         plt.set_cmap('RdBu')
	 ax = plt.gca()
    	 im = ax.imshow(data[p,:].reshape(25,25), interpolation = "None" , origin = "lower",norm=LogNorm())
         plt.title("SR PSF Model")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(2,3,2)

	 ax = plt.gca()
    	 im = ax.imshow(model.reshape(25,25) , interpolation = "None" , origin = "lower",norm=LogNorm())
         plt.title("Model")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(2,3,3)
	 ax = plt.gca()
    	 im = ax.imshow(Z2 , interpolation = "None" , origin = "lower",norm=LogNorm())
         plt.title("SR PSF Model")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(2,3,5)

	 ax = plt.gca()
    	 im = ax.imshow(resi.reshape(25,25) , interpolation = "None" , origin = "lower")
         plt.title("Residual")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(2,3,4)

	 ax = plt.gca()
    	 im = ax.imshow(chi.reshape(25,25) , interpolation =  "None", origin = "lower")
	 plt.title("chi")         
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         #plt.subplot(3,3,6)

	 #ax = plt.gca()
    	 #im = ax.imshow(res.reshape(25,25) , interpolation =  "None", origin = "lower")
	 #plt.title("(residual)x100/data")         
         #divider = make_axes_locatable(ax)
	 #cax = divider.append_axes("right", size="5%", pad=0.2)
         #plt.colorbar(im , cax = cax)
	 #ax.set_xticks(())
	 #ax.set_yticks(())

         plt.subplot(2,3,6)

	 ax = plt.gca()
    	 im = ax.imshow(maskp.reshape(25,25) , interpolation =  "None", origin = "lower")
	 plt.title("Data Quality")         
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())
         
         plt.tight_layout()
         plt.show()
         
         #plt.savefig("all_newstars_%d.png"%(p))
         plt.close() 
