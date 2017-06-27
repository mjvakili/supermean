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
Z2 =  scipy.ndimage.interpolation.shift(Z, [x0,y0], output=None, order=3, mode='nearest', prefilter=True)

from scipy.ndimage.morphology import binary_dilation as grow_mask


#Z2 = ndimage.convolve(Z2, k, mode='nearest')
#Z2 = Z2 * Z.sum() /Z2.sum()


data = pf.open("f160w_25_457_557_457_557_pixels.fits")[0].data
mask = pf.open("f160w_25_457_557_457_557_dq.fits")[0].data


#dat = pf.open("wfc3_f160w_clean_central100.fits")
#data = dat[0].data[0:120,:]
#mask = dat[1].data[0:120,:]

import sampler

q = 1.0

def lnlike(theta , p):
        
         
         cx, cy , F, B = theta
         Kp = sampler.imatrix_new(25, H, cx, cy)
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = F*np.dot(fl+X,Kp) + B
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
        
         chisq = (data[p,:] - model)**2./(f+g*np.abs(model)) + np.log(f+g*np.abs(model))
         chi= (data[p,:] - model)/(f+g*np.abs(model)+q*(model)**2.)**0.5
         maskp = mask[p,:]
         
         chisq = chisq.reshape(25,25)
         chi = chi.reshape(25,25)
         maskp = maskp.reshape(25,25) 
         

         mast = np.abs(chi)**2. > 3      
         mast = ndimage.binary_dilation(mast) 
         mast = ndimage.binary_dilation(mast) 
         bad =  maskp != 0
         
         unhealthy = bad * mast      

         chisq = chisq[unhealthy == False]

         return np.sum(chisq)

import scipy.optimize as op
nll = lambda *args: lnlike(*args)


for p in range(1750,1850):

    	 cx0 , cy0 = 0. , 0.
    	 b0 = np.median(data[p])
    	 datasq = data[p].reshape(25,25) - b0
    	 f0 = np.sum(datasq[11:14,11:14])

    	 print cx0 , cy0 , f0 , b0

    	 #result = op.minimize(nll, [cx0, cy0 , f0, b0], args=(p)) #, method = 'TNC', bounds = ((-0.5,0.5),(-0.5,0.5),(0.,None),(0.,None)))
    	 result = op.minimize(nll, [cx0, cy0 , f0, b0], args=(p), method = 'TNC', bounds = ((-0.5,0.5),(-0.5,0.5),(10.0,None),(0.,None)))
    	 x_ml, y_ml, f_ml , b_ml = result["x"] 
    	 print x_ml , y_ml , f_ml, b_ml


         Kp = sampler.imatrix_new(25, H, x_ml, y_ml)
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = f_ml*np.dot(fl+X,Kp) + b_ml
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
         chi = (data[p,:] - model)/(f+g*np.abs(model))**.5
         chi_clip = (data[p,:] - model)/(f+g*np.abs(model)+q*(model)**2.)**.5
         chi = chi.reshape(25,25)
         chi_clip = chi_clip.reshape(25,25)
         maskp = mask[p,:].reshape(25,25)
	 res[maskp!=0]=np.nan
         chi[maskp!=0]=np.nan
         resi[maskp!=0]=np.nan
         mi = min(data[p].min(), model.min())
         ma = max(data[p].max(), model.max())
         from matplotlib import gridspec 
         	
         fig = plt.figure(figsize=(10,10))

         gs = gridspec.GridSpec(2,2) 
         ax = plt.subplot(gs[0,0])

         plt.set_cmap('RdBu')
    	 im = ax.imshow(data[p,:].reshape(25,25), interpolation = "None" , origin = "lower",norm=LogNorm() , vmin = mi , vmax = ma)
         plt.title("log(Data)")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())


         ax = plt.subplot(gs[0,1])
    	 im = ax.imshow(model.reshape(25,25) , interpolation = "None" , origin = "lower",norm=LogNorm() , vmin = mi , vmax = ma)
         plt.title("log(Model)")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())

	 ax.set_xticks(())
	 ax.set_yticks(())

         ax = plt.subplot(gs[1,0])

         mast = np.abs(chi_clip)**2. > 3      
         mast = ndimage.binary_dilation(mast) 
         mast = ndimage.binary_dilation(mast) 

         chi[mast == True ] = np.nan
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

         ax = plt.subplot(gs[1,1])

         maskp[maskp!=0] = 1
         
         mast = np.abs(chi_clip)**2. > 3      
         mast = ndimage.binary_dilation(mast) 
         mast = ndimage.binary_dilation(mast) 
                  

         maskp[mast == True] = 2
    	 im = ax.imshow(maskp.reshape(25,25) , interpolation =  "None", origin = "lower")
	 plt.title(r"0 : MAST, 1 : FLAG, 2 : $\chi^{2}_{clip}>3 $")         
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())
         
         #plt.tight_layout()
         #plt.show()
         fig.savefig("example"+str(p)+".pdf", bbox_inches='tight')
         plt.close() 
         #plt.savefig("all_newstars_%d.png"%(p))
