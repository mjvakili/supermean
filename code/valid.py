
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
"""
x0 , y0 = c3.find_centroid(Z)
print x0, y0
import scipy
from scipy import ndimage
Z2 =  scipy.ndimage.interpolation.shift(Z, [x0,y0], output=None, order=3, mode='nearest', prefilter=True)

k = np.array([[True, False, True],
              [False, False, False],
              [True, False, True]])
"""
Z2 = ndimage.convolve(Z2, k, mode='nearest')
Z2 = Z2 * Z.sum() /Z2.sum()


data = pf.open("f160w_25_457_557_457_557_pixels.fits")[0].data
mask = pf.open("f160w_25_457_557_457_557_dq.fits")[0].data


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
         
         bol2 = maskp==0 

         mast = np.where(chi>3)       
         
         #drunk = np.zeros((mast[0].shape[0],2), dtype = int)
         #drunk[:,0] = mast[0]
         #drunk[:,1] = mast[1]
         	
           
         #drunk = np.append(drunk , drunk+[1,0] , axis = 0)
         #chisq = chisq[maskp==0]
         #chi = chi[maskp==0]

         #chi = chi[np.abs(chi)<3]
         #bol = np.abs(chi)<3
         #print "before" , bol[:3,:3]
         #bol = ndimage.convolve(bol, k, mode='nearest')   
         #print "after" , bol[:3,:3]
         
         bol = np.abs(chi) < 3
         healthy = bol * bol2       

         chisq = chisq[healthy == True]

         return np.sum(chisq)

import scipy.optimize as op
nll = lambda *args: lnlike(*args)

"""
#Kp = sampler.imatrix_new(25, 3, 0.0, 0.0)
Kp = sampler.phi(0.3 , 3 , 25)[:,:-2]
#print Kp.shape
#print X.shape

plt.figure(figsize=(10,3))
plt.set_cmap('Greys_r')
ax = plt.gca()
im = ax.imshow(Kp, interpolation = "None" , origin = "lower") # , vmin = mi , vmax = ma)
plt.title(r"$K$" , fontsize = 20)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.2)
#plt.colorbar(im , cax = cax)
ax.set_xticks(())
ax.set_yticks(())
plt.tight_layout()
plt.savefig("K.png")
plt.close()

X = np.arange(75)
X0 = np.mean(X)
Y = np.exp(-0.5 * (X - X0) **2. / 10.**2.)
Y = Y/np.sum(Y)

plt.figure(figsize=(10,10))
#plt.set_cmap('Greys_r')
ax = plt.gca()
im = ax.plot(X, Y , lw = 3 , color = 'k') # , vmin = mi , vmax = ma)
plt.title(r"$X$" , fontsize = 20)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.2)
#plt.colorbar(im , cax = cax)
ax.set_xticks(())
ax.set_yticks(())
plt.xlim(0,74)
plt.tight_layout()
plt.show()
plt.close()
Yr = np.dot(Kp , Y.reshape((75 , 1))).flatten()
"""
"""
plt.figure(figsize=(10,10))
#plt.set_cmap('Greys_r')
ax = plt.gca()
im = ax.step(np.arange(25), Yr , lw = 3 , color = 'k') # , vmin = mi , vmax = ma)
plt.title(r"$m$" , fontsize = 20)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.2)
#plt.colorbar(im , cax = cax)
ax.set_xticks(())
ax.set_yticks(())
plt.xlim(0,24)
plt.tight_layout()
plt.show()
plt.close()
"""

for p in range(1399,1600):

    	 cx0 , cy0 = 0. , 0.
    	 b0 = np.median(data[p])
    	 datasq = data[p].reshape(25,25) - b0
    	 f0 = np.sum(datasq[11:14,11:14])

    	 print cx0 , cy0 , f0 , b0

    	 result = op.minimize(nll, [cx0, cy0 , f0, b0], args=(p)) #, method = 'TNC', bounds = ((-0.5,0.5),(-0.5,0.5),(0.,None),(0.,None)))
    	 #result = op.minimize(nll, [cx0, cy0 , f0, b0], args=(p), method = 'TNC', bounds = ((-0.5,0.5),(-0.5,0.5),(0.,None),(0.,None)))
    	 x_ml, y_ml, f_ml , b_ml = result["x"] 
    	 print x_ml , y_ml , f_ml, b_ml


         Kp = sampler.imatrix_new(25, H, x_ml, y_ml)
         #Kp = sampler_new.imatrix_new(25, H , cx[p] , cy[p])
         model = f_ml*np.dot(fl+X,Kp) + b_ml
         resi = (data[p,:] - model).reshape(25,25)
         res = (data[p,:] - model)*100/data[p,:]
         res = res.reshape(25,25)
         chi = (data[p,:] - model)/(f+g*np.abs(model))**.5
         chi_clip = (data[p,:] - model)/(f+g*np.abs(model)+q*model**2.)**.5
         chi = chi.reshape(25,25)
         chi_clip = chi_clip.reshape(25,25)
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
    	 im = ax.imshow(data[p,:].reshape(25,25), interpolation = "None" , origin = "lower",norm=LogNorm() , vmin = mi , vmax = ma)
         plt.title("SR PSF Model")
         divider = make_axes_locatable(ax)
	 cax = divider.append_axes("right", size="5%", pad=0.2)
         plt.colorbar(im , cax = cax)
	 ax.set_xticks(())
	 ax.set_yticks(())

         plt.subplot(2,3,2)

	 ax = plt.gca()
    	 im = ax.imshow(model.reshape(25,25) , interpolation = "None" , origin = "lower",norm=LogNorm() , vmin = mi , vmax = ma)
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


         bol = np.abs(chi_clip)<1

         chi[bol==False] = np.nan
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
         maskp[maskp!=0] = 1
         maskp[np.abs(chi_clip) > 1.] = 2
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
