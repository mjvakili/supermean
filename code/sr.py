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
         
fl=1e-5
f = .05
g = .01 
H = 3
#X = np.exp(np.loadtxt("superb_wfc_mean_iter_5_nfl.txt"))
X = np.exp(np.loadtxt("superb_wfc_mean_iter_0.txt"))
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

Z2 = ndimage.convolve(Z2, k, mode='nearest')
Z2 = Z2 * Z.sum() /Z2.sum()

plt.figure(figsize=(10,10))
         
ax = plt.gca()
plt.set_cmap('RdBu')
im = ax.imshow(Z, origin = "lower",norm = LogNorm(),  interpolation = "None")
#plt.title("SR PSF Model")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im , cax = cax)
ax.set_xticks(())
ax.set_yticks(())
plt.savefig("super.png")

plt.figure(figsize=(10,10))
         
ax = plt.gca()
plt.set_cmap('RdBu')
im = ax.imshow(Z2, origin = "lower",norm = LogNorm(),  interpolation = "None")
#plt.title("SR PSF Model")
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.2)
#plt.colorbar(im , cax = cax)
ax.set_xticks(())
ax.set_yticks(())
plt.savefig("super0.png")
