import pyfits as pf
import matplotlib.pyplot as plt

a = pf.open("wfc3_f160w_clean_central100.fits")[0].data
a = a.reshape(a.shape[0],25,25)
"""
for i in range(10):

  plt.imshow(a[i] , interpolation = "None")
  plt.colorbar()
  plt.show()
"""

import interp
import numpy as np
import matplotlib.pyplot as plt
import ms
from scipy import ndimage

def phi(dx , H , M):

  """
  Interpolator matrix that samples the 1-d model
  at the 1-d data grid. It down-samples the model
  to the resolution of the data and shifts the model
  to the data-grid.
  Prameters:
  dx = subpixel offset along the x-axis,
  H  = upsampling factor.
  M  = dimension along individual axes.  
  """

  a , b = 0. ,  1.
  h1 = (b-a)/(M-1)
  h2 = (b-a)/(H*M-1)
  k = np.arange(M+2) + 1
  
  x = np.arange(H*M+2)*h2  +dx*h2 #In case of H = 1, if I add anything to this(i.e h2*dx), it'll shift the image 
                                  #(i.e. by dx)
  #print interp.pi((x - a)/h1 - k[2:]*0+10 + 2 )
  
  k = k[None,:]
  x = x[:,None]
  y = (x - a)/h1 - k + 2 
  return interp.pi(y)

"""
M , H = 25 , 3
dx = 0.
dy = 0.
#print phi(0,30).shape
#print phi(-dx, H , M).shape
hx = np.dot(phi(dx, H , M) , np.linalg.inv(ms.B(H*M)).T)
hy = np.dot(np.linalg.inv(ms.B(H*M)) , phi(dy, H , M).T)
hf = np.kron(hx.T, hy)
"""
"""
plt.imshow(phi(dx, H , M) , interpolation="None")
plt.colorbar()
plt.show()
"""

import profile
import shifter

y = shifter.shifter(a[13].flatten()).reshape(25,25)
print a[12].min()

ynew = ndimage.interpolation.zoom(y, 3, output = None , order=3, mode='constant', cval=0.0, prefilter=True)
ynew[ynew<0] = y.mean()
ynew[ynew==0] = y.mean()
#print znew.shape
#ynew = znew.reshape(25,25)
vmi = min(ynew.min(),a[13].min())
vma = max(ynew.max(),a[13].max())


from matplotlib.colors import LogNorm
plt.subplot(1,3,1)
plt.imshow(a[13], interpolation = "None" , origin = "lower" , norm = LogNorm(), vmin = vmi , vmax = vma)
#plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(ynew, interpolation = "None" , origin = "lower", norm = LogNorm(), vmin = vmi , vmax = vma)
#plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(ynew, interpolation = "None" , origin = "lower", norm = LogNorm() , vmin = vmi , vmax = vma)
#plt.colorbar()
plt.show()

#plt.imshow(ms.phi(0.1,  M) , interpolation="None")
#plt.colorbar()
#plt.show()
