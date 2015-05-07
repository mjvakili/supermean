import ms
import interp
import c3
import numpy as np
def phi(dx , H , M):

  """
  Interpolator matrix that samples a 1-d model
  at the data grid. It downsamples the model by 
  a factor of H, and shifts the model
  to the data-grid.
  Prameters:
  dx = subpixel offset along the x-axis at the native pixel resolution,
  H  = upsampling factor.
  M  = dimension along individual axes.  
  """

  a , b = 0. ,  1.
  h1 = (b-a)/(H*M-1)
  h2 = (b-a)/(M-1)
  k = np.arange(1, H*M+1) + 1
  x = np.arange(M)*h2+dx*h2 
  k = k[None,:]
  x = x[:,None]
  y = (x - a)/h1 - k + 2 
  return interp.pi(y)

def imatrix(data,H):
  
  M = int((data.shape[0])**.5)
  data = data.reshape(M,M)
  center = c3.find_centroid(data)
  dx  , dy = center[0] , center[1]
  hx = np.dot(phi(-dx, H , M) , np.linalg.inv(ms.B(H*M)).T)
  hy = np.dot(np.linalg.inv(ms.B(H*M)) , phi(-dy, H , M).T)
  hf = np.kron(hx.T, hy)
  return hf
