import numpy as np
import profile


"""dumbass sr"""
n , m = 2*4, 2
v = np.random.rand(m)
D = np.kron(np.eye(n) , np.ones(m).T)

C = np.kron(np.eye(m) , np.ones(n)).T

print D.shape , C.shape

s = np.dot(D,C)/n
w = np.dot(s,v)
print v
print w

print v.sum() , w.sum()


from scipy import ndimage
import profile
from matplotlib.colors import LogNorm
image = profile.makeGaussian(18 , 4. , 0.1 ,(8.1,7.8))

ii = ndimage.interpolation.zoom(image, 3, output=None, order=3)
def srmatrix(n,m):

  D = np.kron(np.eye(n) , np.ones(m).reshape(1,m).T).T 
  C = np.kron(np.eye(m) , np.ones(n).reshape(1,n)).T 
  return np.dot(D,C)#np.kron(np.dot(D,C) , np.dot(D,C))

from pylab import *

i = np.dot(srmatrix(54,18) , np.dot(image , srmatrix(54,18).T)) 
#i = np.dot(image,srmatrix(10,17).T).reshape(10,10)

imshow(image, interpolation ="None")
show()
imshow(ii, interpolation ="None")
colorbar()
show()
imshow(i , interpolation = "None")
show()

