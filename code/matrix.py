import sampler
import profile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

M , H = 25 , 3
y = profile.makeGaussian(H*M, H*M/6. ,0.1, (H*M/2.+0.83*H, H*M/2.+1.45*H))

hx, hy, hf = sampler.test_imatrix_new(M, H, -1.45, -0.83)
qw = hx.dot(y).dot(hy)
qq = y.flatten().dot(hf).reshape(M,M)



plt.subplot(1,3,1)
plt.imshow(y, interpolation = "None" , origin = "lower")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(qw, interpolation = "None" , origin = "lower")
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(qq, interpolation = "None" , origin = "lower")
plt.colorbar()
plt.show()


M , H = 25 , 3

y = profile.makeGaussian(M, M/5. ,0.1, (M/2.+0.83, M/2.+1.45))
y /= y.sum() 

import c3
print c3.find_centroid(y)            #data
model = profile.makeGaussian(H*M, H*M/5. ,0.1, (H*M/2., H*M/2.))               #model

hh = sampler.imatrix(y.flatten() , H)                                      #sampling matrix
ww = model.flatten().dot(hh).reshape(M,M)                                      #model rendered on data grid
ww = ww/ww.sum()

plt.subplot(1,4,1)
plt.imshow(y, interpolation = "None" , origin = "lower")
plt.title("data")
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(model, interpolation = "None" , origin = "lower")
plt.title("model")
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(ww, interpolation = "None" , origin = "lower")
plt.title("rendered model")
plt.colorbar()
plt.subplot(1,4,4)
plt.imshow(ww-y, interpolation = "None" , origin = "lower")
plt.title("residual")
plt.colorbar()
plt.show()

