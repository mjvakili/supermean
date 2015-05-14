import sampler
import profile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

M , H = 25 , 3
y = profile.makeGaussian(H*M, H*M/6. ,0.1, (H*M/2.+0.83*H, H*M/2.+1.45*H))

hx, hy, hf = sampler.imatrix_new(M, H, -1.45, -0.83)
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
