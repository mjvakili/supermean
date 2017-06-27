import numpy as np
import pyfits as pf
import sampler
fl=1e-5
f = .05
g = .01 
H = 4
epsilon = 0.
X = np.exp(np.loadtxt("noregwfc_mean_iter_22.txt"))
F = np.loadtxt("noregwfc_flux_iter_22.txt")
B = np.loadtxt("noregwfc_background_iter_22.txt")
lnX = np.log(X)

N = F.shape[0]
data = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]
mask = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]
dx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")[0:120]
dy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")[0:120]
D = data.shape[1]
import h5py
File = h5py.File('sampler.h5','r') 

#print F                  #loading file
K = np.array(File["MyDataset"])
print K.shape
#K *= 0
data = data.astype(np.double)
mask = mask.astype(np.double)
K = K.astype(np.double)
print K.dtype
#print mask.dtype
#print data.dtype
#print F.dtype

import terminator
import time
fun = terminator.njj
func = terminator.nll

a = time.time()
fun(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl)
#func(data, mask, F, lnX, B, epsilon, dx, dy, H, f, g, fl)
print time.time() - a

a = time.time()
#fun(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl)
func(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl)
print time.time() - a
print func(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl) - fun(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl)
def nlll():

       X = np.exp(lnX)
       b  = int(D**.5)
       M = b
       Z = X.reshape((H*b, H*b))
       nll = epsilon*((Z[:,1:]-Z[:,:-1])**2.).sum() + epsilon*((Z[1:,:]-Z[:-1,:])**2.).sum()
       model_i = data[1,:]
       for j in range(1): 
        for i in range(N):
         Ki = sampler.imatrix_new(M, H, dx[i], dy[i])
         #print np.sum((Ki - K[i])**2.)
         model_i = F[i]*np.dot(X+fl, Ki) + B[i]
         residual_i = data[i] - model_i
         residual_i[mask[i]!=0] = 0   #excluding flagged pixels from contributing to NLL
         var_i = f + g*np.abs(model_i)
         nll += 0.5*np.sum(((residual_i)**2.)/var_i) + .5*np.sum(np.log(var_i))
       return nll

a = time.time()
#nlll()
print time.time() - a
