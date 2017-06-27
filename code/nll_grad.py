import numpy as np
import h5py
import time
a = time.time()
#MS = h5py.File('masked_sampler.hdf5','r')
#MD = h5py.File('masked_data.hdf5','r')
#masked_samplers = MS["masked_sampler"]
#masked_data = MD["masked_data"]
print time.time()-a
#assert False
#r = np.loadtxt("tr.txt").astype(int)
#ts = np.loadtxt("ts.txt").astype(int)
def nll_grad_lnX(lnX, F, B, fl, f, g, H, s, e):

        """
          inputs: 
                  lnX : Log of PSF shared for all data
                  data : patches of stars
                  masks: masks containing flaged pixels for each patch
                  F : list of flux values of the stars in different patches
                  B : list of backgrounds of patches
                  K : Array of sampling matrices. Each matrix samples
                      the PSF model at the data grid
                  fl: floor of the PSF model
		  f : floor variance of the pixel-noise
                  g : gain (variance = f + g*model)
                  H : upsampling factor
                
          outputs:
                 negative-log-likelihood
                 Gradient w.r.t Log(X). has the same dimension as X
        """
        #a = time.time()
        MS = h5py.File('masked_samplerx3.hdf5','r')
	MD = h5py.File('masked_data.hdf5','r')
	masked_samplers = MS["masked_samplerx3"]#[random]
	masked_data = MD["masked_data"]#[random]
        #print time.time() - a
        n_samples= e - s

        #X = lnX #np.exp(lnX)
        X = np.exp(lnX)
        grad = np.zeros_like(X)               
        func  = 0.0

        for p in np.arange(s,e):

         #a = time.time()
         Kp = masked_samplers[str(p)]
         #print p, time.time() - a
         #a = time.time()                        #masked_sampler for datum p 
         Y = np.array(masked_data[str(p)])
         #print time.time() - a                  #masked datum p
         modelp = F[p]*np.dot(X+fl, Kp) + B[p]   #resulting masked model of datum p

         ep = Y - modelp
         varp = f + g*np.abs(modelp)

         gainp = (g/2.)*(1./varp - ep*ep/(varp*varp))
 	 #gradp = Kp*(ep/varp - gainp)[None , :]
         gradp = X[:,None]*Kp*(ep/varp - gainp)[None,:]
         Gradp = np.sum(gradp, axis = 1) 
         grad += -1.*F[p]*Gradp
         func += 0.5*np.sum(((ep)**2.)/varp) + 0.5*np.sum(np.log(varp))
        #print func
        MS.close()
        MD.close()
        return func, grad

def reg_lnX(lnX, data, masks, F, B, K, H, epsilon):

        """
          inputs: 
                  lnX : Log of PSF shared for all data
                  data : patches of stars
                  masks: masks containing flaged pixels for each patch
                  F : list of flux values of the stars in different patches
                  B : list of backgrounds of patches
                  K : Array of sampling matrices. Each matrix samples
                      the PSF model at the data grid
                
          outputs:
                  regularization term in negative log-likelihood
        """
         
        
        n_samples, D = data.shape[0], data.shape[1]

        X = np.exp(lnX)

        b  = int((D)**.5)
        Z = X.reshape((H*b, H*b))
        c=np.zeros_like(Z)
        c[:,:-1] += Z[:, 1:]
        c[:, 1:] += Z[:,:-1]
        c[1:, :] += Z[:-1,:]
        c[:-1,:] += Z[1:, :]               
        func  = epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.+ (Z[1:,:]-Z[:-1,:])**2.)
                 

        return func
