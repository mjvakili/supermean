import numpy as np

def nll_grad_lnX(lnX, data, masks, F, B, K, fl, f, g, H):

        """
          inputs: 
                  lnX : Log of PSF shared for all data
                  data : patches of stars
                  masks: masks containing flaged pixels for each patch
                  F : list of flux values of the stars in different patches
                  B : list of flat-fields of patches
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
         
        
        n_samples, D = data.shape[0], data.shape[1]

        X = np.exp(lnX)


        grad = np.zeros_like(X)               
        func  = 0.0

        for p in range(n_samples):

         Kp = K[p]
         Y = data[p]
         modelp = F[p]*np.dot(X+fl, Kp) + B[p]
         mask = masks[p]
	 Y = Y[mask]
         modelp = modelp[mask]
         
         ep = Y - modelp
         varp = f + g*np.abs(modelp)
         gradp = -1.*F[p]*Kp
         gradp = gradp[:,mask]
         gainp = (g/2.)*(varp**-1. - ep**2./varp**2.)

         #gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid
	 
         gradp = X[:,None]*gradp*(ep/varp - gainp)[None,:]
         Gradp = gradp.sum(axis = 1) 
         grad += Gradp
         func += 0.5*np.sum(((ep)**2.)/varp) + .5*np.sum(np.log(varp))
	
        return func, grad

def reg_lnX(lnX, data, masks, F, B, K, H, epsilon):

        """
          inputs: 
                  lnX : Log of PSF shared for all data
                  data : patches of stars
                  masks: masks containing flaged pixels for each patch
                  F : list of flux values of the stars in different patches
                  B : list of flat-fields of patches
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
