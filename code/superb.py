import numpy as np
import ms
import shifter
import numpy as np
import scipy.optimize as op
from scipy import ndimage
import h5py
import time
from scipy.linalg import cho_factor, cho_solve
from interruptible_pool import InterruptiblePool
from nll_grad import nll_grad_lnX
from nll_grad_fb import v2_fit_single_patch , v3_fit_single_patch
from nll_ctr import fit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sampler

F = h5py.File('samplerx3.hdf5','r')#["sampler"] 
K = F["samplerx3"]
Nthreads = 15  #number of threads

def fg(args):
    return nll_grad_lnX(*args)


def fit_single_patch(data, mask, psf, old_flux, old_back, floor, gain):
    C = floor + gain * np.abs(old_flux * psf + old_back)[mask]
    A = np.ones((C.size, 2))
    A[:, 1] = psf[mask]
    AT = A.T
    ATICA = np.dot(AT, A / C[:, None])
    ATICY = np.dot(AT, data[mask] / C)
    return np.dot(np.linalg.inv(ATICA), ATICY)

class stuff(object):
   
     def __init__(self, data, cx, cy, masks,
                         f = 5e-2, g = 1e-2, fl = 1e-5, H = 3, epsilon = 1e-2,
                        min_iter=5, max_iter=10, check_iter=5, tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD mask matrix;
            data contains images of stars, and mask contains questionable
            pixels in each image.
            N = the number of stars
            D = the number of pixels in each patch
            H = upsampling factor
            cx, cy = centroiding offsets
            f = floor variance of the nosie
            fl = floor of the PSF model                 
        """
        #self.tr = tr
	#self.ts = ts
        self.N = data.shape[0]             #number of observations
        self.D = data.shape[1]             #input dimensionality
        self.H = H                         #upsampling factor
        self.epsilon = epsilon             #smoothness parameter                                  
        self.data = np.atleast_2d(data)    #making sure the data has the right dimension
        self.masks = np.atleast_2d(masks)  #making sure the mask has the right dimension
        self.dx = cx                      #list of centroid offsets x
        self.dy = cy                       #list of centroid offsets y   
        self.M = int(self.D**.5)
        self.f = f                         #floor variance of noise model
        self.fl = fl			   #floor of the PSF model
        self.g = g
        """ outputs of the code:
                                 H*H*D-dimensional mean vector:  X
                                 N-dimensional flux vector:      F
                                 N-dimensional background: B
        """
               
        self.F = np.zeros((self.N))                 #Creating an N-dimensional Flux vector.
        self.B = np.zeros((self.N))                 #one flat-field per star 
        self.lnX = np.ones((self.D*self.H*self.H))  #log(X)
        self.g  = g                             #gain of the noise model
        """ initialization of X, F, B by means of subtracting the median!(to initialize the background B),
                                                  normalizing (to intialize the flux F),
                                                  shifting, and upsampling (to initialize the mean X)"""
        self.initialize()
        
        #""" recording parameters after each iteration """

        self.write_pars_to_file(0)

        """ updating F, B, centroids, X"""
        self.update(max_iter, check_iter, min_iter, tol)
        
        
     def initialize(self):

        """
        initializing the parameters
        """        
        self.masks[self.masks == 0] = -1
	self.masks[self.masks > 0] = False
	self.masks[self.masks < 0] = True
        self.masks = self.masks == True
        #print self.masks[0]
        m = int((self.D)**.5)
        self.d2d = self.data.reshape(self.N , m , m)
        self.dm = np.zeros((self.N, self.D))
        X = np.zeros_like(self.lnX)

        for i in range(self.N):
                          
          self.B[i]   =  np.array([self.d2d[i,m/2-4:m/2+5,-1:].mean(),
	                           self.d2d[i,m/2-4:m/2+5,:1].mean(),
                                   self.d2d[i,:1,m/2-4:m/2+5].mean(),
                                   self.d2d[i,-1:,m/2-4:m/2+5].mean()]).mean()
          
          self.dm[i]  =  self.data[i]-self.B[i]
          self.dm    -= self.dm.min()
          self.F[i]   =  np.sum(self.dm[i])
          self.dm[i] /=  self.F[i]                    
          shifted = shifter.shifter(self.dm[i], self.dx[i], self.dy[i])
          obs = ndimage.interpolation.zoom(shifted.reshape(25,25), self.H,
                                           output = None, order=3, mode='constant', 
 					   cval=0.0, prefilter=True).flatten()
          
          X += obs.flatten()   
         
        X /= self.N
        X[X<0] = self.fl
        X[X==0] = self.fl
        self.lnX = np.log(X)
        """
        XX = np.exp(self.lnX).reshape(self.H*self.M ,self.H*self.M) + self.fl
        plt.imshow(XX , interpolation = None , norm = LogNorm())
	plt.title(r"$X_{\mathtt{initial}}$")        
	plt.colorbar()
        plt.xticks(())
        plt.yticks(())
        plt.show()

        gradx = self.func_grad_lnX_Nthreads(self.lnX)[1].reshape(75,75)
        plt.imshow(np.abs(gradx) , interpolation = None , norm = LogNorm())
        plt.title(r"$|d\mathtt{NLL}/d(\mathtt{lnX})|_{\mathtt{initial}}$")
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())
        plt.show()
        """
     def write_pars_to_file(self, step):

       f = h5py.File("trial_iter_%d.h5"%(step), 'w')
       grp = f.create_group('model_pars')     # create group
       columns = ['F', 'B', 'lnX']
       n_cols = len(columns)       # number of columns
       col_fmt = []                # column format
       for column in columns:
          column_attr = getattr(self, column)
          # write out file
          grp.create_dataset(column, data=column_attr)
	# save metadata
       metadata = [ 'fl', 'f', 'g']
       for metadatum in metadata:
          grp.attrs[metadatum] = getattr(self, metadatum)

       f.close()

     def lsq_update_FB(self):

       """least square optimization of F&B"""
       for p in range(self.N):
         #fit_single_patch(data, mask, psf, old_flux, old_back, floor, gain)
         Kp = np.array(K[str(p)])
         mask = self.masks[p]
         Y = self.data[p]
         old_flux, old_back = self.F[p], self.B[p]
         self.X = np.exp(self.lnX)
         psf = np.dot(self.X + self.fl, Kp)
         
         for i in range(10):
              old_back, old_flux = fit_single_patch(Y, mask, psf, old_flux, old_back, self.f, self.g)
         self.B[p], self.F[p] = old_back, old_flux

         """
         modelp = np.dot(self.X + fl, K[p])
         A = np.vstack([np.ones(self.D), modelp]).T
         C = f + g*np.abs(modelp)
         Y = self.data[p]
         mask = self.masks[p]
	 A = A[mask]
	 C = C[mask]
	 Y = Y[mask]
         AT = A.T
         # this is the idea : C[self.masks[p]] = np.inf
         ATICA    = np.dot(AT, A/C[:,None])
         factor = cho_factor(ATICA, overwrite_a = True)
         x = cho_solve(factor, np.dot(AT, Y/C))
         self.B[p], self.F[p] = x[0], x[1]         
         """
     def bfgs_update_FB(self):
       
       MS = h5py.File('masked_samplerx3.hdf5','r')
       MD = h5py.File('masked_data.hdf5','r')
       masked_samplers = MS["masked_samplerx3"]#[tr]
       masked_data = MD["masked_data"]#[tr]

       for p in range(self.N):
	 #print masked_samplers				
         Kp = masked_samplers[str(p)]
         Y = masked_data[str(p)]
	 theta = self.B[p], self.F[p]
         psf = np.dot(np.exp(self.lnX) + self.fl, Kp)
	 grad_func = v3_fit_single_patch
         #print grad_func
         x = op.fmin_l_bfgs_b(grad_func, x0=theta, fprime = None, \
                             args=(Y, psf, self.f, self.g), approx_grad = False, \
                             bounds = [(0.,100.), (1.,10.**7.)], m=10, factr=1000., pgtol=1e-08, epsilon=1e-08, maxfun=60)
	 #print p, x
         self.B[p], self.F[p] = x[0]
       MS.close()
       MD.close() 
    
     def update_centroids(self):
       
        MD = h5py.File('masked_data.hdf5','r')
        masked_data = MD["masked_data"]
     
        #updating the sampling matrices: we donot overwrite the original ones because 
        #the new ones depende on the variance model, and the variance model is not 
        #perfect at the moment!

        GG  = h5py.File('masked_samplerx3.hdf5','w')   
        Grp = GG.create_group("masked_samplerx3")
        for p in range(self.N):
    
          xp, yp = fit((self.dx[p], self.dy[p]), \
                       masked_data[str(p)], self.masks[p], \
                       self.X, self.F[p], self.B[p], \
                       self.f, self.g, self.fl)
          masked_dset = sampler.imatrix_new(self.M, self.H, xp , yp)[: , self.masks[p]]
          Grp.create_dataset(str(p), data = masked_dset)
        GG.close()
        MD.close()


 
     def func_grad_lnX_Nthreads(self, params):

        """
        Use multiprocessing to calculate negative-log-likelihood and gradinets
        w.r.t lnX, plus the terms coming from th regularization terms.
        """
        n_samples = self.N
        self.lnX = params
        #self.fl, self.f, self.g, self.H, Nthreads = args


	Pool = InterruptiblePool(Nthreads)
        mapfn = Pool.map
        Nchunk = np.ceil(1. / Nthreads * n_samples).astype(np.int)

        arglist = [None] * Nthreads
        for i in range(Nthreads):
          s = int(i * Nchunk)
          e = int(s + Nchunk)
	  arglist[i] = (self.lnX, self.F, self.B, self.fl, self.f, self.g, self.H, s, e)  
        result = list(mapfn(fg, [ars for ars in arglist]))    
        nll, grad = result[0]
        a = time.time()
        for i in range(1, Nthreads):
           nll += result[i][0]
           grad += result[i][1]
        #print "adding up nll's from individual threads", time.time() - a
        Pool.close()
        Pool.terminate()
        Pool.join()
        #computing the regularization term and its derivative w.r.t lnX
        reg_func, reg_grad = self.reg_func_grad_lnX() 
        return nll + reg_func, grad + reg_grad	
     
     def reg_func_grad_lnX(self):
        """ returns regularization term in NLL
            and its derivative w.r.t lnX""" 
      
        self.X = np.exp(self.lnX)

        b = int((self.D)**.5)
        Z = self.X.reshape((self.H*b, self.H*b))
        c= np.zeros_like(Z)
        c[:,:-1] += Z[:, 1:]
        c[:, 1:] += Z[:,:-1]
        c[1:, :] += Z[:-1,:]
        c[:-1,:] += Z[1:, :]
        grad = 2.*self.epsilon*(4.*Z - c).flatten()*self.X 
        #grad = grad*self.X               
        func  = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.)+ self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.)
        return func , grad         
     
     """
     def func_lnX_grad_lnX(self, params , *args):

           returns Gradient w.r.t Log(X) & NLL,
           replaced by func_lnX_grad_lnX_Nthreads,
           keeping this for sanity check for now
         
        self.data, self.F, self.B, K = args
        self.lnX = params
        
        n_samples = self.data.shape[0]

        self.X = np.exp(self.lnX)

        b  = int((self.D)**.5)
        Z = self.X.reshape((self.H*b, self.H*b))
        c=np.zeros_like(Z)
        c[:,:-1] += Z[:, 1:]
        c[:, 1:] += Z[:,:-1]
        c[1:, :] += Z[:-1,:]
        c[:-1,:] += Z[1:, :]
        grad = 2.*self.epsilon*(4.*Z - c).flatten()*self.X 
        #grad = grad*self.X               
        func  = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.)+ self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.)
        
        for p in range(self.N):

         Kp = np.array(K[str(p)])
         Y = self.data[p]
         modelp = self.F[p]*np.dot(self.X+self.fl, Kp) + self.B[p]
         mask = self.masks[p]

	 Y = Y[mask]
         modelp = modelp[mask]
         ep = Y - modelp
 
         varp = self.f + self.g*np.abs(modelp)
         gradp = -1.*self.F[p]*Kp
         
         gradp = gradp[:,mask]

         gainp = (self.g/2.)*(varp**-1. - ep**2./varp**2.)
	 
         gradp = self.X[:,None]*gradp*(ep/varp - gainp)[None,:]
         Gradp = gradp.sum(axis = 1) 
         grad += Gradp
         func += 0.5*np.sum(((ep)**2.)/varp) + .5*np.sum(np.log(varp))
	
        return func, grad
     """
     """
     def grad_lnX(self, params , *args):

      

        self.F, self.B = args
        self.lnX = params
        return self.func_lnX_grad_lnx[1]

     def func_lnX(self, params , *args):

    
        self.F, self.B = args
        self.lnX = params
        return self.func_lnX_grad_lnx[0]
     
     def grad_F(self, params, *args):

      

        self.lnX, self.B = args
        self.F = params
        
        self.X = np.exp(self.lnX)
        
        grad = np.zeros_like(self.F)
        
        for p in range(self.N):
        
         Kp = K[p]
         y = self.data[p]
         mask = self.masks[p]
         nmodelp = np.dot(self.X+fl,Kp)
         modelp = self.F[p]*nmodelp + self.B[p]
        
         y = y[mask]
	 nmodelp = nmodelp[mask]
	 modelp = modelp[mask]

         residualp = y - modelp
         #residualp[self.mask[p]!=0] = 0   #excluding flagged pixels from contributing to gradient_X
         varp = f + g*np.abs(modelp)
         gradp = -1.*nmodelp
         gainp = (g/2.)*nmodelp*(varp**-1. - residualp**2./varp**2.)
         #gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid
         grad[p] = np.sum(residualp*gradp/varp) + np.sum(gainp)
        return grad
      
     def grad_B(self, params, *args):

        self.lnX, self.F = args
        self.B = params

        self.X = np.exp(self.lnX)

        grad = np.zeros_like(self.B)
        for p in range(self.N):
         y = self.data[p]
         Kp = K[p]
         modelp = self.F[p]*np.dot(self.X+fl,Kp) + self.B[p]
         mask = self.masks[p]
         y = y[mask]
         modelp = modelp[mask]
         varp = f+g*np.abs(modelp)
         residualp = y - modelp
         #residualp[self.mask[p]!=0] = 0   #excluding flagged pixels from contributing to gradient_X
         gainp = - (g/2.)*(residualp**2./varp**2.) + (g/2.)*(varp**-1.)
         #gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid   
         grad[p] = -1.*np.sum(residualp/varp) + np.sum(gainp)      
        return grad
    
     def func_F(self , params, *args):
        self.lnX, self.B = args
        self.F = params
        return self.nll()

     def func_B(self, params, *args):
        self.lnX, self.F = args
        self.B = params
        return self.nll()
     """

     def bfgs_lnX(self, num_funccalls):
       
        x = op.fmin_l_bfgs_b(self.func_grad_lnX_Nthreads, x0=self.lnX, fprime = None, \
                             args=(), approx_grad = False, \
                             bounds = [(np.log(1e-5), 0.) for _ in self.lnX], m=10, factr=10.0, pgtol=1e-5, epsilon=1e-8, maxfun=num_funccalls)
        gx = x[2]["grad"]
        print x
        print gx
        #X = np.exp(self.lnX).reshape(self.H*self.M ,self.H*self.M) + self.fl
        #plt.imshow(np.abs(gx).reshape(100,100), interpolation = None, norm = LogNorm())
        #plt.colorbar()
        #plt.show()
        self.lnX  = x[0]

     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.lnX, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=1000., pgtol=1e-02, epsilon=1e-02, maxfun=20)
        #print x
        self.F  = x[0]

     def bfgs_B(self):
        x = op.fmin_l_bfgs_b(self.func_B,x0=self.B, fprime = self.grad_B,args=(self.lnX, self.F), approx_grad = False, \
                              bounds = None, m=10, factr=1000., pgtol=1e-02, epsilon=1e-02, maxfun=20)
        #print x
        self.B  = x[0]
     
     
     def nll(self):

       self.X = np.exp(self.lnX)
       b  = int((self.D)**.5)
       Z = self.X.reshape((self.H*b, self.H*b))
       nll = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.) + self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.) 
       for i in range(self.N):
         Ki = np.array(K[str(i)])
         Y = self.data[i]
         model_i = self.F[i]*np.dot(self.X+self.fl, Ki) + self.B[i]
         
         mask = self.masks[i]
	 Y = Y[mask]
         model_i = model_i[mask]         
 
         var_i = self.f + self.g*np.abs(model_i)
         residual_i = Y - model_i
         nll += 0.5*np.sum(((residual_i)**2.)/var_i) + 0.5*np.sum(np.log(var_i))
       return nll
     
     def update(self, max_iter, check_iter, min_iter, tol):
      

        nll = self.nll()
        print "starting NLL is:", nll
        np.savetxt("superb_wfc_mean_iter_%d.txt"%(0)       , self.lnX ,fmt='%.64f')
     
        for i in range(1, max_iter+1):
            
	    a = time.time()
            self.bfgs_update_FB()
	    print time.time() - a
            a = time.time()
            #self.update_centroids()
	    print time.time() - a
            a = time.time()
            self.bfgs_lnX(200)
	    print time.time() - a
            np.savetxt("superb_wfc_mean_iter_%d_nfljadid.txt"%(i)       , self.lnX ,fmt='%.64f')
            np.savetxt("superb_wfc_flux_iter_%d_nfjadid.txt"%(i)       , self.F ,fmt='%.64f')
	    np.savetxt("superb_wfc_bkg_iter_%d_nfljadid.txt"%(i)        , self.B ,fmt='%.64f')
            """
            if (i==max_iter):
             X = (np.exp(self.lnX)+self.fl).reshape(self.H*self.M ,self.H*self.M)
             plt.imshow(X , interpolation = None , norm = LogNorm())
             plt.title(r"$X_{\mathtt{final}}$")
             plt.colorbar()
             plt.xticks(())
             plt.yticks(())
             plt.show()"""
             #np.savetxt("superb_wfc_mean_iter_%d.txt"%(i+1)       , self.lnX ,fmt='%.64f')
            """  
             gradx = self.func_grad_lnX_Nthreads(self.lnX)[1].reshape(75,75)
             plt.imshow(np.abs(gradx) , interpolation = None , norm = LogNorm())
             plt.title(r"$|d\mathtt{NLL}/d(\mathtt{lnX})|_{\mathtt{final}}$")
             plt.colorbar()
             plt.xticks(())
             plt.yticks(())
             plt.show()"""
             
            """
	    a = time.time()
            W = self.func_lnX_grad_lnX(self.lnX, self.data, self.F, self.B, K )
            print time.time() - a
            print W
            
	    a = time.time()
	    Q = self.func_lnX_grad_lnX_Nthreads(self.lnX)
            #self.fl, self.f, self.g, self.H, Nthreads = args
	    print time.time() - a
	    print Q
            
	    print Q[0] - W[0], np.sum((Q[1] - W[1])**2.)s
            """
            if np.mod(i, check_iter) == 0:
                new_nll =  new_nll = self.nll()
                print 'NLL at step %d is:' % (i+1), new_nll
            if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll
        F.close()
