import numpy as np
import ms
import shifter
import numpy as np
import scipy.optimize as op
from scipy import ndimage
import h5py
import time
from scipy.linalg import cho_factor, cho_solve

f  = 0.05
g  = 0.01
fl = 1e-5

F = h5py.File('sampler.h5','r') 
K = F["MyDataset"]
Nthreads = 1

from multiprocessing import Pool
#from interruptible_pool import InterruptiblePool
from nll_grad import nll_grad_lnX

def fg(args):

    return nll_grad_lnX(*args)
"""
def mem():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    print 'Memory usage is %0.1f GB' % (mem / 1000.)
"""

def fit_single_patch(data, mask, psf, old_flux, old_back, floor, gain):
    C = floor + gain * np.abs(old_flux * psf + old_back)[mask]
    A = np.ones((C.size, 2))
    A[:, 1] = psf[mask]
    AT = A.T
    ATICA = np.dot(AT, A / C[:, None])
    ATICY = np.dot(AT, data[mask] / C)
    return np.dot(np.linalg.inv(ATICA), ATICY)

class stuff(object):
   
     def __init__(self, data, cx, cy, masks, H = 3, epsilon = .01 , min_iter=5, max_iter=10, check_iter=5 , tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD mask matrix;
            data contains images of stars, and mask contains questionable
            pixels in each image.
            N = the number of stars
            D = the number of pixels in each patch
            H = upsampling factor
            cx, cy = centroiding offsets
                             
        """

        self.N = data.shape[0]             #number of observations
        self.D = data.shape[1]             #input dimensionality
        self.H = H                         #upsampling factor
        self.epsilon = epsilon             #smoothness parameter                                  
        self.data = np.atleast_2d(data)    #making sure the data has the right dimension
        self.masks = np.atleast_2d(masks)  #making sure the mask has the right dimension
        self.dx = cx                       #list of centroid offsets x
        self.dy = cy                       #list of centroid offsets y   
        self.M = int(self.D**.5)
         
        """ outputs of the code:
                                 H*H*D-dimensional mean vector:  X
                                 N-dimensional flux vector:      F
                                 N-dimensional flat-field background: B
        """
               
        
        self.F = np.zeros((self.N))                 #Creating an N-dimensional Flux vector.
        self.B = np.zeros((self.N))                 #one flat-field per star 
        self.lnX = np.ones((self.D*self.H*self.H))  #log(X)
 
        """ initialization of X, F, B by means of subtracting the median!(to initialize the background B),
                                                  normalizing (to intialize the flux F),
                                                  shifting, and upsampling (to initialize the mean X)"""
        self.initialize()

        """ updating X, F, B by X-step, F-step, and B-step optimization"""
        self.update(max_iter, check_iter, min_iter, tol)
        
        
     def initialize(self):

        """
        initializing the parameters
        """         
        self.masks[self.masks == 0] = -1
	self.masks[self.masks > 0] = False
	self.masks[self.masks  < 0] = True
        self.masks = self.masks==True
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
        self.lnX = np.log(X)

     def update_FB(self):

       """least square optimization of F&B"""
       for p in range(self.N):
         #fit_single_patch(data, mask, psf, old_flux, old_back, floor, gain)
         
         mask = self.masks[p]
         Y = self.data[p]
         old_flux, old_back = self.F[p], self.B[p]
         psf = np.dot(self.X + fl, K[p])
         
         for i in range(10):
              old_back, old_flux = fit_single_patch(Y, mask, psf, old_flux, old_back, f, g)
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
     def func_lnX_grad_lnX_Nthreads(self, params, *args):

        """
        Use multiprocessing to calculate negative-log-likelihood and gradinets.
        """
        n_samples = self.N
        self.lnX = params
        self.data, self.masks, self.F, self.B, K, fl, f, g, self.H, Nthreads = args

        #pool = InterruptiblePool(Nthreads)
	pool = Pool(Nthreads)
        mapfn = pool.map
        Nchunk = np.ceil(1. / Nthreads * n_samples).astype(np.int)

        arglist = [None] * Nthreads
        for i in range(Nthreads):
          s = i * Nchunk
          e = s + Nchunk
	  arglist[i] = (self.lnX, self.data[s:e], self.masks[s:e], self.F[s:e], self.B[s:e], K[s:e], fl, f, g, self.H)
        result = list(mapfn(fg, [args for args in arglist]))
        
        nll, grad = result[0]
        for i in range(1, Nthreads):
           nll += result[i][0]
           grad += result[i][1]
       
        pool.close()
        pool.terminate()
        pool.join()
        
        return nll, grad 
        

     def func_lnX_grad_lnX(self, params , *args):

        """returns Gradient w.r.t Log(X) & NLL"""
         
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
        grad = 2.*self.epsilon*(4.*Z - c).flatten()
        grad = grad*self.X               
        func  = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.)+ self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.)

        for p in range(self.N):

         Kp = K[p]
         Y = self.data[p]
         modelp = self.F[p]*np.dot(self.X+fl, Kp) + self.B[p]
         mask = self.masks[p]

	 Y = Y[mask]
         modelp = modelp[mask]
         ep = Y - modelp
 
         varp = f + g*np.abs(modelp)
         gradp = -1.*self.F[p]*Kp
         
         gradp = gradp[:,mask]

         gainp = (g/2.)*(varp**-1. - ep**2./varp**2.)
	 
         gradp = self.X[:,None]*gradp*(ep/varp - gainp)[None,:]
         Gradp = gradp.sum(axis = 1) 
         grad += Gradp
         func += 0.5*np.sum(((ep)**2.)/varp) + .5*np.sum(np.log(varp))
	
        return func, grad
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
         gainp = - (g/2.)*(residualp**2./varp**2.) + (g/2.)*(varp**-1.)
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
     

     def bfgs_lnX(self):
       
        x = op.fmin_l_bfgs_b(self.func_lnX_grad_lnX, x0=self.lnX, fprime = None, \
                             args=(self.F, self.B), approx_grad = False, \
                             bounds = [(np.log(fl), 0.) for _ in self.lnX], m=10, factr=1000., pgtol=1e-04, epsilon=1e-04, maxfun=60)
        print x
        self.lnX  = x[0]

     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.lnX, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=1000., pgtol=1e-02, epsilon=1e-02, maxfun=20)
        print x
        self.F  = x[0]

     def bfgs_B(self):
        x = op.fmin_l_bfgs_b(self.func_B,x0=self.B, fprime = self.grad_B,args=(self.lnX, self.F), approx_grad = False, \
                              bounds =None, m=10, factr=1000., pgtol=1e-02, epsilon=1e-02, maxfun=20)
        print x
        self.B  = x[0]
     """
     def nll(self):

       self.X = np.exp(self.lnX)
       b  = int((self.D)**.5)
       Z = self.X.reshape((self.H*b, self.H*b))
       nll = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.) + self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.) 
       for i in range(self.N):
         Ki = K[i]
         Y = self.data[i]
         model_i = self.F[i]*np.dot(self.X+fl, Ki) + self.B[i]
         
         mask = self.masks[i]
	 Y = Y[mask]
         model_i = model_i[mask]         
 
         var_i = f + g*np.abs(model_i)
         residual_i = Y - model_i
         nll += 0.5*np.sum(((residual_i)**2.)/var_i) + 0.5*np.sum(np.log(var_i))
       return nll
     
     def update(self, max_iter, check_iter, min_iter, tol):
      

        nll = self.nll()

        for i in range(max_iter):

	    a = time.time()
	    Q = self.func_lnX_grad_lnX_Nthreads(self.lnX, self.data, self.masks, self.F, self.B, K, fl, f, g, self.H, Nthreads)
	    print time.time() - a
	 
            a = time.time()
            W = self.func_lnX_grad_lnX(self.lnX, self.data, self.F, self.B, K )
            print time.time() - a
            
	    
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
