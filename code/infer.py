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
from scipy.signal import convolve2d

F = h5py.File('samplerx3.hdf5','r')#["sampler"] 
K = F["samplerx3"]
Nthreads = 15  #number of threads

""" Utility Functions"""

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

def grower(array):
    """grows masked regions by one pixel
    """
    grower = np.array([[0,1,0],[1,1,1],[0,1,0]])   
    ag = convolve2d(array , grower , mode = "same")
    ag  = ag != 0
    return ag   

""" End of Utility Functions """

""" PSF Inference Class """

class infer(object):
   
     def __init__(self, data, cx, cy, masks,
                         f = 5e-2, g = 1e-1, fl = 1e-5, H = 3, epsilon = 1e-2,
                        min_iter=5, max_iter=20, check_iter=5, tol=1.e-8):

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
        self.N = data.shape[0]             #number of observations
        self.D = data.shape[1]             #input dimensionality
        self.H = H                         #upsampling factor
        self.epsilon = epsilon             #smoothness parameter                                  
        self.data = np.atleast_2d(data)    #making sure the data has the right dimension
        self.masks = np.atleast_2d(masks)  #making sure the mask has the right dimension
        self.M = int(self.D**.5)
        self.fl = fl			   #floor of the PSF model

        """ outputs of the code:
                                 H*H*D-dimensional mean vector:  X
                                 N-dimensional flux vector:      F
                                 N-dimensional background:       B
                                 N-dimensional centroid offset vector: dx
                                 N-dimensional centroid offset vector: dy
        """
               
        iself.F = np.zeros((self.N))                #Creating an N-dimensional Flux vector.
        self.B = np.zeros((self.N))                 #one flat-field per star 
        self.lnX = np.ones((self.D*self.H*self.H))  #log(X)
        self.dx = cx                                #list of centroid offsets x
        self.dy = cy                                #list of centroid offsets y   
        self.f = f                                  #floor variance of noise model
        self.g = g                                  #gain of the instrument
        """ initialization of X, F, B by means of subtracting the median!(to initialize the background B),
                                                  normalizing (to intialize the flux F),
                                                  shifting, and upsampling (to initialize the mean X)"""
        self.initialize()
        
        """ recording parameters after each iteration """

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
        m = int((self.D)**.5)
        self.d2d = self.data.reshape(self.N , m , m)
        self.dm = np.zeros((self.N, self.D))
        X = np.zeros_like(self.lnX)

        for i in range(self.N):
          
          dat_i = data[i].reshape(25,25)
          self.dx[i] , self.dy[i] = c3.find_centroid(dat_i)                
          #self.B[i]   =  np.array([dat_i[m/2-4:m/2+5,-1:].mean(),
	  #                         dat_i[m/2-4:m/2+5,:1].mean(),
          #                         dat_i[:1,m/2-4:m/2+5].mean(),
          #                         dat_i[-1:,m/2-4:m/2+5].mean()]).mean()
          self.B[i] = np.median(dat_i)  
          self.F[i]   =  np.sum(dat_i - self.B[i])
          self.dm[i] /=  self.F[i]                    
          shifted = shifter.shifter(self.dm[i], self.dx[i], self.dy[i])
          obs = ndimage.interpolation.zoom(shifted.reshape(25,25), self.H,
                                           output = None, order=3, mode='constant', 
 					   cval=0.0, prefilter=True).flatten()
          
          X += obs.flatten()   
         
        X /= self.N
        X[X<0] = self.fl
        X[X==0] = self.fl
        X = X / np.sum(X)
        self.lnX = np.log(X)
        self.f = 0.05
        self.g = 0.1
        self.mg = np.array([[0,1,0],[1,1,1],[0,1,0]])
      

     def write_pars_to_file(self, step):

       f = h5py.File("trial_iter_%d.h5"%(step), 'w')
       grp = f.create_group('model_pars')     # create group
       columns = ['F', 'B', 'dx', 'dy', 'lnX', 'f', 'g']
       n_cols = len(columns)       # number of columns
       col_fmt = []                # column format
       for column in columns:
          column_attr = getattr(self, column)
          # write out file
          grp.create_dataset(column, data=column_attr)
       f.close()

     def patch_nll(self, p, theta):

         """Return NLL of patcht p
         """

         X = np.exp(self.lnX)
         cx, cy , F, B, lf, lg = theta
         f = np.exp(lf)
         g = np.exp(lg)
         Kp = sampler.imatrix_new(25, H, cx, cy)
         model = F*np.dot(fl+X,Kp) + B
         resi = (self.data[p,:] - model).reshape(25,25)
         chisq = (self.data[p,:] - model)**2./(f+g*np.abs(model)) + np.log(f+g*np.abs(model))
         chi= (self.data[p,:] - model)/(f+g*np.abs(model)+1.*(model - B)**2.)**0.5
         maskp = self.mask[p,:]
         chisq = chisq.reshape(25,25)
         chi = chi.reshape(25,25)
         maskp = maskp.reshape(25,25) 
         bol2 = maskp==0                    #masks bad pixels from MAST
         
         #bol = np.abs(chi) < np.sqrt(3.0)  #chi-clipping masks

         bol = np.abs(chi) > np.sqrt(3.0)   #regions with modified chi-squared greater than three
         bol = grower(bol)                  #growing the masked regions by one pixel to be more conservative
         bol = ~bol + 2                     #flipping the bulian values of outliers masks : healthy = True(1), sick = False(0)
         healthy = bol * bol2       

         chisq = chisq[healthy == True]     #masking bad pixels and outliers from the chisq map
                  
         return np.sum(chisq)

     def patch_nll_grad_lnX(self, p, theta):
         
         """Returns dnll(p)/dlnX
         """
         
    
     def patch_nll_fbxy(self, p, theta):

         """Return NLL of patcht p as a function 
            of f, b, x, y of patch p
         """

         X = np.exp(self.lnX)
         xp, yp , fp, bp  = theta
         Kp = sampler.imatrix_new(25, H, xp, yp)
         model = fp * np.dot(fl+X, Kp) + bp
         resi = (self.data[p,:] - model).reshape(25,25)
         chisq = (self.data[p,:] - model)**2./(self.f + self.g*np.abs(model)) + np.log(self.f + self.g*np.abs(model))
         chi= (self.data[p,:] - model)/(self.f + self.g*np.abs(model) + 1.*(model - bp)**2.)**0.5
         maskp = self.mask[p,:]
         chisq = chisq.reshape(25,25)
         chi = chi.reshape(25,25)
         maskp = maskp.reshape(25,25) 
         bol2 = maskp==0                    #masks bad pixels from MAST
         
         #bol = np.abs(chi) < np.sqrt(3.0)  #chi-clipping masks

         bol = np.abs(chi) > np.sqrt(3.0)   #regions with modified chi-squared greater than three
         bol = grower(bol)                  #growing the masked regions by one pixel to be more conservative
         bol = ~bol + 2                     #flipping the bulian values of outliers masks : healthy = True(1), sick = False(0)
         healthy = bol * bol2       

         chisq = chisq[healthy == True]     #masking bad pixels and outliers from the chisq map
                  
         return np.sum(chisq)
     
     def update_centroids(self, p, theta):

         """update centroid of patch p
         """
         return None

     def update_FB(self, p):

        """update flux and background of patch p
        """
        return None
     
     def update_centroids_FB(self, p):

        """update flux and background and centroids of patch p
        """
        return None

     def total_nll_grad_lnX(self, params):

       return None    
 
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
     
     def grad_lnX(self, params , *args):

      

        self.F, self.B = args
        self.lnX = params
        return self.func_lnX_grad_lnx[1]

     def func_lnX(self, params , *args):

    
        self.F, self.B = args
        self.lnX = params
        return self.func_lnX_grad_lnx[0]
    
     def func_F(self , params, *args):
        self.lnX, self.B = args
        self.F = params
        return self.nll()

     def func_B(self, params, *args):
        self.lnX, self.F = args
        self.B = params
        return self.nll()

     def bfgs_lnX(self, num_funccalls):
       
        x = op.fmin_l_bfgs_b(self.func_grad_lnX_Nthreads, x0=self.lnX, fprime = None, \
                             args=(), approx_grad = False, \
                             bounds = [(np.log(1e-5), 0.) for _ in self.lnX], m=10, factr=10.0, pgtol=1e-5, epsilon=1e-8, maxfun=num_funccalls)
        gx = x[2]["grad"]
        print gx
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
