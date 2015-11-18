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

class EM(object):
   
     def __init__(self, data, masks, X, flux, bkg, cx, cy,
                        f = 5.e-2, g = 1.e-2, epsilon = 1e-2,
                        min_iter=5, max_iter=10, check_iter=1, 
                        tol=1.e-8):

         """ 
            inputs of the code: NxD data matrix and NxD mask matrix;
            data contains images of stars, and mask contains questionable
            pixels in each image.
            N = the number of stars
            D = the number of pixels in each patch
            psf = initial guess for the psf
            flux = initial guess for the amplitudes
            bkg = initial guess for the bkgs
            cx, cy = centroiding offsets
            f = floor variance of the nosie
            g = gain 
            fl = floor of the PSF model                 
         """
         self.N = data.shape[0]             #number of observations
         self.D = data.shape[1]             #input dimensionality
         self.epsilon = epsilon             #smoothness parameter                                  
         self.data = np.atleast_2d(data)    #making sure the data has the right dimension
         self.masks = np.atleast_2d(masks)  #making sure the mask has the right dimension
         self.dx = cx                       #list of centroid offsets x
         self.dy = cy                       #list of centroid offsets y   
         self.M = int(self.D**.5)
         self.f = f                         #floor variance of noise model
         self.fl = fl			   #floor of the PSF model
         self.g = g                         #gain 
       
	 self.X = X       
         self.F = flux                 #Creating an N-dimensional Flux vector.
         self.B = bkg
         self.cx = cx
         self.cy = cy
         """setting up constants throughout iterations"""
         self.sr_grid()
         self.Kernel()
         self.get_matrix()
         self.get_inv_matrix()
        
         #self.initialize()
         self.alpha_mean()
         self.update(max_iter, check_iter, min_iter, tol)
        
     def sr_grid(self):
         """
         initializing the super-resolution grid
         """
         hh = 1./101
         x, y = np.arange(0, 101)*hh + .5*hh , np.arange(0,101)*hh + .5*hh
         x, y = np.meshgrid(x, y, indexing="ij")
         self.samples = np.vstack((x.flatten(),y.flatten())).T
     
     def Kernel(self):
         """
         initializing the kernel
         """
         np.random.seed(12345)
         self.kernel = ExpSquaredKernel(.001, ndim=2) + 
                       WhiteKernel(.01, ndim=2)
   

     def get_matrix(self):
         """
         computing Kxx matrix, note: this remains constant
         """
         self.Kxx = self.kernel.value(self.samples , self.samples)
     
     def get_inv_matrix(self):
         """
         computing the inverse of Kxx, this also remains constant
         """
    	 self.Kxxinv = np.linalg.solve(self.Kxx , np.eye(self.X.shape[0])) 

     def alpha_mean(self):
         """
         computing the mean of X
         computing alpha = Kxxinv.Xmm,
         note: this changes after each update of X
         """
         self.mean = np.mean(self.X)
         Xmm = self.X - self.mean
         self.alpha = np.linalg.solve(self.Kxx , Xmm)
     
     def get_samplers(self):
         """
         computing and writing out the matrices 
         that sample the PSF at the data grid
         """
         f1 = h5py.File("sampler.hdf5","w")
         f2 = h5py.File("masked_sampler.hdf5","w")
         g1 = f1.create_group("sampler")
 	 g2 = f2.create_group("masked_sampler")
    	 for p in range(self.N):
           h = 1./25
           x2 = np.arange(0, 25)*h + .5*h -self.cx[p]
           y2 = np.arange(0, 25)*h + .5*h -self.cy[p]
           x2, y2 = np.meshgrid(x2, y2, indexing = "ij")
  	   samples2 = np.vstack((x2.flatten(), y2.flatten())).T
           Kxsx = self.kernel.value(samples2, self.samples)
	   K = np.dot(Kxsx, self.Kxxinv)
    	   
           g1.create_dataset(str(p), data = K)
	   g2.create_dataset(str(p), data = K[self.masks[i], :])

         f1.close()
         f2.close()
     
     def write_out_pars(self, step):
         """
         writing out updated parameters after
         every iteration
         """
         f = h5py.File("params"+str(step)+".hdf5" , "w")
         g1 = par.create_group("amp")
         g2 = par.create_group("bkg")
         g3 = par.create_group("cx")
         g4 = par.create_group("cy")
         g1.create_dataset("a", data = self.F)
         g2.create_dataset("a", data = self.B)
         g3.create_dataset("a", data = self.cx)
         g4.create_dataset("a", data = self.cy)
         f.close()                   
   
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
                          
           self.B[i] = np.array([self.d2d[i,m/2-4:m/2+5,-1:].mean(),
	                         self.d2d[i,m/2-4:m/2+5,:1].mean(),
                                 self.d2d[i,:1,m/2-4:m/2+5].mean(),
                                 self.d2d[i,-1:,m/2-4:m/2+5].mean()]).mean()
          
           self.dm[i] = self.data[i]-self.B[i]
           self.dm -= self.dm.min()
           self.F[i] = np.sum(self.dm[i])
           self.dm[i] /= self.F[i]                    
           shifted = shifter.shifter(self.dm[i], self.dx[i], self.dy[i])
           obs = ndimage.interpolation.zoom(shifted.reshape(25,25), 
					    self.H, output = None, order=3, 
                                            mode='constant', cval=0.0, 
					    prefilter=True).flatten()
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

     def lsq_update_FB(self):

         """least square optimization of F&B"""
         for p in range(self.N):
           Kp = np.array(K[str(p)])
           mask = self.masks[p]
           Y = self.data[p]
           old_flux, old_back = self.F[p], self.B[p]
           self.X = np.exp(self.lnX)
           psf = np.dot(self.X + self.fl, Kp)
         
         for i in range(10):
           old_back, old_flux = fit_single_patch(Y, mask, psf, 
                                                 old_flux, old_back, 
                                                 self.f, self.g)
         self.B[p], self.F[p] = old_back, old_flux

     
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
