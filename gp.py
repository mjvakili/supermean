import numpy as np
import scipy.optimize as op
from scipy import ndimage
import h5py
import time
from scipy.linalg import cho_factor, cho_solve
from interruptible_pool import InterruptiblePool
from nll_grad import nll_grad_X
from nll_grad_fb import v3_fit_single_patch , v4_fit_single_patch
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from matplotlib.colors import LogNorm
from george.kernels import ExpSquaredKernel , WhiteKernel

def fg(args):
    return nll_grad_X(*args)

class EM(object):
   
     def __init__(self, data, masks, X, flux, bkg, cx, cy, Nthreads = 20,
                        f = 5.e-2, g = 1.e-2, epsilon = 1e-2,
                        min_iter=5, max_iter=10, check_iter=1, 
                        tol=1.e-8, precompute = False,
                        precompute_alpha = False):

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
         self.f = f                         #floor variance of noise model
         #self.fl = fl			   #floor of the PSF model
         self.g = g                         #gain 
	 self.X = X       
         self.F = flux                 #Creating an N-dimensional Flux vector.
         self.B = bkg
         self.cx = cx
         self.cy = cy
         self.Nthreads = Nthreads
         self.epsilon = epsilon
         self.precompute = precompute
	 self.precompute_alpha = precompute_alpha
         """setting up objects that remain constant throughout optimization"""
         self.sr_grid()
         self.Kernel()
         self.get_matrix()
         self.get_inv_matrix()
               

         self.run_EM(max_iter, check_iter, min_iter, tol)
        
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
         self.kernel = ExpSquaredKernel(.001,ndim=2) + WhiteKernel(.01,ndim=2)

     def get_matrix(self):
         """
         computing Kxx matrix, note: this remains constant
         """
         self.Kxx = self.kernel.value(self.samples , self.samples)
     
     def get_inv_matrix(self):
         """
         computing the inverse of Kxx, this also remains constant
         """
         if (self.precompute == False):
    	   self.Kxxinv = np.linalg.solve(self.Kxx , np.eye(self.X.shape[0])) 
           f = h5py.File("kxxinv.hdf5","w")
           g = f.create_group("kxxinv")
           g.create_dataset("a", data = self.Kxxinv)
           f.close()
         else:
           f = h5py.File("kxxinv.hdf5","r")
           self.Kxxinv = f["kxxinv"]["a"][:]
           f.close()
          
     def alpha_mean_initial(self):
         """
         computing the initial alpha
         """
         if (self.precompute_alpha==False):
           self.mean = np.mean(self.X)
           Xmm = self.X - self.mean
           self.alpha = np.linalg.solve(self.Kxx , Xmm)
           f = h5py.File("init_alpha.hdf5","w")
           g = f.create_group("alpha")
           g.create_dataset("a", data = self.alpha)
           f.close()
         else:
           f = h5py.File("init_alpha.hdf5","r")
           self.alpha = f["alpha"]["a"][:]
           self.mean = np.mean(self.X)

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
         the function nll_grad_X in nll_grad.py
         needs the samplers
         """
         f1 = h5py.File("sampler.hdf5","w")
         f2 = h5py.File("masked_sampler.hdf5","w")
         g1 = f1.create_group("sampler")
 	 g2 = f2.create_group("masked_sampler")
          
    	 #self.Kxxinv = np.linalg.solve(self.Kxx , np.eye(self.X.shape[0])) 
    	 for p in range(self.N):
            
           h = 1./25
           x2 = np.arange(0, 25)*h + .5*h -self.cx[p]*h
           y2 = np.arange(0, 25)*h + .5*h -self.cy[p]*h
           x2, y2 = np.meshgrid(x2, y2, indexing = "ij")
  	   samples2 = np.vstack((x2.flatten(), y2.flatten())).T
           Kxsx = self.kernel.value(samples2, self.samples)
           #a = time.time()
	   K = np.dot(Kxsx, self.Kxxinv)
           #self.alpha = np.loadtxt("alpha.dat")
           #nm = np.dot(Kxsx, self.alpha)+ self.X.mean()
           #plt.imshow(nm.reshape(25,25), norm=LogNorm())
           #plt.savefig("/home/mj/public_html/nm"+str(p)+".png")
           #plt.close()
           #print time.time() - a 
           #g1.create_dataset(str(p), data = K)
	   #g2.create_dataset(str(p), data = K[self.masks[p], :])
           print p
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
         in case they are not provided
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
     
     def update_patch_FB(self, p):
         """
         updating the amplitude, and background
         of patch p
         """
         theta = self.B[p], self.F[p]
         h = 1./25
         Dx, Dy = self.cx[p], self.cy[p]
         x2 = np.arange(0, 25)*h + .5*h - Dx*h
         y2 = np.arange(0, 25)*h + .5*h - Dx*h
         x2, y2 = np.meshgrid(x2, y2, indexing = "ij")
  	 samples2 = np.vstack((x2.flatten(), y2.flatten())).T
         Kxsx = self.kernel.value(samples2, self.samples)
         psf = np.dot(Kxsx, self.alpha) + self.mean
         maskp = self.masks[p]
         masked_y = self.data[p][maskp!=0]
         masked_psf = psf[maskp!=0]
         grad_func = v3_fit_single_patch
         result = op.fmin_l_bfgs_b(grad_func, x0 = theta, fprime = None, \
                                   args=(masked_y,masked_psf,self.f,self.g),\
			           approx_grad = False, \
                                   bounds = [(1.e-10,100.),(1.e-2,10**7.)], \
                                   factr=10., pgtol=1.e-16, epsilon=1.e-16, \
			           maxfun=60)
         #print result[0]
         #print self.B[p], self.F[p]
         self.B[p], self.F[p] = result[0][0], result[0][1]
     
     def update_patch_centroid(self, p):
         """
         updating the sub-pixel centroid 
         shifts of patch p
         """
         theta = [self.cx[p] , self.cy[p]]
         func = v4_fit_single_patch
         result = op.fmin_powell(func, [theta[0],theta[1]], \
                                 args=(self.data[p],self.masks[p], \
                                       self.F[p], self.B[p], \
                                       self.alpha, self.mean, \
                                       self.f, self.g))
         print result[0]
         print self.cx[p], self.cy[p]
         self.cx[p], self.cy[p] = result[0][0], result[0][1]

     def update_FB_centroid(self):
         """
         updating all the backgrounds,
         amplitudes, and centroids
         """
         for p in xrange(self.N):
           self.update_patch_FB(p)
         for p in xrange(self.N):
           self.update_patch_centroid(p)

     def nll_grad_X(self, params):
         """
         Computing the NLL and its gradient wrt X,
         Computing the reg term and its gradient, 
         and adding them up.
         """
         self.X = params
	 Pool = InterruptiblePool(self.Nthreads)
         mapfn = Pool.map
         Nchunk = np.ceil(1. / self.Nthreads * self.N).astype(np.int)
         arglist = [None] * self.Nthreads
         for i in xrange(self.Nthreads):
           s = int(i * Nchunk)
           e = int(s + Nchunk)
	   arglist[i] = (self.X, self.F, self.B, \
                         self.f, self.g, s, e)
         #result = list(fg(ars) for ars in arglist)
         result = list(mapfn(fg, [ars for ars in arglist]))    
         nll, grad = result[0]
         for i in xrange(1,self.Nthreads):
           nll += result[i][0]
           grad += result[i][1]
         #print nll
         #print grad
         #plt.imshow(self.X.reshape(101,101), interpolation = "None", norm = LogNorm())
         #plt.savefig("/home/mj/public_html/x.png")
         #plt.close()
         Pool.close()
         Pool.terminate()
         Pool.join()
         reg_func, reg_grad = self.reg_func_grad_X() 
         return nll + reg_func, grad + reg_grad	

     def reg_func_grad_X(self):
         """ 
         returns regularization term
         and its derivative wrt X
         """ 
         b = int((self.X.shape[0])**.5)
         Z = self.X.reshape(b,b)
         c= np.zeros_like(Z)
         c[:,:-1] += Z[:, 1:]
         c[:, 1:] += Z[:,:-1]
         c[1:, :] += Z[:-1,:]
         c[:-1,:] += Z[1:, :]
         grad = 2.*self.epsilon*(4.*Z - c).flatten()*self.X 
         func = self.epsilon*np.sum((Z[:,1:]-Z[:,:-1])**2.) + \
                self.epsilon*np.sum((Z[1:,:]-Z[:-1,:])**2.)
         return func , grad         

     def update_X(self, maxfuncall):
         """
         updaing the super-resolution psf X
         optimizing the objective function,
         given the current values of the bkgs,
         amplitudes, and the centroids.
         """ 
         result = op.fmin_l_bfgs_b(self.nll_grad_X,x0=self.X,fprime = None,args=(), approx_grad = False,bounds=[(1e-5, 100.) for _ in self.X],factr=10.0, pgtol=1e-5, epsilon=1e-8)
         self.X = result[0]
     
     def run_EM(self, max_iter, check_iter, min_iter, tol):
         
         self.alpha_mean_initial()
         print "a"
         self.get_samplers()
         print "b"
         #nll = self.nll_grad_X()[0]
         #print "starting NLL is:", nll
          
         for t in range(1, max_iter+1):

           #self.update_FB_centroid()
	   self.update_X(20)
           self.write_out_pars()
           self.alpha_mean()
           self.get_samplers()


           if np.mod(i, check_iter) == 0:
                new_nll = self.nll_grad_X()[0]
                print 'NLL at step %d is:' % (i+1), new_nll
           if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
           else:
                nll = new_nll
         self.nll = new_nll
