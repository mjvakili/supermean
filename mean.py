import numpy as np
import ms
import shifter
import sampler
import numpy as np
import scipy.optimize as op
from scipy import ndimage
f  = .05
g  = .01
fl = 1e-4

class stuff(object):
   
     def __init__(self, data, cx, cy, mask, H = 3, epsilon = .01 , min_iter=5, max_iter=10, check_iter=5 , tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD uncertainty matrix;
            N = the number of stars
            D = the number of pixels in each star                             
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.H = H                       #upsampling factor
        self.epsilon = epsilon           #smoothness parameter                                  
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        self.mask = mask                 #data quality
        self.dx = cx                     #list of centroid offsets x
        self.dy = cy                     #list of centroid offsets y   
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
        
        m = int((self.D)**.5)
        self.d2d = self.data.reshape(self.N , m , m)
        self.dm = np.zeros((self.N, self.D))
        X = np.zeros_like(self.lnX)
        for i in range(self.N):
                          
          self.B[i]   =  np.array([self.d2d[i,m/2-4:m/2+5,-1:].mean(),self.d2d[i,m/2-4:m/2+5,:1].mean(),self.d2d[i,:1,m/2-4:m/2+5].mean(),self.d2d[i,-1:,m/2-4:m/2+5].mean()]).mean()
          
          self.dm[i]  =  self.data[i]-self.B[i]
          self.F[i]   =  np.sum(self.dm[i])
          self.dm[i] /=  self.F[i]                    

          obs = ndimage.interpolation.zoom(self.dm[i].reshape(25,25), self.H, output = None, order=3, mode='constant', cval=0.0, prefilter=True).flatten()
          shifted_sr = shifter.shifter(obs)
          shifted_sr[shifted_sr<0] = np.median(shifted_sr)
          X += shifted_sr.flatten()   
          
        X /= self.N
        X[X<0] = fl                #setting the initial negative pixels in X to fl                           
        m = int((self.D)**.5)*self.H
        #self.X = self.X.reshape(m,m)
        #self.X[self.X<0]=1.#np.mean(self.X)
        #self.X[0:1,:]*=0
        #self.X[-1:,:]*=0
        #self.X[:,0:1]*=0
        #self.X[:,-1:]*=0
        X = X.flatten()
        self.lnX = np.log(X)
        
     def grad_lnX(self , params , *args):

        """Gradient w.r.t Log(X)"""

        self.F, self.B = args
        self.lnX = params

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
        
        #grad = np.zeros_like(self.X)
        for p in range(self.N):
         Kp = sampler.imatrix_new(self.M, self.H, self.dx[p], self.dy[p])
         modelp = self.F[p]*(self.X+fl).dot(Kp) + self.B[p]
         ep = self.data[p] - modelp
         varp = f + g*np.abs(modelp)
         gradp = -1.*self.F[p]*Kp
         
         gainp = (g/2.)*(varp**-1. - ep**2./varp**2.)

         gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid

         gradp = self.X[:,None]*gradp*(ep/varp - gainp)[None,:]
         Gradp = gradp.sum(axis = 1) 
         grad += Gradp
        return grad
     
     def grad_F(self, params, *args):

        """Gradient w.r.t F """

        self.lnX, self.B = args
        self.F = params
        
        self.X = np.exp(self.lnX)
        
        grad = np.zeros_like(self.F)
        
        for p in range(self.N):
        
         Kp = sampler.imatrix_new(self.M, self.H, self.dx[p], self.dy[p])
         nmodelp = np.dot(self.X+fl,Kp)
         modelp = self.F[p]*nmodelp + self.B[p]
         residualp = self.data[p] - modelp
         varp = f + g*np.abs(modelp)
         gradp = -1.*nmodelp
         gainp = (g/2.)*nmodelp*(varp**-1. - residualp**2./varp**2.)
         gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid
         grad[p] = np.sum(residualp*gradp/varp) + np.sum(gainp)
        return grad
      
     def grad_B(self, params, *args):
        
        """Gradient w.r.t B """

        self.lnX, self.F = args
        self.B = params

        self.X = np.exp(self.lnX)

        grad = np.zeros_like(self.B)
        for p in range(self.N):
         Kp = sampler.imatrix_new(self.M, self.H, self.dx[p], self.dy[p])
         modelp = self.F[p]*np.dot(self.X+fl,Kp) + self.B[p]
         varp = f+g*np.abs(modelp)
         residualp = self.data[p] - modelp
         gainp = - (g/2.)*(residualp**2./varp**2.) + (g/2.)*(varp**-1.)
         gainp[modelp<0] *= -1.   #var=f+g|model| to account for numerical artifacts when sr model is sampled at the data grid   
         grad[p] = -1.*np.sum(residualp/varp) + np.sum(gainp)      
        return grad
     


     def func_lnX(self , params, *args):
        self.F, self.B = args
        self.lnX = params   
        return self.nll()
     
     def func_F(self , params, *args):
        self.lnX, self.B = args
        self.F = params
        return self.nll()

     def func_B(self, params, *args):
        self.lnX, self.F = args
        self.B = params
        return self.nll()
     

     def bfgs_lnX(self):
        x = op.fmin_l_bfgs_b(self.func_lnX,x0=self.lnX, fprime = self.grad_lnX,args=(self.F, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-04, epsilon=1e-04, maxfun=20)
        print x
        self.lnX  = x[0]

     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.lnX, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-04, epsilon=1e-04, maxfun=20)
        print x
        self.F  = x[0]

     def bfgs_B(self):
        x = op.fmin_l_bfgs_b(self.func_B,x0=self.B, fprime = self.grad_B,args=(self.lnX, self.F), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-04, epsilon=1e-04, maxfun=20)
        print x
        self.B  = x[0]
 
     def nll(self):

       self.X = np.exp(self.lnX)
       b  = int((self.D)**.5)
       Z = self.X.reshape((self.H*b, self.H*b))
       nll = self.epsilon*((Z[:,1:]-Z[:,:-1])**2.).sum() + self.epsilon*((Z[1:,:]-Z[:-1,:])**2.).sum() 
       for i in range(self.N):
         Ki = sampler.imatrix_new(self.M, self.H, self.dx[i], self.dy[i])
         model_i = self.F[i]*np.dot(self.X+fl, Ki) + self.B[i]
         model_square = model_i
         data_square = self.data[i,:]
         var_i = f + g*np.abs(model_i)
         
         nll += 0.5*np.sum(((model_square - data_square)**2.)/var_i) + .5*np.sum(np.log(var_i))
       return nll
     
     def update(self, max_iter, check_iter, min_iter, tol):
      
        np.savetxt("wfc_mean_iter_0.txt"       , self.lnX ,fmt='%.12f')
        np.savetxt("wfc_flux_iter_0.txt"       , self.F ,fmt='%.12f')
        np.savetxt("wfc_background_iter_0.txt" , self.B ,fmt='%.12f')

        print 'Starting NLL =', self.nll()
        nll = self.nll()
        chi = []
        chi.append(self.nll())
        for i in range(max_iter):



            self.bfgs_F()
            obj = self.nll()
            print "NLL after F-step", obj
            self.bfgs_lnX()
            obj = self.nll()
            print "NLL after X-step", obj
            self.bfgs_B()
            obj = self.nll()
            print "NLL after B-step", obj
            
            np.savetxt("wfc_mean_iter_%d.txt"%(i+1)       , self.lnX ,fmt='%.12f')
            np.savetxt("wfc_flux_iter_%d.txt"%(i+1)       , self.F ,fmt='%.12f')
            np.savetxt("wfc_background_iter_%d.txt"%(i+1) , self.B ,fmt='%.12f')
            chi.append(obj)
                        

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
        print chi
