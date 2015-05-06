import numpy as np
import ms
import shifter
import sampler
import numpy as np
import scipy.optimize as op
from scipy import ndimage

class stuff(object):
   
     def __init__(self, data,  H = 3, min_iter=5, max_iter=10, check_iter=5 , tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD uncertainty matrix;
            N = the number of stars
            D = the number of pixels in each star                             
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.H = H                       #upsampling factor          
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        #self.ivar = ivar                 #variance with the same dimensiona as the data
        
        """ outputs of the code:
                                 H*H*D-dimensional mean vector:  X
                                 N-dimensional flux vector:      F
                                 1-dimensional flat-field background: B
        """
               
        self.X = np.zeros((self.D*self.H*self.H))   #Creating the mean matrix which is a H^2*D-dimensional object!
        self.F = np.zeros((self.N))                 #Creating an N-dimensional Flux vector.

        #self.B = 0.0                                #Creating an 1-dimensional background vector.
        self.B = np.zeros((self.N))                 #one flat-field per star 
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
        
        
        #self.dm = self.data - self.B
        #self.F  = np.sum(self.dm, axis = 1)
        m = int((self.D)**.5)
        self.d2d = self.data.reshape(self.N , m , m)
        self.dm = np.zeros((self.N, self.D))

        for i in range(self.N):
                          
          self.B[i]   =  np.array([self.d2d[i,m/2-4:m/2+5,-1:].mean(),self.d2d[i,m/2-4:m/2+5,:1].mean(),self.d2d[i,:1,m/2-4:m/2+5].mean(),self.d2d[i,-1:,m/2-4:m/2+5].mean()]).mean()
          
          #print self.B[i] , self.d2d[i].min()
          self.dm[i]  =  self.data[i]-self.B[i]
          self.F[i]   =  np.sum(self.dm[i])
          self.dm[i] /=  self.F[i]                    

          #plt.imshow(self.dm[i].reshape(25,25), interpolation="None",norm = LogNorm())
          #plt.colorbar()
          #plt.show()
          obs = ndimage.interpolation.zoom(self.dm[i].reshape(25,25), self.H, output = None, order=3, mode='constant', cval=0.0, prefilter=True).flatten()
          self.X  =  self.X + shifter.shifter(obs)   
          #plt.imshow(shifter.shifter(obs).reshape(75,75), interpolation="None",norm = LogNorm())
          #plt.colorbar()
          #plt.show()
        self.X /= self.N                           #take the mean of the upsampled normalized shifted ff-subtracted stars
        #self.X[(self.X < 0)] = np.mean(self.X)       #replace negative mean brightnesses with mean pixel-brightness of the initial X  
        #plt.imshow(self.X.reshape(75,75), interpolation="None" , norm = LogNorm())
        #plt.colorbar()
        #plt.show()

     def grad_X(self , params , *args):
        
        self.F, self.B = args
        self.X = params
        grad = np.zeros_like(self.X)
        for p in range(self.N):
         Kp = sampler.imatrix(self.data[p,:],self.H)
         residualp = self.data[p] - self.F[p]*np.dot(self.X,Kp) - self.B[p]
         gradp = -1.*self.F[p]*Kp
         gradp = gradp*residualp[None,:]
         Gradp = gradp.sum(axis = 1) 
         grad += Gradp
        return grad
     
     def grad_F(self, params, *args):

        self.X, self.B = args
        self.F = params
        grad = np.zeros_like(self.F)
        for p in range(self.N):
         Kp = sampler.imatrix(self.data[p,:],self.H)
         residualp = self.data[p] - self.F[p]*np.dot(self.X,Kp)
         gradp = -1.*np.dot(self.X,Kp)
         grad[p] = np.sum(residualp*gradp)
        return grad
      
     def grad_B(self, params, *args):

        self.X, self.F = args
        self.B = params
        grad = np.zeros_like(self.B)
        for p in range(self.N):
         Kp = sampler.imatrix(self.data[p,:],self.H)
         residualp = self.data[p] - self.F[p]*np.dot(self.X,Kp) - self.B[p]
         grad[p] = -1.*np.sum(residualp)        
        return grad
     


     def func_X(self , params, *args):
        self.F, self.B = args
        self.X = params   
        return self.nll()
     
     def func_F(self , params, *args):
        self.X, self.B = args
        self.F = params
        return self.nll()

     def func_B(self, params, *args):
        self.X, self.F = args
        self.B = params
        return self.nll()
     

     def bfgs_X(self):
        x = op.fmin_l_bfgs_b(self.func_X,x0=self.X, fprime = self.grad_X,args=(self.F, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        print x
        self.X  = x[0]

     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.X, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        print x
        self.F  = x[0]

     def bfgs_B(self):
        x = op.fmin_l_bfgs_b(self.func_B,x0=self.B, fprime = self.grad_B,args=(self.X, self.F), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        print x
        self.B  = x[0]
 
     def nll(self):
   
       nll = 0.

       for i in range(self.N):
         Ki = sampler.imatrix(self.data[i,:],self.H)
         #print np.dot(self.A[i,:],self.G).shape
         #print Ki.shape
         model_i = self.F[i]*np.dot(self.X, Ki) + self.B[i]
         b  = int((self.D)**.5)
         model_square = model_i.reshape(b,b)
         data_square = self.data[i,:].reshape(b,b)
         nll += 0.5*np.sum((model_square - data_square)**2.)
       return nll
     
     def update(self, max_iter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        chi = []
        chi.append(self.nll())
        for i in range(max_iter):

            np.savetxt("wfc_mean_10%d.txt"%(i)       , self.X ,fmt='%.12f')
            np.savetxt("wfc_flux_10%d.txt"%(i)       , self.F ,fmt='%.12f')
            np.savetxt("wfc_background_10%d.txt"%(i) , self.B ,fmt='%.12f')

            oldobj = self.nll()
            self.bfgs_F()
            obj = self.nll()
            print "delta NLL after f STEP" , obj - oldobj
            assert (obj < oldobj)or(obj == oldobj)
            print "NLL after F-step", obj
            
            oldobj = self.nll()
            self.bfgs_X()
            obj = self.nll()
            assert (obj < oldobj)or(obj == oldobj)
            print "delta NLL after G STEP" , obj - oldobj
            print "NLL after G-step", obj

            oldobj = self.nll()
            self.bfgs_B()
            obj = self.nll()
            assert (obj < oldobj)or(obj == oldobj)
            print "delta NLL after B STEP" , obj - oldobj
            print "NLL after B-step", obj
            
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
