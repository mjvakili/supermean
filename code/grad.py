import numpy as np
import ms
import shifter
import sampler
import numpy as np
import scipy.optimize as op
from scipy import ndimage

class stuff(object):
   
     def __init__(self, data,  H = 3, epsilon = .01 , min_iter=5, max_iter=10, check_iter=5 , tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD uncertainty matrix;
            N = the number of stars
            D = the number of pixels in each star                             
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.H = H                       #upsampling factor
        self.epsilon = epsilon           #smoothness parameter                                  
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        #self.ivar = ivar                #variance with the same dimensiona as the data
        
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
        
        m = int((self.D)**.5)
        self.d2d = self.data.reshape(self.N , m , m)
        self.dm = np.zeros((self.N, self.D))
        X = np.zeros_like(self.lnX)
        for i in range(self.N):
                          
          self.B[i]   =  np.array([self.d2d[i,m/2-4:m/2+5,-1:].mean(),self.d2d[i,m/2-4:m/2+5,:1].mean(),self.d2d[i,:1,m/2-4:m/2+5].mean(),self.d2d[i,-1:,m/2-4:m/2+5].mean()]).mean()
          
          self.dm[i]  =  self.data[i]-self.B[i]
          self.dm    -= self.dm.min()
          self.F[i]   =  np.sum(self.dm[i])
          self.dm[i] /=  self.F[i]                    
          shifted = shifter.shifter(self.dm[i], self.dx[i], self.dy[i])
          obs = ndimage.interpolation.zoom(shifted.reshape(25,25), self.H, output = None, order=3, mode='constant', cval=0.0, prefilter=True).flatten()
          
          
          X += obs.flatten()   
          
        X /= self.N
	X = X.flatten()
        self.lnX = np.log(X)
   
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
         Kp = sampler.imatrix(self.data[p,:], self.H)
         nmodelp = np.dot(self.X,Kp)
         residualp = self.data[p] - self.F[p]*nmodelp -self.B[p]
         gradp = -1.*nmodelp
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
        return self.nll_X(self.X, self.F, self.B )
    
     def nll_X(self):
   
       
       b  = int((self.D)**.5)

       nll = 0.
       for i in range(self.N):
         Ki = sampler.imatrix(self.data[i,:],self.H)
         model_i = self.F[i]*np.dot(self.X[None,:], Ki) + self.B[i]
         model_square = model_i.reshape(b,b)
         data_square = self.data[i,:].reshape(b,b)
         nll += 0.5*np.sum((model_square - data_square)**2.)
       return nll
     
     def func_F(self , params, *args):
        self.X, self.B = args
        self.F = params
        return self.nll(self.X, self.F, self.B )

     def func_B(self, params, *args):
        self.X, self.F = args
        self.B = params
        return self.nll(self.X, self.F, self.B )
     
     def nll(self , *args):

   
       self.X, self.F, self.B = args       
       b  = int((self.D)**.5)
       
       nll = 0.
       for i in range(self.N):
         Ki = sampler.imatrix(self.data[i,:],self.H)
         model_i = self.F[i]*np.dot(self.X, Ki) + self.B[i]
         model_square = model_i.reshape(b,b)
         data_square = self.data[i,:].reshape(b,b)
         nll += 0.5*np.sum((model_square - data_square)**2.)
       return nll
     
     def numgrad_X(self):

        h = 1e-6
        return (self.func_X(self.X+h ,self.F, self.B) - self.func_X(self.X-h ,self.F, self.B))/(2*h) 
     def numgrad_F(self):

        h = 5.e-5
        return (self.func_F(self.F+h ,self.X, self.B) - self.func_F(self.F-h ,self.X, self.B))/(2*h)    
    
 
     def test_grad(self, max_iter, check_iter, min_iter, tol):
  
        chi = []
        for i in range(max_iter):
         chi=[]
         for h in  (10.)**np.linspace(-4,4,40):

          
          q = self.grad_F(self.F , self.X, self.B).sum()
          self.F += h
          nll1 = self.nll(self.X , self.F, self.B).copy()
          self.F -= 2.*h
          nll2 = self.nll(self.X , self.F, self.B)
          qq = (nll1-nll2)/(2.*h)
          chi.append(np.abs(qq-q)/(np.abs(qq)+np.abs(q)))
         chi = np.array(chi)
         x =  10.**np.linspace(-4,4,40)
         import pylab as p
         p.loglog(x, chi)
         p.xlabel("h")
         p.ylabel("relative gradient error")
         p.show()
        
           
