import numpy as np
import ms
import shifter
import sampler
import numpy as np
import scipy.optimize as op
from scipy import ndimage
f = .01
g = .05
class stuff(object):
   
     def __init__(self, data,  mask, H = 3, epsilon = .01 , min_iter=5, max_iter=10, check_iter=5 , tol=1.e-8):

        """ inputs of the code: NxD data matrix and NxD uncertainty matrix;
            N = the number of stars
            D = the number of pixels in each star                             
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.H = H                       #upsampling factor
        self.epsilon = epsilon           #smoothness parameter                                  
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        self.mask = mask                #variance with the same dimensiona as the data
        f = .01
        g = .05
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
        m = int((self.D)**.5)*self.H
        self.X = self.X.reshape(m,m)
        self.X[self.X<0]=np.median(self.X)
        #self.X[0:1,:]*=0
        #self.X[-1:,:]*=0
        #self.X[:,0:1]*=0
        #self.X[:,-1:]*=0
        self.X = self.X.flatten()
        
     def grad_X(self , params , *args):

        b  = int((self.D)**.5)
        Z = self.X.reshape((self.H*b, self.H*b))
        c=np.zeros_like(Z)
        c[:,:-1] += Z[:, 1:]
        c[:, 1:] += Z[:,:-1]
        c[1:, :] += Z[:-1,:]
        c[:-1,:] += Z[1:, :]
        grad = 2.*self.epsilon*(4.*Z - c).flatten() 
        self.F, self.B = args
        self.X = params
        #grad = np.zeros_like(self.X)
        for p in range(self.N):
         Kp = sampler.imatrix(self.data[p,:],self.H)
         modelp = self.F[p]*np.dot(self.X,Kp) - self.B[p]
         residualp = self.data[p] - modelp
         varp = f + g*modelp
         gradp = -1.*self.F[p]*Kp 
         gradp = gradp*(residualp/varp - (g/2.)*(varp**-1. - residualp**2./varp**2.))[None,:]
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
         varp = f + g*(self.F[p]*nmodelp + self.B[p])
         gradp = -1.*nmodelp
         grad[p] = np.sum(residualp*gradp/varp) +(g/2.)*np.sum(nmodelp/varp) - (g/2.)*np.sum(nmodelp*residualp**2./varp**2.)
        return grad
      
     def grad_B(self, params, *args):

        self.X, self.F = args
        self.B = params
        grad = np.zeros_like(self.B)
        for p in range(self.N):
         Kp = sampler.imatrix(self.data[p,:],self.H)
         modelp = self.F[p]*np.dot(self.X,Kp) + self.B[p]
         varp = f+g*modelp
         residualp = self.data[p] - modelp
         grad[p] = -1.*np.sum(residualp/varp) - (g/2.)*np.sum(residualp**2./varp**2.) + (g/2.)*np.sum(varp**-1.)        
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
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=20)
        print x
        self.X  = x[0]

     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.X, self.B), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=40)
        print x
        self.F  = x[0]

     def bfgs_B(self):
        x = op.fmin_l_bfgs_b(self.func_B,x0=self.B, fprime = self.grad_B,args=(self.X, self.F), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=20)
        print x
        self.B  = x[0]
 
     def nll(self):
   
       
       b  = int((self.D)**.5)
       Z = self.X.reshape((self.H*b, self.H*b))
       nll = self.epsilon*((Z[:,1:]-Z[:,:-1])**2.).sum() + self.epsilon*((Z[1:,:]-Z[:-1,:])**2.).sum() 
       for i in range(self.N):
         Ki = sampler.imatrix(self.data[i,:],self.H)
         #print np.dot(self.A[i,:],self.G).shape
         #print Ki.shape
         model_i = self.F[i]*np.dot(self.X, Ki) + self.B[i]
         model_square = model_i#.reshape(b,b)#[2:-2,2:-2]
         data_square = self.data[i,:]#.reshape(b,b)#[2:-2,2:-2]
         var_i = f + g*model_i
         nll += 0.5*np.sum(((model_square - data_square)**2.)/var_i) + .5*np.sum(np.log(var_i+self.epsilon))
       return nll
     
     def update(self, max_iter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        chi = []
        chi.append(self.nll())
        for i in range(max_iter):



            #oldobj = self.nll()
            self.bfgs_F()
            #obj = self.nll()
            #print "delta NLL after f STEP" , obj - oldobj
            #assert (obj < oldobj)or(obj == oldobj)
            #print "NLL after F-step", obj
            
            #oldobj = self.nll()
            self.bfgs_X()
            #obj = self.nll()
            #assert (obj < oldobj)or(obj == oldobj)
            #print "delta NLL after G STEP" , obj - oldobj
            #print "NLL after G-step", obj

            oldobj = self.nll()
            self.bfgs_B()
            obj = self.nll()
            #assert (obj < oldobj)or(obj == oldobj)
            #print "delta NLL after B STEP" , obj - oldobj
            #print "NLL after B-step", obj
            
            np.savetxt("wfc_mean_10%d.txt"%(i)       , self.X ,fmt='%.12f')
            np.savetxt("wfc_flux_10%d.txt"%(i)       , self.F ,fmt='%.12f')
            np.savetxt("wfc_background_10%d.txt"%(i) , self.B ,fmt='%.12f')
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
