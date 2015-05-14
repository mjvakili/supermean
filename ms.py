import numpy as np
import interp as inter

_pi = inter.PI

###initializing the pi functions#####

def _phi(x):

  p = _pi(2+x),_pi(1+x),_pi(x),_pi(-1+x),_pi(-2+x)
  return p

def imext(y):

  """ input = 
  image model """

  n, m = y.shape[0], y.shape[1]
  yy = np.zeros((n+2, m+2))
  yy[1:-1,1:-1] = y
  
  return yy


def ext(y):
   n = int(np.sqrt(y.shape[0]))
   y = y.reshape(n,n)
   yy = np.zeros((n+2, n+2))
   yy[1:-1,1:-1] = y
   return yy.reshape(n+2,n+2)  

def B(n):

  """ n+1 = number of rows 
      (& colomns) of the
  model"""
  
  b = np.zeros((n+2,n+2))
  b[0,0:3] = np.array([1, -2, 1]) #first row of b matrix
  b[-1,-3:] = np.array([1, -2, 1]) #last row of b matrix
  for i in range(1,n+1):
    b[i , i-1:i-1+3] = np.array([1,4,1])

  return b

def I(n):
   b = np.zeros((n+2,n))
   b[1:-1] = np.eye(n)
   return b

def phi(x, n):

  """ x = amount of shift
  in either x(or y) direction
  """
  p1,p2,p3,p4,p5 = _phi(x)[0], _phi(x)[1], _phi(x)[2], _phi(x)[3], _phi(x)[4]

  matrix = np.zeros((n+2,n+2))
  matrix[1,0:4] = np.array([p2,p3,p4,p5])
  matrix[-2,-4:] = np.array([p1,p2,p3,p4])
  #print np.array([p1,p2,p3,p4,p5])
  for i in range(2, n-1):
    #print i
    #print matrix[i, i-2:i-2+5]
    matrix[i, i-2:i-2+5] = np.array([p1,p2,p3,p4,p5])
  return matrix#s[1:-1,1:-1]eme
