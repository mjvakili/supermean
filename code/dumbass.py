import numpy as np

def f(x):

 return ((x-5.)**2.).sum(axis=1)


x = np.arange(10)

def ngrad(x,h):

  return (f(x[:,None]+h)-f(x[:,None]-h))/(2.*h)

def agrad(x):

  return 2.*(x-5.)


print ngrad(x,1e-6), agrad(x)

 
