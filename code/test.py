import interp
import numpy as np
import pylab as p

x = np.linspace(-100,100,10000)
y = interp.pi(x)


def f(x):
  #return 4 - 6*np.abs(x)**2 + 3 * np.abs(x)**3
  return (2.-np.abs(x))**3.


p.plot(x,y)
p.show()
