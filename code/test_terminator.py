import numpy as np
import terminator

A = np.random.normal(0,1,(100,100))
#print A
B = np.ones_like(A)
#print B
fl = 1e-3

import time

a = time.time()

def fun(A,
         B,
         fl):

    return _fun(A,B,fl)

def _fun(A,B,fl):
    a = A.shape[0] 
    obj = 0.0
    #for i1 in xrange(0, a):
    #  obj  += np.sum(np.dot(A,B))
    
    return np.dot(A,B)
fun(A,B,fl)

print time.time() - a

x = fun(A, B, fl)

f = terminator.npmultiply

b = time.time()

f(A,B)

print time.time()-b
print x - f(A,B)
"""
from timeit import timeit
timeit -n2 -r3 fun(A,B,fl)

timeit -n2 -r3 f(A,B)"""
