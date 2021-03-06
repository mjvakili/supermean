from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
#from libc.math cimport log


DTYPE = np.double
ctypedef np.double_t DTYPE_t

def njj(np.ndarray[DTYPE_t, ndim=2, mode="c"] data,
        np.ndarray[DTYPE_t, ndim=2, mode="c"] mask,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] F,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] lnX,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] B,
        double epsilon,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dx,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dy,
        np.ndarray[DTYPE_t, ndim=3, mode="c"] K,
        int H,
        double f, 
        double g,
        double fl,):
    return _njj(data, mask, F, lnX, B, epsilon, dx, dy, K, H, f, g, fl)


@cython.boundscheck(False)
cdef double _njj(np.ndarray[DTYPE_t, ndim=2, mode="c"] data,
        np.ndarray[DTYPE_t, ndim=2, mode="c"] mask,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] F,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] lnX,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] B,
        double epsilon,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dx,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dy,
        np.ndarray[DTYPE_t, ndim=3, mode="c"] K,
        int H,
        double f, 
        double g,
        double fl,):
        
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] X = np.exp(lnX)
    cdef int N = data.shape[0]
    cdef int D = data.shape[1]
    cdef DTYPE_t [:] model = np.empty(D, dtype=DTYPE)
    cdef DTYPE_t [:] residual = np.empty(D, dtype=DTYPE)
    cdef DTYPE_t [:] var = np.empty(D, dtype=DTYPE)
    cdef DTYPE_t [:] chi = np.empty(D, dtype=DTYPE)
    cdef DTYPE_t [:,:] Ki = np.empty((D*H, D), dtype=DTYPE)
    cdef int i, v
    cdef double obj = 0.0
    for i in range(N):
       
       Ki = K[i]
       model[:] = F[i]*np.dot(X+fl, Ki) + B[i]
       residual[:] = data[i] - model[:]
       var = f + g*np.abs(model)
      
       
       for v in range(D):
         if (mask[i,v] != 0 ):
           residual[v] = 0
         chi[v] = residual[v]*residual[v]
         chi[v] /= var[v]
       obj += .5*np.sum(np.log(var)) + 0.5*np.sum(chi)
      
    return obj

def nll(np.ndarray[DTYPE_t, ndim=2, mode="c"] data,
        np.ndarray[DTYPE_t, ndim=2, mode="c"] mask,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] F,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] lnX,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] B,
        double epsilon,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dx,
        np.ndarray[DTYPE_t, ndim=1, mode="c"] dy,
        np.ndarray[DTYPE_t, ndim=3, mode="c"] K,
        int H,
        double f, 
        double g,
        double fl,):
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] X = np.exp(lnX)
    cdef int N = data.shape[0]
    cdef int D = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] residual = np.empty(D, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] model = np.empty(D, dtype=DTYPE)
    cdef int i
    cdef double objj = 0.0
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] var = np.empty(D, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,  ndim=2, mode="c"] Ki = np.empty((D*H, D), dtype=DTYPE)

    for i in range(N):
    
       Ki = K[i]
       model = F[i]*np.dot(X+fl, Ki) + B[i]
       
       residual = data[i] - model
       var = f + g*np.abs(model)
      
       residual[mask[i]!=0.0] = 0.0
       objj +=  0.5*np.sum(np.log(var))+ 0.5*np.sum(((residual)**2.)/var) +
      
    return objj
