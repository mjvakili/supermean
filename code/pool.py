from multiprocessing import pool
import time
import numpy as np
def f(x):
  return np.log(x)
y_serial=[]
x = range(100)

