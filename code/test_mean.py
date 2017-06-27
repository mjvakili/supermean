import pyfits as pf
import numpy as np
from superb import stuff

#from mean import stuff
#print pf.open("wfc3_f160w_clean_central100.fits")[0].data.shape

a  = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]
b  = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]
cx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")[0:120]
cy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")[0:120]

H = 3
epsilon = 10000.
f = 5e-2
fl = 0.0
g = 0.01
tr = np.loadtxt("tr.txt").astype(int)

ts = np.loadtxt("ts.txt").astype(int)

					
h = stuff(a, cx, cy, b, f, g, fl, H, epsilon, min_iter=1, max_iter=1, check_iter=1, tol=.0001)

