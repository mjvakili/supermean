import pyfits as pf
import numpy as np
from mean import stuff


a  = pf.open("wfc3_f160w_clean_central100.fits")[0].data
b  = pf.open("wfc3_f160w_clean_central100.fits")[1].data
cx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")
cy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")

H = 5
epsilon = .1

h = stuff(a, cx, cy, b, H, epsilon, min_iter=1, max_iter=20, check_iter=1, tol=.01)

