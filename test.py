import pyfits as pf
from mean import stuff

epsilon = .1
H = 5

a = pf.open("wfc3_f160w_clean_central100.fits")
data, mask = a[0].data, a[1].data
h = stuff(data, mask , H , epsilon ,min_iter=1, max_iter=30, check_iter=1, tol=1.e-14)

