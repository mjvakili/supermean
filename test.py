import pyfits as pf
from mean import stuff


a = pf.open("wfc3_f160w_clean_central100.fits")[0]
h = stuff(a, 3 , min_iter=1, max_iter=30, check_iter=1, tol=1.e-14)

