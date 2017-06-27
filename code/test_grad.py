import pyfits as pf
from grad import stuff


a = pf.open("wfc3_f160w_clean_central100.fits")[0].data[20:22,:]

h = stuff(a, 4 , .01, min_iter=1, max_iter=2, check_iter=1, tol=.01)


