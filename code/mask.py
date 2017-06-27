import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt


mask = pf.open("wfc3_f160w_clean_central100.fits")[1].data[20:28,:]
mask = mask.reshape(mask.shape[0],25,25)
#print mask.reshape(25,25)
for i in range(mask.shape[0]):
 plt.imshow(mask[i], interpolation = "None")
 plt.colorbar()
 plt.show()
