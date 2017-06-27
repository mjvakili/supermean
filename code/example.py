import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

d = pf.open("f160w_25_457_557_457_557_pixels.fits")[0].data

for i in range(1000,2000):
    plt.imshow(d[i,:].reshape(25,25) , interpolation = "None", norm = LogNorm() , origin = "lower" , cmap = plt.cm.Greys_r)
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.show()
    plt.close()
