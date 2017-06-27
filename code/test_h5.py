import h5py
import numpy as np
import pyfits as pf
data  = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]

F = h5py.File("data&mask.hdf5",'r')
FF = h5py.File("masked_data.hdf5",'r')


FD = h5py.File('sampler.hdf5','r') 
GD = h5py.File('masked_sampler.hdf5','r')

r = np.array(F["data_mask"]["data"])
w=  np.array(F["data_mask"]["mask"])
zz=  np.array(FF["masked_data"]["0"])


kk = np.array(FD["sampler"]["0"])
print kk
#print w[1]
#print zz,zz.shape
