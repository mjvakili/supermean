#hdfyinh data, mask, and masked data

import h5py
import pyfits as pf
import numpy as np

data  = pf.open("wfc3_f160w_clean_central100.fits")[0].data[0:120,:]
masks  = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]
print data.dtype
#sampling_file = h5py.File('sampler.h5','r') 
#K = sampling_file["MyDataset"]
#print K.shape
masks[masks == 0] = -1
masks[masks > 0] = False
masks[masks < 0] = True
masks = masks == True

f1 = h5py.File("masked_data.hdf5", 'w')         # hdf5 file format (open file)
#f2 = h5py.File("masked_sampler.hdf5", 'w')
f3 = h5py.File("data_and_mask.hdf5", 'w')
grp1 = f1.create_group('masked_data')     # create group
#grp2 = f2.create_group('masked_samplers')
grp3 = f3.create_group('data_mask')
grp3.create_dataset("data", data = data)#(data.shape[0],data.shape[1]))

grp3.create_dataset("mask", data = masks)

f3.close()

columns = []
for i in range(data.shape[0]):
  columns.append(str(i))

for i in range(data.shape[0]):

  #masked_data.append(data[i][masks[i]])
  masked_datum_i = data[i][masks[i]]
  #masked_sampler_i = K[i][:,masks[i]]
  #print masked_sampler_i.shape
  #print masked_datum_i
  #shape_i = masked_datum_i.shape[0]
  
  grp1.create_dataset(columns[i], data = masked_datum_i)# (shape_i,), dtype='f')
  #f2.create_dataset(columns[i], masked_sampler_i)
  #grp["qw"] = masked_datum_i
  #print (shape_i)
f1.close()
#f2.close()
  
#print masked_data
"""
if self.sfr is None:
     # output columns
     columns = ['mass', 'parent', 'child', 'ilk', 'snap_index']
elif 'columns' in kwargs.keys():
     columns = kwargs['columns']
else:       # if SFR/SSFR have been assigned
            # output columns
     columns = ['mass', 'sfr', 'ssfr', 'gal_type',
                     'parent', 'child', 'ilk', 'snap_index']

n_cols = len(columns)       # number of columns
col_fmt = []                # column format

# save each column to dataset within 'cenque_data' group
for datum in data:
     column_attr = getattr(self, column)
     # write out file
     grp.create_dataset(column, data=column_attr )

# save metadata
metadata = [ 'nsnap', 'zsnap', 't_cosmic', 't_step' ]
for metadatum in metadata:
     if self.nsnap is not None:
grp.attrs[metadatum] = getattr(self, metadatum)

f.close()
"""

