import h5py
import numpy as np
import sampler
import time
import pyfits as pf

cx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")
cy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")
masks  = pf.open("wfc3_f160w_clean_central100.fits")[1].data[0:120,:]
masks[masks == 0] = -1
masks[masks > 0] = False
masks[masks < 0] = True
masks = masks == True


N = cx.shape[0]
H = 3
M = 25

s1 , s2 = sampler.imatrix_new(M, H, cx[0], cy[0]).shape
#F = h5py.File('sampler.h5')                                         #saving 
#dset = F.create_dataset("MyDataset" , (N , s1 , s2) , 'f8') 
F = h5py.File('samplerx3.hdf5','w') 
G = h5py.File('masked_samplerx3.hdf5','w')

#print F                  #loading file

grp = F.create_group("samplerx3")
Grp = G.create_group("masked_samplerx3")

for p in range(120):
  #a = time.time()
  dset = sampler.imatrix_new(M, H, cx[p], cy[p])
  masked_dset = dset[:,masks[p]]
  grp.create_dataset(str(p), data = dset)
  #print grp[str(p)]
  Grp.create_dataset(str(p), data = masked_dset)
  #print Grp[str(p)]
  #Kp = DATA[p]
  #print Kp
  #print Kp.shape
  #plt.imshow(Kp)
  #plt.show()
  #print time.time()-a
  #dset[p] = Kp 
F.close()
G.close()
