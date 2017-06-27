import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import h5py

i = np.exp(np.loadtxt("superb_wfc_mean_iter_0.txt").reshape(75,75))
f = np.exp(np.loadtxt("superb_wfc_mean_iter_6_nfljadid.txt").reshape(75,75))

from matplotlib.colors import LinearSegmentedColormap
y=(f-i)/i
print y.shape
"""
vmax,vmin = y.max(),y.min()
cmap = LinearSegmentedColormap.from_list('mycmap', [(vmin, 'red'),
                                                    ((vmin+vmax)/2, 'Grey'),
                                                    (vmax, 'blue')]
                                        )
"""
"""
im = ax.pcolor(data, cmap=cmap, vmin=0, vmax=vmax, edgecolors='black')
cbar = fig.colorbar(im)

cbar.set_ticks(range(4)) # Integer colorbar tick locations
ax.set(frame_on=False, aspect=1, xticks=[], yticks=[])
ax.invert_yaxis()
"""
#fig, ax = plt.subplots()

plt.imshow(f , interpolation="None" , norm = LogNorm())#,cmap = cmap)
#ax.set(frame_on=False, aspect=1, xticks=[], yticks=[])
#cbar = fig.colorbar(im)
plt.set_cmap('RdBu')
plt.title(r"$(\mathtt{final} - \mathtt{initial})/\mathtt{initial}$")
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.show()


F = np.loadtxt("superb_wfc_flux_iter_0_nfljadid.txt")#.reshape(75,75)
B = np.loadtxt("superb_wfc_bkg_iter_0_nfljadid.txt")#.reshape(75,75)

MS = h5py.File('masked_samplerx3.hdf5','r')
MD = h5py.File('masked_data.hdf5','r')
masked_samplers = MS["masked_samplerx3"]#[random]
masked_data = MD["masked_data"]#[random]

res = []
brightness = []

for i in range(120):
  modeli = F[i]*np.dot(f.flatten()+1e-5, masked_samplers[str(i)])+ B[i]
  
  resi = ((masked_data[str(i)] - modeli)**2.).flatten()
  
  for j in range(resi.shape[0]):
    res.append(resi[j])
    brightness.append(modeli[j])

#print brightness
"""
res = np.array(res).flatten()
brightnes = np.array(brightness).flatten()
"""
bs = np.argsort(np.array(brightness))
fig = plt.figure()
ax = fig.add_subplot(111)

#print bs
ax.loglog(np.array(brightness)[bs] , np.array(res)[bs] , 'ko'  , alpha = .1)
ax.loglog(np.array(brightness)[bs] , .05*np.array(brightness)[bs] , 'r'  , alpha = 1.)
xmin , xmax =  np.array(brightness)[bs].min() , np.array(brightness)[bs].max()
#xmin , xmax =
#ax.set_xlim((xmin , xmax))
#scatter(aaaa , bbbb**.5 ,  s=1 , c = 'y' , marker='+'  , alpha = .2)
ax.set_xlabel(r"$\mathtt{Brightness-of-the-pixel-model}$")
ax.set_ylabel(r"$\mathtt{squared-residual}$")
#plt.gca().set_aspect('equal', adjustable='box')
#ax.set_autoscaleon('False')
fig.set_size_inches(7,7)
plt.show()
#plt.savefig("gain.png" , dpi=200)

#print xmax - xmin

#print .05*np.array(brightness)[bs].max() - .05*np.array(brightness)[bs].min()

"""
Colormap BuRd is not recognized. Possible values are: Spectral, summer, coolwarm, Wistia_r, pink_r, Set1, Set2, Set3, brg_r, Dark2, prism, PuOr_r, afmhot_r, terrain_r, PuBuGn_r, RdPu, gist_ncar_r, gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern, PuBu_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r, gist_rainbow, gist_heat_r, Wistia, OrRd_r, CMRmap, bone, gist_stern_r, RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, winter_r, PuBu, RdGy_r, spectral, rainbow, flag_r, jet_r, RdPu_r, gist_yarg, BuGn, Paired_r, hsv_r, bwr, cubehelix, Greens, PRGn, gist_heat, spectral_r, Paired, hsv, Oranges_r, prism_r, Pastel2, Pastel1_r, Pastel1, gray_r, jet, Spectral_r, gnuplot2_r, gist_earth, YlGnBu_r, copper, gist_earth_r, Set3_r, OrRd, gnuplot_r, ocean_r, brg, gnuplot2, PuRd_r, bone_r, BuPu, Oranges, RdYlGn_r, PiYG, CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, gist_gray, flag, bwr_r, RdBu_r, BrBG, Reds, Set1_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy, PuRd, Accent_r, Blues, autumn_r, autumn, cubehelix_r, nipy_spectral_r, ocean, PRGn_r, Greys_r, pink, binary, winter, gnuplot, RdYlBu_r, hot, YlOrBr, coolwarm_r, rainbow_r, Purples_r, PiYG_r, YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, seismic_r, RdBu, Greys, BuGn_r, YlOrRd, PuOr, PuBuGn, nipy_spectral, afmhot
"""
