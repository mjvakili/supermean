import os
from nll_grad_fb import v3_fit_single_patch
import sys
import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
pl.switch_backend('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
rc('text', usetex=True)
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import george
from george.kernels import ExpSquaredKernel , ExpKernel , Matern32Kernel , WhiteKernel
import scipy.optimize as op
import h5py

sp = pf.open("anderson.fits")[0].data.T
stars = pf.open("wfc3_f160w_clean_central100.fits")[0].data
masks = pf.open("wfc3_f160w_clean_central100.fits")[1].data

f = .05
g = .01
fl = 1.e-5

masks[masks == 0] = -1
masks[masks  > 0] = False
masks[masks  < 0] = True
masks = masks == True



hh = 1./101
x, y = np.arange(0, 101)*hh + .5*hh , np.arange(0, 101)*hh + .5*hh
x, y = np.meshgrid(x, y, indexing="ij")
shape = x.shape
samples = np.vstack((x.flatten(), y.flatten())).T
 
np.random.seed(12345)
kernel = ExpSquaredKernel(.001, ndim=2) + WhiteKernel(0.01 , ndim = 2)


#Kxx = kernel.value(samples , samples)
mean = np.mean(sp)
spmm = sp.flatten() - mean
#alpha2 = np.linalg.solve(Kxx , spmm)

#np.savetxt("alpha2.dat",alpha2)
alpha = np.loadtxt("alpha.dat")

bkg  = np.loadtxt("superb_wfc_bkg_iter_5_nfl.txt") 				#background	
flux = np.loadtxt("superb_wfc_flux_iter_5_nfl.txt")				#flux
dx = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cx.txt")		#delta x
dy = np.loadtxt("wfc3_f160w_clean_central100_matchedfilterpoly_cy.txt")		#delta y

"""initializing stuff"""

f_pars = h5py.File("anderson_pars.hdf5", 'w')
#f_bkg = h5py.File("anderson_bkg.hdf5", 'w')
#f_cx  = h5py.File("anderson_cx.hdf5", 'w')
#f_cy  = h5py.File("anderson_cy.hdf5", 'w')

#amp_set = f_amp.create_dataset("amp" ,'f8') 
#bkg_set = f_amp.create_dataset("bkg" , 'f8') 
#cx_set = f_amp.create_dataset("cx" , 'f8') 
#cy_set = f_amp.create_dataset("cy" , 'f8') 
grp_amp = f_pars.create_group('amp')
grp_bkg = f_pars.create_group('bkg')
grp_cx = f_pars.create_group('cx')
grp_cy = f_pars.create_group('cy')

def bfgs_update_FB(p):
    t0 = time.time()
    datap = stars[p]
    maskp = masks[p]
    Y = datap[maskp!=0]  #masked data
    theta = bkg[p], flux[p]

    h = 1./25.
    Dx , Dy = dx[p] , dy[p]
    x2 = np.arange(0, 25)*h + 0.5*h - Dx*h
    y2 = np.arange(0, 25)*h + 0.5*h - Dy*h
    x2, y2 = np.meshgrid(x2, y2, indexing="ij")
    shape2 = x2.shape
    samples2 = np.vstack((x2.flatten(), y2.flatten())).T
    Kxsx = kernel.value(samples2 , samples)
    psf =  np.dot(Kxsx , alpha) + mean
    masked_psf = psf[maskp!=0]  #masked psf
        
    grad_func = v3_fit_single_patch
    x = op.fmin_l_bfgs_b(grad_func, x0=theta, fprime = None, \
                         args=(Y, masked_psf, f, g), approx_grad = False, \
                         bounds = [(1.e-10,100.), (1.e-2,10.**7.)], \
                         factr=10., pgtol=1e-16, epsilon=1e-16, maxfun=60)
    print time.time() - t0
    print "old background=", bkg[p]
    print "old amplitude=" , flux[p] 
    print "new_background=", x[0][0]
    print "new-amplitude=" , x[0][1] 
    return x[0] , psf, Y 

def nll_ctr(theta, data, mask, spmm, mean, flux, bkg, f, g):

    Dx , Dy = theta[0], theta[1]
    h = 1./25.
    
    x2 = np.arange(0, 25)*h + 0.5*h - Dx*h
    y2 = np.arange(0, 25)*h + 0.5*h - Dy*h
    x2, y2 = np.meshgrid(x2, y2, indexing="ij")
    shape2 = x2.shape
    samples2 = np.vstack((x2.flatten(), y2.flatten())).T
    Kxsx = kernel.value(samples2 , samples)
    model = flux*(np.dot(Kxsx, alpha)+mean) +bkg

    data = data.reshape(25,25)[10:-10,10:-10]
    model= model.reshape(25,25)[10:-10,10:-10]
    mask = mask.reshape(25,25)[10:-10,10:-10]
    data = data[mask!=0]
    model = model[mask!=0]
    res = data - model
    var = f + g*np.abs(model)
    
    func = np.sum(res**2. / var + np.log(var))
    return func
    
def fit_ctr(p, amp, bg):

    func = nll_ctr
    theta = [dx[p] , dy[p]]
    
    x = op.fmin_powell(func, [theta[0], theta[1]], \
                args = (stars[p], masks[p], spmm, mean, amp, bg, f, g), disp = False)
    print "old dx=", dx[p] 
    print "old dy=", dy[p] 
    print "new dx=", x[0] 
    print "new dy=", x[1]
 
    return x[0], x[1]   

for p in range(120):

  datap = stars[p]
  maskp = masks[p]
  print "p=", p
  
  
  bgamp, psf, Y = bfgs_update_FB(p)
  bkg[p], flux[p] = bgamp[0] , bgamp[1]
  dx[p], dy[p] = fit_ctr(p, flux[p], bkg[p])

  grp_bkg.create_dataset(str(p) , data = bkg[p])
  grp_amp.create_dataset(str(p) , data = flux[p])
  grp_cx.create_dataset(str(p) , data = dx[p])
  grp_cy.create_dataset(str(p) , data = dy[p])
  """
  h = 1./25.
  x2 = np.arange(0, 25)*h + 0.5*h - dx[p]*h
  y2 = np.arange(0, 25)*h + 0.5*h - dy[p]*h
  x2, y2 = np.meshgrid(x2, y2, indexing="ij")
  shape2 = x2.shape
  samples2 = np.vstack((x2.flatten(), y2.flatten())).T
  Kxsx = kernel.value(samples2 , samples)
  model = flux[p]*(np.dot(Kxsx, alpha)+mean) +bkg[p]
  
  chi = (datap - model)/(f + g * np.abs(model))**.5
  ma = max(np.max(datap[maskp]) , np.max(model[maskp]))

  datap[maskp==0] = np.nan
  model[maskp==0] = np.nan
  chi[maskp==0]   = np.nan
 
  datap = datap.reshape(25,25)
  model = model.reshape(25,25)
  chi = chi.reshape(25,25)

  pl.subplot(1,3,1)

  ax = pl.gca()
  im = ax.imshow(datap , interpolation="None" , origin="lower" , norm = LogNorm(), vmin = .1 , vmax = ma)
  pl.title("Data")
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.2)
  pl.colorbar(im , cax = cax)
  ax.set_xticks(())
  ax.set_yticks(())

  pl.subplot(1,3,2)
  ax = pl.gca()
  im = ax.imshow(model, interpolation = "None" , origin = "lower",norm=LogNorm(), vmin = .1 , vmax = ma)
  pl.title("Model")
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.2)
  pl.colorbar(im , cax = cax)
  ax.set_xticks(())                                             
  ax.set_yticks(()) 

  pl.subplot(1,3,3)
  ax = pl.gca()
  im = ax.imshow(chi , interpolation = "None" , origin = "lower")
  pl.title("Chi")
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.2)
  pl.colorbar(im , cax = cax)
  ax.set_xticks(())
  ax.set_yticks(())
  pl.show()

  pl.savefig("/home/mj/public_html/psf/ctr_powell_kernel__"+str(p)+".png")
  
  pl.close()
  """
f_pars.close()
