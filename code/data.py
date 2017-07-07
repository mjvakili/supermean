'''
Module for handling the data and mask files
'''

import pyfits as pf
import numpy as np
import util


def load_data():
    '''load data
    '''
    data_file = ''.join([util.dat_dir(), 'f160w_25_457_557_457_557_pixels.fits'])
    data = pf.open(data_file)[0].data

    return data


def load_mask():
    '''load maks
    '''
    mask_file = ''.join([util.dat_dir(), 'f160w_25_457_557_457_557_dq.fits'])
    mask = pf.open(mask_file)[0].data
    return mask


