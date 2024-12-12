import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils
from roman_imsim import *
import astropy.table as tb
import warnings 
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)
import scipy.sparse as sp 
from scipy.linalg import block_diag, lstsq
from numpy.linalg import LinAlgError
from astropy.nddata import Cutout2D
from coord import *
import requests
from astropy.table import Table
import os
import scipy
import time
import galsim

import sklearn
from sklearn import linear_model
from scipy.interpolate import RectBivariateSpline

roman_path = '/hpc/group/cosmology/OpenUniverse2024'
sn_path = '/hpc/group/cosmology/OpenUniverse2024/roman_rubin_cats_v1.1.2_faint/'

def test_find_parq():
    parq_file_ID = find_parq(50134575, sn_path)
    assert parq_file_ID == 10430

def test_radec2point():
    p, s = radec2point(7.731890048839705, -44.4589649005717, 'Y106', path = roman_path)
    assert p == 10535
    assert s == 14

def test_SNID_to_loc():
    RA, DEC, p, s, start, end, peak, host_ra, host_dec = SNID_to_loc(50134575, 10430, 'Y106', date = True,\
     snpath = sn_path, roman_path = roman_path, host = True)
    assert RA == 7.731890048839705
    assert DEC ==  -44.4589649005717
    assert p == 10535
    assert s == 14
    assert start[0] == 62654.
    assert end[0] == 62958.
    assert peak[0] == np.float32(62683.98)
    assert host_ra == 7.731832
    assert host_dec == -44.459011


    
