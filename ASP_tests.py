import numpy as np
#from AllASPFuncs import find_parq, radec2point, SNID_to_loc

from AllASPFuncs import *
from astropy.io import ascii
roman_path = '/hpc/group/cosmology/OpenUniverse2024'
sn_path = '/hpc/group/cosmology/OpenUniverse2024/roman_rubin_cats_v1.1.2_faint/'

#Add more tests
#move to a python executable file

def test_find_parq():
    parq_file_ID = find_parq(50134575, sn_path)
    assert parq_file_ID == 10430

def test_radec2point():
    p, s = radec2point(7.731890048839705, -44.4589649005717, 'Y106', path = roman_path)
    assert p == 10535
    assert s == 14

def test_get_object_info():
    ra, dec, p, s, start, end, peak  = get_object_info(50134575, 10430, 'Y106', \
     snpath = sn_path, roman_path = roman_path, obj_type = 'SN')
    assert ra == 7.731890048839705
    assert dec ==  -44.4589649005717
    assert p == 10535
    assert s == 14
    assert start[0] == 62654.
    assert end[0] == 62958.
    assert peak[0] == np.float32(62683.98)


def test_findAllExposures():
    explist = findAllExposures(50134575, 7.731890048839705, -44.4589649005717,62654.,62958.,62683.98, 'Y106', maxbg = 24, maxdet = 24, \
                        return_list = True, stampsize = 25, roman_path = roman_path,\
                    pointing_list = None, SCA_list = None, truth = 'simple_model')
    compare_table = ascii.read('tests/testdata/findallexposurestest.dat') 
    assert explist['Pointing'].all() == compare_table['Pointing'].all()
    assert explist['SCA'].all() == compare_table['SCA'].all()
    assert explist['date'].all() == compare_table['date'].all()

def test_simulateImages():
    lam = 1293  # nm
    lam_over_diam = 0.11300864172775239   #This is the roman value
    band=  'F184'
    airy = galsim.ChromaticOpticalPSF(lam, diam = 2.36, aberrations=galsim.roman.getPSF(1,band, pupil_bin = 1).aberrations)
    images, im_wcs_list, cutout_wcs_list, psf_storage, sn_storage = simulateImages(testnum = 10, detim = 5, ra = 7.541534306163982,\
         dec = -44.219205940734625, do_xshift = True, do_rotation = True, \
        supernova = [10, 100, 1000, 10**4, 10**5], noise = 0, use_roman = False, band=  'F184', size=11, \
            deltafcn_profile = False, input_psf = airy,  bg_gal_flux = 9e5)
    compare_images = np.load('tests/testdata/images.npy')
    assert compare_images.all() == images.all()


def test_savelightcurve():
    data_dict = {'MJD': [1,2,3,4,5], 'true_flux': [1,2,3,4,5], 'measured_flux': [1,2,3,4,5]}
    units = {'MJD':u.d, 'true_flux': '',  'measured_flux': ''}
    meta_dict = {}
    lc = QTable(data = data_dict, meta = meta_dict, units = units)
    save_lightcurve(lc, 'test', 'test', 'test')

    output_path = os.path.join(os.getcwd(), 'results/lightcurves/')
    lc_file = os.path.join(output_path, 'test_test_test_lc.ecsv')
    assert os.path.exists(lc_file) == True

def test_run_on_star():
    os.system('python RomanASP.py -s 40973149150 -b Y106 -t 1 -d 1')

