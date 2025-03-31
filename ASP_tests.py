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

def test_findAllExposures():
    explist = findAllExposures(50134575, 7.731890048839705, -44.4589649005717,62654.,62958.,62683.98, 'Y106', maxbg = 24, maxdet = 24, \
                        return_list = True, stampsize = 25, roman_path = roman_path,\
                    pointing_list = None, SCA_list = None, truth = 'simple_model')
    compare_table = ascii.read('tests/testdata/findallexposurestest.dat') 
    assert explist['Pointing'].all() == compare_table['Pointing'].all()
    assert explist['SCA'].all() == compare_table['SCA'].all()
    assert explist['date'].all() == compare_table['date'].all()

def test_simulateImages():
    images, im_wcs_list, cutout_wcs_list = simulateImages(10,5,7.541534306163982, -44.219205940734625, True, True, \
        [10, 100, 1000, 10**4, 10**5], 0, False, 'F184', size=11)
    compare_images = np.load('tests/testdata/images.npy')
    assert compare_images.all() == images.all()


def test_savelightcurve():
    lcdict = {'MJD': [1,2,3,4,5], 'true_flux': [1,2,3,4,5], 'measured_flux': [1,2,3,4,5]}
    lc = pd.DataFrame(lcdict)
    save_lightcurve(lc, 'test', 'test', 'test')
    output_path = os.path.join(os.getcwd(), 'results/lightcurves/')
    lc_file = os.path.join(output_path, 'test_test_test_lc.csv')
    assert os.path.exists(lc_file) == True

