import os
import pathlib
import sys
import tempfile
import warnings

import astropy.units as u
import galsim
import numpy as np
import pandas as pd
import pytest
from astropy.io import ascii
from astropy.table import QTable
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import matplotlib
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils

import snappl
from snappl.image import OpenUniverse2024FITSImage
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger as Lager

from campari import RomanASP
from campari.AllASPFuncs import (
    calc_mag_and_err,
    calculate_background_level,
    construct_psf_background,
    construct_psf_source,
    extract_sn_from_parquet_file_and_write_to_csv,
    extract_star_from_parquet_file_and_write_to_csv,
    find_parquet,
    findAllExposures,
    get_galsim_SED,
    get_galsim_SED_list,
    get_object_info,
    get_weights,
    load_SEDs_from_directory,
    make_adaptive_grid,
    make_contour_grid,
    make_regular_grid,
    open_parquet,
    radec2point,
    save_lightcurve,
)
from campari.simulation import simulate_galaxy, simulate_images, simulate_supernova, simulate_wcs

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


@pytest.fixture(scope="module")
def roman_path(cfg):
    return cfg.value("photometry.campari.paths.roman_path")


@pytest.fixture(scope="module")
def sn_path(cfg):
    return cfg.value("photometry.campari.paths.sn_path")


def test_find_parquet(sn_path):
    parq_file_ID = find_parquet(50134575, sn_path)
    assert parq_file_ID == 10430


def test_radec2point(roman_path):
    p, s = radec2point(7.731890048839705, -44.4589649005717, "Y106",
                       path=roman_path)
    assert p == 10535
    assert s == 14


def test_get_object_info(roman_path, sn_path):
    ra, dec, p, s, start, end, peak = get_object_info(50134575, 10430, "Y106",
                                                      snpath=sn_path,
                                                      roman_path=roman_path,
                                                      obj_type="SN")
    assert ra == 7.731890048839705
    assert dec == -44.4589649005717
    assert p == 10535
    assert s == 14
    assert start[0] == 62654.
    assert end[0] == 62958.
    assert peak[0] == np.float32(62683.98)


def test_findAllExposures(roman_path):
    explist = findAllExposures(50134575, 7.731890048839705, -44.4589649005717,
                               62654., 62958., "Y106", maxbg=24,
                               maxdet=24, return_list=True,
                               roman_path=roman_path,
                               pointing_list=None, SCA_list=None,
                               truth="simple_model")
    compare_table = ascii.read(pathlib.Path(__file__).parent
                               / "testdata/findallexposurestest.dat")
    assert explist["Pointing"].all() == compare_table["Pointing"].all()
    assert explist["SCA"].all() == compare_table["SCA"].all()
    assert explist["date"].all() == compare_table["date"].all()


def test_simulate_images(roman_path):
    lam = 1293  # nm
    band = "F184"
    airy = \
        galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=galsim.roman.
                                   getPSF(1, band, pupil_bin=1).aberrations)
    # Fluxes for the simulated supernova, days arbitrary.
    test_lightcurve = [10, 100, 1000, 10**4, 10**5]
    images, im_wcs_list, cutout_wcs_list, sim_lc, util_ref = \
        simulate_images(num_total_images=10, num_detect_images=5,
                        ra=7.541534306163982,
                        dec=-44.219205940734625,
                        sim_gal_ra_offset=1e-5,
                        sim_gal_dec_offset=1e-5, do_xshift=True,
                        do_rotation=True, sim_lc=test_lightcurve,
                        noise=0, use_roman=False, band=band,
                        deltafcn_profile=False, roman_path=roman_path, size=11,
                        input_psf=airy, bg_gal_flux=9e5)

    compare_images = np.load(pathlib.Path(__file__).parent
                             / "testdata/images.npy")
    assert compare_images.all() == np.asarray(images).all()


def test_simulate_wcs(roman_path):
    wcs_dict = simulate_wcs(angle=np.pi/4, x_shift=0.1, y_shift=0,
                            roman_path=roman_path, base_sca=11,
                            base_pointing=662, band="F184")
    b = np.load(pathlib.Path(__file__).parent / "testdata/wcs_dict.npz",
                allow_pickle=True)
    assert wcs_dict == b, "WCS simulation does not match test example"


def test_simulate_galaxy():
    band = "F184"
    roman_bandpasses = galsim.roman.getBandpasses()
    lam = 1293  # nm
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"), wave_type="nm",
                     flux_type="fphotons")
    sim_psf = \
        galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=galsim.roman.
                                   getPSF(1, band, pupil_bin=1).aberrations)
    convolved = simulate_galaxy(bg_gal_flux=9e5, deltafcn_profile=False,
                                band=band, sim_psf=sim_psf, sed=sed)

    a = convolved.drawImage(roman_bandpasses[band], method="no_pixel",
                            use_true_center=True)
    b = np.load(pathlib.Path(__file__).parent
                / "testdata/test_galaxy.npy")
    assert (a.array - b).all() == 0, "The two galaxy images are not the same!"


def test_simulate_supernova():
    wcs_data = np.load(pathlib.Path(__file__).parent
                       / "testdata/wcs_dict.npz", allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}

    wcs, origin = galsim.wcs.readFromFitsHeader(wcs_dict)

    stamp = galsim.Image(11, 11, wcs=wcs)
    band = "F184"
    lam = 1293  # nm
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"), wave_type="nm",
                     flux_type="fphotons")
    sim_psf = \
        galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=galsim.roman.
                                   getPSF(1, band, pupil_bin=1).aberrations)
    supernova_image = simulate_supernova(snx=6, sny=6, stamp=stamp,
                                         flux=1000, sed=sed, band=band,
                                         sim_psf=sim_psf, source_phot_ops=True,
                                         base_pointing=662, base_sca=11,
                                         random_seed=12345)
    test_sn = np.load(pathlib.Path(__file__).parent
                      / "testdata/supernova_image.npy")
    np.testing.assert_allclose(supernova_image, test_sn, rtol=1e-7)


def test_savelightcurve():
    output_dir = pathlib.Path(Config.get().value("photometry.campari.paths.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    lc_file = output_dir / "test_test_test_lc.ecsv"
    assert not lc_file.exists(), f"File {lc_file} eixsts; delete it before running tests"

    try:
        data_dict = {"MJD": [1, 2, 3, 4, 5], "true_flux": [1, 2, 3, 4, 5],
                     "measured_flux": [1, 2, 3, 4, 5]}
        units = {"MJD": u.d, "true_flux": "",  "measured_flux": ""}
        meta_dict = {}
        lc = QTable(data=data_dict, meta=meta_dict, units=units)
        # save_lightcurve defaults to saving to photometry.campari.paths.output_dir
        save_lightcurve(lc, "test", "test", "test")
        assert lc_file.is_file()
        # TODO: look at contents?
    finally:
        # Make sure to clean up after ourselves
        lc_file.unlink(missing_ok=True)


def test_run_on_star():
    # Call it as a function first so we can pdb and such
    args = ["_", "-s", "40973149150", "-f", "Y106", "-t", "1", "-d", "1",
             "--object_type", "star", "--photometry-campari-grid_options-type", "none"]
    orig_argv = sys.argv
    try:
        sys.argv = args
        RomanASP.main()
    except Exception as ex:
        assert False, str(ex)
    finally:
        sys.argv = orig_argv

    # Make sure it runs from the command line
    err_code = os.system("python ../RomanASP.py -s 40973149150 -f Y106 -t 1 -d 1 "
                         "--object_type star --photometry-campari-grid_options-type none")
    assert err_code == 0, "The test run on a star failed. Check the logs"


def test_regression_function():
    # This runs the same test as test_regression, with a different
    # interface.  This one calls the main() function (so is useful if
    # you want to, e.g., do things with pdb).  test_regression runs it
    # from the command line.  (And we do want to make sure that works!)

    cfg = Config.get()
    curfile = pathlib.Path( cfg.value("photometry.campari.paths.output_dir") ) / "40120913_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    a = ["_", "-s", "40120913", "-f", "Y106", "-t", "2", "-d", "1",
         "--photometry-campari-use_roman",
         "--photometry-campari-use_real_images",
         "--no-photometry-campari-fetch_SED",
         "--photometry-campari-grid_options-type", "contour",
         "--photometry-campari-cutout_size", "19",
         "--photometry-campari-weighting",
         "--photometry-campari-subtract_background",
         "--no-photometry-campari-source_phot_ops"]
    orig_argv = sys.argv
    try:
        sys.argv = a
        RomanASP.main()
        cfg = Config.get()
        current = pd.read_csv(curfile, comment="#", delimiter=" ")
        comparison = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_lc.ecsv",
                                 comment="#", delimiter=" ")

        for col in current.columns:
            Lager.debug(f"Checking col {col}")
            # According to Michael and Rob, this is roughly what can be expected
            # due to floating point precision.
            #
            # (Rob here: 32-bit IEEE-754 floats have a 24-bit mantissa
            # (cf: https://en.wikipedia.org/wiki/IEEE_754), which means
            # roughly log10(2^24)=7 significant figures.  As such,
            # errors of 1e-7 can very easily come from things like order
            # of operations (even in system libraries).  64-bit floats
            # (i.e. doubles) have a 53-bit mantissa, and log10(2^53)=16,
            # so you have 15 or 16 sig figs which "ought to be enough
            # for anybody".  HOWEVER, you *can* get errors much larger
            # than this, depending on your order of operations.  For
            # example, try the following code:
            #
            #   import numpy
            #   a = numpy.float32( 1e8 )
            #   print( f"a={a}" )
            #   b = numpy.float32( 1 )
            #   print( f"b={b}" )
            #   print( a - ( a - b ) )
            #   print( a - a + b )
            #
            # If you know algebra, you know that the last two numbers
            # printed out should be exactly the same.  However, you get
            # either a 100% differerence, or an *infinite* difference,
            # depending on how you define relative difference in this
            # case.
            #
            # The numpy libraries try to be a bit clever when doing
            # things like .sum() to avoid the worst of floating-point
            # underflow, but it's a thing worth being aware of.
            # Relative errors of 1e-7 (for floats) or 1e-16 (for
            # doubles) can easily arise from floating-point underflow;
            # whether or not you're worried about those errors depends
            # on how confident you are that the order of operations is
            # identical in two different test cases.  Bigger errors
            # *can* arise from floating point underflow, but never just
            # wave your hands and say, "eh, the tests are passing, it's
            # just underflow!"  Understand how underflow did it.  If it
            # did, and you're not worried, document that.  But,
            # probably, you should be worried, and you should
            # restructure the order of operations in your code to avoid
            # underflow errors bigger than the number of sig figs in a
            # floating point number.)

            msg = f"The lightcurves do not match for column {col}"
            if col == "band":
                # band is the only string column, so we check it with array_equal
                np.testing.assert_array_equal(current[col], comparison[col]), msg
            else:
                percent = 100 * np.max((current[col] - comparison[col])
                                    / comparison[col])
                msg2 = f"difference is {percent} %"
                msg = msg+msg2
                # Switching from one type of WCS to another gave rise in a
                # difference of about 1e-9 pixels for the grid, which led to a
                # change in flux of 2e-7. I don't want switching WCS types to make
                # this fail, so I put the rtol at just above that level.
                np.testing.assert_allclose(current[col], comparison[col], rtol=3e-7), msg

            # check output
    finally:
        sys.argv = orig_argv


def test_regression():
    # Regression lightcurve was changed on June 6th 2025 because we were on an
    # outdated version of snappl.
    # Weighting is a Gaussian width 1000 when this was made
    # In the future, this should be True, but random seeds not working rn.

    cfg = Config.get()

    curfile = pathlib.Path( cfg.value("photometry.campari.paths.output_dir") ) / "40120913_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    output = os.system("python ../RomanASP.py -s 40120913 -f Y106 -t 2 -d 1 "
                        "--photometry-campari-use_roman "
                        "--photometry-campari-use_real_images "
                        "--no-photometry-campari-fetch_SED "
                        "--photometry-campari-grid_options-type contour "
                        "--photometry-campari-cutout_size 19 "
                        "--photometry-campari-weighting "
                        "--photometry-campari-subtract_background "
                        "--no-photometry-campari-source_phot_ops ")
    assert output == 0, "The test run on a SN failed. Check the logs"

    current = pd.read_csv(curfile, comment="#", delimiter=" ")
    comparison = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_lc.ecsv",
                             comment="#", delimiter=" ")

    for col in current.columns:
        Lager.debug(f"Checking col {col}")
        # According to Michael and Rob, this is roughly what can be expected
        # due to floating point precision.
        msg = f"The lightcurves do not match for column {col}"
        if col == "band":
            # band is the only string column, so we check it with array_equal
            np.testing.assert_array_equal(current[col], comparison[col]), msg
        else:
            percent = 100 * np.max((current[col] - comparison[col])
                                   / comparison[col])
            msg2 = f"difference is {percent} %"
            msg = msg+msg2
            # Switching from one type of WCS to another gave rise in a
            # difference of about 1e-9 pixels for the grid, which led to a
            # change in flux of 2e-7. I don't want switching WCS types to make
            # this fail, so I put the rtol at just above that level.
            np.testing.assert_allclose(current[col], comparison[col], rtol=3e-7), msg


def test_get_galsim_SED(sn_path):
    sed = get_galsim_SED(40973149150, 000, sn_path, obj_type="star",
                         fetch_SED=True)
    lam = sed._spec.x
    flambda = sed._spec.f

    star_lam_test = np.load(pathlib.Path(__file__).parent
                            / "testdata/star_lam_test.npy")
    np.testing.assert_array_equal(lam, star_lam_test)
    star_flambda_test = np.load(pathlib.Path(__file__).parent
                                / "testdata/star_flambda_test.npy")

    np.testing.assert_array_equal(flambda, star_flambda_test)

    sed = get_galsim_SED(40120913, 62535.424, sn_path, obj_type="SN",
                         fetch_SED=True)
    lam = sed._spec.x
    flambda = sed._spec.f

    sn_lam_test = np.load(pathlib.Path(__file__).parent
                          / "testdata/sn_lam_test.npy")
    sn_flambda_test = np.load(pathlib.Path(__file__).parent
                              / "testdata/sn_flambda_test.npy")

    np.testing.assert_array_equal(lam, sn_lam_test)
    np.testing.assert_array_equal(flambda, sn_flambda_test)


def test_get_galsim_SED_list(sn_path):
    dates = 62535.424
    fetch_SED = True
    object_type = "SN"
    ID = 40120913
    with tempfile.TemporaryDirectory() as sed_path:
        get_galsim_SED_list(ID, dates, fetch_SED, object_type, sn_path,
                            sed_out_dir=sed_path)
        sedlist = load_SEDs_from_directory(sed_path)
        assert len(sedlist) == 1, "The length of the SED list is not 1"
        sn_lam_test = np.load(pathlib.Path(__file__).parent
                              / "testdata/sn_lam_test.npy")
        np.testing.assert_allclose(sedlist[0]._spec.x, sn_lam_test, atol=1e-7)
        sn_flambda_test = np.load(pathlib.Path(__file__).parent
                                  / "testdata/sn_flambda_test.npy")
        np.testing.assert_allclose(sedlist[0]._spec.f, sn_flambda_test,
                                   atol=1e-7)


def test_plot_lc():
    from campari.AllASPFuncs import plot_lc
    output = plot_lc(pathlib.Path(__file__).parent
                     / "testdata/test_lc_plot.ecsv",
                     return_data=True)
    assert output[0][0] == 23.34624211038908
    assert output[1][0] == 62535.424
    assert output[2][0] == 0.3464661982648008
    assert output[3][0] == 23.164154309471726
    assert output[4] == 182.088
    assert output[5] == 0.0


def test_extract_sn_from_parquet_file_and_write_to_csv(sn_path):
    cfg = Config.get()
    new_snid_file = ( pathlib.Path( cfg.value("photometry.campari.paths.debug_dir") ) /
                      "test_extract_sn_from_parquet_file_and_write_to_csv_snids.csv" )
    new_snid_file.unlink( missing_ok=True )
    # Make sure we're really writing a new file so that this
    #   test is really meaningful
    assert not new_snid_file.exists()

    # TODO don't write to testdata
    extract_sn_from_parquet_file_and_write_to_csv(10430, sn_path, new_snid_file, mag_limits=[20, 21])
    sn_ids = pd.read_csv(new_snid_file, header=None).values.flatten()
    test_sn_ids = pd.read_csv(pathlib.Path(__file__).parent
                              / "testdata/test_snids.csv",
                              header=None).values.flatten()
    np.testing.assert_array_equal(sn_ids, test_sn_ids), "The SNIDs do not match the test example"


def test_extract_star_from_parquet_file_and_write_to_csv(sn_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)\
            as temp_file:
        output_path = temp_file.name
        extract_star_from_parquet_file_and_write_to_csv(10430, sn_path,
                                                        output_path,
                                                        ra=7.1,
                                                        dec=-44.1,
                                                        radius=0.25)
        star_ids = pd.read_csv(output_path, header=None).values.flatten()
        test_star_ids = pd.read_csv(pathlib.Path(__file__).parent
                                    / "testdata/test_star_ids.csv",
                                    header=None).values.flatten()
    np.testing.assert_array_equal(star_ids, test_star_ids), \
        "The star IDs do not match the test example"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)\
            as temp_file:
        output_path = temp_file.name
        extract_star_from_parquet_file_and_write_to_csv(10430, sn_path,
                                                        output_path)
        star_ids = pd.read_csv(output_path, header=None).values.flatten()
        parq = open_parquet(10430, sn_path, obj_type="star")
        assert len(star_ids) == parq["id"].size, \
            "extract_star_from_parquet_file_and_write_to_csv did not return" +\
            "all stars when no radius was passed"


def test_make_regular_grid():
    wcs_data = np.load(pathlib.Path(__file__).parent
                       / "testdata/wcs_dict.npz",
                       allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}
    ra_center = wcs_dict["CRVAL1"]
    dec_center = wcs_dict["CRVAL2"]

    test_ra = np.array([7.67363133, 7.67373506, 7.67383878, 7.67355803,
                        7.67366176, 7.67376548, 7.67348473, 7.67358845,
                        7.67369218])
    test_dec = np.array([-44.26396874, -44.26391831, -44.26386787,
                        -44.26389673, -44.26384629, -44.26379586,
                        -44.26382471, -44.26377428, -44.26372384])
    for wcs in [snappl.wcs.GalsimWCS.from_header(wcs_dict),
                snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        ra_grid, dec_grid = make_regular_grid(ra_center, dec_center, wcs,
                                              size=25, spacing=3.0)
        np.testing.assert_allclose(ra_grid, test_ra, atol=1e-9), \
            "RA vals do not match"
        np.testing.assert_allclose(dec_grid, test_dec, atol=1e-9), \
            "Dec vals do not match"


def test_make_adaptive_grid():
    wcs_data = np.load("./testdata/wcs_dict.npz", allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}
    ra_center = wcs_dict["CRVAL1"]
    dec_center = wcs_dict["CRVAL2"]
    for wcs in [snappl.wcs.GalsimWCS.from_header(wcs_dict),
                snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        image = compare_images[:11**2].reshape(11, 11)
        ra_grid, dec_grid = make_adaptive_grid(ra_center, dec_center, wcs,
                                               image=image, percentiles=[99])
        test_ra = [7.67356034, 7.67359491, 7.67362949, 7.67366407, 7.67369864,]
        test_dec = [-44.26425446, -44.26423765, -44.26422084, -44.26420403,
                    -44.26418721]
        # Only testing the first 5 to save memory.
        np.testing.assert_allclose(ra_grid[:5], test_ra, atol=1e-9), \
            "RA vals do not match"
        np.testing.assert_allclose(dec_grid[:5], test_dec, atol=1e-9), \
            "Dec vals do not match"


def test_make_contour_grid():
    wcs_data = np.load(pathlib.Path(__file__).parent
                       / "testdata/wcs_dict.npz",
                       allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}
    test_ra = [7.67357048, 7.67360506, 7.67363963, 7.67367421]
    test_dec = [-44.26421364, -44.26419683, -44.26418002, -44.26416321]
    atol = 1e-9
    for wcs in [snappl.wcs.GalsimWCS.from_header(wcs_dict),
                snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        image = compare_images[:11**2].reshape(11, 11)
        ra_grid, dec_grid = make_contour_grid(image, wcs)
        msg = f"RA vals do not match to {atol:.1e} using galsim wcs."
        np.testing.assert_allclose(ra_grid[:4], test_ra,
                                   atol=atol, rtol=1e-9), msg
        msg = f"Dec vals do not match to {atol:.1e} using galsim wcs."
        np.testing.assert_allclose(dec_grid[:4], test_dec,
                                   atol=atol, rtol=1e-9), msg


def test_calculate_background_level():
    test_data = np.ones((12, 12))
    test_data[5:7, 5:7] = 1000

    # Add some outliers to prevent all of
    # the data from being sigma clipped.
    test_data[0:2, 0:12:2] = 123
    test_data[-3:-1, 0:12:2] = 123
    test_data[0:12:2, 0:2] = 123
    test_data[0:12:2, -1:-3] = 123

    expected_output = 1
    output = calculate_background_level(test_data)
    msg = f"Expected {expected_output}, but got {output}"
    assert np.isclose(output, expected_output, rtol=1e-7), msg


def test_calc_mag_and_err():
    flux = np.array([-1e2, 1e2, 1e3, 1e4])
    sigma_flux = np.array([10, 10, 10, 10])
    band = "Y106"
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)

    test_mag = np.array([np.nan, 27.66165575,  25.16165575,  22.66165575])
    test_magerr = np.array([np.nan, 1.0857362e-01,
                            1.0857362e-02, 1.0857362e-03])
    test_zp = 15.023547191066587

    np.testing.assert_allclose(mag, test_mag, atol=1e-7, equal_nan=True), \
        f"The magnitudes do not match {mag} vs. {test_mag}"
    np.testing.assert_allclose(magerr, test_magerr,
                               atol=1e-7, equal_nan=True), \
        "The magnitude errors do not match"
    np.testing.assert_allclose(zp, test_zp, atol=1e-7), \
        "The zeropoint does not match"


def test_construct_psf_background( cfg ):
    wcs_data = np.load("./testdata/wcs_dict.npz", allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}

    ra_grid = np.array([7.67357048, 7.67360506, 7.67363963, 7.67367421])
    dec_grid = np.array([-44.26421364, -44.26419683, -44.26418002,
                         -44.26416321])

    config_file = pathlib.Path(cfg.value("photometry.campari.galsim.tds_file"))
    pointing = 43623  # These numbers are arbitrary for this test.
    SCA = 7

    size = 9
    util_ref = roman_utils(config_file=config_file, visit=pointing, sca=SCA)

    for wcs in [snappl.wcs.GalsimWCS.from_header(wcs_dict),
                snappl.wcs.AstropyWCS.from_header(wcs_dict)]:

        psf_background = construct_psf_background(ra_grid, dec_grid, wcs,
                                                  x_loc=2044, y_loc=2044,
                                                  stampsize=size, band="Y106",
                                                  util_ref=util_ref)
        test_psf_background = np.load(pathlib.Path(__file__).parent
                                      / "testdata/test_psf_bg.npy")
        np.testing.assert_allclose(psf_background, test_psf_background,
                                   atol=1e-7)


def test_get_weights(roman_path):
    test_snra = np.array([7.34465537])
    test_sndec = np.array([-44.91932581])
    size = 7
    pointing = 111
    SCA = 13
    truth = "simple_model"
    band = "Y106"
    imagepath = roman_path + (f"/RomanTDS/images/{truth}/{band}/{pointing}"
                              f"/Roman_TDS_{truth}_{band}_{pointing}_"
                              f"{SCA}.fits.gz")
    snappl_image = OpenUniverse2024FITSImage(imagepath, None, SCA)
    snappl_cutout = snappl_image.get_ra_dec_cutout(test_snra, test_sndec, size)
    wgt_matrix = get_weights([snappl_cutout], test_snra, test_sndec,
                             gaussian_var=1000, cutoff=4)

    test_wgt_matrix = np.load(pathlib.Path(__file__).parent
                              / "testdata/test_wgt_matrix.npy")
    np.testing.assert_allclose(wgt_matrix, test_wgt_matrix, atol=1e-7)


def test_construct_psf_source():
    lam, flambda = [1000, 26000], [1, 1]
    sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                     wave_type="Angstrom",
                     flux_type="fphotons")

    comparison_image = np.load(pathlib.Path(__file__).parent
             / "testdata/test_psf_source.npy")

    psf_image = construct_psf_source(x=2044, y=2044, pointing=43623, SCA=7,
                                     stampsize=25, x_center=2044,
                                     y_center=2044, sed=sed,
                                     flux=1, photOps=False)

    np.testing.assert_allclose(np.sum(psf_image), np.sum(comparison_image),
                               atol=1e-6, verbose=True)

    try:
        np.testing.assert_allclose(psf_image, comparison_image, atol=1e-7,
                                   verbose=True)

    except AssertionError as e:
        matplotlib.use("pdf")
        plt.subplot(1, 3, 1)
        plt.title("Constructed PSF Source")
        plt.grid(True)
        plt.imshow(psf_image.reshape(25, 25), origin="lower")

        plt.subplot(1, 3, 2)
        plt.title("Comparison PSF Source")
        plt.grid(True)
        plt.imshow(comparison_image.reshape(25, 25), origin="lower")

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.grid(True)
        plt.imshow(np.log10(np.abs(psf_image.reshape(25, 25) -
                                   comparison_image.reshape(25, 25))),
                   origin="lower")
        plt.colorbar(label="log10( |constructed - comparison| )")

        im_path = pathlib.Path(__file__).parent / "test_psf_source_comparison.png"
        Lager.debug(f"Saving diagnostic image to {im_path}")
        plt.savefig(im_path)
        plt.close()

        assert False, f"PSF source images do not match, a diagnostic " \
                      f"image has been saved to {im_path}. Error: {e}"
