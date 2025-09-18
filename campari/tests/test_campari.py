# Standard Libary
import os
import pathlib
import sys
import tempfile
import warnings

# Common Library
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Astronomy Library
from astropy.table import QTable, Table
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import galsim
from roman_imsim.utils import roman_utils


# SNPIT
from campari import RomanASP
from campari.data_construction import find_all_exposures
from campari.io import (
    add_truth_to_lc,
    build_lightcurve,
    extract_id_using_ra_dec,
    extract_object_from_healpix,
    extract_sn_from_parquet_file_and_write_to_csv,
    extract_star_from_parquet_file_and_write_to_csv,
    find_parquet,
    open_parquet,
    read_healpix_file,
    save_lightcurve,
)
from campari.model_building import (
    construct_static_scene,
    construct_transient_scene,
    make_adaptive_grid,
    make_contour_grid,
    make_regular_grid,
)
from campari.plotting import plot_lc
from campari.utils import (calc_mag_and_err,
                           calculate_background_level,
                           calculate_local_surface_brightness,
                           get_weights,
                           make_sim_param_grid,
                           campari_lightcurve_model)
import snappl
from snappl.diaobject import DiaObject
from snappl.image import ManualFITSImage
from snappl.imagecollection import ImageCollection
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


@pytest.fixture(scope="module")
def campari_test_data(cfg):
    return cfg.value("photometry.campari.paths.campari_test_data")


@pytest.fixture(scope="module")
def roman_path(cfg):
    return cfg.value("photometry.campari.paths.roman_path")


@pytest.fixture(scope="module")
def sn_path(cfg):
    return cfg.value("photometry.campari.paths.sn_path")


def test_find_parquet(sn_path):
    parq_file_ID = find_parquet(50134575, sn_path)
    assert parq_file_ID == 10430


def test_find_all_exposures(roman_path):
    diaobj = DiaObject.find_objects(id=1, ra=7.731890048839705, dec=-44.4589649005717, collection="manual")[0]
    diaobj.mjd_start = 62654.0
    diaobj.mjd_end = 62958.0
    image_list = find_all_exposures(diaobj=diaobj, band="Y106", maxbg=24,
                                    maxdet=24,
                                    roman_path=roman_path,
                                    pointing_list=None, sca_list=None,
                                    truth="simple_model")

    compare_table = np.load(pathlib.Path(__file__).parent / "testdata/findallexposures.npy")
    argsort = np.argsort(compare_table["date"])
    compare_table = compare_table[argsort]

    np.testing.assert_array_equal(
        np.array([img.mjd for img in image_list]),
        compare_table["date"]
    )

    np.testing.assert_array_equal(
        np.array([img.sca for img in image_list]),
        compare_table["sca"]
    )

    np.testing.assert_array_equal(
        np.array([img.pointing for img in image_list]),
        compare_table["pointing"]
    )


def test_savelightcurve():
    with tempfile.TemporaryDirectory() as output_dir:
        lc_file = output_dir + "/" + "test_test_test_lc.ecsv"
        lc_file = pathlib.Path(lc_file)

        data_dict = {"MJD": [1, 2, 3, 4, 5], "true_flux": [1, 2, 3, 4, 5],
                     "measured_flux": [1, 2, 3, 4, 5]}
        units = {"MJD": u.d, "true_flux": "",  "measured_flux": ""}
        meta_dict = {}
        lc = QTable(data=data_dict, meta=meta_dict, units=units)
        lc["filter"] = "test"
        # save_lightcurve defaults to saving to photometry.campari.paths.output_dir
        save_lightcurve(lc=lc, identifier="test", psftype="test", output_path=output_dir)
        assert lc_file.is_file()
        # TODO: look at contents?


def test_run_on_star(roman_path, campari_test_data, cfg):
    # Call it as a function first so we can pdb and such

    curfile = pathlib.Path(cfg.value("photometry.campari.paths.output_dir")) / "40973166870_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    args = ["_", "-s", "40973166870", "-f", "Y106", "-i",
            f"{campari_test_data}/test_image_list_star.csv", "--object_collection", "manual",
            "--object_type", "star", "--photometry-campari-grid_options-type", "none",
            "--no-photometry-campari-source_phot_ops", "--ra", "7.5833264", "--dec", "-44.809659"]
    orig_argv = sys.argv

    try:
        sys.argv = args
        RomanASP.main()
    except Exception as ex:
        assert False, str(ex)
    finally:
        sys.argv = orig_argv

    current = pd.read_csv(curfile, comment="#", delimiter=" ")
    comparison = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_star_lc.ecsv", comment="#", delimiter=" ")

    for col in current.columns:
        SNLogger.debug(f"Checking col {col}")
        if col == "filter":
            # filter is the only string column, so we check it with array_equal
            np.testing.assert_array_equal(current[col], comparison[col])
        else:
            # We check agreement against a few times 32-bit ulp epsilon, rtol ~1e-7.
            np.testing.assert_allclose(current[col], comparison[col], rtol=3e-7)

    curfile = pathlib.Path(cfg.value("photometry.campari.paths.output_dir")) / "40973166870_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()
    # Make sure it runs from the command line
    err_code = os.system(
        "python ../RomanASP.py -s 40973166870 -f Y106 -i"
        f" {campari_test_data}/test_image_list_star.csv --object_collection manual "
        "--object_type star --photometry-campari-grid_options-type none "
        "--no-photometry-campari-source_phot_ops "
        "--ra 7.5833264 --dec -44.809659"
    )
    assert err_code == 0, "The test run on a star failed. Check the logs"

    current = pd.read_csv(curfile, comment="#", delimiter=" ")
    comparison = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_star_lc.ecsv",
                             comment="#", delimiter=" ")

    for col in current.columns:
        SNLogger.debug(f"Checking col {col}")
        if col == "filter":
            # filter is the only string column, so we check it with array_equal
            np.testing.assert_array_equal(current[col], comparison[col])
        else:
            # We check agreement against a few times 32-bit ulp epsilon, rtol ~1e-7.
            np.testing.assert_allclose(current[col], comparison[col], rtol=3e-7)


def test_regression_function(campari_test_data):
    # This runs the same test as test_regression, with a different
    # interface.  This one calls the main() function (so is useful if
    # you want to, e.g., do things with pdb).  test_regression runs it
    # from the command line.  (And we do want to make sure that works!)

    cfg = Config.get()
    curfile = pathlib.Path(cfg.value("photometry.campari.paths.output_dir")) / "20172782_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    a = ["_", "-s", "20172782", "-f", "Y106", "-i",
         f"{campari_test_data}/test_image_list.csv",
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
            SNLogger.debug(f"Checking col {col}")
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
            if col == "filter":
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


def test_regression(campari_test_data):
    # Regression lightcurve was changed on June 6th 2025 because we were on an
    # outdated version of snappl.
    # Weighting is a Gaussian width 1000 when this was made
    # In the future, this should be True, but random seeds not working rn.

    cfg = Config.get()

    curfile = pathlib.Path(cfg.value("photometry.campari.paths.output_dir")) / "20172782_Y106_romanpsf_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    output = os.system(
        f"python ../RomanASP.py -s 20172782 -f Y106 -i {campari_test_data}/test_image_list.csv "
        "--photometry-campari-use_roman "
        "--photometry-campari-use_real_images "
        "--no-photometry-campari-fetch_SED "
        "--photometry-campari-grid_options-type contour "
        "--photometry-campari-cutout_size 19 "
        "--photometry-campari-weighting "
        "--photometry-campari-subtract_background "
        "--no-photometry-campari-source_phot_ops "
    )
    assert output == 0, "The test run on a SN failed. Check the logs"

    current = pd.read_csv(curfile, comment="#", delimiter=" ")
    comparison = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_lc.ecsv",
                             comment="#", delimiter=" ")

    for col in current.columns:
        SNLogger.debug(f"Checking col {col}")
        msg = f"The lightcurves do not match for column {col}"
        if col == "filter":
            # band is the only string column, so we check it with array_equal
            np.testing.assert_array_equal(current[col], comparison[col]), msg
        else:
            percent = 100 * np.max((current[col] - comparison[col])
                                   / comparison[col])
            msg2 = f"difference is {percent} %"
            msg = msg+msg2
            # We check agreement against a few times 32-bit ulp epsilon, rtol ~1e-7.
            np.testing.assert_allclose(current[col], comparison[col], rtol=3e-7), msg


def test_plot_lc():
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
    new_snid_file = (pathlib.Path(cfg.value("photometry.campari.paths.debug_dir")) /
                     "test_extract_sn_from_parquet_file_and_write_to_csv_snids.csv")
    new_snid_file.unlink(missing_ok=True)
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
    image_size = 25
    wcs_dict["NAXIS1"] = image_size
    wcs_dict["NAXIS2"] = image_size

    test_ra = np.array([7.673631, 7.673558, 7.673485, 7.673735, 7.673662, 7.673588,
                        7.673839, 7.673765, 7.673692])
    test_dec = np.array([-44.263969, -44.263897, -44.263825, -44.263918, -44.263846,
                         -44.263774, -44.263868, -44.263796, -44.263724])
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        img = ManualFITSImage(header=wcs_dict, data=np.zeros((25, 25)))
        ra_grid, dec_grid = make_regular_grid(img,
                                              spacing=3.0)
        np.testing.assert_allclose(ra_grid, test_ra, atol=1e-9), \
            "RA vals do not match"
        np.testing.assert_allclose(dec_grid, test_dec, atol=1e-9), \
            "Dec vals do not match"


def test_make_adaptive_grid():
    wcs_data = np.load("./testdata/wcs_dict.npz", allow_pickle=True)
    # Loading the data in this way, the data is packaged in an array,
    # this extracts just the value so that we can build the WCS.
    wcs_dict = {key: wcs_data[key].item() for key in wcs_data.files}
    image_size = 11
    wcs_dict["NAXIS1"] = image_size
    wcs_dict["NAXIS2"] = image_size
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        SNLogger.debug(f"compare_images shape: {compare_images.shape}")
        image = compare_images[0].reshape(11, 11)
        img_obj = ManualFITSImage(header=wcs_dict, data=image)
        ra_grid, dec_grid = make_adaptive_grid(img_obj, percentiles=[99])
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
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        image = compare_images[0].reshape(11, 11)
        img_obj = ManualFITSImage(header=wcs_dict, data=image)
        ra_grid, dec_grid = make_contour_grid(img_obj)
        msg = f"RA vals do not match to {atol:.1e}."
        np.testing.assert_allclose(ra_grid[:4], test_ra, atol=atol, rtol=1e-9), msg
        msg = f"Dec vals do not match to {atol:.1e}."
        np.testing.assert_allclose(dec_grid[:4], test_dec, atol=atol, rtol=1e-9), msg


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


def test_construct_static_scene(cfg, roman_path):
    config_file = pathlib.Path(cfg.value("photometry.campari.galsim.tds_file"))
    pointing = 43623  # These numbers are arbitrary for this test.
    sca = 7

    pointing = 5934
    sca = 3
    size = 9
    band = "Y106"

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)

    util_ref = roman_utils(config_file=config_file, visit=pointing, sca=sca)

    wcs = snappl_image.get_wcs()

    ra_grid = np.array([7.47193824, 7.47204612, 7.472154, 7.4718731, 7.47198098])
    dec_grid = np.array([-44.8280889, -44.82804109, -44.82799327, -44.82801657, -44.82796875])

    psf_background = construct_static_scene(ra_grid, dec_grid, wcs, x_loc=2044, y_loc=2044,
                                            stampsize=size, band="Y106", util_ref=util_ref)

    test_psf_background = np.load(pathlib.Path(__file__).parent / "testdata/test_psf_bg.npy")

    np.testing.assert_allclose(psf_background, test_psf_background, atol=1e-7)


def test_get_weights(roman_path):
    size = 7
    test_snra = np.array([7.471881246770769])
    test_sndec = np.array([-44.82824910386988])
    pointing = 5934
    sca = 3
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)
    wcs = snappl_image.get_wcs()
    SNLogger.debug(wcs.pixel_to_world(2044, 2044))
    snappl_cutout = snappl_image.get_ra_dec_cutout(test_snra, test_sndec, size)
    wgt_matrix = get_weights([snappl_cutout], test_snra, test_sndec,
                             gaussian_var=1000, cutoff=4)

    test_wgt_matrix = np.load(pathlib.Path(__file__).parent
                              / "testdata/test_wgt_matrix.npy")
    np.testing.assert_allclose(wgt_matrix, test_wgt_matrix, atol=1e-7)


def test_construct_transient_scene():
    lam, flambda = [1000, 26000], [1, 1]
    sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                     wave_type="Angstrom",
                     flux_type="fphotons")

    comparison_image = np.load(pathlib.Path(__file__).parent
                               / "testdata/test_psf_source.npy")

    psf_image = construct_transient_scene(x=2044, y=2044, pointing=43623, sca=7,
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
        SNLogger.debug(f"Saving diagnostic image to {im_path}")
        plt.savefig(im_path)
        plt.close()

        assert False, f"PSF source images do not match, a diagnostic " \
                      f"image has been saved to {im_path}. Error: {e}"


def test_extract_id_using_ra_dec(sn_path):
    ra = 7.3447740
    dec = -44.919229
    ID, dist = extract_id_using_ra_dec(sn_path, ra, dec, radius=5 * u.arcsec, object_type="SN")
    np.testing.assert_equal(ID, 40120913), "The ID extracted from the RA/Dec does not match the expected value."
    np.testing.assert_allclose(dist, 0.003364, rtol=1e-3), \
        "The distance from the RA/Dec to the SN does not match the expected value of 0.003364 arcsec."


def test_build_lc_and_add_truth(roman_path, sn_path):
    exposures = pd.DataFrame(
        {
            "pointing": [5934, 35198],
            "sca": [3, 2],
            "date": [62000.40235, 62495.605],
            "detected": [False, True],
            "filter": ["Y106", "Y106"],
            "x": [2044, 2044],
            "y": [2044, 2044],
            "x_cutout": [5, 5],
            "y_cutout": [5, 5]
        }
    )

    explist = Table.from_pandas(exposures)
    explist.sort(["detected", "sca"])

    # Getting a WCS to use
    pointing = 5934
    sca = 3
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)

    wcs = snappl_image.get_wcs()

    image_list = []
    cutout_image_list = []

    for i in range(len(explist["date"])):
        img = ManualFITSImage(
            header=None, data=np.zeros((4085, 4085)), noise=np.zeros((4085, 4085)), flags=np.zeros((4085, 4085)),
        )
        img.mjd = explist["date"][i]
        img.filter = explist["filter"][i]
        img.pointing = explist["pointing"][i]
        img.sca = explist["sca"][i]
        img._wcs = wcs
        img.band = "Y106"
        image_list.append(img)
        cutout_image_list.append(img)

    lc_model = campari_lightcurve_model(flux=100, sigma_flux=10,
                                        image_list=image_list, cutout_image_list=cutout_image_list, LSB=25.0, diaobj=diaobj)

    diaobj = DiaObject.find_objects(id=20172782, ra=7, dec=-41,  collection="manual")[0]
    diaobj.mjd_start = 62001.0
    diaobj.mjd_end = np.inf

    # The data values are arbitary, just to check that the lc is constructed properly.
    lc = build_lightcurve(diaobj, lc_model)
    saved_lc = Table.read(pathlib.Path(__file__).parent / "testdata/saved_lc_file.ecsv", format="ascii.ecsv")

    for i in lc.columns:
        if not isinstance(saved_lc[i][0], str):
            SNLogger.debug(f"Checking column {i}, lc: {lc[i].value}, saved_lc: {saved_lc[i]}")
            np.testing.assert_allclose(lc[i].value, saved_lc[i])
        else:
            np.testing.assert_array_equal(lc[i].value, saved_lc[i])
    for key in list(lc.meta.keys()):
        if not isinstance(saved_lc.meta[key], str):
            np.testing.assert_allclose(lc.meta[key], saved_lc.meta[key])
        else:
            np.testing.assert_array_equal(lc.meta[key], saved_lc.meta[key])

    # Now add the truth to the lightcurve
    # NOTE: The truth_path thing is a hacky fix, but since I have another issue raised to remove this from
    # campari entirely, I'm leaving it for now. It will be gone soon anyway.
    lc = add_truth_to_lc(lc, lc_model, sn_path, roman_path)
    saved_lc = Table.read(pathlib.Path(__file__).parent / "testdata/saved_lc_file_with_truth.ecsv", format="ascii.ecsv")

    for i in lc.columns:
        if not isinstance(saved_lc[i][0], str):
            np.testing.assert_allclose(lc[i].value, saved_lc[i])
        else:
            np.testing.assert_array_equal(lc[i].value, saved_lc[i])
    for key in list(lc.meta.keys()):
        if not isinstance(saved_lc.meta[key], str):
            np.testing.assert_allclose(lc.meta[key], saved_lc.meta[key])
        else:
            np.testing.assert_array_equal(lc.meta[key], saved_lc.meta[key])


def test_wcs_regression(roman_path):
    pointing = 5934
    sca = 3
    band = "Y106"

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)

    wcs = snappl_image.get_wcs()

    x_test, y_test = 2044, 2044
    ra, dec = wcs.pixel_to_world(x_test, y_test)
    np.testing.assert_allclose(ra, 7.471881246770769, atol=1e-7)
    np.testing.assert_allclose(dec, -44.82824910386988, atol=1e-7)

    ra_test, dec_test = 7.471881246770769, -44.82824910386988
    x, y = wcs.world_to_pixel(ra_test, dec_test)
    np.testing.assert_allclose(x, x_test, atol=1e-7)
    np.testing.assert_allclose(y, y_test, atol=1e-7)


def test_find_all_exposures_with_img_list(roman_path):
    band = "Y106"
    columns = ["pointing", "SCA"]
    image_df = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_image_list.csv", header=None, names=columns)
    SNLogger.debug(image_df)
    ra = 7.551093401915147
    dec = -44.80718106491529
    transient_start = 62450.
    transient_end = 62881.
    max_no_transient_images = None
    max_transient_images = None
    image_selection_start = None
    image_selection_end = None
    diaobj = DiaObject.find_objects(id=1, ra=ra, dec=dec, collection="manual")[0]
    diaobj.mjd_start = transient_start
    diaobj.mjd_end = transient_end

    image_list = find_all_exposures(diaobj=diaobj, roman_path=roman_path, maxbg=max_no_transient_images,
                                    maxdet=max_transient_images, band=band,
                                    image_selection_start=image_selection_start,
                                    image_selection_end=image_selection_end, pointing_list=image_df["pointing"].values)

    SNLogger.debug(f"Found {len(image_list)} images")

    compare_table = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_img_list_exposures.csv")

    np.testing.assert_array_equal(np.array([img.mjd for img in image_list]), compare_table["date"])
    np.testing.assert_array_equal(np.array([img.sca for img in image_list]), compare_table["sca"])
    np.testing.assert_array_equal(np.array([img.pointing for img in image_list]), compare_table["pointing"])


def test_extract_object_from_healpix():
    healpix = 42924408
    nside = 2**11
    object_type = "SN"
    source = "OpenUniverse2024"
    id_array = extract_object_from_healpix(healpix, nside, object_type, source=source)
    test_id_array = np.load(pathlib.Path(__file__).parent / "testdata/test_healpix_id_array.npy")
    np.testing.assert_array_equal(id_array, test_id_array), \
        "The IDs extracted from the healpix do not match the expected values."

    object_type = "star"
    id_array = extract_object_from_healpix(healpix, nside, object_type, source=source)
    test_id_array = np.load(pathlib.Path(__file__).parent / "testdata/test_healpix_star_id_array.npy")
    np.testing.assert_array_equal(id_array, test_id_array), \
        "The IDs extracted from the healpix do not match the expected values."


def test_read_healpix_file():
    healpix_file = pathlib.Path(__file__).parent / "testdata/test_healpix.dat"
    healpixes, nside = read_healpix_file(healpix_file)
    assert nside == 2048, "The nside of the healpix file does not match the expected value."
    np.testing.assert_array_equal(healpixes, [41152726, 41095375, 41005298, 41210086, 41079022, 41251041])


def test_make_sim_param_grid():
    param1 = [1, 2, 3]
    param2 = [4, 5]
    param3 = 9

    grid = make_sim_param_grid([param1, param2, param3])

    grid = np.array(grid)

    testgrid = np.array([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                         [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
                         [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]])
    np.testing.assert_array_equal(grid, testgrid), "The parameter grid does not match the expected values."


def test_handle_partial_overlap():
    cfg = Config.get()

    curfile = pathlib.Path(cfg.value("photometry.campari.paths.debug_dir")) / "30617531_Y106_romanpsf_images.npy"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    image_file = pathlib.Path(__file__).parent / "testdata/partial_overlap.txt"
    output = os.system(
        f"python ../RomanASP.py -s 30617531 -f Y106 -i {image_file}"
        " --ra 7.446894 --dec -44.771605 --object_collection manual"
        " --photometry-campari-use_roman --photometry-campari-use_real_images "
        "--no-photometry-campari-fetch_SED --photometry-campari-grid_options-type regular"
        " --photometry-campari-grid_options-spacing 5.0 --photometry-campari-cutout_size 101 "
        "--photometry-campari-weighting --photometry-campari-subtract_background --photometry-campari-source_phot_ops"
    )
    assert output == 0, "The test run on a SN failed. Check the logs"

    current = np.load(curfile, allow_pickle=True)
    comparison_weights = np.load(pathlib.Path(__file__).parent / "testdata/partial_overlap_weights.npy")
    np.testing.assert_allclose(current[2], comparison_weights, atol=1e-7), \
        "The weights do not match the expected values."


def test_calculate_surface_brightness():
    size = 25
    pointing = 5934
    sca = 3

    band = "Y106"

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)

    pointing = 13205
    sca = 1
    snappl_image_2 = img_collection.get_image(pointing=35198, sca=2, band=band)

    # Both of these test images contain this SN
    diaobj = DiaObject.find_objects(id=20172782,  collection="ou2024")[0]
    ra, dec = diaobj.ra, diaobj.dec
    cutout_1 = snappl_image.get_ra_dec_cutout(np.array([ra]), np.array([dec]), xsize=size)
    cutout_2 = snappl_image_2.get_ra_dec_cutout(np.array([ra]), np.array([dec]), xsize=size)

    LSB = calculate_local_surface_brightness([cutout_1, cutout_2])
    # We check against a pre-calculated value up to 32-bit ulp epsilon, rtol ~1e-7.
    (
        np.testing.assert_allclose(LSB, 26.068841696087837, rtol=1e-7),
        "The local surface brightness does not match the expected value.",
    )
