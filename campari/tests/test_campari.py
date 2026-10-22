# Standard Libary
import os
import uuid
import pathlib
import sys
import tempfile
import warnings

# Common Library
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
import pytest

# Astronomy Library
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import galsim


# SNPIT
from campari import RomanASP
from campari.data_construction import find_all_exposures, construct_one_image
from campari.image_simulator_run import run_sim
from campari.io import (
    build_lightcurve,
    read_healpix_file,
    save_lightcurve,
)
from campari.model_building import (
    construct_static_scene,
    construct_transient_scene,
    make_adaptive_grid,
    make_contour_grid,
    make_regular_grid,
    build_model_for_one_image,
)
from campari.plotting import plot_lc
from campari.utils import (calc_mag_and_err,
                           calculate_background_level,
                           calculate_local_surface_brightness,
                           get_weights,
                           make_sim_param_grid,
                           campari_lightcurve_model)
import snappl
from snappl.dbclient import SNPITDBClient
from snappl.diaobject import DiaObject
from snappl.image import FITSImageStdHeaders
from snappl.imagecollection import ImageCollection
from snappl.config import Config
from snappl.lightcurve import Lightcurve
from snappl.logger import SNLogger
from snappl.provenance import Provenance

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)

cfg = Config.get()
output_dir = cfg.value("photometry.campari_io.output_dir")
debug_dir = cfg.value("photometry.campari_io.debug_dir")


@pytest.fixture(scope="module")
def photometry_test_data(cfg):
    return cfg.value("photometry.test_data")


@pytest.fixture(scope="module")
def campari_test_data(cfg):
    return cfg.value("photometry.campari_io.test_data")


def compare_lightcurves(lc1_path, lc2_path, overwrite_meta=False, check_truth=False):
    # lc1 is new, lc2 is old

    lc1 = QTable.read(lc1_path, format="ascii.ecsv")
    lc2 = QTable.read(lc2_path, format="ascii.ecsv")
    col1s = lc1.columns
    col2s = lc2.columns
    col1s = set(col1s)
    col2s = set(col2s)

    bothcols = col1s & col2s

    metacols1 = lc1.meta.keys()
    metacols2 = lc2.meta.keys()
    metacols1 = set(metacols1)
    metacols2 = set(metacols2)

    bothmetacols = metacols1 & metacols2
    SNLogger.debug("OLD METADATA:")
    SNLogger.debug(lc2.meta)
    SNLogger.debug("NEW METADATA:")
    SNLogger.debug(lc1.meta)

    SNLogger.debug("OLD LIGHTCURVE DATA:")
    SNLogger.debug(lc2)
    SNLogger.debug("NEW LIGHTCURVE DATA:")
    SNLogger.debug(lc1)

    # This list is used to keep track of which columns failed the comparison, so we can report them all at once.
    # This is more informative than failing on just one. For instance, if the location changed and the flux changed,
    # we'd know that the flux changed because the location changed, but we wouldn't know that if we failed
    # on the flux changing first. This is much more useful in my opinion.
    failed_cols = []

    for col in bothcols:
        SNLogger.debug(f"Checking col {col}")

        if "tru" in col and not check_truth:
            SNLogger.debug(f"Skipping truth column {col} because check_truth is False.")
            continue

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
        """
          import numpy
          a = numpy.float32( 1e8 )
          print( f"a={a}" )
          b = numpy.float32( 1 )
          print( f"b={b}" )
          print( a - ( a - b ) )
          print( a - a + b )
        """
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
        if isinstance(lc1[col][0], str) or isinstance(lc1[col][0], np.str_):
            try:
                np.testing.assert_array_equal(lc1[col], lc2[col])
            except Exception:
                failed_cols.append(col)
        else:
            # Switching from one type of WCS to another gave rise in a
            # difference of about 1e-9 pixels for the grid, which led to a
            # change in flux of 2e-7. I don't want switching WCS types to make
            # this fail, so I put the rtol at just above that level.
            # December 2025: In addition, when running with nprocs > 1
            # the WCS objects are pickled and unpickled which can
            # slightly change the numerical results. I found that it altered recovered flux
            # by about 1.4 MICRO mags. So I am increasing the rtol to 7e-6 to allow for this.
            try:
                np.testing.assert_allclose(lc1[col], lc2[col], rtol=7e-6)
            except Exception:
                failed_cols.append(col)

    if len(failed_cols) > 0:
        for f in failed_cols:
            SNLogger.debug(f"Column {f} values in lc1: {lc1[f]}")
            SNLogger.debug(f"Column {f} values in lc2: {lc2[f]}")
            SNLogger.debug("###################################")
        raise AssertionError(f"The lightcurves do not match for columns {failed_cols}")
    if overwrite_meta:
        SNLogger.debug("At this point, all the data columns match."
                       " I am now overwriting the metadata of lc2 with that of lc1.")
        # print the difference between the two meta dictionaries
        for key in lc1.meta:
            if key not in lc2.meta:
                SNLogger.debug(f"Metadata key {key} is missing in lc2")
            elif lc1.meta[key] != lc2.meta[key]:
                SNLogger.debug(f"Metadata difference: {key} - {lc1.meta[key]} vs {lc2.meta[key]}")

        lc2.meta = lc1.meta
        lc2.write(lc2_path, format="ascii.ecsv", overwrite=True)
        return None

    for col in bothmetacols:
        msg = f"The lightcurves do not match for meta column {col}"
        if "provenance" in col or "diaobject_id" in col:
            continue
            # We want to check provenances at the end. Otherwise, the code will fail when it detects that the
            # provenance is different before it can tell us which column is different.
        SNLogger.debug(f"Checking meta col {col}")
        if isinstance(lc1.meta[col], str) or isinstance(lc2.meta[col], str):
            np.testing.assert_array_equal(str(lc1.meta[col]), str(lc2.meta[col])), msg
        elif isinstance(lc1.meta[col], type(None)) or isinstance(lc2.meta[col], type(None)):
            np.testing.assert_equal(lc1.meta[col], lc2.meta[col]), msg
        elif isinstance(lc1.meta[col], dict) or isinstance(lc2.meta[col], dict):
            np.testing.assert_equal(lc1.meta[col], lc2.meta[col]), msg
        else:
            np.testing.assert_allclose(lc1.meta[col], lc2.meta[col], rtol=3e-7), msg

    unique_to_col1s = metacols1.difference(metacols2)
    unique_to_col2s = metacols2.difference(metacols1)
    assert len(unique_to_col1s) == 0 and len(unique_to_col2s) == 0, "The meta columns in the LCs do not match." + \
        f" Unique to lc1: {unique_to_col1s}, Unique to lc2: {unique_to_col2s}. However, all common columns matched."

    for col in bothmetacols:
        msg = f"The lightcurves do not match for meta column {col}"
        if "provenance" in col:
            SNLogger.debug(f"Checking meta col {col}")
            np.testing.assert_array_equal(str(lc1.meta[col]), str(lc2.meta[col])), msg


def test_find_all_exposures():
    SNLogger.debug(f"URL {Config.get().value('system.db.url')}")
    diaobj = DiaObject.find_objects(name=1, ra=7.731890048839705, dec=-44.4589649005717, collection="manual")[0]
    diaobj.mjd_start = 62654.0
    diaobj.mjd_end = 62958.0
    image_list, _ = find_all_exposures(diaobj=diaobj, band="Y106", maxbg=24,
                                       maxdet=24,
                                       truth="simple_model", image_collection="ou2024")

    compare_table = np.load(pathlib.Path(__file__).parent / "testdata/findallexposures.npy", allow_pickle=True)
    compare_table = pd.DataFrame(compare_table, columns=["pointing", "sca", "date", "band",
                                 "detected", "observation_id"])

    compare_table = compare_table.sort_values("date")

    image_date_list = np.array([img.mjd for img in image_list])
    order = np.argsort(image_date_list)

    np.testing.assert_array_equal(
        np.sort(image_date_list),
        compare_table["date"]
    )

    np.testing.assert_array_equal(
        np.array([img.sca for img in image_list])[order],
        compare_table["sca"]
    )

    np.testing.assert_array_equal(
        np.array([img.observation_id for img in image_list])[order],
        compare_table["observation_id"]
    )


def test_savelightcurve():
    with tempfile.TemporaryDirectory() as output_dir:
        lc_file = output_dir + "/" + "test_Y106_test_lc.ecsv"
        lc_file = pathlib.Path(lc_file)

        meta_dict = {
                "provenance_id": uuid.uuid4(),
                "diaobject_id": uuid.uuid4(),
                "iau_name": "ACoolLightcurve",
                "ra": 70.0,
                "ra_err": 1e-5,
                "dec": 40.0,
                "dec_err": 1e-5,
                "local_surface_brightness_Y106": 18.0,
                "band": "Y106",
                "ra_dec_covar": 0.0,
                "diaobject_position_id": uuid.uuid4(),

            }

        data_dict = {
                "mjd": [60000.0, 60001.0],
                "flux": [1000.0, 1100.0],
                "flux_err": [50.0, 55.0],
                "zpt": [25.0, 25.0],
                "NEA": [5.0, 5.0],
                "sky_background": [200.0, 210.0],
                "observation_id": [str(12345), str(12346)],
                "sca": [3, 3],
                "sky_rms": [30, 30],
                "pix_x": [1, 1],
                "pix_y": [1, 1]
            }
        lc = Lightcurve(data=data_dict, meta=meta_dict)
        # save_lightcurve defaults to saving to system.paths.lightcurves
        save_lightcurve(lc=lc, identifier="test", psftype="test", output_path=output_dir)
        assert lc_file.is_file()
        # TODO: look at contents?


@pytest.mark.run_last
def test_run_on_star(campari_test_data, cfg, overwrite_meta):
    # Call it as a function first so we can pdb and such

    curfile = pathlib.Path(output_dir) / "40973166870_Y106_ou24psf_slow_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    args = ["_", "--diaobject-name", "40973166870", "-f", "Y106", "-i",
            f"{campari_test_data}/test_image_list_star.csv", "--diaobject-collection", "manual",
            "--object_type", "star", "--photometry-campari-grid_options-type", "none",
            "--photometry-campari-subtract_background_method", "SKY_MEAN",
            "--ra", "7.5833264", "--dec", "-44.809659", "--photometry-campari-psf-transient_class", "ou24PSF_slow",
            "--image-collection", "ou2024", "--no-save-to-db", "--photometry-campari-grid_options-gaussian_var", "1000"]
    orig_argv = sys.argv
    orig_config = Config.get(clone=cfg)

    try:
        sys.argv = args
        RomanASP.main()
    except Exception as ex:
        assert False, str(ex)
    finally:
        sys.argv = orig_argv
        # ugly :( never do in real life
        cfg._data = orig_config._data

    compare_lightcurves(curfile, pathlib.Path(__file__).parent / "testdata/test_star_lc.ecsv",
                        overwrite_meta=overwrite_meta)

    curfile = pathlib.Path(output_dir) / "40973166870_Y106_ou24psf_slow_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()
    # Make sure it runs from the command line
    err_code = os.system(
        "python ../RomanASP.py --diaobject-name 40973166870 -f Y106 -i"
        f" {campari_test_data}/test_image_list_star.csv --diaobject-collection manual "
        "--object_type star --photometry-campari-grid_options-type none "
        "--photometry-campari-subtract_background_method SKY_MEAN "
        "--photometry-campari-psf-transient_class ou24PSF_slow "
        "--ra 7.5833264 --dec -44.809659 --image-collection ou2024"
        " --no-save-to-db --photometry-campari-grid_options-gaussian_var 1000",
    )
    assert err_code == 0, "The test run on a star failed. Check the logs"

    compare_lightcurves(curfile, pathlib.Path(__file__).parent / "testdata/test_star_lc.ecsv",
                        overwrite_meta=overwrite_meta)

    if overwrite_meta:
        SNLogger.debug("Overwrote metadata in test_run_on_star so I am rerunning this test.")
        test_run_on_star(campari_test_data, cfg, overwrite_meta=False)


@pytest.mark.run_last
def test_regression_function(campari_test_data, cfg, overwrite_meta):
    # This runs the same test as test_regression, with a different
    # interface.  This one calls the main() function (so is useful if
    # you want to, e.g., do things with pdb).  test_regression runs it
    # from the command line.  (And we do want to make sure that works!)

    cfg = Config.get()
    curfile = pathlib.Path(output_dir) / "20172782_Y106_ou24psf_slow_lc.ecsv"
    curfile.unlink(missing_ok=True)

    debug_file = pathlib.Path(cfg.value("photometry.campari_io.debug_dir")) / "20172782_Y106_ou24psf_slow_images.npy"
    debug_file.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()
    assert not debug_file.exists()

    a = ["_", "--diaobject-name", "20172782", "-f", "Y106", "-i", f"{campari_test_data}/test_image_list.csv",
         "--no-photometry-campari-fetch_SED", "--photometry-campari-grid_options-type",
         "contour", "--photometry-campari-cutout_size", "19", "--photometry-campari-weighting",
         "--photometry-campari-subtract_background_method", "SKY_MEAN",
         "--photometry-campari-psf-galaxy_class", "ou24PSF",
         "--photometry-campari-psf-transient_class", "ou24PSF_slow",
         "--photometry-campari-grid_options-gaussian_var", "1000",
         "--prebuilt_static_model", str(pathlib.Path(__file__).parent / "testdata/reg_psf_matrix.npy"),
         "--prebuilt_transient_model", str(pathlib.Path(__file__).parent / "testdata/reg_sn_matrix.npy"),
         "--image-collection", "ou2024", "--diaobject-collection", "ou2024", "--no-save-to-db",
         # "--add-truth-to-lc", # uncomment this to test getting truth data. Not available on all machines!
         ]

    SNLogger.debug(f"Args for test: {' '.join(a)}")

    orig_argv = sys.argv
    orig_config = Config.get(clone=cfg)
    try:
        sys.argv = a
        RomanASP.main()
        cfg = Config.get()

        compare_lightcurves(curfile, pathlib.Path(__file__).parent / "testdata/test_lc.ecsv",
                            overwrite_meta=overwrite_meta)

    finally:
        sys.argv = orig_argv
        # ugly :( never do in real life
        cfg._data = orig_config._data

    if overwrite_meta:
        SNLogger.debug("Overwrote metadata in test_regression_function so I am rerunning this test.")
        test_regression_function(campari_test_data, cfg, overwrite_meta=False)


@pytest.mark.run_last
@pytest.mark.parametrize("nprocs", [(1), (2)])
def test_regression(campari_test_data, overwrite_meta, nprocs, cfg):
    # Regression lightcurve was changed on June 6th 2025 because we were on an
    # outdated version of snappl.
    # Weighting is a Gaussian width 1000 when this was made
    # In the future, this should be True, but random seeds not working rn.

    curfile = pathlib.Path(output_dir) / "20172782_Y106_ou24psf_slow_lc.ecsv"
    curfile.unlink(missing_ok=True)
    debug_file = pathlib.Path(cfg.value("photometry.campari_io.debug_dir")) / "20172782_Y106_ou24psf_slow_images.npy"
    debug_file.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()
    assert not debug_file.exists()

    output = os.system(
        f"python ../RomanASP.py --diaobject-name 20172782 -f Y106 -i {campari_test_data}/test_image_list.csv "
        "--photometry-campari-psf-galaxy_class ou24PSF "
        "--no-photometry-campari-fetch_SED "
        "--photometry-campari-grid_options-type contour "
        "--photometry-campari-cutout_size 19 "
        "--photometry-campari-weighting "
        "--photometry-campari-subtract_background_method SKY_MEAN "
        "--photometry-campari-psf-transient_class ou24PSF_slow "
        "--save_model --image-collection ou2024 "
        " --no-save-to-db"
        # "--add-truth-to-lc" Uncomment this to test getting truth information. Not available on all machines!
        " --diaobject-collection ou2024"
        f" --nprocs {nprocs}"
        " --photometry-campari-grid_options-gaussian_var 1000"
    )
    assert output == 0, "The test run on a SN failed. Check the logs"

    compare_lightcurves(curfile, pathlib.Path(__file__).parent / "testdata/test_lc.ecsv", overwrite_meta=overwrite_meta)
    if overwrite_meta:
        SNLogger.debug("Overwrote metadata in test_regression so I am rerunning this test.")
        test_regression(campari_test_data, cfg=cfg, overwrite_meta=False, nprocs=nprocs)


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
    wcs_dict = fits.Header(wcs_dict)
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        img = FITSImageStdHeaders(header=fits.Header(wcs_dict), path="/dev/null", data=np.zeros((25, 25)))
        ra_grid, dec_grid = make_regular_grid(img,
                                              spacing=3.0, subsize=9.0)
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
    wcs_dict = fits.Header(wcs_dict)
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        SNLogger.debug(f"compare_images shape: {compare_images.shape}")
        image = compare_images[0].reshape(11, 11)
        img_obj = FITSImageStdHeaders(header=fits.Header(wcs_dict), data=image, path="/dev/null")
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
    wcs_dict = fits.Header(wcs_dict)
    for wcs in [snappl.wcs.AstropyWCS.from_header(wcs_dict)]:
        compare_images = np.load(pathlib.Path(__file__).parent
                                 / "testdata/images.npy")
        image = compare_images[0].reshape(11, 11)
        img_obj = FITSImageStdHeaders(header=fits.Header(wcs_dict), data=image, path="/dev/null")
        ra_grid, dec_grid = make_contour_grid(img_obj)
        msg = f"RA vals do not match to {atol:.1e}."
        np.testing.assert_allclose(ra_grid[:4], test_ra, atol=atol, rtol=1e-9), msg
        msg = f"Dec vals do not match to {atol:.1e}."
        np.testing.assert_allclose(dec_grid[:4], test_dec, atol=atol, rtol=1e-9), msg


@pytest.mark.parametrize("BG_SIZE", [39, 71, 101])
@pytest.mark.parametrize("BG_TRUTH", [0.23])
@pytest.mark.parametrize("source_brightness", [0, 1e0, 1e1, 1e2])
def test_calculate_background_level_with_sources(BG_SIZE, BG_TRUTH, source_brightness):
    sigma = .02
    size = BG_SIZE
    rng1 = np.random.default_rng(seed=4242424242424242)
    test_data = rng1.normal(loc=BG_TRUTH, scale=sigma, size=(size, size))

    # Now with an injected fake source
    test_data_with_sources = test_data.copy()

    gaussian = np.zeros((size, size))
    gaussian = source_brightness * np.exp(-((np.arange(size) - size // 2)**2 +
         (np.arange(size)[:, None] - size // 2)**2) / (2 * 1.0**2))
    test_data_with_sources += gaussian
    output_with_sources = calculate_background_level(test_data_with_sources)
    np.testing.assert_allclose(output_with_sources, BG_TRUTH, rtol=1e-3)


def test_calc_mag_and_err():
    flux = np.array([-1e2, 1e2, 1e3, 1e4])
    sigma_flux = np.array([10, 10, 10, 10])
    band = "Y106"
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)

    test_mag = np.array([np.nan, 27.66165575, 25.16165575, 22.66165575])
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


def test_construct_static_scene(cfg):

    observation_id = "5934"
    sca = 3
    size = 9
    band = "Y106"

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(observation_id=observation_id, sca=sca, band=band)

    wcs = snappl_image.get_wcs()

    ra_grid = np.array([7.47193824, 7.47204612, 7.472154, 7.4718731, 7.47198098])
    dec_grid = np.array([-44.8280889, -44.82804109, -44.82799327, -44.82801657, -44.82796875])

    psf_background = construct_static_scene(ra_grid, dec_grid, wcs, x_loc=2044, y_loc=2044,
                                            stampsize=size, band="Y106", image=snappl_image)

    test_psf_background = np.load(pathlib.Path(__file__).parent / "testdata/test_psf_bg.npy")

    np.testing.assert_allclose(psf_background, test_psf_background, atol=1e-7)


def test_get_weights():
    size = 7
    test_snra = np.array([7.471881246770769])
    test_sndec = np.array([-44.82824910386988])
    observation_id = "5934"
    sca = 3
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(observation_id=observation_id, sca=sca, band=band)
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

    cfg = Config.get()
    orig_fetch_SED = cfg.value("photometry.campari.fetch_SED")
    # This will need to go away once the PSF object is split in phot ops and non phot ops

    try:
        cfg._static = False
        cfg.set_value("photometry.campari.fetch_SED", False)
        cfg._static = True

        psf_image = construct_transient_scene(x0=2044, y0=2044, observation_id="43623", sca=7,
                                              stampsize=25, x=2044,
                                              y=2044, sed=sed,
                                              flux=1)

    finally:
        cfg._static = False
        cfg.set_value("photometry.campari.fetch_SED", orig_fetch_SED)
        cfg._static = True

    comparison_image = np.load(pathlib.Path(__file__).parent
                               / "testdata/test_psf_source.npy")

    try:
        np.testing.assert_allclose(np.sum(psf_image), np.sum(comparison_image),
                                   atol=1e-6, verbose=True)

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


def test_build_lc(cfg, overwrite_meta):
    exposures = pd.DataFrame(
        {
            "observation_id": ["5934", "35198"],
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

    sca_x_locations = np.array([68565.85410581, 68565.85410581])
    sca_y_locations = np.array([127021.05784706, 127021.05784706])
    # Getting a WCS to use
    observation_id = "5934"
    sca = 3
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(observation_id=observation_id, sca=sca, band=band)
    SNLogger.debug(f"Snappl image: {snappl_image.provenance_id}")

    wcs = snappl_image.get_wcs()

    image_list = []
    cutout_image_list = []

    for i in range(len(explist["date"])):
        img = FITSImageStdHeaders(
            header=None,
            data=np.zeros((4088, 4088)),
            noise=np.zeros((4088, 4088)),
            flags=np.zeros((4088, 4088)),
            path="/dev/null"
        )
        img.mjd = explist["date"][i]
        img.filter = explist["filter"][i]
        img.observation_id = explist["observation_id"][i]
        img.sca = explist["sca"][i]
        img._wcs = wcs
        img.band = "Y106"
        image_list.append(img)
        cutout_image_list.append(img)

    diaobj = DiaObject.find_objects(name=20172782, ra=7, dec=-41, collection="manual")[0]
    diaobj.mjd_start = 62001.0
    diaobj.mjd_end = np.inf

    lc_model = campari_lightcurve_model(flux=100.0, sigma_flux=10.0, image_list=image_list,
                                        cutout_image_list=cutout_image_list, LSB=25.0, diaobj=diaobj,
                                        sky_background=[0.0] * len(image_list), pre_transient_images=1,
                                        post_transient_images=0, sca_x_locations=sca_x_locations,
                                        sca_y_locations=sca_y_locations)

    upstreams = []
    cam_prov = Provenance(
        process="campari",
        major=0,
        minor=42,
        params=cfg,
        keepkeys=["photometry.campari"],
        omitkeys=None,
        upstreams=upstreams,
    )

    # The data values are arbitary, just to check that the lc is constructed properly.
    lc = build_lightcurve(diaobj, lc_model, cam_prov=cam_prov)

    lc = Table(data=lc.data, meta=lc.meta)
    lc.write(pathlib.Path(__file__).parent / "testdata/newly_built_lc.ecsv", format="ascii.ecsv", overwrite=True)

    compare_lightcurves(
        pathlib.Path(__file__).parent / "testdata/newly_built_lc.ecsv",
        pathlib.Path(__file__).parent / "testdata/saved_lc_file.ecsv",
        overwrite_meta=overwrite_meta
    )
    if overwrite_meta:
        SNLogger.debug("Overwrote metadata in test_build_lc so I am rerunning this test.")
        test_build_lc(cfg, overwrite_meta=False)


def test_wcs_regression():
    observation_id = "5934"
    sca = 3
    band = "Y106"

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(observation_id=observation_id, sca=sca, band=band)

    wcs = snappl_image.get_wcs()

    x_test, y_test = 2044, 2044
    ra, dec = wcs.pixel_to_world(x_test, y_test)
    np.testing.assert_allclose(ra, 7.471881246770769, atol=1e-7)
    np.testing.assert_allclose(dec, -44.82824910386988, atol=1e-7)

    ra_test, dec_test = 7.471881246770769, -44.82824910386988
    x, y = wcs.world_to_pixel(ra_test, dec_test)
    np.testing.assert_allclose(x, x_test, atol=1e-7)
    np.testing.assert_allclose(y, y_test, atol=1e-7)


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

    curfile = pathlib.Path(debug_dir) / "30617531_Y106_ou24psf_slow_photonshoot_images.npy"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    #  we know we're really running this test!
    assert not curfile.exists()

    image_file = pathlib.Path(__file__).parent / "testdata/partial_overlap.txt"

    orig_footprint_thresh = cfg.value("photometry.campari.footprint_threshold")

    try:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", None)
        cfg._static = True

        output = os.system(
            f"python ../RomanASP.py --diaobject-name 30617531 -f Y106 -i {image_file}"
            " --ra 7.446894 --dec -44.771605 --diaobject-collection manual"
            " --photometry-campari-psf-galaxy_class ou24PSF_photonshoot "
            " --photometry-campari-psf-transient_class ou24PSF_slow_photonshoot "
            # This was OU24 PSF and PSF_slow in a different branch?
            "--no-photometry-campari-fetch_SED --photometry-campari-grid_options-type regular"
            " --photometry-campari-grid_options-spacing 5.0 --photometry-campari-cutout_size 101 "
            "--photometry-campari-weighting --photometry-campari-subtract_background_method calculate "
            "--transient_start 63000 --transient_end 63000.0001 --no-save-to-db --image-collection ou2024"
            " --photometry-campari-grid_options-gaussian_var 1000"
            " --photometry-campari-footprint_threshold -1"  # Don't want to confuse with weights
        # lost due to footprint thresholding
        )
        assert output == 0, "The test run on a SN failed. Check the logs"

        current = np.load(curfile, allow_pickle=True)
        comparison_weights = np.load(pathlib.Path(__file__).parent / "testdata/partial_overlap_weights.npy")

        np.testing.assert_allclose(current[2], comparison_weights, atol=1e-7), \
            "The weights do not match the expected values."
    finally:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", orig_footprint_thresh)
        cfg._static = True


@pytest.mark.requires_simdex  # I think this image is in the photometry_test_data, fix this at some point.
def test_calculate_surface_brightness():
    size = 25
    observation_id = str(5934)
    sca = 3

    band = "Y106"
    dbclient = SNPITDBClient()
    image_collection = "ou2024"
    SNLogger.debug(f"getting image collection with name {image_collection}")
    img_collection = ImageCollection().get_collection(
        collection=image_collection
    )
    SNLogger.debug(f"Got image collection: {img_collection}")
    SNLogger.debug(f"Getting image with observation_id={observation_id}, sca={sca}, band={band}")
    snappl_image = img_collection.get_image(observation_id=observation_id, sca=sca, band=band)

    observation_id = "13205"
    sca = 1
    snappl_image_2 = img_collection.get_image(observation_id="35198", sca=2, band=band)

    # Both of these test images contain this SN
    SNLogger.debug("Trying to load diaobj")
    diaobj = DiaObject.find_objects(collection="ou2024", dbclient=dbclient, name=20172782)[0]
    ra, dec = diaobj.ra, diaobj.dec
    cutout_1 = snappl_image.get_ra_dec_cutout(np.array([ra]), np.array([dec]), xsize=size)
    cutout_2 = snappl_image_2.get_ra_dec_cutout(np.array([ra]), np.array([dec]), xsize=size)

    LSB = calculate_local_surface_brightness([cutout_1, cutout_2])
    # We check against a pre-calculated value up to 32-bit ulp epsilon, rtol ~1e-7.
    (
        np.testing.assert_allclose(LSB, 26.068841696087837, rtol=1e-7),
        "The local surface brightness does not match the expected value.",
    )


@pytest.mark.parametrize("nprocs", [(10), (1)])
def test_construct_one_image(cfg, campari_test_data, nprocs):
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")

    observation_ids = ["5934", "35198"]
    scas = [3, 2]
    band = "Y106"

    image_list = []
    for oid, sca in zip(observation_ids, scas):
        image_list.append(img_collection.get_image(observation_id=oid, sca=sca, band=band))

    with open(pathlib.Path(__file__).parent / "testdata/reg_test_cutouts.pkl", "rb") as f:
        reg_cutout_list = pickle.load(f)

    ra = 7.551093401915147
    dec = -44.80718106491529
    size = 19
    truth = "simple_model"
    subtract_background_method = "SKY_MEAN"

    for nprocs in [10, 1]:
        SNLogger.debug(f"Testing construct_one_image with nprocs={nprocs}")
        cutout_image_list = []
        results = []
        if nprocs > 1:
            with Pool(nprocs) as pool:
                for indx, image in enumerate(image_list):
                    SNLogger.debug(f"image path {image.full_filepath}")
                    SNLogger.debug(f"Constructing cutout for image {indx + 1} of {image}")
                    results.append(pool.apply_async(construct_one_image, kwds={"indx": indx, "image": image,
                                                                               "ra": ra, "dec": dec, "size": size,
                                                                               "truth": truth,
                                                                               "subtract_background_method":
                                                                               subtract_background_method}))

                pool.close()
                pool.join()
        else:
            for indx, image in enumerate(image_list):
                SNLogger.debug(f"Constructing cutout for image {indx + 1} of {image}")
                results.append(construct_one_image(indx=indx, image=image,
                                                   ra=ra, dec=dec, size=size, truth=truth,
                                                   subtract_background_method=subtract_background_method))

    for r in results:
        if nprocs > 1:
            res = r.get()
        else:
            res = r
        cutout_image_list.append(res[0])

    for cutout, reg_cutout in zip(cutout_image_list, reg_cutout_list):
        np.testing.assert_allclose(cutout.data, reg_cutout.data, atol=1e-7)
        np.testing.assert_allclose(cutout.noise, reg_cutout.noise, atol=1e-7)
        np.testing.assert_array_equal(cutout.flags, reg_cutout.flags)
        np.testing.assert_array_equal(cutout.get_wcs()._wcs.to_header(), reg_cutout.get_wcs()._wcs.to_header())


def test_build_model_one_image():
    ra = 7.551093401915147
    dec = -44.80718106491529
    size = 19

    with open(pathlib.Path(__file__).parent / "testdata/reg_test_imglist.pkl", "rb") as f:
        image_list = pickle.load(f)

    with open(pathlib.Path(__file__).parent / "testdata/reg_grid_and_arrays.pkl", "rb") as f:
        ra_grid, dec_grid, reg_bg_array, reg_sn_array = pickle.load(f)

    observation_ids = ["5934", "35198"]
    scas = [3, 2]
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    image_list = []
    for oid, sca in zip(observation_ids, scas):
        image_list.append(img_collection.get_image(observation_id=oid, sca=sca, band=band))

    bg_array, sn_array = build_model_for_one_image(image=image_list[0], ra=ra, dec=dec,
                                                   grid_type="contour", ra_grid=ra_grid, dec_grid=dec_grid, size=size,
                                                   pixel=False, band="Y106", sedlist=None,
                                                   image_index=0, num_total_images=2,
                                                   num_detect_images=1, prebuilt_psf_matrix=None,
                                                   prebuilt_sn_matrix=None, subtract_background_method="calculate")

    reg_bg_array = np.load(pathlib.Path(__file__).parent / "testdata/reg_bg_array.npy")
    reg_sn_array = np.load(pathlib.Path(__file__).parent / "testdata/reg_sn_array.npy", allow_pickle=True)

    np.testing.assert_allclose(bg_array, reg_bg_array, atol=1e-7)
    np.testing.assert_equal(sn_array, reg_sn_array)  # We expect Nones here


def test_image_simulator_script():
    test_data_path = pathlib.Path(__file__).parent / "testdata"
    simulation_number = 0
    func_name = "image_simulator_script_test"
    run_name = func_name + f"seed{simulation_number}"

    # This is the file spat out by the image simulator sayingthe locations of all the files.
    imagelist_filename = test_data_path / f"image_list_{run_name}.txt"
    # Make sure we delete it beforehand
    imagelist_filename.unlink(missing_ok=True)
    run_sim(
        seed=simulation_number,  # Set seed for reproducibility, this must be the seed that the data was simulated with.
        images_aligned=False,
        poisson_noise=True,
        sky_noise=True,
        static_source="galaxy",
        static_source_mag=22,
        transient_peak_mag=24,
        band="H158",
        mjd=np.arange(60000, 60075, 10),
        psf_class="ou24PSF_slow",
        run_dir=func_name,
        output_path=test_data_path,
        run_name_base=func_name,
        bulge_R=2,
        bulge_n=3,
        disk_R=4,
        disk_n=1,  # Simulated Galaxy Params
        test_data_path=test_data_path,
    )

    regression_path = test_data_path / "image_simulator_script_regression/image_simulator_script_testseed0/"
    regression_images_to_check = list(regression_path.glob("*image.fits"))

    # Open the imagelist file and get the list of generated images
    with open(imagelist_filename, "r") as f:
        generated_image_paths = [line.strip() for line in f.readlines()]
    generated_images = [pathlib.Path(p + "_image.fits") for p in generated_image_paths]

    for gen_img_path in generated_images:
        # Get corresponding regression image path
        reg_img_path = regression_path / gen_img_path.name
        assert reg_img_path.exists(), f"Regression image {reg_img_path} does not exist, but it was simulated just now!"
        # remove this path from the images to check
        regression_images_to_check.remove(reg_img_path)
        assert not os.path.samefile(reg_img_path, gen_img_path), "Regression image and generated image paths should " \
                                                          " not be the same, check the test setup."
        with fits.open(reg_img_path) as reg_hdul, fits.open(gen_img_path) as gen_hdul:
            try:
                reg_data = reg_hdul[0].data
                gen_data = gen_hdul[0].data
                np.testing.assert_allclose(reg_data, gen_data, atol=1e-7, err_msg=f"Image at {gen_img_path} does not"
                                           f" match regression data at {reg_img_path}")
                SNLogger.debug(f"Image {gen_img_path} matches regression data.")
            except AssertionError as e:
                # Plot the images for debugging
                matplotlib.use("pdf")
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.title("Generated Image")
                plt.imshow(gen_data, origin="lower")
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.title("Regression Image")
                plt.imshow(reg_data, origin="lower")
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.title("Difference (log scale)")
                plt.imshow(np.log10(np.abs(gen_data - reg_data)), origin="lower")
                plt.colorbar(label="log10( |generated - regression| )")
                debug_image_path = test_data_path / f"debug_{gen_img_path.stem}.pdf"
                SNLogger.debug(f"Saving debug image to {debug_image_path}")
                plt.savefig(debug_image_path)
                raise AssertionError(e)
    assert len(regression_images_to_check) == 0, "Some regression images were not checked against generated images"\
                                                 f" meaning they were not simulated: {regression_images_to_check}"

    # To update regression, do:
    # mv /scratch/campari/campari/tests/testdata/image_simulator_script_test/*
    #  /scratch/campari/campari/tests/testdata/image_simulator_script_regression


# Full disclosure: These tests were written by Claude but I have checked them. I knew what I wanted the tests to be
# and I knew how to write them, I just figured this would be faster. I also added some comments to make it more
# readable.
SIZE = 31
SIZE_SQ = SIZE**2
N_IMAGES = 2          # 1 non-detection epoch, 1 the detection epoch
BAND = "Y106"
# Pixel (2044, 2044) of this image is at this position (see test_wcs_regression).
RA0, DEC0 = 7.471881246770769, -44.82824910386988
X0 = Y0 = 2044


@pytest.fixture(scope="module")
def setup(cfg):
    """Build the expensive pieces once and share them between the tests.
       Drawing PSFs is slow, so this avoids repeating it per test.
    """
    img_collection = ImageCollection().get_collection("ou2024")
    snappl_image = img_collection.get_image(observation_id="5934", sca=3, band=BAND)
    wcs = snappl_image.get_wcs()

    # 5x5 grid of model points, 1 pixel apart, centred on the stamp centre.
    offsets = np.arange(-2, 3)
    xx, yy = np.meshgrid(X0 + offsets, Y0 + offsets)
    ra_grid, dec_grid = wcs.pixel_to_world(xx.flatten(), yy.flatten())

    psf_matrix = construct_static_scene(ra_grid, dec_grid, wcs, x_loc=X0, y_loc=Y0,
                                       stampsize=SIZE, band=BAND, image=snappl_image)

    # A single model point at the transient location stands in for the SN stamp.
    sn_stamp = construct_static_scene(np.array([RA0]), np.array([DEC0]), wcs, x_loc=X0, y_loc=Y0,
                                      stampsize=SIZE, band=BAND, image=snappl_image)[:, 0]

    return {"image": snappl_image, "psf_matrix": psf_matrix, "sn_stamp": sn_stamp}


def weights(setup, psf_matrix=None, sn_matrix=None, **kwargs):
    """Call get_weights on fresh cutouts of the real image."""
    cutouts = [setup["image"].get_ra_dec_cutout(np.array([RA0]), np.array([DEC0]), SIZE)
               for _ in range(N_IMAGES)]
    return get_weights(cutouts, np.array([RA0]), np.array([DEC0]), gaussian_var=None, cutoff=None,
                       psf_matrix=psf_matrix, sn_matrix=sn_matrix, **kwargs)


def footprint_per_image(psf_matrix):
    """Calculate the footprint: total |model| per pixel."""
    return np.abs(psf_matrix).sum(axis=1).reshape(N_IMAGES, SIZE_SQ)


def tiled_psf(setup):
    # Both "epochs" use the same image, so they see the same grid model. This combines the two into a single matrix for
    # footprint calculation.
    return np.tile(setup["psf_matrix"], (N_IMAGES, 1))


def test_model_covers_only_a_small_part_of_cutout(setup):
    # The addition of these functions only makes sense if the model is a fraction of the cutout.
    # The footprint should be very small in the corners, and most pixels should be barely touched.
    fp = footprint_per_image(tiled_psf(setup))[0]
    peak = fp.max()
    assert fp[0] < 1e-2 * peak and fp[-1] < 1e-2 * peak     # far corners
    assert np.mean(fp > 1e-2 * peak) < 0.5                  # most pixels barely touched


def test_zero_threshold_masks_exactly_the_untouched_pixels(setup):
    psf_matrix = tiled_psf(setup)
    base = weights(setup)

    orig_footprint_threshold = cfg.value("photometry.campari.footprint_threshold")

    try:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", 0)
        cfg._static = True

        masked = weights(setup, psf_matrix=psf_matrix)           # footprint_threshold = 0

        covered = footprint_per_image(psf_matrix) > 0            # boolean array, True = model reaches pixel
        for i in range(N_IMAGES):
            # Multiplying by a boolean array keeps a weight where True, zeroes it where False.
            np.testing.assert_array_equal(masked[i], base[i] * covered[i])
    finally:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", orig_footprint_threshold)
        cfg._static = True


def test_relative_threshold_masks_faint_wings(setup):
    psf_matrix = tiled_psf(setup)
    base = weights(setup)
    fp = footprint_per_image(psf_matrix)

    previous_masked = np.zeros((N_IMAGES, SIZE_SQ), dtype=bool)
    previous_fraction = 0.0
    orig_footprint_threshold = cfg.value("photometry.campari.footprint_threshold")
    for rel in [1e-4, 1e-3, 1e-2]:

        try:
            cfg._static = False
            cfg.set_value("photometry.campari.footprint_threshold", rel)
            cfg._static = True
            masked = weights(setup, psf_matrix=psf_matrix)
            dropped = fp <= rel * fp.max(axis=1, keepdims=True)  # cut is relative to each image's own peak
            for i in range(N_IMAGES):
                np.testing.assert_array_equal(masked[i], base[i] * ~dropped[i])

            # A higher threshold must drop everything a lower one did, and more.
            assert np.all(dropped[previous_masked])
            assert dropped.mean() >= previous_fraction
            previous_masked, previous_fraction = dropped, dropped.mean()
        finally:
            cfg._static = False
            cfg.set_value("photometry.campari.footprint_threshold", orig_footprint_threshold)
            cfg._static = True

    # At 1% of peak, the centre survives and the corners do not.
    centre = (SIZE // 2) * SIZE + SIZE // 2
    for i in range(N_IMAGES):
        assert masked[i][centre] == base[i][centre] and base[i][centre] != 0
        assert masked[i][0] == 0 and masked[i][-1] == 0
    assert previous_fraction > 0.5


def test_fit_mode_sky_columns_prevent_any_masking(setup):
    # In "fit" mode build_model_for_one_image appends one all-ones column per
    # image (zero on the other images). Every pixel is then touched by the model.
    sky = np.zeros((N_IMAGES * SIZE_SQ, N_IMAGES))
    for j in range(N_IMAGES):
        sky[j * SIZE_SQ:(j + 1) * SIZE_SQ, j] = 1.0
    psf_matrix = np.hstack([tiled_psf(setup), sky])

    base = weights(setup)
    for footprint_threshold in [0, 1e-2]:
        orig_footprint_threshold = cfg.value("photometry.campari.footprint_threshold")
        try:
            cfg._static = False
            cfg.set_value("photometry.campari.footprint_threshold", footprint_threshold)
            cfg._static = True

            masked = weights(setup, psf_matrix=psf_matrix)
            for i in range(N_IMAGES):
                np.testing.assert_array_equal(masked[i], base[i])
        finally:
            cfg._static = False
            cfg.set_value("photometry.campari.footprint_threshold", orig_footprint_threshold)
            cfg._static = True


def test_transient_footprint_only_counts_in_detection_image(setup):
    # No background grid; only a transient stamp, in the last image (the detection image).
    psf_matrix = np.empty((N_IMAGES * SIZE_SQ, 0))

    orig_footprint_threshold = cfg.value("photometry.campari.footprint_threshold")
    try:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", 0)
        cfg._static = True

        base = weights(setup)
        masked = weights(setup, psf_matrix=psf_matrix, sn_matrix=[setup["sn_stamp"]])

        # Non-detection image: no model at all, so every weight is dropped.
        assert np.all(masked[0] == 0)
        # Detection image: weight survives where the transient stamp is nonzero.
        np.testing.assert_array_equal(masked[1], base[1] * (np.abs(setup["sn_stamp"]) > 0))
    finally:
        cfg._static = False
        cfg.set_value("photometry.campari.footprint_threshold", orig_footprint_threshold)
        cfg._static = True
