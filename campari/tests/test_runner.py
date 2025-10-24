# Standard Library
import pathlib
from types import SimpleNamespace

# Common Library
import numpy as np
import pandas as pd
import pytest

from astropy.table import Table

# SNPIT
from campari.campari_runner import campari_runner
from campari.tests.test_campari import compare_lightcurves
from campari.utils import campari_lightcurve_model
from snappl.diaobject import DiaObject
from snappl.image import FITSImageStdHeaders
from snappl.imagecollection import ImageCollection
from snappl.config import Config
from snappl.logger import SNLogger
ROMAN_IMAGE_SIZE = 4088  # Roman images are 4088x4088 pixels (4096 minus 4 on each edge)


def create_default_test_args(cfg):
    test_args = SimpleNamespace()

    test_args.filter = "Y106"
    test_args.max_no_transient_images = 24
    test_args.max_transient_images = 24
    test_args.image_selection_start = None
    test_args.image_selection_end = None
    test_args.object_type = "SN"
    test_args.fast_debug = False
    test_args.save_model = False
    test_args.prebuilt_static_model = None
    test_args.prebuilt_transient_model = None
    test_args.SNID_file = None
    test_args.SNID = None
    test_args.img_list = None
    test_args.healpix = None
    test_args.healpix_file = None
    test_args.nside = None
    test_args.object_collection = "ou24"
    test_args.transient_start = None
    test_args.transient_end = None
    test_args.ra = None
    test_args.dec = None

    config = cfg

    test_args.size = config.value("photometry.campari.cutout_size")
    test_args.use_real_images = config.value("photometry.campari.use_real_images")
    test_args.psfclass = config.value("photometry.campari.psfclass")
    test_args.avoid_non_linearity = config.value("photometry.campari.simulations.avoid_non_linearity")
    test_args.deltafcn_profile = config.value("photometry.campari.simulations.deltafcn_profile")
    test_args.do_xshift = config.value("photometry.campari.simulations.do_xshift")
    test_args.do_rotation = config.value("photometry.campari.simulations.do_rotation")
    test_args.noise = config.value("photometry.campari.simulations.noise")
    test_args.method = config.value("photometry.campari.method")
    test_args.make_initial_guess = config.value("photometry.campari.make_initial_guess")
    test_args.subtract_background = config.value("photometry.campari.subtract_background")
    test_args.weighting = config.value("photometry.campari.weighting")
    test_args.pixel = config.value("photometry.campari.pixel")
    test_args.bg_gal_flux_all = config.value("photometry.campari.simulations.bg_gal_flux")
    test_args.sim_galaxy_scale_all = config.value("photometry.campari.simulations.sim_galaxy_scale")
    test_args.sim_galaxy_offset_all = config.value("photometry.campari.simulations.sim_galaxy_offset")
    test_args.source_phot_ops = config.value("photometry.campari.source_phot_ops")
    test_args.mismatch_seds = config.value("photometry.campari.simulations.mismatch_seds")
    test_args.fetch_SED = config.value("photometry.campari.fetch_SED")
    test_args.initial_flux_guess = config.value("photometry.campari.initial_flux_guess")
    test_args.spacing = config.value("photometry.campari.grid_options.spacing")
    test_args.percentiles = config.value("photometry.campari.grid_options.percentiles")
    test_args.grid_type = config.value("photometry.campari.grid_options.type")
    test_args.base_pointing = config.value("photometry.campari.simulations.base_pointing")
    test_args.base_sca = config.value("photometry.campari.simulations.base_sca")
    test_args.run_name = config.value("photometry.campari.simulations.run_name")
    test_args.param_grid = None
    test_args.config = None
    test_args.pointing_list = None
    return test_args


def test_runner_init(cfg):
    test_args = create_default_test_args(cfg)
    runner = campari_runner(**vars(test_args))
    assert isinstance(runner, campari_runner)
    assert runner.band == test_args.filter
    assert runner.max_no_transient_images == test_args.max_no_transient_images
    assert runner.max_transient_images == test_args.max_transient_images
    assert runner.image_selection_start == test_args.image_selection_start
    assert runner.image_selection_end == test_args.image_selection_end
    assert runner.object_type == test_args.object_type
    assert runner.fast_debug == test_args.fast_debug
    assert runner.SNID_file == test_args.SNID_file
    assert runner.SNID == test_args.SNID
    assert runner.img_list == test_args.img_list
    assert runner.healpix == test_args.healpix
    assert runner.healpix_file == test_args.healpix_file
    assert runner.nside == test_args.nside
    assert runner.object_collection == test_args.object_collection
    assert runner.transient_start == test_args.transient_start
    assert runner.transient_end == test_args.transient_end
    assert runner.ra == test_args.ra
    assert runner.dec == test_args.dec

    # Check that the config is set correctly
    assert isinstance(runner.cfg, Config)


def test_decide_run_mode(cfg):
    test_args = create_default_test_args(cfg)

    # First test passing a SNID
    test_args.SNID = 20172782
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    assert runner.SNID == [20172782]
    assert runner.run_mode == "Single SNID"

    # Now Test passing a SNID file
    test_args.SNID = None
    test_args.SNID_file = pathlib.Path(__file__).parent / "testdata/test_snids.csv"
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    assert len(runner.SNID) == 50
    assert runner.run_mode == "SNID File"

    # Now test passing RA and Dec
    test_args.SNID_file = None
    test_args.ra = 10.684
    test_args.dec = 41.269
    runner = campari_runner(**vars(test_args))
    with pytest.raises(ValueError, match="Must specify --transient_start and --transient_end to run campari"):
        runner.decide_run_mode()

    test_args.transient_start = 60000.0
    test_args.transient_end = 60100.0
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    assert runner.ra == 10.684
    assert runner.dec == 41.269
    assert runner.transient_start is not None
    assert runner.transient_end is not None
    assert runner.run_mode == "RA/Dec"


    # Finally, check some cases  that should raise errors
    test_args.healpix_file = None
    test_args.nside = None
    test_args.object_collection = "ou24"
    test_args.SNID = None
    test_args.SNID_file = None
    test_args.ra = None
    test_args.dec = None
    with pytest.raises(ValueError, match="Must specify --SNID, --SNID-file, to run campari "):
        campari_runner(**vars(test_args)).decide_run_mode()

    test_args.object_collection = "manual"
    with pytest.raises(
        ValueError,
        match="Must specify --SNID, --SNID-file, --healpix, --healpix_file, or --ra and --dec to run campari.",
    ):
        campari_runner(**vars(test_args)).decide_run_mode()

    test_args.object_collection = "ou24"
    test_args.SNID = 20172782
    test_args.img_list = pathlib.Path(__file__).parent / "testdata/test_image_list.csv"
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()

    assert runner.SNID == [20172782]
    columns = ["pointing", "sca"]
    SNLogger.debug(pd.read_csv(test_args.img_list))
    np.testing.assert_array_equal(runner.pointing_list,
                                  pd.read_csv(test_args.img_list, names=columns)["pointing"].tolist())


def test_get_exposures(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "ou24"
    test_args.SNID = 20172782

    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    diaobj = DiaObject.find_objects(name=1, ra=7.731890048839705, dec=-44.4589649005717, collection="manual")[0]
    diaobj.mjd_start = 62654.0
    diaobj.mjd_end = 62958.0
    image_list = runner.get_exposures(diaobj)

    compare_table = np.load(pathlib.Path(__file__).parent / "testdata/findallexposures.npy")
    argsort = np.argsort(compare_table["date"])
    compare_table = compare_table[argsort]

    np.testing.assert_array_equal([a.mjd for a in image_list], compare_table["date"])
    np.testing.assert_array_equal([a.sca for a in image_list], compare_table["sca"])
    np.testing.assert_array_equal([a.pointing for a in image_list], compare_table["pointing"])


def test_get_SED_list(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "ou24"
    test_args.SNID = 40120913

    img = FITSImageStdHeaders(
        header=None,
        path="/dev/null",
        data=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
        noise=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
        flags=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
    )
    img.mjd = 62535.424
    img.band = "Y106"
    image_list = [img]

    orig_fetch_sed = cfg.value( "photometry.campari.fetch_SED" )


    try:

        #  I am about to do a bad thing BUT ROB SAID I COULD
        # Essentially, this needs to be edited because the regression tests set this to False for the python instance
        # which causes this test to fail if you try to run all of the tests at once.
        cfg._static = False
        cfg.set_value("photometry.campari.fetch_SED", True)
        cfg._static = True
        # phrosty sets a precedent for my heinous sin:
        # https://github.com/Roman-Supernova-PIT/phrosty/blob/54db2040feff7c183dfb9955904e957f5122f5ac/phrosty/tests/conftest.py#L37

        test_args.object_type = "SN"

        runner = campari_runner(**vars(test_args))
        runner.decide_run_mode()
        sedlist = runner.get_sedlist(test_args.SNID, image_list)
        assert len(sedlist) == 1, "The length of the SED list is not 1"
        sn_lam_test = np.load(pathlib.Path(__file__).parent / "testdata/sn_lam_test.npy")
        np.testing.assert_allclose(sedlist[0]._spec.x, sn_lam_test, atol=1e-7)
        sn_flambda_test = np.load(pathlib.Path(__file__).parent / "testdata/sn_flambda_test.npy")
        np.testing.assert_allclose(sedlist[0]._spec.f, sn_flambda_test, atol=1e-7)
    finally:
        # If it finishes or if something fails, restore the config value.
        cfg._static = False
        cfg.set_value("photometry.campari.fetch_SED", orig_fetch_sed)
        cfg._static = True


def test_build_and_save_lc(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "manual"
    test_args.SNID = 20172782

    runner = campari_runner(**vars(test_args))

    flux = np.array([1.0, 2.0, 3.0])
    sigma_flux = np.array([0.1, 0.2, 0.3])
    images = None
    model_images = None
    exposures = pd.DataFrame(data={"date": [1.0, 2.0, 3.0], "filter": ["Y106", "Y106", "Y106"],
                                   "detected": [True, True, True],
                                   "pointing": [1, 1, 1], "sca": [1, 1, 1], "x": [0, 0, 0], "y": [0, 0, 0],
                                   "x_cutout": [0, 0, 0], "y_cutout": [0, 0, 0]})

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

    for i in range(len(exposures["date"])):
        img = FITSImageStdHeaders(
            header=None,
            path="/dev/null",
            data=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
            noise=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
            flags=np.zeros((ROMAN_IMAGE_SIZE, ROMAN_IMAGE_SIZE)),
        )
        img.mjd = exposures["date"][i]
        img.band = exposures["filter"][i]
        img.pointing = exposures["pointing"][i]
        img.sca = exposures["sca"][i]
        img._wcs = wcs
        image_list.append(img)
        cutout_image_list.append(img)

    ra_grid = np.array([1, 2, 3])
    dec_grid = np.array([1, 2, 3])
    wgt_matrix = None
    LSB = 19.0
    best_fit_model_values = np.array([0] * 16, dtype=float)
    sim_lc = None
    ra = 7.731890048839705
    dec = -44.4589649005717

    lc_model = campari_lightcurve_model(flux=flux, sigma_flux=sigma_flux, images=images, model_images=model_images,
                                        image_list=image_list, cutout_image_list=cutout_image_list, ra_grid=ra_grid,
                                        dec_grid=dec_grid,
                                        wgt_matrix=wgt_matrix, LSB=LSB, sky_background=np.zeros(len(flux)),
                                        best_fit_model_values=best_fit_model_values,
                                        sim_lc=sim_lc)

    diaobj = DiaObject.find_objects(name=test_args.SNID, ra=ra, dec=dec, collection="manual")[0]
    diaobj.mjd_start = -np.inf
    diaobj.mjd_end = np.inf
    runner.build_and_save_lightcurve(diaobj, lc_model, None)

    output_dir = pathlib.Path(cfg.value("photometry.campari.paths.output_dir"))
    filename = "20172782_Y106_romanpsf_lc.ecsv"
    filepath = output_dir / filename

    assert filepath.exists(), f"Lightcurve file {filename} was not created."

    current = Table.read(filepath, format="ascii.ecsv")
    comparison = Table.read(pathlib.Path(__file__).parent / "testdata/test_build_lc.ecsv", format="ascii.ecsv")
    compare_lightcurves(current, comparison)


def test_sim_param_grid(cfg):
    test_args = create_default_test_args(cfg)
    test_args.use_real_images = False
    test_args.object_collection = "ou24"
    test_args.SNID = 20172782
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    runner.bg_gal_flux_all = [1.0, 2.0]
    runner.sim_galaxy_scale_all = [1.0, 2.0, 3.0]
    runner.sim_galaxy_offset_all = 0.0
    # Create the simulation parameter grid
    runner.create_sim_param_grid()

    test_grid = np.array([[1., 2.,  1.,  2.,  1. , 2.],
                         [1., 1.,  2.,  2.,  3. , 3.],
                         [0., 0.,  0.,  0.,  0. , 0.]])
    np.testing.assert_array_equal(runner.param_grid, test_grid)


# Creating the sim param grid is tested in test_campari.py, so we don't need to test it here.
# The __call__ method is also tested in test_campari.py, so we don't need to test it here either.
