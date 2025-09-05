import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from astropy.table import Table

from snpit_utils.config import Config
from snpit_utils.logger import SNLogger
from campari.campari_runner import campari_runner, campari_lightcurve_model


def create_default_test_args(cfg):

    test_args = SimpleNamespace()

    test_args.filter = "Y106"
    test_args.max_no_transient_images = 24
    test_args.max_transient_images = 24
    test_args.image_selection_start = -np.inf
    test_args.image_selection_end = np.inf
    test_args.object_type = "SN"
    test_args.fast_debug = False
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
    test_args.use_roman = config.value("photometry.campari.use_roman")
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
    test_args.roman_path = config.value("photometry.campari.paths.roman_path")
    test_args.sn_path = config.value("photometry.campari.paths.sn_path")
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

    # Now test passing a healpix
    test_args.ra = None
    test_args.dec = None
    test_args.transient_start = None
    test_args.transient_end = None
    test_args.healpix = 42924408
    test_args.nside = 2**11
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    assert runner.healpixes == [42924408]
    assert runner.run_mode == "Healpix"
    # We don't need to check that it gets the right SNIDs, because that is tested in test_campari.py

    # Now test passing a healpix file
    test_args.healpix = None
    test_args.healpix_file = pathlib.Path(__file__).parent / "testdata/test_healpix.dat"
    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    SNLogger.debug(len(runner.healpixes))
    assert len(runner.healpixes) == 6
    assert runner.nside == 2048
    assert runner.run_mode == "Healpix File"
    # We don't need to check that it gets the right SNIDs, because that is tested in test_campari.py

    # Finally, check some cases  that should raise errors
    test_args.healpix_file = None
    test_args.nside = None
    test_args.object_collection = "ou24"
    test_args.SNID = None
    test_args.SNID_file = None
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


def test_lookup_object_info(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "ou24"
    test_args.SNID = 20172782
    runner = campari_runner(**vars(test_args))
    ra, dec, start, end = runner.lookup_object_info(test_args.SNID)
    assert ra == 7.551093401915147
    assert dec == -44.80718106491529
    assert start[0] == 62450.0
    assert end[0] == 62881.0


def test_get_exposures(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "ou24"
    test_args.SNID = 20172782

    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    explist = runner.get_exposures(7.731890048839705, -44.4589649005717, 62654.0, 62958.0)

    compare_table = np.load(pathlib.Path(__file__).parent / "testdata/findallexposures.npy")
    np.testing.assert_array_equal(explist["date"], compare_table["date"])

    np.testing.assert_array_equal(explist["sca"], compare_table["sca"])

    np.testing.assert_array_equal(explist["pointing"], compare_table["pointing"])


def test_get_SED_list(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "ou24"
    test_args.SNID = 40120913

    exposures = pd.DataFrame({"date": [62535.424]})

    test_args.fetch_SED = True
    test_args.object_type = "SN"

    runner = campari_runner(**vars(test_args))
    runner.decide_run_mode()
    sedlist = runner.get_sedlist(test_args.SNID, exposures)
    assert len(sedlist) == 1, "The length of the SED list is not 1"
    sn_lam_test = np.load(pathlib.Path(__file__).parent / "testdata/sn_lam_test.npy")
    np.testing.assert_allclose(sedlist[0]._spec.x, sn_lam_test, atol=1e-7)
    sn_flambda_test = np.load(pathlib.Path(__file__).parent / "testdata/sn_flambda_test.npy")
    np.testing.assert_allclose(sedlist[0]._spec.f, sn_flambda_test, atol=1e-7)


def test_build_and_save_lc(cfg):
    test_args = create_default_test_args(cfg)
    test_args.object_collection = "manual"
    test_args.SNID = 20172782

    runner = campari_runner(**vars(test_args))

    flux = np.array([1.0, 2.0, 3.0])
    sigma_flux = np.array([0.1, 0.2, 0.3])
    images = None
    model_images = None
    exposures = pd.DataFrame(data={"date": [1, 2, 3], "filter": ["Y106", "Y106", "Y106"],
                                   "detected": [True, True, True],
                                   "pointing": [1, 1, 1], "sca": [1, 1, 1], "x": [0, 0, 0], "y": [0, 0, 0],
                                   "x_cutout": [0, 0, 0], "y_cutout": [0, 0, 0]})
    exposures = Table.from_pandas(exposures)
    ra_grid = np.array([1, 2, 3])
    dec_grid = np.array([1, 2, 3])
    wgt_matrix = None
    confusion_metric = None
    best_fit_model_values = np.array([0] * 16, dtype=float)
    cutout_wcs_list = None
    sim_lc = None
    ra = 7.731890048839705
    dec = -44.4589649005717

    lc_model = campari_lightcurve_model(flux=flux, sigma_flux=sigma_flux, images=images, model_images=model_images,
                                        exposures=exposures, ra_grid=ra_grid, dec_grid=dec_grid,
                                        wgt_matrix=wgt_matrix, confusion_metric=confusion_metric,
                                        best_fit_model_values=best_fit_model_values, cutout_wcs_list=cutout_wcs_list,
                                        sim_lc=sim_lc)

    runner.build_and_save_lightcurve(test_args.SNID, lc_model, ra, dec, None)

    output_dir = pathlib.Path(cfg.value("photometry.campari.paths.output_dir"))
    filename = "20172782_Y106_romanpsf_lc.ecsv"
    filepath = output_dir / filename

    assert filepath.exists(), f"Lightcurve file {filename} was not created."


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
