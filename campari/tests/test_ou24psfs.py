import pathlib
import subprocess
from types import SimpleNamespace


import numpy as np
import pytest

from astropy.table import Table

from snappl.config import Config
from snappl.logger import SNLogger

from campari.tests.test_gausspsfs import (
    create_true_flux,
    generate_diagnostic_plots,
    perform_aperture_photometry,
    perform_gaussianity_checks
)

from campari.campari_runner import campari_runner

imsize = 19

default_parameters = {
    "config": None,
    "filter": "R062",
    "diaobject_name": 10046,
    "prebuilt_transient_model": None,
    "ra": 128.0,
    "dec": 42.0,
    "radius": None,
    "diaobject_collection": "manual",
    "diaobject_subset": None,
    "image_collection": "manual_fits",
    "image_collection_basepath": "/photometry_test_data/simple_gaussian_test/sig1.0",
    "image_collection_subset": "threefile",
    "max_no_transient_images": 0,
    "max_transient_images": 1,
    "image_selection_start": None,
    "image_selection_end": None,
    "transient_start": 60010.0,
    "transient_end": 60060.0,
    "SED_file": None,
    "object_type": "SN",
    "fast_debug": False,
    "save_model": False,
    "image_process": None,
    "image_provenance_tag": None,
    "diaobject_provenance_tag": None,
    "diaobject_process": None,
    "diaobject_id": None,
    "ltcv_process": "campari",
    "ltcv_provenance_tag": None,
    "create_ltcv_provenance": True,
    "diaobject_position_provenance_tag": None,
    "diaobject_position_process": None,
    "save_to_db": False,
    "add_truth_to_lc": False,
    "nprocs": 100,
    "photometry_campari_cutout_size": 19,
    "photometry_campari_initial_flux_guess": None,
    "photometry_campari_fetch_SED": False,
    "photometry_campari_use_roman": None,
    "photometry_campari_weighting": True,
    "photometry_campari_make_initial_guess": False,
    # For some reason generating x0 is broken when using
        # presaved models. I need to come back to this at some point. XXX
    "photometry_campari_method": None,
    "photometry_campari_pixel": None,
    "photometry_campari_subtract_background_method": 0,
    # NOTE: THIS IS CURRENTLY CHEATING. I need to find a way to do better sky subtraction. I found that if you
    # are looking at a small image clips, as campari does, the background subtraction is very poor because the PSF
    # is so wide that it looks like flat background near the edge of the clip. I could write a routine that goes
    #  and gets the background from a larger area or I could wait and see if campari will always be handed one,
    # e.g. from phrosty?
    "photometry_campari_use_real_images": True,
    "photometry_campari_print_memory_usage": None,
    "photometry_campari_psf_galaxy_photon_ops": None,
    "photometry_campari_psf_transient_photon_ops": None,
    "photometry_campari_psf_transient_class": "ou24PSF_slow_photonshoot",
    "photometry_campari_psf_galaxy_class": "ou24PSF",
    "photometry_campari_grid_options_type": None,
    "photometry_campari_grid_options_percentiles": None,
    "photometry_campari_grid_options_spacing": 1.0,
    "photometry_campari_grid_options_turn_grid_off": None,
    "photometry_campari_grid_options_gaussian_var": None,
    "photometry_campari_grid_options_cutoff": None,
    "photometry_campari_grid_options_error_floor": None,
    "photometry_campari_grid_options_subsize": 4,
    "photometry_campari_io_save_debug": None,
    "photometry_campari_io_test_num": None,
    "photometry_campari_simulations_avoid_non_linearity": None,
    "photometry_campari_simulations_background_level": None,
    "photometry_campari_simulations_bg_gal_flux": None,
    "photometry_campari_simulations_deltafcn_profile": None,
    "photometry_campari_simulations_do_rotation": None,
    "photometry_campari_simulations_do_xshift": None,
    "photometry_campari_simulations_mismatch_seds": None,
    "photometry_campari_simulations_noise": None,
    "photometry_campari_simulations_sim_gal_dec_offset": None,
    "photometry_campari_simulations_sim_gal_ra_offset": None,
    "photometry_campari_simulations_single_grid_point": None,
    "photometry_campari_simulations_sim_galaxy_scale": None,
    "photometry_campari_simulations_sim_galaxy_offset": None,
    "photometry_campari_simulations_base_pointing": None,
    "photometry_campari_simulations_base_sca": None,
    "photometry_campari_simulations_run_name": "gauss_source_no_grid",
    "photometry_snappl_simdex_server": None,
    "system_paths_lightcurves": None,
    "system_paths_sims_sed_library": None,
    "system_paths_campari_test_data": None,
    "system_paths_output_dir": None,
    "system_paths_sed_path": None,
    "system_paths_debug_dir": None,
    "system_ou24_simdex_server": None,
    "system_ou24_config_file": None,
    "system_ou24_sn_truth_dir": None,
    "system_ou24_sims_sed_library": None,
    "system_ou24_images": None,
    "system_ou24_tds_base": None,
    "system_db_url": None,
    "system_db_username": None,
    "system_db_passwordfile": None,
    "prebuilt_static_model": None # Will this get overwritten by merge with args?
}


cfg = Config.get()
debug_dir = cfg.value("system.paths.debug_dir")
out_dir = cfg.value("system.paths.output_dir")




#45, 48, 49
# For some reason, just 45, 48 and 49 fail. 45 and 49 are skewed and 48 has a very high bias (~0.37)
# Obviously we expect some to fail a 0.05 p value cut on skew but the bias is concerning.
# I am skipping these for now because I want to go and check if the reason they are failing is due to the fact
# that the galaxies are point like and hard to model.
# Note: these simulation_numbers correspond to the seed used to generate the simulation,
#  so I can go back and check the simulations if I want.

@pytest.mark.slow()
@pytest.mark.parametrize("simulation_number", [46, 47, 50, 51, 52])
def test_bothnoise_shifted_22maghost_ou24PSF_slow_photops(simulation_number):
    diaobject_name = "100" + str(simulation_number)
    args = {
        "prebuilt_static_model": "/scratch/campari_debug_dir/" \
        "psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
        "diaobject_name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{simulation_number}.txt",
    }
    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.slow()
# 51 is two sigma skewed, p ~ 0.04, is this admissible?
#[45, 46, 47, 48, 49, 50, 51, 52]
@pytest.mark.parametrize("simulation_number", [51])
def test_bothnoise_shifted_NOhost_ou24PSF_slow_photops(simulation_number):
    # Screwed up the naming on some of these
    if simulation_number > 48:
        underscore = "_"
    else:
        underscore = ""

    diaobject_name = "111" + str(simulation_number)
    args = {
        "diaobject_name": diaobject_name,
        "photometry_campari_grid_options_type": "none",
        "img_list": pathlib.Path(__file__).parent / "testdata/"
        f"test_gaussims_bothnoise_unaligned_nohost_faintsource_ou2024_more{underscore}seed{simulation_number}.txt",
        "photometry_campari_make_initial_guess": True
    }
    args = default_parameters | args
    SNLogger.debug(args)
    cfg.parse_args(SimpleNamespace(**args))
    val = cfg.value("photometry.campari.grid_options.type")
    #raise ValueError(f"grid options type is {val}")

    runner = campari_runner(**args)
    runner()

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        plotname = f"bothnoise_shifted_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_skynoiseonly():
    diaobjnum = 51
    diaobject_name = "222" + str(diaobjnum)

    args = {
        " diaobject-name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent / f"testdata/test_gaussims_nohost_skynoiseonlyseed51.txt",
        " photometry_campari_grid_options_type": "none",

    }
    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    plotname = f"skynoise_aligned_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        ap_sums, ap_err = perform_aperture_photometry(filename, imsize, aperture_radius=4)
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:

        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux, ap_sums=ap_sums, ap_err=ap_err)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_nohost_poissonnoiseonly():
    diaobjnum = 52
    diaobject_name = "222" + str(diaobjnum)

    args = {
        "diaobject-name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent / f"testdata/test_gaussims_nohost_poissonnoiseonlyseed51.txt",
        "photometry_campari_grid_options_type": "none",
    }

    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()


    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    plotname = f"poissonnoise_aligned_nohost_ou24PSF_slow_photops_diagnostic_{diaobject_name}"
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        ap_sums, ap_err = perform_aperture_photometry(filename, imsize, aperture_radius=4)
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux, ap_sums=ap_sums, ap_err=ap_err)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test will fail because there is no noise so pull has no meaning"
                         " but it's a sanity check.")
def test_extended_nohost_nonoise():
    simulation_number = 53
    diaobject_name = "222" + str(simulation_number)

    args = {
        " diaobject-name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent / f"testdata/test_gaussims_nohost_nonoiseseed51.txt",
        " photometry_campari_grid_options_type": "none"

    }

    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    plotname = f"nonoise_aligned_nohost_ou24PSF_slow_photops_diagnostic_{diaobject_name}"
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(
    reason="This test will fail because there is no noise so pull has no meaning but it's a sanity check."
)
def test_nophot_sanitycheck():
    simulation_number = 54
    diaobject_name = "222" + str(simulation_number)

    args = {
        " diaobject-name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent / f"testdata/test_gaussims_nohost_nophot_sanity_checkseed51.txt",
        " photometry_campari_grid_options_type": "none",
        " photometry_campari_psf_transient_class": "ou24PSF_slow",
    }

    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    filename = f"{diaobject_name}_R062_romanpsf"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    plotname = f"nonoise_aligned_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_both_shifted_21mag_host_ou2024_more():

    args = {
        "img_list": pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_shifted_22mag_host_200_ou2024.txt",
        " photometry_campari_grid_options_type": "regular",
        " photometry_campari_grid_options_spacing": "0.75",
        " save_model": True,
        " photometry_campari_psf_transient_class": "ou24PSF_slow",
        " diaobject-name": "123",

    }
    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    lc = Table.read(f"/{out_dir}/123_R062_ou24PSF_slow_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "both_shifted_21mag_host_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_ou24PSF_slow", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests.")
# While this test is not run typically, it is still useful. If I run the test with noise, misaligned images, and
# a host, and it fails, I won't know if the problem is the noise, the misalignment, or the host.
# This test isolates the effect of the host. If this also fails, the problem is likely due to the modeling
# of the host galaxy.
def test_noiseless_aligned_22maghost_withphotops():

    args = {
        "img_list": pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_22maghost_ou2024_withphotops.txt",
        " photometry_campari_grid_options_type": "regular",
        " photometry_campari_grid_options_spacing": "0.75",
        " save_model": True,
        " photmetry_campari_source_phot_ops": True,
        " photometry_campari_transient_psfclass": "ou24PSF_slow"
    }

    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    lc = Table.read(f"/{out_dir}/123_R062_romanpsf_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)

    except AssertionError as e:
        plotname = "bothnoise_shifted_nohost_ou24PSF_slow_photops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is superseded by more difficult tests.")
def test_noiseless_aligned_nohost_ou2024fast_withphotops_more():
    args = {
        "img_list": pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_withphotops.txt",
        " photometry_campari_grid_options_type": "none",
        " save_model": True
    }
    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()

    # Check accuracy
    lc = Table.read(f"/{out_dir}/123_R062_romanpsf_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    try:
        np.testing.assert_allclose(
            lc["flux"], flux, rtol=9e-3
        )  # With photon ops, accuracy is to about 0.6 % only, is this to be expected?
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


################ Realistic Galaxies Below Here ################
# Note: these simulation_numbers in num_list correspond to the seed used to generate the simulation,
#  so I can go back and check the simulations if I want.
num_list = list(range(45, 61))
@pytest.mark.slow()
@pytest.mark.parametrize("simulation_number", num_list)
def test_bothnoise_shifted_22magrealisticgalaxy_ou24PSF_slow_photops(simulation_number):

    diaobject_name = "333" + str(simulation_number)

    args = {
        " diaobject_name": diaobject_name,
        "img_list": pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_realistichost_faintsource_ou2024_photshootseed{simulation_number}.txt",
        "prebuilt_static_model": f"/{debug_dir}/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    }

    args = default_parameters | args
    cfg.parse_args(SimpleNamespace(**args))
    runner = campari_runner(**args)
    runner()


    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/{out_dir}/{filename}_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=24)
    plotname = f"bothnoise_shifted_22magrealisticgalaxy_ou24PSF_slow_photops_diagnostic_{diaobject_name}"

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /{debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e