import inspect
import pathlib
import pytest
import subprocess

import numpy as np

from astropy.table import Table
from astropy.io import fits

from snappl.logger import SNLogger

from campari.plotting import generate_diagnostic_plots
from campari.tests.test_gausspsfs import perform_gaussianity_checks, create_true_flux
from campari.tests.test_ou24psfs import run_test_and_check_against_truth_flux_using_pull_distribution

from snappl.config import Config


imsize = 19
base_cmd = [
        "python", "../RomanASP.py",
        "--diaobject-name", "123",
        "-t", "1",
        "-n", "0",
        "-f", "R062",
        "--ra", "128.0",
        "--dec", "42.0",
        "--transient_start", "60010",
        "--transient_end", "60060",
        "--photometry-campari-use_real_images",
        "--photometry-campari-psf-transient_class", "gaussian",
        "--photometry-campari-psf-galaxy_class", "gaussian",
        "--diaobject-collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-spacing", "1",
        "--photometry-campari-grid_options-subsize", "4",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background_method", "calculate",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/scratch/",
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]


cfg = Config.get()
debug_dir = cfg.value("system.paths.debug_dir")
out_dir = cfg.value("system.paths.output_dir")


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
    "photometry_campari_grid_options_spacing": 0.75,
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
    "prebuilt_static_model": None,  # Will this get overwritten by merge with args?
}


# Right now this is failing due to a skew. I believe this is
# due to some bias resulting from the low SNR + Poisson noise when doing PSF fitting. More work is needed, come
# back to this! XXX XXX XXX TODO XXX
#@pytest.mark.skip(reason="This test fails due to skew. I need to"
#" figure out if this is due to the fact that it is noiseless.")
def test_faint_transient_nonoise_unlaligned_realisticgalaxy():
    args = {
        "img_list": pathlib.Path(__file__).parent
        / "testdata/test_imagelists/test_gaussims_nonoise_faintsource_unaligned_positionfixed_realisticgal_seed45.txt",
        "photometry_campari_grid_options_type": "regular",
        "photometry_campari_grid_options_spacing": 0.75,
        "prebuilt_static_model": pathlib.Path(__file__).parent / "testdata/prebuilt_models/gauss250images_36points.npy",
    }

    run_test_and_check_against_truth_flux_using_pull_distribution(args)

    # result = subprocess.run(cmd, capture_output=False, text=True)

    # if result.returncode != 0:
    #     raise RuntimeError(
    #         f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    #     )

    # # Check accuracy
    # lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    # flux = create_true_flux(lc["mjd"], peakmag=24)

    # try:
    #     # With the more realistic galaxies, it is harder for campari to perfectly model the galaxy.
    #     # At this grid scale, it seems like the impact is about a scatter of 75 counts in flux.
    #     # Aka, I beleive that at the current grid scale, there is an error floor of 75 counts
    #     # even without noise. This is determined empiraclly from the code. It is possible this could
    #     # be reduced through improvements to the algorithm or finer grid spacing, but for now we
    #     # will just account for it here.
    #     lc["flux_err"] = np.sqrt(lc["flux_err"] ** 2 + 75**2)
    #     residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
    #     perform_gaussianity_checks(residuals_sigma)
    # except AssertionError as e:
    #     plotname = "fainttransient_nonoise_hostrealistic_diagnostic"
    #     generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux, err_fudge=75)

    #     SNLogger.debug(e)
    #     raise e


generate_simulations = False
from campari.image_simulator_run import run_sim


@pytest.mark.self_generating()
def test_faint_transient_bothnoise_unlaligned_realisticgalaxy():
    func_name = inspect.currentframe().f_code.co_name
    test_data_path = pathlib.Path(__file__).parent / "testdata"
    seed = 45
    run_name = func_name + f"seed{seed}"
    if generate_simulations:
        SNLogger.debug(f"Generating new simulations for {func_name}. This may take a while.")
        # The image data does not currently exist, so we will create it.
        run_sim(
            seed=seed, # Set seed for reproducibility, this is the seed that Cole started with.
            images_aligned=False,
            poisson_noise=True,
            sky_noise=True,
            static_source="galaxy",
            static_source_mag=22,
            transient_peak_mag=24,
            mjd=np.arange(60000, 60075, 0.3),
            psf_class="gaussian",
            run_dir=func_name,
            output_path=test_data_path,
            run_name_base=func_name,
            bulge_R=2,
            bulge_n=3,
            disk_R=4,
            disk_n=1,  # Simulated Galaxy Params
            test_data_path=test_data_path
        )

    # Perform sanity check against cached image.
    cached_image = fits.open(f"{test_data_path}/test_faint_transient_bothnoise_unlaligned_realisticgalaxy/"
        "test_faint_transient_bothnoise_unlaligned_realisticgalaxyseed45/test_faint_transient_bothnoise"
        "_unlaligned_realisticgalaxyseed45_sanity_image.fits")
    new_image = fits.open(
        f"{test_data_path}/test_faint_transient_bothnoise_unlaligned_realisticgalaxy/"
        "test_faint_transient_bothnoise_unlaligned_realisticgalaxyseed45/"
        "test_faint_transient_bothnoise_unlaligned_realisticgalaxyseed45_60000.0_image.fits"
    )

    np.testing.assert_allclose(cached_image[0].data, new_image[0].data, rtol=1e-5, err_msg="The newly generated image"
        " does not match the cached image. This suggests that there may be a problem with the simulation code or that"
        " the simulation parameters have changed. Please investigate this issue before proceeding, as it may impact the"
        " validity of the test results.")

    # Check if the image list exists at the expected location. If not, raise an error.
    imagelist_filename = test_data_path / f"image_list_{run_name}.txt"

    imagelist_filename = pathlib.Path(__file__).parent / "testdata/test_imagelists/test_gaussims_bothnoise_faintsource_unaligned_positionfixed_realisticgal_seed45.txt"
    if not pathlib.Path(imagelist_filename).exists():
        raise FileNotFoundError(f"Expected image list at {imagelist_filename} not found. Sim may have"
        " failed to run.")
    args = {
        "img_list": pathlib.Path(__file__).parent / imagelist_filename,
        "photometry_campari_grid_options_type": "regular",
        "photometry_campari_grid_options_spacing": 0.75,
        "prebuilt_static_model": pathlib.Path(__file__).parent / "testdata/prebuilt_models/gauss250images_36points.npy",
        #"save_model": True,
    }

    run_test_and_check_against_truth_flux_using_pull_distribution(args)



    # result = subprocess.run(cmd, capture_output=False, text=True)

    # if result.returncode != 0:
    #     raise RuntimeError(
    #         f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    #     )

    # # Check accuracy
    # lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    # flux = create_true_flux(lc["mjd"], peakmag=24)

    # try:
    #     residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
    #     perform_gaussianity_checks(residuals_sigma)
    # except AssertionError as e:
    #     plotname = "fainttransient_nonoise_hostrealistic_diagnostic"
    #     generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
    #     SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
    #     SNLogger.debug(e)
    #     raise e
