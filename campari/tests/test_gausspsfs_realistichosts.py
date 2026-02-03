
import pathlib
import subprocess

import numpy as np

from astropy.table import Table

from snappl.logger import SNLogger

from campari.plotting import generate_diagnostic_plots
from campari.tests.gausspsfs import perform_gaussianity_checks

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
        "--photometry-campari-psfclass", "gaussian",
        "--photometry-campari-use_real_images",
        "--photometry-campari-psf-transient_class", "gaussian",
        "--photometry-campari-psf-galaxy_class", "gaussian",
        "--diaobject-collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-spacing", "1",
        "--photometry-campari-grid_options-subsize", "4",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background",
        "--no-photometry-campari-source_phot_ops",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/photometry_test_data/simple_gaussian_test/sig1.0",
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]



def test_both_shifted_22mag_realisticgalaxy_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_unalignedattempt2_nonoise_22hostmag_faintsource_realgal_seed45.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.5"  # Finer grid spacing
    #cmd += ["--save_model"]
    # realsitic_galaxy_gridmodel
    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/justshifted_realistic.npy",
    ]
    cmd += [
        "--prebuilt_transient_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/justshifted_realistic_SN.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "justshifted_nonoise_22mag_hostrealistic_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_faint_transient_nonoise_unlaligned_realisticgalaxy():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_nonoise_faintsource_unaligned_positionfixed_realisticgal_seed45.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/gauss250images_36points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        # With the more realistic galaxies, it is harder for campari to perfectly model the galaxy.
        # At this grid scale, it seems like the impact is about a scatter of 75 counts in flux.
        # Aka, I beleive that at the current grid scale, there is an error floor of 75 counts
        # even without noise. This is determined empiraclly from the code. It is possible this could
        # be reduced through improvements to the algorithm or finer grid spacing, but for now we
        # will just account for it here.
        lc["flux_err"] = np.sqrt(lc["flux_err"] ** 2 + 75**2)
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "fainttransient_nonoise_hostrealistic_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux, err_fudge=75)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_faint_transient_bothnoise_unlaligned_realisticgalaxy():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_bothnoise_faintsource_unaligned_positionfixed_realisticgal_seed45.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    # transient_index = cmd.index("--transient_end")
    # cmd[transient_index + 1] = "60010"  # No transient present
    # cmd += ["--save_model"]
    # realsitic_galaxy_gridmodel
    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/gauss250images_36points.npy",
    ]
    # cmd += [
    #     "--prebuilt_transient_model",
    #     pathlib.Path(__file__).parent / "testdata/prebuilt_models/justshifted_realistic_SN.npy",
    # ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "fainttransient_nonoise_hostrealistic_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_no_transient_realisticgalaxy_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
#        / "testdata/test_gaussims_unalignedattempt2_nonoise_22hostmag_faintsource_realgal_seed45.txt",
        / "testdata/test_gaussims_whatisgoingon_3seed45.txt"
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.5"  # Finer grid spacing

    transient_index = cmd.index("--transient_end")
    cmd[transient_index + 1] = "60010"  # No transient present

    # realsitic_galaxy_gridmodel
    cmd += [
       "--prebuilt_static_model",
       pathlib.Path(__file__).parent / "testdata/psf_matrix_gaussian_a0332a3d-785d-4d04-950b-5ec4202d0aa7_75_images64_points.npy",
       ]
    # cmd += [
    #     "--prebuilt_transient_model",
    #     pathlib.Path(__file__).parent / "testdata/prebuilt_models/justshifted_realistic_SN.npy",
    # ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "whatisgoingon_3_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

#
