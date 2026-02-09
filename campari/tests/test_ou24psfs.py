import pathlib
import subprocess

import numpy as np
import pytest

from astropy.table import Table

from snappl.logger import SNLogger

from campari.tests.test_gausspsfs import generate_diagnostic_plots, perform_gaussianity_checks

import pytest


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
        "--photometry-campari-psf-transient_class", "ou24PSF_slow_photonshoot",
        "--photometry-campari-psf-galaxy_class", "ou24PSF",
        "--photometry-campari-use_real_images",
        "--diaobject-collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-spacing", "1",
        "--photometry-campari-grid_options-subsize", "4",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        # This is bleed over from another pull, this file isn't tracked yet
#        "--photometry-campari-subtract_background_method", "SKY_MEAN", # This cheats and sets the background to zero
        # because these images don't have this column. This is set as default so that the tests that are trying to
        # hit machine precision can pass. For more realistic tests, we should use "calculate".
#        "--no-photometry-campari-source_phot_ops",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/photometry_test_data/simple_gaussian_test/sig1.0",
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]



def test_noiseless_aligned_nohost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        # pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_nophotops.txt",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_aligned_noiseless_nohost_ou24PSF_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(
            lc["flux"],
            flux,
            rtol=5e-6,  # Why does this need to be ~50x higher than the gaussian version?
        )
        SNLogger.debug(lc["flux_err"])
        np.testing.assert_allclose(lc["flux_err"], 2.311065590128104, atol=1e-7)  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        #generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_bothnoise_aligned_nohost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_aligned_bothnoise_nohost_ou24PSF_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"


    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

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
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_bothnoise_shifted_nohost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_bothnoise_nohost_ou24PSF_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

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
        plotname = "bothnoise_shifte_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_noiseless_aligned_22maghost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_bothnoise_nohost_ou24PSF_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

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
        plotname = "bothnoise_shifte_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

def test_bothnoise_aligned_22maghost_ou24PSF_slow_photops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed45.txt",
    ]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    #cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        #"/campari_debug_dir/psf_matrix_ou24PSF_f0d255a4-5744-487c-8061-94e5ccf154ee_75_images207_points.npy",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = "123_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = "bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic"

        perform_gaussianity_checks(residuals_sigma)
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = "bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

def test_extended_seed_46():

    seed = 46
    diaobject_name =  "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_47():
    seed = 47
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_48():
    seed = 48
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_49():
    seed = 49
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        #generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_50():
    seed = 50
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        #generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_51():
    seed = 51
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        #generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_seed_52():
    seed = 52
    diaobject_name = "100" + str(seed)
    diaobject_name_index = base_cmd.index("--diaobject-name") + 1
    base_cmd[diaobject_name_index] = diaobject_name

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / f"testdata/test_gaussims_bothnoise_unaligned_withhost_faintsource_ou2024_more_seed{seed}.txt",
    ]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
        plotname = f"bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

        perform_gaussianity_checks(residuals_sigma)
        #generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = f"bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_extended_nohost():
    for seed in [49, 50, 51, 52]:
        diaobject_name = "111" + str(seed)
        diaobject_name_index = base_cmd.index("--diaobject-name") + 1
        base_cmd[diaobject_name_index] = diaobject_name

        cmd = base_cmd + [
            "--img_list",
            pathlib.Path(__file__).parent
            / f"testdata/test_gaussims_bothnoise_unaligned_nohost_faintsource_ou2024_more_seed{seed}.txt",
        ]
        cmd += ["--photometry-campari-grid_options-type", "none"]
        # cmd += [
        #     "--prebuilt_static_model",
        #     "/campari_debug_dir/psf_matrix_ou24PSF_d2605d96-d155-4aa0-9d65-445d1b869dfb_150_images204_points.npy",
        # ]
        cmd += ["--nprocs", "15"]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        # Check accuracy
        filename = f"{diaobject_name}_R062_ou24psf_slow_photonshoot"
        lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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
            plotname = f"bothnoise_shifted_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"

            #perform_gaussianity_checks(residuals_sigma)
            # generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        except AssertionError as e:
            plotname = f"bothnoise_aligned_nohost_ou24PSF_slow_nophotops_diagnostic_{diaobject_name}"
            generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
            SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
            SNLogger.debug(e)
            raise e

def test_just_plot():
    filename = "123_R062_ou24psf_slow_photonshoot"
    lc = Table.read(f"/campari_out_dir/{filename}_lc.ecsv")

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

        plotname = "bothnoise_shifte_22maghost_ou24PSF_slow_nophotops_diagnostic"

        #perform_gaussianity_checks(residuals_sigma)
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
    except AssertionError as e:
        plotname = "bothnoise_aligned_22maghost_ou24PSF_slow_nophotops_diagnostic"
        generate_diagnostic_plots(filename, imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)

#### Tests below here I found in the gauss psf test file. Some may be repeats, sort through this.
@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_both_shifted_21mag_host_ou2024_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_shifted_22mag_host_200_ou2024.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_ou24PSF_slow_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "both_shifted_21mag_host_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_ou24PSF_slow", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


# ########### Tests with OU2024 PSF ##############################################################


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_noiseless_aligned_nohost_ou2024_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_nophotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(lc["flux"], flux, rtol=6e-6)
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_noiseless_aligned_nohost_ou2024_withphotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_withphotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(lc["flux"], flux, rtol=9e-3)  # With photon ops, accuracy is to about 0.6 % only,
        # is this to be expected?
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_bothnoise_aligned_nohost_ou2024_withphotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_aligned_nohost_ou2024_withphotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "bothnoise_aligned_nohost_ou24PSF_slow_photops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_bothnoise_shifted_nohost_ou2024_withphotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_shifted_nohost_ou2024_withphotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)

    except AssertionError as e:
        plotname = "bothnoise_shifted_nohost_ou24PSF_slow_photops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_noiseless_aligned_22maghost_withphotops():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_22maghost_ou2024_withphotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF_slow"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)

    except AssertionError as e:
        plotname = "bothnoise_shifted_nohost_ou24PSF_slow_photops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_noiseless_aligned_nohost_ou2024fast_withphotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_withphotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(
            lc["flux"], flux, rtol=9e-3
        )  # With photon ops, accuracy is to about 0.6 % only, is this to be expected?
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_noiseless_aligned_nohost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_nohost_ou2024_nophotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(
            lc["flux"],
            flux,
            rtol=5e-6,  # Why does this need to be ~50x higher than the gaussian version?
        )
        SNLogger.debug("flux", lc["flux"])
        np.testing.assert_allclose(lc["flux_err"], 2.3313916, atol=1e-7)  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "noiseless_aligned_nohost_ou24PSF_slow_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_bothnoise_aligned_nohost_ou2024fast_nophotops_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_aligned_nohost_ou2024_fast_nophotops.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    # spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    # cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "/campari_debug_dir/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "10"]
    # phot_ops_index = cmd.index("--no-photometry-campari-source_phot_ops")
    # cmd[phot_ops_index] = "--photometry-campari-source_phot_ops"

    psfclass_index = cmd.index("--photometry-campari-psfclass")
    cmd[psfclass_index + 1] = "ou24PSF"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_romanpsf_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "bothnoise_aligned_nohost_ou24PSF_nophotops_diagnostic"
        generate_diagnostic_plots("123_R062_romanpsf", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e