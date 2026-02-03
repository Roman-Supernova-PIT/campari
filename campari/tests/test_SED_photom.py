import pathlib
import subprocess

import numpy as np
import pytest

from astropy.table import Table

from snappl.logger import SNLogger

from campari.tests.test_gausspsfs import generate_diagnostic_plots, perform_gaussianity_checks



imsize = 19
base_cmd = [
        "python", "../RomanASP.py",
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
        "--photometry-campari-subtract_background_method", "calculate", # This cheats and sets the background to zero
        # because these images don't have this column. This is set as default so that the tests that are trying to
        # hit machine precision can pass. For more realistic tests, we should use "calculate".
#        "--no-photometry-campari-source_phot_ops",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/photometry_test_data/simple_gaussian_test/sig1.0",
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]


def test_nohost_bothnoise_HsiaoTemplate():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "123",]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        generate_diagnostic_plots("123_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_bothnoise_FlatSED():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_flat_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "124"]

    cmd.append("--photometry-campari-psf-transient_class")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "flatSED_diagnostic_comparison"
        generate_diagnostic_plots("124_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e



def test_nohost_bothnoise_deltaSED():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_deltafunc_SEDseed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "125"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/125_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "deltaSED_diagnostic_comparison"
        generate_diagnostic_plots("125_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_bothnoise_deltaSED_file():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_delta_sed_file_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "125"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/125_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "deltaSED_diagnostic_comparison"
        generate_diagnostic_plots("125_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_bothnoise_HsiaoSEDsimulated_BBSEDfit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "129"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "test_bb_fit_sed.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/129_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "BBSED_diagnostic_comparison"
        generate_diagnostic_plots("129_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_bothnoise_HsiaoSEDsimulated_Hsiaofit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "130"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "snflux_1a_peakmjd.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/130_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "HsiaoRecoverySED_diagnostic_comparison"
        generate_diagnostic_plots("130_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_nonoise_HsiaoSEDsimulated_Hsiaofit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_nonoise_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "131"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "snflux_1a_peakmjd.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/131_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "HsiaoRecoverySED_nonoise_diagnostic_comparison"
        generate_diagnostic_plots("131_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

def test_nohost_bothnoise_HsiaoSEDsimulated_improvedBBSEDfit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "129"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "test_bb_sed_improved.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/129_R062_ou24psf_slow_photonshoot_lc.ecsv")

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
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "BBSED_diagnostic_comparison_improved"
        generate_diagnostic_plots("129_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e