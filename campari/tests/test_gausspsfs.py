import inspect
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import pytest
import subprocess

from scipy.stats import norm, skewtest, skew

from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry

from snappl.config import Config
from snappl.logger import SNLogger

SNLogger.set_level("DEBUG")

from campari.plotting import generate_diagnostic_plots

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
        "--photometry-campari-psf-transient_class", "gaussian",
        "--photometry-campari-psf-galaxy_class", "gaussian",
        "--photometry-campari-use_real_images",
        "--diaobject-collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-spacing", "1",
        "--photometry-campari-grid_options-subsize", "4",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background", "calculate",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/scratch/", # update in
        # SNPIT environment necessitated this change.
        # but you may need to run this code:
        # ln -s /scratch/photometry_test_data /photometry_test_data
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]

cfg = Config.get()
debug_dir = cfg.value("system.paths.debug_dir")
out_dir = cfg.value("system.paths.output_dir")

def create_true_flux(mjd, peakmag):
    # This creates a linear up-down lightcurve peaking at peakmag. Looks like a triangle.
    peakflux = 10 ** ((peakmag - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros_like(mjd)
    before_peak = np.where(mjd < peak_mjd)
    after_peak = np.where(mjd >= peak_mjd)
    flux[before_peak] = peakflux * (mjd[before_peak] - start_mjd) / (peak_mjd - start_mjd)
    flux[after_peak] = peakflux * (mjd[after_peak] - end_mjd) / (peak_mjd - end_mjd)
    return flux


def perform_aperture_photometry(fileroot, imsize, aperture_radius=4):
    noise_maps = np.load(f"{debug_dir}/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"{debug_dir}/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    lc = Table.read(f"{out_dir}/{fileroot}_lc.ecsv")

    ap_sums = []
    ap_err = []
    lc_index = 0
    for i in range(ims.shape[0]):
        if i < lc.meta["pre_transient_images"] + lc.meta["post_transient_images"]:
            # Skipping non detect images!
            continue
        im = ims[i]
        positions = [lc["x_cutout"][lc_index], lc["y_cutout"][lc_index]]
        aperture = CircularAperture(positions, r=4)
        phot_table = aperture_photometry(im.reshape(imsize, imsize), aperture, error=noise_maps[i])
        ap_sums.append(phot_table["aperture_sum"][0])
        ap_err.append(phot_table["aperture_sum_err"][0])
        lc_index += 1

    ap_sums = np.array(ap_sums)
    ap_err = np.array(ap_err)

    return ap_sums, ap_err


def perform_gaussianity_checks(residuals_sigma, measuredflux=None, trueflux=None):
    """Most of these tests apply the same checks, so just put them in a function."""
    sub_one_sigma = np.sum(np.abs(residuals_sigma) < 1)
    SNLogger.debug(f"Campari fraction within 1 sigma: {sub_one_sigma / len(residuals_sigma)}")
    np.testing.assert_allclose(sub_one_sigma / len(residuals_sigma), 0.68, atol=0.2, err_msg="Errors non gaussian!?")

    mu, sig = norm.fit(residuals_sigma)
    # If I am not mistaken, this is equivalent to checking that the residuals are unbiased at 3 sigma confidence.
    mu_atol = 3 / np.sqrt(len(residuals_sigma))
    SNLogger.debug("Function call stack for Gaussianity Checks:")
    SNLogger.info(inspect.stack()[0][3])
    SNLogger.info(inspect.stack()[1][3])
    SNLogger.info(f"NUM POINTS: {len(residuals_sigma)}")
    if measuredflux is not None and trueflux is not None:
        delta_mag = (measuredflux/trueflux - 1) * 1000  # in mmag
        delta_mag = delta_mag[np.isfinite(delta_mag)]
        SNLogger.info(f"Campari delta mag (mmag) mean: {np.nanmean(delta_mag)}, std: {np.nanstd(delta_mag)}")
    SNLogger.info("Fitted residuals mu: " + str(mu) + ", sig: " + str(sig) + "\n mu tolerance: " + str(mu_atol))
    np.testing.assert_allclose(mu, 0, atol=mu_atol, err_msg="Residuals biased!")
    np.testing.assert_allclose(sig, 1, atol=0.2, err_msg="Residuals too broad or narrow!")
    # Check to make sure our distribution is not skewed.
    p_value = skewtest(residuals_sigma).pvalue
    SNLogger.info("Skewness p_value: " + str(p_value))
    np.testing.assert_array_less(0.05, p_value, err_msg="Residuals skewed!")


# TESTS BEGIN HERE #########################################################

def test_noiseless_aligned_no_host():
    # Test 1. Noiseless, perfectly aligned images. No host galaxy, We expect campari to do extremely well here.
    # Get the images we need
    # Can we define this in the config?

    cmd = base_cmd + ["--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_1.txt"]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    np.testing.assert_allclose(lc["flux"], flux, atol=1e-7)

    # The 3.691204 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    np.testing.assert_allclose(lc["flux_err"], 3.691204, atol=1e-7)

#@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_poisson_noise_aligned_no_host():
    # Now we add just poisson noise. This will introduce scatter, but campari should still agree with aperture
    # photometry.

    cmd = base_cmd + [
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_aligned_nohost_200.txt"
        ]

    # For this test I bumped it up to 60 images because the small number statistics caused it to not pass gaussianity
    # checks. Thankfully 60 images run in about ~1 min without a bg static model. This will be a bit of a pain for
    # tests with static models unfortunately.

    cmd += ["--photometry-campari-grid_options-type", "none"]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    np.testing.assert_allclose(lc["flux"], ap_sums, rtol=3e-3)

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)

    except AssertionError as e:
        plotname = "poisson_aligned_nohost_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0 --no-star-noise -b junkskynoise
# --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055
# 60060 --image-centers 128 42 -θ 0 -r 30 -s 0 --transient-ra 128 --transient-dec 42
# --no-star-noise --no-transient-noise -n 1

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_sky_noise_aligned_no_host():
    # Now we add just sky noise. This will introduce scatter, but campari should still agree with aperture
    # photometry.

    # cmd = base_cmd + [
    #     "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_sky.txt",
    # ]
    cmd = base_cmd + ["--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_sky_no_host_more.txt"]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    SNLogger.debug("Loaded this lc: " + str(lc))
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    flux = create_true_flux(lc["mjd"], peakmag=21)
    flux = create_true_flux(lc["mjd"], peakmag=21)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)

    except AssertionError as e:
        plotname = "sky_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0 --no-star-noise -b bothnoise --width 256
# --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055 60060
#  --image-centers 128 42 -θ 0 -r 30 -s 0 --transient-ra 128 --transient-dec 42 --no-star-noise -n 1


def test_both_noise_aligned_no_host():
    # Now we add sky and poisson noise.

    imsize = 19
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_more.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    flux = create_true_flux(lc["mjd"], peakmag=21)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "both_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0 --no-star-noise -b shifted_noiseless
#  --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055
# 60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999   41.999
#  128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999   42.0005 128.001
#   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330. 360. -r 0 -s 0
# --transient-ra 128 --transient-dec 42 --no-star-noise --no-transient-noise -n 1

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_noiseless_shifted_no_host():

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_noiseless.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    np.testing.assert_allclose(lc["flux"], flux, rtol=1e-7, atol=0)

    # The 3.691204 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    # We use a looser tolerance here because of numerical issues with the shifts.
    np.testing.assert_allclose(lc["flux_err"], 3.691204, rtol=1e-4)

# 'python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0  --no-star-noise -b shifted_poisson
# --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055
#  60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999   41.999
#   128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999   42.0005
# 128.001   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330. 360. -r 0 -s 0
# --transient-ra 128 --transient-dec 42 --no-star-noise  -n 1'

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_poisson_shifted_no_host():

    cmd = base_cmd + [
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_poisson_more.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    np.testing.assert_allclose(lc["flux"], ap_sums, rtol=3e-3)
    # np.testing.assert_allclose(lc["flux_err"], ap_err, rtol=3e-3)

    flux = create_true_flux(lc["mjd"], peakmag=21)
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "poisson_shifted_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0  --no-star-noise -b shifted_sky
#  --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050
#  60055 60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999
#  41.999  128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999
#   42.0005 128.001   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330.
#  360. -r 30 -s 0 --transient-ra 128 --transient-dec 42 --no-star-noise  -n 1 --no-transient-noise

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_sky_shifted_no_host():
    # Now we add just sky noise.

    imsize = 19
    cmd = base_cmd + [
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_sky_more.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "shifted_sky_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0  --no-star-noise -b shifted_both
# --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050
# 60055 60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999
# 41.999  128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999   42.0005
#  128.001   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330. 360. -r 30 -s 0
# --transient-ra 128 --transient-dec 42 --no-star-noise  -n 1'


def test_both_shifted_no_host():

    cmd = base_cmd + [
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_shifted_both_more.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    flux = create_true_flux(lc["mjd"], peakmag=21)
    SNLogger.debug(f"Expected fluxes: {flux}")

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)

    except AssertionError as e:
        plotname = "both_noise_shifted_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


# Just background tests
@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_aligned_noiseless_just_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_aligned_host_only.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]

    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    plotname = "noiseless_aligned_just_host_diagnostic"
    generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=None)
    SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")

    fileroot = "123_R062_gaussian"
    ims = np.load(f"{debug_dir}/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"{debug_dir}/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    np.testing.assert_allclose(residuals, 0, atol=20)

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_both_aligned_just_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_aligned_host_only.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]

    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    # cmd += ['--save_model']
    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent
        / "testdata/prebuilt_models/psf_matrix_gaussian_123_10_images36_points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    fileroot = "123_R062_gaussian"
    noise_maps = np.load(f"{debug_dir}/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"{debug_dir}/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"{debug_dir}/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    pixel_pulls = residuals / noise_maps
    try:
        perform_gaussianity_checks(pixel_pulls.flatten())
    except AssertionError as e:
        plotname = "both_aligned_just_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=None)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_both_shifted_just_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_shifted_host_only.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]

    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent
        / "testdata/prebuilt_models/psf_matrix_gaussian_123_10_images_rotated_36_points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    fileroot = "123_R062_gaussian"
    noise_maps = np.load(f"{debug_dir}/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"{debug_dir}/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"{debug_dir}/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    pixel_pulls = residuals / noise_maps
    try:
        perform_gaussianity_checks(pixel_pulls.flatten())
    except AssertionError as e:
        plotname = "both_shifted_just_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=None)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# 22 mag delta function galaxy tests ############################################################################

# Note: These tests are maintained rather than full removal because they are useful for debugging when
# the more complicated tests fail. Since they take a few minutes to run each, we skip them in normal test runs.
@pytest.mark.skip(reason="this test is subsumed by following tests")
def test_noiseless_aligned_22mag_host():

    cmd = base_cmd + ["--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_host_mag22.txt"]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    cmd += ["--prebuilt_static_model", pathlib.Path(__file__).parent /
            "testdata/prebuilt_models/psf_matrix_gaussian_123_13_images.npy"]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        np.testing.assert_allclose(lc["flux"], flux, atol=1e-7)

        # The error is small, but higher than that in the no host case. I believe this is due to extra uncertainty
        # thanks to more free parameters in the fit.
        np.testing.assert_allclose(lc["flux_err"], 4.520784, atol=1e-7)

    except AssertionError as e:
        plotname = "noiseless_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_poisson_aligned_22mag_host():
    # I think is failing because there are so few no transient compared to with transient images.

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_aligned_hostmag22_evenmore.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
         "--prebuilt_static_model",
         pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_123_75_images36_points.npy",
     ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "poisson_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_hostnoiseonly_aligned_22mag_host():
    # I think is failing because there are so few no transient compared to with transient images.

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_hostpoisson_aligned_hostmag22.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_123_38_images36_points.npy",
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "hostnoiseonly_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_transientnoiseonly_aligned_22mag_host():
    # I think is failing because there are so few no transient compared to with transient images.

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_transientpoisson_aligned_hostmag22.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_123_38_images36_points.npy",
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "transientnoiseonly_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# Note: These tests are maintained rather than full removal because they are useful for debugging when
# the more complicated tests fail. Since they take a few minutes to run each, we skip them in normal test runs.
@pytest.mark.skip(reason="this test is subsumed by following tests")
def test_both_aligned_22mag_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_aligned_hostmag22_evenmore.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_123_75_images36_points.npy",
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "both_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_noiseless_shifted_22mag_host():
    # This test is kinda unncessary since the more difficult tests pass, but I want to
    # make a direct comparison to the same version of this test with a varying gaussian PSF.
    # By removing noise we can isolate any differences to just the PSF modeling.
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_shifted_host.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_rotated_75images_36points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    np.testing.assert_allclose(lc["flux"], flux, atol=1, rtol=1e-3)

    # The 4.14 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    # The tolerance has to be higher here because the rotation and changing in PSF sizes.
    np.testing.assert_allclose(lc["flux_err"], 4.14, rtol=1.1e-2)


@pytest.mark.skip(reason="This test is superseded by more difficult tests with more noise.")
def test_skynoise_shifted_22mag_host():
    # This test is kinda unncessary since the more difficult tests pass, but I want to
    # make a direct comparison to the same version of this test with a varying gaussian PSF.
    # By removing noise we can isolate any differences to just the PSF modeling.
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_skynoise_shifted_host.txt",
    ]

    # Delete saved lc file if it exists from previous test runs
    lc_path = pathlib.Path(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    if lc_path.exists():
        lc_path.unlink()

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    # cmd += ["--save_model"]

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_rotated_75images_36points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    flux = create_true_flux(lc["mjd"], peakmag=21)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)

    except AssertionError as e:
        plotname = "skynoise_shifted_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with more noise.")
def test_poisson_shifted_22mag_host():
    # This test is kinda unncessary since the more difficult tests pass, but I want to
    # make a direct comparison to the same version of this test with a varying gaussian PSF.
    # By removing noise we can isolate any differences to just the PSF modeling.
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_shifted_host.txt",
    ]

    # Delete saved lc file if it exists from previous test runs
    lc_path = pathlib.Path(f"{out_dir}/123_R062_gaussian_lc.ecsv")
    if lc_path.exists():
        lc_path.unlink()

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    # cmd += ["--save_model"]

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_rotated_75images_36points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

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
        plotname = "poisson_shifted_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_both_shifted_22mag_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_shifted_host_v2.txt",
    ]

    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += [
        "--prebuilt_static_model",
        pathlib.Path(__file__).parent / "testdata/prebuilt_models/psf_matrix_gaussian_rotated_75images_36points.npy",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

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
        plotname = "both_shifted_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e


# ###### Tests with varying gaussian PSF ##############################################################

def test_both_shifted_22mag_host_varying_gaussian():

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_allnoise_varyingPSF_22mag_host_evenmore.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        f"{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    ]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "both_shifted_22mag_host_varying_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_noiseless_aligned_no_host_varying():
    # Test 1. Noiseless, perfectly aligned images. No host galaxy, We expect campari to do extremely well here.
    # Get the images we need
    # Can we define this in the config?

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_varyingPSF_nohost.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    np.testing.assert_allclose(lc["flux"], flux, atol=1e-7)

    # The 3.691204 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    np.testing.assert_allclose(lc["flux_err"], 3.691204, atol=1e-7)

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_noiseless_shifted_no_host_varying():
    # Test 1. Noiseless, perfectly aligned images. No host galaxy, We expect campari to do extremely well here.
    # Get the images we need
    # Can we define this in the config?

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_shifted_varyingPSF_nohost.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    np.testing.assert_allclose(lc["flux"], flux, atol=1e-7)

    # The 3.691204 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    # The tolerance has to be higher here because the rotation and changing in PSF sizes.
    np.testing.assert_allclose(lc["flux_err"], 3.691204, rtol=1e-2)

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_noiseless_shifted_22mag_host_varying():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_noiseless_shifted_varyingPSF_host.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    ]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    np.testing.assert_allclose(lc["flux"], flux, atol=1, rtol=1e-3)

    # The 3.691204 comes from the fact that we set an error floor of 1 on each pixel. If we did not, this error
    # would be zero for the noiseless case, but this causes issues with the inverse variance in other cases so
    # we keep it.
    # The tolerance has to be higher here because the rotation and changing in PSF sizes.
    np.testing.assert_allclose(lc["flux_err"], 4.14, rtol=1.1e-2)

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_skynoise_shifted_22mag_host_varying():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_skynoise_shifted_varyingPSF_host.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    ]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "skynoise_shifted_22mag_host_varying"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with noise.")
def test_poisson_shifted_22mag_host_varying():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_shifted_host_varyingPSF.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    ]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing
    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "poisson_shifted_22mag_host_varying"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test fails but I believe it is due to small number statistics. Revisit this.")
# XXX TODO
def test_both_shifted_nohost_varying():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_shifted_varyingPSF_nohost.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "both_shifted_nohost_varying"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by more difficult tests with more noise.")
def test_poisson_noise_shifted_no_host_varying():
    # Now we add just poisson noise. This will introduce scatter, but campari should still agree with aperture
    # photometry.

    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_aligned_nohost_200_varying.txt",
    ]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    # For this test I bumped it up to 60 images because the small number statistics caused it to not pass gaussianity
    # checks. Thankfully 60 images run in about ~1 min without a bg static model. This will be a bit of a pain for
    # tests with static models unfortunately.

    cmd += ["--photometry-campari-grid_options-type", "none"]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy

    fileroot = "123_R062_varying_gaussian"

    lc = Table.read(f"{out_dir}/{fileroot}_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry(fileroot, imsize, aperture_radius=4)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    np.testing.assert_allclose(lc["flux"], ap_sums, rtol=3e-3)

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
        plotname = "poisson_aligned_nohost_varying_diagnostic"
        generate_diagnostic_plots(f"{fileroot}", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

def test_both_shifted_22mag_host_varying_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_shifted_22mag_host_200_varying.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    #cmd += ["--save_model"]
    cmd += [
         "--prebuilt_static_model",
         f"{debug_dir}/psf_matrix_varying_gaussian_a823ec9c-d418-4ee0-bd22-df5f4540544b_250_images36_points.npy",
     ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "both_shifted_22mag_host_varying_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_both_shifted_21mag_host_varying_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_shifted_21mag_host_200_varying.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Finer grid spacing

    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

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
        plotname = "both_shifted_21mag_host_varying_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_both_shifted_22mag_host_faint_source_varying_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_bothnoise_shifted_22maghost_faintsource_varyinggaussian_evenfainter.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Even Finer grid spacing

    subsize_index = cmd.index("--photometry-campari-grid_options-subsize")
    cmd[subsize_index + 1] = "4"  # Smaller grid

    #cmd += ["--save_model"]
    cmd += [
         "--prebuilt_static_model",
         "{debug_dir}/psf_matrix_varying_gaussian_bdd61d2f-6083-41d2-891d-421b796bedd3_250_images36_points.npy",
     ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
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
        plotname = "both_shifted_22mag_host_faint_source_varying_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_skynoise_shifted_22mag_host_faint_source_regular_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_skynoise_shifted_22maghost_faintsource_regulargaussian_evenfainter.txt",
    ]
    # File is mislabeled, there is a host present
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Even Finer grid spacing

    subsize_index = cmd.index("--photometry-campari-grid_options-subsize")
    cmd[subsize_index + 1] = "4"  # Smaller grid

    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_gaussian_5f1a0fbb-3a8b-4870-bbca-54fd4985a1e0_250_images36_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "varying_gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "varying_gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
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
        plotname = "both_shifted_22mag_host_faint_source_regular_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_poissonnoise_shifted_22mag_host_faint_source_regular_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_poisson_shifted_22maghost_faintsource_regulargaussian_evenfainter.txt",
    ]
    # File is mislabeled, there is a host present
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Even Finer grid spacing

    subsize_index = cmd.index("--photometry-campari-grid_options-subsize")
    cmd[subsize_index + 1] = "4"  # Smaller grid

    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_gaussian_5f1a0fbb-3a8b-4870-bbca-54fd4985a1e0_250_images36_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
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
        plotname = "poissonnoise_shifted_22mag_host_faint_source_regular_gaussian_diagnostic"
        SNLogger.debug("Generating diagnostic plots...")
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

# Right now this is failing due to a bias. I believe this is
# due to some bias resulting from the low SNR + Poisson noise when doing PSF fitting. More work is needed, come
# back to this! XXX XXX XXX TODO XXX
@pytest.mark.xfail
def test_bothnoise_shifted_22mag_host_faint_source_regular_gaussian_more():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_bothnoise_shifted_22maghost_faintsource_regulargaussian_evenfainter.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Even Finer grid spacing

    subsize_index = cmd.index("--photometry-campari-grid_options-subsize")
    cmd[subsize_index + 1] = "4"  # Smaller grid

    # cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_gaussian_5f1a0fbb-3a8b-4870-bbca-54fd4985a1e0_250_images36_points.npy",
    ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
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
        plotname = "bothnoise_shifted_22mag_host_faint_source_regular_gaussian_diagnostic"
        SNLogger.debug("Generating diagnostic plots...")
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_gaussian_bias_analysis():
    cmd = base_cmd + [
        "--img_list",
        str(pathlib.Path(__file__).parent / "testdata/test_gaussims_bothnoise_lowfluxtest_seed46.txt"),
    ]
    # File is mislabeled, there is a host present
    cmd += ["--photometry-campari-grid_options-type", "regular"]
    spacing_index = cmd.index("--photometry-campari-grid_options-spacing")
    cmd[spacing_index + 1] = "0.75"  # Even Finer grid spacing

    subsize_index = cmd.index("--photometry-campari-grid_options-subsize")
    cmd[subsize_index + 1] = "4"  # Smaller grid

    #cmd += ["--save_model"]
    cmd += [
        "--prebuilt_static_model",
        "{debug_dir}/psf_matrix_gaussbiastest.npy",
    ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "gaussian"

    SNLogger.debug("Running command...")
    SNLogger.debug(" ".join(cmd))

    # result = subprocess.run(cmd, capture_output=False, text=True)

    # if result.returncode != 0:
    #     raise RuntimeError(
    #         f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    #     )

    # # Check accuracy
    # lc = Table.read("{out_dir}/123_R062_gaussian_lc.ecsv")

    # mjd = lc["mjd"]
    # peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
    # start_mjd = 60010
    # peak_mjd = 60030
    # end_mjd = 60060
    # flux = np.zeros(len(mjd))
    # flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    # flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    # plotname = "gaussbiastest"
    # SNLogger.debug("Generating diagnostic plots...")
    # generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
    # SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")

    # try:
    #     residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
    #     perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    # except AssertionError as e:
    #     raise e

@pytest.mark.skip(reason="This test is currently too slow to run every time.")
def test_same_as_above_no_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent
        / "testdata/test_gaussims_bothnoise_shifted_samewithnohost_faintsource_varyinggaussian_evenfainter.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]
    cmd += ["--save_model"]
    # cmd += [
    #     "--prebuilt_static_model",
    #     "{debug_dir}/psf_matrix_varying_gaussian_cb100078-9498-4337-acdf-94789a4039fa_75_images36_points.npy",
    # ]
    cmd += ["--nprocs", "15"]

    psfclass_index = cmd.index("--photometry-campari-psf-transient_class")
    cmd[psfclass_index + 1] = "gaussian"
    psfclass_index = cmd.index("--photometry-campari-psf-galaxy_class")
    cmd[psfclass_index + 1] = "gaussian"

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read(f"{out_dir}/123_R062_varying_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)  # note the new peakmag
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
        plotname = "both_shifted_22mag_host_faint_source_varying_gaussian_diagnostic"
        generate_diagnostic_plots("123_R062_varying_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")
        SNLogger.debug(e)
        raise e



