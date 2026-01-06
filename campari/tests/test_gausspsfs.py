import pathlib
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, skewtest

from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry

from snappl.logger import SNLogger


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


def perform_aperture_photometry(fileroot, imsize, aperture_radius=4):
    noise_maps = np.load(f"/campari_debug_dir/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    lc = Table.read(f"/campari_out_dir/{fileroot}_lc.ecsv")

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


def generate_diagnostic_plots(fileroot, imsize, plotname, ap_sums=None, ap_err=None, trueflux=None):
    SNLogger.debug("Generating diagnostic plots....")
    lc = Table.read(f"/campari_out_dir/{fileroot}_lc.ecsv")
    ims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    noise_maps = np.load(f"/campari_debug_dir/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)

    numcols = 4
    plt.figure(figsize=(numcols * 5, ims.shape[0] * 5))
    for i in range(ims.shape[0]):
        k = 0
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)

        if i == 0:
            plt.title("Input Image")
        im = plt.imshow(ims[i], origin="lower")
        vmin, vmax = im.get_clim()
        xticks = np.arange(0, imsize, 5) - 0.5

        if imsize < 30:
            plt.xticks(xticks)
            plt.yticks(xticks)
            plt.grid(True)
        plt.colorbar()

        ###########################################################################

        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Model Image")
        plt.xlim(-0.5, imsize - 0.5)
        plt.ylim(-0.5, imsize - 0.5)

        if imsize < 30:
            plt.xticks(xticks)
            plt.yticks(xticks)
            plt.grid(True)
        plt.imshow(modelims[i], origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar()

        ################################
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Residuals")
        plt.imshow((ims[i] - modelims[i]), origin="lower", vmin=-200, vmax=200, cmap="seismic")

        # ###############################
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Pixel Pulls")
        bins = np.linspace(-4, 4, 50)
        residuals = (modelims[i] - ims[i]).flatten()
        pixel_pull = (modelims[i].flatten() - ims[i].flatten()) / noise_maps[i].flatten()
        pixel_pull = pixel_pull[np.where(np.abs(residuals) >= 1)]  # Remove zero residuals from the no transient images.
        plt.hist(pixel_pull, bins=bins, density=True, alpha=0.5, label="Pixel Pulls")
        normal_dist = norm(loc=0, scale=1)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, normal_dist.pdf(x), label="Normal Dist", color="black")
        mu, sig = norm.fit(pixel_pull)
        plt.plot(x, norm.pdf(x, mu, sig), label=f"Fit: mu={mu:.2f}, sig={sig:.2f}", color="red")

        plt.legend()

        plt.colorbar()

        ################################
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("/campari_debug_dir/" + plotname + ".png")
    plt.close()

    SNLogger.debug("Generated image diagnostics and saved to /campari_debug_dir/" + plotname + ".png")
    SNLogger.debug("Now generating light curve diagnostics...")
    # Now plot a light curve
    if trueflux is not None:
        plt.subplot(2, 2, 1)
        plt.errorbar(lc["mjd"], lc["flux"] - trueflux, yerr=lc["flux_err"], marker="o", linestyle="None",
                     label="Campari Fit - Truth")

        residuals = lc["flux"] - trueflux
        window_size = 3
        rolling_avg = np.convolve(residuals, np.ones(window_size) / window_size, mode="valid")
        plt.plot(lc["mjd"][window_size - 1:], rolling_avg, label="Rolling Average", color="orange")

        if ap_sums is not None and ap_err is not None:
            SNLogger.debug(f"aperture phot std: {np.std(np.array(ap_sums) - trueflux)}")
            plt.errorbar(lc["mjd"], np.array(ap_sums) - trueflux, yerr=ap_err, marker="o", linestyle="None",
                         label="Aperture Phot - Truth", color="red")
            plt.errorbar(lc["mjd"], lc["flux"] - np.array(ap_sums),
                         yerr=np.sqrt(lc["flux_err"]**2 + np.array(ap_err)**2), marker="o", linestyle="None",
                         label="Campari - Aperture Phot", color="green")

        SNLogger.debug(f"campari std: {np.std(lc['flux_fit'] - trueflux)}")

        plt.axhline(0, color="black", linestyle="--")
        plt.legend()
        plt.xlabel("MJD")
        plt.ylabel("Flux (e-)")
        plt.xlim(np.min(lc["mjd"]) - 10, np.max(lc["mjd"]) + 10)
        plt.title(plotname + " Light Curve Residuals")

        plt.subplot(2, 2, 2)
        pull = (lc["flux"] - trueflux) / lc["flux_err"]
        plt.hist(pull, bins=10, alpha=0.5, label="Campari Pull", density=True)
        normal_dist = norm(loc=0, scale=1)
        x = np.linspace(-5, 5, 100)
        plt.plot(x, normal_dist.pdf(x), label="Normal Dist", color="black")

        mu, sig = norm.fit(pull)
        plt.plot(x, norm.pdf(x, mu, sig), label=f"Fit: mu={mu:.2f}, sig={sig:.2f}", color="red")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.errorbar(lc["mjd"], lc["flux"], yerr=lc["flux_err"], marker="o", linestyle="None",
                     label="Campari Fit - Truth", ms=1)
        plt.errorbar(lc["mjd"], trueflux, yerr=None, marker="o", linestyle="None", label="Truth", color="black", ms=1)
        plt.yscale("log")
        plt.ylim(1e3, 1e5)

        plt.savefig("/campari_debug_dir/" + plotname + "_lc.png")


def perform_gaussianity_checks(residuals_sigma):
    """Most of these tests apply the same checks, so just put them in a function."""
    sub_one_sigma = np.sum(np.abs(residuals_sigma) < 1)
    SNLogger.debug(f"Campari fraction within 1 sigma: {sub_one_sigma / len(residuals_sigma)}")
    np.testing.assert_allclose(sub_one_sigma / len(residuals_sigma), 0.68, atol=0.2, err_msg="Errors non gaussian!?")

    mu, sig = norm.fit(residuals_sigma)
    # If I am not mistaken, this is equivalent to checking that the residuals are unbiased at 3 sigma confidence.
    mu_atol = 3 / np.sqrt(len(residuals_sigma))
    SNLogger.debug("Fitted residuals mu: " + str(mu) + ", sig: " + str(sig) + "\n mu tolerance: " + str(mu_atol))
    np.testing.assert_allclose(mu, 0, atol=mu_atol, err_msg="Residuals biased!")
    np.testing.assert_allclose(sig, 1, atol=0.2, err_msg="Residuals too broad or narrow!")
    # Check to make sure our distribution is not skewed.
    p_value = skewtest(residuals_sigma).pvalue
    SNLogger.debug("Skewness p_value: " + str(p_value))
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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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


def test_poisson_noise_aligned_no_host():
    # Now we add just poisson noise. This will introduce scatter, but campari should still agree with aperture
    # photometry.

    cmd = base_cmd + [
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson_no_host_even_more.txt"
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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "poisson_aligned_nohost_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0 --no-star-noise -b junkskynoise
# --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055
# 60060 --image-centers 128 42 -θ 0 -r 30 -s 0 --transient-ra 128 --transient-dec 42
# --no-star-noise --no-transient-noise -n 1


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    SNLogger.debug("Loaded this lc: " + str(lc))
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "sky_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "both_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0 --no-star-noise -b shifted_noiseless
#  --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050 60055
# 60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999   41.999
#  128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999   42.0005 128.001
#   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330. 360. -r 0 -s 0
# --transient-ra 128 --transient-dec 42 --no-star-noise --no-transient-noise -n 1

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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    np.testing.assert_allclose(lc["flux"], ap_sums, rtol=3e-3)
    # np.testing.assert_allclose(lc["flux_err"], ap_err, rtol=3e-3)

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "poisson_shifted_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


# python /snappl/snappl/image_simulator.py --seed 42 --star-center 128 42 -n 0  --no-star-noise -b shifted_sky
#  --width 256 --height 256 --pixscale 0.11 -t 60000 60005 60010 60015 60020 60025 60030 60035 60040 60045 60050
#  60055 60060 --image-centers 128.      42.     127.999   42.     128.001   42.     128.      41.999  127.999
#  41.999  128.001   41.999  128.      42.001  127.999   42.001  128.001   42.001  128.      42.0005 127.999
#   42.0005 128.001   42.0005 128.      42.     -θ   0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330.
#  360. -r 30 -s 0 --transient-ra 128 --transient-dec 42 --no-star-noise  -n 1 --no-transient-noise

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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "shifted_sky_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    ap_sums, ap_err = perform_aperture_photometry("123_R062_gaussian", imsize, aperture_radius=4)

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)
    SNLogger.debug(f"Expected fluxes: {flux}")

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "shifted_both_noise_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, ap_sums=ap_sums, ap_err=ap_err, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


# Just background tests

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
    SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")

    fileroot = "123_R062_gaussian"
    ims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    np.testing.assert_allclose(residuals, 0, atol=20)


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
    noise_maps = np.load(f"/campari_debug_dir/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    pixel_pulls = residuals / noise_maps
    try:
        perform_gaussianity_checks(pixel_pulls.flatten())
    except AssertionError as e:
        plotname = "both_aligned_just_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=None)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


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
    noise_maps = np.load(f"/campari_debug_dir/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"/campari_debug_dir/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    residuals = ims - modelims
    pixel_pulls = residuals / noise_maps
    try:
        perform_gaussianity_checks(pixel_pulls.flatten())
    except AssertionError as e:
        plotname = "both_shifted_just_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=None)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

# 22 mag delta function galaxy tests ############################################################################


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((21 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(lc["flux"], flux, atol=1e-7)

        # The error is small, but higher than that in the no host case. I believe this is due to extra uncertainty
        # thanks to more free parameters in the fit.
        np.testing.assert_allclose(lc["flux_err"], 4.520784, atol=1e-7)

    except AssertionError as e:
        plotname = "noiseless_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "poisson_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "hostnoiseonly_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "transientnoiseonly_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "both_aligned_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_both_shifted_22mag_host():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_both_shifted_hostmag22_evenmore.txt",
    ]

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
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")

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
        perform_gaussianity_checks(residuals_sigma)
    except AssertionError as e:
        plotname = "both_shifted_22mag_host_diagnostic"
        generate_diagnostic_plots("123_R062_gaussian", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e
