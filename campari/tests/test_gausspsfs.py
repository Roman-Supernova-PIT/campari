import pathlib
import subprocess


import numpy as np

from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry

from snpit_utils.logger import SNLogger

#  base_path = "/photometry_test_data/simple_gaussian_test/sig1.0"
#     all_images = os.listdir(base_path)
#     noiseless_images = [x for x in all_images if "junkpoisson_" in x and not "more" in x]
#     SNLogger.debug(f"Found {len(noiseless_images)} images")
#     SNLogger.debug(noiseless_images)
#     np.testing.assert_equal(len(noiseless_images), 39)

#     with open(pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson.txt", "w") as f:
#         for item in noiseless_images:
#             # There is an image, noise, and flags, and we don"t want to read the image thrice.
#             if "image" not in item and "flags" not in item and "READ" not in item:
#                 whole_path = os.path.join(base_path, item)
#                 print(whole_path.split("_image.fits")[0])
#                 f.write(f"{whole_path.split('_noise.fits')[0]}\n")

# Test 1. Noiseless, perfectly aligned images. No host galaxy, We expect campari to do extremely well here.

def test_noiseless_aligned_no_host():
    # Get the images we need
    # Can we define this in the config?
    cmd = [
        "python", "../RomanASP.py",
        "-s", "123",
        "-t", "1",
        "-n", "0",
        "-f", "R062",
        "--ra", "128.0",
        "--dec", "42.0",
        "--transient_start", "60010",
        "--transient_end", "60060",
        "--photometry-campari-psfclass", "gaussian",
        "--photometry-campari-use_real_images",
        "--object_collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-type", "none",
        "--photometry-campari-grid_options-spacing", "5",
        "--photometry-campari-cutout_size", "19",
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background",
        "--no-photometry-campari-source_phot_ops",
        "--image_source", "manual_fits",
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_1.txt",
        "--photometry-campari-simulations-run_name", "gauss_source_no_grid",
        "--image_path", "/photometry_test_data/simple_gaussian_test/sig1.0",
    ]

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

    np.testing.assert_allclose(lc["flux_fit"], flux, atol=1e-7)
    np.testing.assert_allclose(lc["flux_fit_err"], 0.0, atol=1e-7)



def test_poisson_noise_aligned_no_host():
    # Now we add just poisson noise. This will introduce scatter, but campari should still agree with aperture
    # photometry.

    imsize = 19

    cmd = [
        "python", "../RomanASP.py",
        "-s", "123",
        "-t", "1",
        "-n", "0",
        "-f", "R062",
        "--ra", "128.0",
        "--dec", "42.0",
        "--transient_start", "60010",
        "--transient_end", "60060",
        "--photometry-campari-psfclass", "gaussian",
        "--photometry-campari-use_real_images",
        "--object_collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-type", "none",
        "--photometry-campari-grid_options-spacing", "5",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background",
        "--no-photometry-campari-source_phot_ops",
        "--image_source", "manual_fits",
        "--img_list", pathlib.Path(__file__).parent / "testdata/test_gaussims_poisson.txt",
        "--photometry-campari-simulations-run_name", "gauss_source_no_grid",
        "--image_path", "/photometry_test_data/simple_gaussian_test/sig1.0",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/123_R062_gaussian_lc.ecsv")
    noise_maps = np.load("/campari_debug_dir/123_R062_gaussian_noise_maps.npy").reshape(-1, imsize, imsize)
    ims = np.load("/campari_debug_dir/123_R062_gaussian_images.npy")[0].reshape(-1, imsize, imsize)

    # Perform aperture photometry for comparison
    ap_sums = []
    ap_err = []
    lc_index = 0
    for i in range(ims.shape[0]):

        if i < lc.meta["pre_transient_images"] or i > ims.shape[0] - lc.meta["post_transient_images"] - 1:
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
    # rtol determined empirically. We expect them to be close, but there is the aperture correction etc.
    np.testing.assert_allclose(lc["flux_fit"], ap_sums, rtol=3e-3)
