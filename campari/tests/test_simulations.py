import pathlib
import warnings
import sys

import galsim
from matplotlib import pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

from campari import RomanASP
from campari.simulation import simulate_galaxy, simulate_images, simulate_supernova, simulate_wcs
from snappl.diaobject import DiaObject
from snappl.image import FITSImageStdHeaders
from snappl.logger import SNLogger

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


def test_simulate_images():
    lam = 1293  # nm
    ra = 7.47193824
    dec = -44.8280889
    base_sca = 3
    base_pointing = 5934
    bg_gal_flux = 9e5
    size = 11
    band = "Y106"
    airy = galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=galsim.roman.getPSF(1, band, pupil_bin=1).aberrations)
    # Fluxes for the simulated supernova, days arbitrary.
    test_lightcurve = [10, 100, 1000, 10**4, 10**5]

    diaobj = DiaObject.find_objects(name=1, ra=ra, dec=dec, collection="manual")[0]
    diaobj.mjd_start = 61000
    diaobj.mjd_end = 61200

    dates = np.linspace(60000, diaobj.mjd_start, 5).tolist() + \
        np.linspace(diaobj.mjd_start+1, diaobj.mjd_end-1, 5).tolist()
    dates = np.array(dates)

    image_list = []
    for i in range(10):
        img = FITSImageStdHeaders(
            header=None,
            path="/dev/null",
            data=np.zeros((4088, 4088)),
            noise=np.ones((4088, 4088)),
            flags=np.zeros((4088, 4088)),
        )
        img.mjd = dates[i]
        img.band = band
        img.pointing = base_pointing
        img.sca = base_sca
        image_list.append(img)
        SNLogger.debug(f"Created faux image with MJD {img.mjd}")

    simulated_lightcurve, util_ref = simulate_images(
        image_list=image_list,
        diaobj=diaobj,
        sim_gal_ra_offset=1e-5,
        sim_gal_dec_offset=1e-5,
        do_xshift=True,
        do_rotation=True,
        sim_lc=test_lightcurve,
        noise=0,
        deltafcn_profile=False,
        size=size,
        psfclass="ou24PSF",
        bg_gal_flux=bg_gal_flux,
        base_sca=base_sca,
        base_pointing=base_pointing,
        source_phot_ops=False,
        sim_galaxy_scale=None,
        sim_galaxy_offset=None,
        bulge_hlr=1.6,
        disk_hlr=5.0,
    )
    image_list = simulated_lightcurve.image_list
    cutout_image_list = simulated_lightcurve.cutout_image_list

    compare_images = np.load(pathlib.Path(__file__).parent / "testdata/test_sim_images.npy")

    images = []
    for ci in cutout_image_list:
        images.append(ci.data)
    images = np.array(images)
    images = images.flatten()
    np.testing.assert_allclose(images, compare_images.flatten(), atol=1e-7)


def test_simulate_wcs():
    wcs_dict = simulate_wcs(angle=np.pi/4, x_shift=0.1, y_shift=0,
                            base_sca=3,
                            base_pointing=5934, band="Y106")
    b = np.load(pathlib.Path(__file__).parent / "testdata/wcs_dict.npy",
                allow_pickle=True).item()

    for key in list(wcs_dict.keys()):
        np.testing.assert_equal(wcs_dict[key], b[key])


def test_simulate_galaxy():
    band = "F184"
    roman_bandpasses = galsim.roman.getBandpasses()
    lam = 1293  # nm
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"), wave_type="nm",
                     flux_type="fphotons")
    sim_psf = \
        galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=galsim.roman.
                                   getPSF(1, band, pupil_bin=1).aberrations)
    convolved = simulate_galaxy(bg_gal_flux=9e5, sim_galaxy_scale=1, deltafcn_profile=False,
                                band=band, sim_psf=sim_psf, sed=sed)

    a = convolved.drawImage(roman_bandpasses[band], method="no_pixel",
                            use_true_center=True)
    b = np.load(pathlib.Path(__file__).parent
                / "testdata/test_galaxy.npy")
    np.testing.assert_allclose(a.array, b, rtol=3e-7)


def test_simulate_supernova():
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"), wave_type="nm",
                     flux_type="fphotons")

    supernova_image = simulate_supernova(snx=2044, sny=2044,
                                         snx0=2044, sny0=2044, stampsize=11,
                                         flux=1000, sed=sed,
                                         source_phot_ops=False,
                                         base_pointing=662, base_sca=11,
                                         random_seed=12345)

    test_sn = np.load(pathlib.Path(__file__).parent
                      / "testdata/supernova_image.npy")
    np.testing.assert_allclose(supernova_image, test_sn, rtol=1e-7)


# Broken until we can simulate non point sources with snappl PSFs

# def test_deltafcn_galaxy_test(cfg):
#     """In this test, we generate a galaxy that is a delta function, and fit to it with a grid that is also a single
#     point.  The result should be that the fitted flux is exactly the input flux, to machine precision."""

#     base_sca = 3
#     base_pointing = 5934

#     curfile = pathlib.Path(pathlib.Path(cfg.value("photometry.campari.paths.debug_dir")) /
#                            "deltafcn_test_20172782_Y106_romanpsf_images.npy")
#     curfile.unlink(missing_ok=True)
#     imsize = 19

#     a = ["_", "-s", "20172782", "-f", "Y106", "-n", "3", "-t", "0",
#          "--photometry-campari-psfclass", "ou24PSF_slow",
#          "--no-photometry-campari-use_real_images",
#          "--no-photometry-campari-fetch_SED",
#          "--photometry-campari-grid_options-type", "single",
#          "--photometry-campari-cutout_size", f"{imsize}",
#          "--no-photometry-campari-weighting",
#          "--photometry-campari-subtract_background",
#          "--no-photometry-campari-source_phot_ops",
#          "--photometry-campari-simulations-deltafcn_profile",
#          "--photometry-campari-simulations-base_sca", str(base_sca),
#          "--photometry-campari-simulations-base_pointing", str(base_pointing),
#          "--photometry-campari-simulations-noise", "0",
#          "--photometry-campari-simulations-run_name", "deltafcn_test",
#          "--photometry-campari-simulations-bg_gal_flux", "1"]
#     sys.argv = a
#     RomanASP.main()

#     images = np.load(curfile)
#     data = images[0]
#     model = images[1]

#     # This tolerance value was chosen empirically. Looking at the actual image output, the fit seems to have no biases
#     # or structure.
#     SNLogger.debug(np.max(np.abs(data - model)/data))
#     try:
#         np.testing.assert_allclose(data, model, rtol=3e-7)
#     except AssertionError:
#         plt.subplots(1, 3, figsize=(12, 4))
#         plt.subplot(1, 3, 1)
#         plt.title("Data")
#         plt.imshow(data.reshape(-1, imsize, imsize)[0], origin="lower", cmap="viridis", vmin=0,
#                    vmax=np.max(data)*0.1)
#         plt.colorbar()
#         plt.subplot(1, 3, 2)
#         plt.title("Model")
#         plt.imshow(model.reshape(-1, imsize, imsize)[0], origin="lower", cmap="viridis", vmin=0,
#                    vmax=np.max(data)*0.1)
#         plt.colorbar()
#         plt.subplot(1, 3, 3)
#         plt.title("Log Residual")
#         plt.imshow(np.log10(np.abs(data.reshape(-1, imsize, imsize)[0] - model.reshape(-1, imsize, imsize)[0])),
#                    origin="lower",
#                    cmap="viridis", vmin=-10, vmax=-2)
#         plt.colorbar()
#         plt.tight_layout()
#         savepath = pathlib.Path(
#             pathlib.Path(cfg.value("photometry.campari.paths.debug_dir"))
#             / "deltafcn_test_20172782_Y106_romanpsf_images.png"
#         )
#         plt.savefig(savepath)
#         plt.close()

#         raise AssertionError(f"Data and model do not match to tolerance! See {savepath} for image.")
