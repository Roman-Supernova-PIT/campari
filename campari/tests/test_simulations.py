import pathlib
import warnings
import sys

import galsim
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

from campari import RomanASP
from campari.simulation import simulate_galaxy, simulate_images, simulate_supernova, simulate_wcs
from snpit_utils.logger import SNLogger

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


@pytest.fixture(scope="module")
def roman_path(cfg):
    return cfg.value("photometry.campari.paths.roman_path")


@pytest.fixture(scope="module")
def sn_path(cfg):
    return cfg.value("photometry.campari.paths.sn_path")


def test_simulate_images(roman_path):
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
    sim_lc, util_ref, image_list, cutout_image_list, sim_galra, sim_galdec, galaxy_images, noise_maps = simulate_images(
        num_total_images=10,
        num_detect_images=5,
        ra=ra,
        dec=dec,
        sim_gal_ra_offset=1e-5,
        sim_gal_dec_offset=1e-5,
        do_xshift=True,
        do_rotation=True,
        sim_lc=test_lightcurve,
        noise=0,
        use_roman=False,
        band=band,
        deltafcn_profile=False,
        roman_path=roman_path,
        size=size,
        input_psf=airy,
        bg_gal_flux=bg_gal_flux,
        base_sca=base_sca,
        base_pointing=base_pointing,
        source_phot_ops=False,
        sim_galaxy_scale=None,
        sim_galaxy_offset=None,
        bulge_hlr=1.6,
        disk_hlr=5.0,
    )

    compare_images = np.load(pathlib.Path(__file__).parent
                             / "testdata/test_sim_images.npy")

    images = []
    for ci in cutout_image_list:
        images.append(ci.data)
    images = np.array(images)
    images = images.flatten()
    np.testing.assert_allclose(images, compare_images.flatten(), atol=1e-7)


def test_simulate_wcs(roman_path):
    wcs_dict = simulate_wcs(angle=np.pi/4, x_shift=0.1, y_shift=0,
                            roman_path=roman_path, base_sca=3,
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


def test_deltafcn_galaxy_test(cfg):
    """In this test, we generate a galaxy that is a delta function, and fit to it with a grid that is also a single
    point.  The result should be that the fitted flux is exactly the input flux, to machine precision."""

    base_sca = 3
    base_pointing = 5934

    curfile = pathlib.Path(pathlib.Path(cfg.value("photometry.campari.paths.debug_dir")) /
                     "deltafcn_test_20172782_Y106_romanpsf_images.npy")
    curfile.unlink(missing_ok=True)

    a = ["_", "-s", "20172782", "-f", "Y106", "-n", "3", "-t", "0",
         "--photometry-campari-use_roman",
         "--no-photometry-campari-use_real_images",
         "--no-photometry-campari-fetch_SED",
         "--photometry-campari-grid_options-type", "single",
         "--photometry-campari-cutout_size", "19",
         "--no-photometry-campari-weighting",
         "--photometry-campari-subtract_background",
         "--no-photometry-campari-source_phot_ops",
         "--photometry-campari-simulations-deltafcn_profile",
         "--photometry-campari-simulations-base_sca", str(base_sca),
         "--photometry-campari-simulations-base_pointing", str(base_pointing),
         "--photometry-campari-simulations-noise", "0",
         "--photometry-campari-simulations-run_name", "deltafcn_test",
         "--photometry-campari-simulations-bg_gal_flux", "1"]
    sys.argv = a
    RomanASP.main()

    images = np.load(curfile)
    data = images[0]
    model = images[1]

    # This tolerance value was chosen empirically. Looking at the actual image output, the fit seems to have no biases
    # or structure.
    SNLogger.debug(np.max(np.abs(data - model)/data))
    np.testing.assert_allclose(data, model, rtol=3e-7)
