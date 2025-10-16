import pathlib
import sys

import numpy as np
import pandas as pd


from campari import RomanASP
from campari.simulation import simulate_images
from snappl.diaobject import DiaObject
# from snappl.image import ManualFITSImage
# from snpit_utils.logger import SNLogger

# Sanity check regression test. Make sure that the same images are generated as before. (No galaxy included here.)
def test_simulate_gauss_images():
    ra = 7.47193824
    dec = -44.8280889
    base_sca = 3
    base_pointing = 5934
    bg_gal_flux = 0 # For now we can't simulate galaxies with the Gaussian PSF. Change this once we can. TODO
    size = 11
    band = "Y106"
    # Fluxes for the simulated supernova, days arbitrary.
    test_lightcurve = [10, 100, 1000, 10**4, 10**5]

    diaobj = DiaObject.find_objects(id=1, ra=ra, dec=dec, collection="manual")[0]
    diaobj.mjd_start = 61000
    diaobj.mjd_end = 61200

    dates = (
        np.linspace(60000, diaobj.mjd_start, 5).tolist()
        + np.linspace(diaobj.mjd_start + 1, diaobj.mjd_end - 1, 5).tolist()
    )
    dates = np.array(dates)

    image_list = []
    for i in range(10):
        img = ManualFITSImage(
            header=None,
            data=np.zeros((4088, 4088)),
            noise=np.ones((4088, 4088)),
            flags=np.zeros((4088, 4088)),
            mjd=dates[i],
            band=band,
            pointing=base_pointing,
            sca=base_sca,
        )
        image_list.append(img)

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
        psfclass="gaussian",
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

    images = []
    for ci in cutout_image_list:
        images.append(ci.data)
    images = np.array(images)
    images = images.flatten()

    compare_images = np.load(pathlib.Path(__file__).parent / "testdata/test_gausssim_images.npy")
    np.testing.assert_allclose(images, compare_images.flatten(), atol=1e-7)


def test_nogalaxy_no_noise_SNPhotometry(cfg):
    base_sca = 3
    base_pointing = 5934
    noise = 0

    a = ["_", "-s", "20172782", "-f", "Y106", "-n", "0", "-t", "5",
         "--photometry-campari-psfclass", "gaussian",
         "--no-photometry-campari-use_real_images",
         "--no-photometry-campari-fetch_SED",
         "--photometry-campari-grid_options-type", "none",
         "--photometry-campari-cutout_size", "19",
         "--no-photometry-campari-weighting",
         "--photometry-campari-subtract_background",
         "--no-photometry-campari-source_phot_ops",
         "--photometry-campari-simulations-base_sca", str(base_sca),
         "--photometry-campari-simulations-base_pointing", str(base_pointing),
         "--photometry-campari-simulations-noise", str(noise),
         "--photometry-campari-simulations-run_name", "gauss_test",
         "--photometry-campari-simulations-bg_gal_flux", "0"]
    sys.argv = a
    RomanASP.main()

    df = pd.read_csv("/campari_out_dir/gauss_test_20172782_Y106_gaussian_lc.ecsv", comment="#", delimiter=" ")
    for index, row in df.iterrows():
        np.testing.assert_allclose(row["flux"], row["sim_flux"], rtol=1e-7)


def test_gaussianSMP_withgrid_butnogalaxy(cfg):
    base_sca = 3
    base_pointing = 5934
    noise = 0

    a = ["_", "-s", "20172782", "-f", "Y106", "-n", "5", "-t", "5",
         "--photometry-campari-psfclass", "gaussian",
         "--no-photometry-campari-use_real_images",
         "--no-photometry-campari-fetch_SED",
         "--photometry-campari-grid_options-type", "regular",
         "--photometry-campari-cutout_size", "15",
         "--no-photometry-campari-weighting",
         "--photometry-campari-subtract_background",
         "--no-photometry-campari-source_phot_ops",
         "--photometry-campari-simulations-base_sca", str(base_sca),
         "--photometry-campari-simulations-base_pointing", str(base_pointing),
         "--photometry-campari-simulations-noise", str(noise),
         "--photometry-campari-simulations-run_name", "gauss_test_grid",
         "--photometry-campari-simulations-bg_gal_flux", "0", "--save_model"]
    sys.argv = a
    RomanASP.main()

    df = pd.read_csv("/campari_out_dir/gauss_test_grid_20172782_Y106_gaussian_lc.ecsv", comment="#", delimiter=" ")
    for index, row in df.iterrows():
        np.testing.assert_allclose(row["flux"], row["sim_flux"], rtol=1e-7)


