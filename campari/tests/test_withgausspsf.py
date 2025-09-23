
# import pathlib
# import warnings
# import sys

# import galsim
# import numpy as np
# import pytest
# from astropy.utils.exceptions import AstropyWarning
# from erfa import ErfaWarning

# from campari import RomanASP
# from campari.simulation import simulate_galaxy, simulate_images, simulate_supernova, simulate_wcs
# from snappl.diaobject import DiaObject
# from snappl.image import ManualFITSImage
# from snpit_utils.logger import SNLogger


# def test_nogalaxy_SNPhotometry(cfg):
#     """In this test, we generate a galaxy that is a delta function, and fit to it with a grid that is also a single
#     point.  The result should be that the fitted flux is exactly the input flux, to machine precision."""

#     base_sca = 3
#     base_pointing = 5934

#     curfile = pathlib.Path(pathlib.Path(cfg.value("photometry.campari.paths.debug_dir")) /
#                            "deltafcn_test_20172782_Y106_romanpsf_images.npy")
#     curfile.unlink(missing_ok=True)

#     a = ["_", "-s", "20172782", "-f", "Y106", "-n", "0", "-t", "5",
#          "--photometry-campari-use_roman",
#          "--no-photometry-campari-use_real_images",
#          "--no-photometry-campari-fetch_SED",
#          "--photometry-campari-grid_options-type", "none",
#          "--photometry-campari-cutout_size", "19",
#          "--no-photometry-campari-weighting",
#          "--photometry-campari-subtract_background",
#          "--no-photometry-campari-source_phot_ops",
#          "--photometry-campari-simulations-base_sca", str(base_sca),
#          "--photometry-campari-simulations-base_pointing", str(base_pointing),
#          "--photometry-campari-simulations-noise", "0",
#          "--photometry-campari-simulations-run_name", "gauss_test",
#          "--photometry-campari-simulations-bg_gal_flux", "0"]
#     sys.argv = a
#     RomanASP.main()

#     images = np.load(curfile)
#     data = images[0]
#     model = images[1]

#     # This tolerance value was chosen empirically. Looking at the actual image output, the fit seems to have no biases
#     # or structure.
#     SNLogger.debug(np.max(np.abs(data - model)/data))
#     np.testing.assert_allclose(data, model, rtol=3e-7)