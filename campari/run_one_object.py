# Standard Library
import pathlib
import warnings

# Common Library

import numpy as np
from numpy.linalg import LinAlgError
import scipy.sparse as sp

# Astronomy Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import galsim
from roman_imsim.utils import roman_utils

# SN-PIT
from campari.data_construction import construct_images, prep_data_for_fit
from campari.model_building import construct_static_scene, construct_transient_scene, generate_guess, make_grid
from campari.simulation import simulate_images
from campari.utils import banner, calculate_local_surface_brightness, campari_lightcurve_model, get_weights
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

r"""
Cole Meldorf 2025
Adapted from code by Pedro Bernardinelli

                    ___
                   / _ \___  __ _  ___ ____
                  / , _/ _ \/  ' \/ _ `/ _ \
                 /_/|_|\___/_/_/_/\_,_/_//_/
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣔⣴⣦⣔⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣭⣿⣟⣿⣿⣿⣅⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣷⣾⣿⣿⣿⣿⣿⣿⣿⡶⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⠤⢤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⢒⣿⣿⣿⣠⠋⠀⠀⠀⠀⠀⠀⣀⣀⠤⠶⠿⠿⠛⠿⠿⠿⢻⢿⣿⣿⣿⠿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⡞⢀⣿⣿⣿⡟⠃⠀⠀⠀⣀⡰⠶⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠃⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠘⢧⣤⣈⣡⣤⠤⠴⠒⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀


                 _____  __     ___  __________
                / __/ |/ /    / _ \/  _/_  __/
               _\ \/    /    / ___// /  / /
              /___/_/|_/    /_/  /___/ /_/


"""
# Global variables
huge_value = 1e32


def run_one_object(diaobj=None, object_type=None, image_list=None, size=None, band=None, fetch_SED=None, sedlist=None,
                   use_real_images=None, subtract_background=None, psfclass=None,
                   make_initial_guess=None, initial_flux_guess=None, weighting=None, method=None,
                   grid_type=None, pixel=None, source_phot_ops=None, do_xshift=None, bg_gal_flux=None, do_rotation=None,
                   airy=None, mismatch_seds=None, deltafcn_profile=None, noise=None,
                   avoid_non_linearity=None, spacing=None, percentiles=None, sim_galaxy_scale=1,
                   sim_galaxy_offset=None, base_pointing=662, base_sca=11,
                   draw_method_for_non_roman_psf="no_pixel"):
    psf_matrix = []
    sn_matrix = []

    # This is a catch for when I'm doing my own simulated WCSs
    util_ref = None

    percentiles = []

    num_total_images = len(image_list)
    transient_image_list = [a for a in image_list if a.mjd >= diaobj.mjd_start and a.mjd <= diaobj.mjd_end]
    num_detect_images = len(transient_image_list)

    if use_real_images:
        cutout_image_list, image_list = construct_images(image_list, diaobj, size,
                                                         subtract_background=subtract_background)
        noise_maps = [im.noise for im in cutout_image_list]

        # We didn't simulate anything, so set these simulation only vars to none.
        sim_galra = None
        sim_galdec = None
        galaxy_images = None


    else:
        # Simulate the images of the SN and galaxy.
        banner("Simulating Images")
        simulated_lightcurve, util_ref = \
            simulate_images(image_list=image_list, diaobj=diaobj,
                            sim_galaxy_scale=sim_galaxy_scale, sim_galaxy_offset=sim_galaxy_offset,
                            do_xshift=do_xshift, do_rotation=do_rotation, noise=noise,
                            size=size, psfclass=psfclass,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            source_phot_ops=source_phot_ops,
                            mismatch_seds=mismatch_seds, base_pointing=base_pointing,
                            base_sca=base_sca)
        sim_lc = simulated_lightcurve.sim_lc
        image_list = simulated_lightcurve.image_list
        cutout_image_list = simulated_lightcurve.cutout_image_list
        galaxy_images = simulated_lightcurve.galaxy_images
        noise_maps = simulated_lightcurve.noise_maps
        sim_galra = simulated_lightcurve.galra
        sim_galdec = simulated_lightcurve.galdec
        object_type = "SN"

    # Build the background grid
    if not grid_type == "none":
        if object_type == "star":
            SNLogger.warning("For fitting stars, you probably dont want a grid.")
        ra_grid, dec_grid = make_grid(grid_type, cutout_image_list, diaobj.ra, diaobj.dec,
                                      percentiles=percentiles, single_ra=sim_galra,
                                      single_dec=sim_galdec, spacing=spacing)
    else:
        ra_grid = np.array([])
        dec_grid = np.array([])

    # Using the images, hazard an initial guess.
    # The num_total_images - num_detect_images check is to ensure we have
    # pre-detection images. Otherwise, initializing the model guess does not
    # make sense.
    num_nondetect_images = num_total_images - num_detect_images
    if make_initial_guess and num_nondetect_images != 0 and grid_type != "none":
        SNLogger.debug("Making initial guess for the model")
        x0test = generate_guess(cutout_image_list[:num_nondetect_images],
                                ra_grid, dec_grid)
        x0_vals_for_sne = np.full(num_total_images, initial_flux_guess)
        x0test = np.concatenate([x0test, x0_vals_for_sne], axis=0)
        SNLogger.debug(f"setting initial guess to {initial_flux_guess}")

    else:
        x0test = None

    banner("Building Model")

    no_transient_cutouts = [a for a in cutout_image_list if a.mjd < diaobj.mjd_start or a.mjd > diaobj.mjd_end]
    if len(no_transient_cutouts) > 0:
        LSB = calculate_local_surface_brightness(no_transient_cutouts, cutout_pix=2)
    else:
        LSB = None

    # Build the backgrounds loop
    for i, image in enumerate(image_list):
        pointing, sca = image.pointing, image.sca

        whole_sca_wcs = image.get_wcs()
        object_x, object_y = whole_sca_wcs.world_to_pixel(diaobj.ra, diaobj.dec)

        # Build the model for the background using the correct psf and the
        # grid we made in the previous section.

        # TODO: Put this in snappl
        SNLogger.debug(f"Image {i} pointing, sca: {pointing, sca}")
        if pointing >= 0 and pointing <= 57364:
            util_ref = roman_utils(config_file=pathlib.Path(Config.get().value
                                   ("photometry.campari.galsim.tds_file")),
                                   visit=pointing, sca=sca)
        else:
            util_ref = None
            # Rob's simulated images have big placeholder pointings. This is a catch for those images.
            SNLogger.warning("Pointing value is outside of the range of the TDS file.")

        # If no grid, we still need something that can be concatenated in the
        # linear algebra steps, so we initialize an empty array by default.
        background_model_array = np.empty((size**2, 0))
        SNLogger.debug("Constructing background model array for image " + str(i) + " ---------------")
        if grid_type != "none":
            background_model_array = \
                construct_static_scene(ra_grid, dec_grid,
                                       whole_sca_wcs,
                                       object_x, object_y, size,
                                       pixel=pixel, image=image, psfclass=psfclass,
                                       util_ref=util_ref, band=band)

        if not subtract_background:
            # If we did not manually subtract the background, we need to fit in the forward model. Since the
            # background is a constant, we add a term to the model that is all ones. But we only want the background
            # to be present in the model for the image it is associated with. Therefore, we only add the background
            # model term when we are on the image that is being modeled, otherwise we add a term that is all zeros.
            # This is the same as to why we have to make the rest of the SN model zeroes in the other images.
            for j in range(num_total_images):
                if i == j:
                    bg = np.ones(size**2).reshape(-1, 1)
                else:
                    bg = np.zeros(size**2).reshape(-1, 1)
                background_model_array =\
                    np.concatenate([background_model_array, bg], axis=1)

        # Add the array of the model points and the background (if using)
        # to the matrix of all components of the model.
        psf_matrix.append(background_model_array)

        # The arrays below are the length of the number of images that contain the object
        # Therefore, when we iterate onto the
        # first object image, we want to be on the first element
        # of sedlist. Therefore, we subtract by the number of
        # predetection images: num_total_images - num_detect_images.
        # I.e., sn_index is the 0 on the first image with an object, 1 on the second, etc.
        sn_index = i - (num_total_images - num_detect_images)
        SNLogger.debug(f"i, sn_index: {i, sn_index}")
        SNLogger.debug(f"Image mjd: {image.mjd}, diaobj mjd_start, mjd_end: {diaobj.mjd_start, diaobj.mjd_end}")
        if image.mjd >= diaobj.mjd_start and image.mjd <= diaobj.mjd_end:
            SNLogger.debug("Constructing transient model array for image " + str(i) + " ---------------")
            if use_real_images:
                pointing = pointing
                sca = sca
            else:
                pointing = base_pointing
                sca = base_sca
            # sedlist is the length of the number of supernova
            # detection images. Therefore, when we iterate onto the
            # first supernova image, we want to be on the first element
            # of sedlist. Therefore, we subtract by the number of
            # predetection images: num_total_images - num_detect_images.
            sn_index = i - (num_total_images - num_detect_images)
            SNLogger.debug(f"Using SED #{sn_index}")
            sed = sedlist[sn_index]
            # object_x and object_y are the exact coords of the SN in the SCA frame.
            # x and y are the pixels the image has been cut out on, and
            # hence must be ints. Before, I had object_x and object_y as SN coords in the cutout frame, hence this
            # switch.
            # In snappl, centers of pixels occur at integers, so the center of the lower left pixel is (0,0).
            # Therefore, if you are at (0.2, 0.2), you are in the lower left pixel, but at (0.6, 0.6), you have
            # crossed into the next pixel, which is (1,1). So we need to round everything between -0.5 and 0.5 to 0,
            # and everything between 0.5 and 1.5 to 1, etc. This code below does that, and follows how snappl does
            # it. For more detail, see the docstring of get_stamp in the PSF class definition of snappl.
            x = int(np.floor(object_x + 0.5))
            y = int(np.floor(object_y + 0.5))
            SNLogger.debug(f"x, y, object_x, object_y, {x, y, object_x, object_y}")
            psf_source_array = construct_transient_scene(x0=x, y0=y, pointing=pointing, sca=sca,
                                                         stampsize=size, x=object_x,
                                                         y=object_y, sed=sed, psfclass=psfclass,
                                                         photOps=source_phot_ops, image=image)

            sn_matrix.append(psf_source_array)

    banner("Lin Alg Section")
    psf_matrix = np.vstack(np.array(psf_matrix))
    SNLogger.debug(f"{psf_matrix.shape} psf matrix shape")

    # Add in the supernova images to the matrix in the appropriate location
    # so that it matches up with the image it represents.
    # All others should be zero.

    # Get the weights

    if weighting:
        wgt_matrix = get_weights(cutout_image_list, diaobj.ra, diaobj.dec)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[0])

    images, err, sn_matrix, wgt_matrix =\
        prep_data_for_fit(cutout_image_list, sn_matrix, wgt_matrix)
    # Combine the background model and the supernova model into one matrix.
    psf_matrix = np.hstack([psf_matrix, sn_matrix])

    # Calculate amount of the PSF cut out by setting a distance cap
    test_sn_matrix = np.copy(sn_matrix)
    test_sn_matrix[np.where(wgt_matrix == 0), :] = 0
    SNLogger.debug(f"SN PSF Norms Pre Distance Cut:{np.sum(sn_matrix, axis=0)}")
    SNLogger.debug("SN PSF Norms Post Distance Cut:"
                   f"{np.sum(test_sn_matrix, axis=0)}")

    # this is where the hstack was before

    banner("Solving Photometry")

    # These if statements can definitely be written more elegantly.
    if not make_initial_guess:
        x0test = np.zeros(psf_matrix.shape[1])

    if not subtract_background:
        x0test = np.concatenate([x0test, np.zeros(num_total_images)], axis=0)

    SNLogger.debug(f"shape psf_matrix: {psf_matrix.shape}")
    SNLogger.debug(f"shape wgt_matrix: {wgt_matrix.reshape(-1, 1).shape}")
    SNLogger.debug(f"image shape: {images.shape}")

    if method == "lsqr":

        wgt_matrix = np.sqrt(wgt_matrix)

        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix, x0=x0test, atol=1e-12,
                              btol=1e-12, iter_lim=300000, conlim=1e10)
        X, istop, itn, r1norm = lsqr[:4]
        SNLogger.debug(f"Stop Condition {istop}, iterations: {itn}," +
                       f"r1norm: {r1norm}")

    flux = X[-num_detect_images:] if num_detect_images > 0 else None


    inv_cov = psf_matrix.T @ np.diag(wgt_matrix**2) @ psf_matrix


    try:
        cov = np.linalg.inv(inv_cov)
    except LinAlgError:
        cov = np.linalg.pinv(inv_cov)

    if num_detect_images > 0:
        SNLogger.debug(f"flux: {np.array2string(flux, separator=', ')}")
    sigma_flux = np.sqrt(np.diag(cov)[-num_detect_images:]) if num_detect_images > 0 else None

    SNLogger.debug(f"sigma flux: {sigma_flux}")

    # Using the values found in the fit, construct the model images.
    pred = X*psf_matrix
    model_images = np.sum(pred, axis=1)

    galaxy_only_model_images = np.sum(X[:-num_detect_images]*psf_matrix[:, :-num_detect_images], axis=1) \
        if num_detect_images > 0 else np.sum(X*psf_matrix, axis=1)

    if use_real_images:
        # Eventually I might completely separate out simulated SNe, though I
        # am hesitant to do that as I want them to be treated identically as
        # possible. In the meantime, just return zeros for the simulated lc
        # if we aren't simulating.
        sim_lc = np.zeros(num_detect_images)

    mjd = np.array([im.mjd for im in cutout_image_list])
    num_pre_transient_images = np.sum(mjd < diaobj.mjd_start)
    num_post_transient_images = np.sum(mjd > diaobj.mjd_end)


    lightcurve_model = campari_lightcurve_model(
            flux=flux, sigma_flux=sigma_flux, images=images, model_images=model_images,
            ra_grid=ra_grid, dec_grid=dec_grid, wgt_matrix=wgt_matrix,
            galaxy_only_model_images=galaxy_only_model_images,
            LSB=LSB, best_fit_model_values=X, sim_lc=sim_lc, image_list=image_list,
            cutout_image_list=cutout_image_list, galaxy_images=np.array(galaxy_images), noise_maps=np.array(noise_maps),
            diaobj=diaobj, object_type=object_type, pre_transient_images=num_pre_transient_images,
            post_transient_images=num_post_transient_images
        )

    return lightcurve_model

