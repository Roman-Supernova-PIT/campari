# Standard Library
import pathlib
import warnings

# Common Library

import numpy as np
from numpy.linalg import LinAlgError
import multiprocessing as mp
from multiprocessing import Pool
import scipy.sparse as sp

# Astronomy Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from campari.data_construction import construct_images, prep_data_for_fit
from campari.model_building import (
    generate_guess,
    make_grid,
    build_model_for_one_image,
)
from campari.simulation import simulate_images
from campari.utils import banner, calculate_local_surface_brightness, campari_lightcurve_model, get_weights
from snappl.config import Config
from snappl.logger import SNLogger

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
                   use_real_images=None, subtract_background_method=None,
                   make_initial_guess=None, initial_flux_guess=None, weighting=None, method=None,
                   grid_type=None, pixel=None, do_xshift=None, bg_gal_flux=None, do_rotation=None,
                   airy=None, mismatch_seds=None, deltafcn_profile=None, noise=None,
                   avoid_non_linearity=None, spacing=None, percentiles=None, sim_galaxy_scale=1,
                   sim_galaxy_offset=None, base_pointing=662, base_sca=11,
                   save_model=False, prebuilt_psf_matrix=None,
                   prebuilt_sn_matrix=None, gaussian_var=None,
                   cutoff=None, error_floor=None, subsize=None,
                   nprocs=None):
    psf_matrix = []
    sn_matrix = []

    # This is a catch for when I'm doing my own simulated WCSs
    util_ref = None

    percentiles = []

    num_total_images = len(image_list)
    transient_image_list = [a for a in image_list if a.mjd >= diaobj.mjd_start and a.mjd <= diaobj.mjd_end]
    num_detect_images = len(transient_image_list)

    no_transient_images = [a for a in image_list if a.mjd < diaobj.mjd_start or a.mjd > diaobj.mjd_end]

    transient_mjds = [a.mjd for a in transient_image_list]
    no_transient_mjds = [a.mjd for a in no_transient_images]
    transient_argsort = np.argsort(transient_mjds)
    no_transient_argsort = np.argsort(no_transient_mjds)

    transient_image_list = [transient_image_list[i] for i in transient_argsort]
    no_transient_images = [no_transient_images[i] for i in no_transient_argsort]

    image_list = no_transient_images + transient_image_list  # Non detection images first, then detection images,
    # but still sorted by MJD.

    if use_real_images:
        cutout_image_list, image_list, sky_background = construct_images(image_list, diaobj, size,
                                                                         subtract_background_method=
                                                                         subtract_background_method,
                                                                         nprocs=nprocs)
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
                            size=size,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            mismatch_seds=mismatch_seds, base_pointing=base_pointing,
                            base_sca=base_sca)
        sim_lc = simulated_lightcurve.sim_lc
        sky_background = np.zeros(len(sim_lc))
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
                                      single_dec=sim_galdec, spacing=spacing,
                                      subsize=subsize)
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
        # This is used for stars only, essentially. LSB just can't be None.
        LSB = calculate_local_surface_brightness(cutout_image_list, cutout_pix=2)

    # Build the backgrounds loop
    model_results = []
    kwarg_dict = {"ra": diaobj.ra, "dec": diaobj.dec, "use_real_images": use_real_images, "grid_type": grid_type,
                  "ra_grid": ra_grid, "dec_grid": dec_grid, "size": size, "pixel": pixel,
                  "band": band,
                  "sedlist": sedlist,
                  "num_total_images": num_total_images,
                  "num_detect_images": num_detect_images, "prebuilt_psf_matrix": prebuilt_psf_matrix,
                  "prebuilt_sn_matrix": prebuilt_sn_matrix, "subtract_background_method": subtract_background_method,
                  "base_pointing": base_pointing, "base_sca": base_sca}
    if nprocs > 1:
        #mp.set_start_method("spawn")
        SNLogger.debug(f"Using {nprocs} processes for model building")
        with Pool(nprocs) as pool:
            for i, image in enumerate(image_list):
                model_results.append(pool.apply_async(build_model_for_one_image,
                                                      kwds={"image": image, "image_index": i, **kwarg_dict}))
            pool.close()
            pool.join()

    else:
        for i, image in enumerate(image_list):
            model_results.append(build_model_for_one_image(**{"image": image, "image_index": i, **kwarg_dict}))

    for result in model_results:
        if nprocs > 1:
            bg_model, transient_model = result.get()
        else:
            bg_model, transient_model = result
        psf_matrix.append(bg_model)
        if transient_model is not None:
            sn_matrix.append(transient_model)

    banner("Lin Alg Section")
    if prebuilt_psf_matrix is None:
        psf_matrix = np.vstack(np.array(psf_matrix))
        SNLogger.debug(f"{psf_matrix.shape} psf matrix shape")
    else:
        psf_matrix = prebuilt_psf_matrix
        SNLogger.debug(f"Using prebuilt PSF matrix of shape {psf_matrix.shape}")

    if prebuilt_sn_matrix is not None:
        sn_matrix = prebuilt_sn_matrix
        SNLogger.debug(f"Using prebuilt SN matrix of shape {sn_matrix.shape}")

    # Add in the supernova images to the matrix in the appropriate location
    # so that it matches up with the image it represents.
    # All others should be zero.

    # Get the weights

    if weighting:
        wgt_matrix = get_weights(cutout_image_list, diaobj.ra, diaobj.dec, gaussian_var=gaussian_var,
                                 cutoff=cutoff, error_floor=error_floor)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[0])

    galaxy_psfclass = Config.get().value("photometry.campari.psf.galaxy_class")
    sn_psfclass = Config.get().value("photometry.campari.psf.transient_class")

    if save_model:
        psf_matrix_path = pathlib.Path(Config.get().value("system.paths.debug_dir")) \
            / f"psf_matrix_{galaxy_psfclass}_{diaobj.id}_{num_total_images}_images{psf_matrix.shape[1]}_points.npy"
        np.save(psf_matrix_path, psf_matrix)

        sn_matrix_path = pathlib.Path(Config.get().value("system.paths.debug_dir")) \
            / f"sn_matrix_{sn_psfclass}_{diaobj.id}_{num_total_images}_images.npy"
        np.save(sn_matrix_path, sn_matrix)

        SNLogger.debug(f"Saved PSF matrix to {psf_matrix_path}")
        SNLogger.debug(f"Saved SN matrix to {sn_matrix_path}")

    images, err, sn_matrix, wgt_matrix =\
        prep_data_for_fit(cutout_image_list, sn_matrix, wgt_matrix, diaobj)
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

    mjd = np.array([im.mjd for im in cutout_image_list])
    num_pre_transient_images = np.sum(mjd < diaobj.mjd_start)
    num_post_transient_images = np.sum(mjd > diaobj.mjd_end)

    # These if statements can definitely be written more elegantly.
    if not make_initial_guess:
        x0test = np.zeros(psf_matrix.shape[1])

    if subtract_background_method == "fit":
        x0test = np.concatenate([x0test, np.zeros(num_total_images)], axis=0)

    SNLogger.debug(f"shape psf_matrix: {psf_matrix.shape}")
    SNLogger.debug(f"shape wgt_matrix: {wgt_matrix.reshape(-1, 1).shape}")
    SNLogger.debug(f"image shape: {images.shape}")

    if method == "lsqr":

        wgt_matrix = np.sqrt(wgt_matrix)
        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix,  atol=1e-12, x0=x0test,
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

    lightcurve_model = campari_lightcurve_model(
            flux=flux, sigma_flux=sigma_flux, images=images, model_images=model_images,
            ra_grid=ra_grid, dec_grid=dec_grid, wgt_matrix=wgt_matrix,
            galaxy_only_model_images=galaxy_only_model_images,
            LSB=LSB, best_fit_model_values=X, sim_lc=sim_lc, image_list=image_list,
            cutout_image_list=cutout_image_list, galaxy_images=np.array(galaxy_images), noise_maps=np.array(noise_maps),
            diaobj=diaobj, object_type=object_type, sky_background=sky_background,
            pre_transient_images=num_pre_transient_images,
            post_transient_images=num_post_transient_images
        )

    return lightcurve_model
