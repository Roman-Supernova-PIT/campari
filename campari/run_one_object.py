# Standard Library
import pathlib
import warnings

# Common Library

import numpy as np
from numpy.linalg import LinAlgError
import multiprocessing
import scipy.sparse as sp
import sys

# Astronomy Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from campari.data_construction import construct_images, prep_data_for_fit
from campari.model_building import (
    prep_initial_guess,
    make_grid,
    build_model_for_one_image,
)
from campari.plotting import plot_cutouts_if_requested
from campari.utils import (banner, calculate_local_surface_brightness, campari_lightcurve_model,
                           convert_band_name, get_weights, print_mem,
                           load_prebuilt_matrices_if_provided)
from campari.io import save_model_if_requested
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


def _build_model_for_one_image_worker(index, kwarg_dict):
    image = _shared_image_list[index]
    return build_model_for_one_image(image=image, image_index=index, **kwarg_dict)


# Global variables
huge_value = 1e32
SNLogger.set_level("DEBUG")


def run_one_object(diaobj=None, object_type=None, image_list=None, size=None, band=None, fetch_SED=None, sedlist=None,
                   subtract_background_method=None,
                   make_initial_guess=None, initial_flux_guess=None, weighting=None, method=None,
                   grid_type=None, pixel=None, do_xshift=None, bg_gal_flux=None, do_rotation=None,
                   mismatch_seds=None, deltafcn_profile=None, noise=None,
                   avoid_non_linearity=None, spacing=None, percentiles=None,
                   save_model=False, prebuilt_psf_matrix=None,
                   prebuilt_sn_matrix=None, gaussian_var=None,
                   cutoff=None, error_floor=None, subsize=None,
                   nprocs=None):
    """ Run campari on one object."""
    psf_matrix = []
    sn_matrix = []

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

    all_sca_xs, all_sca_ys = \
        map(list, zip(*[img.get_wcs().world_to_pixel(diaobj.ra, diaobj.dec) for img in image_list]))

    # We switched from using different lettered (R062, Y106) bands to F + number bands (F062, F106) in the code at
    # some point, so this catches those cases.
    band = convert_band_name(band)

    cutout_image_list, image_list, sky_background = construct_images(image_list, diaobj, size,
                                                                     subtract_background_method=
                                                                     subtract_background_method,
                                                                     nprocs=nprocs)
    # del image_list  # Save memory
    noise_maps = [im.noise for im in cutout_image_list]

    plot_cutouts_if_requested(cutout_image_list, diaobj.ra, diaobj.dec, diaobj=diaobj,
                     output_path=pathlib.Path(Config.get().value("photometry.campari_io.debug_dir")) /
                     f"cutouts_{diaobj.name}.png")

    print_mem("After constructing images:")

    # Build the background grid
    ra_grid, dec_grid = make_grid(grid_type, cutout_image_list, diaobj.ra, diaobj.dec,
                                      percentiles=percentiles, spacing=spacing,
                                      subsize=subsize, object_type=object_type)

    # The num_total_images - num_detect_images check is to ensure we have
    # pre-detection images. Otherwise, initializing the model guess does not
    # make sense.
    num_nondetect_images = num_total_images - num_detect_images

    banner("Building Model")

    no_transient_cutouts = [a for a in cutout_image_list if a.mjd < diaobj.mjd_start or a.mjd > diaobj.mjd_end]
    if len(no_transient_cutouts) > 0:
        LSB = calculate_local_surface_brightness(no_transient_cutouts, cutout_pix=2)
    else:
        # This is used for stars only, essentially. LSB just can't be None.
        LSB = calculate_local_surface_brightness(cutout_image_list, cutout_pix=2)

    # Build the backgrounds loop
    model_results = []
    kwarg_dict = {"ra": diaobj.ra, "dec": diaobj.dec, "grid_type": grid_type,
                  "ra_grid": ra_grid, "dec_grid": dec_grid, "size": size, "pixel": pixel,
                  "band": band,
                  "sedlist": sedlist,
                  "num_total_images": num_total_images,
                  "num_detect_images": num_detect_images, "prebuilt_psf_matrix": prebuilt_psf_matrix,
                  "prebuilt_sn_matrix": prebuilt_sn_matrix, "subtract_background_method": subtract_background_method}

    if nprocs > 1:
        SNLogger.debug(f"Using {nprocs} processes for model building")
        global _shared_image_list
        _shared_image_list = image_list
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(nprocs) as pool:
            for i, image in enumerate(image_list):
                model_results.append(pool.apply_async(_build_model_for_one_image_worker,
                                                      args=(i, kwarg_dict)))
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

    # Load prebuilt matrices if provided, otherwise stack the matrices we just built.
    psf_matrix, sn_matrix = load_prebuilt_matrices_if_provided(prebuilt_psf_matrix,
                                                               prebuilt_sn_matrix, psf_matrix, sn_matrix)

    # Get the weights. If weighting is false, this will return a list of arrays of ones,
    #  which is equivalent to no weighting.
    wgt_matrix = get_weights(cutout_image_list, diaobj.ra, diaobj.dec, weighting, gaussian_var=gaussian_var,
                                 cutoff=cutoff, error_floor=error_floor)

    galaxy_psfclass = Config.get().value("photometry.campari.psf.galaxy_class")
    sn_psfclass = Config.get().value("photometry.campari.psf.transient_class")

    save_model_if_requested(save_model, psf_matrix, sn_matrix, galaxy_psfclass, sn_psfclass, diaobj, num_total_images)

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

    # Using the images, hazard an initial guess.
    x0 = prep_initial_guess(make_initial_guess, num_nondetect_images, grid_type, cutout_image_list, ra_grid, dec_grid,
                        num_total_images, initial_flux_guess, psf_matrix, subtract_background_method)

    SNLogger.debug(f"shape psf_matrix: {psf_matrix.shape}")
    SNLogger.debug(f"psf matrix size: {sys.getsizeof(psf_matrix) / 1e6:.4f} MB")
    SNLogger.debug(f"shape wgt_matrix: {wgt_matrix.reshape(-1, 1).shape}")
    SNLogger.debug(f"wgt matrix size: {sys.getsizeof(wgt_matrix) / 1e6:.4f} MB")
    SNLogger.debug(f"image shape: {images.shape}")
    SNLogger.debug(f"images size: {sys.getsizeof(images) / 1e6:.4f} MB")

    wgt_matrix = np.sqrt(wgt_matrix)
    lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                            images*wgt_matrix,  atol=1e-12, x0=x0,
                            btol=1e-12, iter_lim=300000, conlim=1e10)
    X, istop, itn, r1norm = lsqr[:4]
    SNLogger.debug(f"Stop Condition {istop}, iterations: {itn}," +
                    f"r1norm: {r1norm}")

    flux = X[-num_detect_images:] if num_detect_images > 0 else None

    w2 = wgt_matrix ** 2
    inv_cov = (psf_matrix * w2[:, np.newaxis]).T @ psf_matrix

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

    lightcurve_model = campari_lightcurve_model(
            flux=flux, sigma_flux=sigma_flux, images=images, model_images=model_images,
            ra_grid=ra_grid, dec_grid=dec_grid, wgt_matrix=wgt_matrix,
            galaxy_only_model_images=galaxy_only_model_images,
            LSB=LSB, best_fit_model_values=X, sca_x_locations = all_sca_xs,
            sca_y_locations = all_sca_ys,
            cutout_image_list=cutout_image_list, noise_maps=np.array(noise_maps),
            diaobj=diaobj, object_type=object_type, sky_background=sky_background,
            pre_transient_images=num_pre_transient_images,
            post_transient_images=num_post_transient_images
        )

    return lightcurve_model
