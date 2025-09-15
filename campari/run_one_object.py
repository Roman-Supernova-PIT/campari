# Standard Library
import pathlib
import warnings

# Common Library
import galsim
import numpy as np
import scipy.sparse as sp
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from numpy.linalg import LinAlgError
from roman_imsim.utils import roman_utils

# SN-PIT
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger

from campari.data_construction import construct_images, prep_data_for_fit
from campari.model_building import construct_static_scene, construct_transient_scene, generate_guess, make_grid

# Campari
from campari.simulation import simulate_images
from campari.utils import banner, get_weights

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

r"""
Cole Meldorf 2024
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


def run_one_object(diaobj=None, object_type=None, image_list=None,
                   roman_path=None, sn_path=None, size=None, band=None, fetch_SED=None, sedlist=None,
                   use_real_images=None, use_roman=None, subtract_background=None,
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
    roman_bandpasses = galsim.roman.getBandpasses()

    num_total_images = len(image_list)
    transient_image_list = [a for a in image_list if a.mjd > diaobj.mjd_start and a.mjd < diaobj.mjd_end]
    num_detect_images = len(transient_image_list)

    if use_real_images:
        cutout_image_list, image_list = construct_images(image_list, diaobj, size,
                                                         subtract_background=subtract_background)

        # We didn't simulate anything, so set these simulation only vars to none.
        sim_galra = None
        sim_galdec = None
        galaxy_images = None
        noise_maps = None

    else:
        # Simulate the images of the SN and galaxy.
        banner("Simulating Images")
        sim_lc, util_ref, image_list, cutout_image_list, sim_galra, sim_galdec, galaxy_images, noise_maps = \
            simulate_images(image_list, diaobj,
                            sim_galaxy_scale, sim_galaxy_offset,
                            do_xshift, do_rotation, noise=noise,
                            use_roman=use_roman, roman_path=roman_path,
                            size=size,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            source_phot_ops=source_phot_ops,
                            mismatch_seds=mismatch_seds, base_pointing=base_pointing,
                            base_sca=base_sca)
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

    # Calculate the Confusion Metric

    # if use_real_images and object_type == "SN" and num_detect_images > 1:
    if False:  # This is a temporary fix to not calculate the confusion metric.
        object_x, object_y = image_list[0].get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
        # object_x and object_y are the exact coords of the SN in the SCA frame.
        # x and y are the pixels the image has been cut out on, and
        # hence must be ints. Before, I had object_x and object_y as SN coords in the cutout frame, hence this switch.
        # In snappl, centers of pixels occur at integers, so the center of the lower left pixel is (0,0).
        # Therefore, if you are at (0.2, 0.2), you are in the lower left pixel, but at (0.6, 0.6), you have
        # crossed into the next pixel, which is (1,1). So we need to round everything between -0.5 and 0.5 to 0,
        # and everything between 0.5 and 1.5 to 1, etc. This code below does that, and follows how snappl does it.
        # For more detail, see the docstring of get_stamp in the PSF class definition of snappl.
        x = int(np.floor(object_x + 0.5))
        y = int(np.floor(object_y + 0.5))
        pointing, sca = image_list[0].pointing, image_list[0].sca
        snx = x
        sny = y
        x = int(np.floor(x + 0.5))
        y = int(np.floor(y + 0.5))
        SNLogger.debug(f"x, y, snx, sny, {x, y, snx, sny}")
        psf_source_array = construct_transient_scene(x, y, pointing, sca,
                                                     stampsize=size,
                                                     x_center=object_x, y_center=object_y,
                                                     sed=sed)
        confusion_metric = np.dot(cutout_image_list[0].data.flatten(), psf_source_array)

        SNLogger.debug(f"Confusion Metric: {confusion_metric}")
    else:
        confusion_metric = 0
        SNLogger.debug("Confusion Metric not calculated")

    # Build the backgrounds loop
    for i, image in enumerate(image_list):
        # Passing in None for the PSF means we use the Roman PSF.
        drawing_psf = None if use_roman else airy
        pointing, sca = image.pointing, image.sca

        whole_sca_wcs = image.get_wcs()
        object_x, object_y = whole_sca_wcs.world_to_pixel(diaobj.ra, diaobj.dec)

        # Build the model for the background using the correct psf and the
        # grid we made in the previous section.

        # TODO: Put this in snappl
        if use_real_images:
            util_ref = roman_utils(config_file=pathlib.Path(Config.get().value
                                   ("photometry.campari.galsim.tds_file")),
                                   visit=pointing, sca=sca)

        # If no grid, we still need something that can be concatenated in the
        # linear algebra steps, so we initialize an empty array by default.
        background_model_array = np.empty((size**2, 0))
        SNLogger.debug("Constructing background model array for image " + str(i) + " ---------------")
        if grid_type != "none":
            background_model_array = \
                construct_static_scene(ra_grid, dec_grid,
                                       whole_sca_wcs,
                                       object_x, object_y, size, psf=drawing_psf,
                                       pixel=pixel,
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
        if sn_index >= 0:
            if use_roman:
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
                psf_source_array =\
                    construct_transient_scene(x, y, pointing, sca,
                                              stampsize=size, x_center=object_x,
                                              y_center=object_y, sed=sed,
                                              photOps=source_phot_ops, sca_wcs=whole_sca_wcs)
            else:
                # NOTE: cutout_wcs_list is not being included in the zip above because in a different branch
                # I am updating the simulations to use snappl image objects. That will be changed here once that
                # is done.
                stamp = galsim.Image(size, size, wcs=cutout_wcs_list[i])
                profile = galsim.DeltaFunction()*sed
                profile = profile.withFlux(1, roman_bandpasses[band])
                convolved = galsim.Convolve(profile, drawing_psf)
                psf_source_array =\
                    convolved.drawImage(roman_bandpasses[band],
                                        method=draw_method_for_non_roman_psf,
                                        image=stamp,
                                        wcs=cutout_wcs_list[i],
                                        center=(object_x, object_y),
                                        use_true_center=True,
                                        add_to_image=False)
                psf_source_array = psf_source_array.array.flatten()

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
        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix, x0=x0test, atol=1e-12,
                              btol=1e-12, iter_lim=300000, conlim=1e10)
        X, istop, itn, r1norm = lsqr[:4]
        SNLogger.debug(f"Stop Condition {istop}, iterations: {itn}," +
                       f"r1norm: {r1norm}")

    flux = X[-num_detect_images:] if num_detect_images > 0 else None

    inv_cov = psf_matrix.T @ np.diag(wgt_matrix) @ psf_matrix

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
            confusion_metric=confusion_metric, best_fit_model_values=X, sim_lc=sim_lc, image_list=image_list,
            cutout_image_list=cutout_image_list, galaxy_images=np.array(galaxy_images), noise_maps=np.array(noise_maps)
        )

    return lightcurve_model


class campari_lightcurve_model:
    """This class holds the output of the Campari pipeline for a single SNID."""

    def __init__(
        self,
        diaobj=None,
        flux=None,
        sigma_flux=None,
        images=None,
        model_images=None,
        image_list=None,
        cutout_image_list=None,
        ra_grid=None,
        dec_grid=None,
        wgt_matrix=None,
        confusion_metric=None,
        best_fit_model_values=None,
        sim_lc=None,
        galaxy_images=None,
        noise_maps=None,
        galaxy_only_model_images=None,
    ):
        """Initialize the Campari lightcurve model with the SNID and its properties.
        Parameters
        ----------
        diaobj: snappl.diaobject.DiaObject
            The DiaObject representing the transient.
        flux : np.ndarray
            The flux values for the lightcurve.
        sigma_flux : np.ndarray
            The uncertainties in the flux values.
        images : np.ndarray
            The image data used in the lightcurve analysis.
        model_images : np.ndarray
            The model images generated by Campari.
        image_list : list of snappl.image.Image objects
            list of images used in the lightcurve analysis.
        cutout_image_list : list of snappl.image.Image objects
            list of images used in the lightcurve analysis that have been cutout at transient location.
        ra_grid : np.ndarray
            The RA coordinates of the points used to construct the background model.
        dec_grid : np.ndarray
            The Dec coordinates of the points used to construct the background model.
        wgt_matrix : np.ndarray
            The weight matrix used in the lightcurve analysis.
        confusion_metric : np.ndarray
            The confusion metric for the images. Currently defined as the dot product of PSF rendered
            at the location of the transient and an image of the background galaxy. This is analogous to
            local surface brightness, so it is possible this will be replaced with local surface brightness
            in the future.
        best_fit_model_values : np.ndarray
            The best fit model values for the lightcurve. The last n values,
            where n is the number of images considered a transient detection,
            are the flux values for the transient. All other values are the
            flux values assigned to the points that make up the background model.
        cutout_wcs_list : list
            List of WCS objects for the cutouts used in the lightcurve analysis.
        sim_lc : pd.DataFrame
            The simulated lightcurve data, if applicable.
        """
        self.diaobj = diaobj
        self.flux = flux
        self.sigma_flux = sigma_flux
        self.images = images
        self.model_images = model_images
        self.image_list = image_list
        self.cutout_image_list = cutout_image_list
        self.ra_grid = ra_grid
        self.dec_grid = dec_grid
        self.wgt_matrix = wgt_matrix
        self.confusion_metric = confusion_metric
        self.best_fit_model_values = best_fit_model_values
        self.sim_lc = sim_lc
        self.galaxy_images = galaxy_images
        self.noise_maps = noise_maps
        self.galaxy_only_model_images = galaxy_only_model_images
