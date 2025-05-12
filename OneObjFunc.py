# TODO -- remove these next few lines!
# This needs to be set up in an environment
# where snappl is available.  This will happen "soon"
# Get Rob to fix all of this.  For now, this is a hack
# so you can work short term.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent/"extern/snappl"))
# End of lines that will go away once we do this right

import numpy as np
from astropy.io import fits
import pandas as pd
from roman_imsim.utils import roman_utils
import warnings
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import scipy.sparse as sp
from numpy.linalg import LinAlgError
import galsim
from AllASPFuncs import banner, fetchImages, save_lightcurve, \
                        build_lightcurve, build_lightcurve_sim, \
                        construct_psf_background, construct_psf_source, \
                        makeGrid, get_galsim_SED, getWeights, generateGuess, \
                        get_galsim_SED_list, prep_data_for_fit



from simulation import simulate_images
import yaml
import argparse
import os

from snappl.logger import Lager
from snappl.config import Config
from snappl.image import OpenUniverse2024FITSImage

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


def run_one_object(ID, num_total_imgs, num_detect_imgs, roman_path,
                   sn_path, size, band, fetch_SED, use_real_images, use_roman,
                   fit_background, turn_grid_off, adaptive_grid, npoints,
                   make_initial_guess, initial_flux_guess, weighting, method,
                   make_contour_grid, single_grid_point, pixel, source_phot_ops,
                   lc_start, lc_end, do_xshift, bg_gal_flux, do_rotation, airy,
                   mismatch_seds, deltafcn_profile, noise, check_perfection,
                   avoid_non_linearity):

    Lager.debug(f'ID: {ID}')
    psf_matrix = []
    sn_matrix = []
    cutout_wcs_list = []
    im_wcs_list = []
    # This is a catch for when I'm doing my own simulated WCS's
    util_ref = None
    percentiles = []
    roman_bandpasses = galsim.roman.getBandpasses()

    banner('Finding and Preparing Images')
    if use_real_images:
        # Find SN Info, find exposures containing it,
        # and load those as images.
        # TODO: Calculate peak MJD outside of the function
        images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, \
            exposures, object_type = fetchImages(num_total_imgs,
                                                 num_detect_imgs, ID,
                                                 sn_path, band, size,
                                                 fit_background,
                                                 roman_path,
                                                 lc_start=lc_start,
                                                 lc_end=lc_end)
        if len(exposures[~exposures['DETECTED']]) == 0 and object_type == 'SN':
            raise ValueError('No pre-detection images found in time range ' +
                             'provided, skipping this object.')

        if num_total_imgs != np.inf and len(exposures) != num_total_imgs:
            raise ValueError('Not Enough Exposures. ' +
                             f'Found {len(exposures)} out of' +
                             f'{num_total_imgs} requested')

        num_total_imgs = len(exposures)
        num_detect_imgs = len(exposures[exposures['DETECTED']])
        _ = f'Updating image numbers to {num_total_imgs} and {num_detect_imgs}'
        Lager.debug(_)

    else:
        # Simulate the images of the SN and galaxy.
        banner('Simulating Images')
        images, im_wcs_list, cutout_wcs_list, sim_lc, util_ref = \
            simulate_images(num_total_imgs, num_detect_imgs, ra, dec,
                            do_xshift, do_rotation, noise=noise,
                            use_roman=use_roman, roman_path=roman_path,
                            size=size, band=band,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            source_phot_ops=source_phot_ops,
                            mismatch_seds=mismatch_seds)
        object_type = 'SN'
        err = np.ones_like(images)

    sedlist = get_galsim_SED_list(ID, exposures, fetch_SED, object_type,
                                  sn_path)

    # Build the background grid
    if not turn_grid_off:
        if object_type == 'star':
            Lager.warning('For fitting stars, you probably dont want a grid.')
        ra_grid, dec_grid = makeGrid(adaptive_grid, images, size, ra, dec,
                                     cutout_wcs_list,
                                     single_grid_point=single_grid_point,
                                     percentiles=percentiles,
                                     npoints=npoints,
                                     makecontourGrid=make_contour_grid)
    else:
        ra_grid = np.array([])
        dec_grid = np.array([])

    # Using the images, hazard an initial guess.
    # The num_total_imgs - num_detect_imgs check is to ensure we have
    # pre-detection images. Otherwise, initializing the model guess does not
    # make sense.
    if make_initial_guess and num_total_imgs != num_detect_imgs:
        if num_detect_imgs != 0:
            x0test = generateGuess(images[:-num_detect_imgs], cutout_wcs_list,
                                   ra_grid, dec_grid)
            x0_vals_for_sne = np.full(num_total_imgs, initial_flux_guess)
            x0test = np.concatenate([x0test, x0_vals_for_sne], axis=0)
            print(x0test.shape)
            Lager.debug(f'setting initial guess to {initial_flux_guess}')
        else:
            x0test = generateGuess(images, cutout_wcs_list, ra_grid,
                                   dec_grid)

    else:
        x0test = None

    banner('Building Model')

    # Calculate the Confusion Metric

    confusion_metric = 0
    Lager.debug('Confusion Metric not calculated')

    if use_real_images and object_type == 'SN':
        sed = get_galsim_SED(ID, exposures, sn_path, fetch_SED=False)
        x, y = im_wcs_list[0].toImage(ra, dec, units='deg')
        snx, sny = cutout_wcs_list[0].toImage(snra, sndec, units='deg')
        pointing, SCA = exposures['Pointing'][0], exposures['SCA'][0]
        array = construct_psf_source(x, y, pointing, SCA, stampsize=size,
                                     x_center=snx, y_center=sny, sed=sed)
        confusion_metric = np.dot(images[0].flatten(), array)

        Lager.debug(f'Confusion Metric: {confusion_metric}')
    else:
        confusion_metric = 0
        Lager.debug('Confusion Metric not calculated')

    # Build the backgrounds loop
    # TODO: Zip all the things you index [i] on directly and loop over
    # them.
    for i in range(num_total_imgs):
        if use_roman:
            sim_psf = galsim.roman.getPSF(1, band, pupil_bin=8,
                                          wcs=cutout_wcs_list[i])
        else:
            sim_psf = airy

        x, y = im_wcs_list[i].toImage(ra, dec, units='deg')

        # Build the model for the background using the correct psf and the
        # grid we made in the previous section.

        # TODO: Put this in snappl
        if use_real_images:
            util_ref = roman_utils(config_file='./temp_tds.yaml',
                                   visit=exposures['Pointing'][i],
                                   sca=exposures['SCA'][i])

        # TODO: better name for array
        # TODO: Why is band here twice?
        array, bgpsf = construct_psf_background(ra_grid, dec_grid,
                                                cutout_wcs_list[i], x, y,
                                                size,
                                                roman_bandpasses[band],
                                                color=0.61, psf=sim_psf,
                                                pixel=pixel,
                                                include_photonOps=False,
                                                util_ref=util_ref,
                                                use_roman=use_roman,
                                                band=band)
        # TODO comment this

        if fit_background:
            for j in range(num_total_imgs):
                if i == j:
                    bg = np.ones(size**2).reshape(-1, 1)
                else:
                    bg = np.zeros(size**2).reshape(-1, 1)
                array = np.concatenate([array, bg], axis=1)

        # Add the array of the model points and the background (if using)
        # to the matrix of all components of the model.
        psf_matrix.append(array)

        # TODO make this not bad
        if num_detect_imgs != 0 and i >= num_total_imgs - num_detect_imgs:
            snx, sny = cutout_wcs_list[i].toImage(snra, sndec, units='deg')
            if use_roman:
                if use_real_images:
                    pointing = exposures['Pointing'][i]
                    SCA = exposures['SCA'][i]
                else:
                    pointing = 662
                    SCA = 11
                # sedlist is the length of the number of supernova
                # detection images. Therefore, when we iterate onto the
                # first supernova image, we want to be on the first element
                # of sedlist. Therefore, we subtract by the number of
                # predetection images: num_total_imgs - num_detect_imgs.
                sn_index = i - (num_total_imgs - num_detect_imgs)
                Lager.debug(f'Using SED #{sn_index}')
                sed = sedlist[sn_index]
                Lager.debug(f'x, y, snx, sny, {x, y, snx, sny}')
                array = construct_psf_source(x, y, pointing, SCA,
                                             stampsize=size, x_center=snx,
                                             y_center=sny, sed=sed,
                                             photOps=source_phot_ops)
            else:
                stamp = galsim.Image(size, size, wcs=cutout_wcs_list[i])
                profile = galsim.DeltaFunction()*sed
                profile = profile.withFlux(1, roman_bandpasses[band])
                convolved = galsim.Convolve(profile, sim_psf)
                array = convolved.drawImage(roman_bandpasses[band],
                                            method='no_pixel',
                                            image=stamp,
                                            wcs=cutout_wcs_list[i],
                                            center=(snx, sny),
                                            use_true_center=True,
                                            add_to_image=False)
                array = array.array.flatten()

            sn_matrix.append(array)

    banner('Lin Alg Section')
    psf_matrix = np.vstack(np.array(psf_matrix))
    Lager.debug(f'{psf_matrix.shape} psf matrix shape')

    # Add in the supernova images to the matrix in the appropriate location
    # so that it matches up with the image it represents.
    # All others should be zero.

    # Get the weights
    if weighting:
        wgt_matrix = getWeights(cutout_wcs_list, size, snra, sndec,
                                error=err)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[1])

    images, err, sn_matrix, wgt_matrix =\
        prep_data_for_fit(images, err, sn_matrix, wgt_matrix)

    # Calculate amount of the PSF cut out by setting a distance cap
    test_sn_matrix = np.copy(sn_matrix)
    test_sn_matrix[np.where(wgt_matrix == 0), :] = 0
    Lager.debug(f'SN PSF Norms Pre Distance Cut:{np.sum(sn_matrix, axis=0)}')
    Lager.debug(f'SN PSF Norms Post Distance Cut:{np.sum(test_sn_matrix, axis=0)}')

    # Combine the background model and the supernova model into one matrix.

    psf_matrix = np.hstack([psf_matrix, sn_matrix])

    banner('Solving Photometry')
    # These if statements can definitely be written more elegantly.
    if not make_initial_guess:
        x0test = np.zeros(psf_matrix.shape[1])

    if fit_background:
        x0test = np.concatenate([x0test, np.zeros(num_total_imgs)], axis=0)

    if method == 'lsqr':
        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix, x0=x0test, atol=1e-12,
                              btol=1e-12, iter_lim=300000, conlim=1e10)
        X, istop, itn, r1norm = lsqr[:4]
        Lager.debug(f'Stop Condition {istop}, iterations: {itn},' +
                    f'r1norm: {r1norm}')
    flux = X[-num_detect_imgs:]
    inv_cov = psf_matrix.T @ np.diag(wgt_matrix) @ psf_matrix
    Lager.debug(f'inv_cov shape: {inv_cov.shape}')
    Lager.debug(f'psf_matrix shape: {psf_matrix.shape}')
    Lager.debug(f'wgt_matrix shape: {wgt_matrix.shape}')
    try:
        cov = np.linalg.inv(inv_cov)
    except LinAlgError:
        cov = np.linalg.pinv(inv_cov)

    Lager.debug(f'cov diag: {np.diag(cov)[-num_detect_imgs:]}')
    sigma_flux = np.sqrt(np.diag(cov)[-num_detect_imgs:])
    Lager.debug(f'sigma flux: {sigma_flux}')

    # Using the values found in the fit, construct the model images.
    pred = X*psf_matrix
    sumimages = np.sum(pred, axis=1)

    # TODO: Move this to a separate function
    if check_perfection:
        if avoid_non_linearity:
            f = 1
        else:
            f = 5000
        if single_grid_point:
            X[0] = f
        else:
            X = np.zeros_like(X)
            X[106] = f

    if use_real_images:
        # Eventually I might completely separate out simulated SNe, though I
        # am hesitant to do that as I want them to be treated identically as
        # possible. In the meantime, just return zeros for the simulated lc
        # if we aren't simulating.
        sim_lc = np.zeros(num_detect_imgs)
    return flux, sigma_flux, images, sumimages, exposures, ra_grid, dec_grid, \
        wgt_matrix, confusion_metric, object_type, X, cutout_wcs_list, sim_lc