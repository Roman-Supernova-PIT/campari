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
                        makeGrid, get_SED, getWeights, generateGuess
from simulation import simulate_images
import yaml
import argparse
from galsim import roman

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)

r'''
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


'''


config_path = './config.yaml'


def load_config(config_path):
    """Load parameters from a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():

    parser = argparse.ArgumentParser(description="Can overwrite config file")

    parser.add_argument('-b', '--band', type=str, required=True, help='filter')
    parser.add_argument('-s', '--SNID', type=int, required=True,
                        help='Supernova ID')
    parser.add_argument('-t', '--testnum', type=int, required=True,
                        help='Number of images to use')
    # TODO:change all instances of this variable to tot_images
    parser.add_argument('-d', '--detim', type=int, required=True,
                        help='Number of images to use with SN detections')
    # TODO:change all instances of this variable to det_images

    config = load_config(config_path)

    npoints = config['npoints']
    size = config['size']
    use_real_images = config['use_real_images']
    use_roman = config['use_roman']
    check_perfection = config['check_perfection']
    make_exact = config['make_exact']
    avoid_non_linearity = config['avoid_non_linearity']
    deltafcn_profile = config['deltafcn_profile']
    single_grid_point = config['single_grid_point']
    do_xshift = config['do_xshift']
    do_rotation = config['do_rotation']
    noise = config['noise']
    method = config['method']
    make_initial_guess = config['make_initial_guess']
    adaptive_grid = config['adaptive_grid']
    fit_background = config['fit_background']
    weighting = config['weighting']
    pixel = config['pixel']
    roman_path = config['roman_path']
    sn_path = config['sn_path']
    turn_grid_off = config['turn_grid_off']
    bg_gal_flux = config['bg_gal_flux']
    source_phot_ops = config['source_phot_ops']
    mismatch_seds = config['mismatch_seds']
    fetch_SED = config['fetch_SED']
    makecontourGrid = config['makecontourGrid']

    args = parser.parse_args()
    band = args.band
    SNID = args.SNID
    testnum = args.testnum
    detim = args.detim

    roman_bandpasses = galsim.roman.getBandpasses()

    # PSF for when not using the Roman PSF:
    lam = 1293  # nm
    aberrations = galsim.roman.getPSF(1, band, pupil_bin=1).aberrations
    airy = galsim.ChromaticOpticalPSF(lam, diam=2.36,
                                      aberrations=aberrations)

    if detim == 0:
        supernova = 0
    else:
        d = np.linspace(5, 20, detim)
        mags = -5 * np.exp(-d/10) + 6
        fluxes = 10**(mags)
        supernova = list(fluxes)
    if make_exact:
        assert single_grid_point
    if avoid_non_linearity:
        assert deltafcn_profile
    assert detim <= testnum
    if isinstance(supernova, list):
        assert len(supernova) == detim

    galsim.roman.roman_psfs._make_aperture.clear()  # clear cache

    banner('Finding and Preparing Images')

    if not isinstance(SNID, list):
        SNID = [SNID]

    for ID in SNID:
        print('ID:', ID)

        psf_matrix = []
        sn_matrix = []
        cutout_wcs_list = []
        im_wcs_list = []
        # This is a catch for when I'm doing my own simulated WCS's
        util_ref = None
        percentiles = []
        cutout_wcs_list = []
        im_wcs_list = []

        if use_real_images:
            # Find SN Info, find exposures containing it,
            # and load those as images.
            images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, \
                exposures, object_type = fetchImages(testnum, detim, ID,
                                                     sn_path, band, size,
                                                     fit_background,
                                                     roman_path)

            if len(exposures) != testnum:
                print('Not enough exposures')
                continue

        else:
            # Simulate the images of the SN and galaxy.
            ra, dec = 7.541534306163982, -44.219205940734625
            snra, sndec = ra, dec
            images, im_wcs_list, cutout_wcs_list, psf_storage, sn_storage = \
                simulate_images(testnum, detim, ra, dec, do_xshift,
                                do_rotation, supernova, noise=noise,
                                use_roman=use_roman, roman_path=roman_path,
                                size=size, band=band,
                                deltafcn_profile=deltafcn_profile,
                                input_psf=airy, bg_gal_flux=bg_gal_flux,
                                source_phot_ops=source_phot_ops,
                                mismatch_seds=mismatch_seds)

        if fetch_SED:
            assert use_real_images, 'Cannot fetch SED if not using \
                                     OpenUniverse sims'
            sedlist = []
            for date in exposures['date'][exposures['DETECTED']]:
                print('Getting SED for date:', date)
                lam, flam = get_SED(ID, date, sn_path, obj_type=object_type)
                sed = galsim.SED(galsim.LookupTable(lam, flam,
                                                    interpolant='linear'),
                                 wave_type='Angstrom', flux_type='fphotons')
                sedlist.append(sed)

        else:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                                                interpolant='linear'),
                             wave_type='nm', flux_type='fphotons')

        imlist = [images[i*size**2:(i+1)*size**2].reshape(size, size)
                  for i in range(testnum)]

        # Build the background grid
        if not turn_grid_off:
            ra_grid, dec_grid = makeGrid(adaptive_grid, images, size, ra, dec,
                                         cutout_wcs_list,
                                         single_grid_point=single_grid_point,
                                         percentiles=percentiles,
                                         npoints=npoints,
                                         makecontourGrid=makecontourGrid)
        else:
            ra_grid = np.array([])
            dec_grid = np.array([])

        # Get the weights
        if weighting:
            wgt_matrix = getWeights(cutout_wcs_list, size, snra, sndec,
                                    error=None)

        # Using the images, hazard an initial guess.
        if make_initial_guess and testnum - detim != 0:
            if supernova != 0:
                x0test = generateGuess(imlist[:-detim], cutout_wcs_list,
                                       ra_grid, dec_grid)
                x0test = np.concatenate([x0test, np.full(testnum, 3000)],
                                        axis=0)
                print('setting initial guess to 3000')
            else:
                x0test = generateGuess(imlist, cutout_wcs_list, ra_grid,
                                       dec_grid)

        else:
            x0test = None

        banner('Building Model')

        # Calculate the Confusion Metric
        if use_real_images and object_type == 'SN':
            x, y = im_wcs_list[0].toImage(ra, dec, units='deg')
            snx, sny = cutout_wcs_list[0].toImage(snra, sndec, units='deg')
            pointing, SCA = exposures['Pointing'][0], exposures['SCA'][0]
            array = construct_psf_source(x, y, pointing, SCA, stampsize=size,
                                         x_center=snx, y_center=sny, sed=sed)
            confusion_metric = np.dot(images[:size**2], array)
            print('Confusion Metric:', confusion_metric)
        else:
            confusion_metric = 0
            print('No confusion metric calculated')

        # Build the backgrounds loop

        for i in range(testnum):
            if use_roman:
                sim_psf = galsim.roman.getPSF(1, band, pupil_bin=8,
                                              wcs=cutout_wcs_list[i])
            else:
                sim_psf = airy

            x, y = im_wcs_list[i].toImage(ra, dec, units='deg')

            # Build the model for the background using the correct psf and the
            # grid we made in the previous section.

            if use_real_images:
                util_ref = roman_utils(config_file='./temp_tds.yaml',
                                       visit=exposures['Pointing'][i],
                                       sca=exposures['SCA'][i])
            else:
                util_ref = roman_utils(config_file='./temp_tds.yaml',
                                       visit=662, sca=11)

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

            if fit_background:
                for j in range(testnum):
                    if i == j:
                        bg = np.ones(size**2).reshape(-1, 1)
                    else:
                        bg = np.zeros(size**2).reshape(-1, 1)
                    array = np.concatenate([array, bg], axis=1)

            # Add the array of the model points and the background (if using)
            # to the matrix of all components of the model.
            psf_matrix.append(array)

            if supernova != 0 and i >= testnum - detim:
                snx, sny = cutout_wcs_list[i].toImage(snra, sndec, units='deg')
                if use_roman:
                    if use_real_images:
                        pointing = exposures['Pointing'][i]
                        SCA = exposures['SCA'][i]
                    else:
                        pointing = 662
                        SCA = 11
                    if fetch_SED:
                        print('Using SED #', i - (testnum - detim))
                        sed = sedlist[i - (testnum - detim)]
                    else:
                        print('Using default SED')
                    print(x, y, snx, sny)
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

        psf_matrix = np.array(psf_matrix)
        psf_matrix = np.vstack(psf_matrix)
        print(psf_matrix.shape, 'psf matrix shape')
        matrix_list = []
        matrix_list.append(psf_matrix)
        psf_zeros = np.zeros((psf_matrix.shape[0], testnum))

        # Add in the supernova images to the matrix in the appropriate location
        # so that it matches up with the image it represents.
        # All others should be zero.

        if supernova != 0:
            for i in range(detim):
                psf_zeros[
                    (testnum - detim + i) * size * size:
                    (testnum - detim + i + 1) * size * size,
                    (testnum - detim) + i] = sn_matrix[i]
            sn_matrix = psf_zeros
            sn_matrix = np.array(sn_matrix)
            sn_matrix = np.vstack(sn_matrix)
            matrix_list.append(sn_matrix)

        print(sn_matrix)
        print(sn_matrix.shape, 'sn matrix shape')

        # Combine the background model and the supernova model into one matrix.
        psf_matrix_all = np.hstack(matrix_list)

        print(psf_matrix_all.shape, 'psf matrix all shape')
        print(psf_matrix_all)
        psf_matrix = psf_matrix_all

        if weighting:
            wgt_matrix = np.array(wgt_matrix)
            wgt_matrix = np.hstack(wgt_matrix)

        banner('Solving Photometry')
        # These if statements can definitely be written more elegantly.
        if not make_initial_guess:
            x0test = np.zeros(psf_matrix.shape[1])

        if fit_background:
            x0test = np.concatenate([x0test, np.zeros(testnum)], axis=0)

        if not weighting:
            wgt_matrix = np.ones(psf_matrix.shape[1])

        #
        if method == 'lsqr':
            lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                                  images*wgt_matrix, x0=x0test, atol=1e-12,
                                  btol=1e-12, iter_lim=300000, conlim=1e10)
            X, istop, itn, r1norm = lsqr[:4]
            print(istop, itn, r1norm)

        exptime = {'F184': 901.175,
                   'J129': 302.275,
                   'H158': 302.275,
                   'K213': 901.175,
                   'R062': 161.025,
                   'Y106': 302.275,
                   'Z087': 101.7}

        flux = X[-detim:]
        area_eff = roman.collecting_area
        galsim_zp = roman.getBandpasses()[band].zeropoint
        mags = -2.5*np.log10(flux) + 2.5*np.log10(exptime[band]*area_eff) \
            + galsim_zp

        print('Measured mags:', mags)

        inv_cov = psf_matrix.T @ np.diag(wgt_matrix) @ psf_matrix
        print(np.shape(inv_cov), 'inv cov shape')
        print(np.shape(wgt_matrix), 'wgt shape')
        print(np.shape(psf_matrix), 'psf shape')
        try:
            cov = np.linalg.inv(inv_cov)
        except LinAlgError:
            cov = np.linalg.pinv(inv_cov)

        print(np.shape(cov), 'cov shape')
        sigma_flux = np.sqrt(np.diag(cov))[-detim:]
        print('sigma flux', sigma_flux)

        # Using the values found in the fit, construct the model images.
        pred = X*psf_matrix
        sumimages = np.sum(pred, axis=1)

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

        # Saving the output. The output needs two sections, one where we
        # create a lightcurve compared to true values, and one where we save
        # the images.

        if use_real_images:
            identifier = str(ID)
            lc = build_lightcurve(ID, exposures, sn_path, confusion_metric,
                                  flux, use_roman, band, object_type,
                                  sigma_flux)
        else:
            identifier = 'simulated'
            lc = build_lightcurve_sim(supernova, flux, sigma_flux)
        if use_roman:
            psftype = 'romanpsf'
        else:
            psftype = 'analyticpsf'

        save_lightcurve(lc, identifier, band, psftype)

        # Now, save the images
        images_and_model = np.array([images, sumimages, wgt_matrix])
        print('Saving images to ./results/images/' +
              f'{identifier}_{band}_{psftype}_images.npy')
        np.save(f'./results/images/{identifier}_{band}_{psftype}_images.npy',
                images_and_model)

        # Save the ra and decgrid
        np.save(f'./results/images/{identifier}_{band}_{psftype}_grid.npy',
                [ra_grid, dec_grid, X[:np.size(ra_grid)]])

        # save wcses
        primary_hdu = fits.PrimaryHDU()
        hdul = [primary_hdu]
        for i, galsimwcs in enumerate(cutout_wcs_list):
            hdul.append(fits.ImageHDU(header=galsimwcs.wcs.to_header(),
                        name="WCS" + str(i)))
        hdul = fits.HDUList(hdul)
        filepath = f'./results/images/{identifier}_{band}_{psftype}_wcs.fits'
        hdul.writeto(filepath, overwrite=True)

        '''
        except Exception as e:
            print('Failed on ID:', ID)
            print(e)
            continue
        '''


if __name__ == "__main__":
    main()
