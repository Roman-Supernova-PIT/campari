import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils
from roman_imsim import *
import astropy.table as tb
import warnings
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import scipy.sparse as sp
from scipy.linalg import block_diag, lstsq
from numpy.linalg import LinAlgError
from astropy.nddata import Cutout2D
from coord import *
import requests
from astropy.table import Table
import os
import scipy
import time
import galsim
import sklearn
from sklearn import linear_model
from scipy.interpolate import RectBivariateSpline
import AllASPFuncs
from AllASPFuncs import *
from simulation import *
import yaml
import h5py
import argparse
import pickle

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


'''
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
    # Your main code logic goes here
    print("Running the main function")

    parser = argparse.ArgumentParser(description="Can overwrite config file")

    parser.add_argument('-b', '--band', type=str, required=True, help='filter')

    parser.add_argument('-s', '--SNID', type=int, required=True, help='Supernova ID')
    parser.add_argument('-t', '--testnum', type=int, required=True, help='Number of images to use')
    #TODO:change all instances of this variable to tot_images
    parser.add_argument('-d', '--detim', type=int, required=True, help='Number of images to use with SN detections')
    #TODO:change all instances of this variable to det_images

    parser.add_argument('-o', '--output_path', type=str, required=False,
                        help='relative output path')

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
    background_level = config['background_level']
    method = config['method']
    make_initial_guess = config['make_initial_guess']
    adaptive_grid = config['adaptive_grid']
    spline_grid = config['spline_grid']
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

    output_path = args.output_path

    roman_bandpasses = galsim.roman.getBandpasses()

    #PSF for when not using the Roman PSF:
    lam = 1293  # nm
    lam_over_diam = 0.11300864172775239   #This is the roman value
    airy = galsim.ChromaticOpticalPSF(lam, diam = 2.36, aberrations=galsim.roman.getPSF(1,band, pupil_bin = 1).aberrations)

    if detim == 0:
        supernova = 0
    else:
        d = np.linspace(5,20,detim)
        mags = -5 * np.exp(-d/10) + 6
        fluxes = 10**(mags)
        supernova = list(fluxes)   #This is my faux lightcurve, you can edit this if you want to test with a different lightcurve.

    if make_exact:
        assert single_grid_point
    if avoid_non_linearity:
        assert deltafcn_profile
    assert detim <= testnum
    if type(supernova) == 'list':
        assert len(supernova) == detim


    galsim.roman.roman_psfs._make_aperture.clear() #clear cache




    ################### Finding and Preparing Images Section #########

    if type(SNID) != 'list':
        SNID = [SNID]

    for ID in SNID:
        print('ID:', ID)
        #check if file exists
        #if os.path.exists(f'./results/{ID}_{band}_detections.csv'):
            #print('File exists, skipping')
            #continue

        #try:

        psf_matrix = []
        imagelist = []
        sn_matrix = []
        cutout_wcs_list = []
        im_wcs_list = []
        gridmade = False
        #This is a catch for when I'm doing my own simulated WCS's
        image = None
        util_ref = None
        percentiles = []
        imagelist = []
        sn_matrix = []
        cutout_wcs_list = []
        im_wcs_list = []

        if use_real_images:
            #Find SN Info, find exposures containig it, and load those as images.
            images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, exposures, object_type = fetchImages(testnum, detim, ID, sn_path, band, size, fit_background, roman_path)

            if len(exposures) != testnum:
                    print('Not enough exposures')
                    continue

        else:
            #Simulate the images of the SN and galaxy.
            ra, dec = 7.541534306163982, -44.219205940734625
            snra, sndec = ra, dec
            galra = ra + 1.5e-5
            galdec = dec + 1.5e-5
            images, im_wcs_list, cutout_wcs_list, psf_storage, sn_storage = simulateImages(testnum,detim,ra,dec,do_xshift,\
                do_rotation,supernova,noise = noise,use_roman=use_roman, roman_path = roman_path, size = size, band = band, \
                    deltafcn_profile = deltafcn_profile, input_psf = airy, bg_gal_flux = bg_gal_flux, source_phot_ops = source_phot_ops, mismatch_seds = mismatch_seds)


        if fetch_SED:
            assert use_real_images, 'Cannot fetch SED if not using OpenUniverse sims'
            sedlist = []
            for date in exposures['date'][exposures['DETECTED']]:
                print('Getting SED for date:', date)
                lam, flam = get_SED(ID, date, sn_path, obj_type = object_type)
                sed = galsim.SED(galsim.LookupTable(lam, flam, interpolant='linear'), wave_type='Angstrom', flux_type='fphotons')
                sedlist.append(sed)


        else:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                                        wave_type='nm', flux_type='fphotons')



        imlist = [images[i*size**2:(i+1)*size**2].reshape(size,size) for i in range(testnum)]

        #Build the background grid
        if not turn_grid_off:
            ra_grid, dec_grid = makeGrid(adaptive_grid, images,size,ra,dec,cutout_wcs_list, single_grid_point=single_grid_point, percentiles=percentiles, npoints = npoints, makecontourGrid = makecontourGrid)
        else:
            ra_grid = np.array([])
            dec_grid = np.array([])


        #Get the weights
        if weighting:
            wgt_matrix = getWeights(cutout_wcs_list,size,snra,sndec, error = None)

        #Using the images, hazard an initial guess.
        if make_initial_guess and testnum - detim != 0:
            if supernova != 0:
                x0test = generateGuess(imlist[:-detim], cutout_wcs_list, ra_grid, dec_grid)
                x0test = np.concatenate([x0test, np.full(testnum, 3000)], axis = 0)
                print('setting initial guess to 3000')
            else:
                x0test = generateGuess(imlist, cutout_wcs_list, ra_grid, dec_grid)

        else:
            x0test = None

        ############################################### Fitting Section ###############################################

        #Calculate the Confusion Metric
        if use_real_images and object_type == 'SN':
            x,y = im_wcs_list[0].toImage(ra,dec, units = 'deg')
            snx, sny = cutout_wcs_list[0].toImage(snra, sndec, units = 'deg')
            pointing, SCA = exposures['Pointing'][0], exposures['SCA'][0]
            array = construct_psf_source(x, y, pointing, SCA, \
                            stampsize = size, x_center = snx, y_center = sny, sed = sed)
            confusion_metric = np.dot(images[:size**2], array)
            print('Confusion Metric:', confusion_metric)
        else:
            confusion_metric = 0
            print('No confusion metric calculated')

        #Build the backgrounds loop

        for i in range(testnum):
            spinner = ['|', '/', '-', '\\']
            #print('Constructing Model ' + str(i) + '   ' + spinner[i%4], end = '\r')
            if use_roman:
                sim_psf = galsim.roman.getPSF(1,band, pupil_bin=8, wcs = cutout_wcs_list[i])
            else:
                sim_psf = airy

            x,y = im_wcs_list[i].toImage(ra,dec, units = 'deg')

            #Build the model for the background using the correct psf and the grid we made in the previous section.

            if use_real_images:
                util_ref = roman_utils(config_file='./temp_tds.yaml', visit = exposures['Pointing'][i], sca = exposures['SCA'][i])
            else:
                util_ref = roman_utils(config_file='./temp_tds.yaml', visit = 662, sca = 11)

            array, bgpsf = construct_psf_background(ra_grid, dec_grid, cutout_wcs_list[i],\
                x, y, size, roman_bandpasses[band], color=0.61, \
                    psf = sim_psf, pixel = pixel, include_photonOps = False, util_ref = util_ref, use_roman = use_roman, band = band)

            if fit_background:
                for j in range(testnum):
                    if i == j:
                        bg = np.ones(size**2).reshape(-1,1)
                    else:
                        bg = np.zeros(size**2).reshape(-1,1)
                    array = np.concatenate([array,bg], axis = 1)

            #Add the array of the model points and the background (if using) to the matrix of all components of the model.
            psf_matrix.append(array)

            #The if statements in this section could be written much more elegantly
            if supernova != 0 and i >= testnum - detim:
                snx, sny = cutout_wcs_list[i].toImage(snra, sndec, units = 'deg')
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
                    print(x,y,snx,sny)
                    array = construct_psf_source(x, y, pointing, SCA, \
                            stampsize = size, x_center = snx, y_center = sny, sed = sed, photOps = source_phot_ops)
                else:
                    stamp = galsim.Image(size,size,wcs=cutout_wcs_list[i])
                    profile = galsim.DeltaFunction()*sed
                    profile = profile.withFlux(1,roman_bandpasses[band])
                    convolved = galsim.Convolve(profile, sim_psf)
                    array = convolved.drawImage(roman_bandpasses[band], method='no_pixel', image = stamp, \
                                wcs = cutout_wcs_list[i], center = (snx, sny), \
                                    use_true_center = True, add_to_image = False).array.flatten()

                sn_matrix.append(array)


        psf_matrix = np.array(psf_matrix)
        psf_matrix = np.vstack(psf_matrix)
        matrix_list = []
        matrix_list.append(psf_matrix)
        psf_zeros = np.zeros((psf_matrix.shape[0], testnum))

        #Add in the supernova images to the matrix in the appropriate location so that it matches up with the image
        #it represent. All others should be zero.

        if supernova != 0:
            for i in range(detim):
                psf_zeros[
                    (testnum- detim + i) * size * size : (testnum - detim + i + 1) * size * size, (testnum - detim) + i
                ] = sn_matrix[i]
            sn_matrix =psf_zeros
            sn_matrix = np.array(sn_matrix)
            sn_matrix = np.vstack(sn_matrix)
            matrix_list.append(sn_matrix)


        #Combine the background model and the supernova model into one matrix.
        psf_matrix_all = np.hstack(matrix_list)
        psf_matrix = psf_matrix_all

        if weighting:
            wgt_matrix = np.array(wgt_matrix)
            wgt_matrix = np.hstack(wgt_matrix)



        #These if statements can definitely be written more elegantly.
        if not make_initial_guess:
            x0test = np.zeros(psf_matrix.shape[1])

        if fit_background:
            x0test = np.concatenate([x0test, np.zeros(testnum)], axis = 0)

        if not weighting:
            wgt_matrix = np.ones(psf_matrix.shape[1])

        #
        if method == 'lsqr':
            lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1,1), images*wgt_matrix, \
                            x0 = x0test, atol = 1e-12, btol = 1e-12, iter_lim=300000, conlim = 1e10)

            X, istop, itn, r1norm = lsqr[:4]
            print(istop, itn, r1norm)


        #Using the values found in the fit, construct the model images.
        pred = X*psf_matrix
        sumimages = np.sum(pred, axis = 1)
        res = sumimages - images
        true_mags = -2.5*np.log10(supernova) + 14
        model_mags = -2.5*np.log10(X[-detim:]) + 14
        res = true_mags - model_mags


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


        #Saving the output. The output needs two sections, one where we create a lightcurve compared to true values, and one where we save the images.
        if use_real_images:
            identifier = str(ID)
            lc = build_lightcurve(ID, exposures, sn_path, confusion_metric, detim, X, use_roman, band, object_type)
        else:
            identifier = 'simulated'
            lc = build_lightcurve_sim(supernova, detim, X)

        if use_roman:
            psftype = 'romanpsf'
        else:
            psftype = 'analyticpsf'

        save_lightcurve(lc, identifier, band, psftype,
                        output_path=output_path)

        #Now, save the images
        images_and_model = np.array([images, sumimages, wgt_matrix])
        print('Saving images to ./results/images/'+ f'{identifier}_{band}_{psftype}_images.npy')
        np.save(f'./results/images/{identifier}_{band}_{psftype}_images.npy', images_and_model)

        #Save the ra and decgrid
        np.save(f'./results/images/{identifier}_{band}_{psftype}_grid.npy', [ra_grid, dec_grid, X[:np.size(ra_grid)]])


        #save wcses
        primary_hdu = fits.PrimaryHDU()
        hdul = [primary_hdu]
        for i, galsimwcs in enumerate(cutout_wcs_list):
            hdul.append(fits.ImageHDU(header=galsimwcs.wcs.to_header(), name="WCS" + str(i)))
        hdul = fits.HDUList(hdul)
        hdul.writeto(f'./results/images/{identifier}_{band}_{psftype}_wcs.fits', overwrite = True)


        '''
        except Exception as e:
            print('Failed on ID:', ID)
            print(e)
            continue
        '''

    #Run method, main, main parses arguments
    #If block or for block longer than 5 lines? --> function
    #Save un-run notebook
    #git .ignore
    #tests directory
    #Tests running a function and see if it returns the expected value
    #Make big chunks of code into functions
    #Export as python


if __name__ == "__main__":
    main()