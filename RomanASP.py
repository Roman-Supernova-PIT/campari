# TODO -- remove these next few lines!
# This needs to be set up in an environment
# where snappl is available.  This will happen "soon"
# Get Rob to fix all of this.  For now, this is a hack
# so you can work short term.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent/"extern/snappl"))
# End of lines that will go away once we do this right

from AllASPFuncs import banner, fetchImages, save_lightcurve, \
                        build_lightcurve, build_lightcurve_sim, \
                        construct_psf_background, construct_psf_source, \
                        makeGrid, get_galsim_SED, getWeights, generateGuess, \
                        get_galsim_SED_list, prep_data_for_fit, run_one_object
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import argparse
from erfa import ErfaWarning
import galsim
import numpy as np
from numpy.linalg import LinAlgError
import os
import pandas as pd
from roman_imsim.utils import roman_utils
import scipy.sparse as sp
from simulation import simulate_images
from snpit_utils.logger import SNLogger as Lager
from snpit_utils.config import Config
from snappl.image import OpenUniverse2024FITSImage
import warnings
import yaml

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter('ignore', category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
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


def load_config(config_path):
    """Load parameters from a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Can overwrite config file")

    parser.add_argument('-f', '--filter', type=str, required=True,
                        help='Roman filter')
    parser.add_argument('-s', '--SNID', type=int, required=True,
                        help='Supernova ID')
    parser.add_argument('-t', '--num_total_images', type=int, required=False,
                        help='Number of images to use', default=np.inf)
    # TODO:change all instances of this variable to tot_images
    parser.add_argument('-d', '--num_detect_images', type=int, required=False,
                        help='Number of images to use with SN detections',
                        default=np.inf)
    # TODO:change all instances of this variable to det_images
    parser.add_argument('-o', '--output_path', type=str, required=False,
                        help='relative output path')

    parser.add_argument('-c', '--config', type=str, required=False,
                        help='relative config file path')

    parser.add_argument('-b', '--beginning', type=int, required=False,
                        help='start of desired lightcurve in days from peak.',
                        default=-np.inf)

    parser.add_argument('-e', '--end', type=int, required=False,
                        help='end of desired light curve in days from peak.',
                        default=np.inf)

    parser.add_argument('--object_type', type=str, required=False,
                        choices=["star", "SN"],
                        help='If star, will run on stars. If SN, will run  ' +
                             'on supernovae. If no argument is passed,' +
                             'assumes supernova.',
                        default='SN')

    args = parser.parse_args()
    band = args.filter
    SNID = args.SNID
    num_total_images = args.num_total_images
    num_detect_images = args.num_detect_images
    output_path = args.output_path
    lc_start = args.beginning
    lc_end = args.end
    object_type = args.object_type

    if args.config is not None:
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'config.yaml')

    config = load_config(config_path)

    size = config['size']
    use_real_images = config['use_real_images']
    use_roman = config['use_roman']
    check_perfection = config['check_perfection']
    make_exact = config['make_exact']
    avoid_non_linearity = config['avoid_non_linearity']
    deltafcn_profile = config['deltafcn_profile']
    do_xshift = config['do_xshift']
    do_rotation = config['do_rotation']
    noise = config['noise']
    method = config['method']
    make_initial_guess = config['make_initial_guess']
    subtract_background = config['subtract_background']
    weighting = config['weighting']
    pixel = config['pixel']
    roman_path = config['roman_path']
    sn_path = config['sn_path']
    bg_gal_flux = config['bg_gal_flux']
    source_phot_ops = config['source_phot_ops']
    mismatch_seds = config['mismatch_seds']
    fetch_SED = config['fetch_SED']
    initial_flux_guess = config['initial_flux_guess']
    deltafcn_profile = config['deltafcn_profile']
    sim_gal_ra_offset = config['sim_gal_ra_offset']
    sim_gal_dec_offset = config['sim_gal_dec_offset']

    grid_type = config['grid_type']
    er = f'{grid_type} is not a recognized grid type. Available options are '
    er += 'regular, adaptive, contour, or single. Details in documentation.'
    assert grid_type in ['regular', 'adaptive', 'contour',
                         'single', 'none'], er

    # PSF for when not using the Roman PSF:
    lam = 1293  # nm
    aberrations = galsim.roman.getPSF(1, band, pupil_bin=1).aberrations
    airy = galsim.ChromaticOpticalPSF(lam, diam=2.36,
                                      aberrations=aberrations)

    if make_exact:
        assert grid_type == 'single'
    if avoid_non_linearity:
        assert deltafcn_profile
    assert num_detect_images <= num_total_images

    galsim.roman.roman_psfs._make_aperture.clear()  # clear cache

    banner('Finding and Preparing Images')

    if not isinstance(SNID, list):
        SNID = [SNID]

    # run one supernova function TODO
    for ID in SNID:
        flux, sigma_flux, images, sumimages, exposures, ra_grid, dec_grid, wgt_matrix, \
            confusion_metric, X, cutout_wcs_list, sim_lc = \
            run_one_object(ID, object_type, num_total_images, num_detect_images, roman_path,
                           sn_path, size, band, fetch_SED, use_real_images,
                           use_roman, subtract_background,
                           make_initial_guess, initial_flux_guess,
                           weighting, method, grid_type,
                           pixel, source_phot_ops,
                           lc_start, lc_end, do_xshift, bg_gal_flux,
                           do_rotation, airy, mismatch_seds, deltafcn_profile,
                           noise, check_perfection, avoid_non_linearity,
                           sim_gal_ra_offset, sim_gal_dec_offset)

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
            lc = build_lightcurve_sim(sim_lc, flux, sigma_flux)
        if use_roman:
            psftype = 'romanpsf'
        else:
            psftype = 'analyticpsf'

        save_lightcurve(lc, identifier, band, psftype,
                        output_path=output_path)

        # Now, save the images
        images_and_model = np.array([images, sumimages, wgt_matrix])
        Lager.info('Saving images to ./results/images/' +
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

if __name__ == "__main__":
    main()
