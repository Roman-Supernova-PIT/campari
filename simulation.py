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
import galsim
import pandas as pd
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from roman_imsim.utils import roman_utils
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from astropy.nddata import Cutout2D
from snpit_utils.logger import SNLogger as Lager
from snpit_utils.config import Config

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


def simulate_images(num_total_images, num_detect_images, ra, dec,
                    sim_gal_ra_offset, sim_gal_dec_offset, do_xshift,
                    do_rotation, noise, use_roman, band, deltafcn_profile,
                    roman_path, size=11, input_psf=None, constant_imgs=False,
                    bg_gal_flux=None, source_phot_ops=True, sim_lc=None,
                    mismatch_seds=False, base_pointing=662, base_sca=11):
    '''
    This function simulates images using galsim for testing purposes. It is not
     used in the main pipeline.
    Inputs:
    num_total_images: int, the number of images to simulate
    num_detect_images: int, the number of images to simulate with a supernova
    ra, dec: floats, the RA and DEC of the center of the images to simulate,
        and the RA and DEC of the supernova.
    do_xshift:, bool whether to shift the images in the x direction (they will
    still be centered on the same point, this is just to emulate Roman taking a
    series of images at different locations.)
    do_rotation: bool, whether to rotate the images
    noise: float, the noise level to add to the images.
    use_roman: bool, whether to use the Roman PSF or a simple airy PSF.
    size: nt, the size of the images to simulate.
    sim_lc: list, the light curve of the supernova to simulate. If None,
        a default light curve will be generated.

    Returns:
    images: a numpy array of the images, with shape (num_total_images, size, size)
    im_wcs_list: a list of the wcs objects for each full SCA image
    cutout_wcs_list: a list of the wcs objects for each cutout image
    '''

    if not use_roman:
        assert input_psf is not None, 'you must provide an input psf if not \
             using roman'
    else:
        input_psf = None
    galra = ra + sim_gal_ra_offset
    galdec = dec + sim_gal_dec_offset

    if sim_lc is None:
        # Here, if the user has not provided a light curve that they want
        # simulated, we generate a default one.
        if num_detect_images == 0:
            sim_lc = 0
        else:
            d = np.linspace(5, 20, num_detect_images)
            mags = -5 * np.exp(-d/10) + 6
            fluxes = 10**(mags)
            sim_lc = list(fluxes)

    snra = ra
    sndec = dec
    im_wcs_list = []
    cutout_wcs_list = []
    imagelist = []
    roman_bandpasses = galsim.roman.getBandpasses()
    psf_storage = []
    sn_storage = []

    for i in range(num_total_images):

        if do_xshift:
            x_shift = 1e-5/3 * i
            y_shift = 0
        else:
            x_shift = 0
            y_shift = 0

        if do_rotation:
            rotation_angle = np.pi/10 * i
        else:
            rotation_angle = 0

        wcs_dict = simulate_wcs(rotation_angle, x_shift, y_shift, roman_path,
                                base_sca, base_pointing, band)
        imwcs = WCS(wcs_dict)

        # Just using this astropy tool to get the cutout wcs.
        cutoutstamp = Cutout2D(np.zeros((4088, 4088)), SkyCoord(ra=ra*u.degree,
                               dec=dec*u.degree), size, wcs=imwcs)
        cutoutgalwcs = galsim.AstropyWCS(wcs=cutoutstamp.wcs)

        # Overall WCS for the image
        galwcs, origin = galsim.wcs.readFromFitsHeader(wcs_dict)
        im_wcs_list.append(galwcs)

        if mismatch_seds:
            Lager.debug('INTENTIONALLY MISMATCHING SEDS, 1a SED')
            file_path = r"snflux_1a.dat"
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Day',
                             'Wavelength', 'Flux'])
            a = df.loc[df.Day == 0]
            del df
            sed = galsim.SED(galsim.LookupTable(a.Wavelength/10, a.Flux,
                             interpolant='linear'), wave_type='nm',
                             flux_type='fphotons')

        else:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                             interpolant='linear'), wave_type='nm',
                             flux_type='fphotons')

        stamp = galsim.Image(size, size, wcs=cutoutgalwcs)
        pointx, pointy = cutoutgalwcs.toImage(galra, galdec, units='deg')

        if use_roman:
            sim_psf = galsim.roman.getPSF(1, band, pupil_bin=8,
                                          wcs=cutoutgalwcs)

        else:
            sim_psf = input_psf

        # Draw the galaxy.
        convolved = simulate_galaxy(bg_gal_flux, deltafcn_profile, band,
                                    sim_psf, sed)

        a = convolved.drawImage(roman_bandpasses[band], method='no_pixel',
                                image=stamp, wcs=cutoutgalwcs,
                                center=galsim.PositionD(pointx, pointy),
                                use_true_center=True)
        a = a.array

        stamp2 = galsim.Image(size, size, wcs=cutoutgalwcs)
        psf_storage.append((sim_psf*sed).drawImage(roman_bandpasses[band],
                           wcs=cutoutgalwcs, center=(5, 5),
                           use_true_center=True, image=stamp2).array)

        # Noise it up!
        if noise > 0:
            a += np.random.normal(0, noise, size**2).reshape(size, size)

        # Inject a supernova! If using.
        if sim_lc != 0:
            # Here we want to count which supernova image we are on. The
            # following is zero on the first sn image and counts up:
            sn_im_index = i - num_total_images + num_detect_images
            if sn_im_index >= 0:
                snx, sny = cutoutgalwcs.toImage(snra, sndec, units='deg')
                stamp = galsim.Image(size, size, wcs=cutoutgalwcs)
                Lager.debug(f'sed: {sed}')
                supernova_image = \
                    simulate_supernova(snx, sny, stamp,
                                       sim_lc[sn_im_index],
                                       sed, band, sim_psf, source_phot_ops,
                                       base_pointing, base_sca)

                a += supernova_image
                sn_storage.append(supernova_image)

        cutout_wcs_list.append(cutoutgalwcs)
        imagelist.append(a)

    images = imagelist
    Lager.debug(f'images shape: {images[0].shape}')
    Lager.debug(f'images length {len(images)}')
    file_path = pathlib.Path(__file__).parent/'temp_tds.yaml'
    util_ref = roman_utils(config_file=file_path,
                           visit=base_pointing, sca=base_sca)


    return images, im_wcs_list, cutout_wcs_list, sim_lc, util_ref


def simulate_wcs(angle, x_shift, y_shift, roman_path, base_sca, base_pointing,
                 band):
    rotation_matrix = np.array([np.cos(angle), -np.sin(angle), np.sin(angle),
                               np.cos(angle)]).reshape(2, 2)
    image = fits.open(roman_path + f'/RomanTDS/images/truth/{band}/' +
                      f'{base_pointing}/Roman_TDS_truth_{band}_{base_pointing}'
                      + f'_{base_sca}.fits.gz')

    CD_matrix = np.zeros((2, 2))
    CD_matrix[0, 0] = image[0].header['CD1_1']
    CD_matrix[0, 1] = image[0].header['CD1_2']
    CD_matrix[1, 0] = image[0].header['CD2_1']
    CD_matrix[1, 1] = image[0].header['CD2_2']

    CD_matrix_rotated = CD_matrix @ rotation_matrix

    wcs_dict = {
            'CTYPE1': image[0].header['CTYPE1'],
            'CTYPE2': image[0].header['CTYPE2'],
            'CRPIX1': image[0].header['CRPIX1'],
            'CRPIX2': image[0].header['CRPIX2'],
            'CD1_1': CD_matrix_rotated[0, 0],
            'CD1_2': CD_matrix_rotated[0, 1],
            'CD2_1': CD_matrix_rotated[1, 0],
            'CD2_2': CD_matrix_rotated[1, 1],
            'CUNIT1': image[0].header['CUNIT1'],
            'CUNIT2': image[0].header['CUNIT2'],
            'CRVAL1':   image[0].header['CRVAL1'] + x_shift,
            'CRVAL2':  image[0].header['CRVAL2'] + y_shift,
            'NAXIS1': image[0].header['NAXIS1'],
            'NAXIS2': image[0].header['NAXIS2']
        }

    return wcs_dict


def simulate_galaxy(bg_gal_flux, deltafcn_profile, band, sim_psf, sed):
    roman_bandpasses = galsim.roman.getBandpasses()
    if deltafcn_profile:
        profile = galsim.DeltaFunction()
    else:
        bulge = galsim.Sersic(n=3, half_light_radius=1.6)
        disk = galsim.Exponential(half_light_radius=5)
        profile = bulge + disk

    profile *= sed
    profile = profile.withFlux(bg_gal_flux, roman_bandpasses[band])
    convolved = galsim.Convolve(profile, sim_psf)
    return convolved


def simulate_supernova(snx, sny, stamp, flux, sed, band, sim_psf,
                       source_phot_ops, base_pointing, base_sca,
                       random_seed=0):
    roman_bandpasses = galsim.roman.getBandpasses()
    profile = galsim.DeltaFunction()*sed
    profile = profile.withFlux(flux, roman_bandpasses[band])

    # Code below copied from galsim largely
    if not source_phot_ops:
        result = sim_psf.drawImage(roman_bandpasses[band], image=stamp,
                                   wcs=stamp.wcs, method='no_pixel',
                                   center=galsim.PositionD(snx, sny),
                                   use_true_center=True)
        return result.array

    config_file = './temp_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit=base_pointing,
                           sca=base_sca)
    photon_ops = [sim_psf] + util_ref.photon_ops

    # If random_seed is zero, galsim will use the current time to make a seed
    rng = galsim.BaseDeviate(random_seed)
    result = profile.drawImage(roman_bandpasses[band], wcs=stamp.wcs,
                               method='phot', photon_ops=photon_ops,
                               rng=rng, n_photons=int(1e6),
                               maxN=int(1e6), poisson_flux=False,
                               center=galsim.PositionD(snx, sny),
                               use_true_center=True, image=stamp)
    return result.array
