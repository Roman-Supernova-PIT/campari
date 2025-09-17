# Standard Library
import warnings

# Common Library
import numpy as np

# Astronomy Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from galsim import roman

# SN-PIT
from snpit_utils.logger import SNLogger

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

ROMAN_PIXEL_SCALE = 0.11  # arcsec/pixel


def gaussian(x, A, mu, sigma):
    """See name of function. :D"""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def calculate_background_level(im):
    """A function for naively estimating the background level from a given
    image. This may be replaced by a more sophisticated function later.
    For now, we take the corners of the image, sigma clip, and then return
    the median as the background level.

    Inputs:
    im, numpy array of floats, the image to be used.

    Returns:
    bg, float, the estimated background level.

    """
    size = im.shape[0]
    bgarr = np.concatenate(
        (
            im[0 : size // 4, 0 : size // 4].flatten(),
            im[0:size, 3 * (size // 4) : size].flatten(),
            im[3 * (size // 4) : size, 0 : size // 4].flatten(),
            im[3 * (size // 4) : size, 3 * (size // 4) : size].flatten(),
        )
    )
    if len(bgarr) == 0:
        bg = 0
    else:
        pc = np.percentile(bgarr, 84)
        bgarr = bgarr[bgarr < pc]
        bg = np.median(bgarr)

    return bg


def get_weights(images, ra, dec, gaussian_var=1000, cutoff=4):
    """This function calculates the weights for each pixel in the cutout
        images.

    The weights come from two sources. Firstly, the error in the image pixels
    is accounted for, i.e. higher error = less weight in the fit.
    Secondly, we can optionally apply a gaussian weighting to the fit
        centered on the supernova, since we do not care about pixels far away
        from the supernova.

    Inputs:
    images: list of snappl Image objects, used to get wcs, error, and size.
    ra, dec: floats, the RA and DEC of the supernova
    gaussian_var: float, the standard deviation squared of the Gaussian used
                    to weight   the pixels. This is in pixels.
    cutoff: float, the cutoff distance in pixels. Pixels further than this
                    distance from the supernova are given a weight of 0.

    Outputs:
    wgt_matrix: list of numpy arrays of floats, each array is the weights for
                the pixels in each cutout. Each array is size: (size x size)

    """
    size = images[0].image_shape[0]
    wcs_list = [im.get_wcs() for im in images]
    error = [im.noise for im in images]

    wgt_matrix = []
    SNLogger.debug(f"Gaussian Variance in get_weights {gaussian_var}")
    for i, wcs in enumerate(wcs_list):
        xx, yy = np.meshgrid(np.arange(0, size, 1), np.arange(0, size, 1))
        xx = xx.flatten()
        yy = yy.flatten()
        object_x, object_y = wcs.world_to_pixel(ra, dec)
        dist = np.sqrt((xx - object_x) ** 2 + (yy - object_y) ** 2)

        wgt = np.ones(size**2)
        wgt = 5 * np.exp(-(dist**2) / gaussian_var)
        # NOTE: This 5 is here because when I made this function I was
        # checking my work by plotting and the *5 made it easier to see. I
        # thought the overall normalization
        # of the weights did not matter. I was half right, they don't matter
        # for the flux but they do matter for the size of the errors. Therefore
        # there is some way that these weights are normalized, but I don't
        # know exactly how that should be yet. Online sources speaking about
        # weighted linear regression never seem to address normalization. TODO

        # Here, we throw out pixels that are more than 4 pixels away from the
        # SN. The reason we do this is because by choosing an image size one
        # has set a square top hat function centered on the SN. When that image
        # is rotated pixels in the corners leave the image, and new pixels
        # enter. By making a circular cutout, we minimize this problem. Of
        # course this is not a perfect solution, because the pixellation of the
        # circle means that still some pixels will enter and leave, but it
        # seems to minimize the problem.
        wgt[np.where(dist > cutoff)] = 0
        if error is None:
            error = np.ones_like(wgt)
        SNLogger.debug(f"wgt before: {np.mean(wgt)}")
        inv_var = 1 / (error[i].flatten()) ** 2
        wgt *= inv_var

        SNLogger.debug(f"wgt after: {np.mean(wgt)}")
        wgt_matrix.append(wgt)
    return wgt_matrix


def calc_mag_and_err(flux, sigma_flux, band, zp=None):
    """This function calculates the magnitude and magnitude error from the
       flux.

    flux: float or array of floats, the flux
    sigma_flux: float or array of floats, the flux error
    band: str, the bandpass of the images used
    zp: float, the zeropoint of the bandpass. If None, use the galsim-
                calculated value.

    Returns:
    mag: float or array of floats, the AB magnitude
    magerr: float or array of floats, the magnitude error
    zp: float, the zeropoint of the bandpass
    """

    exptime = {
        "F184": 901.175,
        "J129": 302.275,
        "H158": 302.275,
        "K213": 901.175,
        "R062": 161.025,
        "Y106": 302.275,
        "Z087": 101.7,
    }
    flux = np.atleast_1d(flux)
    sigma_flux = np.atleast_1d(sigma_flux)

    area_eff = roman.collecting_area
    zp = roman.getBandpasses()[band].zeropoint if zp is None else zp
    mag = -2.5 * np.log10(flux) + 2.5 * np.log10(exptime[band] * area_eff) + zp
    magerr = 2.5 / np.log(10) * (sigma_flux / flux)
    magerr[flux < 0] = np.nan
    return mag, magerr, zp


def banner(text):
    length = len(text) + 8
    message = "\n" + "#" * length + "\n" + "#   " + text + "   # \n" + "#" * length
    SNLogger.debug(message)


def make_sim_param_grid(params):
    nd_grid = np.meshgrid(*params)
    flat_grid = np.array(nd_grid, dtype=float).reshape(len(params), -1)
    SNLogger.debug(f"Created a grid of simulation parameters with a total of {flat_grid.shape[1]} combinations.")
    return flat_grid


def calculate_local_surface_brightness(image_object_list, cutout_pix=2):
    """A function to calculate the local surface brightness in a nondetection image.

    Parameters
    ----------
    image_object_list : list of snappl.image.Image objects
        The image objects to calculate the local surface brightness from.
    cutout_pix : int, optional
        The radius in pixels around the center of the image to use for the calculation. Since it must be odd, the
        total width will be 2*cutout_pix + 1. The default is 2 for a 5x5 cutout.
    Returns
    -------
    LSB : float
        The mean local surface brightness in mag/arcsec^2.
    """

    band = image_object_list[0].band

    cutout_pix = 2
    center_fluxes = []
    for i in image_object_list:
        image = i.data
        imsize = image.shape[0]
        center_fluxes.append(np.mean(
                image[
                    imsize // 2 - cutout_pix : imsize // 2 + cutout_pix,
                    imsize // 2 - cutout_pix : imsize // 2 + cutout_pix,
                ]))
    flux_in_image_center = np.array(center_fluxes)

    # Because the images are background subtracted, It's possible that the flux is negative, which would cause
    # an error when calculating the magnitude. Therefore, we set any negative fluxes to one.
    # This is more useful than setting them to zero, which would cause the magnitude to be infinite, as it will just
    # show that the surface brightness is very low post subtraction, which is the case if the background subtraction
    # gets the image to about ~0 brightness.
    if np.any(flux_in_image_center < 0):
        SNLogger.debug("Some fluxes in the cutout center are negative. Setting them to 1"
                       " to avoid errors in magnitude calculation.")
    flux_in_image_center[flux_in_image_center < 0] = 1
    mag_in_image_center, _, _ = calc_mag_and_err(
        flux_in_image_center, np.ones_like(flux_in_image_center), band
    )

    cutout_width = (2 * cutout_pix + 1) * ROMAN_PIXEL_SCALE
    cutout_area = cutout_width**2

    LSB = np.mean(mag_in_image_center + 2.5 * np.log10(cutout_area))
    SNLogger.debug(f"Local Surface Brightness: {LSB} mag/arcsec^2")

    return LSB
