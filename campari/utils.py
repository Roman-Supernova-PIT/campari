# Standard Library
import warnings

# Common Library
import numpy as np
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


def generate_guess(imlist, ra_grid, dec_grid):
    """This function initializes the guess for the optimization. For each grid
    point, it finds the average value of the pixel it is sitting in on
    each image. In some cases, this has offered minor improvements but it is
    not make or break for the algorithm.

    Inputs:
    imlist: list of snappl.image.Image objects, the images to use for the
            guess.
    ra_grid, dec_grid: numpy arrays of floats, the RA and DEC of the
                       grid points.

    Outputs:
    all_vals: numpy array of floats, the proposed initial guess for each model
                point.

    """
    size = imlist[0].image_shape[0]
    imx = np.arange(0, size, 1)
    imy = np.arange(0, size, 1)
    imx, imy = np.meshgrid(imx, imy)
    all_vals = np.zeros_like(ra_grid)

    wcslist = [im.get_wcs() for im in imlist]
    imdata = [im.data.flatten() for im in imlist]
    for i, imwcs in enumerate(zip(imdata, wcslist)):
        im, wcs = imwcs
        xx, yy = wcs.world_to_pixel(ra_grid, dec_grid)
        grid_point_vals = np.atleast_1d(np.zeros_like(xx))
        # For testing purposes, sometimes the grid is exactly one point, so we force it to be 1d.
        xx = np.atleast_1d(xx)
        yy = np.atleast_1d(yy)
        for imval, imxval, imyval in zip(im.flatten(), imx.flatten(), imy.flatten()):
            grid_point_vals[np.where((np.abs(xx - imxval) < 0.5) & (np.abs(yy - imyval) < 0.5))] = imval
        all_vals += grid_point_vals
    return all_vals / len(wcslist)


def gaussian(x, A, mu, sigma):
    """See name of function. :D"""
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


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
    bgarr = np.concatenate((im[0:size//4, 0:size//4].flatten(),
                            im[0:size, 3*(size//4):size].flatten(),
                            im[3*(size//4):size, 0:size//4].flatten(),
                            im[3*(size//4):size, 3*(size//4):size].flatten()))
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
