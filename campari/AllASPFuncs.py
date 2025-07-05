# Standard Library
import os
import pathlib
import warnings

# Common Library
import astropy.table as tb
import galsim
import h5py
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from galsim import roman
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from roman_imsim.utils import roman_utils
from scipy.interpolate import RegularGridInterpolator

# SN-PIT
import snappl
from snappl.image import OpenUniverse2024FITSImage
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger as Lager
from snappl.psf import PSF

# Campari
from campari.simulation import simulate_images

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


def make_regular_grid(ra_center, dec_center, wcs, size, spacing=1.0,
                      subsize=9):
    """ Generates a regular grid around a (RA, Dec) center, choosing step size.

    ra_center, dec_center: floats, coordinate center of the image
    wcs: the WCS of the image, snappl.wcs.BaseWCS object
    spacing: int, spacing of grid points in pixels.
    subsize: int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.


    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.
    """

    if subsize > size:
        Lager.warning("subsize is larger than the image size. " +
                      f"{size} > {subsize}. This would cause model points to" +
                      " be placed outside the image. Reducing subsize to" +
                      " match the image size.")
        subsize = size

    Lager.debug("Grid type: regularly spaced")
    difference = int((size - subsize)/2)

    x = difference + np.arange(0, subsize, spacing)
    y = difference + np.arange(0, subsize, spacing)
    Lager.debug(f"Grid spacing: {spacing}")

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    Lager.debug(f"Built a grid with {np.size(xx)} points")

    ra_grid, dec_grid = wcs.pixel_to_world(xx, yy)

    return ra_grid, dec_grid


def make_adaptive_grid(ra_center, dec_center, wcs,
                       image, percentiles=[45, 90], subsize=9,
                       subpixel_grid_width=1.2):
    """ Construct an "adaptive grid" which allocates model grid points to model
    the background galaxy according to the brightness of the image.

    Inputs:
    ra_center, dec_center: floats, coordinate center of the image
    wcs: the WCS of the image, snappl.wcs.BaseWCS
    image: 2D numpy array of floats of shape (size x size), the image to build
    the grid on. This is used to determine the size of the grid, and once we
                switch to snappl Image objects, will also determine the wcs.
    percentiles: list of floats, the percentiles to use to bin the image. The
                more bins, the more possible grid points could be placed in
                that pixel. For instance, say if you had bins [45, 90],
                as is default. A pixel that lies in the 30th percentile for
                brightness would get 1 point. A pixel at the 50th percentiile
                for brightness would get a 2x2 grid of points.
                A pixel above the 90th percentile would get a 3x3 grid of
                points. If you have more bins, you could go even higher to
                4x4 and 5x5 etc. These points are evenly spaced within the
                pixel.
    subsize: int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.
    subpixel_grid_width: When we place the model points in a pixel, we place
                        them on a small range of locations within the pixel.
                        For instance, 0.25, 0.5, and 0.75 for x values. However
                        I've found this leads to awkward gaps in grid points
                        between pixels. For instance, the point at 0.25 would
                        be half a pixel from the point located at 0.75 in the
                        next lower pixel, and only 0.25 from the point at 0.5
                        in the same pixel.
                        Therefore, subpixel_grid_width can be
                        set to something larger than 1 so that the evenly
                        spaced points are spaced out more, reducing these gaps.
                        For instance, if you set to 1.2, the x values of the
                        pixels would be [0.2, 0.5, 0.8] instead, reducing
                        inter-pixel gaps.

    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.
    """
    size = np.shape(image)[0]
    if subsize > size:
        Lager.warning("subsize is larger than the image size " +
                      f"{size} > {subsize}. This would cause model points to" +
                      " be placed outside the image. Reducing subsize to" +
                      " match the image size.")
        subsize = size

    Lager.debug("image shape: {}".format(np.shape(image)))
    Lager.debug("Grid type: adaptive")
    # Bin the image in logspace and allocate grid points based on the
    # brightness.

    difference = int((size - subsize)/2)

    x = difference + np.arange(0, subsize, 1)
    y = difference + np.arange(0, subsize, 1)

    if percentiles.sort() != percentiles:
        Lager.warning("Percentiles not in ascending order. Sorting them.")
        percentiles.sort()
        Lager.warning(f"Percentiles: {percentiles}")

    imcopy = np.copy(image)
    # We need to make sure that the image is not zero, otherwise we get
    # infinities in the log space.
    imcopy[imcopy <= 0] = 1e-10
    imcopy = np.log(imcopy)
    bins = [0]
    bins.extend(np.nanpercentile(imcopy, percentiles))
    bins.append(100)
    Lager.debug(f"BINS: {bins}")

    brightness_levels = np.digitize(imcopy, bins)
    xs = []
    ys = []
    # Round y and x locations to the nearest pixel. This is necessary because
    # we want to check the brightness for each pixel within the grid, and by
    # rounding we can index the brightness_levels array.
    yvals = np.rint(y).astype(int)
    xvals = np.rint(x).astype(int)
    for xindex in xvals:
        x = xindex
        for yindex in yvals:
            y = yindex
            # xindex and yindex are the indices within the numpy array, while
            # x and y are the actual locations in pixel space.
            # This used to be x and y in here:
            num = int(brightness_levels[xindex][yindex])
            if num == 0:
                pass
            elif num == 1:
                xs.append(y)
                ys.append(x)  # I know I swap this because Astropy takes (y,x)
                # order but I'd really like to iron out all the places I do
                # this rather than doing it so off the cuff. TODO
            else:
                xx = np.linspace(x - subpixel_grid_width/2,
                                 x + subpixel_grid_width/2, num+2)[1:-1]
                yy = np.linspace(y - subpixel_grid_width/2,
                                 y + subpixel_grid_width/2, num+2)[1:-1]
                X, Y = np.meshgrid(xx, yy)
                ys.extend(list(X.flatten()))
                xs.extend(list(Y.flatten()))  # ...Like here. TODO

    xx = np.array(xs).flatten()
    yy = np.array(ys).flatten()

    Lager.debug(f"Built a grid with {np.size(xx)} points")

    ra_grid, dec_grid = wcs.pixel_to_world(xx, yy)

    return ra_grid, dec_grid


def generateGuess(imlist, ra_grid, dec_grid):
    """ This function initializes the guess for the optimization. For each grid
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
        grid_point_vals = np.zeros_like(xx)
        for imval, imxval, imyval in zip(im.flatten(),
                                         imx.flatten(), imy.flatten()):
            grid_point_vals[np.where((np.abs(xx - imxval) < 0.5) &
                                     (np.abs(yy - imyval) < 0.5))] = imval
        all_vals += grid_point_vals
    return all_vals/len(wcslist)


def construct_psf_background(ra, dec, wcs, x_loc, y_loc, stampsize,
                             psf=None, pixel=False,
                             util_ref=None, band=None):

    """Constructs the background model around a certain image (x,y) location
    and a given array of RA and DECs.
    Inputs:
    ra, dec: arrays of floats, RA and DEC values for the grid
    wcs: the wcs of the image, if the image is a cutout, this MUST be the wcs
        of the cutout. A snappl.wcs.BaseWCS object.
    x_loc, y_loc: floats,the pixel location of the image in the FULL image,
        i.e. x y location in the SCA.
    stampsize: int, the size of the stamp being used
    band: str, the bandpass being used
    psf: Here you can provide a PSF to use, if you don't provide one, you must
        provide a util_ref, and this function will calculate the Roman PSF
        instead.
    pixel: bool, If True, use a pixel tophat function to convolve the PSF with,
        otherwise use a delta function. Does not seem to hugely affect results.
    util_ref: A roman_imsim.utils.roman_utils object, which is used to
        calculate the PSF. If you provide this, you don't need to provide a PSF
        and the Roman PSF will be calculated. Note
        that this needs to be for the correct SCA/Pointing combination.

    Returns:
    A numpy array of the PSFs at each grid point, with the shape
    (stampsize*stampsize, npoints)
    """

    assert util_ref is not None or psf is not None, "you must provide at \
        least util_ref or psf"
    assert util_ref is not None or band is not None, "you must provide at \
        least util_ref or band"

    # This is the WCS galsim uses to draw the PSF.
    galsim_wcs = wcs.get_galsim_wcs()
    x, y = wcs.world_to_pixel(ra, dec)

    # With plus ones here I recover the values pre-refactor!

    if psf is None:
        # How different are these two methods? TODO XXX
        pupil_bin = 8
        # psf = util_ref.getPSF(x_loc, y_loc, pupil_bin=pupil_bin)
        psf = galsim.roman.getPSF(1, band, pupil_bin=pupil_bin, wcs=galsim_wcs)

    bpass = roman.getBandpasses()[band]

    psfs = np.zeros((stampsize * stampsize, np.size(x)))

    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"),
                     wave_type="nm", flux_type="fphotons")

    if pixel:
        point = galsim.Pixel(0.1)*sed
    else:
        point = galsim.DeltaFunction()
        point *= sed

    point = point.withFlux(1, bpass)
    oversampling_factor = 1
    convolvedpsf = galsim.Convolve(point, psf)
    stamp = galsim.Image(stampsize*oversampling_factor,
                         stampsize*oversampling_factor, wcs=galsim_wcs)
    # Loop over the grid points, draw a PSF at each one, and append to a list.
    for a, ij in enumerate(zip(x.flatten(), y.flatten())):
        if a % 50 == 0:
            Lager.debug(f"Drawing PSF {a} of {np.size(x)}")
        i, j = ij
        psfs[:, a] = convolvedpsf.drawImage(bpass, method="no_pixel",
                                            center=galsim.PositionD(i, j),
                                            use_true_center=True, image=stamp,
                                            wcs=galsim_wcs).array.flatten()

    return psfs


def findAllExposures(snid, ra, dec, start, end, band, maxbg=24,
                     maxdet=24, return_list=False,
                     roman_path=None, pointing_list=None, SCA_list=None,
                     truth="simple_model", lc_start=-np.inf, lc_end=np.inf):
    """ This function finds all the exposures that contain a given supernova,
    and returns a list of them. Utilizes Rob's awesome database method to
    find the exposures. Humongous speed up thanks to this.

    Inputs:
    snid: the ID of the supernova
    ra, dec: the RA and DEC of the supernova (TODO: Is this necessary if we're
            passing the ID?)
    peak: the peak of the supernova
    start, end: the start and end of the observing window
    maxbg: the maximum number of background images to consider
    maxdet: the maximum number of detected images to consider
    return_list: whether to return the exposures as a list or not
    stampsize: the size of the stamp to use
    roman_path: the path to the Roman data
    pointing_list: If this is passed in, only consider these pointings
    SCA_list: If this is passed in, only consider these SCAs
    truth: If "truth" use truth images, if "simple_model" use simple model
            images.
    band: the band to consider
    lc_start, lc_end: the start and end of the light curve window, in MJD.

    explist: astropy.table.Table, the table of exposures that contain the
    supernova. The columns are:
        - Pointing: the pointing of the exposure
        - SCA: the SCA of the exposure
        - BAND: the band of the exposure
        - date: the MJD of the exposure
        - DETECTED: whether the exposure contains a detection or not.
    """

    f = fits.open(roman_path +
                  "/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits")[1]
    f = f.data

    explist = tb.Table(names=("Pointing", "SCA", "BAND", "date"),
                       dtype=("i8", "i4", "str",  "f8"))

    # Rob's database method! :D

    server_url = "https://roman-desc-simdex.lbl.gov"
    req = requests.Session()
    result = req.post(f"{server_url}/findromanimages/containing=({ra},{dec})")
    if result.status_code != 200:
        raise RuntimeError(f"Got status code {result.status_code}\n"
                           "{result.text}")

    res = pd.DataFrame(result.json())[["filter", "pointing", "sca", "mjd"]]
    res.rename(columns={"mjd": "date", "pointing": "Pointing", "sca": "SCA"},
               inplace=True)

    res = res.loc[res["filter"] == band]
    # The first date cut selects images that are detections, the second
    # selects detections within the requested light curve window
    start = start[0] if not isinstance(start, float) else start
    end = end[0] if not isinstance(end, float) else end
    det = res.loc[(res["date"] >= start) & (res["date"] <= end)].copy()
    det = det.loc[(det['date'] >= lc_start) & (det['date'] <= lc_end)]
    if isinstance(maxdet, int):
        det = det.iloc[:maxdet]
    det["DETECTED"] = True

    if pointing_list is not None:
        det = det.loc[det["Pointing"].isin(pointing_list)]

    bg = res.loc[(res["date"] < start) | (res["date"] > end)].copy()
    bg = bg.loc[(bg['date'] >= lc_start) & (bg['date'] <= lc_end)]
    if isinstance(maxbg, int):
        bg = bg.iloc[:maxbg]
    bg["DETECTED"] = False

    all_images = pd.concat([det, bg])
    all_images["BAND"] = band

    explist = Table.from_pandas(all_images)
    explist.sort(["DETECTED", "SCA"])
    Lager.info("\n" + str(explist))

    if return_list:
        return explist


def find_parquet(ID, path, obj_type="SN"):
    """Find the parquet file that contains a given supernova ID."""

    files = os.listdir(path)
    file_prefix = {"SN": "snana", "star": "pointsource"}
    files = [f for f in files if file_prefix[obj_type] in f]
    files = [f for f in files if ".parquet" in f]
    files = [f for f in files if "flux" not in f]

    for f in files:
        pqfile = int(f.split("_")[1].split(".")[0])
        df = open_parquet(pqfile, path, obj_type=obj_type)
        # The issue is SN parquet files store their IDs as ints and star
        # parquet files as strings.
        # Should I convert the entire array or is there a smarter way to do
        # this?
        if ID in df.id.values or str(ID) in df.id.values:
            return pqfile


def open_parquet(parq, path, obj_type="SN", engine="fastparquet"):
    """Convenience function to open a parquet file given its number."""
    file_prefix = {"SN": "snana", "star": "pointsource"}
    base_name = "{:s}_{}.parquet".format(file_prefix[obj_type], parq)
    file_path = os.path.join(path, base_name)
    df = pd.read_parquet(file_path, engine=engine)
    return df


def radec2point(RA, DEC, filt, path, start=None, end=None):
    """This function takes in RA and DEC and returns the pointing and SCA with
    center closest to desired RA/DEC
    """
    f = fits.open(path+"/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits")[1]
    f = f.data

    allRA = f["RA"]
    allDEC = f["DEC"]

    pointing_sca_coords = SkyCoord(allRA*u.deg, allDEC*u.deg, frame="icrs")
    search_coord = SkyCoord(RA*u.deg, DEC*u.deg, frame="icrs")
    dist = pointing_sca_coords.separation(search_coord).arcsec

    dist[np.where(f["filter"] != filt)] = np.inf
    reshaped_array = dist.flatten()
    # Find the indices of the minimum values along the flattened slices
    min_indices = np.argmin(reshaped_array, axis=0)
    # Convert the flat indices back to 2D coordinates
    rows, cols = np.unravel_index(min_indices, dist.shape[:2])

    # The plus 1 is because the SCA numbering starts at 1
    return rows, cols + 1


def construct_psf_source(x, y, pointing, SCA, stampsize=25, x_center=None,
                         y_center=None, sed=None, flux=1, photOps=True):
    """Constructs the PSF around the point source (x,y) location, allowing for
        some offset from the center.
    Inputs:
    x, y: ints, pixel coordinates where the cutout is centered in the SCA
    pointing, SCA: ints, the pointing and SCA of the image
    stampsize = int, size of cutout image used
    x_center and y_center: floats, x and y location of the object in the SCA.
    sed: galsim.sed.SED object, the SED of the source
    flux: float, If you are using this function to build a model grid point,
        this should be 1. If you are using this function to build a model of
        a source, this should be the flux of the source.
    Outputs:
    psf_image: numpy array of floats of size stampsize**2, the image
                of the PSF at the (x,y) location.
    """
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError(f"x and y must be integers, not {type(x), type(y)}")
    Lager.debug(f"ARGS IN PSF SOURCE: \n x, y: {x, y} \n" +
                f" Pointing, SCA: {pointing, SCA} \n" +
                f" stamp size: {stampsize} \n" +
                f" x_center, y_center: {x_center, y_center} \n" +
                f" sed: {sed} \n" +
                f" flux: {flux}")

    assert sed is not None, "You must provide an SED for the source"

    if not photOps:
        # While I want to do this sometimes, it is very rare that you actually
        # want to do this. Thus if it was accidentally on while doing a normal
        # run, I'd want to know.
        Lager.warning("NOT USING PHOTON OPS IN PSF SOURCE")

    psf_object = PSF.get_psf_object("ou24PSF_slow", pointing=pointing, sca=SCA,
                                    size=stampsize, include_photonOps=photOps)
    psf_image = psf_object.get_stamp(x0=x, y0=y, x=x_center, y=y_center,
                                     flux=1., seed=None)

    return psf_image.flatten()


def gaussian(x, A, mu, sigma):
    """See name of function. :D"""
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def constructImages(exposures, ra, dec, size=7, subtract_background=True,
                    roman_path=None, truth="simple_model"):

    """Constructs the array of Roman images in the format required for the
    linear algebra operations.

    Inputs:
    exposures is a list of exposures from findAllExposures
    ra,dec: the RA and DEC of the SN
    subtract_background: If False, the background level is fit as a free
        parameter in the forward modelling. Otherwise, we subtract it here.
    roman_path: the path to the Roman data

    Returns:
    cutout_image_list: list of snappl.image.Image objects, cutouts on the
                       object location.
    image_list: list of snappl.image.Image objects of the entire SCA.

    """

    bgflux = []
    image_list = []
    cutout_image_list = []

    Lager.debug(f"truth in construct images: {truth}")

    for indx, i in enumerate(exposures):
        Lager.debug(f"Constructing image {indx} of {len(exposures)}")
        band = i["BAND"]
        pointing = i["Pointing"]
        SCA = i["SCA"]

        # TODO : replace None with the right thing once Exposure is implemented

        imagepath = roman_path + (f"/RomanTDS/images/{truth}/{band}/{pointing}"
                                  f"/Roman_TDS_{truth}_{band}_{pointing}_"
                                  f"{SCA}.fits.gz")
        image = OpenUniverse2024FITSImage(imagepath, None, SCA)
        imagedata, errordata, flags = image.get_data(which="all")
        image_cutout = image.get_ra_dec_cutout(ra, dec, size)

        if truth == "truth":
            raise RuntimeError("Truth is broken.")
            # In the future, I'd like to manually insert an array of ones for
            # the error, or something.

        """
        try:
            zero = np.power(10, -(i["zeropoint"] - self.common_zpt)/2.5)
        except:
            zero = -99

        if zero < 0:
            zero =
        im = cutout * zero
        """

        # If we are not fitting the background we subtract it here.
        # When subtract_background is False, we are including the background
        # level as a free parameter in our fit, so it should not be subtracted
        # here.
        bg = 0
        if subtract_background:
            if not truth == "truth":
                # However, if we are subtracting the background, we want to get
                # rid of it here, either by reading the SKY_MEAN value from the
                # image header...
                bg = image_cutout._get_header()["SKY_MEAN"]
            elif truth == "truth":
                # ....or manually calculating it!
                bg = calculate_background_level(imagedata)

        bgflux.append(bg)  # This currently isn't returned, but might be a good
        # thing to put in output? TODO

        image_cutout._data -= bg
        Lager.debug(f"Subtracted a background level of {bg}")

        image_list.append(image)
        cutout_image_list.append(image_cutout)

    return cutout_image_list, image_list


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


def getPSF_Image(self, stamp_size, x=None, y=None, x_center=None,
                 y_center=None, pupil_bin=8, sed=None, oversampling_factor=1,
                 include_photonOps=False, n_phot=1e6, pixel=False, flux=1):

    if pixel:
        point = galsim.Pixel(1)*sed
        Lager.debug("Building a Pixel shaped PSF source")
    else:
        point = galsim.DeltaFunction()*sed

    # Note the +1s in galsim.PositionD below; galsim uses 1-indexed pixel positions,
    # whereas snappl uses 0-indexed pixel positions
    x_center += 1
    y_center += 1
    x += 1
    y += 1

    point = point.withFlux(flux, self.bpass)
    local_wcs = self.getLocalWCS(x, y)
    wcs = galsim.JacobianWCS(dudx=local_wcs.dudx/oversampling_factor,
                             dudy=local_wcs.dudy/oversampling_factor,
                             dvdx=local_wcs.dvdx/oversampling_factor,
                             dvdy=local_wcs.dvdy/oversampling_factor)
    stamp = galsim.Image(stamp_size*oversampling_factor,
                         stamp_size*oversampling_factor, wcs=wcs)

    if not include_photonOps:
        Lager.debug(f'in getPSF_Image: {self.bpass}, {x_center}, {y_center}')

        psf = galsim.Convolve(point, self.getPSF(x, y, pupil_bin))
        return psf.drawImage(self.bpass, image=stamp, wcs=wcs,
                             method="no_pixel",
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True)

    photon_ops = [self.getPSF(x, y, pupil_bin)] + self.photon_ops
    Lager.debug(f"Using {n_phot:e} photons in getPSF_Image")
    result = point.drawImage(self.bpass, wcs=wcs, method="phot",
                             photon_ops=photon_ops, rng=self.rng,
                             n_photons=int(n_phot), maxN=int(n_phot),
                             poisson_flux=False,
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True, image=stamp)
    return result


def fetchImages(exposures, ra, dec, size, subtract_background, roman_path, object_type):
    """This function gets the list of exposures to be used for the analysis.

    Inputs:
    exposures: astropy.table.table.Table, the table of exposures to be used.
    num_total_images: total images used in analysis (detection + no detection)
    num_detect_images: number of images used in the analysis that contain a
                       detection.
    size: int, cutout will be of shape (size, size)e
    subtract_background: If True, subtract sky bg from images. If false, leave
            bg as a free parameter in the forward modelling.
    roman_path: str, the path to the Roman data
    object_type: str, the type of object to be used (SN or star)

    Returns:
    ra, dec: floats, the RA and DEC of the supernova, a single float is
                         used for both of these as we assume the object is
                         not moving between exposures.
    exposures: astropy.table.table.Table, table of exposures used
    cutout_image_list: list of snappl.image.Image objects, the cutout images
    image_list: list of snappl.image.Image objects, the full images
    """

    num_predetection_images = len(exposures[~exposures["DETECTED"]])
    num_total_images = len(exposures)
    if num_predetection_images == 0 and object_type == "SN":
        raise ValueError("No pre-detection images found in time range " +
                         "provided, skipping this object.")

    if num_predetection_images == 0:
        raise ValueError("No detection images found in time range " +
                         "provided, skipping this object.")

    if num_total_images != np.inf and len(exposures) != num_total_images:
        raise ValueError(f"Not Enough Exposures. \
            Found {len(exposures)} out of {num_total_images} requested")

    cutout_image_list, image_list =\
        constructImages(exposures, ra, dec, size=size,
                        subtract_background=subtract_background,
                        roman_path=roman_path)

    return cutout_image_list, image_list


def get_object_info(ID, parq, band, snpath, roman_path, obj_type):

    """Fetch some info about an object given its ID.
    Inputs:
    ID: the ID of the object
    parq: the parquet file containing the object
    band: the band to consider
    date: whether to return the start end and peak dates of the object
    snpath: the path to the supernova data
    roman_path: the path to the Roman data
    host: whether to return the host RA and DEC

    Returns:
    ra, dec: the RA and DEC of the object
    pointing, sca: the pointing and SCA of the object
    start, end, peak: the start, end, and peak dates of the object
    """

    df = open_parquet(parq, snpath, obj_type=obj_type)
    if obj_type == "star":
        ID = str(ID)

    df = df.loc[df.id == ID]
    ra, dec = df.ra.values[0], df.dec.values[0]

    if obj_type == "SN":
        start = df.start_mjd.values
        end = df.end_mjd.values
        peak = df.peak_mjd.values
    else:
        start = [0]
        end = [np.inf]
        peak = [0]

    pointing, sca = radec2point(ra, dec, band, roman_path)

    return ra, dec, pointing, sca, start, end, peak


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
    Lager.debug(f"Gaussian Variance in get_weights {gaussian_var}")
    for i, wcs in enumerate(wcs_list):
        xx, yy = np.meshgrid(np.arange(0, size, 1), np.arange(0, size, 1))
        xx = xx.flatten()
        yy = yy.flatten()
        object_x, object_y = wcs.world_to_pixel(ra, dec)
        dist = np.sqrt((xx - object_x)**2 + (yy - object_y)**2)


        wgt = np.ones(size**2)
        wgt = 5*np.exp(-dist**2/gaussian_var)
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
        Lager.debug(f"wgt before: {np.mean(wgt)}")
        wgt /= (error[i].flatten())**2  # Define an inv variance TODO
        Lager.debug(f"wgt after: {np.mean(wgt)}")
        wgt_matrix.append(wgt)
    return wgt_matrix


def makeGrid(grid_type, images, ra, dec, percentiles=[],
             make_exact=False):
    """This is a function that returns the locations for the model grid points
    used to model the background galaxy. There are several different methods
    for building the grid, listed below, and this parent function calls the
    correct function for which type of grid you wish to construct.

    Inputs:
    grid_type: str, type of grid method to use.
              regular: A regularly spaced grid.
              adaptive: Points are placed in the image based on the brightness
                        in each pixel.
              contour: Points are placed by placing finer and finer regularly
                        spaced grids in different contour levels of a linear
                        interpolation of the image. See make_contour_grid for
                        a more detailed explanation.
              single: Place a single grid point. This is for sanity checking
                      that the algroithm is drawing points where expected.
    images: list of snappl.image.Image objects, the images to be used for the
            grid. The first image in the list is used to get the WCS and
            design the grid.
    ra, dec: floats, the RA and DEC of the supernova. As of now, this is only
                    used if grid_type is "single", TODO remove this?
    percentiles: list of floats, the percentiles to use for the adaptive grid.
    make_exact: Currently not implemented, but will construct the grid in such
                a way on a simulated image that the recovered model is accurate
                to machine precision. TODO

    Returns:
    ra_grid, dec_grid: numpy arrays of floats of the ra and dec locations for
                    model grid points.
    """
    size = images[0].image_shape[0]
    snappl_wcs = images[0].get_wcs()
    image_data = images[0].data

    Lager.debug(f"Grid type: {grid_type}")
    if grid_type not in ["regular", "adaptive", "contour", "single"]:
        raise ValueError("Grid type must be one of: regular, adaptive, "
                         "contour, single")
    if grid_type == "contour":
        ra_grid, dec_grid = make_contour_grid(image_data, snappl_wcs)

    # TODO: de-hardcode spacing and percentiles. These should be passable
    # options.
    elif grid_type == "adaptive":
        ra_grid, dec_grid = make_adaptive_grid(ra, dec, snappl_wcs,
                                               image=image_data,
                                               percentiles=percentiles)
    elif grid_type == "regular":
        ra_grid, dec_grid = make_regular_grid(ra, dec, snappl_wcs,
                                              size=size, spacing=0.75)

    if grid_type == "single":
        ra_grid, dec_grid = [ra], [dec]

    if make_exact:
        if grid_type == "single":
            raise NotImplementedError
            # I need to figure out how to turn the single grid point test
        else:
            raise NotImplementedError
            # I need to figure out how to turn the single grid point test

    ra_grid = np.array(ra_grid)
    dec_grid = np.array(dec_grid)
    return ra_grid, dec_grid


def plot_lc(filepath, return_data=False):
    fluxdata = pd.read_csv(filepath, comment="#", delimiter=" ")
    truth_mag = fluxdata["SIM_true_mag"]
    mag = fluxdata["mag"]
    sigma_mag = fluxdata["mag_err"]

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    dates = fluxdata["MJD"]

    plt.scatter(dates, truth_mag, color="k", label="Truth")
    plt.errorbar(dates, mag, yerr=sigma_mag,  color="purple", label="Model",
                 fmt="o")

    plt.ylim(np.max(truth_mag) + 0.2, np.min(truth_mag) - 0.2)
    plt.ylabel("Magnitude (Uncalibrated)")

    residuals = mag - truth_mag
    bias = np.mean(residuals)
    bias *= 1000
    bias = np.round(bias, 3)
    scatter = np.std(residuals)
    scatter *= 1000
    scatter = np.round(scatter, 3)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    textstr = "Overall Bias: " + str(bias) + " mmag \n" + \
        "Overall Scatter: " + str(scatter) + " mmag"
    plt.text(np.percentile(dates, 60), np.mean(truth_mag), textstr,
             fontsize=14, verticalalignment="top", bbox=props)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.errorbar(dates, residuals, yerr=sigma_mag, fmt="o", color="k")
    plt.axhline(0, ls="--", color="k")
    plt.ylabel("Mag Residuals (Model - Truth)")

    plt.ylabel("Mag Residuals (Model - Truth)")
    plt.xlabel("MJD")
    plt.ylim(np.min(residuals) - 0.1, np.max(residuals) + 0.1)

    plt.axhline(0.005, color="r", ls="--")
    plt.axhline(-0.005, color="r", ls="--", label="5 mmag photometry")

    plt.axhline(0.02, color="b", ls="--")
    plt.axhline(-0.02, color="b", ls="--", label="20 mmag photometry")
    plt.legend()

    if return_data:
        return mag.values, dates.values, \
            sigma_mag.values, truth_mag.values, bias, scatter


def plot_images(fileroot, size=11):

    imgdata = np.load("./results/images/"+str(fileroot)+"_images.npy")
    num_total_images = imgdata.shape[1]//size**2
    images = imgdata[0]
    sumimages = imgdata[1]

    fluxdata = pd.read_csv("./results/lightcurves/"+str(fileroot)+"_lc.csv")

    ra, dec = fluxdata["sn_ra"][0], fluxdata["sn_dec"][0]
    galra, galdec = fluxdata["host_ra"][0], fluxdata["host_dec"][0]

    hdul = fits.open("./results/images/"+str(fileroot)+"_wcs.fits")
    cutout_wcs_list = []
    for i, savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = snappl.AstropyWCS.from_header(savedwcs.header)
        cutout_wcs_list.append(newwcs)

    ra_grid, dec_grid, gridvals = np.load("./results/images/"
                                          + str(fileroot)+"_grid.npy")

    plt.figure(figsize=(15, 3*num_total_images))

    for i, wcs in enumerate(cutout_wcs_list):

        extent = [-0.5, size-0.5, -0.5, size-0.5]
        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)
        object_x, object_y = wcs.world_to_pixel(ra, dec)
        galx, galy = wcs.world_to_pixel(galra, galdec)

        plt.subplot(len(cutout_wcs_list), 4, 4*i+1)
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx, yy, s=1, c="k", vmin=vmin, vmax=vmax)
        plt.title("True Image")
        plt.scatter(object_x, object_y, c="r", s=8, marker="*")
        plt.scatter(galx, galy, c="b", s=8, marker="*")
        imshow = plt.imshow(images[i*size**2:
                            (i+1)*size**2].reshape(size, size),
                            origin="lower", extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)

        ############################################

        plt.subplot(len(cutout_wcs_list), 4, 4*i+2)
        plt.title("Model")

        im1 = sumimages[i*size**2:(i+1)*size**2].reshape(size, size)
        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)

        vmin = imshow.get_clim()[0]
        vmax = imshow.get_clim()[1]

        plt.imshow(im1, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        ############################################
        plt.subplot(len(cutout_wcs_list), 4, 4*i+3)
        plt.title("Residuals")
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx, yy, s=1, c=gridvals,  vmin=vmin, vmax=vmax)
        res = images - sumimages
        current_res = res[i*size**2:(i+1)*size**2].reshape(size, size)
        plt.imshow(current_res, extent=extent, origin="lower", cmap="seismic",
                   vmin=-100, vmax=100)
        plt.colorbar(fraction=0.046, pad=0.14)

    plt.subplots_adjust(wspace=0.4, hspace=0.3)


def get_galsim_SED(SNID, date, sn_path, fetch_SED, obj_type="SN"):
    """Return the appropriate SED for the object on the day. Since SN SEDs
    are time dependent but stars are not, we need to handle them differently.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data
    fetch_SED: If true, fetch true SED from the database, otherwise return a
                flat SED.
    obj_type: the type of object (SN or star)

    Internal Variables:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom

    Returns:
    sed: galsim.SED object
    """
    if fetch_SED:
        if obj_type == "SN":
            lam, flambda = get_SN_SED(SNID, date, sn_path)
        if obj_type == "star":
            lam, flambda = get_star_SED(SNID, sn_path)
    else:
        lam, flambda = [1000, 26000], [1, 1]

    sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                     wave_type="Angstrom", flux_type="fphotons")

    return sed


def get_star_SED(SNID, sn_path):
    """Return the appropriate SED for the star.
    Inputs:
    SNID: the ID of the object
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom (numpy  array of floats)
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
             (numpy array of floats)
    """
    filenum = find_parquet(SNID, sn_path, obj_type="star")
    pqfile = open_parquet(filenum, sn_path, obj_type="star")
    file_name = pqfile[pqfile["id"] == str(SNID)]["sed_filepath"].values[0]
    # SED needs to move out to snappl
    fullpath = pathlib.Path(Config.get().value("photometry.campari." +
                            "paths.sims_sed_library")) / file_name
    sed_table = pd.read_csv(fullpath,  compression="gzip", sep=r"\s+",
                            comment="#")
    lam = sed_table.iloc[:, 0]
    flambda = sed_table.iloc[:, 1]
    return np.array(lam), np.array(flambda)


def get_SN_SED(SNID, date, sn_path):
    """Return the appropriate SED for the supernova on the given day.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
    """
    filenum = find_parquet(SNID, sn_path, obj_type="SN")
    file_name = "snana" + "_" + str(filenum) + ".hdf5"
    fullpath = os.path.join(sn_path, file_name)
    # Setting locking=False on the next line becasue it seems that you can't
    #   open an h5py file unless you have write access to... something.
    #   Not sure what.  The directory where it exists?  We won't
    #   always have that.  It's scary to set locking to false, because it
    #   subverts all kinds of safety stuff that hdf5 does.  However,
    #   because these files were created once in this case, it's not actually
    #   scary, and we expect them to be static.  Locking only matters if you
    #   think somebody else might change the file
    #   while you're in the middle of reading bits of it.
    sed_table = h5py.File(fullpath, "r", locking=False)
    sed_table = sed_table[str(SNID)]
    flambda = sed_table["flambda"]
    lam = sed_table["lambda"]
    mjd = sed_table["mjd"]
    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    max_days_cutoff = 10
    closest_days_away = np.min(np.abs(np.array(mjd) - date))

    if closest_days_away > max_days_cutoff:
        Lager.warning(f"WARNING: No SED data within {max_days_cutoff} days of "
                      f"date. \n The closest SED is {closest_days_away} days away.")
    return np.array(lam), np.array(flambda[bestindex])


def make_contour_grid(image, wcs, numlevels=None, percentiles=[0, 90, 98, 100],
                      subsize=4):
    """Construct a "contour grid" which allocates model grid points to model
    the background galaxy according to the brightness of the image. This is
    an alternate version of make_adaptive_grid that results in a more
    continuous model grid point layout than make_adaptive_grid.
    While make_adaptive_grid visits each pixel and places a certain number of
    points, this function creates a smooth interpolation of the image to choose
    model point locations more densely in brighter regions.

    It does this as follows:
        1. Create a linear interoplation of the image.
        Start a loop:
        2. Create a grid of points that are evenly spaced in pixel space.
        3. For each of these points, check which brightness bin they fall into,
           using the linear interpolation.
        4. If this point is in the correct brightness bin, add it to the grid.
            If not, it does not get added.
        5. Increase the point density, and move to the next higher brightness
            bin.

    Here's a schematic:
    Our Image:  Binned by brightness:
                          ───────          ·····              ·····
            ░░░░░░        │     │          ·   ·              ·:::·
            ░▒▒▒▒░        │ ┌─┐ │          ·   ·              ·:::·
            ░▒██▒░        │ │ │ │          ·   ·              ·:::·
            ░▒██▒░        │ └─┘ │          ·   ·              ·:::·
            ░▒▒▒▒░        │     │          ·   ·              ·:::·
            ░░░░░░        │     │          ·····              ·····
                          ───────            ^                 ^
                            Add sparse model points, then dense model points.


    This model allows for the grid density to change smoothly across pixels,
    and avoids the problem of awkward gaps between model points across pixels.

    Inputs:
    image: 2D numpy array of floats of shape (size x size), the image to build
    the grid on.
    wcs: snappl.wcs.BaseWCS object

    percentiles: list of floats, the percentiles to use to bin the image. The
                more bins, the more possible grid points could be placed in
                that pixel.

    subsize: int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.

    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.
    """
    size = image.shape[0]
    x = np.arange(0, size, 1.0)
    y = np.arange(0, size, 1.0)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    xg = xg.ravel()
    yg = yg.ravel()
    Lager.debug("Grid type: contour")

    if numlevels is not None:
        levels = list(np.linspace(np.min(image), np.max(image), numlevels))
    else:
        levels = list(np.percentile(image, percentiles))

    Lager.debug(f"Using levels: {levels} in make_contour_grid")

    interp = RegularGridInterpolator((x, y), image, method="linear",
                                     bounds_error=False, fill_value=None)

    aa = interp((xg, yg))

    x_totalgrid = []
    y_totalgrid = []

    for i in range(len(levels) - 1):
        zmin = levels[i]
        zmax = levels[i+1]
        # Generate a grid that gets finer each iteration of the loop. For
        # instance, in brightness bin 1, 1 point per pixel, in brightness bin
        # 2, 4 points per pixel (2 in each direction), etc.
        x = np.arange(0, size, 1/(i+1))
        y = np.arange(0, size, 1/(i+1))
        if i == 0:
            x = x[np.where(np.abs(x - size/2) < subsize)]
            y = y[np.where(np.abs(y - size/2) < subsize)]
        xg, yg = np.meshgrid(x, y, indexing="ij")
        aa = interp((xg, yg))
        xg = xg[np.where((aa > zmin) & (aa <= zmax))]
        yg = yg[np.where((aa > zmin) & (aa <= zmax))]
        x_totalgrid.extend(xg)
        y_totalgrid.extend(yg)

    xx, yy = y_totalgrid, x_totalgrid  # Here is another place I need to flip
    # x and y. I'd like this to be more rigorous or at least clear.
    xx = np.array(xx)
    yy = np.array(yy)
    xx = xx.flatten()
    yy = yy.flatten()
    Lager.debug(f"Built a grid with {np.size(xx)} points")
    first_n = 5
    Lager.debug(f"First {first_n} grid points: {xx[:first_n]}, {yy[:first_n]}")

    ra_grid, dec_grid = wcs.pixel_to_world(xx, yy)

    return ra_grid, dec_grid


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

    exptime = {"F184": 901.175,
               "J129": 302.275,
               "H158": 302.275,
               "K213": 901.175,
               "R062": 161.025,
               "Y106": 302.275,
               "Z087": 101.7}

    area_eff = roman.collecting_area
    zp = roman.getBandpasses()[band].zeropoint if zp is None else zp
    mag = -2.5 * np.log10(flux) + 2.5*np.log10(exptime[band]*area_eff) + zp
    magerr = (2.5 / np.log(10) * (sigma_flux / flux))
    magerr[flux < 0] = np.nan
    return mag, magerr, zp


def build_lightcurve(ID, exposures, sn_path, roman_path, confusion_metric, flux,
                     use_roman, band, object_type, sigma_flux):

    """This code builds a lightcurve datatable from the output of the SMP
       algorithm.

    Input:
    ID (int): supernova ID
    exposures (table): table of exposures used in the SMP algorithm
    sn_path (str): path to supernova data
    confusion_metric (float): the confusion metric derived in the SMP algorithm
    num_detect_images (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm
    use_roman (bool): whether or not the lightcurve was built using Roman PSF
    band (str): the bandpass of the images used

    Returns:
    lc: a QTable containing the lightcurve data
    """

    detections = exposures[np.where(exposures["DETECTED"])]
    parq_file = find_parquet(ID, path=sn_path, obj_type=object_type)
    df = open_parquet(parq_file, path=sn_path, obj_type=object_type)

    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)
    sim_true_flux = []
    sim_realized_flux = []
    for pointing, SCA in zip(detections['Pointing'], detections['SCA']):
        catalogue_path = roman_path+f'/RomanTDS/truth/{band}/{pointing}/' \
                        + f'Roman_TDS_index_{band}_{pointing}_{SCA}.txt'
        cat = pd.read_csv(catalogue_path, sep=r"\s+", skiprows=1,
                          names=['object_id', 'ra', 'dec', 'x', 'y',
                                 'realized_flux', 'flux', 'mag', 'obj_type'])
        cat = cat[cat['object_id'] == ID]
        sim_true_flux.append(cat['flux'].values[0])
        sim_realized_flux.append(cat['realized_flux'].values[0])
    sim_true_flux = np.array(sim_true_flux)
    sim_realized_flux = np.array(sim_realized_flux)

    sim_sigma_flux = 0  # These are truth values!
    sim_realized_mag, _, _ = calc_mag_and_err(sim_realized_flux,
                                              sim_sigma_flux, band)
    sim_true_mag, _, _ = calc_mag_and_err(sim_true_flux,
                                          sim_sigma_flux, band)

    if object_type == "SN":
        df_object_row = df.loc[df.id == ID]
    if object_type == "star":
        df_object_row = df.loc[df.id == str(ID)]

    if object_type == "SN":
        meta_dict = {"confusion_metric": confusion_metric,
                     "host_sep": df_object_row["host_sn_sep"].values[0],
                     "host_mag_g": df_object_row["host_mag_g"].values[0],
                     "obj_ra": df_object_row["ra"].values[0],
                     "obj_dec": df_object_row["dec"].values[0],
                     "host_ra": df_object_row["host_ra"].values[0],
                     "host_dec": df_object_row["host_dec"].values[0]}
    else:
        meta_dict = {"ra": df_object_row["ra"].values[0],
                     "dec": df_object_row["dec"].values[0]}

    data_dict = {"MJD": detections["date"], "flux": flux,
                 "flux_error": sigma_flux, "mag": mag,
                 "mag_err": magerr,
                 "band": np.full(np.size(mag), band),
                 "zeropoint": np.full(np.size(mag), zp),
                 "SIM_realized_flux": sim_realized_flux,
                 "SIM_true_flux": sim_true_flux,
                 "SIM_realized_mag": sim_realized_mag,
                 "SIM_true_mag": sim_true_mag}
    units = {"MJD": u.d, "SIM_realized_flux": "",  "flux": "",
             "flux_error": "", "SIM_realized_mag": "",
             "SIM_true_flux": "", "SIM_true_mag": ""}

    return QTable(data=data_dict, meta=meta_dict, units=units)


def build_lightcurve_sim(supernova, flux, sigma_flux):
    """This code builds a lightcurve datatable from the output of the SMP
        algorithm if the user simulated their own lightcurve.

    Inputs
    supernova (array): the true lightcurve
    num_detect_images (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm

    Returns
    lc: a QTable containing the lightcurve data
    """
    sim_MJD = np.arange(0, np.size(supernova), 1)
    data_dict = {"MJD": sim_MJD, "flux": flux,
                 "flux_error": sigma_flux, "SIM_flux": supernova}
    meta_dict = {}
    units = {"MJD": u.d, "SIM_flux": "",  "flux": "", "flux_error": ""}
    return QTable(data=data_dict, meta=meta_dict, units=units)


def save_lightcurve(lc, identifier, band, psftype, output_path=None,
                    overwrite=True):
    """This function parses settings in the SMP algorithm and saves the
    lightcurve to an ecsv file with an appropriate name.
    Input:
    lc: the lightcurve data
    identifier (str): the supernova ID or "simulated"
    band (str): the bandpass of the images used
    psftype (str): "romanpsf" or "analyticpsf"
    output_path (str): the path to save the lightcurve to.  Defaults to
      config value phtometry.campari.paths.output_dir

    Returns:
    None, saves the lightcurve to a ecsv file.
    The file name is:
    <output_path>/identifier_band_psftype_lc.ecsv
    """

    output_path = Config.get().value("photometry.campari.paths.output_dir") \
        if output_path is None else output_path
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    lc_file = output_path / f"{identifier}_{band}_{psftype}_lc.ecsv"

    Lager.info(f"Saving lightcurve to {lc_file}")
    lc.write(lc_file, format="ascii.ecsv", overwrite=overwrite)


def banner(text):
    length = len(text) + 8
    message = "\n" + "#" * length + "\n"+"#   " + text + "   # \n" + "#" \
              * length
    Lager.debug(message)


def get_galsim_SED_list(ID, dates, fetch_SED, object_type, sn_path,
                        sed_out_dir=None):
    """Return the appropriate SED for the object for each observation.
    If you are getting truth SEDs, this function calls get_SED on each exposure
    of the object. Then, get_SED calls get_SN_SED or get_star_SED depending on
    the object type.
    If you are not getting truth SEDs, this function returns a flat SED for
    each exposure.

    Inputs:
    ID: the ID of the object
    exposures: the exposure table returned by fetchImages.
    fetch_SED: If true, get the SED from truth tables.
               If false, return a flat SED for each expsoure.
    object_type: the type of object (SN or star)
    sn_path: the path to the supernova data

    Returns:
    sedlist: list of galsim SED objects, length equal to the number of
             detection images.
    """
    sed_list = []
    if isinstance(dates, float):
        dates = [dates]  # If only one date is given, make it a list.
    for date in dates:
        sed = get_galsim_SED(ID, date, sn_path, obj_type=object_type,
                             fetch_SED=fetch_SED)
        sed_list.append(sed)
        if sed_out_dir is not None:
            sed_df = pd.DataFrame({"lambda": sed._spec.x,
                                   "flux": sed._spec.f})
            sed_df.to_csv(f"{sed_out_dir}/sed_{ID}_{date}.csv", index=False)

    return sed_list


def prep_data_for_fit(images, sn_matrix, wgt_matrix):
    """This function takes the data from the images and puts it into the form
    such that we can analytically solve for the best fit using linear algebra.

    n = total number of images
    s = image size (so the image is s x s)
    d = number of detection images

    Inputs:
    images: list of snappl Image objects. List of length n objects.
    sn_matrix: list of np arrays of SN models. List of length d of sxs arrays.
    wgt_matrix: list of np arrays of weights. List of length n of sxs arrays.

    Outputs:
    images: 1D array of image data. Length n*s^2
    err: 1D array of error data. Length n*s^2
    sn_matrix: A 2D array of SN models, with the SN models placed in the
                correct rows and columns, see comment below. Shape (n*s^2, n)
    wgt_matrix: 1D array of weights. Length n*s^2
    """
    Lager.debug("Prep data for fit")
    size_sq = images[0].image_shape[0]**2
    tot_num = len(images)
    det_num = len(sn_matrix)

    # Flatten into 1D arrays
    err = np.concatenate([im.noise.flatten() for im in images])
    image_data = np.concatenate([im.data.flatten() for im in images])

    # The final design matrix for our fit should have dimensions:
    # (total number of pixels in all images, number of model components)
    # Then, the first s^2 rows of the matrix correspond to the first image,
    # the next s^2 rows to the second image, etc.,  where s is the size of the
    # image. For the SN model, the flux in each image is ostensibly different.
    # Therefore we need a unique flux for each image, and we don't want the
    # flux of the supernova in one image to affect the flux in another image.
    # Therefore, we need to place the supernova model in the correct image
    # (i.e. the correct rows of the design matrix) and zero out all of the
    # others. We'll do this by initializing a matrix of zeros, and then filling
    # in the SN model in the correct place in the loop below:

    psf_zeros = np.zeros((np.size(image_data), tot_num))
    for i in range(det_num):
        sn_index = tot_num - det_num + i  # We only want to edit SN columns.
        psf_zeros[
            (sn_index) * size_sq:  # Fill in rows s^2 * image number...
            (sn_index + 1) * size_sq,  # ... to s^2 * (image number + 1) ...
            sn_index] = sn_matrix[i]  # ...in the correct column.
    sn_matrix = np.vstack(psf_zeros)
    wgt_matrix = np.array(wgt_matrix)
    wgt_matrix = np.hstack(wgt_matrix)

    return image_data, err, sn_matrix, wgt_matrix


def extract_sn_from_parquet_file_and_write_to_csv(parquet_file, sn_path,
                                                  output_path,
                                                  mag_limits=None):
    """Convenience function for getting a list of SN IDs that obey some
    conditions from a parquet file. This is not used anywhere in the main
    algorithm.

    Inputs:
    parquet_file: the path to the parquet file
    sn_path: the path to the supernova data
    mag_limits: a tuple of (min_mag, max_mag) to filter the SNe by
                peak magnitude. If None, no filtering is done.

    Output:
    Saves a csv file of the SN_IDs of supernovae from the parquet file that
    pass mag cuts. If none are found, raise a ValueError.
    """
    # Get the supernova IDs from the parquet file
    df = open_parquet(parquet_file, sn_path, obj_type="SN")
    if mag_limits is not None:
        min_mag, max_mag = mag_limits
        # This can't always be just g band I think. TODO
        df = df[(df["peak_mag_g"] >= min_mag) & (df["peak_mag_g"] <= max_mag)]
    SN_ID = df.id.values
    SN_ID = SN_ID[np.log10(SN_ID) < 8]  # The 9 digit SN_ID SNe are weird for
    # some reason. They only seem to have 1 or 2 images ever. TODO
    SN_ID = np.array(SN_ID, dtype=int)
    Lager.info(f"Found {np.size(SN_ID)} supernovae in the given range.")
    if np.size(SN_ID) == 0:
        raise ValueError("No supernovae found in the given range.")

    pd.DataFrame(SN_ID).to_csv(output_path, index=False, header=False)
    Lager.info(f"Saved to {output_path}")


def extract_star_from_parquet_file_and_write_to_csv(parquet_file, sn_path,
                                                    output_path,
                                                    ra=None,
                                                    dec=None,
                                                    radius=None):
    """Convenience function for getting a list of star IDs
    from a parquet file. The stars can be cone-searched for by passing a
    central coordinate and a radius.
    This is not used anywhere in the main algorithm.

    Inputs:
    parquet_file: int,  the number label of the parquet file to use.
    sn_path: str, the path to the supernova data
    ra: float, the central RA of the region to search in
    dec: float, the central Dec of the region to search in
    radius: float, the radius over which cone search is performed. Can have
                    any angular astropy.unit attached to it. If no unit is
                    included, the function will produce a warning and then
                    automatically assume you meant degrees.
    If no ra, dec, and radius are passed, no cone search
    is performed and the IDs of the entire parquet file are returned.
    If one or two of the above arguments is passed but not all three, the
    cone search is not performed.

    Output:
    Saves a csv file to output_path of the IDs of stars from the parquet
    file that pass location cuts. If none are found, raise a ValueError.
    """
    if not hasattr(radius, "unit") and radius is not None:
        Lager.warning("extract_star_from_parquet_file_and_write_to_csv " +
                      "a radius argument with no units. Assuming degrees.")
        radius *= u.deg

    df = open_parquet(parquet_file, sn_path, obj_type="star")
    df = df[df["object_type"] == "star"]

    if radius is not None and (ra is not None and dec is not None):
        center_coord = SkyCoord(ra*u.deg, dec*u.deg)
        df_coords = SkyCoord(ra=df["ra"].values*u.deg,
                             dec=df["dec"].values*u.deg)
        sep = center_coord.separation(df_coords)
        df = df[sep < radius]

    star_ID = df.id.values
    star_ID = np.array(star_ID, dtype=int)
    Lager.info(f"Found {np.size(star_ID)} stars in the given range.")
    if np.size(star_ID) == 0:
        raise ValueError("No stars found in the given range.")
    pd.DataFrame(star_ID).to_csv(output_path, index=False, header=False)
    Lager.info(f"Saved to {output_path}")


def run_one_object(ID, ra, dec, object_type, exposures, num_total_images, num_detect_images,
                   roman_path, sn_path, size, band, fetch_SED, sedlist,
                   use_real_images, use_roman, subtract_background,
                   make_initial_guess, initial_flux_guess, weighting, method,
                   grid_type, pixel, source_phot_ops,
                   lc_start, lc_end, do_xshift, bg_gal_flux, do_rotation, airy,
                   mismatch_seds, deltafcn_profile, noise, check_perfection,
                   avoid_non_linearity, sim_gal_ra_offset, sim_gal_dec_offset,
                   spacing, percentiles,
                   draw_method_for_non_roman_psf="no_pixel"):
    Lager.debug(f"ID: {ID}")
    psf_matrix = []
    sn_matrix = []

    # This is a catch for when I'm doing my own simulated WCSs
    util_ref = None

    percentiles = []
    roman_bandpasses = galsim.roman.getBandpasses()

    if use_real_images:
        # Using exposures Table, load those Pointing/SCAs as images.
        cutout_image_list, image_list = fetchImages(exposures, ra, dec, size, subtract_background, roman_path,
                                                    object_type)

        if num_total_images != len(exposures) or num_detect_images != len(exposures[exposures["DETECTED"]]):
            Lager.debug(f"Updating image numbers to {num_total_images}" + f" and {num_detect_images}")
            num_total_images = len(exposures)
            num_detect_images = len(exposures[exposures["DETECTED"]])

    else:
        # Simulate the images of the SN and galaxy.
        banner("Simulating Images")
        images, im_wcs_list, cutout_wcs_list, sim_lc, util_ref = \
            simulate_images(num_total_images, num_detect_images, ra, dec,
                            sim_gal_ra_offset, sim_gal_dec_offset,
                            do_xshift, do_rotation, noise=noise,
                            use_roman=use_roman, roman_path=roman_path,
                            size=size, band=band,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            source_phot_ops=source_phot_ops,
                            mismatch_seds=mismatch_seds)
        object_type = "SN"
        err = np.ones_like(images)

    # Build the background grid
    if not grid_type == "none":
        if object_type == "star":
            Lager.warning("For fitting stars, you probably dont want a grid.")
        ra_grid, dec_grid = makeGrid(grid_type, cutout_image_list, ra, dec,
                                     percentiles=percentiles)
    else:
        ra_grid = np.array([])
        dec_grid = np.array([])

    # Using the images, hazard an initial guess.
    # The num_total_images - num_detect_images check is to ensure we have
    # pre-detection images. Otherwise, initializing the model guess does not
    # make sense.
    num_nondetect_images = num_total_images - num_detect_images
    if make_initial_guess and num_nondetect_images != 0:
        x0test = generateGuess(cutout_image_list[:num_nondetect_images],
                               ra_grid, dec_grid)
        x0_vals_for_sne = np.full(num_total_images, initial_flux_guess)
        x0test = np.concatenate([x0test, x0_vals_for_sne], axis=0)
        Lager.debug(f"setting initial guess to {initial_flux_guess}")

    else:
        x0test = None

    banner("Building Model")

    # Calculate the Confusion Metric

    if use_real_images and object_type == "SN" and num_detect_images > 1:
        sed = get_galsim_SED(ID, exposures, sn_path, fetch_SED=False)
        object_x, object_y = image_list[0].get_wcs().world_to_pixel(ra, dec)
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
        pointing, SCA = exposures["Pointing"][0], exposures["SCA"][0]
        psf_source_array = construct_psf_source(x, y, pointing, SCA,
                                                stampsize=size,
                                                x_center=object_x, y_center=object_y,
                                                sed=sed)
        confusion_metric = np.dot(images[0].flatten(), psf_source_array)

        Lager.debug(f"Confusion Metric: {confusion_metric}")
    else:
        confusion_metric = 0
        Lager.debug("Confusion Metric not calculated")

    # Build the backgrounds loop
    # TODO: Zip all the things you index [i] on directly and loop over
    # them.
    for i in range(num_total_images):
        # Passing in None for the PSF means we use the Roman PSF.
        drawing_psf = None if use_roman else airy

        whole_sca_wcs = image_list[i].get_wcs()
        object_x, object_y = whole_sca_wcs.world_to_pixel(ra, dec)

        # Build the model for the background using the correct psf and the
        # grid we made in the previous section.

        # TODO: Put this in snappl
        if use_real_images:
            util_ref = roman_utils(config_file=pathlib.Path(Config.get().value
                                   ("photometry.campari.galsim.tds_file")),
                                   visit=exposures["Pointing"][i],
                                   sca=exposures["SCA"][i])

        # If no grid, we still need something that can be concatenated in the
        # linear algebra steps, so we initialize an empty array by default.
        background_model_array = np.empty((size**2, 0))
        Lager.debug(f"ra_grid {ra_grid[:5]}")
        Lager.debug(f"dec_grid {dec_grid[:5]}")
        Lager.debug("Constructing background model array for image " + str(i))
        if grid_type != "none":
            background_model_array = \
                construct_psf_background(ra_grid, dec_grid,
                                         cutout_image_list[i].get_wcs(),
                                         object_x, object_y, size, psf=drawing_psf,
                                         pixel=pixel,
                                         util_ref=util_ref, band=band)

        # TODO comment this
        if not subtract_background:
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

        # TODO make this not bad
        if num_detect_images != 0 and \
           i >= num_total_images - num_detect_images:
            object_x, object_y = cutout_image_list[i]\
                       .get_wcs().world_to_pixel(ra, dec)

            if use_roman:
                if use_real_images:
                    pointing = exposures["Pointing"][i]
                    SCA = exposures["SCA"][i]
                else:
                    pointing = 662
                    SCA = 11
                # sedlist is the length of the number of supernova
                # detection images. Therefore, when we iterate onto the
                # first supernova image, we want to be on the first element
                # of sedlist. Therefore, we subtract by the number of
                # predetection images: num_total_images - num_detect_images.
                sn_index = i - (num_total_images - num_detect_images)
                Lager.debug(f"Using SED #{sn_index}")
                sed = sedlist[sn_index]
                # object_x and object_y are the exact coords of the SN in the SCA frame.
                # x and y are the pixels the image has been cut out on, and
                # hence must be ints. Before, I had object_x and object_y as SN coords in the cutout frame, hence this switch.
                # In snappl, centers of pixels occur at integers, so the center of the lower left pixel is (0,0).
                # Therefore, if you are at (0.2, 0.2), you are in the lower left pixel, but at (0.6, 0.6), you have
                # crossed into the next pixel, which is (1,1). So we need to round everything between -0.5 and 0.5 to 0,
                # and everything between 0.5 and 1.5 to 1, etc. This code below does that, and follows how snappl does
                # it. For more detail, see the docstring of get_stamp in the PSF class definition of snappl.
                x = int(np.floor(object_x + 0.5))
                y = int(np.floor(object_y + 0.5))
                Lager.debug(f"x, y, object_x, object_y, {x, y, object_x, object_y}")
                psf_source_array =\
                    construct_psf_source(x, y, pointing, SCA,
                                         stampsize=size, x_center=object_x,
                                         y_center=object_y, sed=sed,
                                         photOps=source_phot_ops)
            else:
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
    Lager.debug(f"{psf_matrix.shape} psf matrix shape")

    # Add in the supernova images to the matrix in the appropriate location
    # so that it matches up with the image it represents.
    # All others should be zero.

    # Get the weights
    if weighting:
        wgt_matrix = get_weights(cutout_image_list, ra, dec)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[1])

    images, err, sn_matrix, wgt_matrix =\
        prep_data_for_fit(cutout_image_list, sn_matrix, wgt_matrix)

    # Calculate amount of the PSF cut out by setting a distance cap
    test_sn_matrix = np.copy(sn_matrix)
    test_sn_matrix[np.where(wgt_matrix == 0), :] = 0
    Lager.debug(f"SN PSF Norms Pre Distance Cut:{np.sum(sn_matrix, axis=0)}")
    Lager.debug("SN PSF Norms Post Distance Cut:"
                f"{np.sum(test_sn_matrix, axis=0)}")

    # Combine the background model and the supernova model into one matrix.

    psf_matrix = np.hstack([psf_matrix, sn_matrix])

    banner("Solving Photometry")

    # These if statements can definitely be written more elegantly.
    if not make_initial_guess:
        x0test = np.zeros(psf_matrix.shape[1])

    if not subtract_background:
        x0test = np.concatenate([x0test, np.zeros(num_total_images)], axis=0)

    Lager.debug(f"shape psf_matrix: {psf_matrix.shape}")
    Lager.debug(f"shape wgt_matrix: {wgt_matrix.reshape(-1, 1).shape}")
    Lager.debug(f"image shape: {images.shape}")

    if method == "lsqr":
        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix, x0=x0test, atol=1e-12,
                              btol=1e-12, iter_lim=300000, conlim=1e10)
        X, istop, itn, r1norm = lsqr[:4]
        Lager.debug(f"Stop Condition {istop}, iterations: {itn}," +
                    f"r1norm: {r1norm}")
    flux = X[-num_detect_images:]
    inv_cov = psf_matrix.T @ np.diag(wgt_matrix) @ psf_matrix

    try:
        cov = np.linalg.inv(inv_cov)
    except LinAlgError:
        cov = np.linalg.pinv(inv_cov)

    Lager.debug(f"cov diag: {np.diag(cov)[-num_detect_images:]}")
    sigma_flux = np.sqrt(np.diag(cov)[-num_detect_images:])
    Lager.debug(f"sigma flux: {sigma_flux}")

    # Using the values found in the fit, construct the model images.
    pred = X*psf_matrix
    sumimages = np.sum(pred, axis=1)

    # TODO: Move this to a separate function
    if check_perfection:
        if avoid_non_linearity:
            f = 1
        else:
            f = 5000
        if grid_type == "single":
            X[0] = f
        else:
            X = np.zeros_like(X)
            X[106] = f

    if use_real_images:
        # Eventually I might completely separate out simulated SNe, though I
        # am hesitant to do that as I want them to be treated identically as
        # possible. In the meantime, just return zeros for the simulated lc
        # if we aren't simulating.
        sim_lc = np.zeros(num_detect_images)
    return flux, sigma_flux, images, sumimages, exposures, ra_grid, dec_grid, \
        wgt_matrix, confusion_metric, X, \
        [im.get_wcs() for im in cutout_image_list], sim_lc


def plot_image_and_grid(image, wcs, ra_grid, dec_grid):
    Lager.debug(f"WCS: {type(wcs)}")
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    plt.imshow(image, origin="lower", cmap="gray")
    plt.scatter(ra_grid, dec_grid)


def load_SEDs_from_directory(sed_directory, wave_type="Angstrom",
                             flux_type="fphotons"):
    """This function loads SEDs from a directory of SED files.
    Inputs:
    sed_directory: str, the path to the directory containing the SED files.

    Returns:
    sed_list: list of galsim SED objects. (Temporary until we remove galsim)
    """
    sed_list = []
    for file in os.listdir(sed_directory):
        sed_path = os.path.join(sed_directory, file)
        sed_table = pd.read_csv(sed_path)
        flambda = sed_table["flux"]
        lam = sed_table["lambda"]
        # Assuming units are Angstroms how can I check this?
        sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                         wave_type=wave_type, flux_type=flux_type)
        sed_list.append(sed)
    return sed_list
