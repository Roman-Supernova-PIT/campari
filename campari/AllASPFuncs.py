# Standard Library
import os
import pathlib
import warnings

# Common Library
import astropy.table as tb
import galsim
import glob
import h5py
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.io import fits
from astropy.table import QTable, Table, hstack
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from galsim import roman
import healpy as hp
from numpy.linalg import LinAlgError
from roman_imsim.utils import roman_utils
from scipy.interpolate import RegularGridInterpolator
import yaml

# SN-PIT
from snappl.image import OpenUniverse2024FITSImage
from snappl.diaobject import DiaObject
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger
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
    SNLogger.debug(f'CRPIX1 {wcs.to_fits_header()["CRPIX1"]}')
    if wcs.to_fits_header()["CRPIX1"] == 2044 and wcs.to_fits_header()["CRPIX2"] == 2044:

        SNLogger.warning("This WCS is centered exactly on the center of the image, make_regular_grid is expecting a"
                         "cutout WCS, this is likely not a cutout WCS.")
    if subsize > size:
        SNLogger.warning("subsize is larger than the image size. " +
                         f"{size} > {subsize}. This would cause model points to" +
                         " be placed outside the image. Reducing subsize to" +
                         " match the image size.")
        subsize = size

    SNLogger.debug("Grid type: regularly spaced")
    difference = int((size - subsize)/2)

    x = difference + np.arange(0, subsize, spacing)
    y = difference + np.arange(0, subsize, spacing)
    SNLogger.debug(f"Grid spacing: {spacing}")

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    SNLogger.debug(f"Built a grid with {np.size(xx)} points")

    # Astropy takes (y, x) order:
    ra_grid, dec_grid = wcs.pixel_to_world(yy, xx)

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
        SNLogger.warning("subsize is larger than the image size " +
                         f"{size} > {subsize}. This would cause model points to" +
                         " be placed outside the image. Reducing subsize to" +
                         " match the image size.")
        subsize = size

    SNLogger.debug("image shape: {}".format(np.shape(image)))
    SNLogger.debug("Grid type: adaptive")
    # Bin the image in logspace and allocate grid points based on the
    # brightness.

    difference = int((size - subsize)/2)

    x = difference + np.arange(0, subsize, 1)
    y = difference + np.arange(0, subsize, 1)

    if percentiles.sort() != percentiles:
        SNLogger.warning("Percentiles not in ascending order. Sorting them.")
        percentiles.sort()
        SNLogger.warning(f"Percentiles: {percentiles}")

    imcopy = np.copy(image)
    # We need to make sure that the image is not zero, otherwise we get
    # infinities in the log space.
    imcopy[imcopy <= 0] = 1e-10
    imcopy = np.log(imcopy)
    bins = [0]
    bins.extend(np.nanpercentile(imcopy, percentiles))
    bins.append(100)
    SNLogger.debug(f"BINS: {bins}")

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
                xs.append(x)
                ys.append(y)
            else:
                xx = np.linspace(x - subpixel_grid_width/2,
                                 x + subpixel_grid_width/2, num+2)[1:-1]
                yy = np.linspace(y - subpixel_grid_width/2,
                                 y + subpixel_grid_width/2, num+2)[1:-1]
                X, Y = np.meshgrid(xx, yy)
                ys.extend(list(Y.flatten()))
                xs.extend(list(X.flatten()))

    xx = np.array(xs).flatten()
    yy = np.array(ys).flatten()

    SNLogger.debug(f"Built a grid with {np.size(xx)} points")

    # Astropy takes (y,x) order:
    ra_grid, dec_grid = wcs.pixel_to_world(yy, xx)

    return ra_grid, dec_grid


def generate_guess(imlist, ra_grid, dec_grid):
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
        grid_point_vals = np.atleast_1d(np.zeros_like(xx))
        # For testing purposes, sometimes the grid is exactly one point, so we force it to be 1d.
        xx = np.atleast_1d(xx)
        yy = np.atleast_1d(yy)
        for imval, imxval, imyval in zip(im.flatten(),
                                         imx.flatten(), imy.flatten()):
            grid_point_vals[np.where((np.abs(xx - imxval) < 0.5) &
                                     (np.abs(yy - imyval) < 0.5))] = imval
        all_vals += grid_point_vals
    return all_vals/len(wcslist)


def construct_static_scene(ra, dec, sca_wcs, x_loc, y_loc, stampsize, psf=None, pixel=False,
                           util_ref=None, band=None):

    """Constructs the background model around a certain image (x,y) location
    and a given array of RA and DECs.
    Inputs:
    ra, dec: arrays of floats, RA and DEC values for the grid
    sca_wcs: the wcs of the entire image, i.e. the entire SCA. A snappl.wcs.BaseWCS object.
    x_loc, y_loc: floats,the pixel location of the object in the FULL image,
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

    assert util_ref is not None or psf is not None, "you must provide at least util_ref or psf"
    assert util_ref is not None or band is not None, "you must provide at least util_ref or band"

    # I call this x_sca to highlight that it's the location in the SCA, not the cutout.
    x_sca, y_sca = sca_wcs.world_to_pixel(ra, dec)
    # For testing purposes, sometimes the grid is exactly one point, so we force it to be 1d.
    x_sca = np.atleast_1d(x_sca)
    y_sca = np.atleast_1d(y_sca)
    bpass = roman.getBandpasses()[band]

    num_grid_points = np.size(x_sca)

    psfs = np.zeros((stampsize * stampsize, num_grid_points))

    sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                     interpolant="linear"),
                     wave_type="nm", flux_type="fphotons")

    if pixel:
        point = galsim.Pixel(0.1)*sed
    else:
        point = galsim.DeltaFunction()
        point *= sed

    point = point.withFlux(1, bpass)

    pointing = util_ref.visit
    sca = util_ref.sca

    psf_object = PSF.get_psf_object("ou24PSF", pointing=pointing, sca=sca, size=stampsize, include_photonOps=False)
    # See run_one_object documentation to explain this pixel coordinate conversion.
    x_loc = int(np.floor(x_loc + 0.5))
    y_loc = int(np.floor(y_loc + 0.5))

    # Loop over the grid points, draw a PSF at each one, and append to a list.
    for a, (x, y) in enumerate(zip(x_sca.flatten(), y_sca.flatten())):
        if a % 50 == 0:
            SNLogger.debug(f"Drawing PSF {a} of {num_grid_points}")
        psfs[:, a] = psf_object.get_stamp(x0=x_loc, y0=y_loc, x=x, y=y, flux=1.0, seed=None, input_wcs=sca_wcs).flatten()

    return psfs


def find_all_exposures(ra, dec, transient_start, transient_end, band, maxbg=None,
                       maxdet=None, return_list=False,
                       roman_path=None, pointing_list=None, sca_list=None,
                       truth="simple_model", image_selection_start=-np.inf, image_selection_end=np.inf):
    """This function finds all the exposures that contain a given supernova,
    and returns a list of them. Utilizes Rob's awesome database method to
    find the exposures. Humongous speed up thanks to this.

    Inputs:
    ra, dec: the RA and DEC of the supernova
    peak: the peak of the supernova
    transient_start, transient_end: floats, the first and last MJD of a detection of the transient,
        defines what which images contain transient light (and therefore recieve a single model point
        at the location of the transient) and which do not.
    maxbg: the maximum number of background images to consider
    maxdet: the maximum number of detected images to consider
    return_list: whether to return the exposures as a list or not
    stampsize: the size of the stamp to use
    roman_path: the path to the Roman data
    pointing_list: If this is passed in, only consider these pointings
    sca_list: If this is passed in, only consider these SCAs
    truth: If "truth" use truth images, if "simple_model" use simple model
            images.
    band: the band to consider
    image_selection_start, image_selection_end: floats, the first and last MJD of images to be used in the algorithm.
    explist: astropy.table.Table, the table of exposures that contain the
    supernova. The columns are:
        - pointing: the pointing of the exposure
        - sca: the SCA of the exposure
        - band: the band of the exposure
        - date: the MJD of the exposure
        - detected: whether the exposure contains a detection or not.
    """
    f = fits.open(roman_path +
                  "/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits")[1]
    f = f.data

    explist = tb.Table(names=("pointing", "sca", "filter", "date"),
                       dtype=("i8", "i4", "str",  "f8"))

    transient_start = np.atleast_1d(transient_start)[0]
    transient_end = np.atleast_1d(transient_end)[0]
    if not (isinstance(maxdet, (int, type(None))) & isinstance(maxbg, (int, type(None)))):
        raise TypeError("maxdet and maxbg must be integers or None, " +
                        f"not {type(maxdet), type(maxbg)}. Their values are {maxdet, maxbg}")

    # Rob's database method! :D

    server_url = "https://roman-desc-simdex.lbl.gov"
    req = requests.Session()
    result = req.post(f"{server_url}/findromanimages/containing=({ra},{dec})")
    if result.status_code != 200:
        raise RuntimeError(f"Got status code {result.status_code}\n"
                           "{result.text}")

    res = pd.DataFrame(result.json())[["pointing", "sca", "mjd", "filter"]]
    res.rename(columns={"mjd": "date"}, inplace=True)
    res = res.loc[res["filter"] == band].copy()

    # The first date cut selects images that are detections, the second
    # selects detections within the requested light curve window.
    det = res.loc[(res["date"] >= transient_start) & (res["date"] <= transient_end)].copy()
    det = det.loc[(det["date"] >= image_selection_start) & (det["date"] <= image_selection_end)]
    if maxdet is not None:
        det = det.iloc[:maxdet]
    det["detected"] = True

    if pointing_list is not None:
        det = det.loc[det["pointing"].isin(pointing_list)]
    bg = res.loc[(res["date"] < transient_start) | (res["date"] > transient_end)].copy()
    bg = bg.loc[(bg["date"] >= image_selection_start) & (bg["date"] <= image_selection_end)]

    if pointing_list is not None:
        bg = bg.loc[bg["pointing"].isin(pointing_list)]
    if isinstance(maxbg, int):
        bg = bg.iloc[:maxbg]
    bg["detected"] = False

    all_images = pd.concat([det, bg])
    all_images["filter"] = band

    explist = Table.from_pandas(all_images)
    explist.sort(["detected", "sca"])
    SNLogger.info("\n" + str(explist))

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
            SNLogger.debug(f"Found {obj_type} {ID} in {f}")
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


def construct_transient_scene(x, y, pointing, sca, stampsize=25, x_center=None,
                              y_center=None, sed=None, flux=1, photOps=True, sca_wcs=None):
    """Constructs the PSF around the point source (x,y) location, allowing for
        some offset from the center.
    Inputs:
    x, y: ints, pixel coordinates where the cutout is centered in the SCA
    pointing, sca: ints, the pointing and SCA of the image
    stampsize = int, size of cutout image used
    TODO: this defn below isn't correct
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

    SNLogger.debug(f"ARGS IN PSF SOURCE: \n x, y: {x, y} \n" +
                   f" pointing, sca: {pointing, sca} \n" +
                   f" stamp size: {stampsize} \n" +
                   f" x_center, y_center: {x_center, y_center} \n" +
                   f" sed: {sed} \n" +
                   f" flux: {flux}")

    assert sed is not None, "You must provide an SED for the source"

    if not photOps:
        # While I want to do this sometimes, it is very rare that you actually
        # want to do this. Thus if it was accidentally on while doing a normal
        # run, I'd want to know.
        SNLogger.warning("NOT USING PHOTON OPS IN PSF SOURCE")

    psf_object = PSF.get_psf_object("ou24PSF_slow", pointing=pointing, sca=sca,
                                    size=stampsize, include_photonOps=photOps)
    psf_image = psf_object.get_stamp(x0=x, y0=y, x=x_center, y=y_center, flux=1.0, seed=None, input_wcs=sca_wcs)

    return psf_image.flatten()


def gaussian(x, A, mu, sigma):
    """See name of function. :D"""
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def construct_images(exposures, ra, dec, size=7, subtract_background=True,
                     roman_path=None, truth="simple_model"):

    """Constructs the array of Roman images in the format required for the
    linear algebra operations.

    Inputs:
    exposures is a list of exposures from find_all_exposures
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

    SNLogger.debug(f"truth in construct images: {truth}")
    x_list = []
    y_list = []
    x_cutout_list = []
    y_cutout_list = []

    for indx, exp in enumerate(exposures):
        SNLogger.debug(f"Constructing image {indx} of {len(exposures)}")
        band = exp["filter"]
        pointing = exp["pointing"]
        sca = exp["sca"]

        # TODO : replace None with the right thing once Exposure is implemented

        imagepath = roman_path + (f"/RomanTDS/images/{truth}/{band}/{pointing}"
                                  f"/Roman_TDS_{truth}_{band}_{pointing}_"
                                  f"{sca}.fits.gz")
        image = OpenUniverse2024FITSImage(imagepath, None, sca)
        imagedata, errordata, flags = image.get_data(which="all", cache=True)

        image_cutout = image.get_ra_dec_cutout(ra, dec, size)

        sca_loc = image.get_wcs().world_to_pixel(ra, dec)
        cutout_loc = image_cutout.get_wcs().world_to_pixel(ra, dec)

        x_list.append(sca_loc[0])
        y_list.append(sca_loc[1])
        x_cutout_list.append(cutout_loc[0])
        y_cutout_list.append(cutout_loc[1])

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
                bg = image_cutout.get_fits_header()["SKY_MEAN"]
            elif truth == "truth":
                # ....or manually calculating it!
                bg = calculate_background_level(imagedata)

        bgflux.append(bg)

        image_cutout._data -= bg
        SNLogger.debug(f"Subtracted a background level of {bg}")

        image_list.append(image)
        cutout_image_list.append(image_cutout)

    exposures["x"] = x_list
    exposures["y"] = y_list
    exposures["x_cutout"] = x_cutout_list
    exposures["y_cutout"] = y_cutout_list

    SNLogger.debug("updated exposures with x, y, x_cutout, y_cutout:")
    SNLogger.debug(exposures)
    return cutout_image_list, image_list, exposures


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


def get_psf_image(self, stamp_size, x=None, y=None, x_center=None,
                  y_center=None, pupil_bin=8, sed=None, oversampling_factor=1,
                  include_photonOps=False, n_phot=1e6, pixel=False, flux=1):

    if pixel:
        point = galsim.Pixel(1)*sed
        SNLogger.debug("Building a Pixel shaped PSF source")
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
        SNLogger.debug(f"in get_psf_image: {self.bpass}, {x_center}, {y_center}")

        psf = galsim.Convolve(point, self.getPSF(x, y, pupil_bin))
        return psf.drawImage(self.bpass, image=stamp, wcs=wcs,
                             method="no_pixel",
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True)

    photon_ops = [self.getPSF(x, y, pupil_bin)] + self.photon_ops
    SNLogger.debug(f"Using {n_phot:e} photons in get_psf_image")
    result = point.drawImage(self.bpass, wcs=wcs, method="phot",
                             photon_ops=photon_ops, rng=self.rng,
                             n_photons=int(n_phot), maxN=int(n_phot),
                             poisson_flux=False,
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True, image=stamp)
    return result


def fetch_images(exposures, ra, dec, size, subtract_background, roman_path, object_type):
    """This function gets the list of exposures to be used for the analysis.

    Inputs:
    exposures: astropy.table.table.Table, the table of exposures to be used.
    num_total_images: total images used in analysis (detection + no detection)
    num_detect_images: number of images used in the analysis that contain a
                       detection.
    size: int, cutout will be of shape (size, size)
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

    num_predetection_images = len(exposures[~exposures["detected"]])
    num_total_images = len(exposures)
    if num_predetection_images == 0 and object_type == "SN":
        raise ValueError("No pre-detection images found in time range " +
                         "provided, skipping this object.")

    if num_total_images != np.inf and len(exposures) != num_total_images:
        raise ValueError(f"Not Enough Exposures. \
            Found {len(exposures)} out of {num_total_images} requested")

    cutout_image_list, image_list, exposures =\
        construct_images(exposures, ra, dec, size=size,
                         subtract_background=subtract_background,
                         roman_path=roman_path)

    return cutout_image_list, image_list, exposures


def get_object_info(ID, parq, band, snpath, roman_path, obj_type, collection='ou2024'):

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

    #### Implementing dia object
    obj = DiaObject.find_objects(collection=collection, id=ID)[0]
    return obj.ra, obj.dec, obj.mjd_start, obj.mjd_end, obj.mjd_peak


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
        SNLogger.debug(f"wgt before: {np.mean(wgt)}")
        inv_var = 1 / (error[i].flatten()) ** 2
        wgt *= inv_var

        SNLogger.debug(f"wgt after: {np.mean(wgt)}")
        wgt_matrix.append(wgt)
    return wgt_matrix


def make_grid(grid_type, images, ra, dec, percentiles=[0, 90, 95, 100],
              make_exact=False, single_ra=None, single_dec=None, cut_points_close_to_sn=False, spacing=0.75):
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
    ra, dec: floats, the RA and DEC of the supernova.
    percentiles: list of floats, the percentiles to use for the adaptive grid.
    make_exact: Currently not implemented, but will construct the grid in such
                a way on a simulated image that the recovered model is accurate
                to machine precision. TODO
    sim_galra, sim_galdec: floats, the RA and DEC of a single simulated galaxy, only
                used if grid_type is "single". This is used to place a single
                grid point at the location of the simulated galaxy, for sanity
                checking that the algorithm is drawing points where expected.

    Returns:
    ra_grid, dec_grid: numpy arrays of floats of the ra and dec locations for
                    model grid points.
    """
    size = images[0].image_shape[0]
    snappl_wcs = images[0].get_wcs()
    image_data = images[0].data

    SNLogger.debug(f"Grid type: {grid_type}")
    if grid_type not in ["regular", "adaptive", "contour", "single"]:
        raise ValueError("Grid type must be one of: regular, adaptive, "
                         "contour, single")
    if grid_type == "contour":
        ra_grid, dec_grid = make_contour_grid(image_data, snappl_wcs)

    elif grid_type == "adaptive":
        ra_grid, dec_grid = make_adaptive_grid(ra, dec, snappl_wcs,
                                               image=image_data,
                                               percentiles=percentiles)
    elif grid_type == "regular":
        ra_grid, dec_grid = make_regular_grid(ra, dec, snappl_wcs,
                                              size=size, spacing=spacing)

    if grid_type == "single":
        if single_ra is None or single_dec is None:
            raise ValueError("You did not simulate a galaxy, so you should not be using the single grid type.")
        ra_grid, dec_grid = [single_ra], [single_dec]

    if make_exact:
        if grid_type == "single":
            raise NotImplementedError
            # I need to figure out how to turn the single grid point test
        else:
            raise NotImplementedError
            # I need to figure out how to turn the single grid point test

    ra_grid = np.array(ra_grid)
    dec_grid = np.array(dec_grid)

    if cut_points_close_to_sn:
        min_distance = 0.5 * 0.11 * u.arcsec  # 0.11 arcsec is the pixel scale of Roman, so this is 1/2 a pixel
        SNLogger.debug(f"Cutting points closer than {min_distance} from SN")
        distances = angular_separation(ra*u.deg, dec*u.deg, ra_grid*u.deg, dec_grid*u.deg)
        SNLogger.debug(f"Old Grid size: {len(ra_grid)}")
        ra_grid = ra_grid[distances > min_distance]
        dec_grid = dec_grid[distances > min_distance]
        SNLogger.debug(f"New grid size: {len(ra_grid)}")

    return ra_grid, dec_grid


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


def get_SN_SED(SNID, date, sn_path, max_days_cutoff=10):
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
    SNLogger.debug(f"MJD values in SED: {np.array(mjd)}")
    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    closest_days_away = np.min(np.abs(np.array(mjd) - date))

    if np.abs(closest_days_away) > max_days_cutoff:
        SNLogger.warning(f"WARNING: No SED data within {max_days_cutoff} days of "
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
    SNLogger.debug(f"Grid type: contour, with percentiles: {percentiles} and subsize: {subsize}")


    if numlevels is not None:
        levels = list(np.linspace(np.min(image), np.max(image), numlevels))
    else:
        levels = list(np.percentile(image, percentiles))

    SNLogger.debug(f"Using levels: {levels} in make_contour_grid")

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

    xx, yy = x_totalgrid, y_totalgrid
    xx = np.array(xx)
    yy = np.array(yy)
    xx = xx.flatten()
    yy = yy.flatten()
    SNLogger.debug(f"Built a grid with {np.size(xx)} points")
    first_n = 5
    SNLogger.debug(f"First {first_n} grid points: {xx[:first_n]}, {yy[:first_n]}")

    # Astropy takes (y ,x) order:
    ra_grid, dec_grid = wcs.pixel_to_world(yy, xx)

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


def build_lightcurve(ID, exposures, confusion_metric, flux, sigma_flux, ra, dec):

    """This code builds a lightcurve datatable from the output of the SMP
       algorithm.

    Input:
    ID (int): supernova ID
    exposures (table): table of exposures used in the SMP algorithm
    confusion_metric (float): the confusion metric derived in the SMP algorithm
    flux (array): the output flux of the SMP algorithm. If no flux is received,
                    then no lightcurve is built.
    sigma_flux (array): the output flux error of the SMP algorithm
    ra, dec (float): the RA and DEC of the object.

    Returns:
    lc: a QTable containing the lightcurve data
    """
    flux = np.atleast_1d(flux)
    sigma_flux = np.atleast_1d(sigma_flux)
    band = exposures["filter"][0]
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)
    detections = exposures[np.where(exposures["detected"])]
    meta_dict = {"ID": ID, "obj_ra": ra, "obj_dec": dec}
    if confusion_metric is not None:
        meta_dict["confusion_metric"] = confusion_metric

    data_dict = {"mjd": detections["date"], "flux_fit": flux,
                 "flux_fit_err": sigma_flux, "mag_fit": mag,
                 "mag_fit_err": magerr,
                 "filter": np.full(np.size(mag), band),
                 "zpt": np.full(np.size(mag), zp),
                 "pointing": detections["pointing"],
                 "sca": detections["sca"],
                 "x": detections["x"],
                 "y": detections["y"],
                 "x_cutout": detections["x_cutout"],
                 "y_cutout": detections["y_cutout"]}

    units = {"mjd": u.d,  "flux_fit": "",
             "flux_fit_err": "", "mag_fit": u.mag,
             "mag_fit_err": u.mag, "filter": ""}

    return QTable(data=data_dict, meta=meta_dict, units=units)


def add_truth_to_lc(lc, exposures, sn_path, roman_path, object_type):

    detections = exposures[np.where(exposures["detected"])]
    band = exposures["filter"][0]
    ID = lc.meta["ID"]
    parq_file = find_parquet(ID, path=sn_path, obj_type=object_type)
    df = open_parquet(parq_file, path=sn_path, obj_type=object_type)

    sim_true_flux = []
    sim_realized_flux = []
    for pointing, sca in zip(detections["pointing"], detections["sca"]):
        catalogue_path = (
            roman_path + f"/RomanTDS/truth/{band}/{pointing}/" + f"Roman_TDS_index_{band}_{pointing}_{sca}.txt"
        )
        cat = pd.read_csv(
            catalogue_path,
            sep=r"\s+",
            skiprows=1,
            names=["object_id", "ra", "dec", "x", "y", "realized_flux", "flux", "mag", "obj_type"],
        )
        cat = cat[cat["object_id"] == ID]
        sim_true_flux.append(cat["flux"].values[0])
        sim_realized_flux.append(cat["realized_flux"].values[0])
    sim_true_flux = np.array(sim_true_flux)
    sim_realized_flux = np.array(sim_realized_flux)

    sim_sigma_flux = 0  # These are truth values!
    sim_realized_mag, _, _ = calc_mag_and_err(sim_realized_flux, sim_sigma_flux, band)
    sim_true_mag, _, _ = calc_mag_and_err(sim_true_flux, sim_sigma_flux, band)

    if object_type == "SN":
        df_object_row = df.loc[df.id == ID]
        meta_dict = {
            "host_sep": df_object_row["host_sn_sep"].values[0].item(),
            "host_mag_g": df_object_row["host_mag_g"].values[0].item(),
            "host_ra": df_object_row["host_ra"].values[0].item(),
            "host_dec": df_object_row["host_dec"].values[0].item(),
        }
    else:
        meta_dict = None

    data_dict = {
        "sim_realized_flux": sim_realized_flux,
        "sim_true_flux": sim_true_flux,
        "sim_realized_mag": sim_realized_mag,
        "sim_true_mag": sim_true_mag,
    }
    units = {
        "sim_realized_flux": "",
        "sim_realized_mag": u.mag,
        "sim_true_flux": "",
        "sim_true_mag": u.mag,
    }

    lc = hstack([lc, QTable(data=data_dict, meta=meta_dict, units=units)])

    return lc


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

    sim_mjd = np.arange(0, np.size(supernova), 1)
    data_dict = {"mjd": sim_mjd, "flux": flux,
                 "flux_error": sigma_flux, "sim_flux": supernova}
    meta_dict = {}
    units = {"mjd": u.d, "sim_flux": "",  "flux": "", "flux_error": ""}
    return QTable(data=data_dict, meta=meta_dict, units=units)


def save_lightcurve(lc, identifier, band, psftype, output_path=None, overwrite=True):
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
    SNLogger.info(f"Saving lightcurve to {lc_file}")
    lc.write(lc_file, format="ascii.ecsv", overwrite=overwrite)


def banner(text):
    length = len(text) + 8
    message = "\n" + "#" * length + "\n"+"#   " + text + "   # \n" + "#" \
              * length
    SNLogger.debug(message)


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
    exposures: the exposure table returned by fetch_images.
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
    SNLogger.debug("Prep data for fit")
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


def extract_sn_from_parquet_file_and_write_to_csv(parquet_file, sn_path, output_path, mag_limits=None):
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
    SNLogger.info(f"Found {np.size(SN_ID)} supernovae in the given range.")
    if np.size(SN_ID) == 0:
        raise ValueError("No supernovae found in the given range.")

    pd.DataFrame(SN_ID).to_csv(output_path, index=False, header=False)
    SNLogger.info(f"Saved to {output_path}")


def extract_id_using_ra_dec(sn_path, ra=None, dec=None, radius=None, object_type="SN"):
    """Convenience function for getting a list of SN RA and Dec that can be
    cone-searched for by passing a central coordinate and a radius. For now, this solely
    pulls objects from the OpenUniverse simulations.

    Parameters
    ----------
    sn_path: str, the path to the supernova data
    ra: float, the central RA of the region to search in
    dec: float, the central Dec of the region to search in
    radius: float, the radius over which cone search is performed. Can have
            any angular astropy.unit attached to it. If no unit is
            included, the function will produce a warning and then
            automatically assume you meant degrees.
    object_type: str, the type of object to search for. Can be "SN" or "star".
                  Defaults to "SN".

    Returns
    -------
    all_SN_ID: numpy array of int, the IDs of the objects found in the
               given range.
    all_dist: numpy array of float, the distances of the objects found in the
                given range, in arcseconds.
    """

    if not hasattr(radius, "unit") and radius is not None:
        SNLogger.warning("extract_id_using_ra_dec got a radius argument with no units. Assuming degrees.")
        radius *= u.deg

    file_prefix = {"SN": "snana", "star": "pointsource"}
    file_prefix = file_prefix[object_type]
    parquet_files = sorted(glob.glob(os.path.join(sn_path, f"{file_prefix}_*.parquet")))
    SN_ID_list = []
    dist_list = []
    SNLogger.debug(f"Found {len(parquet_files)} parquet files in {sn_path} with prefix {file_prefix}")
    for file in parquet_files:
        p = file.split(f"{file_prefix}_")[-1].split(".parquet")[0]
        df = open_parquet(p, sn_path, obj_type="SN")

        if radius is not None and (ra is not None and dec is not None):
            center_coord = SkyCoord(ra * u.deg, dec * u.deg)
            df_coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
            sep = center_coord.separation(df_coords)
            df = df[sep < radius]
            dist_list.extend(sep[sep < radius].to(u.arcsec).value)
        SN_ID = df.id.values
        SN_ID = SN_ID[np.log10(SN_ID) < 8]  # The 9 digit SN_ID SNe are weird for
        # some reason. They only seem to have 1 or 2 images ever. TODO
        SN_ID_list.extend(SN_ID)
    all_SN_ID = np.array(SN_ID_list, dtype=int)
    all_dist = np.array(dist_list, dtype=float)
    SNLogger.info(f"Found {np.size(all_SN_ID)} {object_type}s in the given range.")
    if np.size(all_SN_ID) == 0:
        raise ValueError(f"No {object_type}s found in the given range.")

    return all_SN_ID, all_dist


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
        SNLogger.warning("extract_star_from_parquet_file_and_write_to_csv " +
                         "got a radius argument with no units. Assuming degrees.")

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
    SNLogger.info(f"Found {np.size(star_ID)} stars in the given range.")
    if np.size(star_ID) == 0:
        raise ValueError("No stars found in the given range.")
    pd.DataFrame(star_ID).to_csv(output_path, index=False, header=False)
    SNLogger.info(f"Saved to {output_path}")


def run_one_object(ID=None, ra=None, dec=None, object_type=None, exposures=None,
                   roman_path=None, sn_path=None, size=None, band=None, fetch_SED=None, sedlist=None,
                   use_real_images=None, use_roman=None, subtract_background=None,
                   make_initial_guess=None, initial_flux_guess=None, weighting=None, method=None,
                   grid_type=None, pixel=None, source_phot_ops=None, do_xshift=None, bg_gal_flux=None, do_rotation=None,
                   airy=None, mismatch_seds=None, deltafcn_profile=None, noise=None, check_perfection=None,
                   avoid_non_linearity=None, spacing=None, percentiles=None, sim_galaxy_scale=1,
                   sim_galaxy_offset=None, base_pointing=662, base_sca=11,
                   draw_method_for_non_roman_psf="no_pixel"):
    psf_matrix = []
    sn_matrix = []

    # This is a catch for when I'm doing my own simulated WCSs
    util_ref = None

    percentiles = []
    roman_bandpasses = galsim.roman.getBandpasses()

    num_total_images = len(exposures)
    num_detect_images = len(exposures[exposures["detected"]])

    if use_real_images:
        # Using exposures Table, load those Pointing/SCAs as images.
        cutout_image_list, image_list, exposures = fetch_images(exposures, ra, dec, size, subtract_background,
                                                                roman_path, object_type)
        # We didn't simulate anything, so set these simulation only vars to none.
        sim_galra = None
        sim_galdec = None

    else:
        # Simulate the images of the SN and galaxy.
        banner("Simulating Images")
        sim_lc, util_ref, image_list, cutout_image_list, sim_galra, sim_galdec = \
            simulate_images(num_total_images, num_detect_images, ra, dec,
                            sim_galaxy_scale, sim_galaxy_offset,
                            do_xshift, do_rotation, noise=noise,
                            use_roman=use_roman, roman_path=roman_path,
                            size=size, band=band,
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
        ra_grid, dec_grid = make_grid(grid_type, cutout_image_list, ra, dec,
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

    #if use_real_images and object_type == "SN" and num_detect_images > 1:
    if False: # This is a temporary fix to not calculate the confusion metric.
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
        pointing, sca = exposures["pointing"][0], exposures["sca"][0]
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
    for i, (image, pointing, sca) in enumerate(zip(image_list, exposures["pointing"], exposures["sca"])):
        # Passing in None for the PSF means we use the Roman PSF.
        drawing_psf = None if use_roman else airy

        whole_sca_wcs = image.get_wcs()
        object_x, object_y = whole_sca_wcs.world_to_pixel(ra, dec)

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
        wgt_matrix = get_weights(cutout_image_list, ra, dec)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[1])

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
    sumimages = np.sum(pred, axis=1)

    # TODO: Move this to a separate function.
    # NOTE: This todo is being worked on in the simulations branch.
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


def load_SED_from_directory(sed_directory, wave_type="Angstrom", flux_type="fphotons"):
    """This function loads SEDs from a directory of SED files. The files must be in CSV format with
    two columns: "lambda" and "flux". The "lambda" column should contain the
    wavelengths in Angstroms, and the "flux" column should contain the fluxes in
    the appropriate units for the specified wave_type and flux_type.
    Inputs:
    sed_directory: str, the path to the directory containing the SED files.

    Returns:
    sed_list: list of galsim SED objects. (Temporary until we remove galsim)
    """
    SNLogger.debug(f"Loading SEDs from {sed_directory}")
    sed_list = []
    for file in pathlib.Path(sed_directory).glob("*.csv"):
        sed_table = pd.read_csv(file)

        flambda = sed_table["flux"]
        lam = sed_table["lambda"]
        # Assuming units are Angstroms how can I check this?
        sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                         wave_type=wave_type, flux_type=flux_type)
        sed_list.append(sed)
    return sed_list


def extract_object_from_healpix(healpix, nside, object_type="SN", source="OpenUniverse2024"):
    """This function takes in a healpix and nside and extracts all of the objects of the requested type in that
    healpix. Currently, the source the objects are extracted from is hardcoded to OpenUniverse2024 sims, but that will
    change in the future with real data.

    Parameters
    ----------
    healpix: int, the healpix number to extract objects from
    nside: int, the nside of the healpix to extract objects from
    object_type: str, the type of object to extract. Can be "SN" or "star". Defaults to "SN".
    source: str, the source of the table of objects to extract. Defaults to "OpenUniverse2024".

    Returns;
    -------
    id_array: numpy array of int, the IDs of the objects extracted from the healpix.
    """
    assert isinstance(healpix, int), "Healpix must be an integer."
    assert isinstance(nside, int), "Nside must be an integer."
    SNLogger.debug(f"Extracting {object_type} objects from healpix {healpix} with nside {nside} from {source}.")
    if source == "OpenUniverse2024":
        path = Config.get().value("photometry.campari.paths.sn_path")
        files = os.listdir(path)
        file_prefix = {"SN": "snana", "star": "pointsource"}
        files = [f for f in files if file_prefix[object_type] in f]
        files = [f for f in files if ".parquet" in f]
        files = [f for f in files if "flux" not in f]

        ra_array = np.array([])
        dec_array = np.array([])
        id_array = np.array([])

        for f in files:
            pqfile = int(f.split("_")[1].split(".")[0])
            df = open_parquet(pqfile, path, obj_type=object_type)

            ra_array = np.concatenate([ra_array, df["ra"].values])
            dec_array = np.concatenate([dec_array, df["dec"].values])
            id_array = np.concatenate([id_array, df["id"].values])

    else:
        # With real data, we will have to choose the first detection, as ra/dec might shift slightly.
        raise NotImplementedError(f"Source {source} not implemented yet.")

    healpix_array = hp.ang2pix(nside, ra_array, dec_array, lonlat=True)
    mask = healpix_array == healpix
    id_array = id_array[mask]

    return id_array.astype(int)


def read_healpix_file(healpix_file):
    """This function reads a healpix file and returns the healpix number and nside

    Parameters
    ----------
    healpix_file: str, the path to the healpix file

    Returns
    -------
    healpix: numpy array of int, the healpix numbers
    nside: int, the nside of the healpix
    """
    nside = None
    healpix_file = str(healpix_file)
    if healpix_file.endswith(".dat") or healpix_file.endswith(".yaml") or healpix_file.endswith(".yml"):
        with open(healpix_file, "r") as f:
            data = yaml.safe_load(f)
        nside = int(data["NSIDE"])
        healpix_list = data["HEALPIX"]
    else:
        healpix_list = pd.read_csv(healpix_file, header=None).values.flatten().tolist()

    return healpix_list, nside


def make_sim_param_grid(params):
    nd_grid = np.meshgrid(*params)
    flat_grid = np.array(nd_grid, dtype=float).reshape(len(params), -1)
    SNLogger.debug(f"Created a grid of simulation parameters with a total of {flat_grid.shape[1]} combinations.")
    return flat_grid

