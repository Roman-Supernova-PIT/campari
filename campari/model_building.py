# Standard Library
import warnings

# Common Library
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Astronomy Library
from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import galsim
from galsim import roman

# SN-PIT
from snappl.psf import PSF
from snappl.logger import SNLogger

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


def make_regular_grid(image_object, spacing=1.0, subsize=4):
    """Generates a regular grid around a (RA, Dec) center, choosing step size.

    Parameters
    ----------
    image_object: snappl.image.Image
        The image to build the grid upon.
    spacing: int
        Spacing of grid points in pixels.
    subsize: int
        Width of the grid in pixels.
        Specify the width of the grid, which can be smaller than the
        image. For instance I could have an image that is 11x11 but a grid that is only 9x9.
        This is useful and different from making a smaller image because
        when the image rotates, model points near the corners of the image
        may be rotated out. By taking a smaller grid, we can avoid this.

    Returns
    ----------
    ra_grid, dec_grid: 1D numpy arrays of floats
        The RA and DEC of the grid points.
    """
    SNLogger.debug(f"Making regular grid with spacing {spacing} and subsize {subsize}")
    wcs = image_object.get_wcs()
    size = image_object.image_shape[0]
    if wcs.to_fits_header()["CRPIX1"] == 2044 and wcs.to_fits_header()["CRPIX2"] == 2044:
        SNLogger.warning(
            "This WCS is centered exactly on the center of the image, make_regular_grid is expecting a"
            "cutout WCS, this is likely not a cutout WCS."
        )
    if subsize > size:
        SNLogger.warning(
            "subsize is larger than the image size. "
            + f"{size} > {subsize}. This would cause model points to"
            + " be placed outside the image. Reducing subsize to"
            + " match the image size."
        )
        subsize = size

    SNLogger.debug("Grid type: regularly spaced")
    difference = int((size - subsize) / 2)

    x = difference + np.arange(0, subsize, spacing)
    y = difference + np.arange(0, subsize, spacing)
    SNLogger.debug(f"Grid spacing: {spacing}")
    SNLogger.debug(f"Grid x values: {x}")
    old_difference = int((size - 4) / 2)
    SNLogger.debug(f"Old woudld have been {old_difference + np.arange(0, 4, 0.75)} ")

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    SNLogger.debug(f"Built a grid with {np.size(xx)} points")
    import pdb; pdb.set_trace()

    # Astropy takes (y, x) order:
    ra_grid, dec_grid = wcs.pixel_to_world(yy, xx)

    return ra_grid, dec_grid


def make_adaptive_grid(image_object, percentiles=[45, 90], subsize=9, subpixel_grid_width=1.2):
    """Construct an "adaptive grid" which allocates model grid points to model
    the background galaxy according to the brightness of the image.

    Parameters
    ----------
    image_object: snappl.image.Image
        The image to build the grid upon.
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
    size = image_object.image_shape[0]
    image = image_object.data
    wcs = image_object.get_wcs()
    if subsize > size:
        SNLogger.warning(
            "subsize is larger than the image size "
            + f"{size} > {subsize}. This would cause model points to"
            + " be placed outside the image. Reducing subsize to"
            + " match the image size."
        )
        subsize = size

    SNLogger.debug("image shape: {}".format(np.shape(image)))
    SNLogger.debug("Grid type: adaptive")
    # Bin the image in logspace and allocate grid points based on the
    # brightness.

    difference = int((size - subsize) / 2)

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
                xx = np.linspace(x - subpixel_grid_width / 2, x + subpixel_grid_width / 2, num + 2)[1:-1]
                yy = np.linspace(y - subpixel_grid_width / 2, y + subpixel_grid_width / 2, num + 2)[1:-1]
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
    all_vals = np.atleast_1d(np.zeros_like(ra_grid))

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


def construct_static_scene(ra=None, dec=None, sca_wcs=None, x_loc=None, y_loc=None, stampsize=None,
                           pixel=False, band=None, image=None, psfclass="ou24PSF"):
    """Constructs the background model around a certain image (x,y) location
    and a given array of RA and DECs.

    Parameters
    ----------
    ra, dec: arrays of floats, RA and DEC values for the grid
    sca_wcs: the wcs of the entire image, i.e. the entire SCA. A snappl.wcs.BaseWCS object.
    x_loc, y_loc: floats,the pixel location of the object in the FULL image,
        i.e. x y location in the SCA.
    stampsize: int, the size of the stamp being used
    band: str, the bandpass being used
    psf: Here you can provide a PSF to use, if you don't provide one,this function will calculate the Roman PSF
        instead.
    pixel: bool, If True, use a pixel tophat function to convolve the PSF with,
        otherwise use a delta function. Does not seem to hugely affect results.

    Returns:
    A numpy array of the PSFs at each grid point, with the shape
    (stampsize*stampsize, npoints)
    """
    # I call this x_sca to highlight that it's the location in the SCA, not the cutout.
    x_sca, y_sca = sca_wcs.world_to_pixel(ra, dec)
    # For testing purposes, sometimes the grid is exactly one point, so we force it to be 1d.
    x_sca = np.atleast_1d(x_sca)
    y_sca = np.atleast_1d(y_sca)
    bpass = roman.getBandpasses()[band]

    num_grid_points = np.size(x_sca)

    psfs = np.zeros((stampsize * stampsize, num_grid_points))

    sed = galsim.SED(
        galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"), wave_type="nm", flux_type="fphotons"
    )

    if pixel:
        point = galsim.Pixel(0.1) * sed
    else:
        point = galsim.DeltaFunction()
        point *= sed

    point = point.withFlux(1, bpass)

    pointing = image.pointing if image is not None else None
    sca = image.sca if image is not None else None

    psf_object = PSF.get_psf_object(psfclass, pointing=pointing, sca=sca, size=stampsize, stamp_size=stampsize,
                                    include_photonOps=False, seed=None, image=image)
    # See run_one_object documentation to explain this pixel coordinate conversion.
    x_loc = int(np.floor(x_loc + 0.5))
    y_loc = int(np.floor(y_loc + 0.5))

    # Loop over the grid points, draw a PSF at each one, and append to a list.
    for a, (x, y) in enumerate(zip(x_sca.flatten(), y_sca.flatten())):
        psfs[:, a] = psf_object.get_stamp(
            x0=x_loc, y0=y_loc, x=x, y=y, flux=1.0
        ).flatten()

    return psfs


def construct_transient_scene(
    x0=None, y0=None, pointing=None, sca=None, stampsize=25, x=None,
    y=None, sed=None, flux=1, photOps=True, image=None, psfclass="ou24PSF_slow"
):
    """Constructs the PSF around the point source (x,y) location, allowing for
        some offset from the center.
    Parameters:
    -----------
    x0, y0: int, default None
        The pixel position on the image corresponding to the center
        pixel of the returned PSF.  If either is None, they default
        to x0=floor(x+0.5) and y0=floor(y+0.5).
    x, y: floats
            Position on the image of the center of the psf.

    For more on the above two parameters, see snappl.psf.PSF.get_stamp documentation.

    pointing, sca: ints
        The pointing and SCA of the image
    stampsize: int
        Size of cutout image used.
    sed: galsim.sed.SED object
        The SED of the source TODO: this needs to be implemented.
    flux: float
        If you are using this function to build a model grid point,
        this should be 1. If you are using this function to build a model of
        a source, this should be the flux of the source.

    Returns:
    -----------
    psf_image: numpy array of floats of size stampsize**2
        The image of the PSF at the (x,y) location.
    """

    SNLogger.debug(
        f"ARGS IN PSF SOURCE: \n x, y: {x, y} \n"
        + f" pointing, sca: {pointing, sca} \n"
        + f" stamp size: {stampsize} \n"
        + f" x0, y0: {x0, y0} \n"
        + f" sed: {sed} \n"
        + f" flux: {flux}"
    )

    if not photOps:
        # While I want to do this sometimes, it is very rare that you actually
        # want to do this. Thus if it was accidentally on while doing a normal
        # run, I'd want to know.
        SNLogger.warning("NOT USING PHOTON OPS IN PSF SOURCE")

    # We want to use the slower PSF class for supernovae
    snpsfclass = "ou24PSF_slow" if psfclass == "ou24PSF" else psfclass

    SNLogger.debug(f"Using psf class {snpsfclass}")
    psf_object = PSF.get_psf_object(
        snpsfclass, pointing=pointing, sca=sca, size=stampsize, include_photonOps=photOps,
        image=image, stamp_size=stampsize
    )
    psf_image = psf_object.get_stamp(x0=x0, y0=y0, x=x, y=y, flux=1.0)

    return psf_image.flatten()


def make_grid(
    grid_type=None,
    images=None,
    ra=None,
    dec=None,
    percentiles=[0, 90, 95, 100],
    make_exact=False,
    single_ra=None,
    single_dec=None,
    cut_points_close_to_sn=False,
    spacing=0.75,
    subsize=9,
):
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
    sim_galra, sim_galdec: floats, the RA and DEC of a single simulated galaxy, only
                used if grid_type is "single". This is used to place a single
                grid point at the location of the simulated galaxy, for sanity
                checking that the algorithm is drawing points where expected.

    Returns:
    ra_grid, dec_grid: numpy arrays of floats of the ra and dec locations for
                    model grid points.
    """

    SNLogger.debug(f"Grid type: {grid_type}")
    if grid_type not in ["regular", "adaptive", "contour", "single"]:
        raise ValueError("Grid type must be one of: regular, adaptive, contour, single")
    if grid_type == "contour":
        ra_grid, dec_grid = make_contour_grid(images[0], subsize=subsize)

    elif grid_type == "adaptive":
        ra_grid, dec_grid = make_adaptive_grid(images[0], percentiles=percentiles, subsize=subsize)
    elif grid_type == "regular":
        ra_grid, dec_grid = make_regular_grid(images[0], spacing=spacing, subsize=subsize)

    if grid_type == "single":
        if single_ra is None or single_dec is None:
            raise ValueError("You did not simulate a galaxy, so you should not be using the single grid type.")
        ra_grid, dec_grid = [single_ra], [single_dec]

    ra_grid = np.array(ra_grid)
    dec_grid = np.array(dec_grid)

    if cut_points_close_to_sn:
        min_distance = 0.5 * 0.11 * u.arcsec  # 0.11 arcsec is the pixel scale of Roman, so this is 1/2 a pixel
        SNLogger.debug(f"Cutting points closer than {min_distance} from SN")
        distances = angular_separation(ra * u.deg, dec * u.deg, ra_grid * u.deg, dec_grid * u.deg)
        SNLogger.debug(f"Old Grid size: {len(ra_grid)}")
        ra_grid = ra_grid[distances > min_distance]
        dec_grid = dec_grid[distances > min_distance]
        SNLogger.debug(f"New grid size: {len(ra_grid)}")

    return ra_grid, dec_grid


def make_contour_grid(img_obj, numlevels=None, percentiles=[0, 90, 98, 100], subsize=4):
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
    wcs = img_obj.get_wcs()
    size = img_obj.image_shape[0]
    image = img_obj.data
    size = image.shape[0]
    x = np.arange(0, size, 1.0)
    y = np.arange(0, size, 1.0)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    xg = xg.ravel()
    yg = yg.ravel()
    SNLogger.debug(f"Grid type: contour, with percentiles: {percentiles} and subsize: {subsize}")

    if numlevels is not None:
        levels = list(np.linspace(np.nanmin(image), np.nanmax(image), numlevels))
    else:
        levels = list(np.nanpercentile(image, percentiles))

    SNLogger.debug(f"Using levels: {levels} in make_contour_grid")

    interp = RegularGridInterpolator((x, y), image, method="linear", bounds_error=False, fill_value=None)

    aa = interp((xg, yg))

    x_totalgrid = []
    y_totalgrid = []

    for i in range(len(levels) - 1):
        zmin = levels[i]
        zmax = levels[i + 1]
        # Generate a grid that gets finer each iteration of the loop. For
        # instance, in brightness bin 1, 1 point per pixel, in brightness bin
        # 2, 4 points per pixel (2 in each direction), etc.
        x = np.arange(0, size, 1 / (i + 1))
        y = np.arange(0, size, 1 / (i + 1))
        if i == 0:
            x = x[np.where(np.abs(x - size / 2) < subsize)]
            y = y[np.where(np.abs(y - size / 2) < subsize)]
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


def build_model_for_one_image(image=None, ra=None, dec=None, use_real_images=None, grid_type=None, ra_grid=None,
                              dec_grid=None, size=None, pixel=False, psfclass=None, band=None, sedlist=None,
                              source_phot_ops=None, i=None, num_total_images=None, num_detect_images=None,
                              prebuilt_psf_matrix=None, prebuilt_sn_matrix=None, subtract_background=None,
                              base_pointing=None, base_sca=None):
    # Passing in None for the PSF means we use the Roman PSF.
    pointing, sca = image.pointing, image.sca
    SNLogger.debug(f"Building model for image {i} with pointing {pointing} and sca {sca}")

    whole_sca_wcs = image.get_wcs()
    object_x, object_y = whole_sca_wcs.world_to_pixel(ra, dec)

    # Build the model for the background using the correct psf and the
    # grid we made in the previous section.
    # If no grid, we still need something that can be concatenated in the
    # linear algebra steps, so we initialize an empty array by default.
    background_model_array = np.empty((size**2, 0))
    SNLogger.debug("Constructing background model array for image " + str(i) + " ---------------")
    if grid_type != "none" and prebuilt_psf_matrix is None:
        background_model_array = construct_static_scene(
            ra_grid,
            dec_grid,
            whole_sca_wcs,
            object_x,
            object_y,
            size,
            pixel=pixel,
            image=image,
            psfclass=psfclass,
            band=band,
        )
    elif grid_type != "none" and prebuilt_psf_matrix is not None:
        SNLogger.debug("Using prebuilt PSF matrix for background model")
        background_model_array = None

    if not subtract_background and prebuilt_psf_matrix is None:
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
            background_model_array = np.concatenate([background_model_array, bg], axis=1)

    # Add the array of the model points and the background (if using)
    # to the matrix of all components of the model.
    # if prebuilt_psf_matrix is None:
    #     psf_matrix.append(background_model_array)

    # The arrays below are the length of the number of images that contain the object
    # Therefore, when we iterate onto the
    # first object image, we want to be on the first element
    # of sedlist. Therefore, we subtract by the number of
    # predetection images: num_total_images - num_detect_images.
    # I.e., sn_index is the 0 on the first image with an object, 1 on the second, etc.
    sn_index = i - (num_total_images - num_detect_images)

    if sn_index >= 0 and prebuilt_sn_matrix is None:
        SNLogger.debug("Constructing transient model array for image " + str(i) + " ---------------")
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
        psf_source_array = construct_transient_scene(
            x0=x,
            y0=y,
            pointing=pointing,
            sca=sca,
            stampsize=size,
            x=object_x,
            y=object_y,
            sed=sed,
            psfclass=psfclass,
            photOps=source_phot_ops,
            image=image,
        )


#        sn_matrix.append(psf_source_array)

    elif sn_index >= 0 and prebuilt_sn_matrix is not None:
        SNLogger.debug("Using prebuilt SN matrix for transient model")
        psf_source_array = None
    else:
        psf_source_array = None

    return background_model_array, psf_source_array
