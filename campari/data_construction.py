# Standard Library
import warnings

# Common Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import numpy as np

# SN-PIT
from snappl.imagecollection import ImageCollection
from snpit_utils.logger import SNLogger

# Campari
from campari.utils import calculate_background_level

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

# Global variables
huge_value = 1e32


def construct_images(image_list, diaobj, size, subtract_background=True):
    """Constructs the array of Roman images in the format required for the
    linear algebra operations.

    Inputs:
    image_list: list of snappl.image.Image objects, the images to be used.
    ra,dec: the RA and DEC of the SN
    subtract_background: If False, the background level is fit as a free
        parameter in the forward modelling. Otherwise, we subtract it here.

    Returns:
    cutout_image_list: list of snappl.image.Image objects, cutouts on the
                       object location.
    image_list: list of snappl.image.Image objects of the entire SCA.

    """
    ra = diaobj.ra
    dec = diaobj.dec
    truth = "simple_model"

    bgflux = []
    cutout_image_list = []

    x_list = []
    y_list = []
    x_cutout_list = []
    y_cutout_list = []

    for indx, image in enumerate(image_list):

        imagedata, errordata, flags = image.get_data(which="all", cache=True)

        image_cutout = image.get_ra_dec_cutout(ra, dec, size, mode="partial", fill_value=np.nan)
        num_nans = np.isnan(image_cutout.data).sum()
        if num_nans > 0:
            SNLogger.warning(
                f"Cutout contains {num_nans} NaN values, likely because the cutout is near the edge of the"
                " image. These will be given a weight of zero."
            )
            SNLogger.warning(f"Fraction of NaNs in cutout: {num_nans / size**2:.2%}")

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
                # Clean this up before pushing TODO!
                try:
                    bg = image_cutout.get_fits_header()["SKY_MEAN"]
                except KeyError:
                    SNLogger.warning("Using an override of 0")
                    bg = 0

            elif truth == "truth":
                # ....or manually calculating it!
                bg = calculate_background_level(imagedata)

        bgflux.append(bg)

        image_cutout._data -= bg

        cutout_image_list.append(image_cutout)

    return cutout_image_list, image_list


def prep_data_for_fit(images, sn_matrix, wgt_matrix, diaobj):
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
    size_sq = images[0].image_shape[0] ** 2
    tot_num = len(images)
    mjd = np.array([im.mjd for im in images])

    num_pre_transient_images = np.sum(mjd < diaobj.mjd_start)
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

    SNLogger.debug("sn_matrix shape before: " + str(np.array(sn_matrix).shape))
    psf_zeros = np.zeros((np.size(image_data), tot_num))
    for i in range(det_num):
        #SNLogger.debug(f"Filling in SN model for detection image {i}")
        #SNLogger.debug(f"totnum, pre_trans: {tot_num, num_pre_transient_images}")
        #sn_index = num_pre_transient_images + i  # We only want to edit SN columns.
        sn_index = tot_num - det_num + i  # We only want to edit SN columns.
        #SNLogger.debug(f"Which corresponds to image {sn_index} in the full set of images.")
        psf_zeros[
            (sn_index) * size_sq :   # Fill in rows s^2 * image number...
            (sn_index + 1) * size_sq,  # ... to s^2 * (image number + 1) ...
            sn_index,
        ] = sn_matrix[i]  # ...in the correct column.
    sn_matrix = np.vstack(psf_zeros)
    wgt_matrix = np.array(wgt_matrix)
    wgt_matrix = np.hstack(wgt_matrix)

    # Now handle masked pixels:
    wgt_matrix[np.isnan(image_data)] = 0
    image_data[np.isnan(image_data)] = 0
    err[np.isnan(err)] = huge_value  # Give a huge error to masked pixels.

    return image_data, err, sn_matrix, wgt_matrix


def find_all_exposures(
    diaobj=None,
    band=None,
    maxbg=None,
    maxdet=None,
    sca_list=None,
    truth="simple_model",
    image_selection_start=None,
    image_selection_end=None,
    image_source="ou2024",
    image_path=None
):
    """This function finds all the exposures that contain a given supernova,
    and returns a list of them.

    Inputs:
    ra, dec: the RA and DEC of the supernova
    peak: the peak of the supernova
    transient_start, transient_end: floats, the first and last MJD of a detection of the transient,
        defines what which images contain transient light (and therefore recieve a single model point
        at the location of the transient) and which do not.
    maxbg: the maximum number of background images to consider
    maxdet: the maximum number of detected images to consider
    pointing_list: If this is passed in, only consider these pointings
    sca_list: If this is passed in, only consider these SCAs
    truth: If "truth" use truth images, if "simple_model" use simple model
            images.
    band: the band to consider
    image_selection_start, image_selection_end: floats, the first and last MJD of images to be used in the algorithm.
    image_source: str, the source of the images to be used. If "ou2024", use the Open Universe 2024 images.
    image_path: str, the path to the images to be used. If given, will use these images
                     for image sources that require a base_path.
    """
    SNLogger.debug(f"Finding all exposures for diaobj {diaobj.mjd_start, diaobj.mjd_end, diaobj.ra, diaobj.dec}")
    transient_start = diaobj.mjd_start
    transient_end = diaobj.mjd_end
    ra = diaobj.ra
    dec = diaobj.dec

    img_collection = ImageCollection()
    # De-harcode this
    img_collection = img_collection.get_collection(image_source, subset="threefile", base_path=image_path)

    if (image_selection_start is None or transient_start > image_selection_start) and transient_start is not None:

        pre_transient_images = img_collection.find_images(
            mjd_min=image_selection_start, mjd_max=transient_start, ra=ra, dec=dec, filter=band
        )
    else:
        pre_transient_images = []

    if (image_selection_end is None or transient_end < image_selection_end) and transient_end is not None:
        post_transient_images = img_collection.find_images(
            mjd_min=transient_end, mjd_max=image_selection_end, ra=ra, dec=dec, filter=band
        )
    else:
        post_transient_images = []

    no_transient_images = pre_transient_images + post_transient_images

    transient_images = img_collection.find_images(
        mjd_min=transient_start, mjd_max=transient_end, ra=ra, dec=dec, filter=band
    )

    no_transient_images = np.array(no_transient_images)
    transient_images = np.array(transient_images)
    if maxbg is not None:
        no_transient_images = no_transient_images[:maxbg]
    if maxdet is not None:
        transient_images = transient_images[:maxdet]
    all_images = np.hstack((transient_images, no_transient_images))

    argsort = np.argsort([img.pointing for img in all_images])
    all_images = all_images[argsort]
    return all_images
