# Standard Library
import warnings

# Common Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from multiprocessing import Pool
import numpy as np


# SN-PIT
from snappl.imagecollection import ImageCollection
from snappl.logger import SNLogger

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


def construct_images(image_list, diaobj, size, subtract_background_method=True, nprocs=1):
    """Constructs the array of Roman images in the format required for the
    linear algebra operations.

    Inputs:
    image_list: list of snappl.image.Image objects, the images to be used.
    diaobj: snappl.diaobj.DiaObj object, the Difference Imaging Object to find images for.
    size: int, the size of the cutout to be made (size x size)
    subtract_background_method: str, the method used to calculate the background to be removed. 
        - If "fit", the background level is fit as a free parameter in the forward modelling.
        - If "calc" or "calculate", the background level is calculated using photutils.
        - If it is any other string, it is assumed that this is a column name in the fits header that the background 
        level is stored in.
    nprocs: int, the number of processors to use for parallel processing.

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

    results = []

    SNLogger.debug(f"subtract_background_method: {subtract_background_method}")

    if nprocs > 1:
        with Pool(nprocs) as pool:
            for indx, image in enumerate(image_list):
                SNLogger.debug(f"Constructing cutout for image {indx+1} of {image}")
                results.append(pool.apply_async(construct_one_image, kwds={"indx": indx, "image": image,
                                                                           "ra": ra, "dec": dec, "size": size,
                                                                           "truth": truth,
                                                                           "subtract_background_method": subtract_background_method}))

            pool.close()
            pool.join()
    else:
        for indx, image in enumerate(image_list):
            SNLogger.debug(f"Constructing cutout for image {indx+1} of {image}")
            results.append(construct_one_image(indx=indx, image=image,
                                               ra=ra, dec=dec, size=size, truth=truth,
                                               subtract_background_method=subtract_background_method))

    for r in results:
        if nprocs > 1:
            res = r.get()
        else:
            res = r
        cutout_image_list.append(res[0])
        bgflux.append(res[1])

    return cutout_image_list, image_list, bgflux


def construct_one_image(indx=None, image=None, ra=None, dec=None, size=None, truth=None, subtract_background_method=None):
    """Constructs a single Roman image in the format required for the
    linear algebra operations. This is the function that is called in parallel
    by campari.data_construction.construct_images

    Inputs:
    image: snappl.image.Image object, the image to be used.
    indx: int, index of the image in the list.
    ra/dec: float, the RA and DEC of the SN
    size: int, the size of the cutout to be made (size x size)
    subtract_background_method: str, the method used to calculate the background to be removed. 
        - If "fit", the background level is fit as a free parameter in the forward modelling.
        - If "calculate", the background level is calculated using photutils.
        - If it is any other string, it is assumed that this is a keyword name in the FITS header that the background 
        level is stored in. If the keyword is not found, an error is raised.
    truth: str, either "truth" or "simple_model", whether to use truth images
        or OU2024 simple model images.
    

    Returns:
    cutout_image: snappl.image.Image object, cutout on the object location.
    image: snappl.image.Image object of the entire SCA.

    """
    imagedata, errordata, flags = image.get_data(which="all", cache=True)

    image_cutout = image.get_ra_dec_cutout(ra, dec, size, mode="partial", fill_value=np.nan)
    num_nans = np.isnan(image_cutout.data).sum()
    if num_nans > 0:
        SNLogger.warning(
            f"Cutout contains {num_nans} NaN values, likely because the cutout is near the edge of the"
            " image. These will be given a weight of zero."
        )
        SNLogger.warning(f"Fraction of NaNs in cutout: {num_nans / size**2:.2%}")

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
    if subtract_background_method == "calculate":
        bg = calculate_background_level(imagedata)
    elif subtract_background_method == "fit":
        bg = 0
    else:
        SNLogger.debug(f"Trying to get background from header: {subtract_background_method}")
        bg = image_cutout.get_fits_header()[subtract_background_method] if subtract_background_method in image_cutout.get_fits_header() else None

        if bg is None:
            raise ValueError(f"Could not find background level in header with keyword "
                             f"'{subtract_background_method}' for image {indx}.")

    image_cutout._data -= bg
    return image_cutout, bg


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
        sn_index = tot_num - det_num + i  # We only want to edit SN columns.
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
    truth="simple_model",
    image_selection_start=None,
    image_selection_end=None,
    image_collection="snpitdb",
    image_collection_subset=None,
    image_collection_basepath=None,
    dbclient=None,
    provenance_tag=None,
    process=None
):
    """This function finds all the exposures that contain a given supernova,
    and returns a list of them.

    Inputs:
    diaobj: snappl.diaobj.DiaObj object, the Difference Imaging Object to find images for.
    band: the band to consider
    maxbg: the maximum number of background images to consider
    maxdet: the maximum number of detected images to consider
    truth: If "truth" use truth images, if "simple_model" use simple model
            images. For Open Universe 2024 simulations only.
    image_selection_start, image_selection_end: floats, the first and last MJD of images to be used in the algorithm.
    image_collection: str, the source of the images to be used. If "ou2024", use the Open Universe 2024 images.
    image_collection_subset: str, subset argument provided to the image collection object to use for lookup.
    image_collection_basepath: str, the path to the images to be used. If given, will use these images
                     for image sources that require a base_path when using the ImageCollection object.
    dbclient: snappl.dbclient.DBClient object, the database client to use to query for images.
    provenance_tag: str, the provenance tag to use to find images.
    process: str, the process name to use to find images.

    """
    SNLogger.debug(f"Finding all exposures for diaobj {diaobj.mjd_start, diaobj.mjd_end, diaobj.ra, diaobj.dec}")
    SNLogger.debug(f"Using image collection: {image_collection}")
    SNLogger.debug(f"image_selection_start: {image_selection_start}")
    SNLogger.debug(f"image_selection_end: {image_selection_end}")
    transient_start = diaobj.mjd_start
    if transient_start is None and diaobj.mjd_discovery is not None:
        # From Sidecar, the start date is called mjd_discovery
        transient_start = diaobj.mjd_discovery
    transient_end = diaobj.mjd_end
    ra = diaobj.ra
    dec = diaobj.dec

    # Database can't handle Nones
    temp_image_selection_start = 0 if image_selection_start is None else image_selection_start
    temp_image_selection_end = 1e30 if image_selection_end is None else image_selection_end
    temp_transient_start = 0 if transient_start is None else transient_start
    temp_transient_end = 1e30 if transient_end is None else transient_end

    SNLogger.debug(f"Using image collection: {image_collection}")
    SNLogger.debug(f"Using provenance tag: {provenance_tag}")
    SNLogger.debug(f"Using process: {process}")
    SNLogger.debug(f"db_client: {dbclient}")

    # Dehardcode the 3 file thing
    img_collection = ImageCollection().get_collection(collection=image_collection, provenance_tag=provenance_tag,
                                                      process=process, dbclient=dbclient,
                                                      subset=image_collection_subset,
                                                      base_path=image_collection_basepath)

    img_collection_prov = getattr(img_collection, "provenance", None)
    if (image_selection_start is None or transient_start > image_selection_start) and transient_start is not None:
        SNLogger.debug(f"Looking for Pre Transient images between {temp_image_selection_start}"
                       f" and {temp_transient_start}")
        pre_transient_images = img_collection.find_images(
            mjd_min=temp_image_selection_start, mjd_max=temp_transient_start, ra=ra, dec=dec, band=band,

        )
        SNLogger.debug(f"Found {len(pre_transient_images)}")
    else:
        pre_transient_images = []

    if (image_selection_end is None or transient_end < image_selection_end) and transient_end is not None:
        SNLogger.debug(f"Looking for Post Transient images between {temp_transient_end} and {temp_image_selection_end}")

        post_transient_images = img_collection.find_images(
            mjd_min=temp_transient_end, mjd_max=temp_image_selection_end, ra=ra, dec=dec, band=band,
        )
        SNLogger.debug(f"Found {len(post_transient_images)}")
    else:
        post_transient_images = []

    no_transient_images = pre_transient_images + post_transient_images
    SNLogger.debug(f"Looking for Transient images between {temp_transient_start} and {temp_transient_end}")
    transient_images = img_collection.find_images(
        mjd_min=temp_transient_start, mjd_max=temp_transient_end, ra=ra, dec=dec, band=band,
    )
    SNLogger.debug(f"Found {len(transient_images)} Transient images")

    no_transient_images = np.array(no_transient_images)
    transient_images = np.array(transient_images)
    if maxbg is not None:
        no_transient_images = no_transient_images[:maxbg]
    if maxdet is not None:
        transient_images = transient_images[:maxdet]
    all_images = np.hstack((transient_images, no_transient_images))
    SNLogger.debug(f"Found {len(all_images)} total images")

    argsort = np.argsort([img.pointing for img in all_images])
    all_images = all_images[argsort]
    return all_images, img_collection_prov
