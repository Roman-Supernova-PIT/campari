# Standard Library
import warnings

# Common Library
import astropy.table as tb
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from snappl.image import OpenUniverse2024FITSImage
from snpit_utils.logger import SNLogger

# Campari
from campari.utils import calculate_background_level

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


def construct_images(exposures, ra, dec, size=7, subtract_background=True, roman_path=None, truth="simple_model"):
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

        imagepath = roman_path + (
            f"/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{sca}.fits.gz"
        )
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

    return image_data, err, sn_matrix, wgt_matrix


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
        raise ValueError("No pre-detection images found in time range " + "provided, skipping this object.")

    if num_total_images != np.inf and len(exposures) != num_total_images:
        raise ValueError(
            f"Not Enough Exposures. \
            Found {len(exposures)} out of {num_total_images} requested"
        )

    cutout_image_list, image_list, exposures = construct_images(
        exposures, ra, dec, size=size, subtract_background=subtract_background, roman_path=roman_path
    )

    return cutout_image_list, image_list, exposures


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
