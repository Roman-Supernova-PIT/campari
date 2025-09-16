# Standard Library
import glob
import os
import pathlib
import warnings

# Common Library
import numpy as np
import pandas as pd
import yaml

# Astronomy Library
from astropy.coordinates import SkyCoord
from astropy.table import QTable, hstack
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
import healpy as hp

# SN-PIT
from snappl.imagecollection import ImageCollection
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger

# Campari
from campari.utils import calc_mag_and_err

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


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


def build_lightcurve(diaobj, lc_model):
    """This code builds a lightcurve datatable from the output of the SMP algorithm.

    Input:
    Parameters
    ----------
    diaobj: snappl.diaobject.DiaObject
        The DiaObject representing the transient.
    lc_model: campari.campari_lightcurve_model
        The lightcurve model output from the SMP algorithm.

    Returns:
        lc: a QTable containing the lightcurve data
    """
    flux = np.atleast_1d(lc_model.flux)
    sigma_flux = np.atleast_1d(lc_model.sigma_flux)
    confusion_metric = lc_model.confusion_metric
    image_list = lc_model.image_list
    cutout_image_list = lc_model.cutout_image_list
    band = image_list[0].band
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)
    meta_dict = {"ID": diaobj.id, "obj_ra": diaobj.ra, "obj_dec": diaobj.dec}
    if confusion_metric is not None:
        meta_dict["confusion_metric"] = confusion_metric

    data_dict = {
        "mjd": [],
        "flux_fit": flux,
        "flux_fit_err": sigma_flux,
        "mag_fit": mag,
        "mag_fit_err": magerr,
        "filter": [],
        "zpt": np.full(np.size(mag), zp),
        "pointing": [],
        "sca": [],
        "x": [],
        "y": [],
        "x_cutout": [],
        "y_cutout": [],
    }

    for i, img in enumerate(image_list):
        if img.mjd > diaobj.mjd_start and img.mjd < diaobj.mjd_end:
            data_dict["mjd"].append(img.mjd)
            data_dict["filter"].append(img.band)
            data_dict["pointing"].append(img.pointing)
            data_dict["sca"].append(img.sca)
            x, y = img.get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x"].append(x)
            data_dict["y"].append(y)
            x_cutout, y_cutout = cutout_image_list[i].get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x_cutout"].append(x_cutout)
            data_dict["y_cutout"].append(y_cutout)

    units = {"mjd": u.d, "flux_fit": "", "flux_fit_err": "", "mag_fit": u.mag, "mag_fit_err": u.mag, "filter": ""}
    SNLogger.debug(f"data dict in build_lightcurve: {data_dict}")

    return QTable(data=data_dict, meta=meta_dict, units=units)


def add_truth_to_lc(lc, lc_model, diaobj, sn_path, roman_path, object_type):
    """This code adds the truth flux and magnitude to a lightcurve datatable."""

    ID = lc.meta["ID"]
    parq_file = find_parquet(ID, path=sn_path, obj_type=object_type)
    df = open_parquet(parq_file, path=sn_path, obj_type=object_type)

    sim_true_flux = []
    sim_realized_flux = []
    for img in lc_model.image_list:
        if img.mjd < diaobj.mjd_start or img.mjd > diaobj.mjd_end:
            # If the image is outside the time range of the transient, skip it.
            continue
        try:
            catalogue_path = img.truthpath
        except AttributeError:
            # If using a ManualFITSImage, load the truthpath for a OU24 Image with that pointing and SCA
            img_collection = ImageCollection()
            img_collection = img_collection.get_collection("ou2024")
            dummy_image = img_collection.get_image(pointing=img.pointing, sca=img.sca, band=img.band)
            catalogue_path = dummy_image.truthpath

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
    sim_realized_mag, _, _ = calc_mag_and_err(sim_realized_flux, sim_sigma_flux, lc_model.image_list[0].band)
    sim_true_mag, _, _ = calc_mag_and_err(sim_true_flux, sim_sigma_flux, lc_model.image_list[0].band)

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
    data_dict = {"mjd": sim_mjd, "flux": flux, "flux_error": sigma_flux, "sim_flux": supernova}
    meta_dict = {}
    units = {"mjd": u.d, "sim_flux": "", "flux": "", "flux_error": ""}
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
    output_path = Config.get().value("photometry.campari.paths.output_dir") if output_path is None else output_path
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    lc_file = output_path / f"{identifier}_{band}_{psftype}_lc.ecsv"
    SNLogger.info(f"Saving lightcurve to {lc_file}")
    lc.write(lc_file, format="ascii.ecsv", overwrite=overwrite)


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


def extract_star_from_parquet_file_and_write_to_csv(parquet_file, sn_path, output_path, ra=None, dec=None, radius=None):
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
        SNLogger.warning(
            "extract_star_from_parquet_file_and_write_to_csv "
            + "got a radius argument with no units. Assuming degrees."
        )

        radius *= u.deg

    df = open_parquet(parquet_file, sn_path, obj_type="star")
    df = df[df["object_type"] == "star"]

    if radius is not None and (ra is not None and dec is not None):
        center_coord = SkyCoord(ra * u.deg, dec * u.deg)
        df_coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
        sep = center_coord.separation(df_coords)
        df = df[sep < radius]

    star_ID = df.id.values
    star_ID = np.array(star_ID, dtype=int)
    SNLogger.info(f"Found {np.size(star_ID)} stars in the given range.")
    if np.size(star_ID) == 0:
        raise ValueError("No stars found in the given range.")
    pd.DataFrame(star_ID).to_csv(output_path, index=False, header=False)
    SNLogger.info(f"Saved to {output_path}")


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
