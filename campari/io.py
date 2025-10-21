# Standard Library
import os
import pathlib
import warnings

# Common Library
import numpy as np
import pandas as pd
import yaml

# Astronomy Library
from astropy.table import QTable
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
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
    image_list = lc_model.image_list
    cutout_image_list = lc_model.cutout_image_list
    band = image_list[0].band
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)
    # ra and dec errors are set to 0 for now as they are currently being taken from truth tables.
    # I am not yet sure how to get provenance_id or the iau_name
    meta_dict = {"provenance_id": 0, "diaobject_id": diaobj.id, "iau_name": "placeholder", "ra": diaobj.ra, "ra_err": 0,
                 "dec": diaobj.dec, "dec_err": 0}
    meta_dict["local_surface_brightness"] = lc_model.LSB

    data_dict = {
        "mjd": [],
        "flux": flux,
        "flux_err": sigma_flux,
        "mag": mag,
        "mag_err": magerr,
        "band": [],
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
            data_dict["band"].append(img.band)
            data_dict["pointing"].append(img.pointing)
            data_dict["sca"].append(img.sca)
            x, y = img.get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x"].append(x)
            data_dict["y"].append(y)
            x_cutout, y_cutout = cutout_image_list[i].get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x_cutout"].append(x_cutout)
            data_dict["y_cutout"].append(y_cutout)

    units = {"mjd": u.d, "flux": "", "flux_err": "", "mag": u.mag, "mag_err": u.mag, "band": ""}

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

    sim_mjd = np.arange(0, np.size(supernova), 1)
    data_dict = {"mjd": sim_mjd, "flux": flux, "flux_error": sigma_flux, "sim_flux": supernova}
    meta_dict = {}
    units = {"mjd": u.d, "sim_flux": "", "flux": "", "flux_error": ""}
    return QTable(data=data_dict, meta=meta_dict, units=units)


def save_lightcurve(lc=None, identifier=None, psftype=None, output_path=None, overwrite=True, filetype="ecsv"):
    """This function parses settings in the SMP algorithm and saves the
    lightcurve to an ecsv or parquet file with an appropriate name.
    Input:
    lc: the lightcurve data
    identifier (str): the supernova ID or "simulated"
    band (str): the bandpass of the images used
    psftype (str): "romanpsf" or "analyticpsf"
    output_path (str): the path to save the lightcurve to.  Defaults to
      config value phtometry.campari.paths.output_dir
    filetype (str): "ecsv" or "parquet", the type of file to save.  Defaults to "ecsv"

    Returns:
    None, saves the lightcurve to a ecsv or parquet file.
    The file name is:
    <output_path>/identifier_band_psftype_lc.ecsv
    """
    band = lc["band"][0]
    output_path = Config.get().value("photometry.campari.paths.output_dir") if output_path is None else output_path
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    lc_file = output_path / f"{identifier}_{band}_{psftype}_lc.{filetype}"
    SNLogger.info(f"Saving lightcurve to {lc_file}")
    lc.write(lc_file, overwrite=overwrite)


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
