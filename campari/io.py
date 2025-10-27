# Standard Library
import os
import pathlib
import uuid
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
from snappl.provenance import Provenance
from snappl.config import Config
from snappl.lightcurve import Lightcurve
from snappl.logger import SNLogger

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

    diaobj_prov = getattr(diaobj, "provenance_Id", None)
    provs = [diaobj_prov, lc_model.image_collection_prov]
    upstream_list = [p for p in provs if p is not None]

    cfg = Config.get()
    cam_prov = Provenance(
        process="campari",
        major=0,
        minor=42,
        params=cfg,  # keepkeys=["photometry.campari"],
        omitkeys=["photometry.campari.galsim", "photometry.campari.simulations"],
        upstreams=upstream_list,
    )

    meta_dict = cam_prov.params["photometry"]["campari"]
    meta_dict.update({"ID": diaobj.name, "ra": diaobj.ra, "dec": diaobj.dec})
    SNLogger.debug(f"meta dict in build_lightcurve: {meta_dict}")

    data_dict = {
        "mjd": [],
        "flux": flux,
        "flux_err": sigma_flux,
        "mag_fit": mag,
        "mag_fit_err": magerr,
        "zpt": np.full(np.size(mag), zp),
        "pointing": [],
        "sca": [],
        "pix_x": [],
        "pix_y": [],
        "x_cutout": [],
        "y_cutout": [],
        "sky_background": [],
        "sky_rms": []
    }

    for i, img in enumerate(image_list):
        if img.mjd > diaobj.mjd_start and img.mjd < diaobj.mjd_end:
            data_dict["mjd"].append(img.mjd)
            data_dict["pointing"].append(img.pointing)
            data_dict["sca"].append(img.sca)
            x, y = img.get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["pix_x"].append(x)
            data_dict["pix_y"].append(y)
            x_cutout, y_cutout = cutout_image_list[i].get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x_cutout"].append(x_cutout)
            data_dict["y_cutout"].append(y_cutout)
            data_dict["sky_background"].append(lc_model.sky_background[i])
            data_dict["sky_rms"].append(0.0) # placeholder for now XXX TODO

    SNLogger.debug(f"data dict in build_lightcurve: {data_dict}")

    SNLogger.debug("trying to build a lightcurve object")
    meta_dict["band"] = band  # I don't ever expect campari to do multi-band fitting so just store the one band.
    meta_dict["diaobject_position_id"] = "e98e579f-0ab3-4ad6-8042-2606d7d53014"  # placeholder for now XXX TODO
    meta_dict["provenance_id"] = cam_prov.id
    meta_dict["diaobject_id"] = diaobj.id
    meta_dict["iau_name"] = diaobj.name  # I am not sure this is what IAUname is but it's a placeholder for now.
    meta_dict["ra_err"] = 0.0
    meta_dict["dec_err"] = 0.0  # This is a placeholder for now
    meta_dict["ra_dec_covar"] = 0.0  # This is a placeholder for now
    # Note that this is only allowing for one band, not multiple bands. I don't think campari will ever
    # do multi-band fitting so this is probably fine.
    meta_dict[f"local_surface_brightness_{band}"] = lc_model.LSB
    data_dict["NEA"] = [0.0] * len(data_dict["pix_x"]) # snappl will calculate this

    SNLogger.debug("building lightcurve object")
    SNLogger.debug(f"data dict: {data_dict}")
    SNLogger.debug(f"meta dict: {meta_dict}")
    lc = Lightcurve(data=data_dict, meta=meta_dict)

    # return QTable(data=data_dict, meta=meta_dict, units=units)
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


def save_lightcurve(lc=None, identifier=None, psftype=None, output_path=None, overwrite=True):
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
    band = lc.meta["band"][0]
    output_path = Config.get().value("photometry.campari.paths.output_dir") if output_path is None else output_path
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    band_map = {
            "r": "R062",
            "z": "Z087",
            "y": "Y106",
            "j": "J129",
            "h": "H158",
            "f": "F184",
            "w": "W146",
        }

    # Here we handle band abbreviations
    if band not in list(band_map.values()):
        band = band_map[band.lower()]

    lc.write(
        base_dir=output_path, filepath=f"{identifier}_{band}_{psftype}_lc.ecsv", filetype="ecsv", overwrite=overwrite
    )


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
