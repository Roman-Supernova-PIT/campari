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


def build_lightcurve(diaobj, lc_model, obj_pos_prov=None):
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
    SNLogger.debug(f"building lightcurve for diaobj {diaobj.name} in band {band}")
    mag, magerr, zp = calc_mag_and_err(flux, sigma_flux, band)

    diaobj_prov = getattr(diaobj, "provenance_Id", None)
    provs = [diaobj_prov, lc_model.image_collection_prov]
    upstream_list = [p for p in provs if p is not None]

    cfg = Config.get()
    cam_prov = Provenance(
        process="campari",
        major=0,
        minor=42, # THIS CAN'T BE HARDCODED FOREVER XXX TODO
        params=cfg,
        keepkeys=["photometry.campari"],
        omitkeys=None,
        upstreams=upstream_list,
    )

    meta_dict = cam_prov.params["photometry"]["campari"]
    meta_dict.update({"ID": diaobj.name, "ra": diaobj.ra, "dec": diaobj.dec})

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
        "sky_rms": [],
        "NEA": [],
    }

    for i, img in enumerate(image_list):
        if img.mjd > diaobj.mjd_start and img.mjd < diaobj.mjd_end:
            data_dict["mjd"].append(img.mjd)
            SNLogger.debug(f"Adding image with MJD {img.mjd} to lightcurve")
            data_dict["pointing"].append(img.pointing)
            data_dict["sca"].append(img.sca)
            x, y = img.get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["pix_x"].append(x)
            data_dict["pix_y"].append(y)
            x_cutout, y_cutout = cutout_image_list[i].get_wcs().world_to_pixel(diaobj.ra, diaobj.dec)
            data_dict["x_cutout"].append(x_cutout)
            data_dict["y_cutout"].append(y_cutout)
            data_dict["sky_background"].append(lc_model.sky_background[i])
            data_dict["sky_rms"].append(0.0)  # placeholder for now XXX TODO
            data_dict["NEA"].append(0.0)  # placeholder for now XXX TODO

    meta_dict["band"] = band  # I don't ever expect campari to do multi-band fitting so just store the one band.
    meta_dict["diaobject_position_id"] = None  # placeholder for now XXX TODO
    meta_dict["provenance_id"] = cam_prov.id
    meta_dict["diaobject_id"] = diaobj.id
    meta_dict["iau_name"] = diaobj.iauname
    meta_dict["ra_err"] = 0.0
    meta_dict["dec_err"] = 0.0  # This is a placeholder for now
    meta_dict["ra_dec_covar"] = 0.0  # This is a placeholder for now
    # Note that this is only allowing for one band, not multiple bands. I don't think campari will ever
    # do multi-band fitting so this is probably fine.
    meta_dict[f"local_surface_brightness_{band}"] = lc_model.LSB

    lc = Lightcurve(data=data_dict, meta=meta_dict)
    # Some extra info needed to save
    lc.image_list = image_list
    lc.diaobj = diaobj

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


def save_lightcurve(lc=None, identifier=None, psftype=None, output_path=None,
                    overwrite=True, save_to_database=False, dbclient=None,
                    new_provenance=False, diaobj_pos=None,
                    ltcvprocess=None):
    """This function parses settings in the SMP algorithm and saves the
    lightcurve to an ecsv file with an appropriate name.
    Input:
    lc: the lightcurve data, in the form of a snappl.lightcurve.Lightcurve object
    identifier (str): the supernova ID or "simulated"
    band (str): the bandpass of the images used
    psftype (str): "romanpsf" or "analyticpsf"
    output_path (str): the path to save the lightcurve to.  Defaults to
      config value system.paths.lightcurves

    Returns:
    None, saves the lightcurve to a ecsv file.
    The file name is:
    <output_path>/identifier_band_psftype_lc.ecsv
    """
    band = lc.meta["band"]
    SNLogger.debug(f"saving lightcurve for id={identifier}, band={band}, psftype={psftype}")

    if save_to_database:
        if output_path is not None:
            raise ValueError("output_path must be None when save_to_database is True.")
        else:
            base_output_path = Config.get().value("system.paths.lightcurves")
    else:
        if output_path is None:
            base_output_path = Config.get().value("system.paths.output_dir")
        else:
            base_output_path = output_path

    base_output_path = pathlib.Path(base_output_path)
    base_output_path.mkdir(exist_ok=True, parents=True)

    filepath = f"{identifier}_{band}_{psftype}_lc.ecsv" if not save_to_database else None

    if save_to_database:

        # process = "campari"
        # major = 0
        # minor = 42  # THIS CAN'T BE HARDCODED FOREVER XXX TODO
        # params = Config.get()
        # keepkeys = ["photometry.campari"]
        # omitkeys = None
        # imgprov = Provenance.get_by_id(lc.image_list[0].provenance_id, dbclient=dbclient)
        # objprov = Provenance.get_by_id(lc.diaobj.provenance_id, dbclient=dbclient)
        # if diaobj_pos is not None:
        #     objposprov = Provenance.get_by_id(diaobj_pos, dbclient=dbclient)
        #     upstreams = [imgprov, objprov, objposprov]
        # else:
        #     objposprov = None
        #     upstreams = [imgprov, objprov]
        SNLogger.debug("Creating new provenance for lightcurve")

        # ltcvprov = Provenance(process=process, major=major, minor=minor, params=params,
        #                       upstreams=upstreams, keepkeys=keepkeys,
        #                       omitkeys=omitkeys)
        SNLogger.debug("Using provenance to save lightcurve to database")
        SNLogger.debug(lc.meta.get("provenance_id"))
        ltcvprov = Provenance.get_by_id(lc.meta.get("provenance_id"), dbclient=dbclient)
        ltcv_provenance_tag = "campari_test_provenance_lightcurve"
        if new_provenance:
            ltcvprov.save_to_db(tag=ltcv_provenance_tag)
        lc.save_to_db(dbclient=dbclient)
    else:
        lc.write(
            base_dir=output_path, filepath=filepath, filetype="ecsv", overwrite=overwrite
        )
    # Return the lc so we can have the snappl generated filepath
    return lc


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
