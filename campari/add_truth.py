#!/usr/bin/env python
# Standard Library
import argparse
import os
import sys
import warnings

# Common Library
import numpy as np
import pandas as pd

# Astronomy Library
from astropy.table import QTable, hstack
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from campari.utils import calc_mag_and_err
from snappl.imagecollection import ImageCollection
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger


# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


def main():
    # Run one arg pass just to get the config file, so we can augment
    #   the full arg parser later with config options
    configparser = argparse.ArgumentParser(add_help=False)
    configparser.add_argument("-c", "--config", default=None, help="Location of the .yaml config file")
    args, leftovers = configparser.parse_known_args()

    desc = "Run the campari pipeline."
    try:
        cfg = Config.get(args.config, setdefault=True)
    # except RuntimeError:
    #     # If it failed to load the config file, just move on with life.  This
    #     #   may mean that things will fail later, but it may also just mean
    #     #   that somebody is doing "--help"
    #     cfg = None
    except RuntimeError as e:
        if str(e) == "No default config defined yet; run Config.init(configfile)":
            sys.stderr.write(
                "Error, no configuration file defined.\n"
                "Either run campari with -c <configfile>\n"
                "or set the SNPIT_CONFIG environment varaible.\n"
            )
            sys.exit(1)
        else:
            raise
    desc += (
        " Include --config <configfile> before --help "
        "(or set SNPIT_CONFIG) for help to show you all config "
        "options that can be passed on the command line."
    )

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "lightcurve", type=str, required=True, help="Path to lightcurve file to process."
    )


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


def add_truth_to_lc(lc, sn_path, object_type="SN"):
    """This code adds the truth flux and magnitude to a lightcurve datatable."""

    ID = lc.meta["ID"]
    parq_file = find_parquet(ID, path=sn_path, obj_type=object_type)
    df = open_parquet(parq_file, path=sn_path, obj_type=object_type)
    band = lc["filter"][0]

    sim_true_flux = []
    sim_realized_flux = []
    for pointing, sca in zip(lc["pointing"], lc["sca"]):
        # Load the truthpath for a OU24 Image with that pointing and SCA
        img_collection = ImageCollection()
        img_collection = img_collection.get_collection("ou2024")
        dummy_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)
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
