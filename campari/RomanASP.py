#!/usr/bin/env python
# Standard Library
import argparse
import warnings
import sys

# Common Library
import numpy as np
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from snpit_utils.config import Config
from campari.campari_runner import campari_runner
from snpit_utils.logger import SNLogger


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


def main():
    # Run one arg pass just to get the config file, so we can augment
    #   the full arg parser later with config options
    configparser = argparse.ArgumentParser(add_help=False)
    configparser.add_argument("-c", "--config", default=None,
                              help="Location of the .yaml config file")
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
            sys.stderr.write( "Error, no configuration file defined.\n"
                              "Either run campari with -c <configfile>\n"
                              "or set the SNPIT_CONFIG environment varaible.\n")
            sys.exit(1)
        else:
            raise
    desc += (" Include --config <configfile> before --help "
             "(or set SNPIT_CONFIG) for help to show you all config "
             "options that can be passed on the command line.")

    parser = argparse.ArgumentParser(description=desc)

    ####################
    # BASE CONFIG

    # This next argument will have been consumed by configparser above, and
    #   thus will never be parsed here, but include it so it shows up
    #   with --help.
    parser.add_argument("-c", "--config", default=None,
                        help="Location of the .yaml config file.  Defaults to "
                        "env var SNPIT_CONFIG.")

    parser.add_argument("-f", "--filter", type=str, required=True,
                        help="Roman filter")

    ####################
    # FINDING THE LOCATION ON THE SKY TO SCENE MODEL

    # If you specify -s or --SNID_file, then campari will look up (WHERE? TODO)
    # to find supernova RA and Dec
    parser.add_argument("-s", "--SNID", "--oid", type=int, default=None,
                        required=False, nargs="*",
                        help="Object IDs to run on. Meaning is dependent on the collection used."
                             "Will be ignored if --SNID-file is given")
    parser.add_argument("--SNID-file", "--oid_file", type=str, default=None, required=False,
                        help="Path to a csv file containing a list of "
                             "Object IDs to run on. Meaning of Object ID dependent on the collection used.")

    parser.add_argument("--healpix", type=int, default=None, required=False, nargs="*",
                        help="Healpix ID or IDs to run on. If given, will run on all "
                             "supernovae in that healpix. nside must be specified with the"
                             "--nside argument.")
    parser.add_argument("--nside", type=int, default=None, required=False,
                        help="Nside of the healpix to run on. If given, will run "
                             "on all supernovae in that healpix. This is a required argument if --healpix is given.")
    parser.add_argument("--healpix_file", type=str, default=None,  required=False,
                        help="A file of healpix IDs to run on. If given, "
                        "will run on all supernovae in those healpixes. nside must be specified with the"
                        "--nside argument.")

    # If instead you give --ra and --dec, it will assume there is a
    # point source at that position and will scene model a stamp around
    # it. (NOT YET SUPPORTED.)
    parser.add_argument("--ra", type=float, default=None,
                        help="RA of transient point source")
    parser.add_argument("--dec", type=float, default=None,
                        help="Dec of transient point source")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radius in degrees to search for supernovae "
                             "around the given RA and Dec. If not given, "
                             "will return the closest.")

    parser.add_argument("--object_collection", type=str, default="ou24", required=False,
                        help="Which collection of objects to use for lookup. "
                             "Default is 'ou24', the Open Universe 2024 catalog. 'manual'"
                             "will use the input ra and dec given by the user, and not perform any lookup.")
    ####################
    # FINDING THE IMAGES TO RUN SCENE MODELLING ON

    # If you give -n, -t, -b, and/or -e, then campari will decide
    #  (HOW?) what images to use.
    parser.add_argument("-n", "--max_no_transient_images", type=int, required=False,
                        help="Number of images to use that are treated as having no transient."
                        " Note that this is ignored if img_list is passed.",
                        default=None)
    parser.add_argument("-t", "--max_transient_images", type=int, required=False,
                        help="Number of images to use as having a transient present."
                        " Note that this is ignored if img_list is passed.",
                        default=None)
    parser.add_argument("-b", "--image_selection_start", type=float, required=False,
                        help="First MJD of images to be selected for use.",
                        default=None)
    parser.add_argument("-e", "--image_selection_end", type=float, required=False,
                        help="Last MJD of images to be selected for use.",
                        default=None)

    parser.add_argument(
        "--transient_start",
        type=float,
        required=False,
        help="MJD of first epoch of transient. Only used when --object_collection is manual. Otherwise, "
        " then the catalog values for transient_start, transient_end will be used."
        "If not given but transient_end is, will assume the first detection is at -inf.",
        default=None,
    )
    parser.add_argument(
        "--transient_end",
        type=float,
        required=False,
        help="MJD of last epoch of transient. Only used when --object_collection is manual. Otherwise, "
        " the catalog values for transient_start, transient_end will be used."
        " If not given but transient_start is, will assume the last detection is at +inf.",
        default=None,
    )

    # If instead you give img_list, then you expliclty list the images
    # used.  TODO: specify type of image, and adapt the code to handle
    # that.  Right now it will just assume openuniverse 2024.

    parser.add_argument(
        "-i",
        "--img_list",
        default=None,
        help="File with list of images. Note that if you pass an image list, the arguments "
        "--max_no_transient_images and --max_transient_images"
        " will be ignored, and campari will use all the images in the list.",
    )

    ####################
    # What does it mean to run on stars??????  Assume constant flux? No
    # host galaxy?  Ideally, the code to run on stars should be exactly
    # the same as the code to run on supernova.  Or does this have to do
    # with looking up the data in the opensim tables?
    parser.add_argument("--object_type", type=str, required=False,
                        choices=["star", "SN"],
                        help="If star, will run on stars. If SN, will run  " +
                             "on supernovae. If no argument is passed," +
                             "assumes supernova.",
                        default="SN")

    parser.add_argument("--fast_debug", action=argparse.BooleanOptionalAction, default=False,
                        help="If True, will run campari in fast debug mode, "
                             "which will enforce a very sparse grid. Data collected "
                             "using this method should not be used. ")

    if cfg is not None:
        cfg.augment_argparse(parser)
    args = parser.parse_args(leftovers)

    if cfg is None:
        raise ValueError("Must pass a config file, or must set SNPIT_CONFIG")
    cfg.parse_args(args)

    runner = campari_runner(**vars(args))
    runner()


if __name__ == "__main__":
    main()
