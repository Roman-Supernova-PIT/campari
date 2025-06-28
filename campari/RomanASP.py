# Standard Library
import argparse
import pathlib
import warnings

# Common Library
import galsim
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
import snappl
from snappl.sed import OU2024_Truth_SED
from snappl.sed import Flat_SED
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger as Lager

# Campari
from campari.AllASPFuncs import (banner,
                                 build_lightcurve,
                                 build_lightcurve_sim,
                                 findAllExposures,
                                 find_parquet,
                                 get_object_info,
                                 run_one_object,
                                 save_lightcurve)

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
    except RuntimeError:
        # If it failed to load the config file, just move on with life.  This
        #   may mean that things will fail later, but it may also just mean
        #   that somebody is doing "--help"
        cfg = None
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
    parser.add_argument("-s", "--SNID", type=int, default=None,
                        required=False, nargs="*",
                        help="OpenUniverse2024 Supernova IDs; ignored if"
                             " --SNID-file is given")
    parser.add_argument("--SNID-file", type=str, default=None, required=False,
                        help="Path to a csv file containing a list of "
                             "OpenUniverse SNIDs to run.")

    # If instead you give --ra and --dec, it will assume there is a
    # point source at that position and will scene model a stamp around
    # it. (NOT YET SUPPORTED.)
    parser.add_argument("--ra", type=float, default=None,
                        help="RA of transient point source")
    parser.add_argument("--dec", type=float, default=None,
                        help="Dec of transient point source")

    ####################
    # FINDING THE IMAGES TO RUN SCENE MODELLING ON

    # If you give -t, -d, -b, and/or -e, then campari will decide
    #  (HOW?) what images to use.
    parser.add_argument("-t", "--num_total_images", type=int, required=False,
                        help="Number of images to use", default=np.inf)
    # TODO:change all instances of this variable to tot_images
    parser.add_argument("-d", "--num_detect_images", type=int, required=False,
                        help="Number of images to use with SN detections",
                        default=np.inf)
    # TODO:change all instances of this variable to det_images
    parser.add_argument("-b", "--beginning", type=int, required=False,
                        help="start of desired lightcurve in days from peak.",
                        default=-np.inf)
    parser.add_argument("-e", "--end", type=int, required=False,
                        help="end of desired light curve in days from peak.",
                        default=np.inf)

    # If instead you give imglist, then you expliclty list the images
    # used.  TODO: specify type of image, and adapt the code to handle
    # that.  Right now it will just assume openuniverse 2024.

    parser.add_argument("-i", "--img-list", default=None,
                        help="File with list of images")

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

    if cfg is not None:
        cfg.augment_argparse(parser)
    args = parser.parse_args(leftovers)

    if cfg is None:
        raise ValueError("Must pass a config file, or must set SNPIT_CONFIG")
    cfg.parse_args(args)

    band = args.filter

    if (args.ra is not None) or (args.dec is not None):
        raise NotImplementedError("--ra and --dec not yet supported.")
    if args.SNID_file is not None:
        SNID = pd.read_csv(args.SNID_file, header=None).values.flatten().tolist()
    else:
        SNID = args.SNID

    if args.img_list is not None:
        raise NotImplementedError("--img-list not yet supported.")

    num_total_images = args.num_total_images
    num_detect_images = args.num_detect_images
    lc_start = args.beginning
    lc_end = args.end
    object_type = args.object_type

    config = Config.get(args.config, setdefault=True)

    size = config.value("photometry.campari.cutout_size")
    use_real_images = config.value("photometry.campari.use_real_images")
    use_roman = config.value("photometry.campari.use_roman")
    check_perfection = config.value("photometry.campari.simulations.check_perfection")
    make_exact = config.value("photometry.campari.simulations.make_exact")
    avoid_non_linearity = config.value("photometry.campari.simulations.avoid_non_linearity")
    deltafcn_profile = config.value("photometry.campari.simulations.deltafcn_profile")
    do_xshift = config.value("photometry.campari.simulations.do_xshift")
    do_rotation = config.value("photometry.campari.simulations.do_rotation")
    noise = config.value("photometry.campari.simulations.noise")
    method = config.value("photometry.campari.method")
    make_initial_guess = config.value("photometry.campari.make_initial_guess")
    subtract_background = config.value("photometry.campari.subtract_background")
    weighting = config.value("photometry.campari.weighting")
    pixel = config.value("photometry.campari.pixel")
    roman_path = config.value("photometry.campari.paths.roman_path")
    sn_path = config.value("photometry.campari.paths.sn_path")
    bg_gal_flux = config.value("photometry.campari.simulations.bg_gal_flux")
    source_phot_ops = config.value("photometry.campari.source_phot_ops")
    mismatch_seds = config.value("photometry.campari.simulations.mismatch_seds")
    fetch_SED = config.value("photometry.campari.fetch_SED")
    initial_flux_guess = config.value("photometry.campari.initial_flux_guess")
    sim_gal_ra_offset = config.value("photometry.campari.simulations.sim_gal_ra_offset")
    sim_gal_dec_offset = config.value("photometry.campari.simulations.sim_gal_dec_offset")
    spacing = config.value("photometry.campari.grid_options.spacing")
    percentiles = config.value("photometry.campari.grid_options.percentiles")
    grid_type = config.value("photometry.campari.grid_options.type")

    er = f"{grid_type} is not a recognized grid type. Available options are "
    er += "regular, adaptive, contour, or single. Details in documentation."
    assert grid_type in ["regular", "adaptive", "contour",
                         "single", "none"], er

    # PSF for when not using the Roman PSF:
    lam = 1293  # nm
    aberrations = galsim.roman.getPSF(1, band, pupil_bin=1).aberrations
    airy = galsim.ChromaticOpticalPSF(lam, diam=2.36,
                                      aberrations=aberrations)

    if make_exact:
        assert grid_type == "single"
    if avoid_non_linearity:
        assert deltafcn_profile
    assert num_detect_images <= num_total_images

    galsim.roman.roman_psfs._make_aperture.clear()  # clear cache

    if not isinstance(SNID, list):
        SNID = [SNID]
    Lager.debug("Snappl version:")
    Lager.debug(snappl.__version__)
    # run one supernova function TODO
    for ID in SNID:
        banner(f"Running SN {ID}")
        try:
            # Pull out list of image exposures so we can be more flexible.
            # Can then call to get the SEDs here.

            pqfile = find_parquet(ID, sn_path, obj_type=object_type)
            Lager.debug(f"Found parquet file {pqfile} for SN {ID}")

            ra, dec, p, s, start, end, peak = get_object_info(ID, pqfile, band=band, snpath=sn_path,
                                                              roman_path=roman_path, obj_type=object_type)
            Lager.debug(f"Object info for SN {ID}: ra={ra}, dec={dec}")

            exposures = findAllExposures(ID, ra, dec, start, end,
                                         roman_path=roman_path,
                                         maxbg=num_total_images - num_detect_images,
                                         maxdet=num_detect_images, return_list=True,
                                         band=band, lc_start=lc_start, lc_end=lc_end)
            if fetch_SED:
                sed_obj = OU2024_Truth_SED(ID, isstar=(object_type == "star"))
            else:
                sed_obj = Flat_SED()

            sedlist = []
            for date in exposures["date"]:
                sedlist.append(sed_obj.get_sed(snid=ID, mjd=date))

            flux, sigma_flux, images, sumimages, exposures, ra_grid, dec_grid, wgt_matrix, \
                confusion_metric, X, cutout_wcs_list, sim_lc = \
                run_one_object(ID, ra, dec, object_type, exposures, num_total_images, num_detect_images, roman_path,
                               sn_path, size, band, fetch_SED, sedlist, use_real_images,
                               use_roman, subtract_background,
                               make_initial_guess, initial_flux_guess,
                               weighting, method, grid_type,
                               pixel, source_phot_ops,
                               lc_start, lc_end, do_xshift, bg_gal_flux,
                               do_rotation, airy, mismatch_seds, deltafcn_profile,
                               noise, check_perfection, avoid_non_linearity,
                               sim_gal_ra_offset, sim_gal_dec_offset,
                               spacing, percentiles)
        # I don't have a particular error in mind for this, but I think
        # it's worth having a catch just in case that one supernova fails,
        # this way the rest of the code doesn't halt.
        except ValueError as e:
            Lager.info(f"ValueError: {e}")
            continue

        # Saving the output. The output needs two sections, one where we
        # create a lightcurve compared to true values, and one where we save
        # the images.

        if use_real_images:
            identifier = str(ID)
            lc = build_lightcurve(ID, exposures, sn_path, roman_path,
                                  confusion_metric, flux, use_roman, band,
                                  object_type, sigma_flux)
        else:
            identifier = "simulated"
            lc = build_lightcurve_sim(sim_lc, flux, sigma_flux)
        if use_roman:
            psftype = "romanpsf"
        else:
            psftype = "analyticpsf"

        output_dir = pathlib.Path(cfg.value("photometry.campari.paths.output_dir"))
        save_lightcurve(lc, identifier, band, psftype,
                        output_path=output_dir)

        # Now, save the images
        images_and_model = np.array([images, sumimages, wgt_matrix])
        debug_dir = pathlib.Path(cfg.value("photometry."
                                 "campari.paths.debug_dir"))
        Lager.info(f"Saving images to {debug_dir}")
        np.save(debug_dir / f"{identifier}_{band}_{psftype}_images.npy",
                images_and_model)

        # Save the ra and decgrid
        np.save(debug_dir / f"{identifier}_{band}_{psftype}_grid.npy",
                [ra_grid, dec_grid, X[:np.size(ra_grid)]])

        # save wcses
        primary_hdu = fits.PrimaryHDU()
        hdul = [primary_hdu]
        for i, wcs in enumerate(cutout_wcs_list):
            hdul.append(fits.ImageHDU(header=wcs.to_fits_header(),
                        name="WCS" + str(i)))
        hdul = fits.HDUList(hdul)
        filepath = debug_dir / f"{identifier}_{band}_{psftype}_wcs.fits"
        hdul.writeto(filepath, overwrite=True)


if __name__ == "__main__":
    main()
