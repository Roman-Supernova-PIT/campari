import pathlib

import numpy as np
import pandas as pd

# Astronomy
from astropy.io import fits
import galsim


# SN-PIT
from snappl.dbclient import SNPITDBClient
from snappl.diaobject import DiaObject
from snappl.image import FITSImageStdHeaders
from snappl.sed import Flat_SED, OU2024_Truth_SED
from snappl.config import Config
from snappl.logger import SNLogger

# Campari
from campari.access_truth import add_truth_to_lc, extract_object_from_healpix
from campari.data_construction import find_all_exposures
from campari.io import (
    build_lightcurve,
    build_lightcurve_sim,
    read_healpix_file,
    save_lightcurve,
)
from campari.run_one_object import run_one_object
from campari.utils import banner


class campari_runner:
    """This class is used to run the Campari pipeline."""

    def __init__(self, **kwargs):
        """Initialize the Campari runner with all of the variables needed to run the pipeline.
        NOTE: Config must be set before running this function."""

        self.cfg = Config.get()

        self.band = kwargs["filter"]
        self.max_no_transient_images = kwargs["max_no_transient_images"]
        self.max_transient_images = kwargs["max_transient_images"]
        self.image_selection_start = kwargs["image_selection_start"]
        self.image_selection_end = kwargs["image_selection_end"]
        self.object_type = kwargs["object_type"]
        self.fast_debug = kwargs["fast_debug"]
        self.SNID_file = kwargs["SNID_file"]
        self.SNID = kwargs["SNID"]
        self.img_list = kwargs["img_list"]
        self.image_source = kwargs["image_source"]

        self.healpix = kwargs["healpix"]
        self.healpix_file = kwargs["healpix_file"]
        self.nside = kwargs["nside"]
        self.object_collection = kwargs["object_collection"]
        self.transient_start = kwargs["transient_start"]
        self.transient_end = kwargs["transient_end"]

        self.ra = kwargs["ra"]
        self.dec = kwargs["dec"]
        self.save_model = kwargs["save_model"]
        self.prebuilt_static_model = kwargs["prebuilt_static_model"]
        self.prebuilt_transient_model = kwargs["prebuilt_transient_model"]

        self.size = self.cfg.value("photometry.campari.cutout_size")
        self.use_real_images = self.cfg.value("photometry.campari.use_real_images")
        self.avoid_non_linearity = self.cfg.value("photometry.campari.simulations.avoid_non_linearity")
        self.deltafcn_profile = self.cfg.value("photometry.campari.simulations.deltafcn_profile")
        self.do_xshift = self.cfg.value("photometry.campari.simulations.do_xshift")
        self.do_rotation = self.cfg.value("photometry.campari.simulations.do_rotation")
        self.psfclass = self.cfg.value("photometry.campari.psfclass")
        self.noise = self.cfg.value("photometry.campari.simulations.noise")
        self.method = self.cfg.value("photometry.campari.method")
        self.make_initial_guess = self.cfg.value("photometry.campari.make_initial_guess")
        self.subtract_background = self.cfg.value("photometry.campari.subtract_background")
        self.weighting = self.cfg.value("photometry.campari.weighting")
        self.pixel = self.cfg.value("photometry.campari.pixel")
        self.sn_truth_dir = self.cfg.value("system.ou24.sn_truth_dir")
        self.bg_gal_flux_all = self.cfg.value("photometry.campari.simulations.bg_gal_flux")
        self.sim_galaxy_scale_all = self.cfg.value("photometry.campari.simulations.sim_galaxy_scale")
        self.sim_galaxy_offset_all = self.cfg.value("photometry.campari.simulations.sim_galaxy_offset")
        self.source_phot_ops = self.cfg.value("photometry.campari.source_phot_ops")
        self.mismatch_seds = self.cfg.value("photometry.campari.simulations.mismatch_seds")
        self.fetch_SED = self.cfg.value("photometry.campari.fetch_SED")
        self.initial_flux_guess = self.cfg.value("photometry.campari.initial_flux_guess")
        self.spacing = self.cfg.value("photometry.campari.grid_options.spacing")
        self.percentiles = self.cfg.value("photometry.campari.grid_options.percentiles")
        self.grid_type = self.cfg.value("photometry.campari.grid_options.type")
        self.base_pointing = self.cfg.value("photometry.campari.simulations.base_pointing")
        self.base_sca = self.cfg.value("photometry.campari.simulations.base_sca")
        self.run_name = self.cfg.value("photometry.campari.simulations.run_name")
        self.save_debug = self.cfg.value("photometry.campari.save_debug")
        self.param_grid = None
        self.run_mode = None
        self.noise_maps = None
        self.galaxy_images = None
        self.galaxy_only_model_images = None
        self.dbclient = SNPITDBClient()
        self.img_coll_prov = None

        if self.fast_debug:
            SNLogger.debug("Overriding config to run in fast debug mode.")
            self.grid_type = "regular"
            self.spacing = 9
            self.size = 11
            self.source_phot_ops = False
            self.fetch_SED = False
            self.make_initial_guess = False

        if self.grid_type == "single" and not self.deltafcn_profile:
            SNLogger.warning("Using a single point on the grid without a delta function profile is not recommended."
                             "The goal of using a single point is to run an exact fit for testing purposes,"
                             "which requires "
                             "the galaxy be a delta function.")

        # PSF for when not using the Roman PSF:
        lam = 1293  # nm
        aberrations = galsim.roman.getPSF(1, self.band, pupil_bin=1).aberrations
        self.airy = galsim.ChromaticOpticalPSF(lam, diam=2.36, aberrations=aberrations)

        er = f"{self.grid_type} is not a recognized grid type. Available options are "
        er += "regular, adaptive, contour, single, or none. Details in documentation."
        if self.grid_type not in ["regular", "adaptive", "contour", "single", "none"]:
            raise ValueError(er)

        if self.max_no_transient_images is None or self.max_transient_images is None:
            self.max_images = None
        else:
            self.max_images = self.max_no_transient_images + self.max_transient_images

        if self.object_type == "star":
            self.max_no_transient_images = 0
            SNLogger.debug("Running on stars, so setting max_no_transient_images to 0.")

    def __call__(self):
        """Run the Campari pipeline."""
        self.decide_run_mode()
        if not self.use_real_images:
            self.create_sim_param_grid()

        for index, ID in enumerate(self.SNID):
            banner(f"Running SN {ID}")

            # These will need to be re included once Issue #93 is resolved.
            # ra=self.ra, dec=self.dec
            #    mjd_discovery_min=self.transient_start, mjd_discovery_max=self.transient_end

            if self.object_collection == "manual":
                provenance_tag = None
                process = None
                diaobjs = DiaObject.find_objects(collection=self.object_collection, dbclient=self.dbclient,
                                                 provenance_tag=provenance_tag, process=process, name=ID,
                                                 ra=self.ra, dec=self.dec, mjd_discovery_min=self.transient_start,
                                                 mjd_discovery_max=self.transient_end)
            else:
                provenance_tag = "ou2024"
                process = "load_ou2024_diaobject"
                diaobjs = DiaObject.find_objects(collection=self.object_collection, dbclient=self.dbclient,
                                             provenance_tag=provenance_tag, process=process, name=ID)

            SNLogger.debug(f"Searching for DiaObject with id={ID}, ra={self.ra}, dec={self.dec},"
                           f" collection={self.object_collection}, provenance_tag={provenance_tag}, process={process}")





            if len(diaobjs) == 0:
                raise ValueError(f"Could not find DiaObject with id={ID}, ra={self.ra}, dec={self.dec}.")
            if len(diaobjs) > 1:
                raise ValueError(f"Found multiple DiaObject with id={ID}, ra={self.ra}, dec={self.dec}.")
            diaobj = diaobjs[0]
            if self.ra is not None:
                if np.fabs(self.ra - diaobj.ra) > 1. / 3600. / np.cos(diaobj.dec * np.pi / 180.):
                    SNLogger.warning(f"Given RA {self.ra} is far from DiaObject nominal RA {diaobj.ra}")
                diaobj.ra = self.ra
            if self.dec is not None:
                if np.fabs(self.dec - diaobj.dec) > 1. / 3600.:
                    SNLogger.warning(f"Given Dec {self.dec} is far from DiaObject nominal Dec {diaobj.dec}")
                diaobj.dec = self.dec

            if (self.transient_start is not None):
                if (diaobj.mjd_start is not None) and np.fabs(self.transient_start - diaobj.mjd_start) > .1:
                    SNLogger.warning(f"Given transient_start {self.transient_start} is far from DiaObject "
                                     f"nominal transient_start {diaobj.mjd_start}")
                diaobj.mjd_start = self.transient_start

            if self.transient_end is not None:
                if (diaobj.mjd_end is not None) and np.fabs(self.transient_end - diaobj.mjd_end) > .1:
                    SNLogger.warning(f"Given transient_end {self.transient_end} is far from DiaObject "
                                     f"nominal transient_end {diaobj.mjd_end}")
                diaobj.mjd_end = self.transient_end

            SNLogger.debug(f"Object info for SN {ID} in collection {self.object_collection}: ra={diaobj.ra},"
                           f" dec={diaobj.dec}, transient_start={diaobj.mjd_start}, transient_end={diaobj.mjd_end}")
            image_list = self.get_exposures(diaobj)
            sedlist = self.get_sedlist(diaobj.name, image_list)

            # This has to go after get_exposures because the infs break the simdex.
            if diaobj.mjd_start is None:
                diaobj.mjd_start = -np.inf
            if diaobj.mjd_end is None:
                diaobj.mjd_end = np.inf

            param_grid_row = self.param_grid[:, index] if self.param_grid is not None else None

            lightcurve_model = self.call_run_one_object(diaobj, image_list, sedlist, param_grid_row)
            self.build_and_save_lightcurve(diaobj, lightcurve_model, param_grid_row)

    def decide_run_mode(self):
        """Decide which run mode to use based on the input configuration."""

        # Option 1, user passes a file of SNIDs
        if self.SNID_file is not None:
            self.SNID = pd.read_csv(self.SNID_file, header=None).values.flatten().tolist()
            self.run_mode = "SNID File"

        # Option 2, user passes a SNID
        elif self.SNID is not None:
            self.run_mode = "Single SNID"

        # Option 3, user passes a ra and dec, meaning we don't search for SNID.
        elif (self.ra is not None) or (self.dec is not None):
            self.run_mode = "RA/Dec"
            if self.transient_start is None and self.transient_end is None:
                raise ValueError("Must specify --transient_start and --transient_end to run campari at a"
                                 " given RA and Dec.")
            SNLogger.debug(
                "Forcing campari to run on the given RA and Dec, "
                f" RA={self.ra}, Dec={self.dec} with transient flux fit for between "
                f"MJD {self.transient_start} and {self.transient_end}."
            )

        # Option 4, user passes a healpix and nside, meaning we search for SNe in healpix via ra/dec.
        elif self.healpix is not None or self.healpix_file is not None:
            if self.healpix is not None:
                self.healpixes = [self.healpix]
                self.run_mode = "Healpix"
            else:
                self.healpixes, self.nside = read_healpix_file(self.healpix_file)
                self.run_mode = "Healpix File"

            if self.nside is None:
                if self.nside is not None:
                    pass
                else:
                    raise ValueError("--nside was not passed, and nside was not found in the healpix file. ")

            SNLogger.debug(f"Running on {len(self.healpixes)} healpixes with nside {self.nside}.")

            SNID = []
            for healpix in self.healpixes:
                SNID.extend(extract_object_from_healpix(healpix, self.nside, object_type=self.object_type,
                            source="OpenUniverse2024"))

        elif self.object_collection != "manual" and (self.SNID is None) and (self.SNID_file is None):
            raise ValueError(
                "Must specify --SNID, --SNID-file, to run campari with a non-manual object collection. Note that"
                " --object_collection is ou2024 by default, so if you want to run campari without looking up a SNID,"
                " you must set --object_collection to manual and provide --ra and --dec."
            )
        else:
            raise ValueError(
                "Must specify --SNID, --SNID-file, --healpix, --healpix_file, or --ra and --dec to run campari."
            )

        if self.img_list is not None:
            columns = ["pointing", "sca"]
            image_df = pd.read_csv(self.img_list, header=None, names=columns)
            # If provided a list, we want to make sure we continue searching until all the images are found. So we set:
            self.max_no_transient_images = None
            self.max_transient_images = None
            self.pointing_list = image_df["pointing"].values
        else:
            image_df = None
            self.pointing_list = None

        if not isinstance(self.SNID, list):
            self.SNID = [self.SNID]

        SNLogger.debug(f"Running campari in {self.run_mode} mode with {len(self.SNID)} SNIDs.")

    def create_sim_param_grid(self):
        """Create a grid of simulation parameters to run the pipeline on."""
        params = [self.bg_gal_flux_all, self.sim_galaxy_scale_all, self.sim_galaxy_offset_all]
        nd_grid = np.meshgrid(*params)
        self.param_grid = np.array(nd_grid, dtype=float).reshape(len(params), -1)
        SNLogger.debug("Created a grid of simulation parameters with a total of"
                       f" {self.param_grid.shape[1]} combinations.")
        self.SNID = self.SNID * self.param_grid.shape[1]  # Repeat the SNID for each combination of parameters

    def get_exposures(self, diaobj):
        """Call the find_all_exposures function to get the exposures for the given RA, Dec, and time frame."""
        if self.use_real_images:
            image_list, self.img_coll_prov = find_all_exposures(diaobj=diaobj,
                                                                maxbg=self.max_no_transient_images,
                                                                maxdet=self.max_transient_images,
                                                                band=self.band,
                                                                image_selection_start=self.image_selection_start,
                                                                image_selection_end=self.image_selection_end,
                                                                image_source=self.image_source,
                                                                pointing_list=self.pointing_list,
                                                                dbclient=self.dbclient)
            mjd_start = diaobj.mjd_start if diaobj.mjd_start is not None else -np.inf
            mjd_end = diaobj.mjd_end if diaobj.mjd_end is not None else np.inf

            no_transient_images = [a for a in image_list if (a.mjd < mjd_start) or (a.mjd > mjd_end)]

            if (
                self.max_no_transient_images != 0
                and len(no_transient_images) == 0
                and self.object_type != "star"
                and self.img_list is None  # If passing an image list, I assume the user knows what they are doing.
            ):
                raise ValueError("No non-detection images were found. This may be because the transient is"
                                 " detected in all images, or because the transient is outside the date range of"
                                 " available images. If you are running on stars, this is expected behavior."
                                 " If you are running on supernovae, consider increasing the date range.")
        else:
            if self.max_no_transient_images is None or self.max_transient_images is None:
                raise ValueError("Must specify --max_no_transient_images and --max_transient_images to run campari with"
                                 " simulated images.")
            num_images = self.max_no_transient_images + self.max_transient_images

            faux_dates = np.linspace(60000, diaobj.mjd_start, self.max_no_transient_images).tolist() + \
                np.linspace(diaobj.mjd_start, diaobj.mjd_end, self.max_transient_images).tolist()
            faux_dates = np.array(faux_dates)
            # fake dates for simulated images
            image_list = []
            for i in range(num_images):
                # These data sizes are arbitary. I just need a data array present in order to perform the cutout,
                # otherwise snappl throws an error. No data is actually placed into these images until they are
                # cutout sized and the 4088 is used nowhere.
                img = FITSImageStdHeaders(
                    header=None, data=np.zeros((4088, 4088)), noise=np.zeros((4088, 4088)),
                    flags=np.zeros((4088, 4088)), path="/dev/null"
                )
                img.mjd = faux_dates[i]
                img.band = self.band
                image_list.append(img)
                img.pointing = self.base_pointing
                img.sca = self.base_sca

        recovered_pointings = [a.pointing for a in image_list]
        if self.img_list is not None and not np.array_equiv(np.sort(recovered_pointings),
                                                            np.sort(self.pointing_list)):
            SNLogger.warning("Unable to find the object in all the pointings in the image list. Specifically, the"
                             " following pointings were not found: "
                             f"{np.setdiff1d(self.pointing_list, [a.pointing for a in image_list])}")

        SNLogger.debug(f"Found {len(image_list)} exposures")
        return image_list

    def get_sedlist(self, ID, image_list):
        """Create a list of SEDs for the given SNID and images."""
        sed_obj = OU2024_Truth_SED(ID, isstar=(self.object_type == "star")) if self.fetch_SED else Flat_SED()
        sedlist = []
        for img in image_list:
            sedlist.append(sed_obj.get_sed(snid=ID, mjd=img.mjd))
        return sedlist

    def call_run_one_object(self, diaobj, image_list, sedlist, param_grid_row):
        """Call the run_one_object function to run the pipeline for a given SNID and exposures."""

        prebuilt_psf_matrix = np.load(self.prebuilt_static_model) if self.prebuilt_static_model is not None else None
        prebuilt_sn_matrix = np.load(self.prebuilt_transient_model) if self.prebuilt_transient_model is not None \
            else None

        if not self.use_real_images:
            bg_gal_flux, sim_galaxy_scale, sim_galaxy_offset = param_grid_row
        else:
            bg_gal_flux, sim_galaxy_scale, sim_galaxy_offset = None, None, None
        SNLogger.debug("Save model is set to " + str(self.save_model))
        lightcurve_model = \
            run_one_object(diaobj=diaobj, object_type=self.object_type, image_list=image_list,
                           size=self.size, band=self.band, psfclass=self.psfclass,
                           fetch_SED=self.fetch_SED, sedlist=sedlist, use_real_images=self.use_real_images,
                           subtract_background=self.subtract_background,
                           make_initial_guess=self.make_initial_guess, initial_flux_guess=self.initial_flux_guess,
                           weighting=self.weighting, method=self.method, grid_type=self.grid_type,
                           pixel=self.pixel, source_phot_ops=self.source_phot_ops, do_xshift=self.do_xshift,
                           bg_gal_flux=bg_gal_flux, do_rotation=self.do_rotation, airy=self.airy,
                           mismatch_seds=self.mismatch_seds, deltafcn_profile=self.deltafcn_profile,
                           noise=self.noise,
                           avoid_non_linearity=self.avoid_non_linearity,
                           spacing=self.spacing, percentiles=self.percentiles, sim_galaxy_scale=sim_galaxy_scale,
                           sim_galaxy_offset=sim_galaxy_offset, base_pointing=self.base_pointing,
                           base_sca=self.base_sca, save_model=self.save_model, prebuilt_psf_matrix=prebuilt_psf_matrix,
                           prebuilt_sn_matrix=prebuilt_sn_matrix)

        return lightcurve_model

    def build_and_save_lightcurve(self, diaobj, lc_model, param_grid_row):

        lc_model.image_collection_prov = self.img_coll_prov if self.use_real_images else None
        if self.psfclass == "ou24PSF" or self.psfclass == "ou24PSF_slow":
            psftype = "romanpsf"
        else:
            psftype = self.psfclass.lower()

        if self.use_real_images:
            identifier = str(diaobj.name)
            if lc_model.flux is not None:
                lc = build_lightcurve(diaobj, lc_model)
                if self.object_collection != "manual":
                    lc = add_truth_to_lc(lc, self.sn_truth_dir, self.object_type)

        else:
            sim_galaxy_scale, bg_gal_flux, sim_galaxy_offset = param_grid_row
            if self.run_name is None:
                identifier = "simulated_" + str(sim_galaxy_scale) + "_" + \
                    str(np.round(np.log10(bg_gal_flux), 2)) + "_" + str(sim_galaxy_offset) + "_" \
                    + self.grid_type
            else:
                identifier = self.run_name + "_" + str(diaobj.name)
            if lc_model.flux is not None:
                lc = build_lightcurve_sim(lc_model.sim_lc, lc_model.flux, lc_model.sigma_flux)
                lc["filter"] = self.band

        if lc_model.flux is not None:
            output_dir = pathlib.Path(self.cfg.value("photometry.campari.paths.output_dir"))
            save_lightcurve(lc=lc, identifier=identifier, psftype=psftype, output_path=output_dir)

        # Now, save the images

        if self.save_debug:
            fileroot = f"{identifier}_{self.band}_{psftype}"
            images_and_model = np.array(
                [lc_model.images, lc_model.model_images, lc_model.wgt_matrix, lc_model.galaxy_only_model_images]
            )
            debug_dir = pathlib.Path(self.cfg.value("photometry.campari.paths.debug_dir"))
            SNLogger.info(f"Saving images to {debug_dir / f'{fileroot}_images.npy'}")
            np.save(debug_dir / f"{fileroot}_images.npy", images_and_model)

            # Save the ra and dec grids
            ra_grid = np.atleast_1d(lc_model.ra_grid)
            dec_grid = np.atleast_1d(lc_model.dec_grid)
            SNLogger.info(f"Saving Ra/Dec grid to {debug_dir}")
            np.save(debug_dir / f"{fileroot}_grid.npy", [ra_grid, dec_grid,
                    lc_model.best_fit_model_values[: np.size(ra_grid)]])

            # save wcses
            primary_hdu = fits.PrimaryHDU()
            hdul = [primary_hdu]
            SNLogger.info(f"Saving Image WCS headers to {debug_dir}")
            if lc_model.cutout_image_list is not None:
                for i, img in enumerate(lc_model.cutout_image_list):
                    hdul.append(fits.ImageHDU(header=img.get_wcs().to_fits_header(), name="WCS" + str(i)))
                hdul = fits.HDUList(hdul)
                filepath = debug_dir / f"{fileroot}_wcs.fits"
                hdul.writeto(filepath, overwrite=True)

            if not self.use_real_images:
                np.save(debug_dir / f"{fileroot}_galaxy_images.npy", lc_model.galaxy_images)
                np.save(debug_dir / f"{fileroot}_noise_maps.npy", lc_model.noise_maps)
                SNLogger.debug(f"Saved galaxy and noise images to {debug_dir}")
        else:
            SNLogger.info("Not saving debug files.")
