import pathlib

import numpy as np

# Astronomy
from astropy.io import fits
import galsim


# SN-PIT
from snappl.dbclient import SNPITDBClient
from snappl.diaobject import DiaObject
from snappl.imagecollection import ImageCollection
from snappl.sed import Flat_SED, OU2024_Truth_SED
from snappl.config import Config
from snappl.logger import SNLogger
from snappl.provenance import Provenance

# Campari
import campari
from campari.access_truth import add_truth_to_lc
from campari.data_construction import find_all_exposures
from campari.io import (
    build_lightcurve,
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
        self.diaobject_name = kwargs["diaobject_name"]
        self.diaobject_id = kwargs["diaobject_id"]
        self.img_list = kwargs["img_list"]
        self.image_collection = kwargs["image_collection"]

        self.diaobject_collection = kwargs["diaobject_collection"]
        self.transient_start = kwargs["transient_start"]
        self.transient_end = kwargs["transient_end"]
        self.image_collection = kwargs["image_collection"]
        self.image_collection_basepath = kwargs["image_collection_basepath"]
        self.image_collection_subset = kwargs["image_collection_subset"]

        self.ra = kwargs["ra"]
        self.dec = kwargs["dec"]
        self.save_model = kwargs["save_model"]
        self.prebuilt_static_model = kwargs["prebuilt_static_model"]
        self.prebuilt_transient_model = kwargs["prebuilt_transient_model"]

        self.diaobject_provenance_tag = kwargs["diaobject_provenance_tag"]
        self.diaobject_process = kwargs["diaobject_process"]
        self.image_provenance_tag = kwargs["image_provenance_tag"]
        self.image_process = kwargs["image_process"]
        self.diaobject_position_provenance_tag = kwargs["diaobject_position_provenance_tag"]
        self.diaobject_position_process = kwargs["diaobject_position_process"]

        self.ltcv_provenance_tag = kwargs["ltcv_provenance_tag"]
        self.ltcv_process = kwargs["ltcv_process"]
        self.create_ltcv_provenance = kwargs["create_ltcv_provenance"]

        self.save_to_db = kwargs["save_to_db"]
        self.add_truth_to_lc = kwargs["add_truth_to_lc"]
        self.nprocs = kwargs["nprocs"]

        self.size = self.cfg.value("photometry.campari.cutout_size")
        self.avoid_non_linearity = self.cfg.value("photometry.campari_simulations.avoid_non_linearity")
        self.deltafcn_profile = self.cfg.value("photometry.campari_simulations.deltafcn_profile")
        self.do_xshift = self.cfg.value("photometry.campari_simulations.do_xshift")
        self.do_rotation = self.cfg.value("photometry.campari_simulations.do_rotation")
        self.noise = self.cfg.value("photometry.campari_simulations.noise")
        self.method = self.cfg.value("photometry.campari.method")
        self.make_initial_guess = self.cfg.value("photometry.campari.make_initial_guess")
        self.subtract_background_method = self.cfg.value("photometry.campari.subtract_background_method")
        self.weighting = self.cfg.value("photometry.campari.weighting")
        self.pixel = self.cfg.value("photometry.campari.pixel")
        self.sn_truth_dir = self.cfg.value("system.ou24.sn_truth_dir")
        self.mismatch_seds = self.cfg.value("photometry.campari_simulations.mismatch_seds")
        self.fetch_SED = self.cfg.value("photometry.campari.fetch_SED")
        self.initial_flux_guess = self.cfg.value("photometry.campari.initial_flux_guess")
        self.spacing = self.cfg.value("photometry.campari.grid_options.spacing")
        self.subsize = self.cfg.value("photometry.campari.grid_options.subsize")
        self.percentiles = self.cfg.value("photometry.campari.grid_options.percentiles")
        self.grid_type = self.cfg.value("photometry.campari.grid_options.type")
        self.run_name = self.cfg.value("photometry.campari_simulations.run_name")
        self.save_debug = self.cfg.value("photometry.campari_io.save_debug")
        self.transient_psfclass = self.cfg.value("photometry.campari.psf.transient_class")
        self.galaxy_psfclass = self.cfg.value("photometry.campari.psf.galaxy_class")
        try:
            self.testrun = self.cfg.value("photometry.campari.testrun")
        except Exception:
            pass
        self.param_grid = None
        self.noise_maps = None
        self.galaxy_images = None
        self.galaxy_only_model_images = None
        self.gaussian_var = self.cfg.value("photometry.campari.grid_options.gaussian_var")
        if self.gaussian_var <= 0:
            self.gaussian_var = None
        self.cutoff = self.cfg.value("photometry.campari.grid_options.cutoff")
        self.error_floor = self.cfg.value("photometry.campari.grid_options.error_floor")
        self.dbclient = SNPITDBClient()
        self.img_coll_prov = None

        if self.fast_debug:
            SNLogger.debug("Overriding config to run in fast debug mode.")
            self.grid_type = "regular"
            self.spacing = 9
            self.size = 11
            self.fetch_SED = False
            self.make_initial_guess = False

        if self.grid_type == "single" and not self.deltafcn_profile:
            SNLogger.warning("Using a single point on the grid without a delta function profile is not recommended."
                             "The goal of using a single point is to run an exact fit for testing purposes,"
                             "which requires "
                             "the galaxy be a delta function.")

        # Lightcurve provenance argument parsing logic:
        SNLogger.debug("save to db is set to " + str(kwargs["save_to_db"]))
        if kwargs["save_to_db"]:
            if not self.create_ltcv_provenance:
                if self.ltcv_provenance_tag is None and self.ltcv_process is None:
                    raise ValueError("Must provide both"
                          " ltcv_provenance_tag and ltcv_process.")

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

        banner(f"Running SN {self.diaobject_name}")

        # These will need to be re included once Issue #93 is resolved.
        # ra=self.ra, dec=self.dec
        #    mjd_discovery_min=self.transient_start, mjd_discovery_max=self.transient_end

        SNLogger.debug(f"Searching for DiaObject with id={self.diaobject_id}, name={self.diaobject_name},"
                       f" ra={self.ra}, dec={self.dec},"
                       f" collection={self.diaobject_collection}, provenance_tag={self.diaobject_provenance_tag}, "
                       f"process={self.diaobject_process}")

        arguments = {
            "collection": self.diaobject_collection,
            "dbclient": self.dbclient,
            "provenance_tag": self.diaobject_provenance_tag,
            "process": self.diaobject_process,
            "name": self.diaobject_name,
            "diaobject_id": self.diaobject_id,
            "ra": self.ra,
            "dec": self.dec,
            "mjd_discovery_min": self.transient_start,
            "mjd_discovery_max": self.transient_end}
        filtered_args = {k: v for k, v in arguments.items() if v is not None}
        # Database can't handle nones.

        diaobjs = DiaObject.find_objects(**filtered_args)

        if len(diaobjs) == 0:
            raise ValueError(
                f"Could not find DiaObject with id={self.diaobject_id}, name={self.diaobject_name},"
                f" ra={self.ra}, dec={self.dec}."
            )
        if len(diaobjs) > 1:
            raise ValueError(f"Found multiple DiaObject with id={self.diaobject_id}, name={self.diaobject_name},"
                             f" ra={self.ra}, dec={self.dec}.")
        diaobj = diaobjs[0]
        SNLogger.debug(f"Immediately after searching for objects the ID is: {diaobj.id}")

        # Get diaobject position using different methods depending on provenance.
        if self.diaobject_position_provenance_tag is None:
            diaobj.ra = diaobj.ra
            diaobj.dec = diaobj.dec
        else:
            if self.ra is not None or self.dec is not None:
                raise ValueError("Cannot provide ra or dec when also providing diaobject_position_provenance_tag."
                                 "This would lead to provenance confusion.")
            diaobj.ra, diaobj.dec = diaobj.get_position(provenance_tag=self.diaobject_position_provenance_tag,
                                                        process=self.diaobject_process, dbclient=self.dbclient)

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

        SNLogger.debug(f"Object info for SN {self.diaobject_name} with ID {self.diaobject_id} in"
                       f" collection {self.diaobject_collection}: ra={diaobj.ra},"
                       f" dec={diaobj.dec}, transient_start={diaobj.mjd_start}, transient_end={diaobj.mjd_end}")
        image_list = self.get_exposures(diaobj)
        sedlist = self.get_sedlist(diaobj.id, image_list)

        SNLogger.debug("Building Campari provenance")
        self.cam_prov = self.build_campari_provenance(image_list=image_list, diaobj=diaobj,
                                                      obj_pos_prov=self.diaobject_position_provenance_tag,
                                                      dbclient=self.dbclient)

        # This has to go after get_exposures because the infs break the simdex.
        if diaobj.mjd_start is None:
            diaobj.mjd_start = -np.inf
        if diaobj.mjd_end is None:
            diaobj.mjd_end = np.inf

        lightcurve_model = self.call_run_one_object(diaobj, image_list, sedlist)
        self.build_and_save_lightcurve(diaobj, lightcurve_model)

    def get_exposures(self, diaobj):
        """Call the find_all_exposures function to get the exposures for the given RA, Dec, and time frame."""
        if self.img_list is not None:
            # If the user provided an image list, use that.
            image_list = self.parse_img_list()
            mjd_list = [im.mjd for im in image_list]
            image_list = [im for mjd, im in sorted(zip(mjd_list, image_list))]  # Sort the images by MJD
        else:
            # Otherwise, go find images that match the criteria.
            SNLogger.debug("max no transient images: " + str(self.max_no_transient_images))
            SNLogger.debug("max transient images: " + str(self.max_transient_images))
            image_list, \
                self.img_coll_prov = find_all_exposures(diaobj=diaobj,
                                                        maxbg=self.max_no_transient_images,
                                                        maxdet=self.max_transient_images,
                                                        band=self.band,
                                                        image_selection_start=self.image_selection_start,
                                                        image_selection_end=self.image_selection_end,
                                                        image_collection=self.image_collection,
                                                        image_collection_subset=self.image_collection_subset,
                                                        image_collection_basepath=self.image_collection_basepath,
                                                        dbclient=self.dbclient,
                                                        provenance_tag=self.image_provenance_tag,
                                                        process=self.image_process)
            mjd_start = diaobj.mjd_start if diaobj.mjd_start is not None else -np.inf
            mjd_end = diaobj.mjd_end if diaobj.mjd_end is not None else np.inf

            no_transient_images = [a for a in image_list if (a.mjd < mjd_start) or (a.mjd > mjd_end)]
            SNLogger.debug(f"Found {len(no_transient_images)} non-detection images for SN {diaobj.id}.")

            if (
                self.max_no_transient_images != 0
                and len(no_transient_images) == 0
                and self.object_type != "star"
            ):
                raise ValueError("No non-detection images were found. This may be because the transient is"
                                 " detected in all images, or because the transient is outside the date range of"
                                 " available images. If you are running on stars, this is expected behavior."
                                 " If you are running on supernovae, consider increasing the date range.")

        mjd_start = diaobj.mjd_start if diaobj.mjd_start is not None else -np.inf
        mjd_end = diaobj.mjd_end if diaobj.mjd_end is not None else np.inf
        no_transient_images = [a for a in image_list if (a.mjd < mjd_start) or (a.mjd > mjd_end)]
        transient_images = [a for a in image_list if (a.mjd >= mjd_start) and (a.mjd <= mjd_end)]

        SNLogger.debug(f"Found a total of {len(image_list)} images for this object, ")
        SNLogger.debug(f"of which {len(no_transient_images)} are non-detection images")
        SNLogger.debug(f"and {len(transient_images)} are detection images.")
        self.image_list = image_list
        SNLogger.debug("setting image list")
        recovered_observation_ids = [a.observation_id for a in image_list]
        self.observation_id_list = np.array(self.observation_id_list) if getattr(self, "observation_id_list", None) \
            is not None else None
        if (self.img_list is not None and self.observation_id_list is not None) \
                and len(np.setdiff1d(self.observation_id_list, recovered_observation_ids)) > 0:
            SNLogger.warning(
                "Unable to find the object in all the observation_ids in the image list. Specifically, the"
                " following observation_ids were not found: "
                f"{np.setdiff1d(self.observation_id_list, recovered_observation_ids)}. A total of "
                f"{len(np.setdiff1d(self.observation_id_list, recovered_observation_ids))} were missing."
            )

        SNLogger.debug(f"Found {len(image_list)} exposures")
        return image_list

    def get_sedlist(self, name, image_list):
        """Create a list of SEDs for the given SNID and images."""
        try:
            sed_obj = OU2024_Truth_SED(name, isstar=(self.object_type == "star")) if self.fetch_SED else Flat_SED()
        except Exception as e:
            SNLogger.error(f"Error creating SED object: {e}. Using flat SED instead.")
            sed_obj = Flat_SED()

        sedlist = []
        for img in image_list:
            sedlist.append(sed_obj.get_sed(snid=name, mjd=img.mjd))
        return sedlist

    def call_run_one_object(self, diaobj, image_list, sedlist):
        """Call the run_one_object function to run the pipeline for a given SNID and exposures."""

        prebuilt_psf_matrix = np.load(self.prebuilt_static_model) if self.prebuilt_static_model is not None else None
        prebuilt_sn_matrix = np.load(self.prebuilt_transient_model) if self.prebuilt_transient_model is not None \
            else None

        SNLogger.debug("Save model is set to " + str(self.save_model))
        lightcurve_model = \
            run_one_object(diaobj=diaobj, object_type=self.object_type, image_list=image_list,
                           size=self.size, band=self.band,
                           fetch_SED=self.fetch_SED, sedlist=sedlist,
                           subtract_background_method=self.subtract_background_method,
                           make_initial_guess=self.make_initial_guess, initial_flux_guess=self.initial_flux_guess,
                           weighting=self.weighting, method=self.method, grid_type=self.grid_type,
                           pixel=self.pixel, do_xshift=self.do_xshift,
                           do_rotation=self.do_rotation, airy=self.airy,
                           mismatch_seds=self.mismatch_seds, deltafcn_profile=self.deltafcn_profile,
                           noise=self.noise,
                           avoid_non_linearity=self.avoid_non_linearity, subsize=self.subsize,
                           spacing=self.spacing, percentiles=self.percentiles, save_model=self.save_model,
                           prebuilt_psf_matrix=prebuilt_psf_matrix,
                           prebuilt_sn_matrix=prebuilt_sn_matrix, nprocs=self.nprocs, gaussian_var=self.gaussian_var,
                           cutoff=self.cutoff, error_floor=self.error_floor)

        return lightcurve_model

    def build_and_save_lightcurve(self, diaobj, lc_model):
        """ Build the lightcurve object and save it locally and/or to the database. Note that if no measurements
        are made, e.g. if no exposures with the transient are found, no lightcurve is saved.
        
        Inputs:
        ---------
        diaobj: DiaObject
            The DiaObject for which the lightcurve is being built.
        lc_model: LightcurveModel
            The lightcurve model returned by run_one_object.

        Returns:
        ---------
        None, but the lightcurve is saved locally and/or to the database.
        
        """
        lc_model.image_collection_prov = self.img_coll_prov
        if self.transient_psfclass == "ou24PSF" or self.transient_psfclass == "ou24PSF_slow":
            psftype = "romanpsf"
        else:
            psftype = self.transient_psfclass.lower()

        # identifier is a string that will be used to name the lightcurve file when saving debug files.
        # TODO: Come up with a better name for this.
        if self.save_to_db:
            identifier = str(diaobj.id if diaobj.id is not None else diaobj.name)
        else:
            identifier = str(diaobj.name)

        # Only save a lightcurve if there were detection images with measured fluxes:
        if lc_model.flux is not None:
            lc = build_lightcurve(diaobj, lc_model, cam_prov=self.cam_prov)
            if self.add_truth_to_lc:
                lc = add_truth_to_lc(lc, self.sn_truth_dir, self.object_type)

        if lc_model.flux is not None:
            if self.save_to_db:
                output_dir = None
            else:
                output_dir = pathlib.Path(self.cfg.value("system.paths.output_dir"))
            testrun = getattr(self, "testrun", None)
            save_lightcurve(lc=lc, identifier=identifier, psftype=psftype, output_path=output_dir,
                            save_to_database=self.save_to_db, new_provenance=self.create_ltcv_provenance,
                            testrun=testrun, dbclient=self.dbclient, ltcv_provenance_tag=self.ltcv_provenance_tag)

        # Now, save the images

        if self.save_debug:
            fileroot = f"{identifier}_{self.band}_{psftype}"

            images_and_model = np.array(
                [lc_model.images, lc_model.model_images, lc_model.wgt_matrix, lc_model.galaxy_only_model_images]
            )

            debug_dir = pathlib.Path(self.cfg.value("system.paths.debug_dir"))
            SNLogger.info(f"Saving images to {debug_dir / f'{fileroot}_images.npy'}")
            np.save(debug_dir / f"{fileroot}_images.npy", images_and_model)
            np.save(debug_dir / f"{fileroot}_noise_maps.npy", lc_model.noise_maps)

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

        else:
            SNLogger.info("Not saving debug files.")

    def parse_img_list(self):
        """Parse the image list file if provided."""
        with open(self.img_list) as ifp:
            img_list_lines = ifp.readlines()
        img_list_lines = [line.strip() for line in img_list_lines if
                          (len(line.strip()) > 0) and (line.strip()[0] != "#")]
        my_image_collection = ImageCollection()
        # De-harcode this threefile thing
        SNLogger.debug(f"Using base path {self.image_collection_basepath}")
        my_image_collection = my_image_collection.get_collection(self.image_collection,
                                                                 subset=self.image_collection_subset,
                                                                 base_path=self.image_collection_basepath)
        images = []
        if all(len(line.split(",")) == 3 for line in img_list_lines):
            self.observation_id_list = []
            # each line of file is observation_id sca band
            for line in img_list_lines:
                vals = line.split(",")
                images.append(my_image_collection.get_image(observation_id=vals[0], sca=int(vals[1]), band=vals[2]))
                self.observation_id_list.append(vals[0])
        elif all(len(line.split(",")) == 2 for line in img_list_lines):
            # each line of file is observation_id sca
            self.observation_id_list = []
            for line in img_list_lines:
                vals = line.split(",")
                images.append(my_image_collection.get_image(observation_id=vals[0], sca=int(vals[1]),
                              band=self.band))
                self.observation_id_list.append(vals[0])
        elif all(len(line.split(",")) == 1 for line in img_list_lines):
            # each line of file is path to image
            self.observation_id_list = None
            for line in img_list_lines:
                SNLogger.debug(f"Looking for path {line}.")
                images.append(my_image_collection.get_image(path=line))
        else:
            raise ValueError("Invalid img_list. Should be either paths, lines of observation_id sca band, or lines of"
                             " observation_id and sca.")

        return images

    def build_campari_provenance(self, image_list=None, diaobj=None, obj_pos_prov=None, dbclient=None):
        upstreams = []

        if image_list[0].provenance_id is not None:
            SNLogger.debug("Getting provenance for images")
            upstreams.append(Provenance.get_by_id(image_list[0].provenance_id, dbclient=dbclient))
        else:
            SNLogger.warning("Image provenance ID is None; setting imgprov to None. This should only happen in tests.")

        if diaobj.provenance_id is not None:
            SNLogger.debug("Getting provenance for diaobject")
            upstreams.append(Provenance.get_by_id(diaobj.provenance_id, dbclient=dbclient))
        else:
            SNLogger.warning(
                "Diaobject provenance ID is None; setting objprov to None. This should only happen in tests."
            )

        if obj_pos_prov is not None:
            SNLogger.debug("Getting provenance for diaobject position")
            upstreams.append(obj_pos_prov)
        else:
            SNLogger.warning("No diaobject position provenance ID provided; skipping.")

        cfg = Config.get()
        SNLogger.debug("Attempting to build provenance for lightcurve")
        campari_version = campari.__version__
        major = int(campari_version.split(".")[0])
        minor = int(campari_version.split(".")[1])
        cam_prov = Provenance(
            process=self.ltcv_process,
            major=major,
            minor=minor,
            params=cfg,
            keepkeys=["photometry.campari"],
            omitkeys=None,
            upstreams=upstreams,
        )
        return cam_prov
