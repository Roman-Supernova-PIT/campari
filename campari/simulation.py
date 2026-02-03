import pathlib
import warnings

import galsim
import numpy as np
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
from astropy.io.fits import Header as header
from erfa import ErfaWarning
from roman_imsim.utils import roman_utils

from campari.utils import campari_lightcurve_model
from snappl.imagecollection import ImageCollection
from snappl.psf import PSF
from snappl.config import Config
from snappl.logger import SNLogger

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)


def simulate_images(image_list=None, diaobj=None,
                    sim_galaxy_scale=None, sim_galaxy_offset=None, do_xshift=None,
                    do_rotation=None, noise=None, deltafcn_profile=None,
                    size=11, input_psf=None,
                    bg_gal_flux=None, sim_lc=None,
                    mismatch_seds=False, base_pointing=662, base_sca=11,
                    sim_gal_ra_offset=None, sim_gal_dec_offset=None,
                    bulge_hlr=None, disk_hlr=None):
    """This function simulates images using galsim for testing purposes. It is not
     used in the main pipeline.
    Inputs:
    num_total_images: int, the number of images to simulate
    num_detect_images: int, the number of images to simulate with a supernova
    ra, dec: floats, the RA and DEC of the center of the images to simulate,
        and the RA and DEC of the supernova.
    sim_gal_ra_offset, sim_gal_dec_offset: floats, the offsets to apply to the
        RA and DEC of the galaxy in the images.
    do_xshift:, bool whether to shift the images in the x direction (they will
        still be centered on the same point, this is just to emulate Roman taking a
        series of images at different locations.)
    do_rotation: bool, whether to rotate the images
    noise: float, the noise level to add to the images.
    band: str, the band to use for the images.
    deltafcn_profile: bool, whether to use a delta function profile for the galaxy.
    size: int, the size of the images to simulate.
    input_psf: galsim.ChromaticOpticalPSF, the PSF to use if not using Roman.
    bg_gal_flux: float, the flux of the background galaxy to simulate.
    source_phot_ops: bool, whether to use photon shooting for the supernova.

    sim_lc: list, the light curve of the supernova to simulate. If None,
        a default light curve will be generated.

    mismatch_seds: bool, whether to use a mismatched SED for the supernova, testing purposes only.
    base_pointing: int, the base pointing to use to simulate the WCS.
    base_sca: int, the base SCA to use to simulate the WCS.
    bulge_hlr: float, the half-light radius of the bulge in arcseconds.
    disk_hlr: float, the half-light radius of the disk in arcseconds.


    Returns:
    sim_lc: list, the light curve of the supernova.
    util_ref: roman_utils object, used to get the PSF.
    image_list: list of FITSImageStdHeaders objects, the full images.
    cutout_image_list: list of FITSImageStdHeaders objects, the cutout images.
    galra: float, the RA of the galaxy.
    galdec: float, the DEC of the galaxy.
    """
    source_phot_ops = Config.get().value("photometry.campari.psf.transient_photon_ops")
    ra = diaobj.ra
    dec = diaobj.dec
    band = image_list[0].band

    galaxy_psfclass = Config.get().value("photometry.campari.psf.galaxy_class")
    if galaxy_psfclass not in ["ou24PSF", "ou24PSF_slow"]:
        raise ValueError("Currently, only the ou24PSF and ou24PSF_slow are supported for simulation galaxy PSFs.")
    transient_psfclass = Config.get().value("photometry.campari.psf.transient_class")

    if sim_gal_ra_offset is not None and sim_gal_dec_offset is not None:
        galra = ra + sim_gal_ra_offset
        galdec = dec + sim_gal_dec_offset
    elif sim_galaxy_offset is not None:
        galra = ra + sim_galaxy_offset / np.sqrt(2)
        galdec = dec + sim_galaxy_offset / np.sqrt(2)
    else:
        raise ValueError("You must provide either sim_gal_ra_offset and sim_gal_dec_offset,"
                         "or sim_galaxy_offset to simulate a galaxy offset.")

    num_detect_images = len([a for a in image_list if (a.mjd > diaobj.mjd_start and a.mjd < diaobj.mjd_end)])
    SNLogger.debug(f"num_detect_images: {num_detect_images}")
    num_total_images = len(image_list)

    if sim_lc is None:
        # Here, if the user has not provided a light curve that they want
        # simulated, we generate a default one.
        if num_detect_images == 0:
            sim_lc = 0
        else:
            d = np.linspace(5, 20, num_detect_images)
            mags = -5 * np.exp(-d/10) + 6
            fluxes = 10**(mags)
            sim_lc = list(fluxes)

    snra = ra
    sndec = dec
    roman_bandpasses = galsim.roman.getBandpasses()
    sn_storage = []
    cutout_image_list = []
    noise_maps = []
    galaxy_images = []

    SNLogger.debug(f"Using base pointing {base_pointing} and SCA {base_sca}")
    file_path = pathlib.Path(Config.get().value("system.ou24.config_file"))
    util_ref = roman_utils(config_file=file_path, visit=base_pointing, sca=base_sca)
    SNLogger.debug(f"image list {image_list}")
    for i, image_object in enumerate(image_list):
        SNLogger.debug(f"Simulating image {i+1} of {num_total_images}. -----------------------------")

        if do_xshift:
            x_shift = 1e-5/3 * i
            y_shift = 0
        else:
            x_shift = 0
            y_shift = 0

        if do_rotation:
            rotation_angle = np.pi/10 * i
        else:
            rotation_angle = 0

        wcs_dict = simulate_wcs(angle=rotation_angle, x_shift=x_shift, y_shift=y_shift,
                                base_sca=base_sca, base_pointing=base_pointing, band=band)
        input_header = header(wcs_dict)
        # Adding in other necessary header keywords.
        input_header["MJD"] = image_object.mjd
        input_header["BAND"] = image_object.band
        input_header["POINTING"] = image_object.pointing
        input_header["SCA"] = image_object.sca
        image_object.data = np.zeros((4088, 4088))
        image_object.noise = np.ones((4088, 4088))
        image_object.flags = np.zeros((4088, 4088)).astype(np.uint8)

        image_object.set_fits_header(input_header)

        full_image_wcs = image_object.get_wcs()

        cutout_object = image_object.get_ra_dec_cutout(ra, dec, xsize=size)
        cutoutgalwcs = cutout_object.get_wcs().get_galsim_wcs()  # rename this
        cutout_loc = full_image_wcs.world_to_pixel(ra, dec)
        cutout_pixel = (int(np.floor(cutout_loc[0] + 0.5)), int(np.floor(cutout_loc[1] + 0.5)))

        image_object.data = None
        image_object.noise = None
        image_object.flags = None # These are meaningless, delete to save memory.

        if mismatch_seds:
            SNLogger.debug("INTENTIONALLY MISMATCHING SEDS, 1a SED")
            file_path = r"snflux_1a.dat"
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["Day",
                             "Wavelength", "Flux"])
            a = df.loc[df.Day == 0]
            del df
            sed = galsim.SED(galsim.LookupTable(a.Wavelength/10, a.Flux,
                             interpolant="linear"), wave_type="nm",
                             flux_type="fphotons")

        else:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1, 1],
                             interpolant="linear"), wave_type="nm",
                             flux_type="fphotons")

        pointx, pointy = cutoutgalwcs.toImage(galra, galdec, units="deg")

        if transient_psfclass in ["ou24PSF", "ou24PSF_slow"]:
            sim_psf = util_ref.getPSF(cutout_pixel[0] + 1, cutout_pixel[1] + 1, pupil_bin=8)
        else:
            sim_psf = input_psf

        # Draw the galaxy.
        if bg_gal_flux > 0:
            convolved = simulate_galaxy(bg_gal_flux=bg_gal_flux, sim_galaxy_scale=sim_galaxy_scale,
                                        deltafcn_profile=deltafcn_profile, band=band,
                                        sim_psf=sim_psf, sed=sed, bulge_hlr=bulge_hlr,
                                        disk_hlr=disk_hlr)

            SNLogger.debug(f"Galaxy being drawn at {pointx, pointy} ")
            localwcs = full_image_wcs.get_galsim_wcs().\
                local(image_pos=galsim.PositionD(cutout_pixel[0] + 1, cutout_pixel[1] + 1))
            stamp = galsim.Image(size, size, wcs=localwcs)
            a = convolved.drawImage(roman_bandpasses[band], method="no_pixel",
                                    image=stamp, wcs=localwcs,
                                    center=galsim.PositionD(pointx, pointy),
                                    use_true_center=True).array
            galaxy_images.append(np.copy(a))
        else:
            a = np.zeros((size, size))
            galaxy_images.append(a)

        # Noise it up!
        if noise > 0:
            rng = np.random.default_rng()
            noise_map = rng.normal(0, noise, size**2).reshape(size, size)
            a += noise_map
            noise_maps.append(noise_map)
        else:
            noise_maps.append(np.zeros((size, size)))

        # Inject a supernova! If using.
        if sim_lc != 0:
            # Here we want to count which supernova image we are on. The
            # following is zero on the first sn image and counts up:
            sn_im_index = i - num_total_images + num_detect_images
            SNLogger.debug(f"On image {i+1} of {num_total_images}, sn_im_index is {sn_im_index}")
            if sn_im_index >= 0:
                snx, sny = cutoutgalwcs.toImage(snra, sndec, units="deg")
                stamp = galsim.Image(size, size, wcs=cutoutgalwcs)
                SNLogger.debug(f"sed: {sed}")

                # When using the ou24PSF, we want to use the slower and more accurate ou24PSF_slow for SN.
                supernova_image = simulate_supernova(
                    snx=cutout_loc[0],
                    sny=cutout_loc[1],
                    snx0=cutout_pixel[0],
                    sny0=cutout_pixel[1],
                    flux=sim_lc[sn_im_index],
                    sed=sed,
                    source_phot_ops=source_phot_ops,
                    base_pointing=base_pointing,
                    base_sca=base_sca,
                    stampsize=size,
                    image=image_object,
                    psfclass=transient_psfclass)

                a += supernova_image
                sn_storage.append(supernova_image)

        cutout_object.data = a
        # TODO: Decide how error is handled for simulated images.
        if noise > 0:
            cutout_object.noise = np.ones_like(a) * noise
        else:
            cutout_object.noise = np.ones_like(a)
        cutout_object.mjd = image_object.mjd  # Temp fix, cutouts should inherit mjd from full image in snappl.
        cutout_object.band = image_object.band  # Temp fix, cutouts should inherit band from full image in snappl.
        cutout_image_list.append(cutout_object)

    lightcurve = campari_lightcurve_model(
        sim_lc=sim_lc,
        image_list=image_list,
        cutout_image_list=cutout_image_list,
        galaxy_images=np.array(galaxy_images),
        noise_maps=np.array(noise_maps),
        galra=galra,
        galdec=galdec
    )

    return lightcurve, util_ref


def simulate_wcs(angle=None, x_shift=None, y_shift=None, base_sca=None, base_pointing=None, band=None,
                 sim_basis="ou2024"):
    """ This function simulates the WCS for a Roman image given a base pointing / SCA combination to start from,
    then applying a rotation and shifts to the WCS.

    Inputs:
    angle: float, the angle to rotate the WCS by in radians.
    x_shift, y_shift: floats, the shifts, in degrees, to apply to the WCS in the x and y directions.
    base_sca: int, the base SCA to use to simulate the WCS.
    base_pointing: int, the base pointing to use to simulate the WCS.
    band: str, the band to use for the images.
    sim_basis: str, the simulation to use to base WCS simulations on, defaults to "ou2024".

    Returns:
    wcs_dict: dict, a dictionary containing the WCS information for the image.
    """
    rotation_matrix = np.array([np.cos(angle), -np.sin(angle), np.sin(angle),
                               np.cos(angle)]).reshape(2, 2)

    img_collection = ImageCollection()
    img_collection = img_collection.get_collection(sim_basis)
    image = img_collection.get_image(pointing=base_pointing, sca=base_sca, band=band)
    header = image.get_fits_header()

    CD_matrix = np.zeros((2, 2))
    CD_matrix[0, 0] = header["CD1_1"]
    CD_matrix[0, 1] = header["CD1_2"]
    CD_matrix[1, 0] = header["CD2_1"]
    CD_matrix[1, 1] = header["CD2_2"]

    CD_matrix_rotated = CD_matrix @ rotation_matrix

    wcs_dict = {
            "CTYPE1": header["CTYPE1"],
            "CTYPE2": header["CTYPE2"],
            "CRPIX1": header["CRPIX1"],
            "CRPIX2": header["CRPIX2"],
            "CD1_1": CD_matrix_rotated[0, 0],
            "CD1_2": CD_matrix_rotated[0, 1],
            "CD2_1": CD_matrix_rotated[1, 0],
            "CD2_2": CD_matrix_rotated[1, 1],
            "CUNIT1": header["CUNIT1"],
            "CUNIT2": header["CUNIT2"],
            "CRVAL1": header["CRVAL1"] + x_shift,
            "CRVAL2":  header["CRVAL2"] + y_shift,
        }

    return wcs_dict


def simulate_galaxy(bg_gal_flux=None, sim_galaxy_scale=None, deltafcn_profile=None, band=None,
                    sim_psf=None, sed=None, bulge_hlr=None, disk_hlr=None):
    """This function simulates a galaxy using galsim. It can simulate either a delta function profile or a bulge+disk
    profile.

    Inputs:
    bg_gal_flux: float, the flux of the background galaxy to simulate.
    deltafcn_profile: bool, whether to use a delta function profile for the galaxy, if false, use a bulge+disk profile.
    band: str, the band to use for the images.
    sim_psf: galsim.ChromaticOpticalPSF, the PSF to use for the galaxy.
    sed: galsim.SED, the spectral energy distribution of the galaxy.
    bulge_hlr: float, the half-light radius of the bulge in arcseconds.
    disk_hlr: float, the half-light radius of the disk in arcseconds.

    Returns:
    convolved: galsim.chromatic.ChromaticConvolution, the convolved galaxy profile. This can then be used to draw an
    image.

    """

    SNLogger.debug(f"Simulating galaxy with band {band} and flux {bg_gal_flux}.")
    SNLogger.debug(f"Using sim_galaxy_scale {sim_galaxy_scale}")
    roman_bandpasses = galsim.roman.getBandpasses()

    if not deltafcn_profile:
        if sim_galaxy_scale is not None:
            sim_galaxy_scale = float(sim_galaxy_scale)
            bulge_hlr = sim_galaxy_scale * 1.6
            disk_hlr = sim_galaxy_scale * 5.0

        if bulge_hlr is None or disk_hlr is None:
            raise ValueError("You must provide either bulge_hlr and disk_hlr, or sim_galaxy_scale"
                             " to simulate a galaxy.")
        else:
            SNLogger.debug("Using bulge+disk profile for galaxy. The bulge has a half-light radius of "
                           f"{bulge_hlr} and the disk has a half-light radius of {disk_hlr}.")
            bulge = galsim.Sersic(n=3, half_light_radius=bulge_hlr)
            disk = galsim.Exponential(half_light_radius=disk_hlr)
            profile = bulge + disk

    elif deltafcn_profile:
        SNLogger.debug("Using delta function profile for galaxy.")
        profile = galsim.DeltaFunction()

    else:
        raise ValueError("You must provide either bulge_hlr and disk_hlr, sim_galaxy_scale, or deltafcn_profile"
                         " to simulate a galaxy.")

    profile *= sed
    profile = profile.withFlux(bg_gal_flux, roman_bandpasses[band])
    convolved = galsim.Convolve(profile, sim_psf)
    return convolved


def simulate_supernova(snx=None, sny=None, snx0=None, sny0=None, flux=None, sed=None,
                       source_phot_ops=None, base_pointing=None, base_sca=None, stampsize=None,
                       random_seed=0, image=None, psfclass="ou24PSF_slow"):
    """This function simulates a supernova using the ou24PSF_slow PSF.

    Inputs:
    snx, sny: floats, the x and y coordinates of the supernova in the image.\
    snx0, sny0: ints, the x and y coordinates of pixel the image was cutout on.
    flux: float, the flux of the supernova.
    sed: galsim.SED, the spectral energy distribution of the supernova.
    source_phot_ops: bool, whether to use photon shooting for the supernova.
    base_pointing: int, the base pointing to use to simulate the WCS.
    base_sca: int, the base SCA to use to simulate the WCS.
    stampsize: int, the size of the stamp to draw the supernova on.
    random_seed: int, the seed to use for the random number generator for the photon shooting.
    sca_wcs: snappl.BaseWCS, the WCS of the entire SCA image.

    Returns:
    psf_image: numpy.ndarray, the image of the supernova convolved with the PSF.

    """

    SNLogger.debug(f"Simulating supernova at ({snx}, {sny}) with flux {flux} ")
    SNLogger.debug(f"Using base pointing {base_pointing} and SCA {base_sca}.")
    SNLogger.debug(f"source_phot_ops: {source_phot_ops} and sed {sed}")
    SNLogger.debug(f"Using SN psfclass {psfclass}")

    psf_object = PSF.get_psf_object(psfclass, pointing=base_pointing, sca=base_sca,
                                    size=stampsize, include_photonOps=source_phot_ops, sed=sed, seed=None, image=image, stamp_size=stampsize)
    psf_image = psf_object.get_stamp(x0=snx0, y0=sny0, x=snx, y=sny, flux=flux)
    return psf_image
