# Standard Library
import multiprocessing
import pathlib
import warnings

# Common Library
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Astronomy Library
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SN-PIT
from snappl.config import Config
from snappl.logger import SNLogger
from snappl.sed import Flat_SED

# Campari
from campari.data_construction import is_number
from campari.model_building import construct_transient_scene
from campari.utils import calculate_background_level, get_weights
from campari.RomanASP import _parse_args_and_instantiate_runner

from pathlib import Path
from astropy.table import Table, vstack, hstack, join
import glob

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter("ignore", category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

r"""
Zeropoint calibration utilities for campari.

Unlike the main campari pipeline (RomanASP.py / campari_runner.py), which
places a single transient PSF (plus background grid points) per image to
measure a supernova, this module places a PSF at the location of EVERY star
in a supplied catalog, on a single image, in order to build up the flux
measurements needed to solve for a photometric zeropoint.

This module is NOT called anywhere in the main pipeline. It's meant to be
run separately (e.g. as a standalone script, or from a notebook) either
before or after a lightcurve run, on whatever set of calibration images and
star catalog you have on hand.

Key design point: each image's pixel data is only read off disk ONCE, no
matter how many stars are measured on it. See calculate_star_fluxes_for_image
for how this works.
"""


def _get_image_background(image, subtract_background_method):
    """Compute a single background level for an entire image (not per star!).

    This mirrors the background-subtraction logic in
    campari.data_construction.construct_one_image, but is written to be
    called exactly ONCE per image rather than once per cutout, since (unlike
    an individual star's PSF) the background level does not depend on any
    particular star's position.

    NOTE: this duplicates some logic that also lives in
    campari.data_construction.construct_one_image. That function couldn't be
    reused directly here because it calls image.free() at the end, which
    would throw away the cached pixel data we rely on to avoid rereading the
    image from disk for every star. If this duplication starts to be a
    maintenance headache, it would be worth factoring the background-level
    logic out into one shared helper that both functions call.

    Parameters
    ----------
    image : snappl.image.Image
        The full SCA image. image.get_data(cache=True) must have already
        been called on it (so that imagedata below is already in memory).
    subtract_background_method : str
        "calculate" (auto-estimate via photutils), a numeric string/float
        (use that constant), or a FITS header keyword name to look up.
        "fit" is intentionally not supported -- see the docstring of
        calculate_star_fluxes_for_image for why.

    Returns
    -------
    bg : float
        The background level to subtract from this image, in the same
        units as the image data.
    """
    if is_number(subtract_background_method):
        bg = float(subtract_background_method)
        SNLogger.debug(f"Using constant background level from user input: {bg}")
        return bg

    if subtract_background_method == "calculate":
        imagedata, _errordata, _flags = image.get_data(which="all", cache=True)
        bg = calculate_background_level(imagedata)
        SNLogger.debug(f"Calculated background level for whole image: {bg}")
        return bg

    if subtract_background_method == "fit":
        raise ValueError(
            "subtract_background_method='fit' is not supported by "
            "calculate_star_fluxes_for_image. In the main campari pipeline, "
            "'fit' means the background is solved for jointly with the "
            "fluxes in one big linear fit. Here, each star gets its own "
            "flux-only fit (see the module docstring for why), so there is "
            "no joint fit for a background level to be part of. Use "
            "'calculate' or a numeric constant instead."
        )

    header = image.get_fits_header()
    if subtract_background_method not in header:
        raise ValueError(
            f"Could not find background level in header with keyword "
            f"'{subtract_background_method}' for this image."
        )
    bg = header[subtract_background_method]
    SNLogger.debug(f"Using background level {bg} from header keyword '{subtract_background_method}'.")
    return bg


# Used together with multiprocessing's "fork" start method to share one
# image's already-loaded pixel data with worker processes without having to
# pickle (serialize) the image object itself. See the "Parallelizing across
# stars" section of calculate_star_fluxes_for_image's docstring for a
# plain-language explanation of why this is necessary and how it works.
_calc_star_flux_shared_image = None


def _fit_one_star_flux(image, star_id, ra, dec, sed, size, bg, whole_sca_wcs,
                       gaussian_var, cutoff, error_floor):
    """Fit a single star's flux on `image`. This is the inner-loop body of
    calculate_star_fluxes_for_image, pulled out into its own function so it
    can be called either directly (nprocs=1) or from a worker process
    (nprocs>1) without duplicating the fitting logic."""

    image_cutout = image.get_ra_dec_cutout(ra, dec, size, mode="partial", fill_value=np.nan)
    image_cutout.data = image_cutout.data - bg

    object_x, object_y = whole_sca_wcs.world_to_pixel(ra, dec)
    # See snappl.psf.PSF.get_stamp docs (also referenced in
    # campari.model_building) for why we round like this: pixel centers
    # sit at integers, so anything within +/-0.5 of an integer belongs
    # to that pixel.
    x0 = int(np.floor(object_x + 0.5))
    y0 = int(np.floor(object_y + 0.5))

    psf_stamp = construct_transient_scene(
        x0=x0, y0=y0,
        observation_id=image.observation_id, sca=image.sca,
        stampsize=size, x=object_x, y=object_y,
        sed=sed, image=image, flux=1.0,
    )

    wgt = get_weights(
        [image_cutout], ra, dec,
        gaussian_var=gaussian_var, cutoff=cutoff, error_floor=error_floor,
    )[0]

    data = image_cutout.data.flatten()

    # Zero out any masked/off-edge pixels, the same way
    # campari.data_construction.prep_data_for_fit does for the main fit.
    bad = np.isnan(data)
    data = np.where(bad, 0.0, data)
    wgt = np.where(bad, 0.0, wgt)

    # The main pipeline (run_one_object) solves a big linear system because
    # it has many free parameters (background grid points + transient) at
    # once, via scipy's lsqr solver. Here, each star has exactly one free
    # parameter (flux) so the weighted least-squares solution
    # has a plain formula:
    #     flux     = sum(w * data * psf) / sum(w * psf^2)
    #     flux_err = sqrt( 1 / sum(w * psf^2) )
    denom = np.sum(wgt * psf_stamp ** 2)
    if denom <= 0:
        SNLogger.warning(f"Star {star_id}: no usable weight/pixels, skipping flux fit.")
        flux, flux_err = np.nan, np.nan
    else:
        flux = np.sum(wgt * data * psf_stamp) / denom
        flux_err = np.sqrt(1.0 / denom)

    return {
        "id": star_id, "ra": ra, "dec": dec, "x_sca": object_x, "y_sca": object_y,
        "flux": flux, "flux_err": flux_err, "n_valid_pixels": int(np.sum(~bad)),
    }


def _fit_one_star_flux_worker(star_id, ra, dec, sed, kwarg_dict):
    """Runs inside a worker process. Pulls the (already-loaded) image out of
    the module-level global set by calculate_star_fluxes_for_image, rather
    than receiving it as a normal argument, since campari image objects
    generally can't be pickled and sent to a worker process directly."""
    return _fit_one_star_flux(image=_calc_star_flux_shared_image,
                              star_id=star_id, ra=ra, dec=dec, sed=sed, **kwarg_dict)


def calculate_star_fluxes_for_image(
    image,
    star_catalog,
    size=11,
    sed=None,
    sed_list=None,
    gaussian_var=None,
    cutoff=4,
    error_floor=1.0,
    subtract_background_method="calculate",
    ra_colname="ra",
    dec_colname="dec",
    id_colname="id",
    nprocs=4,
):
    """Fit a PSF flux to every star in a catalog, on a single image.

    This is the core building block for zeropoint calibration: given a
    catalog of stars with known sky positions, this places a copy of
    campari's PSF model (whatever PSF class is currently configured under
    photometry.campari.psf.transient_class) at each star's location, and
    solves for that star's flux.

    IMPORTANT -- how this avoids rereading the image from disk per star:
    `image.get_data(..., cache=True)` is called exactly ONCE, before the
    star loop starts. The `cache=True` flag tells the image object "keep
    the pixel array you just loaded in memory." Every star's cutout, made
    inside the loop with `image.get_ra_dec_cutout(...)`, is then just a
    slice out of that already-loaded array rather than a fresh disk read.
    Likewise, the background level is computed once for the whole image
    (backgrounds don't depend on where a given star sits), not once per
    star. `image.free()` is only called at the very end, after every star
    has been measured, to release that cached memory.

    Parallelizing across stars (nprocs > 1)
    ----------------------------------------
    If `nprocs` is greater than 1, the star loop is farmed out to a pool of
    worker processes with Python's `multiprocessing`, using the same
    pattern campari already relies on elsewhere (see
    campari.data_construction.construct_images and
    campari.run_one_object). Worth explaining plainly, since it's a bit
    unusual:

    Normally, to hand data to a separate worker process, Python has to
    "pickle" it -- serialize it into a stream of bytes that gets copied
    over to the worker, which then rebuilds an equivalent object on its
    end. Think of it like mailing someone a photocopy. Some campari image
    objects can't be photocopied this way (their internals don't support
    it), so instead we lean on how new processes get created on Linux and
    macOS: with the "fork" start method, a brand-new process starts out as
    an exact duplicate of the current one, including everything already
    sitting in memory at that moment -- more like handing a worker a key
    to a copy of your entire desk mid-task, papers and all, rather than
    mailing them one document. So we make sure `image` (with its pixel
    data already loaded) is sitting in a module-level variable *before*
    the worker processes are created, and each worker just reads that
    variable out of its own inherited copy of memory -- the image itself
    is never pickled. Only the small, genuinely picklable, per-star inputs
    (ID, RA, Dec, SED) get sent to each worker directly, and the fitted
    flux gets sent back the normal (pickled) way.

    Parameters
    ----------
    image : snappl.image.Image
        The full, un-cut SCA image object to measure stars on (i.e. the
        entire detector image, not a cutout).
    star_catalog : astropy.table.Table or pandas.DataFrame
        One row per star. Must contain RA, Dec, and ID columns; see
        `ra_colname`, `dec_colname`, `id_colname` .
    size : int
        Width/height, in pixels, of the square cutout used to fit each
        star. Should generally match (or be close to) the cutout size used
        for the science object, since the PSF model can depend on stamp
        size.
    sed : galsim.SED, optional
        A single SED to use for every star. Ignored if `sed_list` is given.
    sed_list : list of galsim.SED, optional
        A per-star SED, in the same order as `star_catalog`. Use this if
        you have real spectral information for your calibration stars
        (e.g. from stellar type or catalog colors). If neither `sed` nor
        `sed_list` is given, a flat SED is used for every star (the same
        fallback campari uses elsewhere when no SED information is
        available).
    gaussian_var, cutoff, error_floor : float
        Passed straight through to campari.utils.get_weights to build the
        per-pixel weights around each star. See that function's docstring
        for details.
    subtract_background_method : str
        "calculate" (auto-estimate the background for the whole image via
        photutils, once), a numeric string/float (subtract that constant),
        or a FITS header keyword name. "fit" is not supported here -- see
        _get_image_background for why.
    ra_colname, dec_colname, id_colname : str
        Column names to look for in `star_catalog`.
    nprocs : int
        Number of worker processes to fit stars with. Default 1 (no
        parallelism; everything runs in the calling process). See
        "Parallelizing across stars" above.

    Returns
    -------
    astropy.table.Table
        One row per star, with columns: id, ra, dec, x_sca, y_sca, flux,
        flux_err, n_valid_pixels. `x_sca`/`y_sca` are the star's pixel
        location on the full SCA (not the cutout), which is often handy
        for later diagnostics (e.g. flux vs. detector position).
    """
    if not isinstance(star_catalog, Table):
        star_catalog = Table.from_pandas(star_catalog)

    if sed_list is not None and len(sed_list) != len(star_catalog):
        raise ValueError(
            f"sed_list has {len(sed_list)} entries but star_catalog has "
            f"{len(star_catalog)} rows; they must be the same length and "
            "in the same order."
        )

    SNLogger.debug(
        f"Loading full image data once, for {len(star_catalog)} stars, on image "
        f"observation_id={getattr(image, 'observation_id', None)}, sca={getattr(image, 'sca', None)}"
    )
    # This is the one-and-only disk read for this whole function.
    image.get_data(which="all", cache=True)

    # Background is computed once per image, not once per star.
    bg = _get_image_background(image, subtract_background_method)

    # A single flat SED, computed once, to reuse for every star unless the
    # user gave us something more specific.
    default_sed = None
    if sed is None and sed_list is None:
        default_sed = Flat_SED().get_sed(snid=None, mjd=image.mjd)

    whole_sca_wcs = image.get_wcs()

    # Gather the (small, easily-picklable) per-star inputs up front, whether
    # or not we end up running in parallel.
    star_ids, ras, decs, seds = [], [], [], []

    # Cut the star catalog with ra, dec, and mag cuts

    ra_corners = [image.ra_corner_00, image.ra_corner_01, image.ra_corner_10, image.ra_corner_11]
    dec_corners = [image.dec_corner_00, image.dec_corner_01, image.dec_corner_10, image.dec_corner_11]
    min_ra = min(ra_corners)
    max_ra = max(ra_corners)
    min_dec = min(dec_corners)
    max_dec = max(dec_corners)

    SNLogger.debug(f"Before ra / dec cuts star catalog has {len(star_catalog)} stars")
    star_catalog = star_catalog[star_catalog[ra_colname] >= min_ra]
    star_catalog = star_catalog[star_catalog[ra_colname] <= max_ra]
    star_catalog = star_catalog[star_catalog[dec_colname] >= min_dec]
    star_catalog = star_catalog[star_catalog[dec_colname] <= max_dec]
    SNLogger.debug(f"After ra / dec cuts star catalog has {len(star_catalog)} stars")

    # Cut the star catalog with magnitude cuts
    SNLogger.warning(f"REMOVE HARDCODED BAND")
    star_catalog_flux = star_catalog["F129"]
    star_catalog_mag = -2.5 * np.log10(star_catalog_flux)
    star_catalog = star_catalog[star_catalog_mag >= 18]
    star_catalog_flux = star_catalog["F129"]
    star_catalog_mag = -2.5 * np.log10(star_catalog_flux)
    star_catalog = star_catalog[star_catalog_mag <= 22]
    SNLogger.debug(f"After mag cuts star catalog has {len(star_catalog)} stars")

    for i, row in enumerate(star_catalog):
        star_ids.append(row[id_colname])
        ras.append(float(row[ra_colname]))
        decs.append(float(row[dec_colname]))
        if sed_list is not None:
            seds.append(sed_list[i])
        elif sed is not None:
            seds.append(sed)
        else:
            seds.append(default_sed)

    # Everything each star's fit needs, other than the star's own ID/ra/dec/sed.
    kwarg_dict = {
        "size": size, "bg": bg, "whole_sca_wcs": whole_sca_wcs,
        "gaussian_var": gaussian_var, "cutoff": cutoff, "error_floor": error_floor,
    }

    if nprocs > 1:
        SNLogger.debug(f"Using {nprocs} processes to fit {len(star_catalog)} stars")
        # See the "Parallelizing across stars" section of this function's
        # docstring for why `image` goes into a global before the workers
        # are created, rather than being passed in as a normal argument.
        global _calc_star_flux_shared_image
        _calc_star_flux_shared_image = image
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(nprocs) as pool:
            async_results = [
                pool.apply_async(
                    _fit_one_star_flux_worker,
                    args=(star_ids[i], ras[i], decs[i], seds[i], kwarg_dict),
                )
                for i in range(len(star_catalog))
            ]
            pool.close()
            pool.join()
        # .get() blocks until that particular star's result is ready, but
        # since we call it in submission order, star_results comes out in
        # the same order as star_catalog regardless of which worker
        # actually finished first.
        star_results = [r.get() for r in async_results]
    else:
        star_results = [
            _fit_one_star_flux(image=image, star_id=star_ids[i], ra=ras[i], dec=decs[i],
                               sed=seds[i], **kwarg_dict)
            for i in range(len(star_catalog))
        ]

    # Now that every star has been measured, release the cached pixel data.
    image.free()

    results = {
        key: [r[key] for r in star_results]
        for key in ("id", "ra", "dec", "x_sca", "y_sca", "flux", "flux_err", "n_valid_pixels")
    }

    return Table(results)


def calculate_zeropoint_catalog(image_list, star_catalog, output_dir=None, **kwargs):
    """Measure star fluxes across a list of images, and save one ECSV per image.

    This is the "image-outer, star-inner" loop: for each image in
    `image_list`, calculate_star_fluxes_for_image is called once, which in
    turn reads that image's pixel data from disk exactly once and loops
    over every star in `star_catalog` in memory.

    Parameters
    ----------
    image_list : list of snappl.image.Image
        The calibration images to measure stars on.
    star_catalog : astropy.table.Table or pandas.DataFrame
        Passed straight through to calculate_star_fluxes_for_image.
    output_dir : str or pathlib.Path, optional
        Where to write the per-image ECSV files. Defaults to
        photometry.campari_io.output_dir from the current config.
    **kwargs
        Any other keyword arguments are passed straight through to
        calculate_star_fluxes_for_image (size, sed, sed_list, gaussian_var,
        cutoff, error_floor, subtract_background_method, ra_colname,
        dec_colname, id_colname, nprocs). Pass nprocs=N here to fit stars
        within each image in parallel across N worker processes -- see
        calculate_star_fluxes_for_image's docstring for how that works.

    Returns
    -------
    list of astropy.table.Table
        One table per image, in the same order as `image_list`. Each table
        is also written to disk as
        <output_dir>/zeropoint_stars_<observation_id>_<sca>.ecsv
    """
    cfg = Config.get()
    if output_dir is None:
        output_dir = cfg.value("photometry.campari_io.output_dir")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_tables = []
    for image in image_list:
        SNLogger.debug(
            f"Measuring {len(star_catalog)} stars on image "
            f"observation_id={image.observation_id}, sca={image.sca}"
        )
        star_table = calculate_star_fluxes_for_image(image, star_catalog, **kwargs)
        star_table.meta["observation_id"] = str(image.observation_id)
        star_table.meta["sca"] = image.sca
        star_table.meta["mjd"] = image.mjd
        star_table.meta["band"] = image.band

        # Change this naming hardcode
        out_path = output_dir / f"zeropoint_stars_{image.observation_id}_{image.sca}_{image.mjd}_crds.ecsv"
        star_table.write(out_path, format="ascii.ecsv", overwrite=True)
        SNLogger.info(f"Wrote {len(star_table)} star flux measurements to {out_path}")

        all_tables.append(star_table)

    return all_tables


def _load_star_cat():
    star_cat_parent = Path( "/ricktruth/" )
    gaiacat_path = star_cat_parent / "STARS_GAIA_ELAIS.csv.gz"
    syncat_path = star_cat_parent / "STARS_SYN_ELAIS.csv.gz"
    gaiacat = Table.read(gaiacat_path, format="csv")
    syncat = Table.read(syncat_path, format="csv")
    star_catalog = vstack([gaiacat, syncat])
    return star_catalog


def main():
    """Run the zeropoint calibration utility from the command line.

    This is a convenience wrapper around calculate_zeropoint_catalog that
    reads the input star catalog, uses campari's runner to find images, and writes the
    per-image ECSV files.
    """

    cfg = Config.get()
    output_dir = cfg.value("photometry.campari_io.output_dir")
    star_catalog = _load_star_cat()

    # Copy the campari pipeline setup but don't actually run SMP! Just get images in the exact
    # same way!
    runner = _parse_args_and_instantiate_runner()
    diaobjs = runner.find_diaobjs()
    diaobj = runner._setup_diaobj(diaobjs)
    image_list = runner.get_exposures(diaobj)

    calculate_zeropoint_catalog(image_list, star_catalog, output_dir=output_dir)
    calc_all_zeropoints_from_saved_files(f"{output_dir}/zeropoint_stars_*.ecsv")


def _load_saved_ecsv_and_determine_zpt(ecsv_file_path):
    """Load a previously-saved zeropoint star flux table, and compute the
    photometric zeropoint from it.

    Parameters
    ----------
    ecsv_file_path : str or pathlib.Path
        Path to a zeropoint ECSV file previously written by
        calculate_zeropoint_catalog.

    Returns
    -------
    zpt : float
        The photometric zeropoint for this image, computed as the median of
        2.5 * log10(flux) + mag for all stars in the table.
    """
    star_catalog = _load_star_cat()
    star_table = Table.read(ecsv_file_path, format="ascii.ecsv")

    # match star_catalog to star_table by id, and add the mag column to star_table
    SNLogger.warning("HARDCODED BAND HERE")

    star_table = join(star_table, star_catalog, keys='id', join_type='inner')

    def line(x, b):
        return x + b

    from scipy.optimize import curve_fit
    star_table["mag"] = -2.5 * np.log10(star_table["flux"])
    popt, pcov = curve_fit(line, star_table["mag"], -2.5 * np.log10(star_table["F129"]))
    print("Fitted line parameters:", popt, "+/-", np.sqrt(np.diag(pcov)))

    plt.title("Zeropoint Calibration")

    # Fit a line
    plt.scatter(star_table["mag"], -2.5 * np.log10(star_table["F129"]), label="Data")
    plt.xlabel("Measured mag")
    plt.ylabel("Catalog mag")
    plt.legend()
    plt.savefig("zeropoint_calibration.png")

    plt.close()
    zpt = np.median(star_table["mag"] + 2.5 * np.log10(star_table["F129"]))
    SNLogger.info(f"Photometric zeropoint for {ecsv_file_path}: {zpt}")

    zpt = popt[0]
    zpt_err = np.sqrt(np.diag(pcov))[0]

    return zpt, zpt_err


def calc_all_zeropoints_from_saved_files(zpt_glob):
    """Load all zeropoint ECSV files matching a glob, and compute the
    photometric zeropoint for each one.

    Parameters
    ----------
    zpt_glob : str
        A glob pattern to match zeropoint ECSV files previously written by
        calculate_zeropoint_catalog.
    """
    zpt_files = glob.glob(zpt_glob)
    for z in zpt_files:
        print(z)
        zpt, zpterr = _load_saved_ecsv_and_determine_zpt(z)
        print(f"Zeropoint for {z}: {zpt} +/- {zpterr}")

if __name__ == "__main__":
    """
    python /home/snpit/packages/campari/campari/zeropoint.py --photometry-campari-psf-transient_class STPSF \
 --photometry-campari-psf-galaxy_class gaussian --photometry-campari-use_real_images --no-photometry-campari-fetch_SED \
 --photometry-campari-grid_options-type none --photometry-campari-grid_options-spacing 0.75 \
 --photometry-campari-grid_options-subsize 4 --photometry-campari-grid_options-error_floor 0 \
 --photometry-campari-grid_options-gaussian_var 100000 --photometry-campari-grid_options-cutoff 3 \
 --photometry-campari-cutout_size 19 --photometry-campari-weighting --photometry-campari-subtract_background calculate \
 --image-collection manual_rdm --no-save-to-db --diaobject-collection manual --nprocs 4 \
 -p "/ricksims/output_images_SCAx2_ZYJHF_40day//SNP*WFI01*F129*L2.asdf" --image-collection-basepath \
 /ricksims/output_images_SCAx2_ZYJHF_40day/ --ra 9.418392 --dec -43.942912 --transient_end 60400 \
 -f F129 --diaobject-name testing_zpts
    """




    #--bind /home/rkessler/romanisim/input_catalogs/:/ricktruth:ro

    # truth_dir = "/ricktruth/snana_sim_pilot+deep/TRUTH_HL*SNANA*"
    # # search for matching ra/dec
    # truthfiles = glob.glob(truth_dir)
    # print(f"Found {len(truthfiles)} truth files: {truthfiles}")
    # truth_df  = pd.read_csv(truthfiles[0], comment="#", sep = "\s+")
    # truth_df_subset = truth_df[np.abs(truth_df["RA"] - 9.418392) < 0.01]
    # truth_df_subset = truth_df_subset[np.abs(truth_df_subset["DEC"] - -43.942912) < 0.01]
    # print(truth_df_subset)
    # print("RA DEC MATCHES ^")


    # truth_dir = "/ricktruth/snana_sim_pilot+deep/TRUTH_HL*LCPLOT*"
    # truthfiles = glob.glob(truth_dir)
    # print(f"Found {len(truthfiles)} truth files: {truthfiles}")
    # truth_df  = pd.read_csv(truthfiles[0], comment="#", sep = "\s+")

    # truth_df_subset = truth_df[truth_df["CID"] == 56]
    # truth_df_subset = truth_df_subset[truth_df_subset["BAND"] == "J"]
    # print(truth_df_subset)

    # def _flux_err_to_mag_err(flux, flux_err):
    #     return 2.5 / np.log(10) * (flux_err / flux)


    # zpt_files = glob.glob('/dev_storage/campari_out_dir/zeropoint_*crds*.ecsv')
    # from matplotlib import pyplot as plt
    # mjds = []
    # zpts = []
    # zpterrs = []
    # for z in zpt_files:
    #     print(z)
    #     mjd = z.split("_")[-2].split(".")[0]
    #     print(mjd)
    #     zpt, zpterr = _load_saved_ecsv_and_determine_zpt(z)
    #     if float(mjd) > 60000:
    #         mjds.append(float(mjd))
    #         zpts.append(zpt)
    #         zpterrs.append(zpterr)
    # plt.errorbar(mjds, zpts, yerr=zpterrs, linestyle='-', label='Measured')
    # plt.xlabel("MJD")
    # plt.ylabel("Zeropoint")
    # plt.savefig("test_zeropoint.png")
    # plt.close()

    # plt.figure(figsize=(10, 8), dpi = 300)
    # #df = Table.read('/dev_storage/campari_out_dir/testing_zpts_F129_stpsf_lc.ecsv', format="ascii.ecsv")
    # df = Table.read('/dev_storage/campari_out_dir/testing_zpts_crds_no_wgt_F129_stpsf_lc.ecsv', format="ascii.ecsv")
    # mag = -2.5 * np.log10(df['flux'])
    # #mag_cal = mag + 25.530294003
    # #mag_cal = mag + 25.4806851
    # mag_cal = mag + 25.566
    # mag_err = _flux_err_to_mag_err(df['flux'], df['flux_err'])
    # mag_err = np.sqrt(mag_err**2 + 0.001**2)
    # print("MAG ERR:", mag_err)
    # print(truth_df_subset.columns)
    # plt.subplot(2, 1, 1)
    # plt.errorbar(df['mjd'], mag_cal, yerr=mag_err, marker='o', linestyle='-', label='Measured')
    # truth_mag = -2.5 * np.log10(truth_df_subset['FLUXCAL']) + 31.4
    # truth_mag_err = _flux_err_to_mag_err(truth_df_subset['FLUXCAL'], truth_df_subset['FLUXCAL_ERR'])
    # truth_mag_err = np.where(truth_mag_err > 0, truth_mag_err, np.nan)
    # print("TRUTH MAG ERR:", truth_mag_err)
    # plt.errorbar(truth_df_subset['MJD'], truth_mag, yerr=truth_mag_err, marker='s', linestyle='--', color='red', label='Truth')
    # plt.xlim(60100, 60300)
    # plt.ylim(26, 24)
    # plt.ylabel("Magnitude (GAIA Calibrated)")
    # plt.xlabel("MJD")
    # plt.legend()
    # plt.subplot(2, 1, 2)

    # campari_mjd = df["mjd"].astype(int)
    # truth_mjd = truth_df_subset['MJD'].astype(int)

    # truth_mag = truth_mag[np.isin(truth_mjd, campari_mjd)]
    # truth_mag_err = truth_mag_err[np.isin(truth_mjd, campari_mjd)]
    # #total_err = np.sqrt(mag_err**2 + truth_mag_err**2) / np.sqrt(2)
    # total_err = mag_err



    # plt.errorbar(df['mjd'], mag_cal - truth_mag, yerr=mag_err, marker='o', linestyle='-', label='Measured - Truth')

    # chi_sq_terms = ((mag_cal - truth_mag) / mag_err)**2
    # print("CHI SQ TERMS:", chi_sq_terms)
    # chi_sq = np.nansum(chi_sq_terms)
    # dof = len(mag_cal) - 1
    # reduced_chi_sq = chi_sq / dof
    # plt.title(f"Reduced Chi-Squared: {reduced_chi_sq:.2f}")

    # plt.axhline(0, color='black', linestyle='--')
    # plt.xlim(60100, 60300)
    # plt.ylim(-0.5, 0.5)
    # plt.xlabel("MJD")
    # plt.ylabel("Mag Difference")
    # plt.tight_layout()
    # plt.savefig("test_zeropoint_lc_crds.png")

    main()