import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import binned_statistic, norm

from astropy.io import fits
from astropy.table import Table

from snappl.config import Config
from snappl.logger import SNLogger
from snappl.wcs import AstropyWCS

cfg = Config.get()
debug_dir = cfg.value("photometry.campari_io.debug_dir")


def plot_lc(filepath, return_data=False):
    """ Plot a lightcurve from a CSV file containing the columns: sim_true_mag, mag, mag_err, mjd.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the lightcurve data.
    return_data : bool, optional
        If True, return the data used for plotting. Default is False.
    """
    fluxdata = pd.read_csv(filepath, comment="#", delimiter=" ")
    truth_mag = fluxdata["sim_true_mag"]
    mag = fluxdata["mag"]
    sigma_mag = fluxdata["mag_err"]

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    dates = fluxdata["mjd"]

    plt.scatter(dates, truth_mag, color="k", label="Truth")
    plt.errorbar(dates, mag, yerr=sigma_mag, color="purple", label="Model", fmt="o")

    plt.ylim(np.max(truth_mag) + 0.2, np.min(truth_mag) - 0.2)
    plt.ylabel("Magnitude (Uncalibrated)")

    residuals = mag - truth_mag
    bias = np.mean(residuals)
    bias *= 1000
    bias = np.round(bias, 3)
    scatter = np.std(residuals)
    scatter *= 1000
    scatter = np.round(scatter, 3)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    textstr = "Overall Bias: " + str(bias) + " mmag \n" + "Overall Scatter: " + str(scatter) + " mmag"
    plt.text(np.percentile(dates, 60), np.mean(truth_mag), textstr, fontsize=14, verticalalignment="top", bbox=props)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.errorbar(dates, residuals, yerr=sigma_mag, fmt="o", color="k")
    plt.axhline(0, ls="--", color="k")
    plt.ylabel("Mag Residuals (Model - Truth)")

    plt.ylabel("Mag Residuals (Model - Truth)")
    plt.xlabel("MJD")
    plt.ylim(np.min(residuals) - 0.1, np.max(residuals) + 0.1)

    plt.axhline(0.005, color="r", ls="--")
    plt.axhline(-0.005, color="r", ls="--", label="5 mmag photometry")

    plt.axhline(0.02, color="b", ls="--")
    plt.axhline(-0.02, color="b", ls="--", label="20 mmag photometry")
    plt.legend()

    if return_data:
        return mag.values, dates.values, sigma_mag.values, truth_mag.values, bias, scatter


def plot_image_and_grid(image, wcs, ra_grid, dec_grid):
    """ Plot an image with a grid of RA/Dec points overlaid.

    Parameters
    ----------
    image : 2D array
        The image data to plot.
    wcs : astropy.wcs.WCS
        The WCS object for the image.
    ra_grid : 1D array
        The RA coordinates of the grid points to overlay.
    dec_grid : 1D array
        The Dec coordinates of the grid points to overlay.
    """
    SNLogger.debug(f"WCS: {type(wcs)}")
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    plt.imshow(image, origin="lower", cmap="gray")
    plt.scatter(ra_grid, dec_grid)


def generate_diagnostic_plots(fileroot, imsize, plotname, ap_sums=None, ap_err=None, trueflux=None, err_floor=0):
    """ Generate diagnostic plots for the light curve and images.

    Parameters
    -----------
    fileroot : str
        The root name of the files to read.
    imsize : int
        The size of the cutout images.
    plotname : str
        The name of the output plot file.
    ap_sums : list, optional
        The aperture photometry sums for comparison.
    ap_err : list, optional
        The errors associated with the aperture photometry sums.
    trueflux : list, optional
        The true flux values for comparison.
    err_floor : float, optional
        An additional error term to add in quadrature to the flux errors.
    """
    SNLogger.debug("Generating diagnostic plots....")
    cfg = Config.get()
    debug_dir = cfg.value("photometry.campari_io.debug_dir")
    out_dir = cfg.value("photometry.campari_io.output_dir")
    lc = Table.read(f"/{out_dir}/{fileroot}_lc.ecsv")
    ims = np.load(f"/{debug_dir}/{fileroot}_images.npy")[0].reshape(-1, imsize, imsize)
    modelims = np.load(f"/{debug_dir}/{fileroot}_images.npy")[1].reshape(-1, imsize, imsize)
    noise_maps = np.load(f"/{debug_dir}/{fileroot}_noise_maps.npy").reshape(-1, imsize, imsize)

    galra, galdec = 128.00003, 42.00003

    hdul = fits.open(f"/{debug_dir}/" + str(fileroot) + "_wcs.fits")
    cutout_wcs_list = []
    for i, savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = AstropyWCS.from_header(savedwcs.header)
        cutout_wcs_list.append(newwcs)

    numcols = 4
    plt.figure(figsize=(numcols * 5, ims.shape[0] * 5))
    for i in range(ims.shape[0]):
        k = 0
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)

        galx, galy = cutout_wcs_list[i].world_to_pixel(galra, galdec)
        plt.scatter(galx, galy, c="b", s=100, marker="*")
        if i >= lc.meta["pre_transient_images"] and i < ims.shape[0] - lc.meta["post_transient_images"]:
            plt.scatter(
                lc["x_cutout"][i - lc.meta["pre_transient_images"]],
                lc["y_cutout"][i - lc.meta["pre_transient_images"]],
                color="red",
                marker="+",
                s=100,
            )

        if i == 0:
            plt.title("Input Image")
        im = plt.imshow(ims[i], origin="lower")

        vmin, vmax = im.get_clim()
        xticks = np.arange(0, imsize, 5) - 0.5

        plt.xticks(xticks)
        plt.yticks(xticks)
        plt.grid(True)
        plt.colorbar()

        ###########################################################################

        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Model Image")
        plt.scatter(galx, galy, c="b", s=100, marker="*")
        plt.xlim(-0.5, imsize - 0.5)
        plt.ylim(-0.5, imsize - 0.5)

        plt.xticks(xticks)
        plt.yticks(xticks)
        plt.grid(True)
        plt.imshow(modelims[i], origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar()

        ################################
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Residuals")
        maxval = np.max(np.abs(ims[i] - modelims[i]))
        plt.imshow((ims[i] - modelims[i]), origin="lower", vmin=-maxval, vmax=maxval, cmap="seismic")

        # ###############################
        k += 1
        plt.subplot(ims.shape[0], numcols, numcols * i + k)
        if i == 0:
            plt.title("Pixel Pulls")
        bins = np.linspace(-4, 4, 50)
        residuals = (modelims[i] - ims[i]).flatten()
        pixel_pull = (modelims[i].flatten() - ims[i].flatten()) / noise_maps[i].flatten()
        pixel_pull = pixel_pull[np.where(np.abs(residuals) >= 1)]  # Remove zero residuals from the no transient images.
        pixel_pull = pixel_pull[np.isfinite(pixel_pull)]  # Remove any infinite values
        plt.hist(pixel_pull, bins=bins, density=True, alpha=0.5, label="Pixel Pulls")
        normal_dist = norm(loc=0, scale=1)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, normal_dist.pdf(x), label="Normal Dist", color="black")
        mu, sig = norm.fit(pixel_pull)
        plt.plot(x, norm.pdf(x, mu, sig), label=f"Fit: mu={mu:.2f}, sig={sig:.2f}", color="red")

        plt.legend()

        plt.colorbar()

        ################################
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"{debug_dir}/" + plotname + ".png")
    plt.close()

    plt.clf()

    lc["flux_err"] = _add_error_floor(lc, err_floor)
    SNLogger.debug(f"Generated image diagnostics and saved to {debug_dir}/" + plotname + ".png")
    SNLogger.debug("Now generating light curve diagnostics...")
    # Now plot a light curve
    _plot_diagnostic_lc_with_truth_if_provided(lc, trueflux, err_floor, plotname, ap_sums, ap_err, ims)

    SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")


def _plot_diagnostic_lc_with_truth_if_provided(lc, trueflux, err_floor, plotname,
                                            ap_sums=None, ap_err=None, ims=None,
                                            suffix = "_lc.png"):
    """ Plot the light curve with residuals and pulls, optionally comparing to true flux values.

    Parameters
    ----------
    lc : astropy.table.Table
        The light curve data table containing the flux and flux_err columns.
    trueflux : list or None
        The true flux values for comparison. If None, no comparison is made.
    err_floor : float
        An additional error term to add in quadrature to the flux errors.
    plotname : str
        The name of the output plot file.
    ap_sums : list, optional
        The aperture photometry sums for comparison.
    ap_err : list, optional
        The errors associated with the aperture photometry sums.
    ims : numpy.ndarray, optional
        The images used in photometry. Used only to sanity check that photometry roughly scales
        with the sum over the image.
    suffix : str, optional
        The suffix to append to the plot filename. Default is "_lc.png".
    """
    if trueflux is None:
        return

    plt.subplot(2, 2, 1)
    plt.errorbar(
        lc["mjd"],
        lc["flux"] - trueflux,
        yerr=lc["flux_err"],
        marker="o",
        linestyle="None",
        label="Campari Fit - Truth",
    )

    residuals = lc["flux"] - trueflux
    window_size = 3
    if len(residuals) >= window_size:
        rolling_avg = np.convolve(residuals, np.ones(window_size) / window_size, mode="valid")
        plt.plot(lc["mjd"][window_size - 1 :], rolling_avg, label="Rolling Average", color="orange")

    if ap_sums is not None and ap_err is not None:
        SNLogger.debug(f"aperture phot std: {np.std(np.asarray(ap_sums) - trueflux)}")
        plt.errorbar(
            lc["mjd"],
            np.asarray(ap_sums) - trueflux,
            yerr=ap_err,
            marker="o",
            linestyle="None",
            label="Aperture Phot - Truth",
            color="red",
        )
        plt.errorbar(
            lc["mjd"],
            lc["flux"] - np.array(ap_sums),
            yerr=np.sqrt(lc["flux_err"] ** 2 + np.array(ap_err) ** 2),
            marker="o",
            linestyle="None",
            label="Campari - Aperture Phot",
            color="green",
        )

        non_transient_images = lc.meta["post_transient_images"] + lc.meta["pre_transient_images"]
        image_sums = np.array([np.sum(ims[i + non_transient_images]) for i in range(ims.shape[0] - non_transient_images)])
        plt.errorbar(
            lc["mjd"], image_sums - trueflux, yerr=0, marker="o", linestyle="None",
            label="Image Sum - Truth", color="purple"
        )

    SNLogger.debug(f"campari std: {np.std(lc['flux'] - trueflux)}")

    plt.axhline(0, color="black", linestyle="--")
    plt.legend()
    plt.xlabel("MJD")
    plt.ylabel("Flux (e-)")
    plt.xlim(np.min(lc["mjd"]) - 10, np.max(lc["mjd"]) + 10)
    plt.title(plotname + " Light Curve Residuals")

    plt.subplot(2, 2, 2)
    pull = (lc["flux"] - trueflux) / lc["flux_err"]
    plt.hist(pull, bins=10, alpha=0.5, label="Campari Pull", density=True)
    normal_dist = norm(loc=0, scale=1)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, normal_dist.pdf(x), label="Normal Dist", color="black")

    mu, sig = norm.fit(pull)
    plt.plot(x, norm.pdf(x, mu, sig), label=f"Fit: mu={mu:.2f}, sig={sig:.2f}", color="red")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.errorbar(
        lc["mjd"], lc["flux"], yerr=lc["flux_err"], marker="o", linestyle="None", label="Campari Fit - Truth", ms=1
    )
    plt.errorbar(lc["mjd"], trueflux, yerr=None, marker="o", linestyle="None", label="Truth", color="black", ms=1)
    plt.yscale("log")

    plt.subplot(2, 2, 4)
    plt.errorbar(lc["flux"], pull, yerr=None, marker="o", linestyle="None")
    bins = np.linspace(min(lc["flux"]), max(lc["flux"]), 5)
    bin_means, bin_edges, _ = binned_statistic(lc["flux"], pull, statistic="mean", bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Flux")
    plt.ylabel("Pull")
    bin_stds, _, _ = binned_statistic(lc["flux"], pull, statistic="std", bins=bins)
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o", color="red", label="Binned Mean Pull with Std Dev")
    plt.legend()

    plt.savefig(f"/{debug_dir}/" + plotname + suffix)


def _add_error_floor(lc, err_floor):
    """ Add an additional error term in quadrature to the flux errors.

    Parameters
    ----------
    lc : astropy.table.Table
        The light curve table containing the flux and flux_err columns.
    err_floor : float
        The additional error term to add in quadrature to the flux errors.
    """
    if lc["flux_err"] is not None:
        error = np.sqrt(lc["flux_err"] ** 2 + err_floor**2)
    else:
        # Noiseless tests will have no flux error column so we make a new one directly.
        error = np.full_like(lc["flux"], err_floor)
    return error


def plot_cutouts_if_requested(cutout_image_list, ra, dec, diaobj=None, ncols=5, output_path=None):
    """Plot all cutout images labeled with their MJD and the location of the supernova.

    Parameters
    ----------
    cutout_image_list : list of snappl.image.Image objects
        The cutout images to plot, as returned by construct_images().
    ra : float
        RA of the supernova in degrees.
    dec : float
        Dec of the supernova in degrees.
    diaobj : snappl.diaobject.DiaObject, optional
        If provided, images are outlined to indicate whether they fall
        within the transient window (mjd_start to mjd_end).
    ncols : int
        Number of columns in the grid of subplots.
    output_path : str or pathlib.Path, optional
        If provided, save the figure to this path. Otherwise, call plt.show().
    """
    if not Config.get().value("photometry.campari.preplot_cutouts"):
        return
    num_images = len(cutout_image_list)
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    # Flatten so we can iterate even if nrows==1
    axes = np.atleast_1d(axes).flatten()

    # Robust color scale: use 10/99th percentile to avoid hot pixels
    # dominating the colormap
    concat_data = [im.data.flatten() for im in cutout_image_list]
    concat_data = np.concatenate(concat_data)
    vmin = np.nanpercentile(concat_data, 10)
    vmax = np.nanpercentile(concat_data, 99)

    for i, image in enumerate(cutout_image_list):

        plot_noise = False
        ax = axes[i]
        if plot_noise:
            data = image.noise
        else:
            data = image.data

        ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap = "viridis")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

        error = image.noise
        snr = np.abs(data / error)
        peak_snr = np.nanmax(snr)
        # Mark the location of the peak SNR pixel with a blue circle
        peak_y, peak_x = np.unravel_index(np.nanargmax(snr), data.shape)
        if not plot_noise:
            ax.scatter(peak_x, peak_y, marker="o", color="blue", s=100, linewidths=.5,
                       label="Peak SNR" if i == 0 else None)

        # Mark the SN location in cutout pixel coordinates
        _try_to_plot_SN(image, ax, ra, dec, i)
        # Label with MJD; color the title to distinguish detection vs. non-detection
        title_color = _color_title_by_whether_image_contains_transient(diaobj, image)

        ax.set_title(f"MJD {image.mjd:.7f} Texp: {image.exptime:.1f}s PkSNR:"
                     f" {peak_snr:.1f}", fontsize=8, color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])

        # Light border to visually separate panels
        for spine in ax.spines.values():
            spine.set_edgecolor(title_color)
            spine.set_linewidth(1.5 if title_color == "red" else 0.5)

    # Hide any unused axes
    for j in range(num_images, len(axes)):
        axes[j].set_visible(False)

    # Add a shared legend for the SN marker and detection colouring
    legend_elements = [
        plt.Line2D([0], [0], marker="+", color="red", linestyle="None",
                   markersize=10, label="SN position"),
    ]
    if diaobj is not None:
        from matplotlib.patches import Patch
        legend_elements.append(Patch(edgecolor="red",  facecolor="none", label="Detection image"))
        legend_elements.append(Patch(edgecolor="black", facecolor="none", label="Non-detection image"))
    if not plot_noise:
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="blue", linestyle="None",
                   markersize=10, label="Peak SNR pixel"))
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 0.0), fontsize=9, frameon=True)

    name = diaobj.name if diaobj is not None else "(no name)"
    plt.suptitle(f"Cutouts  for {name} (RA={ra:.5f}, Dec={dec:.5f})", fontsize=11, y=1.01)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        SNLogger.info(f"Saved cutout grid to {output_path}")
        plt.close()
    else:
        plt.show()


def _try_to_plot_SN(image, ax, ra, dec, i):
    """Try to plot the SN position on the cutout image. If the WCS is invalid, log a warning.

    Parameters
    ----------
    image : snappl.image.Image
        The cutout image.
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    ra : float
        RA of the supernova in degrees.
    dec : float
        Dec of the supernova in degrees.
    i : int
        Index of the cutout image (for logging purposes).
    """
    try:
        wcs = image.get_wcs()
        sn_x, sn_y = wcs.world_to_pixel(ra, dec)
        ax.scatter(sn_x, sn_y, marker="+", color="red", s=100, linewidths=1.5,
                    label="SN" if i == 0 else None)
    except ValueError:
        SNLogger.warning(f"Could not project SN position onto cutout {i} (mjd={image.mjd:.7f})")


def _color_title_by_whether_image_contains_transient(diaobj, image):
    """ Determine the color of the title for a cutout image based on whether it falls within the
    transient window of a DiaObject.
    Parameters
    ----------

    diaobj : snappl.diaobject.DiaObject
        The DiaObject containing the transient information.
    image : snappl.image.Image
        The cutout image for which to determine the title color.

    Returns
    -------
    title_color : str
        "red" if the image falls within the transient window, "black" otherwise.
    """
    title_color = "black"
    if diaobj is not None:
        mjd_start = getattr(diaobj, "mjd_start", None) or -np.inf
        mjd_end   = getattr(diaobj, "mjd_end",   None) or  np.inf
        if mjd_start <= image.mjd <= mjd_end:
            title_color = "red"
    return title_color
