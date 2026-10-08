import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from scipy.stats import binned_statistic, norm

from astropy.io import fits
from astropy.table import Table

import snappl
from snappl.config import Config
from snappl.logger import SNLogger
from snappl.wcs import AstropyWCS
Config.init("/home/cfmeldorf/campari/examples/SMDC/campari_config_test.yaml")
cfg = Config.get()
debug_dir = cfg.value("photometry.campari_io.debug_dir")


def plot_images(fileroot, size=11):
    imgdata = np.load("./results/images/" + str(fileroot) + "_images.npy")
    num_total_images = imgdata.shape[1] // size**2
    images = imgdata[0]
    sumimages = imgdata[1]

    fluxdata = pd.read_csv("./results/lightcurves/" + str(fileroot) + "_lc.csv")

    ra, dec = fluxdata["sn_ra"][0], fluxdata["sn_dec"][0]
    galra, galdec = fluxdata["host_ra"][0], fluxdata["host_dec"][0]

    hdul = fits.open("./results/images/" + str(fileroot) + "_wcs.fits")
    cutout_wcs_list = []
    for i, savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = snappl.AstropyWCS.from_header(savedwcs.header)
        cutout_wcs_list.append(newwcs)

    ra_grid, dec_grid, gridvals = np.load("./results/images/" + str(fileroot) + "_grid.npy")

    plt.figure(figsize=(15, 3 * num_total_images))

    for i, wcs in enumerate(cutout_wcs_list):
        extent = [-0.5, size - 0.5, -0.5, size - 0.5]
        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)
        object_x, object_y = wcs.world_to_pixel(ra, dec)
        galx, galy = wcs.world_to_pixel(galra, galdec)

        plt.subplot(len(cutout_wcs_list), 4, 4 * i + 1)
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx, yy, s=1, c="k", vmin=vmin, vmax=vmax)
        plt.title("True Image")
        plt.scatter(object_x, object_y, c="r", s=8, marker="*")
        plt.scatter(galx, galy, c="b", s=8, marker="*")
        imshow = plt.imshow(images[i * size**2 : (i + 1) * size**2].reshape(size, size), origin="lower", extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)

        ############################################

        plt.subplot(len(cutout_wcs_list), 4, 4 * i + 2)
        plt.title("Model")

        im1 = sumimages[i * size**2 : (i + 1) * size**2].reshape(size, size)
        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)

        vmin = imshow.get_clim()[0]
        vmax = imshow.get_clim()[1]

        plt.imshow(im1, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        ############################################
        plt.subplot(len(cutout_wcs_list), 4, 4 * i + 3)
        plt.title("Residuals")
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx, yy, s=1, c=gridvals, vmin=vmin, vmax=vmax)
        res = images - sumimages
        current_res = res[i * size**2 : (i + 1) * size**2].reshape(size, size)
        plt.imshow(current_res, extent=extent, origin="lower", cmap="seismic", vmin=-100, vmax=100)
        plt.colorbar(fraction=0.046, pad=0.14)

    plt.subplots_adjust(wspace=0.4, hspace=0.3)


def plot_lc(filepath, return_data=False):
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
    SNLogger.debug(f"WCS: {type(wcs)}")
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    plt.imshow(image, origin="lower", cmap="gray")
    plt.scatter(ra_grid, dec_grid)


def generate_diagnostic_plots(fileroot, imsize, plotname, ap_sums=None, ap_err=None, trueflux=None, err_fudge=0):
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

        if imsize < 30:
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

        if imsize < 30:
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
    if lc["flux_err"] is not None:
        lc["flux_err"] = np.sqrt(lc["flux_err"] ** 2 + err_fudge**2)
    else:
        # Noiseless tests will have no error.
        lc["flux_err"] = np.full_like(lc["flux"], err_fudge)
    SNLogger.debug(f"Generated image diagnostics and saved to {debug_dir}/" + plotname + ".png")
    SNLogger.debug("Now generating light curve diagnostics...")
    # Now plot a light curve
    if trueflux is not None:
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
            SNLogger.debug(f"aperture phot std: {np.std(np.array(ap_sums) - trueflux)}")
            plt.errorbar(
                lc["mjd"],
                np.array(ap_sums) - trueflux,
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
            image_sums = [np.sum(ims[i + non_transient_images]) for i in range(ims.shape[0] - non_transient_images)]
            plt.errorbar(
                lc["mjd"], np.array(image_sums) - trueflux, yerr=0, marker="o", linestyle="None",
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

        plt.savefig(f"/{debug_dir}/" + plotname + "_lc.png")

    SNLogger.debug(f"Generated saved diagnostic plots to {debug_dir}/{plotname}.png")


def plot_cutouts(cutout_image_list, ra, dec, diaobj=None, ncols=5, output_path=None):
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
        try:
            wcs = image.get_wcs()
            sn_x, sn_y = wcs.world_to_pixel(ra, dec)
            ax.scatter(sn_x, sn_y, marker="+", color="red", s=100, linewidths=1.5,
                       label="SN" if i == 0 else None)
        except Exception:
            SNLogger.warning(f"Could not project SN position onto cutout {i} (mjd={image.mjd:.7f})")

        # Label with MJD; color the title to distinguish detection vs. non-detection
        title_color = "black"
        if diaobj is not None:
            mjd_start = getattr(diaobj, "mjd_start", None) or -np.inf
            mjd_end   = getattr(diaobj, "mjd_end",   None) or  np.inf
            if mjd_start <= image.mjd <= mjd_end:
                title_color = "red"

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


def plot_postrun_summary(lc_model, diaobj, output_path):
    """Make a summary figure after a campari run.

    Top: light curve (flux vs. MJD with error bars).
    Below: one row per epoch with [real image | model image | residuals].
    In the residual panels, pixels whose residual is exactly zero (or not
    finite) are masked, so they show up blank.

    Parameters
    ----------
    lc_model : campari_lightcurve_model
        Output of run_one_object.
    diaobj : snappl.diaobject.DiaObject
        Used for mjd_start / mjd_end to flag detection epochs.
    output_path : str or pathlib.Path
        Where to save the .png.
    """
    cutouts = lc_model.cutout_image_list
    n_epochs = len(cutouts)
    size = cutouts[0].image_shape[0]
    mjds = np.array([im.mjd for im in cutouts])

    # lc_model.images / model_images are flat 1D arrays with the pixels of every
    # epoch concatenated, so reshape to (epoch, y, x).
    data = np.asarray(lc_model.images).reshape(n_epochs, size, size)
    model = np.asarray(lc_model.model_images).reshape(n_epochs, size, size)
    resid = data - model

    # Weights are stored in the same flat, epoch-by-epoch order as the images.
    # A weight of zero means the pixel was excluded from the fit (outside the
    # `cutoff` radius, or NaN in the original image).
    if lc_model.wgt_matrix is not None:
        zero_weight = np.asarray(lc_model.wgt_matrix).reshape(n_epochs, size, size) == 0
    else:
        SNLogger.warning("No weights found on lc_model; not masking any pixels in the post-run plot.")
        zero_weight = np.zeros((n_epochs, size, size), dtype=bool)

    is_detection = (mjds >= diaobj.mjd_start) & (mjds <= diaobj.mjd_end)

    # The fit orders images as [non-detections..., detections...], each sorted by
    # MJD. For display we want strict time order.
    order = np.argsort(mjds)

    # Figure setup
    lc_height = 4.0
    row_height = 2.6
    fig_height = lc_height + row_height * n_epochs
    fig = Figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(n_epochs + 1, 3, height_ratios=[lc_height / row_height] + [1] * n_epochs,
                          hspace=0.35, wspace=0.35)

    # Top panel: light curve
    ax_lc = fig.add_subplot(gs[0, :])
    if lc_model.flux is not None:
        det_mjds = mjds[is_detection]  # already in the same order as lc_model.flux
        ax_lc.errorbar(det_mjds, np.atleast_1d(lc_model.flux), yerr=np.atleast_1d(lc_model.sigma_flux),
                       fmt="o", color="purple", capsize=2)
        ax_lc.axhline(0, color="k", ls="--", lw=0.8)
        ax_lc.set_xlabel("MJD")
        ax_lc.set_ylabel("Flux (e-/s)")
    else:
        ax_lc.text(0.5, 0.5, "No detection images, so no light curve", ha="center", va="center",
                   transform=ax_lc.transAxes)
    ax_lc.set_title(f"Campari light curve: {getattr(diaobj, 'name', '')}")

    # Residual colormap: masked pixels are drawn white
    resid_cmap = matplotlib.colormaps["seismic"].copy()
    resid_cmap.set_bad("white")
    model_cmap = matplotlib.colormaps["viridis"].copy()
    model_cmap.set_bad("white")

    # One row per epoch
    for row, i in enumerate(order):
        ax_data = fig.add_subplot(gs[row + 1, 0])
        ax_model = fig.add_subplot(gs[row + 1, 1])
        ax_resid = fig.add_subplot(gs[row + 1, 2])

        # Real and model images share a color scale so they can be compared by eye.
        finite = data[i][np.isfinite(data[i])]
        vmin, vmax = (np.percentile(finite, 1), np.percentile(finite, 99)) if finite.size else (0, 1)

                # The real image is shown in full, for reference.
        im0 = ax_data.imshow(data[i], origin="lower", vmin=vmin, vmax=vmax)

        # Model and residuals hide every pixel that had zero weight in the fit
        # (and any non-finite values).
        hide = zero_weight[i] | ~np.isfinite(model[i]) | ~np.isfinite(resid[i])
        masked_model = np.ma.masked_where(hide, model[i])
        masked_resid = np.ma.masked_where(hide, resid[i])

        im1 = ax_model.imshow(masked_model, origin="lower", cmap=model_cmap, vmin=vmin, vmax=vmax)

        # Symmetric color scale built only from pixels that were actually fit,
        # so the excluded corners can't dominate the scale.
        rmax = np.max(np.abs(masked_resid)) if masked_resid.count() > 0 else 1.0
        im2 = ax_resid.imshow(masked_resid, origin="lower", cmap=resid_cmap, vmin=-rmax, vmax=rmax)

        for ax, im in ((ax_data, im0), (ax_model, im1), (ax_resid, im2)):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])

        label_color = "red" if is_detection[i] else "black"
        ax_data.set_ylabel(f"MJD {mjds[i]:.3f}", color=label_color, fontsize=9)
        if row == 0:
            ax_data.set_title("Real image")
            ax_model.set_title("Model")
            ax_resid.set_title("Residual (data - model)\nzero-weight pixels masked")

    # Matplotlib refuses to write images over ~65000 pixels on a side, so lower
    # the resolution for runs with a very large number of epochs.
    dpi = int(min(100, 60000 / fig_height))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    SNLogger.info(f"Saved post-run summary plot to {output_path}")
