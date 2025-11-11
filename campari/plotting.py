import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from astropy.io import fits

import snappl
from snappl.logger import SNLogger


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
