from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import snappl
import snappl.wcs
from astropy.table import Table


def plot_lc(filepath, return_data=False):

    fluxdata = pd.read_csv(filepath, comment='#', delimiter=' ')

    truth_mag = fluxdata['SIM_true_mag']
    mag = fluxdata['mag']
    sigma_mag = fluxdata['mag_err']

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    dates = fluxdata['MJD']

    plt.scatter(dates, truth_mag, color='k', label='Truth')
    plt.errorbar(dates, mag, yerr=sigma_mag,  color='purple', label='Model',
                 fmt='o')

    plt.ylim(np.max(truth_mag) + 0.2, np.min(truth_mag) - 0.2)
    plt.ylabel('Magnitude (Uncalibrated)')

    residuals = mag - truth_mag
    bias = np.mean(residuals)
    bias *= 1000
    bias = np.round(bias, 3)
    scatter = np.std(residuals)
    scatter *= 1000
    scatter = np.round(scatter, 3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = 'Overall Bias: ' + str(bias) + ' mmag \n' + \
        'Overall Scatter: ' + str(scatter) + ' mmag'
    plt.text(np.percentile(dates, 60), np.mean(truth_mag), textstr,
             fontsize=14, verticalalignment='top', bbox=props)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.errorbar(dates, residuals, yerr=sigma_mag, fmt='o', color='k')
    plt.axhline(0, ls='--', color='k')
    plt.ylabel('Mag Residuals (Model - Truth)')

    plt.ylabel('Mag Residuals (Model - Truth)')
    plt.xlabel('MJD')
    plt.ylim(np.min(residuals) - 0.1, np.max(residuals) + 0.1)

    plt.axhline(0.005, color='r', ls='--')
    plt.axhline(-0.005, color='r', ls='--', label='5 mmag photometry')

    plt.axhline(0.02, color='b', ls='--')
    plt.axhline(-0.02, color='b', ls='--', label='20 mmag photometry')
    plt.legend()

    if return_data:
        return mag.values, dates.values, \
            sigma_mag.values, truth_mag.values, bias, scatter


def plot_image_and_grid(image, wcs, ra_grid, dec_grid):
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    imshow = plt.imshow(image, origin='lower', cmap='gray')
    plt.scatter(ra_grid, dec_grid)
    return imshow


def plot_images(lc_filepath, images_filepath, size=19):
    imgdata = np.load(images_filepath)
    num_total_images = imgdata.shape[1]//size**2
    images = imgdata[0]
    sumimages = imgdata[1]
    wgt_matrix = imgdata[2]
    wcs_filepath = images_filepath.split('_images.npy')[0] + '_wcs.fits'
    grid_filepath = images_filepath.split('_images.npy')[0] + '_grid.npy'
    fluxdata = Table.read(lc_filepath)
    snra, sndec = fluxdata.meta['obj_ra'], fluxdata.meta['obj_dec']
    galra, galdec = fluxdata.meta['host_ra'], fluxdata.meta['host_dec']
    hdul = fits.open(wcs_filepath)

    cutout_wcs_list = []
    for i, savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = snappl.wcs.AstropyWCS.from_header(savedwcs.header)
        cutout_wcs_list.append(newwcs)

    ra_grid, dec_grid, gridvals = np.load(grid_filepath)
    #fig = plt.figure(figsize=(15, 3*num_total_images))
    for i, wcs in enumerate(cutout_wcs_list):

        current_image = images[i*size**2:(i+1)*size**2].reshape(size, size)
        imshow = plot_image_and_grid(current_image, wcs._wcs, ra_grid, dec_grid)

        '''

        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)
        snx, sny = wcs.world_to_pixel(snra, sndec)
        galx, galy = wcs.world_to_pixel(galra, galdec)

        plt.subplot(len(cutout_wcs_list), 4, 4*i+1)
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx, yy, s=1, c='k', vmin=vmin, vmax=vmax)
        plt.title('True Image')
        plt.scatter(snx, sny, c='r', s=8, marker='*', label='Supernova')
        plt.scatter(galx, galy, c='b', s=8, marker='*', label='Host Galaxy')
        imshow = plt.imshow(images[i*size**2:(i+1)*size**2].reshape(size,size), origin = 'lower', extent = extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        trueimage = images[i*size**2:(i+1)*size**2].reshape(size,size)
        '''

        ############################################

        plt.subplot(len(cutout_wcs_list), 4, 4*i+2)
        plt.title('Model')

        extent = [-0.5, size-0.5, -0.5, size-0.5]

        im1 = sumimages[i*size**2:(i+1)*size**2].reshape(size,size)
        xx, yy = cutout_wcs_list[i].world_to_pixel(ra_grid, dec_grid)

        vmin = np.min(images[i*size**2:(i+1)*size**2].reshape(size,size))
        vmax = np.max(images[i*size**2:(i+1)*size**2].reshape(size,size))

        #im1[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) == 0)] = 0


        vmin = imshow.get_clim()[0]
        vmax = imshow.get_clim()[1]

        plt.imshow(im1, extent = extent, origin = 'lower', vmin = vmin, vmax = vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        ############################################
        plt.subplot(len(cutout_wcs_list),4,4*i+3)
        plt.title('Residuals')
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx,yy, s = 1, c= gridvals,  vmin = vmin, vmax = vmax)
        res = images - sumimages

        current_res= res[i*size**2:(i+1)*size**2].reshape(size,size)

        norm = 3*np.std(current_res[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) != 0)])

        plt.imshow(current_res, extent = extent, origin = 'lower', cmap = 'seismic', vmin = -100, vmax = 100)
        plt.colorbar(fraction=0.046, pad=0.14)


    plt.subplots_adjust(wspace = 0.4, hspace = 0.3)
    plt.legend()


def slice_plot(fileroot):
    biases = []
    fig = plt.figure(figsize = (15,2*num_total_images))
    images = imgdata[0]
    sumimages = imgdata[1]
    wgt_matrix = imgdata[2]

    fluxdata = pd.read_csv('./results/lightcurves/'+str(fileroot)+'_lc.csv')
    supernova = fluxdata['true_flux']
    measured_flux = fluxdata['measured_flux']
    snra, sndec = fluxdata['sn_ra'][0], fluxdata['sn_dec'][0]

    hdul = fits.open('./results/images/'+str(fileroot)+'_wcs.fits')
    cutout_wcs_list = []
    for i,savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = snappl.AstropyWCS.from_header(savedwcs.header)
        cutout_wcs_list.append(newwcs)

    magresiduals = -2.5*np.log10(measured_flux)+2.5*np.log10(np.array(supernova))

    galxes = []
    stds = []
    biases = []

    for i, wcs in enumerate(cutout_wcs_list):

        extent = [-0.5, size-0.5, -0.5, size-0.5]
        trueimage = images[i*size**2:(i+1)*size**2].reshape(size,size)
        snx, sny = wcs.world_to_pixel(snra, sndec)

        plt.subplot(len(cutout_wcs_list)//3 + 1,3,i+1)
        if i >= num_total_images - num_detect_images:
            plt.title('MagBias: ' + str(np.round(magresiduals[i - num_total_images + num_detect_images],4)) + ' mag')


        justbgX = np.copy(X)
        justbgX[-num_total_images:] = 0

        justbgpred = justbgX * psf_matrix
        justbgsumimages = np.sum(justbgpred, axis = 1)
        justbgim = justbgsumimages[i*size**2:(i+1)*size**2].reshape(size,size)


        #subtract off the real sn
        #if i >= num_total_images - num_detect_images:
            #justbgim -= sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*supernova[i - num_total_images + num_detect_images]




        justbgres = trueimage - justbgim
        im1 = sumimages[i*size**2:(i+1)*size**2].reshape(size,size)
        plt.axhline(0, ls = '--', color = 'k')
        plt.plot(trueimage[5] - im1[5], label = 'Im-Model', alpha = 0.4)
        plt.ylim(-250,250)


        if i >= num_total_images - num_detect_images:
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*supernova[i - num_total_images + num_detect_images]
            plt.plot(snim[5], label = 'True SN', lw = 3)
            plt.fill_between(np.arange(0,11,1), trueimage[5] - snim[5] + 50, trueimage[5] - snim[5] - 50, label = 'Im-True SN', alpha = 0.4)
            plt.plot(np.arange(0,11,1), trueimage[5] - snim[5] , color = 'k', ls = '--')
            plt.plot(justbgim[5], label = 'BGModel')
            plt.plot(justbgres[5], label = 'Im-BGModel')

            plt.ylim(-500,np.max(trueimage[5]))
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*X[-num_detect_images:][i - num_total_images + num_detect_images]

        else:
            snim = np.zeros_like(justbgres)

        plt.axvline(snx+4, ls = '--', color = 'k')
        plt.axvline(snx-4, ls = '--', color = 'k')
        plt.axvline(snx, ls = '--', color = 'r')

        plt.xlim(snx-size/2, snx+size/2)


        plt.legend(loc = 'upper left')