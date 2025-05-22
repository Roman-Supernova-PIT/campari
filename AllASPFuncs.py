# TODO -- remove these next few lines!
# This needs to be set up in an environment
# where snappl is available.  This will happen "soon"
# Get Rob to fix all of this.  For now, this is a hack
# so you can work short term.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent/"extern/snappl"))
# End of lines that will go away once we do this right

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils
from roman_imsim import *
import astropy.table as tb
import warnings
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from astropy.nddata import Cutout2D
from coord import *
import requests
from astropy.table import Table
from astropy.table import QTable
import os
import time
import galsim
import h5py
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.interpolate import RegularGridInterpolator
from snappl.image import OpenUniverse2024FITSImage
from snappl.logger import Lager

# This supresses a warning because the Open Universe Simulations dates are not
# FITS compliant.
warnings.simplefilter('ignore', category=AstropyWarning)
# Because the Open Universe Sims have dates from the future, we supress a
# warning about using future dates.
warnings.filterwarnings("ignore", category=ErfaWarning)

r'''
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


'''


def make_regular_grid(ra_center, dec_center, wcs, size, spacing=1.0,
                      subsize=9):
    '''
    Generates a regular grid around a (RA, Dec) center, choosing step size.

    ra_center, dec_center: floats, coordinate center of the image
    wcs: the WCS of the image, currently a galsim.fitswcs.AstropyWCS object
    spacing: int, spacing of grid points in pixels.
    subsize: Int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.


    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.


    '''
    if subsize > size:
        Lager.warning('subsize is larger than the image size. ' +
                      f'{size} > {subsize}. This would cause model points to' +
                      ' be placed outside the image. Reducing subsize to' +
                      ' match the image size.')
        subsize = size

    Lager.debug('Grid type: regularly spaced')
    difference = int((size - subsize)/2)

    x_center, y_center = wcs.toImage(ra_center, dec_center, units='deg')

    x = difference + np.arange(0, subsize, spacing)
    y = difference + np.arange(0, subsize, spacing)
    Lager.debug(f'Grid spacing: {spacing}')

    xx, yy = np.meshgrid(x+1, y+1)
    xx = xx.flatten()
    yy = yy.flatten()
    Lager.debug(f'Built a grid with {np.size(xx)} points')

    ra_grid, dec_grid = wcs.toWorld(xx, yy, units='deg')
    return ra_grid, dec_grid


def make_adaptive_grid(ra_center, dec_center, wcs,
                       image, percentiles=[45, 90], subsize=9,
                       subpixel_grid_width=1.2):
    '''
    Construct an "adaptive grid" which allocates model grid points to model
    the background galaxy according to the brightness of the image.

    Inputs:
    ra_center, dec_center: floats, coordinate center of the image
    wcs: the WCS of the image, currently a galsim.fitswcs.AstropyWCS object
    image: 2D numpy array of floats of shape (size x size), the image to build
    the grid on. This is used to determine the size of the grid, and once we
                switch to snappl Image objects, will also determine the wcs.
    percentiles: list of floats, the percentiles to use to bin the image. The
                more bins, the more possible grid points could be placed in
                that pixel. For instance, say if you had bins [45, 90],
                as is default. A pixel that lies in the 30th percentile for
                brightness would get 1 point. A pixel at the 50th percentiile
                for brightness would get a 2x2 grid of points.
                A pixel above the 90th percentile would get a 3x3 grid of
                points. If you have more bins, you could go even higher to
                4x4 and 5x5 etc. These points are evenly spaced within the
                pixel.
    subsize: Int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.
    subpixel_grid_width: When we place the model points in a pixel, we place
                        them on a small range of locations within the pixel.
                        For instance, 0.25, 0.5, and 0.75 for x values. However
                        I've found this leads to awkward gaps in grid points
                        between pixels. For instance, the point at 0.25 would
                        be half a pixel from the point located at 0.75 in the
                        next lower pixel, and only 0.25 from the point at 0.5
                        in the same pixel.
                        Therefore, subpixel_grid_width can be
                        set to something larger than 1 so that the evenly
                        spaced points are spaced out more, reducing these gaps.
                        For instance, if you set to 1.2, the x values of the
                        pixels would be [0.2, 0.5, 0.8] instead, reducing
                        inter-pixel gaps.

    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.
    '''
    size = np.shape(image)[0]
    if subsize > size:
        Lager.warning('subsize is larger than the image size '  +
                      f'{size} > {subsize}. This would cause model points to' +
                      ' be placed outside the image. Reducing subsize to' +
                      ' match the image size.')
        subsize = size

    Lager.debug('image shape: {}'.format(np.shape(image)))
    Lager.debug('Grid type: adaptive')
    # Bin the image in logspace and allocate grid points based on the
    # brightness.

    difference = int((size - subsize)/2)
    x_center, y_center = wcs.toImage(ra_center, dec_center, units='deg')
    x = difference + np.arange(0, subsize, 1)
    y = difference + np.arange(0, subsize, 1)

    if percentiles.sort() != percentiles:
        Lager.warning('Percentiles not in ascending order. Sorting them.')
        percentiles.sort()
        Lager.warning(f'Percentiles: {percentiles}')

    imcopy = np.copy(image)
    # We need to make sure that the image is not zero, otherwise we get
    # infinities in the log space.
    imcopy[imcopy <= 0] = 1e-10
    imcopy = np.log(imcopy)
    bins = [0]
    bins.extend(np.nanpercentile(imcopy, percentiles))
    bins.append(100)
    Lager.debug(f'BINS: {bins}')

    brightness_levels = np.digitize(imcopy, bins)
    xs = []
    ys = []
    # Round y and x locations to the nearest pixel. This is necessary because
    # we want to check the brightness for each pixel within the grid, and by
    # rounding we can index the brightness_levels array.
    yvals = np.rint(y).astype(int)
    xvals = np.rint(x).astype(int)
    for xindex in xvals:
        x = xindex + 1
        for yindex in yvals:
            y = yindex + 1
            # xindex and yindex are the indices within the numpy array, while
            # x and y are the actual locations in pixel space.
            # This used to be x and y in here:
            num = int(brightness_levels[xindex][yindex])
            if num == 0:
                pass
            elif num == 1:
                xs.append(y)
                ys.append(x)  # I know I swap this because Astropy takes (y,x)
                # order but I'd really like to iron out all the places I do
                # this rather than doing it so off the cuff. TODO
            else:
                xx = np.linspace(x - subpixel_grid_width/2,
                                 x + subpixel_grid_width/2, num+2)[1:-1]
                yy = np.linspace(y - subpixel_grid_width/2,
                                 y + subpixel_grid_width/2, num+2)[1:-1]
                X, Y = np.meshgrid(xx, yy)
                ys.extend(list(X.flatten()))
                xs.extend(list(Y.flatten()))  #...Like here. TODO

    xx = np.array(xs).flatten()
    yy = np.array(ys).flatten()

    Lager.debug(f'Built a grid with {np.size(xx)} points')

    ra_grid, dec_grid = wcs.toWorld(xx, yy, units='deg')
    return ra_grid, dec_grid


def generateGuess(imlist, wcslist, ra_grid, dec_grid):
    '''
    This function initializes the guess for the optimization. For each grid
    point, it finds the average value of the pixel it is sitting in on
    each image. In some cases, this has offered minor improvements but it is
    not make or break for the algorithm.
    '''
    size = np.shape(imlist[0])[0]
    imx = np.arange(0, size, 1)
    imy = np.arange(0, size, 1)
    imx, imy = np.meshgrid(imx, imy)
    all_vals = np.zeros_like(ra_grid)

    for i,imwcs in enumerate(zip(imlist, wcslist)):
        im, wcs = imwcs
        if type(wcs) == galsim.fitswcs.AstropyWCS:
            # This actually means that we have a galsim wcs that was loaded from an astropy one
            xx, yy = wcs.toImage(ra_grid, dec_grid,units='deg')
        else:
            xx, yy = wcs.world_to_pixel(SkyCoord(ra = ra_grid*u.degree, dec = dec_grid*u.degree))

        grid_point_vals = np.zeros_like(xx)
        for imval, imxval, imyval in zip(im.flatten(), imx.flatten(), imy.flatten()):
            grid_point_vals[np.where((np.abs(xx - imxval) < 0.5) & (np.abs(yy - imyval) < 0.5))] = imval
        all_vals += grid_point_vals
    return all_vals/len(wcslist)


def construct_psf_background(ra, dec, wcs, x_loc, y_loc, stampsize, bpass,
                             use_roman, color=0.61, psf=None, pixel=False,
                             include_photonOps=False, util_ref=None,
                             band=None):

    '''
    Constructs the background model around a certain image (x,y) location and
    a given array of RA and DECs.
    Inputs:
    ra, dec: arrays of RA and DEC values for the grid
    wcs: the wcs of the image, if the image is a cutout, this MUST be the wcs
    of the CUTOUT
    x_loc, y_loc: the pixel location of the image in the FULL image, i.e. x y
    location in the SCA.
    stampsize: the size of the stamp being used
    bpass: the bandpass being used
    flatten: whether to flatten the output array (REMOVED XXXXXX)
    color: the color of the star being used (currently not used)
    psf: Here you can provide a PSF to use, if you don't provide one, you must
    provide a util_ref, which will calculate the Roman PSF instead.
    pixel: If True, use a pixel tophat function to convolve the PSF with,
    otherwise use a delta function. Does not seem to hugely affect results.
    include_photonOps: If True, use photon ops in the background model.
    This is not recommended for general use, as it is very slow.
    util_ref: A reference to the util object, which is used to calculate the
            PSF. If you provide this, you don't need to provide a PSF. Note
            that this needs to be for the correct SCA/Pointing combination.

    Returns:
    A numpy array of the PSFs at each grid point, with the shape
    (stampsize*stampsize, npoints)
    '''

    assert util_ref is not None or psf is not None, 'you must provide at \
        least util_ref or psf'
    assert util_ref is not None or band is not None, 'you must provide at \
        least util_ref or band'

    if not use_roman:
        assert psf is not None, 'you must provide an input psf if \
                                 not using roman.'
    else:
        psf = None

    if type(wcs) == galsim.fitswcs.AstropyWCS:
        x, y = wcs.toImage(ra,dec,units='deg')
    else:
        x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree))

    psfs = np.zeros((stampsize * stampsize,np.size(x)))

    k = 0

    sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
    point = None
    if pixel:
        point = galsim.Pixel(0.1)*sed
    else:
        point = galsim.DeltaFunction()
        point *= sed

    point = point.withFlux(1,bpass)
    oversampling_factor = 1
    pupil_bin = 8

    newwcs = wcs
    #Loop over the grid points, draw the PSF at each one, and append to a list.

    #How different are these two methods? TODO XXX

    #roman_psf =  util_ref.getPSF(x_loc,y_loc,pupil_bin)
    roman_psf = galsim.roman.getPSF(1,band, pupil_bin=8, wcs = newwcs)

    for a,ij in enumerate(zip(x.flatten(),y.flatten())):
        i,j = ij
        stamp = galsim.Image(stampsize*oversampling_factor,stampsize*oversampling_factor,wcs=newwcs)

        if not include_photonOps:
            if use_roman:
                convolvedpsf = galsim.Convolve(point, roman_psf)
            else:
                convolvedpsf = galsim.Convolve(point, psf)
            result = convolvedpsf.drawImage(bpass, method='no_pixel',\
                center = galsim.PositionD(i, j),use_true_center = True, image = stamp, wcs = newwcs)

        else:
            photon_ops = [util_ref.getPSF(i,j,8)] + util_ref.photon_ops
            result = point.drawImage(bpass,wcs=newwcs, method='phot', photon_ops=photon_ops, rng=util_ref.rng, \
                n_photons=int(1e6),maxN=int(1e6),poisson_flux=False, center = galsim.PositionD(i+1, j+1),\
                    use_true_center = True, image=stamp)

        '''
        plt.figure(figsize = (5,5))
        plt.title('PSF inside CBg')
        plt.imshow(result.array)
        plt.colorbar()
        plt.show()
        '''

        psfs[:,k] = result.array.flatten()
        k += 1

    newstamp = galsim.Image(stampsize*oversampling_factor,stampsize*oversampling_factor,wcs=newwcs)
    #roman_bandpasses[band]
    '''
    if not psf:
        bgpsf = (util_ref.getPSF(2048,2048,pupil_bin)*sed).drawImage(bpass, wcs = newwcs, center = (5, 5), use_true_center = True, image = newstamp)
    else:
        bgpsf = (psf*sed).drawImage(bpass, wcs = newwcs, center = (5, 5), use_true_center = True, image = newstamp)
    bgpsf = bgpsf.array
    '''
    bgpsf = None
    return psfs, bgpsf


def findAllExposures(snid, ra, dec, peak, start, end, band, maxbg=24,
                     maxdet=24, return_list=False, stampsize=25,
                     roman_path=None, pointing_list=None, SCA_list=None,
                     truth='simple_model', lc_start=-np.inf, lc_end=np.inf):
    '''
    This function finds all the exposures that contain a given supernova, and returns a list of them.
    Utilizes Rob's awesome database method to find the exposures. Humongous speed up thanks to this.

    Inputs:
    snid: the ID of the supernova
    ra, dec: the RA and DEC of the supernova (TODO: Is this necessary if we're passing the ID?)
    peak: the peak of the supernova
    start, end: the start and end of the observing window
    maxbg: the maximum number of background images to consider
    maxdet: the maximum number of detected images to consider
    return_list: whether to return the exposures as a list or not
    stampsize: the size of the stamp to use
    roman_path: the path to the Roman data
    pointing_list: If this is passed in, only consider these pointings
    SCA_list: If this is passed in, only consider these SCAs
    truth: If 'truth' use truth images, if 'simple_model' use simple model images.
    band: the band to consider
    lc_start, lc_end: the start and end of the light curve window, in terms of
                      time, in days, away from the peak.
    '''
    g = fits.open(roman_path + '/RomanTDS/Roman_TDS_obseq_11_6_23.fits')[1] #Am I still using this? XXX TODO
    g = g.data
    alldates = g['date']
    f = fits.open(roman_path + '/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits')[1]
    f = f.data

    explist = tb.Table(names=('Pointing', 'SCA', 'BAND', 'zeropoint', 'RA', 'DEC', 'date', 'true mag', 'true flux', 'realized flux'),\
            dtype=('i8', 'i4', 'str', 'f8', 'f8', 'f8', 'f8','f8', 'f8', 'f8'))

    #Rob's database method! :D

    server_url = 'https://roman-desc-simdex.lbl.gov'
    req = requests.Session()
    result = req.post( f'{server_url}/findromanimages/containing=({ra},{dec})' )
    if result.status_code != 200:
        raise RuntimeError( f"Got status code {result.status_code}\n{result.text}" )

    res = pd.DataFrame(result.json())[['filter','pointing','sca', 'mjd']]
    res.rename(columns={'mjd': 'date', 'pointing': 'Pointing', 'sca': 'SCA'},
               inplace=True)

    res = res.loc[res['filter'] == band]
    det = res.loc[(res['date'] >= start) & (res['date'] <= end)].copy()
    det['offpeak_time'] = det['date'] - peak
    det = det.sort_values('offpeak_time')
    if lc_start != -np.inf or lc_end != np.inf:
        det = det.loc[(det['offpeak_time'] >= lc_start) &
                      (det['offpeak_time'] <= lc_end)]
    if isinstance(maxdet, int):
        det = det.iloc[:maxdet]
    det['DETECTED'] = True

    if pointing_list is not None:
        det = det.loc[det['Pointing'].isin(pointing_list)]

    bg = res.loc[(res['date'] < start) | (res['date'] > end)].copy()
    bg['offpeak_time'] = bg['date'] - peak
    if lc_start != -np.inf or lc_end != np.inf:
        bg = bg.loc[(bg['offpeak_time'] >= lc_start) &
                    (bg['offpeak_time'] <= lc_end)]
    if isinstance(maxbg, int):
        bg = bg.iloc[:maxbg]
    bg['DETECTED'] = False

    # combine these two dataframes
    all_images = pd.concat([det, bg])
    all_images['zeropoint'] = np.nan

    # Now we need to loop through the images and get the information we need
    zpts = []
    true_mags = []
    true_fluxes = []
    realized_fluxes = []
    for index, row in all_images.iterrows():
        cat = pd.read_csv(roman_path+f'/RomanTDS/truth/{band}/{row.Pointing}/Roman_TDS_index_{band}_{row.Pointing}_{row.SCA}.txt',\
                                sep="\s+", skiprows = 1,
                                names = ['object_id', 'ra', 'dec', 'x', 'y', 'realized_flux', 'flux', 'mag', 'obj_type'])
        cat_star = cat.loc[cat['obj_type'] == 'star']
        logflux = -2.5*np.log10(cat_star['flux'])
        mag = cat_star['mag']
        zpt = np.mean(mag - logflux)
        zpts.append(zpt)

        if row.DETECTED:
            try:
                true_mags.append(cat.loc[cat['object_id'] == snid].mag.values[0])
                true_fluxes.append(cat.loc[cat['object_id'] == snid].flux.values[0])
                realized_fluxes.append(cat.loc[cat['object_id'] == snid].realized_flux.values[0])

            except:
                Lager.error(f'No truth file found for \
                             {row.Pointing, row.SCA}')
                true_mags.append(np.nan)
                true_fluxes.append(np.nan)
                realized_fluxes.append(np.nan)
                continue

        else:
            true_mags.append(np.nan)
            true_fluxes.append(np.nan)
            realized_fluxes.append(np.nan)
    all_images['zeropoint'] = zpts
    all_images['true mag'] = true_mags
    all_images['true flux'] = true_fluxes
    all_images['realized flux'] = realized_fluxes
    all_images['BAND'] = band

    explist = Table.from_pandas(all_images)
    explist.sort(['DETECTED', 'SCA'])
    Lager.info('\n' + str(explist))

    if return_list:
        return explist

def find_parq(ID, path, obj_type = 'SN'):


    '''
    Find the parquet file that contains a given supernova ID.
    '''
    files = os.listdir(path)
    file_prefix = {"SN": "snana", "star": "pointsource"}
    files = [f for f in files if file_prefix[obj_type] in f]
    files = [f for f in files if '.parquet' in f]
    files = [f for f in files if 'flux' not in f]

    for f in files:
        pqfile = int(f.split('_')[1].split('.')[0])
        df = open_parq(pqfile, path, obj_type = obj_type)
        #The issue is SN parquets store their IDs as ints and star parquets as strings.
        # Should I convert the entire array or is there a smarter way to do this?
        if ID in df.id.values or str(ID) in df.id.values:
            Lager.debug(f'parq file: {pqfile}')
            return pqfile

def open_parq(parq, path, obj_type = 'SN', engine="fastparquet"):
    '''
    Convenience function to open a parquet file given its number.
    '''
    file_prefix = {"SN": "snana", "star": "pointsource"}
    base_name = "{0:s}_{1}.parquet".format(file_prefix[obj_type], parq)
    file_path = os.path.join(path, base_name)
    df = pd.read_parquet(file_path, engine=engine)
    return df


def radec2point(RA, DEC, filt, path, start = None, end = None):
    '''
    This function takes in RA and DEC and returns the pointing and SCA with
    center closest to desired RA/DEC
    '''
    f = fits.open(path+'/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits')[1]
    f = f.data

    g = fits.open(path+'/RomanTDS/Roman_TDS_obseq_11_6_23.fits')[1]
    g = g.data
    alldates = g['date']

    allRA = f['RA']
    allDEC = f['DEC']


    pointing_sca_coords = SkyCoord(allRA*u.deg, allDEC*u.deg, frame='icrs')
    search_coord = SkyCoord(RA*u.deg, DEC*u.deg, frame='icrs')
    dist = pointing_sca_coords.separation(search_coord).arcsec

    dist[np.where(f['filter'] != filt)] = np.inf #Ensuring we only get the filter we want
    reshaped_array = dist.flatten()
    # Find the indices of the minimum values along the flattened slices
    min_indices = np.argmin(reshaped_array, axis=0)
    # Convert the flat indices back to 2D coordinates
    rows, cols = np.unravel_index(min_indices, dist.shape[:2])

    #The plus 1 is because the SCA numbering starts at 1
    return rows, cols + 1

def construct_psf_source(x, y, pointing, SCA, stampsize=25,  x_center = None, y_center = None, sed = None, flux = 1, photOps = True):
    '''
        Constructs the PSF around the point source (x,y) location, allowing for some offset from the center
        Inputs:
        x,y are locations in the SCA
        pointing, SCA: the pointing and SCA of the image
        stampsize = size of cutout image used
        x_center and y_center need to be given in coordinates of the cutout.
        sed: the SED of the source
        flux: If you are using this function to build a model grid point, this should be 1. If
            you are using this function to build a model of a source, this should be the flux of the source.

    '''

    Lager.debug(f'ARGS IN PSF SOURCE: \n x, y: {x, y} \n' +
                f' Pointing, SCA: {pointing, SCA} \n' +
                f' stamp size: {stampsize} \n' +
                f' x_center, y_center: {x_center, y_center} \n' +
                f' sed: {sed} \n' +
                f' flux: {flux}')

    config_file = './temp_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit=pointing, sca=SCA)

    assert sed is not None, 'You must provide an SED for the source'

    if not photOps:
        # While I want to do this sometimes, it is very rare that you actually
        # want to do this. Thus if it was accidentally on while doing a normal
        # run, I'd want to know.
        Lager.warning('NOT USING PHOTON OPS IN PSF SOURCE')

    master = getPSF_Image(util_ref, stampsize, x=x, y=y,  x_center=x_center,
                          y_center=y_center, sed=sed,
                          include_photonOps=photOps, flux=flux).array

    return master.flatten()


def gaussian(x, A, mu, sigma):
    '''
    See name of function. :D
    '''
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def constructImages(exposures, ra, dec, size=7, subtract_background=True,
                    roman_path=None, truth='simple_model'):

    '''
    Constructs the array of Roman images in the format required for the linear
    algebra operations

    Inputs:
    exposures is a list of exposures from findAllExposures
    ra,dec: the RA and DEC of the SN
    subtract_background: If False, the background level is fit as a free
        parameter in the forward modelling. Otherwise, we subtract it here.
    roman_path: the path to the Roman data

    Returns:
    cutout_image_list: list of snappl.image.Image objects, cutouts on the
                       object location.
    image_list: list of snappl.image.Image objects of the entire SCA.

    '''

    bgflux = []
    image_list = []
    cutout_image_list = []

    Lager.debug(f'truth in construct images: {truth}')

    for indx, i in enumerate(exposures):
        Lager.debug(f'Constructing image {indx} of {len(exposures)}')
        band = i['BAND']
        pointing = i['Pointing']
        SCA = i['SCA']

        # TODO : replace None with the right thing once Exposure is implemented

        imagepath = roman_path + (f'/RomanTDS/images/{truth}/{band}/{pointing}'
                                  f'/Roman_TDS_{truth}_{band}_{pointing}_'
                                  f'{SCA}.fits.gz')
        image = OpenUniverse2024FITSImage(imagepath, None, SCA)
        imagedata, errordata, flags = image.get_data(which='all')
        image_cutout = image.get_ra_dec_cutout(ra, dec, size)
        if truth == 'truth':
            raise RuntimeError("Truth is broken.")
            # In the future, I'd like to manually insert an array of ones for
            # the error, or something.

        '''
        try:
            zero = np.power(10, -(i['zeropoint'] - self.common_zpt)/2.5)
        except:
            zero = -99

        if zero < 0:
            zero =
        im = cutout * zero
        '''

        # If we are not fitting the background we subtract it here.
        # When subtract_background is False, we are including the background
        # level as a free parameter in our fit, so it should not be subtracted
        # here.
        bg = 0
        if subtract_background:
            if not truth == 'truth':
                # However, if we are subtracting the background, we want to get
                # rid of it here, either by reading the SKY_MEAN value from the
                # image header...
                bg = image_cutout._get_header()['SKY_MEAN']
            elif truth == 'truth':
                # ....or manually calculating it!
                bg = calculate_background_level(imagedata)

        bgflux.append(bg)  # This currently isn't returned, but might be a good
        # thing to put in output? TODO

        image_cutout._data -= bg
        Lager.debug(f'Subtracted a background level of {bg}')

        image_list.append(image)
        cutout_image_list.append(image_cutout)
        Lager.debug('image type:')
        Lager.debug(type(image))

    return cutout_image_list, image_list


def calculate_background_level(im):
    '''
    A function for naively estimating the background level from a given image.
    This may be replaced by a more sophisticated function later.
    For now, we take the corners of the image, sigma clip, and then return
    the median as the background level.

    Inputs:
    im, numpy array of floats, the image to be used.

    Returns:
    bg, float, the estimated background level.

    '''
    size = im.shape[0]
    bgarr = np.concatenate((im[0:size//4, 0:size//4].flatten(),
                            im[0:size, 3*(size//4):size].flatten(),
                            im[3*(size//4):size, 0:size//4].flatten(),
                            im[3*(size//4):size, 3*(size//4):size].flatten()))
    if len(bgarr) == 0:
        bg = 0
    else:
        pc = np.percentile(bgarr, 84)
        bgarr = bgarr[bgarr < pc]
        bg = np.median(bgarr)

    return bg


def getPSF_Image(self,stamp_size,x=None,y=None, x_center = None, y_center= None, pupil_bin=8,sed=None,
                        oversampling_factor=1,include_photonOps=False,n_phot=1e6, pixel = False, flux = 1):

    """
    This is a roman imsim function that I have repurposed slightly for off center placement.

    Return a Roman PSF image for some image position
    Parameters:
        stamp_size: size of output PSF model stamp in native roman pixel_scale (oversampling_factor=1)
        x: x-position in SCA
        y: y-position in SCA
        pupil_bin: pupil image binning factor
        sed: SED to be used to draw the PSF - default is a flat SED.
        oversampling_factor: factor by which to oversample native roman pixel_scale
        include_photonOps: include additional contributions from other photon operators in effective psf image
    Returns:
        the PSF GalSim image object (use image.array to get a numpy array representation)
    """
    time1 = time.time()
    '''
    if sed is None:
        sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
    '''
    if pixel:
        point = galsim.Pixel(1)*sed
        Lager.debug('Building a Pixel shaped PSF source')
    else:
        point = galsim.DeltaFunction()*sed
    time2 = time.time()

    point = point.withFlux(flux,self.bpass)
    local_wcs = self.getLocalWCS(x,y)
    wcs = galsim.JacobianWCS(dudx=local_wcs.dudx/oversampling_factor,
                                dudy=local_wcs.dudy/oversampling_factor,
                                dvdx=local_wcs.dvdx/oversampling_factor,
                                dvdy=local_wcs.dvdy/oversampling_factor)
    stamp = galsim.Image(stamp_size*oversampling_factor,stamp_size*oversampling_factor,wcs=wcs)

    time3 = time.time()

    if not include_photonOps:
        psf = galsim.Convolve(point, self.getPSF(x,y,pupil_bin))
        return psf.drawImage(self.bpass,image=stamp,wcs=wcs,method='no_pixel',center = galsim.PositionD(x_center, y_center),use_true_center = True)

    photon_ops = [self.getPSF(x,y,pupil_bin)] + self.photon_ops
    Lager.debug(f'Using {n_phot:e} photons in getPSF_Image')
    result = point.drawImage(self.bpass,wcs=wcs, method='phot', photon_ops=photon_ops, rng=self.rng, \
        n_photons=int(n_phot),maxN=int(n_phot),poisson_flux=False, center = galsim.PositionD(x_center, y_center),use_true_center = True, image=stamp)
    return result


def fetchImages(num_total_images, num_detect_images, ID, sn_path, band, size, subtract_background,
                roman_path, object_type, lc_start=-np.inf, lc_end=np.inf):
    '''
    This function gets the list of exposures to be used for the analysis.

    Inputs:
    num_total_images: total images used in analysis (detection + no detection)
    num_detect_images: number of images used in the analysis that contain a
                       detection.
    ID: int, the ID of the object
    sn_path: str, the path to the supernova data
    band: str, the band to be used
    size: int, cutout will be of shape (size, size)
    subtract_background: If True, subtract sky bg from images. If false, leave
            bg as a free parameter in the forward modelling.
    roman_path: str, the path to the Roman data
    obj_type: str, the type of object to be used (SN or star)
    lc_start, lc_end: ints, MJD bounds on where to fetch images.

    Returns:
    images: array, the actual image data, shape (num_total_images, size, size)
    cutout_wcs_list: list of wcs objects for the cutouts
    im_wcs_list: list of wcs objects for the entire SCA
    err: array, the uncertainty in each pixel
                of images, shape (num_total_images, size, size)
    snra, sndec: floats, the RA and DEC of the supernova, a single float is
                         used for both of these as we assume the object is
                         not moving between exposures.
    exposures: astropy.table.table.Table, table of exposures used

    '''

    pqfile = find_parq(ID, sn_path, obj_type=object_type)
    ra, dec, p, s, start, end, peak = \
            get_object_info(ID, pqfile, band = band, snpath = sn_path, roman_path = roman_path, obj_type = object_type)
    snra = ra
    sndec = dec # Why is this here? TODO remove in a less urgent PR
    start = start[0]
    end = end[0]
    exposures = findAllExposures(ID, ra, dec, peak, start, end,
                                 roman_path=roman_path, maxbg=num_total_images - num_detect_images,
                                 maxdet=num_detect_images, return_list=True, band=band,
                                 lc_start=lc_start, lc_end=lc_end)
    cutout_image_list, image_list =\
        constructImages(exposures, ra, dec, size=size,
                        subtract_background=subtract_background,
                        roman_path=roman_path)

    # THIS IS TEMPORARY. In this PR, I am refactoring constructImages to return
    # Image objects. However, the rest of the code is not refactored yet. This
    # returns the Image objects back into the numpy arrays that the rest of the
    # code understands.

    images = []
    cutout_wcs_list = []
    im_wcs_list = []
    err = []
    for cutout, image in zip(cutout_image_list, image_list):
        images.append(cutout._data)
        cutout_wcs_list.append(galsim.AstropyWCS(wcs=cutout._wcs))
        im_wcs_list.append(galsim.AstropyWCS(wcs=image._wcs))
        err.append(cutout._noise)

    ########################### END TEMPORARY SECTION #########################

    return images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, \
           exposures


def get_object_info(ID, parq, band, snpath, roman_path, obj_type):

    '''
    Fetch some info about an object given its ID.
    Inputs:
    ID: the ID of the object
    parq: the parquet file containing the object
    band: the band to consider
    date: whether to return the start end and peak dates of the object
    snpath: the path to the supernova data
    roman_path: the path to the Roman data
    host: whether to return the host RA and DEC

    Returns:
    ra, dec: the RA and DEC of the object
    pointing, sca: the pointing and SCA of the object
    start, end, peak: the start, end, and peak dates of the object
    '''

    df = open_parq(parq, snpath, obj_type = obj_type)
    if obj_type == 'star':
        ID = str(ID)


    df = df.loc[df.id == ID]
    ra, dec = df.ra.values[0], df.dec.values[0]

    if obj_type == 'SN':
        start = df.start_mjd.values
        end = df.end_mjd.values
        peak = df.peak_mjd.values
    else:
        start = [0]
        end = [np.inf]
        peak = [0]

    pointing, sca = radec2point(ra, dec, band, roman_path)

    return ra, dec, pointing, sca, start, end, peak


def getWeights(cutout_wcs_list, size, snra, sndec, error=None,
               gaussian_std=1000, cutoff=np.inf):
    wgt_matrix = []
    Lager.debug(f'Gaussian std in getWeights {gaussian_std}')
    for i, wcs in enumerate(cutout_wcs_list):
        xx, yy = np.meshgrid(np.arange(0, size, 1), np.arange(0, size, 1))
        xx = xx.flatten()
        yy = yy.flatten()

        rara, decdec = wcs.toWorld(xx, yy, units='deg')
        dist = np.sqrt((rara - snra)**2 + (decdec - sndec)**2)

        snx, sny = wcs.toImage(snra, sndec, units='deg')
        dist = np.sqrt((xx - snx + 1)**2 + (yy - sny + 1)**2)

        wgt = np.ones(size**2)
        wgt = 5*np.exp(-dist**2/gaussian_std)
        # Here, we throw out pixels that are more than 4 pixels away from the
        # SN. The reason we do this is because by choosing an image size one
        # has set a square top hat function centered on the SN. When that image
        # is rotated pixels in the corners leave the image, and new pixels
        # enter. By making a circular cutout, we minimize this problem. Of
        # course this is not a perfect solution, because the pixellation of the
        # circle means that still some pixels will enter and leave, but it
        # seems to minimize the problem.
        wgt[np.where(dist > 4)] = 0 # Correction here for flux missed ??? TODO
        if error is None:
            error = np.ones_like(wgt)
        Lager.debug(f'wgt before: {np.mean(wgt)}')
        wgt /= (error[i].flatten())**2 # Define an inv variance TODO
        Lager.debug(f'wgt after: {np.mean(wgt)}')
        # wgt = wgt / np.sum(wgt) # Normalize outside out of the loop TODO
        # What fraction of the flux is contained in the PSF? TODO
        wgt_matrix.append(wgt)
    return wgt_matrix


def makeGrid(grid_type, images, size, ra, dec, cutout_wcs_list,
             percentiles=[],
             make_exact=False):
    '''
    This is a function that returns the locations for the model grid points
    used to model the background galaxy. There are several different methods
    for building the grid, listed below, and this parent function calls the
    correct function for which type of grid you wish to construct.

    TODO: refactor

    Inputs:
    grid_type: str, type of grid method to use.
              regular: A regularly spaced grid.
              adaptive: Points are placed in the image based on the brightness
                        in each pixel.
              contour: Points are placed by placing finer and finer regularly
                        spaced grids in different contour levels of a linear
                        interpolation of the image. See make_contour_grid for
                        a more detailed explanation.
              single: Place a single grid point. This is for sanity checking
                      that the algroithm is drawing points where expected.

    Returns:
    ra_grid, dec_grid: numpy arrays of floats of the ra and dec locations for
                    model grid points.
    '''
    if grid_type == 'contour':
        ra_grid, dec_grid = make_contour_grid(images[0], cutout_wcs_list[0])

    elif grid_type == 'adaptive':
        ra_grid, dec_grid = make_adaptive_grid(ra, dec, cutout_wcs_list[0],
                                               image=images[0],
                                               percentiles=percentiles)
    elif grid_type == 'regular':
        ra_grid, dec_grid = make_regular_grid(ra, dec, cutout_wcs_list[0],
                                              size=size, spacing=0.75)

    if grid_type == 'single':
        ra_grid, dec_grid = [ra], [dec]

    if make_exact:
        if grid_type == 'single':
            galra = ra_grid[0]
            galdec = dec_grid[0]
        else:
            raise NotImplementedError
            # I need to figure out how to turn the single grid point test
            # into a test function.
            galra = ra_grid[106]
            galdec = dec_grid[106]

    ra_grid = np.array(ra_grid)
    dec_grid = np.array(dec_grid)
    return ra_grid, dec_grid


def plot_lc(filepath, return_data=False):
    fluxdata = pd.read_csv(filepath, comment='#', delimiter=' ')
    truth_mags = fluxdata['SIM_true_mag']
    mag = fluxdata['mag']
    sigma_mag = fluxdata['mag_err']

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    dates = fluxdata['MJD']

    plt.scatter(dates, truth_mags, color='k', label='Truth')
    plt.errorbar(dates, mag, yerr=sigma_mag,  color='purple', label='Model',
                 fmt='o')

    plt.ylim(np.max(truth_mags) + 0.2, np.min(truth_mags) - 0.2)
    plt.ylabel('Magnitude (Uncalibrated)')

    residuals = mag - truth_mags
    bias = np.mean(residuals)
    bias *= 1000
    bias = np.round(bias, 3)
    scatter = np.std(residuals)
    scatter *= 1000
    scatter = np.round(scatter, 3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = 'Overall Bias: ' + str(bias) + ' mmag \n' + \
        'Overall Scatter: ' + str(scatter) + ' mmag'
    plt.text(np.percentile(dates, 60), np.mean(truth_mags), textstr,
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
            sigma_mag.values, truth_mags.values, bias, scatter


def plot_images(fileroot, size = 11):

    imgdata = np.load('./results/images/'+str(fileroot)+'_images.npy')
    num_total_images = imgdata.shape[1]//size**2
    images = imgdata[0]
    sumimages = imgdata[1]
    wgt_matrix = imgdata[2]

    fluxdata = pd.read_csv('./results/lightcurves/'+str(fileroot)+'_lc.csv')
    supernova = fluxdata['true_flux']
    measured_flux = fluxdata['measured_flux']

    snra, sndec = fluxdata['sn_ra'][0], fluxdata['sn_dec'][0]
    galra, galdec = fluxdata['host_ra'][0], fluxdata['host_dec'][0]


    hdul = fits.open('./results/images/'+str(fileroot)+'_wcs.fits')
    cutout_wcs_list = []
    for i,savedwcs in enumerate(hdul):
        if i == 0:
            continue
        newwcs = galsim.wcs.readFromFitsHeader(savedwcs.header)[0]
        cutout_wcs_list.append(newwcs)

    biases = []

    ra_grid, dec_grid, gridvals = np.load('./results/images/'+str(fileroot)+'_grid.npy')

    fig = plt.figure(figsize = (15,3*num_total_images))

    for i, wcs in enumerate(cutout_wcs_list):

        extent = [-0.5, size-0.5, -0.5, size-0.5]
        xx, yy = cutout_wcs_list[i].toImage(ra_grid, dec_grid, units = 'deg')
        snx, sny = wcs.toImage(snra, sndec, units = 'deg')
        galx, galy = wcs.toImage(galra, galdec, units = 'deg')

        plt.subplot(len(cutout_wcs_list), 4, 4*i+1)
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx-1, yy-1, s = 1, c= 'k', vmin = vmin, vmax = vmax)
        plt.title('True Image')
        plt.scatter(snx-1, sny-1, c = 'r', s = 8, marker = '*')
        plt.scatter(galx-1,galy-1, c = 'b', s = 8, marker = '*')
        imshow = plt.imshow(images[i*size**2:(i+1)*size**2].reshape(size,size), origin = 'lower', extent = extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        trueimage = images[i*size**2:(i+1)*size**2].reshape(size,size)


        ############################################

        plt.subplot(len(cutout_wcs_list), 4, 4*i+2)
        plt.title('Model')

        im1 = sumimages[i*size**2:(i+1)*size**2].reshape(size,size)
        xx, yy = cutout_wcs_list[i].toImage(ra_grid, dec_grid, units = 'deg')


        xx -= 1
        yy -= 1


        vmin = np.min(images[i*size**2:(i+1)*size**2].reshape(size,size))
        vmax = np.max(images[i*size**2:(i+1)*size**2].reshape(size,size))

        #im1[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) == 0)] = 0


        vmin = imshow.get_clim()[0]
        vmax = imshow.get_clim()[1]

        plt.imshow(im1, extent = extent, origin = 'lower', vmin = vmin, vmax = vmax)
        plt.colorbar(fraction=0.046, pad=0.04)


        #plt.scatter(galx-1,galy-1, c = 'r', s = 8, marker = '*')
        #plt.scatter(snx-1, sny-1, c = 'k', s = 8, marker = '*')

        #plt.xlim(-1,size)
        #plt.ylim(-1,size)


        ############################################
        plt.subplot(len(cutout_wcs_list),4,4*i+3)
        plt.title('Residuals')
        vmin = np.mean(gridvals) - np.std(gridvals)
        vmax = np.mean(gridvals) + np.std(gridvals)
        plt.scatter(xx,yy, s = 1, c= gridvals,  vmin = vmin, vmax = vmax)
        res = images - sumimages



        current_res= res[i*size**2:(i+1)*size**2].reshape(size,size)

        #if i == 0:
        norm = 3*np.std(current_res[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) != 0)])

        #current_res[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) == 0)] = 0
        #current_res[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) != 0)] = \
            #np.log10(np.abs(current_res[np.where(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size) != 0)]))

        plt.imshow(current_res, extent = extent, origin = 'lower', cmap = 'seismic', vmin = -100, vmax = 100)
        #plt.imshow(wgt_matrix[i*size**2:(i+1)*size**2].reshape(size,size), extent = extent, origin = 'lower')
        plt.colorbar(fraction=0.046, pad=0.14)
        #plt.scatter(galx,galy, c = 'r', s = 12, marker = '*', edgecolors='k')

    plt.subplots_adjust(wspace = 0.4, hspace = 0.3)


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
        newwcs = galsim.wcs.readFromFitsHeader(savedwcs.header)[0]
        cutout_wcs_list.append(newwcs)


    magresiduals = -2.5*np.log10(measured_flux)+2.5*np.log10(np.array(supernova))


    galxes = []
    stds = []
    biases = []

    for i, wcs in enumerate(cutout_wcs_list):

        extent = [-0.5, size-0.5, -0.5, size-0.5]
        trueimage = images[i*size**2:(i+1)*size**2].reshape(size,size)
        snx, sny = wcs.toImage(snra, sndec, units = 'deg')

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


        #plt.plot(trueimage[5], label = 'Image')

        plt.axhline(0, ls = '--', color = 'k')
        #plt.plot(im1[5], label = 'Model', lw = 3)
        plt.plot(trueimage[5] - im1[5], label = 'Im-Model', alpha = 0.4)
        plt.ylim(-250,250)


        if i >= num_total_images - num_detect_images:
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*supernova[i - num_total_images + num_detect_images]
            plt.plot(snim[5], label = 'True SN', lw = 3)
            plt.fill_between(np.arange(0,11,1), trueimage[5] - snim[5] + 50, trueimage[5] - snim[5] - 50, label = 'Im-True SN', alpha = 0.4)
            plt.plot(np.arange(0,11,1), trueimage[5] - snim[5] , color = 'k', ls = '--')
            plt.plot(justbgim[5], label = 'BGModel')
            plt.plot(justbgres[5], label = 'Im-BGModel')

            #plt.plot(justbgres[5] - snim[5], label = 'SN Residuals', ls = '--')
            plt.ylim(-500,np.max(trueimage[5]))
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*X[-num_detect_images:][i - num_total_images + num_detect_images]

        else:
            snim = np.zeros_like(justbgres)



        plt.axvline(snx-1+4, ls = '--', color = 'k')
        plt.axvline(snx-1-4, ls = '--', color = 'k')
        plt.axvline(snx-1, ls = '--', color = 'r')

        plt.xlim(snx-1-3.8, snx-1+3.8)


        plt.legend(loc = 'upper left')


def get_galsim_SED(SNID, date, sn_path, fetch_SED, obj_type = 'SN'):
    '''
    Return the appropriate SED for the object on the day. Since SN's SEDs are
    time dependent but stars are not, we need to handle them differently.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data
    fetch_SED: If true, fetch true SED from the database, otherwise return a
                flat SED.
    obj_type: the type of object (SN or star)

    Internal Variables:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom

    Returns:
    sed: galsim.SED object
    '''
    if fetch_SED == True:
        if obj_type == 'SN':
            lam, flambda = get_SN_SED(SNID, date, sn_path)
        if obj_type == 'star':
            lam, flambda = get_star_SED(SNID, sn_path)
    else:
        lam, flambda = [1000, 26000], [1, 1]

    sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant='linear'),
                         wave_type='Angstrom', flux_type='fphotons')

    return sed


def get_star_SED(SNID, sn_path):
    '''
    Return the appropriate SED for the star.
    Inputs:
    SNID: the ID of the object
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom (numpy  array of floats)
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
             (numpy array of floats)
    '''
    filenum = find_parq(SNID, sn_path, obj_type = 'star')
    pqfile = open_parq(filenum, sn_path, obj_type = 'star')
    file_name = pqfile[pqfile['id'] == str(SNID)]['sed_filepath'].values[0]
    #THIS HARDCODE WILL NEED TO BE REMOVED
    #Make hardcodes keyword args until they are fixed
    fullpath = os.path.join('/hpc/home/cfm37/rubin_sim_data/sims_sed_library/', file_name)
    sed_table = pd.read_csv(fullpath,  compression='gzip', sep = '\s+', comment = '#')
    lam = sed_table.iloc[:, 0]
    flambda = sed_table.iloc[:, 1]
    return np.array(lam), np.array(flambda)


def get_SN_SED(SNID, date, sn_path):
    '''
    Return the appropriate SED for the supernova on the given day.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
    '''
    filenum = find_parq(SNID, sn_path, obj_type = 'SN')
    file_name = 'snana' + '_' + str(filenum) + '.hdf5'
    fullpath = os.path.join(sn_path, file_name)
    sed_table = h5py.File(fullpath, 'r')
    sed_table = sed_table[str(SNID)]
    flambda = sed_table['flambda']
    lam = sed_table['lambda']
    mjd = sed_table['mjd']
    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    max_days_cutoff = 10
    closest_days_away = np.min(np.abs(np.array(mjd) - date))

    if closest_days_away > max_days_cutoff:
        Lager.warning(f'WARNING: No SED data within {max_days_cutoff} days of' +
                   'date. \n The closest SED is ' + closest_days_away +
                   ' days away.')
    return np.array(lam), np.array(flambda[bestindex])


def make_contour_grid(image, wcs, numlevels = None, percentiles = [0, 90, 98, 100],
                subsize = 4):
    '''
    Construct a "contour grid" which allocates model grid points to model
    the background galaxy according to the brightness of the image. This is
    an alternate version of make_adaptive_grid that results in a more
    continuous model grid point layout than make_adaptive_grid.
    While make_adaptive_grid visits each pixel and places a certain number of
    points, this function creates a smooth interpolation of the image to choose
    model point locations more densely in brighter regions.

    It does this as follows:
        1. Create a linear interoplation of the image.
        Start a loop:
        2. Create a grid of points that are evenly spaced in pixel space.
        3. For each of these points, check which brightness bin they fall into,
           using the linear interpolation.
        4. If this point is in the correct brightness bin, add it to the grid.
            If not, it does not get added.
        5. Increase the point density, and move to the next higher brightness
            bin.

    Here's a schematic:
    Our Image:  Binned by brightness:
                          ───────          ·····              ·····
            ░░░░░░        │     │          ·   ·              ·:::·
            ░▒▒▒▒░        │ ┌─┐ │          ·   ·              ·:::·
            ░▒██▒░        │ │ │ │          ·   ·              ·:::·
            ░▒██▒░        │ └─┘ │          ·   ·              ·:::·
            ░▒▒▒▒░        │     │          ·   ·              ·:::·
            ░░░░░░        │     │          ·····              ·····
                          ───────            ^                 ^
                            Add sparse model points, then dense model points.


    This model allows for the grid density to change smoothly across pixels,
    and avoids the problem of awkward gaps between model points across pixels.

    Inputs:
    image: 2D numpy array of floats of shape (size x size), the image to build
    the grid on.
    wcs: the WCS of the image, currently a galsim.fitswcs.AstropyWCS object

    percentiles: list of floats, the percentiles to use to bin the image. The
                more bins, the more possible grid points could be placed in
                that pixel.

    subsize: Int, width of the grid in pixels.
             Specify the width of the grid, which can be smaller than the
             image. For instance I could have an image that is 11x11 but a grid
             that is only 9x9.
             This is useful and different from making a smaller image because
             when the image rotates, model points near the corners of the image
             may be rotated out. By taking a smaller grid, we can avoid this.

    Returns:
    ra_grid, dec_grid: 1D numpy arrays of floats, the RA and DEC of the grid.
    '''
    size = image.shape[0]
    x = np.arange(0,size,1.0)
    y = np.arange(0,size,1.0)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    xg = xg.ravel()
    yg = yg.ravel()
    Lager.debug('Grid type: contour')

    if numlevels is not None:
        levels = list(np.linspace(np.min(image), np.max(image), numlevels))
    else:
        levels = list(np.percentile(image, percentiles))

    Lager.debug(f'Using levels: {levels} in make_contour_grid')

    interp = RegularGridInterpolator((x, y), image, method='linear',
                                 bounds_error=False, fill_value=None)

    aa = interp((xg,yg))

    x_totalgrid = []
    y_totalgrid = []

    for i in range(len(levels) - 1):
        zmin = levels[i]
        zmax = levels[i+1]
        # Generate a grid that gets finer each iteration of the loop. For
        # instance, in brightness bin 1, 1 point per pixel, in brightness bin
        # 2, 4 points per pixel (2 in each direction), etc.
        x = np.arange(0,size,1/(i+1))
        y = np.arange(0,size,1/(i+1))
        if i == 0:
            x = x[np.where(np.abs(x - size/2) < subsize)]
            y = y[np.where(np.abs(y - size/2) < subsize)]
        xg, yg = np.meshgrid(x, y, indexing='ij')
        aa = interp((xg,yg))
        xg = xg[np.where((aa > zmin) & (aa <= zmax))]
        yg = yg[np.where((aa > zmin) & (aa <= zmax))]
        x_totalgrid.extend(xg)
        y_totalgrid.extend(yg)

    xx, yy = y_totalgrid, x_totalgrid # Here is another place I need to flip
    # x and y. I'd like this to be more rigorous or at least clear.
    xx = np.array(xx)
    yy = np.array(yy)
    xx = xx.flatten()
    yy = yy.flatten()
    Lager.debug(f'Built a grid with {np.size(xx)} points')

    result = wcs.toWorld(xx, yy, units='deg')
    ra_grid = result[0]
    dec_grid = result[1]

    return ra_grid, dec_grid


def calc_mags_and_err(flux, sigma_flux, band, zp = None):
    exptime = {'F184': 901.175,
            'J129': 302.275,
            'H158': 302.275,
            'K213': 901.175,
            'R062': 161.025,
            'Y106': 302.275,
            'Z087': 101.7}

    area_eff = roman.collecting_area
    zp = roman.getBandpasses()[band].zeropoint if zp is None else zp
    mags = -2.5*np.log10(flux) + 2.5*np.log10(exptime[band]*area_eff) + zp
    magerr = 2.5 * sigma_flux / (flux * np.log(10))
    return mags, magerr


def build_lightcurve(ID, exposures, sn_path, confusion_metric, flux,
                     use_roman, band, object_type, sigma_flux):

    '''
    This code builds a lightcurve datatable from the output of the SMP algorithm.

    Input:
    ID (int): supernova ID
    exposures (table): table of exposures used in the SMP algorithm
    sn_path (str): path to supernova data
    confusion_metric (float): the confusion metric derived in the SMP algorithm
    num_detect_images (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm
    use_roman (bool): whether or not the lightcurve was built using Roman PSF
    band (str): the bandpass of the images used

    Returns:
    lc: a QTable containing the lightcurve data
    '''

    detections = exposures[np.where(exposures['DETECTED'])]
    parq_file = find_parq(ID, path = sn_path, obj_type = object_type)
    df = open_parq(parq_file, path = sn_path, obj_type = object_type)

    mags, magerr = calc_mags_and_err(flux, sigma_flux, band)
    sim_sigma_flux = 0 # These are truth values!
    sim_realized_mags, _ = calc_mags_and_err(detections['realized flux'],
                                             sim_sigma_flux, band)
    sim_true_mags, _ = calc_mags_and_err(detections['true flux'],
                                         sim_sigma_flux, band)
    if object_type == 'SN':
        df_object_row = df.loc[df.id == ID]
    if object_type == 'star':
        df_object_row = df.loc[df.id == str(ID)]

    if object_type == 'SN':
        meta_dict ={'confusion_metric': confusion_metric, \
        'host_sep': df_object_row['host_sn_sep'].values[0],\
        'host_mag_g': df_object_row[f'host_mag_g'].values[0],\
        'sn_ra': df_object_row['ra'].values[0], \
        'sn_dec': df_object_row['dec'].values[0], \
        'host_ra': df_object_row['host_ra'].values[0],\
        'host_dec': df_object_row['host_dec'].values[0]}
    else:
        meta_dict = {'ra': df_object_row['ra'].values[0],
                     'dec': df_object_row['dec'].values[0]}

    data_dict = {'MJD': detections['date'], 'flux': flux,
                 'flux_error': sigma_flux, 'mag': mags,
                 'mag_err': magerr,
                 'SIM_realized_flux': detections['realized flux'],
                 'SIM_true_flux': detections['true flux'],
                 'SIM_realized_mag': sim_realized_mags,
                 'SIM_true_mag': sim_true_mags,}
    units = {'MJD':u.d, 'SIM_realized_flux': '',  'flux': '',
             'flux_error': '', 'SIM_realized_mag': '',
              'SIM_true_flux': '', 'SIM_true_mag': ''}

    return QTable(data = data_dict, meta = meta_dict, units = units)


def build_lightcurve_sim(supernova, flux, sigma_flux):
    '''
    This code builds a lightcurve datatable from the output of the SMP algorithm
    if the user simulated their own lightcurve.

    Inputs
    supernova (array): the true lightcurve
    num_detect_images (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm

    Returns
    lc: a QTable containing the lightcurve data
    2.) Soon I will turn many of these inputs into environment variable and they
    should be deleted from function arguments and docstring.
    '''
    sim_MJD = np.arange(0, num_detect_images, 1)
    data_dict = {'MJD': sim_MJD, 'flux': flux,
                 'flux_error': sigma_flux, 'SIM_flux': supernova}
    meta_dict = {}
    units = {'MJD':u.d, 'SIM_flux': '',  'flux': '', 'flux_error':''}
    return QTable(data = data_dict, meta = meta_dict, units = units)

def save_lightcurve(lc,identifier, band, psftype, output_path = None,
                    overwrite = True):
    '''
    This function parses settings in the SMP algorithm and saves the lightcurve
    to an ecsv file with an appropriate name.
    Input:
    lc: the lightcurve data
    identifier (str): the supernova ID or 'simulated'
    band (str): the bandpass of the images used
    psftype (str): 'romanpsf' or 'analyticpsf'
    output_path (str): the path to save the lightcurve to.

    Returns:
    None, saves the lightcurve to a ecsv file.
    The file name is:
    output_path/identifier_band_psftype_lc.ecsv
    '''

    if not os.path.exists(os.path.join(os.getcwd(), 'results/')):
            Lager.info('Making a results directory for output at ',
                       os.getcwd(), '/results')
            os.makedirs(os.path.join(os.getcwd(), 'results/'))
            os.makedirs(os.path.join(os.getcwd(), 'results/images/'))
            os.makedirs(os.path.join(os.getcwd(), 'results/lightcurves/'))

    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'results/lightcurves/')

    lc_file = os.path.join(output_path,
                           f'{identifier}_{band}_{psftype}_lc.ecsv')

    Lager.info(f'Saving lightcurve to {lc_file}')
    lc.write(lc_file, format = 'ascii.ecsv', overwrite = overwrite)

def banner(text):
    length = len(text) + 8
    message = "\n" + "#" * length +'\n'+'#   ' + text + '   # \n'+ "#" * length
    Lager.debug(message)


def get_galsim_SED_list(ID, exposures, fetch_SED, object_type, sn_path):
    sedlist = []
    '''
    Return the appropriate SED for the object for each observation.
    If you are getting truth SEDs, this function calls get_SED on each exposure
    of the object. Then, get_SED calls get_SN_SED or get_star_SED depending on
    the object type.
    If you are not getting truth SEDs, this function returns a flat SED for
    each exposure.

    Inputs:
    ID: the ID of the object
    exposures: the exposure table returned by fetchImages.
    fetch_SED: If true, get the SED from truth tables.
               If false, return a flat SED for each expsoure.
    object_type: the type of object (SN or star)
    sn_path: the path to the supernova data

    Returns:
    sedlist: list of galsim SED objects, length equal to the number of
             detection images.
    '''
    for date in exposures['date'][exposures['DETECTED']]:
        sed = get_galsim_SED(ID, date, sn_path, obj_type=object_type,
                                 fetch_SED=fetch_SED)
        sedlist.append(sed)

    return sedlist


def prep_data_for_fit(images, err, sn_matrix, wgt_matrix):
    '''
    This function takes the data from the images and puts it into the form such
    that we can analytically solve for the best fit using linear algebra.

    n = total number of images
    s = image size (so the image is s x s)
    d = number of detection images

    Inputs:
    images: list of np arrays of image data. List of length n of sxs arrays.
    err: list of np arrays of error data. List of length n of sxs arrays.
    sn_matrix: list of np arrays of SN models. List of length d of sxs arrays.
    wgt_matrix: list of np arrays of weights. List of length n of sxs arrays.

    Outputs:
    images: 1D array of image data. Length n*s^2
    err: 1D array of error data. Length n*s^2
    sn_matrix: A 2D array of SN models, with the SN models placed in the
                correct rows and columns, see comment below. Shape (n*s^2, n)
    wgt_matrix: 1D array of weights. Length n*s^2
    '''
    size_sq = int((images[0].size))
    tot_num = len(images)
    det_num = len(sn_matrix)

    # Flatten into 1D arrays
    images = np.concatenate([arr.flatten() for arr in images])
    err = np.concatenate([arr.flatten() for arr in err])

    # The final design matrix for our fit should have dimensions:
    # (total number of pixels in all images, number of model components)
    # Then, the first s^2 rows of the matrix correspond to the first image,
    # the next s^2 rows to the second image, etc.,  where s is the size of the
    # image. For the SN model, the flux in each image is ostensibly different.
    # Therefore we need a unique flux for each image, and we don't want the
    # flux of the supernova in one image to affect the flux in another image.
    # Therefore, we need to place the supernova model in the correct image
    # (i.e. the correct rows of the design matrix) and zero out all of the
    # others. We'll do this by initializing a matrix of zeros, and then filling
    # in the SN model in the correct place in the loop below:

    psf_zeros = np.zeros((np.size(images), tot_num))
    for i in range(det_num):
        sn_index = tot_num - det_num + i # We only want to edit SN columns.
        psf_zeros[
            (sn_index) * size_sq:  # Fill in rows s^2 * image number...
            (sn_index + 1) * size_sq, #... to s^2 * (image number + 1) ...
            sn_index] = sn_matrix[i] # ...in the correct column.
    sn_matrix = np.vstack(psf_zeros)
    wgt_matrix = np.array(wgt_matrix)
    wgt_matrix = np.hstack(wgt_matrix)

    return images, err, sn_matrix, wgt_matrix


def run_one_object(ID, object_type, num_total_images, num_detect_images, roman_path,
                   sn_path, size, band, fetch_SED, use_real_images, use_roman,
                   subtract_background, turn_grid_off,
                   make_initial_guess, initial_flux_guess, weighting, method,
                   grid_type, pixel, source_phot_ops,
                   lc_start, lc_end, do_xshift, bg_gal_flux, do_rotation, airy,
                   mismatch_seds, deltafcn_profile, noise, check_perfection,
                   avoid_non_linearity, sim_gal_ra_offset, sim_gal_dec_offset,
                   draw_method_for_non_roman_psf = 'no_pixel'):

    Lager.debug(f'ID: {ID}')
    psf_matrix = []
    sn_matrix = []
    cutout_wcs_list = []
    im_wcs_list = []

    # This is a catch for when I'm doing my own simulated WCSs
    util_ref = None

    percentiles = []
    roman_bandpasses = galsim.roman.getBandpasses()

    if use_real_images:
        # Find SN Info, find exposures containing it,
        # and load those as images.
        # TODO: Calculate peak MJD outside of the function
        images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, \
            exposures = fetchImages(num_total_images,
                                                 num_detect_images, ID,
                                                 sn_path, band, size,
                                                 subtract_background,
                                                 roman_path,
                                                 object_type,
                                                 lc_start=lc_start,
                                                 lc_end=lc_end)
        num_predetection_images = exposures[~exposures['DETECTED']]
        if len(num_predetection_images) == 0 and object_type == 'SN':
            Lager.warning('No pre-detection images found in time range ' +
                            'provided, skipping this object.')
            return None

        if num_total_images != np.inf and len(exposures) != num_total_images:
            Lager.warning(f'Not Enough Exposures. \
                Found {len(exposures)} out of {num_total_images} requested')
            return None

        num_total_images = len(exposures)
        num_detect_images = len(exposures[exposures['DETECTED']])
        _ = f'Updating image numbers to {num_total_images} and {num_detect_images}'
        Lager.debug(_)

    else:
        # Simulate the images of the SN and galaxy.
        banner('Simulating Images')
        images, im_wcs_list, cutout_wcs_list, sim_lc, util_ref = \
            simulate_images(num_total_images, num_detect_images, ra, dec,
                            sim_gal_ra_offset, sim_gal_dec_offset,
                            do_xshift, do_rotation, noise=noise,
                            use_roman=use_roman, roman_path=roman_path,
                            size=size, band=band,
                            deltafcn_profile=deltafcn_profile,
                            input_psf=airy, bg_gal_flux=bg_gal_flux,
                            source_phot_ops=source_phot_ops,
                            mismatch_seds=mismatch_seds)
        object_type = 'SN'
        err = np.ones_like(images)

    sedlist = get_galsim_SED_list(ID, exposures, fetch_SED, object_type,
                                  sn_path)

    # Build the background grid
    if not turn_grid_off:
        if object_type == 'star':
            Lager.warning('For fitting stars, you probably dont want a grid.')
        ra_grid, dec_grid = makeGrid(grid_type, images, size, ra, dec,
                                     cutout_wcs_list,
                                     percentiles=percentiles)
    else:
        ra_grid = np.array([])
        dec_grid = np.array([])

    # Using the images, hazard an initial guess.
    # The num_total_images - num_detect_images check is to ensure we have
    # pre-detection images. Otherwise, initializing the model guess does not
    # make sense.
    if make_initial_guess and num_total_images != num_detect_images:
        if num_detect_images != 0:
            x0test = generateGuess(images[:-num_detect_images], cutout_wcs_list,
                                   ra_grid, dec_grid)
            x0_vals_for_sne = np.full(num_total_images, initial_flux_guess)
            x0test = np.concatenate([x0test, x0_vals_for_sne], axis=0)
            print(x0test.shape)
            Lager.debug(f'setting initial guess to {initial_flux_guess}')
        else:
            x0test = generateGuess(images, cutout_wcs_list, ra_grid,
                                   dec_grid)

    else:
        x0test = None

    banner('Building Model')

    # Calculate the Confusion Metric

    confusion_metric = 0
    Lager.debug('Confusion Metric not calculated')

    if use_real_images and object_type == 'SN':
        sed = get_galsim_SED(ID, exposures, sn_path, fetch_SED=False)
        x, y = im_wcs_list[0].toImage(ra, dec, units='deg')
        snx, sny = cutout_wcs_list[0].toImage(snra, sndec, units='deg')
        pointing, SCA = exposures['Pointing'][0], exposures['SCA'][0]
        array = construct_psf_source(x, y, pointing, SCA, stampsize=size,
                                     x_center=snx, y_center=sny, sed=sed)
        confusion_metric = np.dot(images[0].flatten(), array)

        Lager.debug(f'Confusion Metric: {confusion_metric}')
    else:
        confusion_metric = 0
        Lager.debug('Confusion Metric not calculated')

    # Build the backgrounds loop
    # TODO: Zip all the things you index [i] on directly and loop over
    # them.
    for i in range(num_total_images):
        if use_roman:
            sim_psf = galsim.roman.getPSF(1, band, pupil_bin=8,
                                          wcs=cutout_wcs_list[i])
        else:
            sim_psf = airy

        x, y = im_wcs_list[i].toImage(ra, dec, units='deg')

        # Build the model for the background using the correct psf and the
        # grid we made in the previous section.

        # TODO: Put this in snappl
        if use_real_images:
            util_ref = roman_utils(config_file='./temp_tds.yaml',
                                   visit=exposures['Pointing'][i],
                                   sca=exposures['SCA'][i])

        # TODO: better name for array
        # TODO: Why is band here twice?
        array, bgpsf = construct_psf_background(ra_grid, dec_grid,
                                                cutout_wcs_list[i], x, y,
                                                size,
                                                roman_bandpasses[band],
                                                color=0.61, psf=sim_psf,
                                                pixel=pixel,
                                                include_photonOps=False,
                                                util_ref=util_ref,
                                                use_roman=use_roman,
                                                band=band)
        # TODO comment this

        if not subtract_background:
            for j in range(num_total_images):
                if i == j:
                    bg = np.ones(size**2).reshape(-1, 1)
                else:
                    bg = np.zeros(size**2).reshape(-1, 1)
                array = np.concatenate([array, bg], axis=1)

        # Add the array of the model points and the background (if using)
        # to the matrix of all components of the model.
        psf_matrix.append(array)

        # TODO make this not bad
        if num_detect_images != 0 and i >= num_total_images - num_detect_images:
            snx, sny = cutout_wcs_list[i].toImage(snra, sndec, units='deg')
            if use_roman:
                if use_real_images:
                    pointing = exposures['Pointing'][i]
                    SCA = exposures['SCA'][i]
                else:
                    pointing = 662
                    SCA = 11
                # sedlist is the length of the number of supernova
                # detection images. Therefore, when we iterate onto the
                # first supernova image, we want to be on the first element
                # of sedlist. Therefore, we subtract by the number of
                # predetection images: num_total_images - num_detect_images.
                sn_index = i - (num_total_images - num_detect_images)
                Lager.debug(f'Using SED #{sn_index}')
                sed = sedlist[sn_index]
                Lager.debug(f'x, y, snx, sny, {x, y, snx, sny}')
                array = construct_psf_source(x, y, pointing, SCA,
                                             stampsize=size, x_center=snx,
                                             y_center=sny, sed=sed,
                                             photOps=source_phot_ops)
            else:
                stamp = galsim.Image(size, size, wcs=cutout_wcs_list[i])
                profile = galsim.DeltaFunction()*sed
                profile = profile.withFlux(1, roman_bandpasses[band])
                convolved = galsim.Convolve(profile, sim_psf)
                array =\
                     convolved.drawImage(roman_bandpasses[band],
                                        method=draw_method_for_non_roman_psf,
                                        image=stamp,
                                        wcs=cutout_wcs_list[i],
                                        center=(snx, sny),
                                        use_true_center=True,
                                        add_to_image=False)
                array = array.array.flatten()

            sn_matrix.append(array)

    banner('Lin Alg Section')
    psf_matrix = np.vstack(np.array(psf_matrix))
    Lager.debug(f'{psf_matrix.shape} psf matrix shape')

    # Add in the supernova images to the matrix in the appropriate location
    # so that it matches up with the image it represents.
    # All others should be zero.

    # Get the weights
    if weighting:
        wgt_matrix = getWeights(cutout_wcs_list, size, snra, sndec,
                                error=err)
    else:
        wgt_matrix = np.ones(psf_matrix.shape[1])

    images, err, sn_matrix, wgt_matrix =\
        prep_data_for_fit(images, err, sn_matrix, wgt_matrix)

    # Calculate amount of the PSF cut out by setting a distance cap
    test_sn_matrix = np.copy(sn_matrix)
    test_sn_matrix[np.where(wgt_matrix == 0), :] = 0
    Lager.debug(f'SN PSF Norms Pre Distance Cut:{np.sum(sn_matrix, axis=0)}')
    Lager.debug(f'SN PSF Norms Post Distance Cut:{np.sum(test_sn_matrix, axis=0)}')

    # Combine the background model and the supernova model into one matrix.

    psf_matrix = np.hstack([psf_matrix, sn_matrix])

    banner('Solving Photometry')
    # These if statements can definitely be written more elegantly.
    if not make_initial_guess:
        x0test = np.zeros(psf_matrix.shape[1])

    if not subtract_background:
        x0test = np.concatenate([x0test, np.zeros(num_total_images)], axis=0)

    if method == 'lsqr':
        lsqr = sp.linalg.lsqr(psf_matrix*wgt_matrix.reshape(-1, 1),
                              images*wgt_matrix, x0=x0test, atol=1e-12,
                              btol=1e-12, iter_lim=300000, conlim=1e10)
        X, istop, itn, r1norm = lsqr[:4]
        Lager.debug(f'Stop Condition {istop}, iterations: {itn},' +
                    f'r1norm: {r1norm}')
    flux = X[-num_detect_images:]
    inv_cov = psf_matrix.T @ np.diag(wgt_matrix) @ psf_matrix
    Lager.debug(f'inv_cov shape: {inv_cov.shape}')
    Lager.debug(f'psf_matrix shape: {psf_matrix.shape}')
    Lager.debug(f'wgt_matrix shape: {wgt_matrix.shape}')
    try:
        cov = np.linalg.inv(inv_cov)
    except LinAlgError:
        cov = np.linalg.pinv(inv_cov)

    Lager.debug(f'cov diag: {np.diag(cov)[-num_detect_images:]}')
    sigma_flux = np.sqrt(np.diag(cov)[-num_detect_images:])
    Lager.debug(f'sigma flux: {sigma_flux}')

    # Using the values found in the fit, construct the model images.
    pred = X*psf_matrix
    sumimages = np.sum(pred, axis=1)

    # TODO: Move this to a separate function
    if check_perfection:
        if avoid_non_linearity:
            f = 1
        else:
            f = 5000
        if grid_type == 'single':
            X[0] = f
        else:
            X = np.zeros_like(X)
            X[106] = f

    if use_real_images:
        # Eventually I might completely separate out simulated SNe, though I
        # am hesitant to do that as I want them to be treated identically as
        # possible. In the meantime, just return zeros for the simulated lc
        # if we aren't simulating.
        sim_lc = np.zeros(num_detect_images)
    return flux, sigma_flux, images, sumimages, exposures, ra_grid, dec_grid, \
        wgt_matrix, confusion_metric, X, cutout_wcs_list, sim_lc


def plot_image_and_grid(image, wcs, ra_grid, dec_grid):
    Lager.debug(f'WCS: {type(wcs)}')
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    plt.imshow(image, origin='lower', cmap='gray')
    plt.scatter(ra_grid, dec_grid)