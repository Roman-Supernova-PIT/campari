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
from scipy.interpolate import RegularGridInterpolator


from snappl.image import OpenUniverse2024FITSImage
from snappl.logger import Lager

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter('ignore', category=AstropyWarning)
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


def local_grid(ra_center, dec_center, wcs, npoints, size = 25, spacing = 1.0, image = None, spline_grid = True, percentiles = [], makecontourGrid = True):

    '''
    Generates a local grid around a RA-Dec center, choosing step size and
    number of points
    '''
    Lager.debug('image shape: {}'.format(np.shape(image)))
    # Build the basic grid
    subsize = 9  # Taking a smaller square inside the image to fit on
    difference = int((size - subsize)/2)

    x_center, y_center = wcs.toImage(ra_center, dec_center, units='deg')

    if image is None:
        spacing = 0.5
    else:
        spacing = 1.0

    Lager.debug(f'GRID SPACE {spacing}')

    x = np.arange(difference, subsize+difference, spacing)
    y = np.arange(difference, subsize+difference, spacing)

    if image is not None and not makecontourGrid:
        # Bin the image in logspace and allocate grid points based on the
        # brightness.
        imcopy = np.copy(image)
        imcopy[imcopy <= 0] = 1e-10
        bins = [-np.inf]
        if len(percentiles) == 0:
            percentiles = [45, 90]
        bins.extend(np.nanpercentile(np.log(imcopy[np.where(np.log10(imcopy)>-10)]), percentiles))
        bins.append(np.inf)

        a = np.digitize(np.log(np.copy(imcopy)),bins)
        xes = []
        ys = []

        xvals = x
        yvals = y
        yvals = np.rint(yvals).astype(int)
        xvals = np.rint(xvals).astype(int)
        for xindex in xvals:
            x = xindex + 1
            for yindex in yvals:
                y = yindex + 1
                num = int(a[x][y])
                if num == 0:
                    pass
                elif num == 1:
                    xes.append(y)
                    ys.append(x)
                else:
                    xx = np.linspace(x - 0.6, x + 0.6, num+2)[1:-1]
                    yy = np.linspace(y - 0.6, y + 0.6, num+2)[1:-1]
                    X, Y = np.meshgrid(xx, yy)
                    ys.extend(list(X.flatten()))
                    xes.extend(list(Y.flatten()))

        xx = np.array(xes)
        yy = np.array(ys)

    elif image is not None and makecontourGrid:
        Lager.debug('USING CONTOUR GRID')
        xx, yy = contourGrid(image)
        xx = np.array(xx)
        yy = np.array(yy)

    else:
        xx, yy = np.meshgrid(x+1, y+1)
    '''
        subsize = 8 #Taking a smaller square inside the image to fit on
        difference = int((size - subsize)/2)

        spacing = 1.0
        x = np.arange(difference, subsize+difference, spacing)
        y = np.arange(difference, subsize+difference, spacing)

        x -= np.mean(x)
        x+= x_center

        y -= np.mean(y)
        y+= y_center

        xx, yy = np.meshgrid(x, y)
    '''

    xx = xx.flatten()
    yy = yy.flatten()
    Lager.debug(f'Built a grid with {np.size(xx)} points')

    if type(wcs)==galsim.fitswcs.AstropyWCS:
        result = wcs.toWorld(xx, yy, units='deg')
        ra_grid = result[0]
        dec_grid = result[1]
    else:
        Lager.warning('swapped x and y here')
        result = wcs.pixel_to_world(yy, xx)
        ra_grid = result.ra.deg
        dec_grid = result.dec.deg

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

    Lager.debug('In construct psf bg using flat SED')
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



def findAllExposures(snid, ra,dec,peak,start,end,band, maxbg = 24, maxdet = 24, \
                        return_list = False, stampsize = 25, roman_path = None,\
                    pointing_list = None, SCA_list = None, truth = 'simple_model'):
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
    res.rename(columns = {'mjd':'date', 'pointing': 'Pointing', 'sca': 'SCA'}, inplace = True)

    res = res.loc[res['filter'] == band]
    det = res.loc[(res['date'] >= start) & (res['date'] <= end)]
    det['offpeak_time'] = np.abs(det['date'] - peak)
    det = det.sort_values('offpeak_time')
    det = det.iloc[:maxdet]
    det['DETECTED'] = True

    if pointing_list is not None:
        det = det.loc[det['Pointing'].isin(pointing_list)]

    bg = res.loc[(res['date'] < start) | (res['date'] > end)]
    bg['offpeak_time'] = np.abs(bg['date'] - peak)
    bg = bg.iloc[:maxbg]
    bg['DETECTED'] = False

    #combine these two dataframes
    all_images = pd.concat([det, bg])
    all_images['zeropoint'] = np.nan

    #Now we need to loop through the images and get the information we need
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
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=SCA)

    assert sed is not None, 'You must provide an SED for the source'

    if not photOps:
        Lager.warning('NOT USING PHOTON OPS IN PSF SOURCE')

    master = getPSF_Image(util_ref, stampsize, x=x, y=y,  x_center = x_center, y_center=y_center, sed = sed, include_photonOps=photOps, flux = flux).array

    return master.flatten()

def gaussian(x, A, mu, sigma):
    '''
    See name of function. :D
    '''
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def constructImages(exposures, ra, dec, size = 7, background = False, roman_path = None):

    '''
    Constructs the array of Roman images in the format required for the linear algebra operations

    Inputs:
    exposures is a list of exposures from findAllExposures
    ra,dec: the RA and DEC of the SN
    background: whether to subtract the background from the images
    roman_path: the path to the Roman data

    '''
    m = []
    err = []
    mask = []
    wgt = []
    bgflux = []
    sca_wcs_list = []
    wcs_list = []
    truth = 'simple_model'
    Lager.debug(f'truth in construct images: {truth}')

    for indx, i in enumerate(exposures):
        Lager.debug(f'Constructing image {indx} of {len(exposures)}')
        band = i['BAND']
        pointing = i['Pointing']
        SCA = i['SCA']

        image = fits.open(roman_path + f'/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{SCA}.fits.gz')

        #imagepath = roman_path + f'/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{SCA}.fits.gz'
        # TODO : replace None with the right thing once Exposure is implemented
        #image = OpenUniverse2024FITSImage( imagepath, None, SCA )


        if truth == 'truth':
            raise RuntimeError( "Truth is broken." )
            wcs = WCS(image[0].header)
            a = 0
        else:
            wcs = WCS(image[1].header)
            #wcs = image.get_wcs()
            a = 1

        sca_wcs_list.append(galsim.AstropyWCS(wcs = wcs)) #Made this into a galsim wcs

        pixel = wcs.world_to_pixel(SkyCoord(ra=ra*u.degree, dec=dec*u.degree))

        #imagedata, = image.get_data( which='data' )
        # Use this where you would have used image[...].data below

        result = Cutout2D(image[a].data, pixel, size, mode = 'strict', wcs = wcs)
        wcs_list.append(galsim.AstropyWCS(wcs = result.wcs)) # Made this into a galsim wcs

        ff = 1
        cutout = result.data
        if truth == 'truth':
            img = Cutout2D(image[0].data, pixel, size, mode = 'strict').data
            img += np.abs(np.min(img))
            img += 1
            img = np.sqrt(img)
            err_cutout = 1 / img

        else:
            err_cutout = Cutout2D(image[2].data, pixel, size, mode = 'strict').data
        cutouttime_end = time.time()

        im = cutout

        '''
        try:
            zero = np.power(10, -(i['zeropoint'] - self.common_zpt)/2.5)
        except:
            zero = -99

        if zero < 0:
            zero =
        im = cutout * zero
        '''

        bgarr = np.concatenate((im[0:size//4,0:size//4].flatten(),\
                            im[0:size,size//4:size].flatten(),\
                                im[size//4:size,0:size//4].flatten(),\
                                    im[size//4:size,size//4:size].flatten()))
        bgarr = bgarr[bgarr != 0]

        if len(bgarr) == 0:
            med = 0
            bg = 0
        else:
            pc = np.percentile(bgarr, 84)
            med = np.median(bgarr)
            bgarr = bgarr[bgarr < pc]
            bg = np.median(bgarr)

        bgflux.append(bg)

        #If we are not fitting the background we manually subtract it here.
        if not background and not truth == 'truth':
            im -= image[1].header['SKY_MEAN']
        elif not background and truth == 'truth':
            im -= bg
            Lager.debug(f'Subtracted a BG of {bg}')

        #m.append(im.flatten())
        #err.append(err_cutout.flatten())
        #mask.append(np.zeros(size*size))

        # Switching to not flattening for now

        m.append(im)
        err.append(err_cutout)
        mask.append(np.zeros((size, size)))


    #image = np.hstack(m)
    #err = np.hstack(err)

    # Switching to not flattening for now
    image = m

    return image, wcs_list, sca_wcs_list, err




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
    Lager.debug('Using 1e6 photons in getPSF_Image')
    result = point.drawImage(self.bpass,wcs=wcs, method='phot', photon_ops=photon_ops, rng=self.rng, \
        n_photons=int(1e6),maxN=int(1e6),poisson_flux=False, center = galsim.PositionD(x_center, y_center),use_true_center = True, image=stamp)
    return result

def fetchImages(testnum, detim, ID, sn_path, band, size, fit_background, roman_path):
    if len(str(ID)) != 8:
        object_type = 'star'

    else:
        object_type = 'SN'

    pqfile = find_parq(ID, sn_path, obj_type = object_type)
    ra, dec, p, s, start, end, peak = \
            get_object_info(ID, pqfile, band = band, snpath = sn_path, roman_path = roman_path, obj_type = object_type)



    snra = ra
    sndec = dec
    start = start[0]
    end = end[0]
    exposures = findAllExposures(ID, ra,dec, peak,start,end, roman_path=roman_path, maxbg = testnum - detim, \
        maxdet = detim, return_list = True, band = band)
    images, cutout_wcs_list, im_wcs_list, err = constructImages(exposures, ra, dec, size = size, \
        background = fit_background, roman_path = roman_path)

    return images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, exposures, object_type


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



def getWeights(cutout_wcs_list,size,snra,sndec, error = None, gaussian_std = 1000, cutoff = np.inf):
    wgt_matrix = []
    Lager.debug(f'Gaussian std in getWeights {gaussian_std}')
    for i,wcs in enumerate(cutout_wcs_list):
        xx, yy = np.meshgrid(np.arange(0,size,1), np.arange(0,size,1))
        xx = xx.flatten()
        yy = yy.flatten()

        rara, decdec = wcs.toWorld(xx, yy, units = 'deg')
        dist = np.sqrt((rara - snra)**2 + (decdec - sndec)**2)

        snx, sny = wcs.toImage(snra, sndec, units = 'deg')
        dist = np.sqrt((xx - snx + 1)**2 + (yy - sny + 1)**2)

        wgt = np.ones(size**2)


        wgt = 5*np.exp(-dist**2/gaussian_std)
        wgt[np.where(dist > 4)] = 0

        if not isinstance(error, np.ndarray):
            error = np.ones_like(wgt)
        wgt /= error
        wgt = wgt / np.sum(wgt)
        if i >= cutoff:
            Lager.debug(f'Setting wgt to zero on image {i}')
            wgt = np.zeros_like(wgt)
        wgt_matrix.append(wgt)
    return wgt_matrix


def makeGrid(adaptive_grid, images, size, ra, dec, cutout_wcs_list,
             percentiles=[], single_grid_point=False, npoints=7,
             make_exact=False, makecontourGrid=False):
    if adaptive_grid:
        ra_grid, dec_grid = local_grid(ra, dec, cutout_wcs_list[0],
                                       npoints, size=size,  spacing=0.75,
                                       image=images[0], spline_grid=False,
                                       percentiles=percentiles,
                                       makecontourGrid=makecontourGrid)
    else:
        if single_grid_point:
            ra_grid, dec_grid = [ra], [dec]
        else:
            ra_grid, dec_grid = local_grid(ra,dec, cutout_wcs_list[0], npoints, size = size, spacing = 0.75, spline_grid = False)

        if make_exact:
            if single_grid_point:
                galra = ra_grid[0]
                galdec = dec_grid[0]
            else:
                galra = ra_grid[106]
                galdec = dec_grid[106]


        ra_grid = np.array(ra_grid)
        dec_grid = np.array(dec_grid)
    return ra_grid, dec_grid


def plot_lc(fileroot):

    fluxdata = pd.read_csv('./results/lightcurves/'+str(fileroot)+'_lc.csv')
    supernova = fluxdata['true_flux']
    measured_flux = fluxdata['measured_flux']

    plt.figure(figsize = (10,10))
    plt.subplot(2,1,1)

    dates = fluxdata['MJD']

    plt.scatter(dates, 14-2.5*np.log10(supernova), color = 'k', label = 'Truth')
    plt.scatter(dates, 14-2.5*np.log10(measured_flux), color = 'purple', label = 'Model')

    plt.ylim(14 - 2.5*np.log10(np.min(supernova)) + 0.2, 14 - 2.5*np.log10(np.max(supernova)) - 0.2)
    plt.ylabel('Magnitude (Uncalibrated)')

    bias = np.mean(-2.5*np.log10(measured_flux)+2.5*np.log10(np.array(supernova)))
    bias *= 1000
    bias = np.round(bias, 3)
    scatter = np.std(-2.5*np.log10(measured_flux)+2.5*np.log10(np.array(supernova)))
    scatter *= 1000
    scatter = np.round(scatter, 3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = 'Overall Bias: ' + str(bias) + ' mmag \n' + \
        'Overall Scatter: ' + str(scatter) + ' mmag'
    plt.text(np.percentile(dates,60), 14 - 2.5*np.log10(np.mean(supernova)), textstr,  fontsize=14,
            verticalalignment='top', bbox=props)
    plt.legend()


    plt.subplot(2,1,2)
    flux_mode = False
    if flux_mode:
        plt.scatter(dates, X[-detim:] - supernova, color = 'k')
        for i,dr in enumerate(zip(dates, X[-detim:] - supernova)):
            d,r = dr
            plt.text(d+1,r,i+testnum-detim, fontsize = 8)
    else:
        plt.scatter(dates, -2.5*np.log10(measured_flux)+2.5*np.log10(supernova), color = 'k')
        plt.axhline(0, ls = '--', color = 'k')
        plt.ylabel('Mag Residuals (Model - Truth)')

    plt.ylabel('Mag Residuals (Model - Truth)')
    plt.xlabel('MJD')
    plt.ylim(-0.1, 0.1)


    plt.axhline(0.005, color = 'r', ls = '--')
    plt.axhline(-0.005, color = 'r', ls = '--', label = '5 mmag photometry')

    plt.axhline(0.02, color = 'b', ls = '--')
    plt.axhline(-0.02, color = 'b', ls = '--', label = '20 mmag photometry')
    plt.legend()


def plot_images(fileroot, size = 11):

    imgdata = np.load('./results/images/'+str(fileroot)+'_images.npy')
    testnum = imgdata.shape[1]//size**2
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

    fig = plt.figure(figsize = (15,3*testnum))

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
    fig = plt.figure(figsize = (15,2*testnum))
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
        if i >= testnum - detim:
            plt.title('MagBias: ' + str(np.round(magresiduals[i - testnum + detim],4)) + ' mag')


        justbgX = np.copy(X)
        justbgX[-testnum:] = 0

        justbgpred = justbgX * psf_matrix
        justbgsumimages = np.sum(justbgpred, axis = 1)
        justbgim = justbgsumimages[i*size**2:(i+1)*size**2].reshape(size,size)


        #subtract off the real sn
        #if i >= testnum - detim:
            #justbgim -= sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*supernova[i - testnum + detim]




        justbgres = trueimage - justbgim
        im1 = sumimages[i*size**2:(i+1)*size**2].reshape(size,size)


        #plt.plot(trueimage[5], label = 'Image')

        plt.axhline(0, ls = '--', color = 'k')
        #plt.plot(im1[5], label = 'Model', lw = 3)
        plt.plot(trueimage[5] - im1[5], label = 'Im-Model', alpha = 0.4)
        plt.ylim(-250,250)


        if i >= testnum - detim:
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*supernova[i - testnum + detim]
            plt.plot(snim[5], label = 'True SN', lw = 3)
            plt.fill_between(np.arange(0,11,1), trueimage[5] - snim[5] + 50, trueimage[5] - snim[5] - 50, label = 'Im-True SN', alpha = 0.4)
            plt.plot(np.arange(0,11,1), trueimage[5] - snim[5] , color = 'k', ls = '--')
            plt.plot(justbgim[5], label = 'BGModel')
            plt.plot(justbgres[5], label = 'Im-BGModel')

            #plt.plot(justbgres[5] - snim[5], label = 'SN Residuals', ls = '--')
            plt.ylim(-500,np.max(trueimage[5]))
            snim = sn_matrix[i*size**2:(i+1)*size**2, i].reshape(size,size)*X[-detim:][i - testnum + detim]

        else:
            snim = np.zeros_like(justbgres)



        plt.axvline(snx-1+4, ls = '--', color = 'k')
        plt.axvline(snx-1-4, ls = '--', color = 'k')
        plt.axvline(snx-1, ls = '--', color = 'r')

        plt.xlim(snx-1-3.8, snx-1+3.8)


        plt.legend(loc = 'upper left')






def get_SED(SNID, date, sn_path, obj_type = 'SN'):
    #Is this an ok way to do this?
    if obj_type == 'SN':
        lam, flambda = get_SN_SED(SNID, date, sn_path)
    if obj_type == 'star':
        lam, flambda = get_star_SED(SNID, sn_path)

    return lam, flambda



def get_star_SED(SNID, sn_path):
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
    filenum = find_parq(SNID, sn_path, obj_type = 'SN')
    file_name = 'snana' + '_' + str(filenum) + '.hdf5'
    fullpath = os.path.join(sn_path, file_name)
    sed_table = h5py.File(fullpath, 'r')
    sed_table = sed_table[str(SNID)]
    flambda = sed_table['flambda']
    lam = sed_table['lambda']
    mjd = sed_table['mjd']
    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    if np.min(np.abs(np.array(mjd) - date)) > 10:
        Lager.warning('WARNING: No SED data within 10 days of date. \n \
            The closest SED is ' + str(np.min(np.abs(np.array(mjd) - date))) +
                                       ' days away.')
    return np.array(lam), np.array(flambda[bestindex])



def contourGrid(image, numlevels = 5, subsize = 4):
    size = image.shape[0]
    x = np.arange(0,size,1.0)
    y = np.arange(0,size,1.0)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    xg = xg.ravel()
    yg = yg.ravel()

    levels = list(np.linspace(np.min(image), np.max(image), numlevels))
    levels = list(np.percentile(image, [0,90, 98, 100]))
    Lager.debug(f'Using levels: {levels} in contourGrid')

    interp = RegularGridInterpolator((x, y), image, method='linear',
                                 bounds_error=False, fill_value=None)

    aa = interp((xg,yg))

    x_totalgrid = []
    y_totalgrid = []

    for i in range(len(levels) - 1):

        zmin = levels[i]
        zmax = levels[i+1]
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

    return y_totalgrid, x_totalgrid


def build_lightcurve(ID, exposures, sn_path, confusion_metric, flux,
                     use_roman, band, object_type, sigma_flux):

    '''
    This code builds a lightcurve datatable from the output of the SMP algorithm.

    Input:
    ID (int): supernova ID
    exposures (table): table of exposures used in the SMP algorithm
    sn_path (str): path to supernova data
    confusion_metric (float): the confusion metric derived in the SMP algorithm
    detim (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm
    use_roman (bool): whether or not the lightcurve was built using Roman PSF
    band (str): the bandpass of the images used

    Returns:
    lc: a pandas dataframe containing the lightcurve data
    Notes:
    1.) This will soon be ECSV format instead
    2.) Soon I will turn many of these inputs into environment variable and they
    should be deleted from function arguments and docstring.
    '''

    detections = exposures[np.where(exposures['DETECTED'])]
    parq_file = find_parq(ID, path = sn_path, obj_type = object_type)
    df = open_parq(parq_file, path = sn_path, obj_type = object_type)

    if object_type == 'SN':
        meta_dict ={'confusion_metric': confusion_metric, \
        'host_sep': df['host_sn_sep'][df['id'] == ID].values[0],\
        'host_mag_g': df[f'host_mag_g'][df['id'] == ID].values[0],\
        'sn_ra': df['ra'][df['id'] == ID].values[0], \
        'sn_dec': df['dec'][df['id'] == ID].values[0], \
        'host_ra': df['host_ra'][df['id'] == ID].values[0],\
        'host_dec': df['host_dec'][df['id'] == ID].values[0]}
    else:
        meta_dict = {'ra': df[df['id'] == str(ID)]['ra'].values[0], \
            'dec': df[df['id'] == str(ID)]['dec'].values[0]}

    data_dict = {'MJD': detections['date'], 'true_flux':
    detections['realized flux'],  'measured_flux': flux, 'flux_error':
    sigma_flux}
    units = {'MJD':u.d, 'true_flux': '',  'measured_flux': '',
             'flux_error': ''}

    return QTable(data = data_dict, meta = meta_dict, units = units)


def build_lightcurve_sim(supernova, flux, sigma_flux):
    '''
    This code builds a lightcurve datatable from the output of the SMP algorithm
    if the user simulated their own lightcurve.

    Inputs
    supernova (array): the true lightcurve
    detim (int): number of detection images in the lightcurve
    X (array): the output of the SMP algorithm

    Returns
    lc: a QTable containing the lightcurve data
    2.) Soon I will turn many of these inputs into environment variable and they
    should be deleted from function arguments and docstring.
    '''
    data_dict = {'MJD': np.arange(0, detim, 1), 'true_flux': supernova,
          'measured_flux':flux , 'flux_error': sigma_flux}
    meta_dict = {}
    units = {'MJD':u.d, 'true_flux': '',  'measured_flux': '', 'flux_error':''}
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

    lc_file = os.path.join(output_path, f'{identifier}_{band}_{psftype}_lc.ecsv')

    Lager.info(f'Saving lightcurve to {lc_file}')
    lc.write(lc_file, format = 'ascii.ecsv', overwrite = overwrite)

def banner(text):
    length = len(text) + 8
    message = "\n" + "#" * length +'\n'+'#   ' + text + '   # \n'+ "#" * length
    Lager.debug(message)


def prep_data_for_fit(images, err, sn_matrix, wgt_matrix):
    '''
    This function takes the data from the images and puts it into the form such
    that we can analytically solve for the best fit using linear algebra.
    '''
    size = int(np.sqrt((images[0].size)))
    tot_num = len(images)
    det_num = len(sn_matrix)

    # Flatten into 1D arrays
    images = np.concatenate([arr.flatten() for arr in images])
    err = np.concatenate([arr.flatten() for arr in err])

    psf_zeros = np.zeros((np.size(images), tot_num))

    for i in range(det_num):
        psf_zeros[
            (tot_num - det_num + i) * size * size:
            (tot_num - det_num + i + 1) * size * size,
            (tot_num - det_num) + i] = sn_matrix[i]
    sn_matrix = psf_zeros
    sn_matrix = np.array(sn_matrix)
    sn_matrix = np.vstack(sn_matrix)


    wgt_matrix = np.array(wgt_matrix)
    wgt_matrix = np.hstack(wgt_matrix)

    return images, err, sn_matrix, wgt_matrix