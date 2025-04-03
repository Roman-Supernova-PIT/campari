import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils
from roman_imsim import *
import astropy.table as tb
import warnings 
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)
import scipy.sparse as sp 
from scipy.linalg import block_diag, lstsq
from numpy.linalg import LinAlgError
from astropy.nddata import Cutout2D
from coord import *
import requests
from astropy.table import Table
from astropy.table import QTable
import os
import scipy
import time
import galsim

import h5py
import sklearn
from sklearn import linear_model
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator


'''
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
    Generates a local grid around a RA-Dec center, choosing step size and number of points
    '''
    #Build the basic grid
    subsize = 9 #Taking a smaller square inside the image to fit on
    difference = int((size - subsize)/2)

    x_center, y_center = wcs.toImage(ra_center, dec_center, units = 'deg')

    if image is None:
        spacing = 0.5
    else:
        spacing = 1.0
    print('GRID SPACE', spacing)
    x = np.arange(difference, subsize+difference, spacing) 
    y = np.arange(difference, subsize+difference, spacing) 

    '''
    x -= np.mean(x)
    x+= x_center

    y -= np.mean(y)
    y+= y_center 
    '''

  
 
    if image is not None and not makecontourGrid:

        #Bin the image in logspace and allocate grid points based on the brightness.
        imcopy = np.copy(image)
        imcopy[imcopy <= 0] = 1e-10
        bins = [-np.inf]
        if len(percentiles) == 0:
            #percentiles = [25, 80, 90]
            percentiles = [45, 90]
        bins.extend(np.nanpercentile(np.log(imcopy[np.where(np.log10(imcopy)>-10)]), percentiles))
        #print('added 90 percentile to local grid')
        bins.append(np.inf)

        a = np.digitize(np.log(np.copy(imcopy)),bins)
        xes = []
        ys = []
        

        '''
        xvals = np.array(range(-2, subsize+2)).astype(float)
        print('xvals', xvals)
        xvals -= np.mean(xvals)
        print('xvals', xvals)
        xvals += x_center 
        print('xvals', xvals)
        
        
        
        yvals = np.array(range(-2, subsize+2)).astype(float)
        yvals -= np.mean(yvals)
        yvals += y_center 
        
        '''

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
                    xx = np.linspace(x-0.6,x+0.6,num+2)[1:-1]
                    yy = np.linspace(y-0.6,y+0.6,num+2)[1:-1]
                    X,Y = np.meshgrid(xx,yy)
                    ys.extend(list(X.flatten()))
                    xes.extend(list(Y.flatten()))
        
        xx = np.array(xes)
        yy = np.array(ys)

    elif image is not None and makecontourGrid:
        print('USING CONTOUR GRID')
        xx, yy = contourGrid(image)
        xx = np.array(xx)
        yy = np.array(yy)


    else:

        xx, yy = np.meshgrid(x+1, y+1) 
    '''
        subsize = 8 #Taking a smaller square inside the image to fit on
        difference = int((size - subsize)/2)

        spacing = 1.0
        print('GRID SPACE', spacing)
        x = np.arange(difference, subsize+difference, spacing) 
        y = np.arange(difference, subsize+difference, spacing) 

        x -= np.mean(x)
        x+= x_center

        y -= np.mean(y)
        y+= y_center 

        xx, yy = np.meshgrid(x, y) 
        print(xx)
    '''
    

    xx = xx.flatten()
    yy = yy.flatten()
    print('Built a grid with', np.size(xx), 'points')

    
    if type(wcs)==galsim.fitswcs.AstropyWCS:
        result = wcs.toWorld(xx, yy, units = 'deg')
        ra_grid = result[0]
        dec_grid = result[1]
    else:
        print('swapped x and y here')
        result = wcs.pixel_to_world(yy, xx) #Convert them to RA/DEC and return
        ra_grid = result.ra.deg
        dec_grid = result.dec.deg


    return ra_grid, dec_grid


    

def generateGuess(imlist, wcslist, ra_grid, dec_grid):
    '''
    This function initializes the guess for the optimization. For each grid point, it finds the average value of the pixel it is sitting in on 
    each image. In some cases, this has offered minor improvements but it is not make or break for the algorithm.
    '''
    size = np.shape(imlist[0])[0]
    imx = np.arange(0,size,1)
    imy = np.arange(0,size,1)
    imx, imy = np.meshgrid(imx, imy)
    all_vals = np.zeros_like(ra_grid)

    for i,imwcs in enumerate(zip(imlist,wcslist)):
        im, wcs = imwcs
        if type(wcs) == galsim.fitswcs.AstropyWCS:
            #This actually means that we have a galsim wcs that was loaded from an astropy one
            xx, yy = wcs.toImage(ra_grid, dec_grid,units='deg')
        else:
            xx,yy = wcs.world_to_pixel(SkyCoord(ra = ra_grid*u.degree, dec = dec_grid*u.degree))

        grid_point_vals = np.zeros_like(xx)
        for imval, imxval, imyval in zip(im.flatten(), imx.flatten(), imy.flatten()):
            grid_point_vals[np.where((np.abs(xx - imxval) < 0.5) & (np.abs(yy - imyval) < 0.5))] = imval
        all_vals+= grid_point_vals
    return all_vals/len(wcslist)



def construct_psf_background(ra, dec, wcs, x_loc, y_loc, stampsize, bpass, use_roman, \
    color=0.61, psf = None, pixel = False, include_photonOps = False, util_ref = None, band = None):

    '''
    Constructs the background model around a certain image (x,y) location and a given array of RA and DECs.
    Inputs:
    ra, dec: arrays of RA and DEC values for the grid
    wcs: the wcs of the image, if the image is a cutout, this MUST be the wcs of the CUTOUT
    x_loc, y_loc: the pixel location of the image in the FULL image, i.e. x y location in the SCA.
    stampsize: the size of the stamp being used
    bpass: the bandpass being used
    flatten: whether to flatten the output array (REMOVED XXXXXX)
    color: the color of the star being used (currently not used)
    psf: Here you can provide a PSF to use, if you don't provide one, you must provide a util_ref, which will calculate the Roman PSF instead.
    pixel: If True, use a pixel tophat function to convolve the PSF with, otherwise use a delta function. Does not seem to hugely affect results.
    include_photonOps: If True, use photon ops in the background model. This is not recommended for general use, as it is very slow.
    util_ref: A reference to the util object, which is used to calculate the PSF. If you provide this, you don't need to provide a PSF. Note
            that this needs to be for the correct SCA/Pointing combination.

    Returns:
    A numpy array of the PSFs at each grid point, with the shape (stampsize*stampsize, npoints)    
    '''

    assert util_ref is not None or psf is not None, 'you must provide at least util_ref or psf'

    assert util_ref is not None or band is not None, 'you must provide at least util_ref or band'

    if not use_roman:
        assert psf is not None, 'you must provide an input psf if not using roman'
    else:
        psf = None

    

    


    if type(wcs) == galsim.fitswcs.AstropyWCS:
        #print('using astropy')
        x,y = wcs.toImage(ra,dec,units='deg')
    else:
        #print('using not astropy')
        x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree))



    psfs = np.zeros((stampsize * stampsize,np.size(x)))

    k = 0 

    #For now, we use a flat SED. This is not ideal, but it is a good starting point.
    print('In construct psf bg using flat SED')
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
                #print('PSF x and y in construct psf background', x_loc, y_loc)
                #print('Ive changed this to xloc and yloc for now but its possibly wrong')
                
                #print(stamp)
                #print(newwcs)

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

def simulateImages(testnum,detim,ra,dec,do_xshift,do_rotation,supernova,noise, use_roman,band, deltafcn_profile, size=11, \
    input_psf =None, constant_imgs = False, bg_gal_flux = None, source_phot_ops = True, mismatch_seds = False):
    '''
    This function simulates images using galsim for testing purposes. It is not used in the main pipeline.
    Inputs:
    testnum: the number of images to simulate
    detim: the number of images to simulate with a supernova
    ra, dec: the RA and DEC of the center of the images to simulate, and the RA and DEC of the supernova.
    do_xshift: whether to shift the images in the x direction (they will still be centered on the same point, this is just to emulate\
        Roman taking a series of images at different locations.)
    do_rotation: whether to rotate the images
    supernova: the flux of the supernova to simulate, a list of flux values.
    noise: the noise level to add to the images.
    use_roman: whether to use the Roman PSF or a simple airy PSF.
    size: the size of the images to simulate.

    Returns:
    images: a numpy array of the images, with shape (testnum*size*size)
    im_wcs_list: a list of the wcs objects for each full SCA image
    cutout_wcs_list: a list of the wcs objects for each cutout image
    '''
    if not use_roman:
        assert input_psf is not None, 'you must provide an input psf if not using roman'
    else:
        input_psf = None
    galra = ra + 1.5e-5
    galdec = dec + 1.5e-5
    #print('ra and dec in simulate images', galra, galdec)
    snra = ra
    sndec = dec
    im_wcs_list = []
    cutout_wcs_list = []
    imagelist = []
    roman_bandpasses = galsim.roman.getBandpasses()
    psf_storage = []
    sn_storage = []



    for i in range(testnum):

        #Spinny loader just for fun :D
        spinner = ['|', '/', '-', '\\']
        print('Image ' + str(i) + '   ' + spinner[i%4], end = '\r')
 
        if do_xshift:
                xshift = 1e-5/3 * i
                yshift = 0
        else:
            xshift = 0
            yshift = 0

        if do_rotation:
            rotation_angle = np.pi/10 * i
        else:
            rotation_angle = 0


        #Simulating a WCS object for the full image
        rotation_matrix = np.array([np.cos(rotation_angle), -np.sin(rotation_angle), np.sin(rotation_angle), np.cos(rotation_angle)]).reshape(2,2)
        CD_matrix = np.array([[-2.0951875487247e-05,  -1.9726924681363e-05], [2.11521248003807e-05,  -2.1222586081326e-05]])
        CD_matrix_rotated = CD_matrix @ rotation_matrix
        
        wcs_dict = {
            'CTYPE1': 'RA---TAN-SIP',                                                        
            'CTYPE2': 'DEC--TAN-SIP',

            'CRPIX1':               2044.0,                                                  
            'CRPIX2':               2044.0,                                                    
            'CD1_1': CD_matrix_rotated[0,0],
            'CD1_2': CD_matrix_rotated[0,1],
            'CD2_1': CD_matrix_rotated[1,0],
            'CD2_2': CD_matrix_rotated[1,1],                                               
            'CUNIT1': 'deg     '   ,                                                         
            'CUNIT2': 'deg     '    ,                                                        
            'CRVAL1':   7.5942407686430995 + xshift,   #This is an arbitrary RA / DEC value for the center of the SCA, this should probably be formalized.                                   
            'CRVAL2':  -44.180904726970695 + yshift,    

            'NAXIS1':                 4088,                                                  
            'NAXIS2':                 4088 
        }
        
        imwcs = WCS(wcs_dict)


        #Just using this astropy tool to get the cutout wcs.
        cutoutstamp = Cutout2D(np.zeros((4088,4088)), SkyCoord(ra = ra*u.degree, dec = dec*u.degree), size, wcs=imwcs)
        cutoutgalwcs = galsim.AstropyWCS(wcs = cutoutstamp.wcs)
        
        galwcs = galsim.AstropyWCS(wcs = imwcs)
        

        galwcs2, origin = galsim.wcs.readFromFitsHeader(wcs_dict)
        x,y = galwcs2.toImage(ra, dec, units = 'deg')

        galx2, galy2 = galwcs2.toImage(galra*u.degree,galdec*u.degree, units = 'deg')

        im_wcs_list.append(galwcs2)
        if mismatch_seds:
            print('INTENTIONALLY MISMATCHING SEDS, 1a SED')
            
            file_path = r"snflux_1a.dat"
            df = pd.read_csv(file_path, sep = '\s+', header = None, names = ['Day', 'Wavelength', 'Flux'])
            a = df.loc[df.Day == 0]
            del df
            sed = galsim.SED(galsim.LookupTable(a.Wavelength/10, a.Flux, interpolant='linear'),
                                    wave_type='nm', flux_type='fphotons')
            '''
            sed = galsim.SED(galsim.LookupTable([100, 2600], [0.8,1], interpolant='linear'),
                                wave_type='nm', flux_type='fphotons')
            '''
        else:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                                wave_type='nm', flux_type='fphotons')

        stamp = galsim.Image(size,size,wcs=cutoutgalwcs)
        pointx, pointy = cutoutgalwcs.toImage(galra, galdec, units = 'deg')

        if use_roman:
            sim_psf = galsim.roman.getPSF(1, band, pupil_bin=8, wcs = cutoutgalwcs)
            #pupil_bin = 8
            #util_ref = roman_utils(config_file='./temp_tds.yaml', visit = 502, sca = 13)

            #sim_psf = util_ref.getPSF(x,y,pupil_bin)
            
        else:
            #print('\n')
           # print('Using input PSF')
            #sim_psf = airy
            sim_psf = input_psf

        
        #Draw the galaxy.
        if deltafcn_profile:
            profile = galsim.DeltaFunction()*sed
            profile = profile.withFlux(bg_gal_flux, roman_bandpasses[band]) 
            convolved = galsim.Convolve(profile, sim_psf)
        else:
            bulge = galsim.Sersic(n=3, half_light_radius=1.6)
            disk = galsim.Exponential(half_light_radius=5)
            #bulge = galsim.Sersic(n=3, half_light_radius=3)
            #disk = galsim.Exponential(half_light_radius=6)
            gal = bulge + disk
            profile = gal*sed
            profile = profile.withFlux(bg_gal_flux, roman_bandpasses[band])
            convolved = galsim.Convolve(profile, sim_psf)
        

        a = convolved.drawImage(roman_bandpasses[band], method='no_pixel', image = stamp, \
            wcs = cutoutgalwcs, center = galsim.PositionD(pointx, pointy), use_true_center = True)
        a = a.array

        '''
        plt.figure(figsize = (5,5))
        plt.imshow(a)
        plt.title('Image in sim image')
        plt.colorbar()
        plt.show()
        '''
        
        stamp2 = galsim.Image(size,size,wcs=cutoutgalwcs)
        psf_storage.append((sim_psf*sed).drawImage(roman_bandpasses[band], wcs = cutoutgalwcs, center = (5, 5), use_true_center = True, image = stamp2).array)

        
        #Noise it up!
        if noise > 0:
            a += np.random.normal(0, noise, size**2).reshape(size,size)
        #Inject a supernova! If using.
        if supernova != 0:
            if i >= testnum - detim:
                snx, sny = cutoutgalwcs.toImage(snra, sndec, units = 'deg')
                if use_roman:
                    supernova_image = construct_psf_source(x, y, 662, 11, stampsize=size,  \
                        x_center = snx, y_center = sny, flux = supernova[i - testnum + detim], sed = sed, photOps = source_phot_ops).reshape(size,size)
                    a += supernova_image
                    

                else:
                    stamp = galsim.Image(size,size,wcs=cutoutgalwcs)
                    profile = galsim.DeltaFunction()*sed
                    profile = profile.withFlux(supernova[i - testnum + detim], roman_bandpasses[band]) 
                    
                    convolved = galsim.Convolve(profile, sim_psf)
                    supernova_image = convolved.drawImage(roman_bandpasses[band], method='no_pixel', image = stamp, \
                                wcs = cutoutgalwcs, center = (snx, sny), \
                                    use_true_center = True, add_to_image = False).array
                    a += supernova_image
                sn_storage.append(supernova_image)


        cutout_wcs_list.append(cutoutgalwcs)
        imagelist.append(a.flatten())

    images = np.array(imagelist)
    images = np.hstack(images)

    

    return images, im_wcs_list, cutout_wcs_list, psf_storage, sn_storage


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
                print('No truth file found for ', row.Pointing, row.SCA)
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
    print(explist)

    if return_list:
        return explist

def find_parq(ID, path, star = False):
    '''
    Find the parquet file that contains a given supernova ID.
    '''
    files = os.listdir(path)
    if star:
        files = [f for f in files if 'pointsource' in f]
        files = [f for f in files if 'flux' not in f]
    else:
        files = [f for f in files if 'snana' in f]
    files = [f for f in files if '.parquet' in f]
    for f in files:
        pqfile = int(f.split('_')[1].split('.')[0])
        df = open_parq(pqfile, path)
        if ID in df.id.values:
            return pqfile

def open_parq(ID, path):
    '''
    Convenience function to open a parquet file given a supernova ID.
    '''
    df = pd.read_parquet(path+'/snana_'+str(ID)+'.parquet', engine='fastparquet')
    return df

def SNID_to_loc(SNID, parq, band,\
     snpath, roman_path, host = False, date = False):
    '''
    Fetch some info about a SN given its ID.
    Inputs:
    SNID: the ID of the supernova
    parq: the parquet file containing the supernova
    band: the band to consider
    date: whether to return the start end and peak dates of the supernova
    snpath: the path to the supernova data
    roman_path: the path to the Roman data
    host: whether to return the host RA and DEC

    Returns:
    RA, DEC: the RA and DEC of the supernova
    p, s: the pointing and SCA of the supernova
    start, end, peak: the start, end, and peak dates of the supernova
    host_ra, host_dec: the RA and DEC of the host galaxy
    '''
    df = open_parq(parq, snpath)
    df = df.loc[df.id == SNID]
    RA, DEC = df.ra.values[0], df.dec.values[0]
    start = df.start_mjd.values
    end = df.end_mjd.values
    peak = df.peak_mjd.values

    
    if not date:
        p, s = radec2point(RA, DEC, band, roman_path)
        return RA, DEC, p, s
    else:
        p, s = radec2point(RA, DEC, band, roman_path,  start, end)
        if host:
            return RA, DEC, p, s, start, end, peak, df.host_ra.values[0], df.host_dec.values[0]
        else:
            return RA, DEC, p, s, start, end, peak

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

    dist = np.sqrt((allRA - RA)**2 + (allDEC - DEC)**2)

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

    print('ARGS IN PSF SOURCE', x, y, pointing, SCA, stampsize, x_center, y_center, sed, flux)

    config_file = './temp_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=SCA)

    assert sed is not None, 'You must provide an SED for the source'

    if not photOps:
        print('WARNING: NOT USING PHOTON OPS IN PSF SOURCE')
        print('ARE YOU SURE YOU WANT TO DO THIS?')

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
    print('truth in construct images', truth)

            
    for indx, i in enumerate(exposures):
        spinner = ['|', '/', '-', '\\']
        print('Image ' + str(indx) + '   ' + spinner[indx%4], end = '\r')
        band = i['BAND']
        pointing = i['Pointing']
        SCA = i['SCA']
        image = fits.open(roman_path + f'/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{SCA}.fits.gz')
        if truth == 'truth':
            wcs = WCS(image[0].header)
            a = 0
        else:
            wcs = WCS(image[1].header)
            a = 1

        sca_wcs_list.append(galsim.AstropyWCS(wcs = wcs)) #Made this into a galsim wcs

        pixel = wcs.world_to_pixel(SkyCoord(ra=ra*u.degree, dec=dec*u.degree))

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
            print('failed')
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
            #calimage = image[1]
            #bins = np.linspace(0,1000,100)
            #bincenters = (bins[1:] + bins[:-1])/2
            #x = plt.hist(calimage.data.flatten(), bins = np.linspace(0,1000,100), histtype = 'step', color = 'k', density = True)

            #fit a Gaussian to the truth image

            #popt, pcov = scipy.optimize.curve_fit(gaussian, bincenters, x[0], p0 = [.01, 400, 100])
            
            im -= image[1].header['SKY_MEAN']
            #print('subtracting sky mean not my fit')
            #print('subtracted a bg of', popt[1])
            #print('compared to: ', image[1].header['SKY_MEAN'])
            #print('--------------------')
        elif not background and truth == 'truth':
            im -= bg
            #print('Subtracted a BG of', bg)


        m.append(im.flatten())
        err.append(err_cutout.flatten())
        mask.append(np.zeros(size*size))
        #w = (zero**2)*err_cutout.flatten()
     
    image = np.hstack(m)
    err = np.hstack(err)
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
        print('Building a Pixel shaped PSF source')
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
    print('Using 1e6 photons in getPSF_Image')
    result = point.drawImage(self.bpass,wcs=wcs, method='phot', photon_ops=photon_ops, rng=self.rng, \
        n_photons=int(1e6),maxN=int(1e6),poisson_flux=False, center = galsim.PositionD(x_center, y_center),use_true_center = True, image=stamp)
    return result

def fetchImages(testnum, detim, ID, sn_path, band, size, fit_background, roman_path):
    pqfile = find_parq(ID, sn_path)
    ra, dec, p, s, start, end, peak, galra, galdec = \
        SNID_to_loc(ID, pqfile, date = True, band = band, snpath = sn_path, roman_path = roman_path, host = True)
    snra = ra
    sndec = dec
    start = start[0]
    end = end[0]
    exposures = findAllExposures(ID, ra,dec, peak,start,end, roman_path=roman_path, maxbg = testnum - detim, \
        maxdet = detim, return_list = True, band = band)
    images, cutout_wcs_list, im_wcs_list, err = constructImages(exposures, ra, dec, size = size, \
        background = fit_background, roman_path = roman_path)

    return images, cutout_wcs_list, im_wcs_list, err, snra, sndec, ra, dec, exposures

def getWeights(cutout_wcs_list,size,snra,sndec, error = None, gaussian_std = 1000, cutoff = np.inf):
    wgt_matrix = []
    print('Gaussian std in getWeights', gaussian_std)
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
            print('Setting wgt to zero on image', i)
            wgt = np.zeros_like(wgt)
        wgt_matrix.append(wgt)
    return wgt_matrix

def makeGrid(adaptive_grid, images,size,ra,dec,cutout_wcs_list, percentiles = [], single_grid_point=False, npoints = 7, make_exact = False, makecontourGrid = False):
    if adaptive_grid:
        a = images[:size**2].reshape(size,size)
        ra_grid, dec_grid = local_grid(ra,dec, cutout_wcs_list[0], \
                npoints, size = size,  spacing = 0.75, image = a, spline_grid = False, percentiles = percentiles, makecontourGrid = makecontourGrid)
        print('removed wgt when making adaptive grid')
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
            #print('subtracting sn')
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
    





def get_SED(SNID, date, star = False):
    filenum = find_parq(SNID, star = star)
    if star:
        filename = sn_path + 'pointsource_' + str(filenum) + '.hdf5'
    else:
        filename = sn_path + 'snana_' + str(filenum) + '.hdf5'
    h5 = h5py.File(filename,'r')
    h5 = h5[str(SNID)]
    lam = h5['lambda']
    flambda = h5['flambda']
    mjd = h5['mjd']

    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    if np.min(np.abs(np.array(mjd) - date)) > 10:
        print('WARNING: No SED data within 10 days of date. \n \
            The closest SED is ' + str(np.min(np.abs(np.array(mjd) - date))) + ' days away.')
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
    print(levels)

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

'''
def saveLightcurves(ID, exposures, sn_path, confusion_metric, use_real_images, detim, supernova, X, use_roman, band):   
    #First, build the lc file
    if use_real_images:
        identifier = str(ID)
    else:
        identifier = 'simulated'
    if use_roman:
        psftype = 'romanpsf'
    else:
        psftype = 'analyticpsf'


    lc = pd.DataFrame()
    if use_real_images:
        detections = exposures[np.where(exposures['DETECTED'])]
        parq_file = find_parq(ID, path = sn_path)
        df = open_parq(parq_file, path = sn_path)
        lc['true_flux'] = detections['realized flux']
        lc['MJD'] = detections['date']
        lc['confusion metric'] = confusion_metric
        lc['host_sep'] = df['host_sn_sep'][df['id'] == ID].values[0]
        lc['host_mag_g'] = df[f'host_mag_g'][df['id'] == ID].values[0]
        lc['sn_ra'] = df['ra'][df['id'] == ID].values[0]
        lc['sn_dec'] = df['dec'][df['id'] == ID].values[0]
        lc['host_ra'] = df['host_ra'][df['id'] == ID].values[0]
        lc['host_dec'] = df['host_dec'][df['id'] == ID].values[0]

    else:
        lc['true_flux'] = supernova
        lc['MJD'] = np.arange(0, detim, 1)

    lc['measured_flux'] = X[-detim:]
    
    print('Saving lightcurve to ./results/lightcurves/'+ f'{identifier}_{band}_{psftype}_lc.csv')            
    lc.to_csv(f'./results/lightcurves/{identifier}_{band}_{psftype}_lc.csv', index = False)
'''

def build_lightcurve(ID, exposures, sn_path, confusion_metric, detim, supernova, X, use_roman, band):

    detections = exposures[np.where(exposures['DETECTED'])]
    parq_file = find_parq(ID, path = sn_path)
    df = open_parq(parq_file, path = sn_path)

    meta_dict ={'confusion_metric': confusion_metric, \
    'host_sep': df['host_sn_sep'][df['id'] == ID].values[0],\
     'host_mag_g': df[f'host_mag_g'][df['id'] == ID].values[0],\
      'sn_ra': df['ra'][df['id'] == ID].values[0], \
      'sn_dec': df['dec'][df['id'] == ID].values[0], \
      'host_ra': df['host_ra'][df['id'] == ID].values[0],\
       'host_dec': df['host_dec'][df['id'] == ID].values[0]}

    data_dict = {'MJD': detections['date'], 'true_flux': detections['realized flux'],  'measured_flux': X[-detim:]}
    units = {'MJD':u.d, 'true_flux': '',  'measured_flux': ''}

    return QTable(data = data_dict, meta = meta_dict, units = units)


def build_lightcurve_sim(supernova, detim, X):
    data_dict = {'MJD': np.arange(0, detim, 1), 'true_flux': supernova,  'measured_flux': X[-detim:]}
    meta_dict = {}
    units = {'MJD':u.d, 'true_flux': '',  'measured_flux': ''}
    return QTable(data = data_dict, meta = meta_dict, units = units)


def save_lightcurve(lc,identifier, band, psftype, output_path = None, overwrite = True):

    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'results/lightcurves/')

    lc_file = os.path.join(output_path, f'{identifier}_{band}_{psftype}_lc.ecsv')

    print('Saving lightcurve to ' + lc_file)            
    lc.write(lc_file, format = 'ascii.ecsv', overwrite = overwrite)
