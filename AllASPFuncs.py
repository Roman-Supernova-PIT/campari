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
import os
import scipy
import time
import galsim

import sklearn
from sklearn import linear_model
from scipy.interpolate import RectBivariateSpline

roman_path = '/hpc/group/cosmology/OpenUniverse2024'
sn_path = '/hpc/group/cosmology/OpenUniverse2024/roman_rubin_cats_v1.1.2_faint/'

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

def local_grid(ra_center, dec_center, wcs, npoints, size = 25, spacing = 1.0, image = None, spline_grid = True, percentiles = []):

    '''
    Generates a local grid around a RA-Dec center, choosing step size and number of points
    '''


    x_center, y_center = wcs.toImage(ra_center, dec_center, units = 'deg')


    if spline_grid:
        print('fitting spline grid')

        #testimage = images[0*size**2:(0+1)*size**2].reshape(size,size)
        testimage = image
        x = np.arange(0,size,1)
        y = np.arange(0,size,1)
        xx, yy = np.meshgrid(x,y)
        spline = RectBivariateSpline(x, y, testimage)

        x = np.linspace(0,size-1,100)+1
        y = np.linspace(0,size-1,100)+1
        print('Adding 1 in spline grid')
        xx, yy = np.meshgrid(x,y)



        splineval = spline.__call__(x,y)
        splineval /= np.max(splineval)

        splinederiv = np.sqrt(spline.__call__(x,y,dx = 1)**2 + spline.__call__(x,y,dy = 1)**2) 
        splinederiv /= np.max(splinederiv)
        combo = splinederiv + splineval
        argsort = np.argsort(combo.flatten())
        argsort = argsort[::-1]



        xx_sort = xx.flatten()[argsort]
        yy_sort = yy.flatten()[argsort]

        indices_zero = np.arange(0,100,5)
        indices = np.arange(0,1000,5)

        indices2 = np.arange(1000,3000, 20)
        indices3 = np.arange(3000,np.size(xx_sort), 50)

        totindices = np.concatenate([indices, indices2, indices3])
        print(np.size(totindices))
        plt.subplot(1,2,1)
        plt.imshow(image, origin = 'lower')
        plt.scatter(xx_sort[totindices],yy_sort[totindices], c = 'r', s = 1) 

        plt.subplot(1,2,2)
        plt.imshow(splinederiv, origin = 'lower')
        plt.scatter(xx_sort[totindices]*10,yy_sort[totindices]*10, c = 'r', s = 1) 

        xx = np.array(xx_sort[totindices])
        print('xx from spline grid')
        print(xx[:20])
        yy = np.array(yy_sort[totindices])
        print('overwriting as a test')


    
    elif image is not None:

        #Bin the image in logspace and allocate grid points based on the brightness.
        imcopy = np.copy(image)
        imcopy[imcopy <= 0] = 1e-10
        bins = [-np.inf]
        #bins.extend(np.nanpercentile(np.log(imcopy[np.where(np.log10(imcopy)>-10)]), [30, 85]))
        if len(percentiles) == 0:
            percentiles = [25, 80, 90]
        bins.extend(np.nanpercentile(np.log(imcopy[np.where(np.log10(imcopy)>-10)]), percentiles))
        print('added 90 percentile to local grid')
        #85, 
        #bins.extend(np.nanpercentile(np.log(imcopy[np.where(np.log10(imcopy)>-10)]), [35, 90]))
        bins.append(np.inf)

        a = np.digitize(np.log(np.copy(imcopy)),bins)
        xes = []
        ys = []
        
        subsize = 5 #Taking a smaller square inside the image to fit on
        difference = int((size - subsize)/2)

        a = a.reshape(size,size)
        xvals = np.array(range(-2, subsize+2)).astype(float)
        xvals -= np.mean(xvals)
        xvals += x_center 
        
        xvals = np.rint(xvals).astype(int)

        yvals = np.array(range(-2, subsize+2)).astype(float)
        yvals -= np.mean(yvals)
        yvals += y_center 
        yvals = np.rint(yvals).astype(int)

        for xindex in xvals: 
            x = xindex - 1
            for yindex in yvals: 
                y = yindex - 1
                num = int(a[x][y])
                #if xindex == size//2 + 1 and yindex == size//2 + 1:
                    #continue
                if num == 0:
                    pass
                elif num == 1:
                    xes.append(yindex)
                    ys.append(xindex)
                else: 
                    xx = np.linspace(xindex-0.6,xindex+0.6,num+2)[1:-1]
                    yy = np.linspace(yindex-0.6,yindex+0.6,num+2)[1:-1]
                    X,Y = np.meshgrid(xx,yy)
                    ys.extend(list(X.flatten()))
                    xes.extend(list(Y.flatten()))
        
        xx = np.array(xes)
        yy = np.array(ys)

        print('Built a grid with', np.size(xx), 'points')
        #dist = np.sqrt((xx - x_center)**2 + (yy - y_center)**2)
        #delete xx and yy that are too close
        #xx = xx[dist >= 1]
        #yy = yy[dist >= 1]


        
    else:

        subsize = 6 #Taking a smaller square inside the image to fit on
        difference = int((size - subsize)/2)

        spacing = 0.4
        x = np.arange(difference, subsize+difference, spacing) 
        y = np.arange(difference, subsize+difference, spacing) 

        x -= np.mean(x)
        x+= x_center

        y -= np.mean(y)
        y+= y_center 

        xx, yy = np.meshgrid(x, y) 
        
    xx = xx.flatten()
    yy = yy.flatten()

    
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

def gradientGrid(im, wcs, ra_grid, dec_grid):
    '''

    '''
    plt.imshow(im,origin = 'lower')

    gradient = np.gradient(im)

    plt.quiver(gradient[1], gradient[0])
    imx = np.arange(0,size,1)
    imy = np.arange(0,size,1)
    xx, yy = wcs.toImage(ra_grid, dec_grid,units='deg')



    xx_mod = []
    yy_mod = []
    for xcoord, ycoord in zip(xx,yy):
        xarg = np.argmin(np.abs(imx - xcoord))
        yarg = np.argmin(np.abs(imy - ycoord))
        gradx = gradient[1][yarg-1,xarg-1]
        grady = gradient[0][yarg-1,xarg-1]
        #plt.arrow(xcoord, ycoord, gradx/100, grady/100, color = 'blue', head_width = 0.1, head_length = 0.1, zorder = 10)
        xx_mod.append(gradx/np.max(gradient))
        yy_mod.append(grady/np.max(gradient))

    xx_prime = xx + np.array(xx_mod)
    yy_prime = yy + np.array(yy_mod)


       

    ra_grid, dec_grid = wcs.toWorld(xx_prime,yy_prime,units='deg')
    #plt.scatter(xx,yy, color = 'red')
    plt.scatter(xx_prime-1, yy_prime-1, color = 'blue')
    plt.show()
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



def construct_psf_background(ra, dec, wcs, x_loc, y_loc, stampsize, bpass, \
    color=0.61, psf = None, pixel = False, include_photonOps = False, util_ref = None):

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


    if type(wcs) == galsim.fitswcs.AstropyWCS:
        x,y = wcs.toImage(ra,dec,units='deg')
    else:
        x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree))

    #print('xx in construct bg grid')
    #print(x[:20])
    psfs = np.zeros((stampsize * stampsize,np.size(x)))

    k = 0 

    #For now, we use a flat SED. This is not ideal, but it is a good starting point.
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')

    if pixel:
        point = galsim.Pixel(0.1)*sed
    else:
        point = galsim.DeltaFunction()*sed

    point = point.withFlux(1,bpass)
    oversampling_factor = 1
    pupil_bin = 8

    newwcs = wcs

    #Loop over the grid points, draw the PSF at each one, and append to a list.
    for a,ij in enumerate(zip(x.flatten(),y.flatten())):
        i,j = ij
        stamp = galsim.Image(stampsize*oversampling_factor,stampsize*oversampling_factor,wcs=newwcs)
        
        if not include_photonOps:
            if not psf:
                convolvedpsf = galsim.Convolve(point, util_ref.getPSF(x,y,pupil_bin))
                
            else:
                convolvedpsf = galsim.Convolve(point, psf)
            result = convolvedpsf.drawImage(bpass, method='no_pixel',\
                center = galsim.PositionD(i, j),use_true_center = True, image = stamp, wcs = newwcs) 

        else:
            photon_ops = [util_ref.getPSF(i,j,8)] + util_ref.photon_ops 
            result = point.drawImage(bpass,wcs=newwcs, method='phot', photon_ops=photon_ops, rng=util_ref.rng, \
                n_photons=int(1e6),maxN=int(1e6),poisson_flux=False, center = galsim.PositionD(i+1, j+1),\
                    use_true_center = True, image=stamp)

        psfs[:,k] = result.array.flatten() 
        k += 1

    return psfs

def simulateImages(testnum,detim,ra,dec,do_xshift,do_rotation,supernova,noise, use_roman,band, size=11):
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
        sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                                wave_type='nm', flux_type='fphotons')

        stamp = galsim.Image(size,size,wcs=cutoutgalwcs)

        pointx, pointy = cutoutgalwcs.toImage(galra, galdec, units = 'deg')

        if use_roman:
            sim_psf = galsim.roman.getPSF(1,band, pupil_bin=8, wcs = cutoutgalwcs)
        else:
            sim_psf = airy

        
        #Draw the galaxy.
        if deltafcn_profile:
            profile = galsim.DeltaFunction()*sed
            profile = profile.withFlux(9e6, roman_bandpasses[band]) 
            convolved = galsim.Convolve(profile, sim_psf)
        else:
            bulge = galsim.Sersic(n=3, half_light_radius=1.6)
            disk = galsim.Exponential(half_light_radius=5)
            gal = bulge + disk
            profile = gal*sed
            profile = profile.withFlux(9e6, roman_bandpasses[band])
            convolved = galsim.Convolve(profile, sim_psf)

        a = convolved.drawImage(roman_bandpasses[band], method='no_pixel', image = stamp, \
            wcs = cutoutgalwcs, center = (pointx, pointy), use_true_center = True).array
        
        #Noise it up!
        if noise > 0:
            a += np.random.normal(background_level, noise, size**2).reshape(size,size)
        
        #Inject a supernova! If using.
        if supernova != 0:
            if i >= testnum - detim:
                snx, sny = cutoutgalwcs.toImage(snra, sndec, units = 'deg')
                if use_roman:
                    a += construct_psf_source(x, y, 662, 11, stampsize=size,  \
                        x_center = snx, y_center = sny, flux = supernova[i - testnum + detim], sed = sed).reshape(size,size)
                else:
                    stamp = galsim.Image(size,size,wcs=cutoutgalwcs)
                    profile = galsim.DeltaFunction()*sed
                    profile = profile.withFlux(supernova[i - testnum + detim], roman_bandpasses[band]) 
                    
                    convolved = galsim.Convolve(profile, sim_psf)
                    a += convolved.drawImage(roman_bandpasses[band], method='no_pixel', image = stamp, \
                                wcs = cutoutgalwcs, center = (snx, sny), \
                                    use_true_center = True, add_to_image = False).array


        cutout_wcs_list.append(cutoutgalwcs)
        imagelist.append(a.flatten())

    images = np.array(imagelist)
    images = np.hstack(images)

    

    return images, im_wcs_list, cutout_wcs_list


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

def find_parq(ID, path = '/hpc/group/cosmology/OpenUniverse2024/roman_rubin_cats_v1.1.2_faint/'):
    '''
    Find the parquet file that contains a given supernova ID.
    '''
    files = os.listdir(path)
    files = [f for f in files if 'snana' in f]
    files = [f for f in files if '.parquet' in f]
    for f in files:
        pqfile = int(f.split('_')[1].split('.')[0])
        df = open_parq(pqfile, path)
        if ID in df.id.values:
            return pqfile

def open_parq(ID, path = '/cwork/mat90/RomanDESC_sims_2024/roman_rubin_cats_v1.1.2_faint'):
    '''
    Convenience function to open a parquet file given a supernova ID.
    '''
    df = pd.read_parquet(path+'/snana_'+str(ID)+'.parquet', engine='fastparquet')
    return df

def SNID_to_loc(SNID, parq, band, date = False,\
     snpath = '/cwork/mat90/RomanDESC_sims_2024/roman_rubin_cats_v1.1.2_faint/', roman_path = None, host = False):
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
        p, s = radec2point(RA, DEC, band, start, end, roman_path)
        if host:
            return RA, DEC, p, s, start, end, peak, df.host_ra.values[0], df.host_dec.values[0]
        else:
            return RA, DEC, p, s, start, end, peak

def radec2point(RA, DEC, filt, start = None, end = None, path = '/cwork/mat90/RomanDESC_sims_2024'):
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

def construct_psf_source(x, y, pointing, SCA, stampsize=25,  x_center = None, y_center = None, sed = None, flux = 1):
    '''
        Constructs the PSF around the point source (x,y) location, allowing for some offset from the center
        Inputs:
        x,y are locations in the SCA
        pointing, SCA: the pointing and SCA of the image
        stampsize = size of cutout image used
        x_center and y_center need to be given in coordinates of the cutout.
        sed: the SED of the source (XXX CURRENTLY NOT IMPLEMENTED XXX)
        flux: the flux of the source. If you are using this function to build a model grid point, this should be 1. If
            you are using this function to build a model of a source, this should be the flux of the source.

    '''

    config_file = './temp_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=SCA)

    '''
    file_path = r"snflux_1a.dat"
    df = pd.read_csv(file_path, sep = '\s+', header = None, names = ['Day', 'Wavelength', 'Flux'])
    a = df.loc[df.Day == 0]
    del df
    sed = galsim.SED(galsim.LookupTable(a.Wavelength/10, a.Flux, interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
    '''
    sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                                wave_type='nm', flux_type='fphotons')

    master = getPSF_Image(util_ref, stampsize, x=x, y=y,  x_center = x_center, y_center=y_center, sed = sed, include_photonOps=True, flux = flux).array

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
        #print(wcs)
        sca_wcs_list.append(galsim.AstropyWCS(wcs = wcs)) #Made this into a galsim wcs

        pixel = wcs.world_to_pixel(SkyCoord(ra=ra*u.degree, dec=dec*u.degree))

        result = Cutout2D(image[a].data, pixel, size, mode = 'strict', wcs = wcs)
        wcs_list.append(galsim.AstropyWCS(wcs = result.wcs)) # Made this into a galsim wcs

        #print('PIXEL DIFFERENTIAL')
        #print('Pixel Astropy', result.wcs.world_to_pixel(SkyCoord(ra=ra*u.degree, dec=dec*u.degree)))
        #print('Pixel Galsim', wcs_list[-1].toImage(ra, dec, units = 'deg'))


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
    if sed is None:
        sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
    if pixel:
        point = galsim.Pixel(1)*sed
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

    
    result = point.drawImage(self.bpass,wcs=wcs, method='phot', photon_ops=photon_ops, rng=self.rng, \
        n_photons=int(1e6),maxN=int(1e6),poisson_flux=False, center = galsim.PositionD(x_center, y_center),use_true_center = True, image=stamp)
    return result

def fetchImages(testnum, detim, ID, sn_path, band, size, fit_background):
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

def getWeights(cutout_wcs_list,size,snra,sndec, error = None):
    wgt_matrix = []
    gaussian_std = 2.5
    print('Gaussian std in getWeights', gaussian_std)
    for i,wcs in enumerate(cutout_wcs_list):
        xx, yy = np.meshgrid(np.arange(0,size,1), np.arange(0,size,1))
        xx = xx.flatten()
        yy = yy.flatten()
        rara, decdec = wcs.toWorld(xx, yy, units = 'deg')
        dist = np.sqrt((rara - snra)**2 + (decdec - sndec)**2)

        snx, sny = wcs.toImage(snra, sndec, units = 'deg')
        dist = np.sqrt((xx - snx + 1)**2 + (yy - sny + 1)**2)
        
        #wgt = np.zeros(size**2)
        wgt = np.ones(size**2)
        

        wgt = 5*np.exp(-dist**2/gaussian_std)
        
        wgt[np.where(dist > 4)] = 0


        if not isinstance(error, np.ndarray):
            error = np.ones_like(wgt)
        wgt /= error
        wgt = wgt / np.sum(wgt)
        wgt_matrix.append(wgt)
    return wgt_matrix

def makeGrid(adaptive_grid, images,size,ra,dec,cutout_wcs_list, percentiles = [], single_grid_point=False):
    if adaptive_grid:
        a = images[:size**2].reshape(size,size)
        ra_grid, dec_grid = local_grid(ra,dec, cutout_wcs_list[0], \
                npoints, size = size,  spacing = 0.75, image = a, spline_grid = False, percentiles = percentiles)
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




