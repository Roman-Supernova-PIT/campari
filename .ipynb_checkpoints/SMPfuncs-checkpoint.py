import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from matplotlib import pyplot as plt
from roman_imsim.utils import roman_utils
import matplotlib
from matplotlib import pyplot as plt
import astropy.table as tb
import warnings 
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)
import scipy.sparse as sp 
#from pixmappy import Gnomonic
from scipy.linalg import block_diag, lstsq
from numpy.linalg import LinAlgError
from astropy.nddata import Cutout2D
import galsim
#from reproject import reproject_interp
print('Updated version of SMP')
'''
def calibration(pointing, SCA, band, snid = None):
    cat = pd.read_csv(f'/cwork/mat90/RomanDESC_sims_2024/RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{SCA}.txt',\
                              sep="\s+", skiprows = 1,
                              names = ['object_id', 'ra', 'dec', 'x', 'y', 'realized_flux', 'flux', 'mag', 'obj_type'])
    cat = cat.loc[cat['obj_type'] == 'star']
    logflux = -2.5*np.log10(cat['flux'])
    mag = cat['mag']
    zpt = np.mean(mag - logflux)
    if snid is not None:
        print('SNID:', snid)
        print(cat.loc[cat['object_id'] == snid].mag, 'Mag Goal')
    del cat
    return zpt
'''

def downsample(array,factor):
    """
    Downsample an array by a factor of `factor` in each dimension
    """
    
    xsize = array.shape[0]
    ysize = array.shape[1]
    '''
    output = np.zeros((xsize//factor,ysize//factor))
    assert xsize % factor == 0 and ysize % factor == 0, "Array size must be divisible by factor size is " + str(xsize) + " " + str(ysize) + " factor is " + str(factor)
    for i in range(xsize//factor):
        for j in range(ysize//factor):
            output[i,j] = np.mean(array[i*factor:(i+1)*factor + 1, j*factor:(j+1)*factor + 1]) #I am unclear if this +1 is correct XXX TODO

    return output
    '''
    # Assuming array is your input array and factor is the downsampling factor
    xsize, ysize = array.shape
    assert xsize % factor == 0 and ysize % factor == 0, "Array size must be divisible by factor size is " + str(xsize) + " " + str(ysize) + " factor is " + str(factor)
    new_xsize, new_ysize = xsize // factor, ysize // factor
    # Reshape and compute the mean
    reshaped_array = array[:new_xsize * factor, :new_ysize * factor].reshape(new_xsize, factor, new_ysize, factor)
    output = reshaped_array.mean(axis=(1, 3))
    return output



def construct_psf_background(ra, dec, pointing, scanum, wcs, x_loc, y_loc, stampsize, flatten = True, color=0.61):
    #removed center_Ra and center_Dec

    '''
    Constructs the background model using PIFF's PSFs around a certain image (x,y) location and a given array of RA and DECs.
    The pixel coordinates are found using pixmappy's WCSs 
    stampsize determines how large the image will be (eg stampsize = 30 means a 30x30 image). 
    flatten decides if the image should be flattened (preferred) or not
    '''
    #print('Constructing psf background...')

    
  
    osample = 8
    #print(x_loc, y_loc, 'x_loc and y_loc')

    x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree))
    #print('The following X and Y are the locations of the grid in the cutout frame')
    #print('x and y')
    #print(np.min(x), np.max(x))
    #print(np.min(y), np.max(y))

    #x += 0.5
    #y += 0.5
    #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.


    x_center = np.median(x)
    y_center = np.median(y)



    if type(x_loc) == np.ndarray and np.size(x_loc) > 1:
        x_loc = x_loc[0]
        y_loc = y_loc[0]


    config_file = '/hpc/home/cfm37/my_tds.yaml' 

    xdists = 2*np.abs(x_loc - np.rint(x).astype(int))
    ydists = 2*np.abs(y_loc - np.rint(y).astype(int))
    alldists = np.concatenate((xdists, ydists))
    
    bonussize = stampsize * 2
    total = stampsize + bonussize
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=scanum)
    master = util_ref.getPSF_Image(total, x=x_loc, y=y_loc, oversampling_factor = osample).array 
    center = osample*total//2


    x_over = x * osample
    y_over = y * osample #Oversampling by a factor of 8
    
    x_over = np.rint(x_over).astype(int)
    y_over = np.rint(y_over).astype(int)


    deltax = x_over - x_center *osample
    deltay = y_over - y_center *osample
    psfs = np.zeros((stampsize * stampsize,np.size(deltax)))

    k = 0
    for i,j in zip(deltax.flatten(),deltay.flatten()):
        cutout = Cutout2D(master, (center-i + osample//2, center-j+osample//2), stampsize*osample, mode = 'strict') #I am not sure why this +osample//2 is necessary, need to figure it out before using
        down = downsample(cutout.data,osample)
        psfs[:,k] = down.flatten()
        k += 1

    del master
    del cutout
    del down

    return psfs


def getPSF_Image(refim,stamp_size,x=None,y=None,pupil_bin=8,sed=None,
                        oversampling_factor=1,include_photonOps=True, n_phot=1e6, offset=None, x_center = None, y_center = None):
        """
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
        if sed is None:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
        else:
            pass
            #print('Using input SED')
        point = galsim.DeltaFunction()*sed
        point = point.withFlux(1,refim.bpass)
        local_wcs = refim.getLocalWCS(x,y)
        wcs = galsim.JacobianWCS(dudx=local_wcs.dudx/oversampling_factor,
                                dudy=local_wcs.dudy/oversampling_factor,
                                dvdx=local_wcs.dvdx/oversampling_factor,
                                dvdy=local_wcs.dvdy/oversampling_factor)
        stamp = galsim.Image(stamp_size*oversampling_factor,stamp_size*oversampling_factor,wcs=wcs)
        if not include_photonOps:
            psf = galsim.Convolve(point, refim.getPSF(x,y,pupil_bin))
            return psf.drawImage(refim.bpass,image=stamp,wcs=wcs,method='no_pixel', offset=offset)
        photon_ops = [refim.getPSF(x,y,pupil_bin)] + refim.photon_ops
        return point.drawImage(refim.bpass,
                                method='phot',
                                rng=refim.rng,
                                maxN=int(1e6), #This needs to be changed back to 1e6 once I figure out memory issues XXX TODO
                                n_photons=int(1e6),
                                image=stamp,
                                photon_ops=photon_ops,
                                poisson_flux=False,
                                center = galsim.PositionD(x_center + 1, y_center + 1))

def construct_psf_source(x, y, pointing, SCA, stampsize=25,  x_center = None, y_center = None, sed = None):
    '''
        Constructs the PIFF PSF around the point source (x,y) location, allowing for some offset from the center
        (if so, specify x_center and y_center)
        x,y are locations in the SCA
        x_center and y_center need to be given in coordinates of the cutout

    '''
    
    #Need to customize band stuff here too XXX TODO 
    print('while cwork is down')
    config_file = '../temp_tds.yaml'
    #config_file = '/hpc/home/cfm37/my_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=SCA)

    #sed = sed,
    
    file_path = r"snflux_1a.dat"
    df = pd.read_csv(file_path, sep = '\s+', header = None, names = ['Day', 'Wavelength', 'Flux'])
    a = df.loc[df.Day == 0]
    del df
    sed = galsim.SED(galsim.LookupTable(a.Wavelength/10, a.Flux, interpolant='linear'),
                            wave_type='nm', flux_type='fphotons')
    
    print('For source, x y x cen y cen', x, y, x_center, y_center)
    master = getPSF_Image(util_ref, stampsize, x=x, y=y,  x_center = x_center, y_center=y_center, sed = sed).array
    
    return master.flatten()

#def local_grid(ra_center, dec_center, step, npoints):
def local_grid(ra_center, dec_center, wcs, npoints, size = 25):

    '''
    Generates a local grid around a RA-Dec center, choosing step size and number of points
    '''

    extra = 3
    x = np.linspace(-extra, size+extra, npoints)
    y = np.linspace(-extra, size+extra, npoints)

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    print('Creating Grid with +', extra)
    result = wcs.pixel_to_world(xx, yy)
    ra_grid = result.ra.deg
    dec_grid = result.dec.deg

    return ra_grid, dec_grid

class Detection:
    '''
    Main class for SMP. Requires RA and Dec for the detection, an exposure and CCD numbers for bookkeeping and
    zero-point retrieval, a band (for finding extra exposures) and an optional color (for astrometry) and name for the 
    detection 
    '''
    def __init__(self, ra, dec, pointing, scanum, band, start_mjd, end_mjd, peak_mjd, color = 0.61, name = '', snid = None):
        '''
        Constructor class
        '''
        self.ra = ra 
        self.dec = dec 
        self.pointing = pointing
        self.scanum = scanum
        self.band = band
        self.color = color
        self.name = name
        self.common_zpt = 18.8
        self.start_mjd = start_mjd
        self.end_mjd = end_mjd
        self.peak_mjd = peak_mjd
        if snid is not None:
            self.snid = snid
        else:
            self.snid = None


    
    def findAllExposures(self, truth = 'simple_model', maxnum = None, return_list = False): 
        truth = 'simple_model' #THIS IS A TEST CHANGE THIS BACK XXX XXX TODO
        print('FAE TRUTH:', truth)

        RA = self.ra
        DEC = self.dec
        band = self.band
        date = self.peak_mjd
        start = self.start_mjd
        end = self.end_mjd



        f = fits.open('/cwork/mat90/RomanDESC_sims_2024/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits')[1]
        f = f.data

        if date != None:
            g = fits.open('/cwork/mat90/RomanDESC_sims_2024/RomanTDS/Roman_TDS_obseq_11_6_23.fits')[1]
            g = g.data
            alldates = g['date']
            #datesep = np.abs(alldates - date)
            
        explist = tb.Table(names=('Pointing', 'SCA', 'BAND', 'zeropoint', 'RA', 'DEC'), dtype=('i8', 'i4', 'str', 'f8', 'f8', 'f8'))

        allRA = f['RA']
        allDEC = f['DEC']
        dist = np.sqrt((allRA - RA)**2 + (allDEC - DEC)**2)


        cut = np.any(dist < 0.2, axis = 1)
        testdates = alldates[np.any(dist < 0.2, axis = 1)]

        cut2 = f['filter'] == band
        testdates = alldates[cut2]


        testdates = alldates[np.any(dist < 0.2, axis = 1) & (f['filter'] == band)]




        dist[np.where(f['filter'] != band)] = np.inf #Ensuring we only get the filter we want
        dist[np.where((alldates > start) & (alldates < end))] = np.inf #Ensuring we only get dates where SN is off
        coords = np.where(dist <= 0.2)
        
        print(np.size(coords[0]), 'Potential Candidates')

        testnum = 0
        for pointing, SCA in zip(coords[0],coords[1]+1):
            testnum += 1
            f = fits.open(f'/cwork/mat90/RomanDESC_sims_2024/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{SCA}.fits.gz')
            a = 0 if truth == 'truth' else 1
            fitsfile = f[a] 
            w = WCS(fitsfile.header)

            pixel = w.world_to_pixel(SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)) #Get the pixel at desired RA/DEC
            #pixel = (pixel[0]+0.5, pixel[1]+0.5)
            #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.
            
            if pixel[0] < 0 or pixel[0] > 4095 or pixel[1] < 0 or pixel[1] > 4095:
                continue
            else:
                print('Found Exposure')
                zpt = fitsfile.header['ZPTMAG']
                central = w.pixel_to_world(2048,2048)

                explist.add_row([pointing,SCA,band, zpt, central.ra.deg, central.dec.deg])
                if maxnum != None:
                    if len(explist) == maxnum:
                        break
        del f
        del g

                
        if self.pointing not in explist['Pointing']:
            f = fits.open(f'/cwork/mat90/RomanDESC_sims_2024/RomanTDS/images/{truth}/{self.band}/{self.pointing}/Roman_TDS_{truth}_{self.band}_{self.pointing}_{self.scanum}.fits.gz')
            if truth == 'truth':
                fitsfile = f[0]
            else:
                fitsfile = f[1] 
            zpt = fitsfile.header['ZPTMAG']
            explist.add_row([self.pointing,self.scanum,self.band, zpt, self.ra, self.dec])
            print('Manually Added detection exposure ######################################')
        
        explist['DETECTED'] = False 
        explist['DETECTED'][explist['Pointing'] == self.pointing] = True  

        self.exposures = tb.unique(explist)
        self.num_exposures = np.size(self.exposures)

        #if reduce_band:   #Change this! XXX TODO
            #self.exposures = self.exposures[self.exposures['BAND'] == self.band]
        self.exposures.sort('DETECTED')
        f.close()
        
        if return_list:
            return self.exposures

        
    

    def findPixelCoords(self, pointing = None, scanum = None, return_wcs = False, color = 0.61, ra = None, dec = None, cutout = False):
        '''
        Finds the pixel coordinates of the detection using pixmappy (data provided using the pmc argument)
        for a given exposure/ccdnum pair. Can return the wcs for usage in other functions
        Color (g-i) is optional
        '''
        if pointing is None:
            pointing = self.pointing
        if scanum is None:
            scanum = self.scanum

        if ra is None:
            ra = self.ra
            dec = self.dec

        #truth = 'truth' #THIS IS A TEST CHANGE THIS BACK XXX XXX TODO
        truth = 'simple_model'


        image = fits.open(f'/cwork/mat90/RomanDESC_sims_2024/RomanTDS/images/{truth}/{self.band}/{pointing}/Roman_TDS_{truth}_{self.band}_{pointing}_{scanum}.fits.gz')
        if truth == 'truth':
            wcs = WCS(image[0].header)
        else:
            wcs = WCS(image[1].header)

        #need to get rid of hardcoded 25
        #Trying to adjust images

        if cutout:
            print('Find Pix is using a cutout')
            cutout = Cutout2D(image[1].data, SkyCoord(ra = np.array(self.ra)*u.degree, dec = np.array(self.dec)*u.degree), 25, mode = 'strict', wcs = wcs)
            wcs = cutout.wcs
        #print(wcs)


        #x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree)) #deleted c = color here
        print('Using world 2 pix')
        x, y = wcs.all_world2pix(ra,dec, 0)

        #x += 0.5
        #y += 0.5
        #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.
        #print('Find pixel coords has 0.5 added')
 
        

        image.close()
        if return_wcs:
            return x, y, wcs
        else:
            return x, y
    
    def constructImages(self, size = 25, background = False, roman_path = None):
    
            '''
            Constructs the array of images in the format required for the linear algebra operations
            - zeropoints is a dictionary of ZP for each exposure/ccdnum, all exposures are brought to a common
            zeropoint = 30. 
            - path provides the location of all FITS for the exposures, the stamps should be
            names as {name}_EXPNUM.fits 
            - size is the size for the grid (size = 30 means 30x30 stamps)
            - background applies some background subtraction routines developed for the comet analysis

            '''
            #print('Constructing Images...')

            #will be used for gain corrections later on
            #zpt = self.exposures['zeropoint'][np.where((self.exposures['Pointing'] == self.pointing)&(self.exposures['SCA'] == self.scanum))][0]
            print('This zpt line commented out')
            #self.zp = np.power(10, -(zpt - self.common_zpt)/2.5) #Need to figure out zeropointing XXX TODO


            m = []
            mask = []
            wgt = []
            bgflux = []

            for i in self.exposures:
                #try:   This has to be put back

                truth = 'simple_model'
                band = i['BAND']
                pointing = i['Pointing']
                SCA = i['SCA']
                image = fits.open(roman_path + f'/RomanTDS/images/{truth}/{band}/{pointing}/Roman_TDS_{truth}_{band}_{pointing}_{SCA}.fits.gz')

                #Switched w to wcs here
                if truth == 'truth':
                    wcs = WCS(image[0].header)
                    a = 0
                else:
                    wcs = WCS(image[1].header)
                    a = 1


                pixel = wcs.world_to_pixel(SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree))
                print('pixel', pixel)
                #pixel = (pixel[0] + 0.5, pixel[1] + 0.5)
                #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.


                #XXX This should be put back ! XXX
                #try:
                result = Cutout2D(image[a].data, pixel, size, mode = 'strict', wcs = wcs)
                if i['DETECTED']:
                    self.xpix, self.ypix = result.wcs.world_to_pixel(SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree))
                    #self.xpix += 0.5
                    #self.ypix += 0.5
                    #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.


                '''       
                except:
                    if i['DETECTED']:
                        print('No stamp for the detection!')
                    m.append(np.zeros(size*size))
                    mask.append(np.ones(size*size))
                    wgt.append(np.zeros(size*size))
                    continue
                '''

                ff = 1
                cutout = result.data
                if truth == 'truth':
                    img = Cutout2D(image[0].data, pixel, size, mode = 'strict').data 
                    img += np.abs(np.min(img))
                    img += 1

                    img = np.sqrt(img)
                    err_cutout = 1 / img

                else:
                    err_cutout = Cutout2D(image[2].data, pixel, size, mode = 'strict').data #I think this is supposed to be image[2] XXX TODO
           
                #This also has to be put back
                '''
                except OSError:
                    if i['DETECTED']:
                        print('No stamp for the detection!')
                    m.append(np.zeros(size*size))
                    mask.append(np.ones(size*size))
                    wgt.append(np.zeros(size*size))
                    continue
                '''

                try:
                
                    zero = np.power(10, -(i['zeropoint'] - self.common_zpt)/2.5)
                except:
                    print('failed')
                    zero = -99

                if zero < 0:
                    zero = 0

                im = cutout * zero #removed .data
                bgarr = np.concatenate((im[0:size//10,0:size//10].flatten(),\
                                         im[0:size,size//10:size].flatten(),\
                                              im[size//10:size,0:size//10].flatten(),\
                                                  im[size//10:size,size//10:size].flatten()))
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

                if background:
                    im -= bg 
                    print('Subtracted a BG of', bg)

                m.append(im.flatten())
                mask.append(np.zeros(size*size))
                w = zero**2/err_cutout.flatten()
                w[err_cutout.flatten() == 0] = 0 

                wgt.append(w)
            
                
            self.image = np.hstack(m)
            self.mask = np.hstack(mask)
            self.bgflux = bgflux

            self.wgt = np.hstack(wgt)
            print('WGT shape', np.shape(self.wgt))

            self.invwgt = 1/self.wgt

            self.invwgt[self.mask > 0] = 0
            self.invwgt[self.wgt == 0] = 0

        
    
    def constructPSFs(self, ra_grid=None, dec_grid=None, size = 25, shift_x = 0, shift_y = 0, path = '', sparse = False):
        '''
        Constructs the PIFF PSFs for the detections, requires an array of RA and Decs (ra_grid, dec_grid), a pixmappy instance (pmc),
        a stamp size, a potential offset in pixels for the center (shift_x,y), a path for the 
        PIFF files. 
        sparse turns on the sparse matrix solution (uses less memory and can be faster, but less stable)
        '''
        print('Constructing PSFs...')
        if ra_grid is None:
            print('This probably should not be getting called ##################')
            ra_grid, dec_grid = local_grid(self.ra, self.dec, 0.11/3600, 25)  #changed from 0.11/3600 to 0.11 to 0.13
        #Need to make sure this isn't hardcoded:
        #config_file = '/hpc/home/cfm37/my_tds.yaml'
        psf_matrix = []
        self.x, self.y = self.findPixelCoords()

        x_center, y_center = self.cutout_wcs.world_to_pixel(SkyCoord(ra = self.ra*u.degree, dec = self.dec*u.degree)) 
        #x_center += 0.5
        #y_center += 0.5
        #Astropy defines pixel coordinates as the center of the pixel, so we add 0.5.
        
        self.psf_source = construct_psf_source(self.x, self.y, self.pointing, self.scanum, stampsize = size, x_center = x_center, y_center=y_center)
        

        for i in self.exposures:
            try:
                print('Calculating x_cen using this wcs')
                x_cen, y_cen, wcs = self.findPixelCoords(i['Pointing'], int(i['SCA']), return_wcs=True, color = self.color)
 
            except (OSError, ValueError):
                print(f"Missing {i['Pointing']} {i['SCA']} psf")
                psf_matrix.append(sp.csr_matrix(np.zeros((size * size, len(ra_grid)))))   
                continue 
            psf_matrix.append(sp.csr_matrix(construct_psf_background(ra_grid, dec_grid, i['Pointing'], int(i['SCA']), wcs, x_cen, y_cen, size, flatten=True)))
           
        if sparse:
            self.psf_matrix = sp.vstack(psf_matrix).toarray()
            del psf_matrix
        else:
            dense_psf_matrix = [matrix.toarray() for matrix in psf_matrix]  #Pedro's code didn't have to do this but mine doesn't work w/o it? Confused XXX
            self.psf_matrix = np.vstack(dense_psf_matrix) 
        

        ## Last PSF is the one for the detected exposure 

        #config_file = '/hpc/home/cfm37/my_tds.yaml'
        #util_ref = roman_utils(config_file=config_file, visit=self.pointing, sca=self.scanum)
        #self.source_psf = util_ref.getPSF_Image(size, x = self.x, y = self.y).array 
        

    def constructDesignMatrix(self, size, sparse = False, background = True):
        '''
        Constructs the design matrix for the solution. 
        size is the stamp size, sparse turns on the sparse solution
        background defines whether the background is being fit together with the image or not
        '''
        if not background:
            ones = np.ones((size*size,1))
        else:
            ones = np.zeros((size*size, 1))

        if sparse:
            background = sp.block_diag(len(self.exposures) * [ones] )
        else:
            background = block_diag(*(len(self.exposures) * [ones]))

        psf_zeros = np.zeros((self.psf_matrix.shape[0]))
        psf_zeros[-size*size:] = self.psf_source

        if sparse:
            self.design = sp.hstack([self.psf_matrix, background, np.array([psf_zeros]).T], dtype='float64')
        else:
            #self.design = sp.csc_matrix(self.design)
            self.design = np.column_stack([self.psf_matrix, background, psf_zeros])


    def solvePhotometry(self, res = True, err = True, sparse = False):
        '''
        Solves the system for the flux as well as background sources
        Solution is saved in det.X, the flux is the -1 entry in this array
        - res: defines if the residuals should be computed
        - err: defines if the errors should be computed (requires an expensive matrix inversion)
        - sparse: turns on sparse routines. Less stable, possibly incompatible with `err`
        '''
        if sparse:
            diag = sp.diags(np.sqrt(self.invwgt))
            prod = diag.dot(self.design)
            self.X = sp.linalg.lsqr(prod, self.image*np.sqrt(self.invwgt))[0]
        else:
            self.X = lstsq(np.diag(np.sqrt(self.invwgt)) @ self.design, self.image*np.sqrt(self.invwgt))[0]
        
        self.flux = self.X[-1] 


        #self.flux = 31908.2731 #dont forget to remove this test XXX XXX XXX XXX

        
        #zeropoint = calibration(self.pointing, self.scanum, self.band, snid = self.snid)
      
        print('Using common zpt here now')
        #self.mag = -2.5*np.log10(self.flux) + zeropoint
        self.mag = -2.5*np.log10(self.flux) + self.common_zpt

        if res:
            self.pred = self.design @ self.X 
            self.res = self.pred - self.image
            cut = np.where(np.abs(self.image) > 0)
            print(np.sum(self.res[cut]**2 / np.abs(self.image[cut])) / np.size(self.pred), 'chi2 ############################## <-------')

        if err:
            inv_cov = self.design.T @ np.diag(self.invwgt) @ self.design
            try:
                self.cov = np.linalg.inv(inv_cov)
            except LinAlgError:
                self.cov = np.linalg.pinv(inv_cov)
                
            self.sigma_flux = np.sqrt(self.cov[-1,-1])
            self.sigma_mag = 2.5*np.sqrt(self.cov[-1,-1]/(self.flux**2))/np.log(10)

    def runPhotometry(self, n_grid = 25, size = 25, offset_x = 0, offset_y = 0, \
                      sparse = False, err = True, res = True, background = False):
        '''
        Convenience function that performs all operations required by the photometry
        - se_path: path for the SE postage stamps
        - piff_path: path for the PIFF files
        - zp: zeropoint dictionary
        - survey: `DESTNOSIM` list of exposures
        - pmc: pixmappy instance for astrometry
        - n_grid: grid size for point sources in the background (adds n_grid x n_grid sources)
        - size: stamp size
        - offset_x,y: offset in the x and y pixel coordinates
        - sparse: sparse routines
        - err: turns on error estimation
        - res: computes residuals
        - background: background estimation
        '''
        self.findAllExposures(maxnum = 5) #This is a test change this back XXX TODO
        stepsize =0.07/3600 

        
         #changed from 0.11/3600 to 0.11, to 0.13
        self.constructImages(size = size, background = background)
        #Swapped the order of these two ^ v
        ra_grid, dec_grid = local_grid(self.ra, self.dec, self.cutout_wcs, n_grid, size)
        self.constructPSFs(ra_grid, dec_grid, size, offset_x, offset_y, sparse = sparse)
        self.constructDesignMatrix(size, sparse, background = background)
        self.solvePhotometry(sparse = sparse, err = err, res = res)
        

#old: 10.92716323444293, -43.58891625058177,7977, 6, 'F184'
#new: 8.950999846453639, -43.39788176865921,57278, 1, 'F184'
#even newer 9.897857262526113, -41.88108165662882, 36470, 2, 'F184'

def open_parq(ID):
    df = pd.read_parquet('/cwork/mat90/RomanDESC_sims_2024/roman_rubin_cats_v1.1.2_faint/snana_'+str(ID)+'.parquet', engine='fastparquet')
    return df

def radec2point(RA, DEC, filt, start = None, end = None):
    #This function takes in RA and DEC and returns the pointing and SCA with
    #center closest to desired RA/DEC
    f = fits.open('/cwork/mat90/RomanDESC_sims_2024/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits')[1]
    f = f.data

    g = fits.open('/cwork/mat90/RomanDESC_sims_2024/RomanTDS/Roman_TDS_obseq_11_6_23.fits')[1]
    g = g.data
    alldates = g['date']


    #mask = np.zeros(np.size(alldates))
    #mask[np.where((f['filter'] == filt) & (g['date'] > start) & (g['date'] < end))] = 1




    allRA = f['RA']
    allDEC = f['DEC']

    dist = np.sqrt((allRA - RA)**2 + (allDEC - DEC)**2)



    if start is not None:
        dist[np.where(alldates < start)] = np.inf
        dist[np.where(alldates > end)] = np.inf

    
    #dist = np.sqrt((allRA[:,:,np.newaxis] - RA)**2 + (allDEC[:,:,np.newaxis] - DEC)**2)
    dist[np.where(f['filter'] != filt)] = np.inf #Ensuring we only get the filter we want




    reshaped_array = dist.flatten()
    # Find the indices of the minimum values along the flattened slices
    min_indices = np.argmin(reshaped_array, axis=0)
    # Convert the flat indices back to 2D coordinates
    rows, cols = np.unravel_index(min_indices, dist.shape[:2])
    #f.close()
    #g.close()

    return rows, cols + 1

def SNID_to_loc(SNID, parq, band = 'F184', date = False):
    print('looking in band', band)
    df = open_parq(parq)
    df = df.loc[df.id == SNID]
    RA, DEC = df.ra.values[0], df.dec.values[0]
    start = df.start_mjd.values
    end = df.end_mjd.values
    peak = df.peak_mjd.values
    
    if not date:
        p, s = radec2point(RA, DEC, band)
        return RA, DEC, p, s
    else:
        p, s = radec2point(RA, DEC, band, start, end)
        return RA, DEC, p, s, start, end, peak


def calibration(pointing, SCA, band, snid = None, return_mag = False, roman_path = None):
    cat = pd.read_csv(roman_path+f'/RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{SCA}.txt',\
                              sep="\s+", skiprows = 1,
                              names = ['object_id', 'ra', 'dec', 'x', 'y', 'realized_flux', 'flux', 'mag', 'obj_type'])
    if snid is not None:
        if np.size(cat.loc[cat['object_id'] == snid].flux.values) == 0:
            print('No Flux Found')
            return None, None
        true_mag = cat.loc[cat['object_id'] == snid].mag.values[0]
        print(true_mag, 'Mag Goal')
        print(cat.loc[cat['object_id'] == snid].flux.values[0], 'Flux Goal')
    cat = cat.loc[cat['obj_type'] == 'star']
    logflux = -2.5*np.log10(cat['flux'])
    mag = cat['mag']
    zpt = np.mean(mag - logflux)

    del cat

    if return_mag:
        return zpt, true_mag
    else:
        return zpt
