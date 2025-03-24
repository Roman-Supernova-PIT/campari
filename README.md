## Environment
To create an environment to run this code \
The following 'module load' will be necessary on NERSC \
On other systems 'conda' may be already in your path. \
Consult the documentation for the relevant system. \

```
module load conda
```
### Create our conda environment.  

Will fill it with pip installable Python libraries below.
```
conda create -n multismp ipykernel python=3.11
```
```
conda activate multismp
conda install -c conda-forge fitsio
pip install -r requirements.txt
```
#### Create a Jupyter kernel with this environment
```
python -m ipykernel install --user --name multismp --display-name multismp
```

## Doing a simple run.
The RomanASP code can be run from the command line. For now, the command line takes no arguments and instead all input is done via the input file config.yaml. 
To do a simple test run to ensure everything is installed correctly, you can just run:
```
python RomanASP.py
```

## Modifying the yaml file.
To actually have the code serve your specific needs, you can modify the yaml file to change which SN are measured and how the fit is performed.

### Basics:
**SNID:** int, ID of the supernova you want to fit.
**adaptive_grid:** bool, if true, use the adaptive grid method. If false, use a standard grid. \
**band:** str, which Roman passband to use \
**testnum:** int, total number of images to utilize in the SMP algorithm \
**detim:** int, number of images with a SN detection in them. Rule of thumb, this should be 1/2 or less of testnum. \
**roman_path:** str, path to the roman data on your machine \
**sn_path:** str, path to the roman SN parquet files on your machine. On DCC, this is roman_path + roman_rubin_cats_v1.1.2_faint/ \
**size:** int, size of the image stamps in pixels. Should be an odd number for a well defined central pixel. \
**use_real_images:**  bool, if true, use roman OpenUniverse images. If false, use images you simulate yourself, see the simulating images section below. \
**weighting:** bool, if true, use a Gaussian weighting centered on the SN. This typically sees improved results.\
**fetch_SED:** bool, if true, get the SN SED from the OpenUniverse parquet files. If false, use a flat SED. This may not be perfectly functional yet, as it does not seem to improve results. TODO: see if this is improvable\
**make_initial_guess:** bool, if true, the algorithm uses an average of the pixel values at each model point to set an initial guess for each model point. Have seen slight improvement in certain cases but certainly not pivotal to set to true.\
**source_phot_ops:** bool, if true, use photon shooting to generate the PSF for fitting the SN. This seemingly needs to be true for a quality fit.

### Simulating your own images.
For testing the algorithm, it is often beneficial to simulate our own galaxy and SN rather than use Roman OpenUniverse images. On a normal run, the following options aren't used. If use_real_images is set to false, the following become necessary:  
**bg_gal_flux:** float, total flux of the background galaxy.\
**background_level:** int, sky background in simulated images.\
**noise:** int, std of Gaussian noise added to images.\
**do_rotation:** bool, if true, sucessive images are rotated (as they will be for real Roman images)\
**do_xshift:** bool, if true, sucessive images have their centers offset (as they will be for real Roman images)\
**use_roman:** bool, if true, use a galsim-generated Roman PSF to create images, if false, use an analytic Airy PSF.\
**mismatch_seds:** bool, if true, intentionally use a different SED to generate the SN than to fit it later. This is useful for testing how much the SED matters in our fit.\
**turn_grid_off:** bool, if true, don't generate a background model at all. This is useful for testing just the PSF photometry of the SN if you also set bg_gal_flux to zero.\
**single_grid_point:** see below  
**deltafcn_profile:** If true, the galaxy is no longer a realistic galaxy profile and is rather a dirac delta function. Setting this to true along with single_grid_point above means that it is hypothetically possible that the algorithm can perfectly recover the background, since a single dirac delta is being fit to a single dirac delta at the exact same location. TODO: explain this better.\

### Experimental
**pixel:** bool, if true, use a pixel (tophat) function rather than a delta function to be convolved with the PSF in order to build the model.\
**makecontourGrid:** bool, a new method I am working on to generate the adaptive grid. Seems to be better! TODO: Consider replacing the default method.\
**fit_background:** bool, if true, add an extra parameter that fits for the mean sky background level. Since we have the exact number in the image header, this should be false.\


### Not currently used, to be removed.
npoints
method
spline_grid
avoid_non_linearity
make_exact
check_perfection   TODO: Ensure users can use the avoid non linearity and check perfection options. 

## Output
All output is stored in the results directory. Two sub directories are created. 
### images
3 Outputs are placed in this directory. \
SNID_band_psftype_grid.npy --> ra and dec locations of model points used. \
SNID_band_psftype_wcs.fits --> WCS objects for each image used. \
SNID_band_psftype_images.npy --> pixel values for each image used. 

### lightcurves 
#### SNID_band_psftype_lc.csv 
csv file containing a measured lightcurve for the supernova. \
**true_flux:** flux of the supernova from the OpenUniverse truth files \
**MJD:** MJD date of the current epoch. \
**confusion metric:** A current experimental metric that measures how much contamination the background galaxy imparts. Formally, it is the dot product of an evaluation of the PSF at the SN location with an image of the galaxy with no supernova detection. Essentially, it is the amount of background flux "under" the SN in a detection image! This metric seems to roughly correlate with measurement error but I need to look into this further.\
**host_sep:** Seperation between galaxy center and SN, from OpenUniverse truth files.\
**host_mag_g:** Host galaxy magnitude in g band, from OpenUniverse truth files.\
**sn_ra:** RA location of SN, from OpenUniverse truth files.\
**sn_dec:** DEC location of SN, from OpenUniverse truth files.\
**host_ra:** RA location of host galaxy, from OpenUniverse truth files.\
**host_dec:** DEC location of host galaxy, from OpenUniverse truth files.\
**measured_flux:** Flux as measured by the RomanASP algorithm.














