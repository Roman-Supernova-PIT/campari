## Environment
To create an environment to run this code the following 'module load' will be necessary on NERSC \
On other systems 'conda' may be already in your path. Consult the documentation for the relevant system. \

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

## Doing a simple run on DCC.
The RomanASP code can be run from the command line. Basic arguments are given in the command line and algorithm settings are given via the input file config.yaml.
To do a simple test run to ensure everything is installed correctly, you can request a node:

```
srun -n 1 -N 1 -t 4:00:00 --mem 20000 -p cosmology --account=cosmology --pty bash
conda activate multismp
```

and then run:

```
python RomanASP.py -s 40120913 -b Y106 -t 10 -d 5
```
This will run the algorithm on supernova with SNID 40120913, in band Y106, using 10 images 5 of which contain SN detections.

## Modifying the yaml file.
To actually have the code serve your specific needs, you can modify the yaml file to change which SN are measured and how the fit is performed.

### Basics:

| Parameter             | Type  | Description                                                                                                                           |
|------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------|
| SNID                   | int    | ID of the supernova you want to fit.                                                                                            |
| adaptive_grid          | bool   | If true, use the adaptive grid method. If false, use a standard (rectilinear) grid.                                                        |
| band                   | str    | Which Roman passband to use.                                                                                                   |
| testnum                | int    | Total number of images to utilize in the SMP algorithm.                                                                        |
| detim                  | int    | Number of images with a SN detection in them. Rule of thumb, this should be 1/2 or less of testnum.                           |
| roman_path             | str    | Path to the Roman data on your machine.                                                                                        |
| sn_path                | str    | Path to the Roman SN parquet files on your machine. On DCC, this is roman_path + roman_rubin_cats_v1.1.2_faint/.              |
| size                   | int    | Size of the image stamps in pixels. Should be an odd number for a well-defined central pixel.                                |
| use_real_images        | bool   | If true, use Roman OpenUniverse images. If false, use images you simulate yourself, see the simulating images section below.|
| weighting              | bool   | If true, use a Gaussian weighting centered on the SN. This typically sees improved results.                                 |
| fetch_SED              | bool   | If true, get the SN SED from the OpenUniverse parquet files. If false, use a flat SED. May not be perfectly functional yet. TODO: see if this is improvable. |
| make_initial_guess     | bool   | If true, the algorithm uses an average of the pixel values at each model point to set an initial guess for each model point. Slight improvement in certain cases but not pivotal. |
| source_phot_ops        | bool   | If true, use photon shooting to generate the PSF for fitting the SN. This seemingly needs to be true for a quality fit.     |
| flux_initial_guess | float | When we make the initial guess, every model component gets a starting value. This includes the supernova fluxes. Setting this value chooses the initial flux for the SN in each image before the algorithm solves for the true value. Changing this number has very very little effect on the results and can be safely left at 1. It is only a configuration variable because I thought it was not smart to hard code it. |
|object_type| str | What kind of object are we fitting? 'SN' for supernova, 'star' for star. This difference is important, as a star won't have predetection images, while a transient will. |


### Simulating your own images.
For testing the algorithm, it is often beneficial to simulate our own galaxy and SN rather than use Roman OpenUniverse images. On a normal run, the following options aren't used. If use_real_images is set to false, the following become necessary:

| Parameter             | Type   | Description                                                                                                                           |
|------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------|
| bg_gal_flux            | float  | Total flux of the background galaxy.                                                                                                |
| background_level       | int    | Sky background in simulated images.                                                                                                 |
| noise                  | int    | Standard deviation of Gaussian noise added to images.                                                                              |
| do_rotation            | bool   | If true, successive images are rotated (as they will be for real Roman images).                                                    |
| do_xshift              | bool   | If true, successive images have their centers offset (as they will be for real Roman images).                                     |
| use_roman              | bool   | If true, use a Galsim-generated Roman PSF to create images. If false, use an analytic Airy PSF.                                   |
| mismatch_seds          | bool   | If true, intentionally use a different SED to generate the SN than to fit it later. Useful for testing how much the SED affects the fit. |
| turn_grid_off          | bool   | If true, don't generate a background model at all. Useful for testing just the PSF photometry of the SN if bg_gal_flux is set to zero. |
| single_grid_point      | bool   | See below.                                                                                                                           |
| deltafcn_profile       | bool   | If true, the galaxy is no longer a realistic galaxy profile and instead a Dirac delta function. Combined with single_grid_point, it is hypothetically possible for the algorithm to perfectly recover the background by fitting a Dirac delta to a Dirac delta at the exact same location. TODO: explain this better. |
|sim_ra, sim_dec         | float  | RA and DEC for simulated SN in degrees. |
|base_pointing, base_sca | int    | Pointing and SCA for base Roman image to use for simulation. For instance, this image is used to set the
initial WCS. |


### Experimental
| Parameter          | Type  | Description                                                                                                                                |
|---------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------|
| pixel               | bool  | If true, use a pixel (tophat) function rather than a delta function to be convolved with the PSF in order to build the model.            |
| contour_grid     | bool  | A new method being developed to generate the adaptive grid. Seems to be better! TODO: Consider replacing the default method.             |
| subtract_background      | bool  | If False, add an extra parameter that fits for the mean sky background level. If true, subtract the SKYMEAN from the image header. |


### Not currently used, to be removed.
npoints
method
spline_grid
avoid_non_linearity
make_exact
check_perfection   TODO: Ensure users can use the avoid non linearity and check perfection options.

## Output
All output is stored in the results directory. Two sub directories are created, **images** and **lightcurves**.
### images
3 Outputs are placed in this directory. \
SNID_band_psftype_grid.npy --> ra and dec locations of model points used. \
SNID_band_psftype_wcs.fits --> WCS objects for each image used. \
SNID_band_psftype_images.npy --> pixel values for each image used.

### lightcurves
#### SNID_band_psftype_lc.csv
csv file containing a measured lightcurve for the supernova.

| Parameter            | Type            | Description                                                                                                                                            |
|-----------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| true_flux             | float           | Flux of the supernova from the OpenUniverse truth files.                                                                                              |
| MJD                   | float           | MJD date of the current epoch.                                                                                                                        |
| confusion metric      | float           | An experimental metric measuring how much contamination the background galaxy imparts. It is the dot product of the PSF at the SN location with an image of the galaxy without a supernova detection. Essentially, it is the amount of background flux "under" the SN in a detection image!This metric seems to roughly correlate with measurement error but requires further investigation. |
| host_sep              | float           | Separation between galaxy center and SN, from OpenUniverse truth files.                                                                              |
| host_mag_g            | float           | Host galaxy magnitude in g band, from OpenUniverse truth files.                                                                                      |
| sn_ra                 | float           | RA location of the SN, from OpenUniverse truth files.                                                                                                |
| sn_dec                | float           | DEC location of the SN, from OpenUniverse truth files.                                                                                               |
| host_ra               | float           | RA location of the host galaxy, from OpenUniverse truth files.                                                                                       |
| host_dec              | float           | DEC location of the host galaxy, from OpenUniverse truth files.                                                                                      |
| measured_flux         | float           | Flux as measured by the RomanASP algorithm.                                                                                                           |














