## Environment
To create an environment to run this code the following 'module load' will be necessary on NERSC \
On other systems 'conda' may be already in your path. Consult the documentation for the relevant system. \

```
module load conda
```
### Create our conda environment.

This code uses the sn_pit_dev environment shared by multiple codes from the SN PIT team. See, e.g. phrosty or SFFT.
To install:

```
git clone https://github.com/Roman-Supernova-PIT/environment.git
cd environment/
bash env_setup.sh
```
If you get an error when running the last command referring to `jdavis`, go into the `sn_pit_dev.yaml` file in `environment` and comment out the `- jdavis` line.

Then once that finishes, copy and paste the location it places the environment. For instance, for me, it's `/global/u1/c/cmeldorf/environment/envs/sn-pit-dev` :
```
# To activate this environment, use
#
#     $ conda activate /global/u1/c/cmeldorf/environment/envs/sn-pit-dev
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```
and then run:
```
conda rename -p YOUR_PATH_HERE sn_pit_dev
```
and finally:
```
conda activate sn_pit_dev
```

## Doing a simple run.
The campari code can be run from the command line. Basic arguments are given in the command line and algorithm settings are given via the input file config.yaml. Because of different file paths on different
systems, the steps are slightly different for each machine. Here's how to get a basic run going dpeending on which computer you find yourself using:


### SMDC:
To do a simple test run on SMDC, try the following:

```
salloc --nodes 1 --qos interactive --time 04:00:00 -p mem-med

bash /data/snpit/env/singrun_smdc_ricksim.sh

cd /home/packages/

git clone https://github.com/Roman-Supernova-PIT/campari/

cd campari

git checkout SMDC_updates

export SNPIT_CONFIG=/home/packages/campari/examples/SMDC/campari_config_ricksims.yaml

cd ../..

pip install -e /home/packages/campari -e /home/packages/snappl

mkdir -p /dev_storage/campari_debug_dir

python packages/campari/campari/RomanASP.py -f F106 --ra 9.376416 --dec -43.946209 --diaobject-collection manual --diaobject-name coolsne --image-collection snpitdb --image-provenance-tag ricksim202608 --image-process load_ricksim --transient_start 60400 --nprocs  1  --photometry-campari-psf-transient_class gaussian  --photometry-campari-psf-galaxy_class gaussian -t 1 -n 1 --photometry-campari-grid_options-type regular --no-save-to-db


```
This will run the algorithm on one SNe from Richard Kessler's simulations.Sure

### NERSC
To do a simple test run to ensure everything is installed correctly, you can request a node:

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m4385
conda activate sn_pit_dev
```
cd into your directory where the code is stored.
Then, in the `config.yaml` file, ensure that `roman_path` and `sn_path` read as follows:
```
roman_path: /global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data
sn_path: /global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE
```

Next, in the `temp_tds.yaml` file, make sure `file_name` is:
```
file_name: /global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data/RomanTDS/Roman_TDS_obseq_11_6_23.fits
```
and then run:

```
python RomanASP.py -s 40120913 -f Y106 -t 10 -d 5
```
This will run the algorithm on supernova with SNID 40120913, in band Y106, using 10 images 5 of which contain SN detections.~~


## Modifying the yaml file.
To actually have the code serve your specific needs, you can modify the yaml file to change which SN are measured and how the fit is performed.

### lightcurves
#### SNID_band_psftype_lc.csv
csv file containing a measured lightcurve for the supernova.

| Parameter            | Type            | Description                                                                                                                                            |
|-----------------------|-----------------|

## 1\. Target / object selection

These tell campari *which* transient (or star) to run on, and where to find it.

| Parameter | CLI flag | Description |
| :---- | :---- | :---- |
| `diaobject_name` | `--diaobject-name` | The catalog ID (a plain integer, e.g. a SNANA/OpenUniverse ID) of the object to process. "human-friendly" identifier. |
| `diaobject_id` | `--diaobject-id` | The database UUID of the object, used instead of/alongside `diaobject_name` when looking objects up via `SNPITDBClient`. |
| `ra`, `dec` | `--ra`, `--dec` | Manually supplied sky coordinates (degrees) for the object. If the object was already found by name/ID and has its own nominal position, campari will warn you if your supplied `ra`/`dec` is far (\>1 arcsec) from that nominal position, then use your supplied value. |
| `diaobject_collection` | `--diaobject-collection` | Which backend to search for the object in. Seen values: `"ou24"` (OpenUniverse 2024 sim catalog), `"manual"` (you're constructing a DiaObject yourself, e.g. for tests or one-off runs), `"snpitdb"` (the SNPIT database).  |
| `radius` | `--radius` | Radius, in degrees, to search around the given `ra`/`dec` for a matching supernova. If not given, campari returns the closest match instead of doing a radius search. |
| `diaobject_subset` | `--diaobject-subset` | Subset of the diaobject collection to restrict lookup to. The code itself has a comment right after this argument reading `# Campari currently does not use this?` — so treat this one with suspicion; it's defined on the CLI but may not actually do anything downstream yet. |
| `diaobject_provenance_tag` | `--diaobject-provenance-tag` | Provenance tag used to select which "version" of the diaobject catalog to query (see the provenance explainer above). |
| `diaobject_process` | `--diaobject-process` | The process name paired with the provenance tag above (e.g. `"load_ou2024_diaobject"`). |
| `diaobject_position_provenance_tag` | `--diaobject-position-provenance-tag` | If set, campari fetches the object's RA/Dec from a *separate* position-measurement provenance rather than using the object's own nominal position. Cannot be combined with manually passing `--ra`/`--dec` (campari will raise an error, since it would be ambiguous which position "wins"). |
| `diaobject_position_process` | `--diaobject-position-process` | Process name paired with the position provenance tag above. |
| `transient_start`, `transient_end` | `--transient_start`, `--transient_end` | MJD (Modified Julian Date) bounds for when the transient is considered "active." Images taken inside this window are treated as detection (transient-bearing) images; images outside it are treated as pre/post-transient reference images. If not given, campari falls back to the object's own `mjd_start`/`mjd_end` from the catalog. |
| `object_type` | `--object_type` | `"SN"` (supernova) or `"star"`. Stars are used as calibrators/tests — when set to `"star"`, campari automatically forces `max_no_transient_images` to 0, since a star has no "before/after" the transient. |

---

## 2\. Image selection

These control *which images* get pulled in to build the lightcurve.

| Parameter | CLI flag | Description |
| :---- | :---- | :---- |
| `filter` (stored as `self.band`) | `-f` / `--filter` | The Roman filter/bandpass to use (e.g. `Y106`, `R062`). Only images in this band are used. |
| `image_collection` | `--image-collection` | Which image backend to pull from. `"snpitdb"` (query the database), `"ou2024"` (OpenUniverse 2024 sim images), `"manual_fits"` (you hand campari a list of file-root paths), `"manual_rdm"` (glob a directory of ASDF files). |
| `image_collection_basepath` | `--image-collection-basepath` | Base directory the image collection should look under (used by collections that read from disk rather than a database). |
| `image_collection_subset` | `--image-collection-subset` | A named subset restriction passed to the collection. |
| `image_provenance_tag`, `image_process` | `--image-provenance-tag`, `--image-process` | Provenance tag/process pair used to pick the right "version" of the image catalog when querying the database, same idea as the diaobject provenance above. |
| `img_list` | `-i` / `--img_list` | Path to a plain-text/CSV file listing exactly which images to use, instead of having campari search for them. Each line can be a file path, or `observation_id,sca`, or `observation_id,sca,band`. Mutually exclusive with `img_glob`. |
| `img_glob` | `--img_glob` | A glob pattern (e.g. `/data/*.asdf`) that expands to a group image files, as an alternative to `img_list`. Cannot be used with `image_collection="manual_fits"` |
| `image_selection_start`, `image_selection_end` | `--image_selection_start`, `--image_selection_end` | Optional MJD bounds narrowing the *search window* for images (separate from `transient_start`/`transient_end`, which define the detection window within whatever images are found). For instance, selecting only the Pilot survey. |
| `max_no_transient_images` | `--max_no_transient_images` | Cap on how many non-detection (reference/background) images to use. Useful for tests and time management. |
| `max_transient_images` | `--max_transient_images` | Cap on how many detection (transient-bearing) images to use. |

---

## 3\. Run mode / debugging shortcuts

| Parameter | CLI flag | Description |
| :---- | :---- | :---- |
| `fast_debug` | `--fast_debug` | A "just make it run fast, don't worry about being right" switch. When true, campari overrides several settings for speed: forces a coarse `regular` grid, `spacing=9`, `cutout_size=11`, disables SED fetching, and disables the initial-guess step. This is meant for quickly checking that a pipeline run completes, not for science-quality results. |
| `config` | `-c` / `--config` | Path to the `.yaml` config file to load. Can also be supplied via the `SNPIT_CONFIG` environment variable instead of the flag. |

---

## 4\. Core photometry / fitting settings

*(config path prefix: `photometry.campari.*`)*

| Parameter | Config key | Description |
| :---- | :---- | :---- |
| Cutout size | `cutout_size` | The width (in pixels) of the square image cutout built around the object for fitting. Default in the base config is `11`; other examples use `19`. Must be odd\! |
| Initial flux guess | `initial_flux_guess` | Starting flux value used to seed the optimizer for each transient-image flux parameter. Default `3000`. |
| Fetch SED | `fetch_SED` | If true, campari fetches a "true" SED (spectral energy distribution — basically the object's spectrum as a function of wavelength) from the OpenUniverse truth tables for the object being fit, rather than assuming a flat spectrum. Cannot be combined with `SED_file` (you have to pick one source of SED information). |
| SED file | `SED_file` (CLI: `--SED_file`) | Path to a user-supplied CSV file specifying a custom SED to fit with, as an alternative to `fetch_SED` or the flat-SED default. |
| Weighting | `weighting` | Whether to apply the inverse-variance \+ optional Gaussian spatial weighting scheme (see `get_weights` in `utils.py`) when fitting, versus weighting every pixel equally. |
| Make initial guess | `make_initial_guess` | Whether to compute a data-driven starting guess for the linear fit (by averaging pixel values at each model grid point across the pre-transient images) rather than starting from all-zeros. |
| Pixel | `pixel` | If true, convolves the model with a small pixel tophat function instead of treating each model point as an infinitesimal point source (delta function). Not hugely impactful on results either way. |
| Subtract background method | `subtract_background_method` | How to remove the sky background from each cutout before fitting. Accepted values: a literal number (subtract that constant), `"calculate"` (estimate it per-image via `photutils`\-based sigma-clipped statistics), `"fit"` (treat the background as a free parameter in the linear fit itself, rather than pre-subtracting it), or any other string, which is interpreted as a FITS header keyword to read the background value from. |
| Print memory usage | `print_memory_usage` | If true uses psutils and logs memory snapshots at several checkpoints through the pipeline. |
| Preplot cutouts | `preplot_cutouts` | If true, saves a diagnostic grid image of all the cutouts (via `plot_cutouts`) before fitting begins, so you can visually sanity-check what images/positions are being used. |

---

## 5\. PSF settings

*(config path prefix: `photometry.campari.psf.*`)*

Campari uses two, potentially different, PSF models: one for the static background (host galaxy / field) and one for the transient itself.

| Parameter | Config key | Description |
| :---- | :---- | :---- |
| Galaxy PSF class | `galaxy_class` | Which snappl PSF model to use for the background/galaxy component. See snappl for details. `"ou24PSF"`, `"ou24PSF_photonshoot"`, `"gaussian"`, `"varying_gaussian"`, `"STPSF"`. |
| Transient PSF class | `transient_class` | Which snappl PSF model to use for the transient point source. See snappl for details. `"ou24PSF_slow"`, `"ou24PSF_slow_photonshoot"`, `"gaussian"`, `"varying_gaussian"`, `"STPSF"`, etc. any snappl PSF type. |

---

## 6\. Grid options — how the background galaxy is modeled

*(config path prefix: `photometry.campari.grid_options.*`)*

Campari models the host galaxy as a collection of point sources arranged on a "grid" rather than fitting a parametric galaxy shape. These parameters control how that grid is built.

| Parameter | Config key | Description |
| :---- | :---- | :---- |
| Type | `type` | Which grid-construction method to use: `"regular"` (evenly spaced), `"adaptive"` (denser where the image is brighter, binned per-pixel), `"contour"` (like adaptive, but using a smooth interpolation so density changes continuously rather than jumping pixel-to-pixel), `"single"` (place exactly one grid point — a sanity-check mode), or `"none"` (skip galaxy modeling entirely, e.g. for star-only fits). |
| Percentiles | `percentiles` | The brightness percentile bin edges used by `"adaptive"`/`"contour"` grids to decide how many points to place in each region (e.g. `[0, 90, 98, 100]` means: dimmest 90% of pixels get 1 point, next 8% get a denser sub-grid, brightest 2% get the densest sub-grid). |
| Spacing | `spacing` | Pixel spacing between grid points for the `"regular"` grid type. |
| Subsize | `subsize` | The width (in pixels) of the region, centered on the cutout, over which grid points are actually placed — can be smaller than the full cutout so that points near the very edge (which can rotate in/out of frame between exposures) are excluded. |
| Gaussian variance | `gaussian_var` | Controls the width (variance, in pixels²) of an optional Gaussian spatial weighting centered on the transient, used when `weighting=true`. Set to a value ≤ 0 to disable this Gaussian weighting entirely (campari treats `gaussian_var <= 0` as "use `None`" internally). |
| Cutoff | `cutoff` | Distance (in pixels) from the transient beyond which pixels are given zero weight, when Gaussian weighting is active — this keeps corner pixels that rotate in/out of frame between exposures from contaminating the fit. |
| Error floor | `error_floor` | A minimum per-pixel error value enforced before computing inverse-variance weights, to avoid a handful of anomalously low-noise pixels dominating the fit with enormous weight. |

---

## 7\. Prebuilt / saved models

These let you skip re-computing expensive parts of the model by reusing results from a previous run (handy for iterating quickly, e.g. re-running just the linear solve while holding the PSF model fixed).

| Parameter | CLI flag | Description |
| :---- | :---- | :---- |
| `save_model` | `--save_model` | If true, saves the constructed PSF ("static scene") and transient model matrices to disk (as `.npy` files in the debug directory) so they can be reloaded later via the two options below. |
| `prebuilt_static_model` | `--prebuilt_static_model` | Path to a previously saved `.npy` file containing the background/galaxy PSF matrix; if given, campari loads this instead of reconstructing the galaxy model from scratch. |
| `prebuilt_transient_model` | `--prebuilt_transient_model` | Same idea, but for the transient PSF matrix. |

---

## 8\. Lightcurve provenance, database saving, and output

| Parameter | CLI flag | Description |
| :---- | :---- | :---- |
| `ltcv_provenance_tag` | `--ltcv-provenance-tag` | Provenance tag to attach to (or look up) the lightcurve output. |
| `ltcv_process` | `--ltcv-process` | Process name paired with the lightcurve provenance tag. |
| `create_ltcv_provenance` | `--create-ltcv-provenance` | If true, campari creates a brand-new provenance record for this lightcurve run rather than reusing an existing one. If false, you must supply both `ltcv_provenance_tag` and `ltcv_process` so campari knows which existing provenance to attach the results to. |
| `save_to_db` | `--save-to-db` / `--no-save-to-db` | Whether to write the resulting lightcurve to the SNPIT database, versus just writing a local `.ecsv` file. |
| `add_truth_to_lc` | `--add-truth-to-lc` | If true, appends the "true" simulated flux/magnitude (from the OpenUniverse truth catalogs) to the output lightcurve, for comparison against the fit. Only meaningful for simulated data where truth is actually known. |
| `nprocs` | `--nprocs` | Number of worker processes to use for parallelizable steps (building cutouts, building the per-image PSF model). Uses Python's `multiprocessing` with the `"fork"` start method. |

---

## 9\. Output / debug I/O

*(config path prefix: `photometry.campari_io.*`)*

| Parameter | Config key | Description |
| :---- | :---- | :---- |
| Output directory | `output_dir` | Where finished lightcurve `.ecsv` files get written (when not saving to the database). |
| Debug directory | `debug_dir` | Where diagnostic files go: saved PSF/SN matrices (`save_model`), saved cutout images/noise maps/grid arrays (`save_debug`), and diagnostic plots. |
| Save debug | `save_debug` | Master switch for writing out the extra debug files described above (images, noise maps, RA/Dec grid, WCS headers) after a run. If false, only the lightcurve itself is written. |
| Test number | `test_num` | A dummy/incrementing value used purely to force the config file's hash to change between test runs.  |
| Test data | `test_data` | Path to a directory of test fixtures used by the test suite (not used by production runs). |

---

## 10\. System-level paths and database connection

*(config path prefix: `system.*` — these describe where things live on disk / how to reach the database, as opposed to `photometry.campari.*`, which describes how the algorithm behaves.)*

| Parameter | Config key | Description |
| :---- | :---- | :---- |
| Lightcurves path | `system.paths.lightcurves` | Base directory for lightcurve output when saving to the database (a separate path from `photometry.campari_io.output_dir`, which is used for local/non-database saves). |
| SED library path | `system.paths.sims_sed_library` | Path to the `rubin_sim` SED template library used for SED lookups. |
| Campari test data | `system.paths.campari_test_data` | Path to test data fixtures (used by `pytest` fixtures across the test suite). |
| Output / SED / debug dirs | `system.paths.output_dir`, `system.paths.sed_path`, `system.paths.debug_dir` | Additional path overrides seen in some example configs; largely mirror the `photometry.campari_io.*` equivalents above. |
| DB URL | `system.db.url` | URL of the SNPIT database server to connect to. |
| DB username | `system.db.username` | Username for the database connection. |
| DB password / passwordfile | `system.db.password`, `system.db.passwordfile` | Either the password directly, or (preferred, for not leaving secrets in a config file) a path to a file containing the password. |
| OU24 sim-index server | `system.ou24.simdex_server` | URL of the "simulation index" server used to look up OpenUniverse 2024 simulation metadata. |
| OU24 config file | `system.ou24.config_file` | Path to the GalSim/roman\_imsim `.yaml` config (like `tds.yaml`/`tds_jupyter.yaml` in `examples/perlmutter/`) describing how the OpenUniverse simulated images themselves were generated — this is a *different* config file from campari's own config, one level further down describing the raw image simulation. |
| OU24 truth directory | `system.ou24.sn_truth_dir` | Directory containing the OpenUniverse "truth" parquet files (`snana_*.parquet`, `pointsource_*.parquet`) used by `access_truth.py` to look up true fluxes/magnitudes and host-galaxy properties. |
| OU24 SED library | `system.ou24.sims_sed_library` | Same idea as `system.paths.sims_sed_library`, scoped under the `ou24` section in some configs. |
| OU24 images path | `system.ou24.images` | Base path to the OpenUniverse simulated image files themselves. |
| OU24 TDS base | `system.ou24.tds_base` | Base path used for locating "Time Domain Survey" (TDS) related test data. |
| `ou24.config_file`, `ou24.sn_truth_dir`, `ou24.sims_sed_library` | (top-level, not under `system`) | In `examples/perlmutter/campari_config.yaml` and `campari_config_outside_podman.yaml`, these same three settings also appear as a **top-level** `ou24:` section rather than nested under `system.ou24`. Both forms show up across the example configs — worth double-checking which one your particular branch/version of the code actually reads, since having both a `system.ou24.*` and a top-level `ou24.*` version is the kind of thing that quietly diverges over time. |
| `ou24psf.config_file` | top-level `ou24psf:` section | Path to a config file specifically for PSF generation, seen in `campari_config.yaml`. |
| `preloads` | top-level `preloads:` list | A list of *other* config file paths to load first, before applying the rest of this config file's settings on top. This is how `campari_config_test.yaml` pulls in a shared base config (e.g. `/snpit_env/configs/rknop_dev_container_config.yaml`) and then layers its own overrides on top — same "form \+ sticky notes" idea as the CLI-flag override system, just one config file overriding another instead of a CLI flag overriding a file. |
| `photometry.snappl.simdex_server` | — | A `snappl`\-scoped copy of the sim-index server URL, seen in some configs alongside the `system.ou24.simdex_server` version. |
| `galsim.tds_file` | — | Points to the GalSim TDS config file; alternate/older naming for what's captured by `system.ou24.config_file` above. |

---

Information on the output can be found here:

[Rich Lightcurve Link](https://github.com/Roman-Supernova-PIT/Roman-Supernova-PIT/wiki/rich_lightcurve)