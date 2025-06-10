(**Warning** : some of this readme is currently out of date, especially with regards to paths.  We need to update it.)

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

The RomanASP code can be run from the command line. Basic arguments are given in the command line and algorithm settings are given via the input file config.yaml. Because of different file paths on different
systems, the steps are slightly different for each machine. Here's how to get a basic run going dpeending on which computer you find yourself using:

### DCC:

To do a simple test run to ensure everything is installed correctly, you can request a node:

```
srun -n 1 -N 1 -t 4:00:00 --mem 20000 -p cosmology --account=cosmology --pty bash
conda activate sn_pit_dev
```
cd into your directory where the code is stored.
Then, in the `config.yaml` file, ensure that `roman_path` and `sn_path` read as follows:

```
roman_path: /hpc/group/cosmology/OpenUniverse2024
sn_path: /hpc/group/cosmology/OpenUniverse2024/roman_rubin_cats_v1.1.2_faint/
```

Next, in the `temp_tds.yaml` file, make sure `file_name` is:
```
file_name: /hpc/group/cosmology/OpenUniverse2024/RomanTDS/Roman_TDS_obseq_11_6_23.fits
```

and then run:

```
python RomanASP.py -s 40120913 -f Y106 -t 10 -d 5
```
This will run the algorithm on supernova with SNID 40120913, in band Y106, using 10 images 5 of which contain SN detections.

### NERSC

(Aside: see examples/perlmutter for an example of running campari in a podman container on Perlmutter.)

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
This will run the algorithm on supernova with SNID 40120913, in band Y106, using 10 images 5 of which contain SN detections.


## Arguments and configs

There are a number of command-line arguments.  Run `campari/RomanASP.py` with the `--help` option to get documentation on them.

Campari depends on a config file.  You can find a default config file in `base_campari_config.yaml` (which is also the config file we use in the tests, and in the dockerized perlmutter example in `examples/perlmutter`).  That file documents what the various config options do.  You can make a customized config file by creating a new `.yaml` file that includes this file and then only overrides what you want to change; for an example of this in action, see `examples/campari_cole_dcc_config.yaml`.  Campari will read the config file from the file pointed to by the environment varaible `SNPIT_CONFIG`.

If you have specified a config file (either using `SNPIT_CONFIG`, or passing something to `-c` when running campari), then it's possible to override most (all?) config values from the command line.  When you add `--help`, it will, in addition to the regular command line options, list all of the options necessary to override things from the config files.
