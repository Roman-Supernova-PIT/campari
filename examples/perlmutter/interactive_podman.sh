#!/bin/sh

podman-hpc run --gpu \
    --mount type=bind,source=$PWD,target=/home \
    --mount type=bind,source=$PWD/campari,target=/campari \
    --mount type=bind,source=$PWD/campari_out_dir,target=/campari_out_dir \
    --mount type=bind,source=$SCRATCH/campari_debug_dir,target=/campari_debug_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data,target=/sims_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE,target=/snana_pq_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/sims_sed_library,target=/sims_sed_library \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/ROMAN+LSST_LARGE_SNIa-normal,target=/snid_lc_dir \
    --env LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
    --env PYTHONPATH=/roman_imsim:/phrosty \
    --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --env OMP_NUM_THREADS=1 \
    --env VECLIB_MAXIMUM_THREADS=1 \
    --env SNPIT_CONFIG=/campari/base_campari_config.yaml \
    --env TERM=xterm \
    --annotation run.oci.keep_original_groups=1 \
    -it \
    registry.nersc.gov/m4385/rknop/roman-snpit-env:cpu \
    /bin/bash
