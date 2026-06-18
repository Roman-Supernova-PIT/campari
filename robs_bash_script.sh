#!/bin/bash

n_jobs=2
echo $n_jobs
#file_len=$(wc -l < /rick/romanisim/STARS_GAIA.csv)
file_len=644
echo $file_len
stars_per_job=$((file_len / n_jobs ))
echo $stars_per_job


nohup python -m virtualenv new_venv
source new_venv/bin/activate 


for ((i=0; i<=file_len - stars_per_job; i+=stars_per_job))
do
    echo $i
    #upper=$((i + stars_per_job - 1))
    upper=$((i + 1))
    echo $upper
    echo -----
    #nohup singularity run --overlay pit_overlay /data/snpit/roman-snpit-env-cpu-0.1.36.sif \
    #   bash -c "
    nohup python -m virtualenv new_venv &&
            source new_venv/bin/activate &&
            pip install -e campari/ -e snappl/ &&
            pip install healpy &&
            pip install pytest &&
            export SNPIT_CONFIG=campari/examples/SMDC/campari_config_test.yaml &&
            python -u asdf_phot_run.py --low_index $i --high_index $upper \
    #    "
done
