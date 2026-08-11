#!/bin/bash

n_jobs=5
echo $n_jobs
#file_len=$(wc -l < /rick/romanisim/STARS_GAIA.csv)
file_len=5
echo $file_len
stars_per_job=$((file_len / n_jobs ))
echo $stars_per_job



singularity run --bind /home/rkessler/:/rick:ro \
  --bind /mnt/roman-science-east-2/snpit/snana+romanisim+romancal/:/ricksims:ro \
  --overlay pit_overlay \
  /data/snpit/roman-snpit-env-cpu-0.1.36.sif \
    bash -c '
        echo "Starting job for stars" $i
    '



# for ((i=0; i<file_len+1; i+=1))
#         do
#             echo "Starting job for stars" $i
#             echo here
#         done

# pip install -e /home/cfmeldorf/campari/ -e /home/cfmeldorf/snappl/ &&
#     pip install healpy &&
#     pip install pytest &&
#     mkdir -p /home/cfmeldorf/campari/logs &&
#     export SNPIT_CONFIG=/home/cfmeldorf/campari/examples/SMDC/campari_config_test.yaml &&


# #upper=$((i + stars_per_job - 1))
#         upper=$((i + 1))
#         echo $upper
#         echo -----

#         nohup python -u /home/cfmeldorf/asdf_new_rick_ims.py --low_index $i --high_index $upper --star_mode True\
#         >& /home/cfmeldorf/campari/logs/imsim_log_${i}_${upper}.txt &














#nohup singularity run --overlay pit_overlay /data/snpit/roman-snpit-env-cpu-0.1.36.sif \
    #   bash -c "
    # nohup python -m virtualenv new_venv &&
    #         source new_venv/bin/activate &&
    #         pip install -e campari/ -e snappl/ &&
    #         pip install healpy &&
    #         pip install pytest &&
            # export SNPIT_CONFIG=campari/examples/SMDC/campari_config_test.yaml &&