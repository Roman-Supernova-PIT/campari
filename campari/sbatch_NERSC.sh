#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cmeldorf@sas.upenn.edu
#SBATCH --mem=20GB
source ~/.bashrc
conda activate /global/u1/c/cmeldorf/environment/envs/sn-pit-dev

#python RomanASP.py --SNID_file '/global/homes/c/cmeldorf/MultiSMP/SNe.csv' -f Y106 -t 20 -d 10
python RomanASP.py -s 20172782 -f Y106 --photometry-campari-use_roman --photometry-campari-use_real_images \
--no-photometry-campari-fetch_SED --photometry-campari-grid_options-type contour --photometry-campari-cutout_size 19 \
--photometry-campari-weighting --photometry-campari-subtract_background --photometry-campari-source_phot_ops \
-c /pscratch/sd/c/cmeldorf/campari/examples/perlmutter/campari_config_jupyter.yaml
