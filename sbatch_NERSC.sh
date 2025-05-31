#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cmeldorf@sas.upenn.edu
#SBATCH --mem=20GB
source ~/.bashrc
conda activate sn_pit_dev

python RomanASP.py --SNID_file '/global/homes/c/cmeldorf/MultiSMP/SNe.csv' -f Y106 -t 20 -d 10
