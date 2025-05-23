#!/bin/bash
#SBATCH -J multismp_test
#SBATCH -p cosmology
#SBATCH -o multismp_test.out
#SBATCH -A cosmology
#SBATCH --mail-user=cmeldorf@sas.upenn.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=20G

## Why doesn't this switch over environments?
## Why do we need 20 gigs of memory?

source ~/.bashrc
conda activate ColeRoman


python RomanASP.py -s 40120913 -f Y106 -t 2 -d 1
