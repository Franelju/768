#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=5g
#SBATCH -t 15:00:00 
#SBATCH --mail-type=end 
#SBATCH --mail-user=frane@email.unc.edu 
#SBATCH--wrap="python3 run_paths_pool.py”
#SBATCH -o ./report/output.%a.out # STDOUT

module add python/3.7.9
pip install timebudget

python3 run_paths_pool.py

