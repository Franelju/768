#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=5g
#SBATCH -t 15:00:00 --mail-type=end --mail-user=daftari@email.unc.edu --wrap="python3 run_paths_pool.py”

module add python/3.7.9

