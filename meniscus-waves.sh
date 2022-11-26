#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=5g
#SBATCH -t 15:00:00 --mail-type=end --mail-user=daftari@email.unc.edu --wrap="python3 run_paths_pool.py”

module add python/3.7.9

mkdir -p unpinned-45deg-8Hz-0_2g-LVL4_9uf-t4
mkdir -p "unpinned-45deg-8Hz-0_2g-LVL4_9uf-t4/vtkoutput"
cd unpinned-45deg-8Hz-0_2g-LVL4_9uf-t4
cp ../meniscus-waves.c meniscus-waves.c

qcc -O2 -Wall meniscus-waves.c -o meniscus-waves -lm
./meniscus-waves