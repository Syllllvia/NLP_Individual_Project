#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal       # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024     # only require for multiple projects

module purge
module load Anaconda3 cuda12.2/fft cuda12.2/blas cuda12.2/toolkit
module list
conda activate pytorch

python test.py
