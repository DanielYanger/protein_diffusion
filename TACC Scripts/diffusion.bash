#!/bin/bash

#SBATCH -J lr-prob-diffusion-%j                # job name
#SBATCH -o lr-prob-diffusion-%j-output.o            # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 1                        # total number of tasks to run in parallel
#SBATCH -p gpu-a100-small              # queue (partition) 
#SBATCH -t 32:00:00                 # run time (hh:mm:ss) 
#SBATCH --mail-user=daniel_yanger@utexas.edu
#SBATCH --mail-type=all

# cd $HOME

export PATH="$HOME/.local/bin:$PATH"
source activate base 

cd $WORK/protein_diffusion
conda activate ./envs
# echo $CONDA_DEFAULT_ENV
# pwd
accelerate launch train_diffusion.py
# exit