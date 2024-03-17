#!/bin/bash

#SBATCH -J utr-diffusion-%j                # job name
#SBATCH -o utr-diffusion-%j-output.o            # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 1                        # total number of tasks to run in parallel
#SBATCH -p gpu-a100-small              # queue (partition) 
#SBATCH -t 40:00:00                 # run time (hh:mm:ss) 
#SBATCH --mail-user=daniel_yanger@utexas.edu
#SBATCH --mail-type=all

# cd $HOME

export PATH="$HOME/.local/bin:$PATH"
source activate base 

cd $WORK/protein_diffusion
conda activate ./envs
echo "Saving into UTR"
# pwd
accelerate launch train_diffusion_utrs.py
# exit