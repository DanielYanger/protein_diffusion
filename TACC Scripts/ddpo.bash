#!/bin/bash

#SBATCH -J ddpo-%j                # job name
#SBATCH -o ddpo-%j-output.o            # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 4                        # total number of tasks to run in parallel
#SBATCH -p gpu-a100              # queue (partition) 
#SBATCH -t 20:00:00                 # run time (hh:mm:ss) 
#SBATCH --mail-user=daniel_yanger@utexas.edu
#SBATCH --mail-type=all

# cd $HOME

export PATH="$HOME/.local/bin:$PATH"
source activate base 

cd $WORK/protein_diffusion
conda activate ./envs
# echo $CONDA_DEFAULT_ENV
# pwd
accelerate launch ddpo_trainer.py
# exit