#!/bin/bash

#SBATCH -J working-diffusion-%j                # job name
#SBATCH -o working-diffusion-%j-output.o            # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 4                        # total number of tasks to run in parallel
#SBATCH -p gpu-a100-small              # queue (partition) 
#SBATCH -t 20:00:00                 # run time (hh:mm:ss) 
#SBATCH --mail-user=daniel_yanger@utexas.edu
#SBATCH --mail-type=all

# cd $HOME

export PATH="$HOME/.local/bin:$PATH"
source activate base 

cd $WORK/protein-generation/protein_diffusion
conda activate ./envs
# echo $CONDA_DEFAULT_ENV
# pwd
accelerate launch train_diffusion.py
# exit