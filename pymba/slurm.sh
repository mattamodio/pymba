#!/bin/bash
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=mba_farnam
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<matthew.amodio@yale.edu>

module restore cuda
source activate dlnn


python -u $1 > log.out


# to use: sbatch slurm.sh myPythonProgram.py
# to monitor: squeue -u mba27
