#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -p scc-gpu
#SBATCH -G 1
#SBATCH --mail-type=all
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

module load miniconda3
source $HOME/.bashrc
conda activate tflat

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python3 --version

# Run the script:
python3 optimize.py
