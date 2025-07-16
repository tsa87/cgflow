#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1 # This needs to be equal to the number of GPUs
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cgflow-topo2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=myjob.%j.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL

# Run the wandb agent using the sweep ID provided from wandb sweep
python experiments/scripts/exp3_topo2_debug.py
