#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4 # This needs to be equal to the number of GPUs
#SBATCH --time=3-00:00:00
#SBATCH --job-name=cgflow-plinder
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=myjob.%j.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_P2P_DISABLE=1

srun --cpu-bind=none python scripts/pretrain/train.py --config ./configs/cgflow/train.yaml --name plinder --num_gpus 4
