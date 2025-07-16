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

DATA_DIR="/projects/jlab/to.shen/cgflow-dev/data/complex/plinder/smol"

srun --cpu-bind=none python scripts/_a1_semlaflow_train.py \
  --data_path ${DATA_DIR} \
  --dataset plinder \
  --categorical_strategy no-change \
  --pocket_n_layers 4 \
  --d_message 64 \
  --d_message_hidden 96 \
  --time_alpha 1.0 \
  --dist_loss_weight 10. \
  --type_loss_weight 0. \
  --bond_loss_weight 0. \
  --charge_loss_weight 0. \
  --optimal_transport None \
  --monitor val-strain \
  --monitor_mode min \
  --val_check_epochs 1 \
  --batch_cost 1800 \
  --num_gpus 4
