#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4 # This needs to be equal to the number of GPUs
#SBATCH --time=3-00:00:00
#SBATCH --job-name=plinder
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=myjob.%j.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_P2P_DISABLE=1

DATA_DIR="/projects/jlab/to.shen/cgflow-dev/data/complex/crossdock-no-litpcba/smol"

srun --cpu-bind=none python scripts/_a1_semlaflow_train.py \
  --data_path ${DATA_DIR} \
  --dataset crossdock \
  --categorical_strategy auto-regressive \
  --ordering_strategy connected \
  --decomposition_strategy reaction \
  --pocket_n_layers 4 \
  --d_message 64 \
  --d_message_hidden 96 \
  --time_alpha 1.0 \
  --t_per_ar_action 0.33 \
  --max_interp_time 1.0 \
  --max_action_t 0.66 \
  --max_num_cuts 2 \
  --dist_loss_weight 0. \
  --type_loss_weight 0. \
  --bond_loss_weight 0. \
  --charge_loss_weight 0. \
  --optimal_transport None \
  --monitor val-conformer-no-align-rmsd \
  --monitor_mode min \
  --val_check_epochs 1 \
  --acc_batches 2 \
  --batch_cost 1800 \
  --num_gpus 4
