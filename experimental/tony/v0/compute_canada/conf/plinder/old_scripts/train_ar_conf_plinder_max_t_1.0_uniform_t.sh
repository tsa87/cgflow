#!/bin/bash
#SBATCH --time=3-00:00:00  #(days-hours:minutes:seconds)
#SBATCH --mem=32G # total CPU memory
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL

cd $project/trans-semla
module purge
module load python/3.11 rdkit/2024.03.4 scipy-stack/2024a openbabel/3.1.1
source ~/equinv/bin/activate

nvidia-smi

export PYTHONPATH=$PYTHONPATH:$project/trans-semla
export WANDB_API_KEY=fe74f9b5ba3b6f8a1a5fa3be198bc1cf09cf14e6

echo "train_ar_conf_plinder_all_pro_layers_max_t_1.0.sh"

python semlaflow/train.py \
    --data_path semlaflow/saved/data/plinder/smol \
    --dataset plinder \
    --t_per_ar_action 0.25 \
    --max_interp_time 1.0 \
    --ordering_strategy connected \
    --decomposition_strategy reaction \
    --max_action_t 0.75 \
    --max_num_cuts 3 \
    --dist_loss_weight 0. \
    --type_loss_weight 0. \
    --bond_loss_weight 0. \
    --charge_loss_weight 0. \
    --optimal_transport None  \
    --categorical_strategy auto-regressive \
    --monitor val-strain \
    --monitor_mode min \
    --val_check_epochs 20 \
    --batch_cost 2500 \
    --time_alpha 1.0 \
    --c_alpha_only
