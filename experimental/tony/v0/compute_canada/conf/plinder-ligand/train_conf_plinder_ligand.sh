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
python semlaflow/train.py \
    --data_path semlaflow/saved/data/plinder-ligand/smol \
    --dataset plinder-ligand \
    --type_loss_weight 0. \
    --bond_loss_weight 0. \
    --charge_loss_weight 0. \
    --optimal_transport None \
    --categorical_strategy no-change \
    --val_check_epochs 5 \
    --epochs 1000
