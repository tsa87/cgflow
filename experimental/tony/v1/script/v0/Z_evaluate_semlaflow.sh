#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4 # This needs to be equal to the number of GPUs
#SBATCH --time=01:00:00
#SBATCH --job-name=plinder
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --output=myjob.%j.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_P2P_DISABLE=1

DATA_DIR="/projects/jlab/to.shen/cgflow-dev/experiments/data/complex/plinder_15A"
MODEL_CHECKPOINT="/projects/jlab/to.shen/cgflow-dev/weights/plinder_till_end.ckpt"
OUTPUT_DIR="evaluation_results"

python scripts/_a2_semlaflow_eval.py  \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --data_path ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --n_validation_mols 100 \
    --num_gpus 4 \
    --batch_cost 13000
