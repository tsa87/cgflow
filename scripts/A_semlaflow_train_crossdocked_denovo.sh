DATA_DIR=data/complex/crossdock-no-litpcba/smol

python scripts/_a1_cgflow_train.py \
  --data_path ${DATA_DIR} \
  --dataset crossdock \
  --categorical_strategy uniform-sample \
  --val_check_epochs 1 \
  --batch_cost 6000 \
  --num_gpus 1

