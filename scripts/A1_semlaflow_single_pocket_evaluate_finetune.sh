DATA_DIR='cgflow/saved/data/ALDH1_debug'
MODEL_CHECKPOINT='result/3DSynthflow_result/ALDH1_finetune_epoch39.ckpt'

python scripts/_a1_cgflow_train.py \
  --data_path ${DATA_DIR} \
  --model_checkpoint ${MODEL_CHECKPOINT} \
  --dataset crossdock \
  --categorical_strategy auto-regressive \
  --pocket_encoding gvp \
  --ordering_strategy connected \
  --decomposition_strategy reaction \
  --n_pro_layers 6 \
  --time_alpha 1.0 \
  --t_per_ar_action 0.3 \
  --max_interp_time 0.4 \
  --max_action_t 0.6 \
  --max_num_cuts 2 \
  --dist_loss_weight 0. \
  --type_loss_weight 0. \
  --bond_loss_weight 0. \
  --charge_loss_weight 0. \
  --optimal_transport None \
  --monitor val-strain \
  --monitor_mode min \
  --val_check_epochs 1 \
  --batch_cost 3000 \
  --use_complex_metrics \
  --num_workers 0 \
  --num_gpus 1

