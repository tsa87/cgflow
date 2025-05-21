DATA_DIR="/projects/jlab/to.shen/CGFlow/data/complex/plinder/smol"

python scripts/_a1_cgflow_train.py \
  --data_path ${DATA_DIR} \
  --dataset plinder \
  --categorical_strategy auto-regressive \
  --ordering_strategy connected \
  --decomposition_strategy reaction \
  --pocket_n_layers 4 \
  --d_message 64 \
  --d_message_hidden 96 \
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
  --batch_cost 600 \
  --num_gpus 8
