from types import SimpleNamespace

def get_ar_semla_model_args():
    dataset = "plinder"
    args_dict = {
        # General args
        "data_path":  f'semlaflow/saved/data/{dataset}/smol',
        "dataset": dataset,
        "is_pseudo_complex": False,
        "trial_run": False,
        "val_check_epochs": 1,
        "monitor": "val-validity",
        "monitor_mode": "max",

        # Model args
        "d_model": 384,
        "n_layers": 12,
        "n_pro_layers": 12,
        "d_message": 128,
        "d_edge": 128,
        "n_coord_sets": 64,
        "n_attn_heads": 32,
        "d_message_hidden": 128,
        "coord_norm": "length",
        "size_emb": 64,
        "max_atoms": 256,
        "arch": "semla",
        "integration_steps": 100,

        # Training args
        "epochs": 200,
        "lr": 0.0003,
        "batch_cost": 2048, # 4096,
        "acc_batches": 1,
        "gradient_clip_val": 1.0,
        "type_loss_weight": 0.,
        "bond_loss_weight": 0.,
        "charge_loss_weight": 0.,
        "coord_align": False,
        "t_per_ar_action": 0.25,
        "max_interp_time": 0.5,
        "ordering_strategy": "connected",
        "decomposition_strategy": "reaction",
        "max_action_t": 0.75,
        "max_num_cuts": 3,
        "categorical_strategy": "auto-regressive",
        "lr_schedule": "constant",
        "warm_up_steps": 10000,
        "bucket_cost_scale": "linear",
        "use_ema": True,  # Defaults to True since --no_ema negates it
        "self_condition": True,
        "distill": False,

        # Flow matching and sampling args
        "conf_coord_strategy": "gaussian",
        "complex_debug": False,
        "dist_loss_weight": 0.,
        "ode_sampling_strategy": "linear",
        "n_validation_mols": 50,
        "n_training_mols": 1000,
        "num_inference_steps": 100,
        "cat_sampling_noise_level": 1,
        "coord_noise_std_dev": 0.2,
        "type_dist_temp": 1.0,
        "time_alpha": 1.0,
        "time_beta": 1.0,
        "time_discretization": None,
        "optimal_transport": "None",
        "pocket_encoding": "gvp",
        "num_workers": None,
        "min_group_size": 5
    }

    args = SimpleNamespace(**args_dict)
    return args
