from collections import namedtuple

import cgflow.scriptutil as util
import cgflow.util.rdkit as smolRD
from cgflow.models.complex_fm import ARMolecularCFM
from cgflow.models.fm import Integrator, MolecularCFM
from cgflow.models.pocket_v2 import LigandGenerator as LigandGeneratorV2
from cgflow.models.pocket_v2 import PocketEncoder as PocketEncoderV2
from cgflow.models.pocket import LigandGenerator as LigandGeneratorV1
from cgflow.models.pocket import PocketEncoder as PocketEncoderV1

CategoricalStrategyConfig = namedtuple(
    "CategoricalStrategyConfig",
    [
        "train_strategy",
        "sampling_strategy",
        "type_mask_index",
        "bond_mask_index",
        "n_bond_types",
        "n_extra_atom_feats",
    ],
)

DataConfig = namedtuple("DataConfig", ["coord_scale", "is_complex"])

TrainConfig = namedtuple("TrainConfig", ["train_steps", "train_smiles"])


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "16-mixed"
    # return "16-mixed" if args.mixed_precision else "32"


def get_hparams(args, dm):
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams,
    }
    return hparams


def get_categorical_config(args, vocab):
    type_mask_index = None
    bond_mask_index = None
    n_extra_atom_feats = 1  # time

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "no-change":
        train_strategy = "no-change"
        sampling_strategy = "no-change"

    elif args.categorical_strategy == "auto-regressive":
        train_strategy = "no-change"  # Not used
        sampling_strategy = "no-change"
        n_extra_atom_feats += 1  # for the relative time for each token

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported.")

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    return CategoricalStrategyConfig(
        train_strategy,
        sampling_strategy,
        type_mask_index,
        bond_mask_index,
        n_bond_types,
        n_extra_atom_feats,
    )


def get_dataset_config(dataset, is_pseudo_complex):
    if dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
        is_complex = False
    elif dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
        is_complex = False
    elif dataset == "plinder-ligand":
        coord_scale = util.PLINDER_COORDS_STD_DEV
        is_complex = False
    elif dataset == "plinder":
        coord_scale = util.PLINDER_COORDS_STD_DEV
        is_complex = True
    elif dataset == "zinc15m":
        coord_scale = util.PLINDER_COORDS_STD_DEV
        is_complex = True
    elif dataset == "crossdock":
        coord_scale = util.PLINDER_COORDS_STD_DEV
        is_complex = True
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    is_complex = any((is_complex, is_pseudo_complex))
    return DataConfig(coord_scale, is_complex)


def get_train_config(args, dm):
    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    train_smiles = None if args.trial_run else dm.train_dataset.smiles  # NOTE: much faster
    return TrainConfig(train_steps, train_smiles)


def get_semla_model(args, vocab, cat_config, pocket_enc=None):
    if args.semla_version == "v1":
        assert args.ligand_local_connections is None, "v1 does not support ligand local connections"
        assert args.pocket_local_connections is None, "v1 does not support pocket local connections"
        egnn_gen = LigandGeneratorV1(
            d_equi=args.n_coord_sets,
            d_inv=args.d_model,
            d_message=args.d_message,
            n_layers=args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_ff=args.d_message_hidden,
            d_edge=args.d_edge,
            n_atom_types=vocab.size,
            n_bond_types=cat_config.n_bond_types,
            n_extra_atom_feats=cat_config.n_extra_atom_feats,
            self_cond=args.self_condition,
            pocket_enc=pocket_enc,
        )
    elif args.semla_version == "v2":
        egnn_gen = LigandGeneratorV2(
            d_equi=args.n_coord_sets,
            d_inv=args.d_model,
            d_message=args.d_message,
            n_layers=args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_ff=args.d_message_hidden,
            d_edge=args.d_edge,
            n_atom_types=vocab.size,
            n_bond_types=cat_config.n_bond_types,
            n_extra_atom_feats=cat_config.n_extra_atom_feats,
            self_cond=args.self_condition,
            ligand_local_connections=args.ligand_local_connections,
            cond_local_connections=args.pocket_ligand_local_connections,
            pocket_enc=pocket_enc,
        )
    else:
        raise ValueError(f"Unknown Semla version {args.semla_version}")
    return egnn_gen


def get_pocket_encoder(args, vocab, cat_config):
    pocket_d_equi = 1 if args.fixed_equi else args.n_coord_sets
    if args.semla_version == "v1":
        pocket_enc = PocketEncoderV1(d_equi=pocket_d_equi,
                                     d_inv=args.pocket_d_inv,
                                     d_message=args.d_message,
                                     n_layers=args.pocket_n_layers,
                                     n_attn_heads=args.n_attn_heads,
                                     d_message_ff=args.d_message_hidden,
                                     d_edge=args.d_edge,
                                     n_atom_names=vocab.size,
                                     n_bond_types=cat_config.n_bond_types,
                                     n_res_types=len(smolRD.IDX_RESIDUE_MAP),
                                     fixed_equi=args.fixed_equi)
    elif args.semla_version == "v2":
        pocket_enc = PocketEncoderV2(
            d_equi=pocket_d_equi,
            d_inv=args.pocket_d_inv,
            d_message=args.d_message,
            n_layers=args.pocket_n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_ff=args.d_message_hidden,
            d_edge=args.d_edge,
            n_atom_names=vocab.size,
            n_bond_types=cat_config.n_bond_types,
            n_res_types=len(smolRD.IDX_RESIDUE_MAP),
            fixed_equi=args.fixed_equi,
            local_connections=args.pocket_local_connections,
            virtual_nodes=args.pocket_virtual_nodes,
        )
    else:
        raise ValueError(f"Unknown Semla version {args.semla_version}")
    return pocket_enc


def get_cfm_model(
    args,
    dm,
    egnn_gen,
    vocab,
    data_config,
    cat_config,
    train_config,
    hparams,
):
    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=cat_config.sampling_strategy,
        bond_strategy=cat_config.sampling_strategy,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=cat_config.type_mask_index,
        bond_mask_index=cat_config.bond_mask_index,
    )

    common_params = dict(
        gen=egnn_gen,
        vocab=vocab,
        lr=args.lr,
        integrator=integrator,
        coord_scale=data_config.coord_scale,
        type_strategy=cat_config.train_strategy,
        bond_strategy=cat_config.train_strategy,
        dist_loss_weight=args.dist_loss_weight,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        pairwise_metrics=False,
        complex_metrics=args.use_complex_metrics,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_config.train_steps,
        train_smiles=train_config.train_smiles,
        type_mask_index=cat_config.type_mask_index,
        bond_mask_index=cat_config.bond_mask_index,
        args=args,
        **hparams,
    )

    if args.categorical_strategy != "auto-regressive":
        fm_model = MolecularCFM(**common_params)
    else:
        fm_model = ARMolecularCFM(ar_interpolant=dm.train_interpolant,
                                  **common_params)

    return fm_model


def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = get_hparams(args, dm)
    cat_config = get_categorical_config(args, vocab)
    data_config = get_dataset_config(args.dataset, args.is_pseudo_complex)
    train_config = get_train_config(args, dm)
    print(f"Total training steps {train_config.train_steps}")

    pocket_enc = get_pocket_encoder(
        args, vocab, cat_config) if data_config.is_complex else None
    egnn_gen = get_semla_model(args, vocab, cat_config, pocket_enc)

    print(f"Using model class {egnn_gen.__class__.__name__}")
    fm_model = get_cfm_model(
        args,
        dm,
        egnn_gen,
        vocab,
        data_config,
        cat_config,
        train_config,
        hparams,
    )

    print(f"Using CFM class {fm_model.__class__.__name__}")
    return fm_model
