from collections import namedtuple
from functools import partial

import semlaflow.scriptutil as util
import semlaflow.util.rdkit as smolRD
from semlaflow.models.ar_fm import ARComplexMolecularCFM, ARMolecularCFM
from semlaflow.models.complex import ComplexEquiInvDynamics, ComplexSemlaGenerator
from semlaflow.models.complex_fm import ComplexMolecularCFM
from semlaflow.models.fm import BaseMolecularCFM, Integrator
from semlaflow.models.pocket import PocketEncoder
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

CategoricalStrategyConfig = namedtuple(
    "CategoricalStrategyConfig",
    [
        "train_strategy",
        "sampling_strategy",
        "type_mask_index",
        "bond_mask_index",
        "n_lig_atom_feats",
        "n_bond_types",
        "n_pro_atom_feats",
        "use_gvp",
    ],
)

DataConfig = namedtuple("DataConfig", ["coord_scale", "is_complex"])

TrainConfig = namedtuple("TrainConfig", ["train_steps", "train_smiles"])


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "32"
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

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        train_strategy = "ce"
        sampling_strategy = "dirichlet"

    elif args.categorical_strategy == "no-change":
        # train_strategy doesn't matter since loss weight is 0 for no-change
        assert args.type_loss_weight == 0.0
        assert args.bond_loss_weight == 0.0
        assert args.charge_loss_weight == 0.0

        train_strategy = "no-change"
        sampling_strategy = "no-change"

    elif args.categorical_strategy == "auto-regressive":
        assert args.type_loss_weight == 0.0
        assert args.bond_loss_weight == 0.0
        assert args.charge_loss_weight == 0.0

        train_strategy = "no-change"  # Not used
        sampling_strategy = "no-change"

    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported.")

    # Add 1 for the time (0 <= t <= 1 for flow matching)
    n_lig_atom_feats = vocab.size + 1
    # Add 1 for relative time for each atom if we are doing auto-regressive sampling
    if args.categorical_strategy == "auto-regressive":
        n_lig_atom_feats += 1

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    use_gvp = args.pocket_encoding == "gvp"

    if args.pocket_encoding == "atom":
        n_pro_atom_feats = vocab.size + 1
    elif args.pocket_encoding == "gvp":
        n_pro_atom_feats = 128 + 1
    elif args.pocket_encoding == "c-alpha":
        n_pro_atom_feats = len(smolRD.IDX_RESIDUE_MAP) + 1
    else:
        raise ValueError(f"Unknown pocket encoding {args.pocket_encoding}")

    if args.categorical_strategy == "auto-regressive":
        n_pro_atom_feats += 1

    return CategoricalStrategyConfig(
        train_strategy,
        sampling_strategy,
        type_mask_index,
        bond_mask_index,
        n_lig_atom_feats,
        n_bond_types,
        n_pro_atom_feats,
        use_gvp,
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
    train_smiles = None if args.trial_run else [mols.str_id for mols in dm.train_dataset._data]  # NOTE: much faster
    return TrainConfig(train_steps, train_smiles)


def get_semla_model(args, vocab, cat_config):
    dynamics = EquiInvDynamics(
        args.d_model,
        args.d_message,
        args.n_coord_sets,
        args.n_layers,
        n_attn_heads=args.n_attn_heads,
        d_message_hidden=args.d_message_hidden,
        d_edge=args.d_edge,
        bond_refine=True,
        self_cond=args.self_condition,
        coord_norm=args.coord_norm,
    )
    egnn_gen = SemlaGenerator(
        args.d_model,
        dynamics,
        vocab.size,
        cat_config.n_lig_atom_feats,
        d_edge=args.d_edge,
        n_edge_types=cat_config.n_bond_types,
        self_cond=args.self_condition,
        size_emb=args.size_emb,
        max_atoms=args.max_atoms,
    )
    return egnn_gen


def get_complex_semla_model(args, vocab, cat_config):
    dynamics = ComplexEquiInvDynamics(
        args.d_model,
        args.d_message,
        args.n_coord_sets,
        args.n_layers,
        n_pro_layers=args.n_pro_layers,
        n_attn_heads=args.n_attn_heads,
        d_message_hidden=args.d_message_hidden,
        d_edge=args.d_edge,
        bond_refine=True,
        self_cond=args.self_condition,
        coord_norm=args.coord_norm,
        debug=args.complex_debug,
    )

    egnn_gen = ComplexSemlaGenerator(
        args.d_model,
        dynamics,
        vocab.size,
        n_lig_atom_feats=cat_config.n_lig_atom_feats,
        n_pro_atom_feats=cat_config.n_pro_atom_feats,
        d_edge=args.d_edge,
        n_edge_types=cat_config.n_bond_types,
        self_cond=args.self_condition,
        size_emb=args.size_emb,
        max_atoms=args.max_atoms,
        debug=args.complex_debug,
    )

    return egnn_gen


def get_eqgat_model(args, vocab, cat_config):
    from semlaflow.models.eqgat import EqgatGenerator

    # Hardcode for now since we only need one model size
    d_model_eqgat = 256
    n_equi_feats_eqgat = 256
    n_layers_eqgat = 12
    d_edge_eqgat = 128

    egnn_gen = EqgatGenerator(
        d_model_eqgat,
        n_layers_eqgat,
        n_equi_feats_eqgat,
        vocab.size,
        cat_config.n_lig_atom_feats,
        d_edge_eqgat,
        cat_config.n_bond_types,
    )
    return egnn_gen


def get_egnn_model(args, vocab, cat_config):
    from semlaflow.models.egnn import VanillaEgnnGenerator

    egnn_gen = VanillaEgnnGenerator(
        args.d_model,
        args.n_layers,
        vocab.size,
        cat_config.n_lig_atom_feats,
        d_edge=args.d_edge,
        n_edge_types=cat_config.n_bond_types,
    )

    return egnn_gen


def get_model(args, vocab, cat_config, data_config):
    if args.arch == "semla":
        if data_config.is_complex:
            model = get_complex_semla_model(args, vocab, cat_config)
        else:
            model = get_semla_model(args, vocab, cat_config)
    elif args.arch == "eqgat":
        model = get_eqgat_model(args, vocab, cat_config)
    elif args.arch == "egnn":
        model = get_egnn_model(args, vocab, cat_config)
    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    return model


def get_pocket_encoder():
    return PocketEncoder()


def get_non_autoregressive_cfm(
    args,
    egnn_gen,
    pocket_enc,
    vocab,
    integrator,
    data_config,
    cat_config,
    train_config,
    hparams,
):
    if data_config.is_complex:
        cfm_cls = partial(
            ComplexMolecularCFM,
            pocket_encoder=pocket_enc,
            use_gvp=cat_config.use_gvp,
            use_complex_metrics=args.use_complex_metrics,
        )
    else:
        cfm_cls = BaseMolecularCFM
    fm_model = cfm_cls(
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
        **hparams,
    )
    return fm_model


def get_autoregressive_cfm(
    args,
    dm,
    egnn_gen,
    pocket_enc,
    vocab,
    integrator,
    data_config,
    cat_config,
    train_config,
    hparams,
):
    if data_config.is_complex:
        cfm_cls = partial(
            ARComplexMolecularCFM,
            pocket_encoder=pocket_enc,
            use_gvp=cat_config.use_gvp,
            use_complex_metrics=args.use_complex_metrics,
        )
    else:
        cfm_cls = ARMolecularCFM

    fm_model = cfm_cls(
        ar_interpolant=dm.train_interpolant,
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
        **hparams,
    )
    return fm_model


def get_cfm_model(
    args,
    dm,
    egnn_gen,
    pocket_enc,
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

    if args.categorical_strategy == "auto-regressive":
        fm_model = get_autoregressive_cfm(
            args,
            dm,
            egnn_gen,
            pocket_enc,
            vocab,
            integrator,
            data_config,
            cat_config,
            train_config,
            hparams,
        )
    elif args.categorical_strategy != "auto-regressive":
        fm_model = get_non_autoregressive_cfm(
            args,
            egnn_gen,
            pocket_enc,
            vocab,
            integrator,
            data_config,
            cat_config,
            train_config,
            hparams,
        )
    return fm_model


def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = get_hparams(args, dm)
    cat_config = get_categorical_config(args, vocab)
    data_config = get_dataset_config(args.dataset, args.is_pseudo_complex)
    train_config = get_train_config(args, dm)
    print(f"Total training steps {train_config.train_steps}")

    egnn_gen = get_model(args, vocab, cat_config, data_config)
    pocket_enc = get_pocket_encoder()

    print(f"Using model class {egnn_gen.__class__.__name__}")
    fm_model = get_cfm_model(
        args,
        dm,
        egnn_gen,
        pocket_enc,
        vocab,
        data_config,
        cat_config,
        train_config,
        hparams,
    )

    print(f"Using CFM class {fm_model.__class__.__name__}")
    return fm_model
