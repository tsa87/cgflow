import argparse

from rdkit import RDLogger

import cgflow.scriptutil as util

# turn off rdkit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def load_config(config_path, args):
    config = util.load_config(config_path)
    # TODO: add args override

    # TODO: sync configs; check the config is valid
    config.ligand_decoder.d_pocket_inv = config.pocket_encoder.d_inv
    assert config.ligand_decoder.d_equi == config.pocket_encoder.d_equi, (
        'ligand decoder "d_equi" must match pocket encoder "d_equi"'
    )
    return config


def main(config):
    from cgflow.buildutil import build_dm

    dm = build_dm(config)
    assert dm.train_dataset is not None, "Train dataset is not defined in the config"
    dm.train_dataset[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--config", type=str, default="./configs/cgflow/train.yaml")
    # TODO: add parameters
    args = parser.parse_args()
    config = load_config(args.config, args)
    main(config)
