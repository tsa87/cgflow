import torch

from cgflow.models.model import (
    LigandDecoder,
    LigandDecoderConfig,
    PocketEncoder,
    PocketEncoderConfig,
    PosePrediction,
)
from cgflow.util.registry import Registry


def get_pocket_encoder(config) -> PocketEncoder:
    assert config._registry_ == "model"
    registry = Registry.get_register(config._registry_)
    obj_cls = registry[config._type_]
    return obj_cls(config)


def get_ligand_decoder(config) -> LigandDecoder:
    assert config._registry_ == "model"
    registry = Registry.get_register(config._registry_)
    obj_cls = registry[config._type_]
    return obj_cls(config)


def build_model(config, device: str | torch.device | None = None) -> PosePrediction:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poc_enc_config: PocketEncoderConfig = config.pocket_encoder
    lig_dec_config: LigandDecoderConfig = config.ligand_decoder

    pocket_encoder = get_pocket_encoder(poc_enc_config)
    ligand_decoder = get_ligand_decoder(lig_dec_config)
    model = PosePrediction(pocket_encoder, ligand_decoder)
    model.eval()
    model = model.to(device)
    return model
