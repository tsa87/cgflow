from cgflow.data.interpolate import ARGeometricInterpolant
from cgflow.models.fm import MolecularCFM
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


def get_cfm(config, model: PosePrediction, **kwargs) -> MolecularCFM:
    assert config.cfm._registry_ == "cfm"
    registry = Registry.get_register(config.cfm._registry_)
    obj_cls = registry[config.cfm._type_]
    return obj_cls(config, model, **kwargs)


def build_cfm(config, interpolant: ARGeometricInterpolant):
    model = build_model(config)
    cfm = get_cfm(config, model, interpolant=interpolant)
    return cfm


def build_model(config):
    poc_enc_config: PocketEncoderConfig = config.pocket_encoder
    lig_dec_config: LigandDecoderConfig = config.ligand_decoder

    # Match the dimension
    lig_dec_config.d_pocket_inv = poc_enc_config.d_inv
    lig_dec_config.d_pocket_equi = poc_enc_config.d_equi

    pocket_encoder = get_pocket_encoder(poc_enc_config)
    ligand_decoder = get_ligand_decoder(lig_dec_config)
    return PosePrediction(pocket_encoder, ligand_decoder)
