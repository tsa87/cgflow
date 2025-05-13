import tempfile

import numpy as np
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import HeteroData

from .data import ProteinGraphDataset

three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",  # unknown
}


def construct_protein_data_from_graph_gvp(
    protein_coords,
    protein_seq,
    protein_node_s,
    protein_node_v,
    protein_edge_index,
    protein_edge_s,
    protein_edge_v,
):
    n_protein_node = protein_coords.shape[0]
    keepNode = np.ones(n_protein_node, dtype=bool)
    input_node_xyz = protein_coords[keepNode]
    (
        input_edge_idx,
        input_protein_edge_s,
        input_protein_edge_v,
    ) = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode)

    # construct graph data.
    data = HeteroData()

    # additional information. keep records.
    data.seq = protein_seq[keepNode]
    data["protein"].coords = input_node_xyz
    data["protein"].node_s = protein_node_s[keepNode]  # [num_protein_nodes, num_protein_feautre]
    data["protein"].node_v = protein_node_v[keepNode]
    data["protein", "p2p", "protein"].edge_index = input_edge_idx
    data["protein", "p2p", "protein"].edge_s = input_protein_edge_s
    data["protein", "p2p", "protein"].edge_v = input_protein_edge_v
    pocket_center = data["protein"].coords.mean(axis=0)

    return data


def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode):
    # protein
    input_edge_list = []
    input_protein_edge_feature_idx = []
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    for res in res_list:
        if not (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res)):
            raise ValueError("Residue does not contain N, CA, C, O atoms, cannot merge with holo mols")

    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure["name"] = "placeholder"
    structure["seq"] = "".join([three_to_one.get(res.resname, "X") for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res["N"], res["CA"], res["C"], res["O"]]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure["coords"] = coords
    torch.set_num_threads(1)  # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (
        protein.x,
        protein.seq,
        protein.node_s,
        protein.node_v,
        protein.edge_index,
        protein.edge_s,
        protein.edge_v,
    )
    return x, protein


def get_pocket_data(holo_pock):
    parser = PDBParser()
    with tempfile.NamedTemporaryFile() as f:
        holo_pock.write_pdb(f.name)
        s = parser.get_structure("x", f.name)
        res_list = list(s.get_residues())
        protein_feat = get_protein_feature(res_list)[0]
        (
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
        ) = protein_feat
        pocket_data = construct_protein_data_from_graph_gvp(
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
        )
    return pocket_data


def pad_and_stack(data, ptr, length=None, device="cuda"):
    """Pad and stack a list of data tensors.
    Args:
        data: Tensor of shape (num_data, dim)
        ptr: Tensor of shape (batch_size + 1) with the pointers to the start of each batch
    """
    dim_size = data.shape[1]
    batch_size = ptr.shape[0] - 1

    if length is None:
        length = torch.max(ptr[1:] - ptr[:-1])

    result = torch.zeros((batch_size, length, dim_size))
    for i in range(batch_size):
        start, end = ptr[i], ptr[i + 1]
        result[i, : end - start, :] = data[start:end]

    return result.to(device)
