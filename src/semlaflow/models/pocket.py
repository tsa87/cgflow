import lmdb
import pickle
from torch import Tensor, nn
import torch_geometric.data as gd
import semlaflow.util.functional as smolF

from semlaflow.gvp.models import GVP_embedding
from semlaflow.gvp.pocket import get_pocket_data
from semlaflow.util.pocket import ProteinPocket


class PocketCache:
    def __init__(self, cache_dir="./.gvp_cache", map_size=int(1e12)):
        self.env = lmdb.open(cache_dir, map_size=map_size)

    def _get(self, key):
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is not None:
                return pickle.loads(value)
        return None

    def _set(self, key, value):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(value))

    @staticmethod
    def process_pocket_data(holo_pock: ProteinPocket):
        return get_pocket_data(holo_pock)

    def get_pocket_data(self, holo_pock: ProteinPocket):
        if self._get(holo_pock.str_id) is not None:
            return self._get(holo_pock.str_id)
        else:
            data = self.process_pocket_data(holo_pock)
            self._set(holo_pock.str_id, data)
            return data

    def get_pocket_batch(self, holo_pocks: list[ProteinPocket]):
        data_list = [self.get_pocket_data(holo) for holo in holo_pocks]
        pocket_batch = gd.Batch.from_data_list(data_list)
        return pocket_batch


class PocketEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GVP_embedding((6, 3), (128, 16), (32, 1), (32, 1), seq_in=True)
        self.pocket_cache = PocketCache()

    # Get protein embeddings
    def get_pocket_features(self, pocket_data, device):
        p_node_feature = (
            pocket_data["protein"]["node_s"].to(device),
            pocket_data["protein"]["node_v"].to(device),
        )
        p_edge_index = pocket_data[("protein", "p2p", "protein")]["edge_index"].to(device)
        p_edge_feature = (
            pocket_data[("protein", "p2p", "protein")]["edge_s"].to(device),
            pocket_data[("protein", "p2p", "protein")]["edge_v"].to(device),
        )
        return (
            p_node_feature,
            p_edge_index,
            p_edge_feature,
            pocket_data.seq.to(device),
        )

    def forward(self, holo_pocks: list[ProteinPocket], holo: dict[str, Tensor]) -> Tensor:
        """Encode pocket atomic features

        Parameters
        ----------
        holo_pock : ProteinPocket
            single holo pocket
        holo : dict[str, Tensor]
            batch from the single data

        Returns
        -------
        pocket_atom_encoding: torch.Tensor
            [Nbatch, Natom, Fatom]
        """
        device = holo["atomics"].device
        pocket_batch = self.pocket_cache.get_pocket_batch(holo_pocks)
        p_node_feature, p_edge_index, p_edge_feature, seq = self.get_pocket_features(pocket_batch, device)
        pro_atom_encoding = self.encoder(p_node_feature, p_edge_index, p_edge_feature, seq)

        ptr = pocket_batch["protein"].ptr

        assert holo["mask"].sum() == pro_atom_encoding.size(0), "Mismatch in number of residues"
        pro_atom_encoding = smolF.pad_and_stack(pro_atom_encoding, ptr, length=holo["mask"].size(1), device=device)
        assert pro_atom_encoding.size(0) == holo["atomics"].size(0)
        assert pro_atom_encoding.size(1) == holo["atomics"].size(1)

        return pro_atom_encoding

    def forward_single(self, holo_pock: ProteinPocket, holo: dict[str, Tensor]) -> Tensor:
        """For API

        Parameters
        ----------
        holo_pock : ProteinPocket
            single holo pocket
        holo : dict[str, Tensor]
            batch from the single data

        Returns
        -------
        pocket_atom_encoding: torch.Tensor
            [1, Natom, Fatom]

        """
        device = holo["atomics"].device
        pocket_batch = get_pocket_data(holo_pock)
        p_node_feature, p_edge_index, p_edge_feature, seq = self.get_pocket_features(pocket_batch, device)
        pro_atom_encoding = self.encoder(p_node_feature, p_edge_index, p_edge_feature, seq)
        pro_atom_encoding = pro_atom_encoding.unsqueeze(0)  # match the size
        assert pro_atom_encoding.shape[:2] == holo["atomics"].shape[:2]
        return pro_atom_encoding
