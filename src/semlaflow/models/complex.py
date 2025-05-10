import copy
import torch
from torch import nn

from semlaflow.models import semla
import semlaflow.util.functional as smolF


def validate_all_or_none(*args):
    if any(arg is not None for arg in args):
        return all(arg is not None for arg in args)
    return True  # All are None, which is valid


class ComplexCoordAttention(nn.Module):
    def __init__(self, n_coord_sets, proj_sets=None, coord_norm="length", eps=1e-6):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.eps = eps

        self.complex_coord_norm = semla.ComplexCoordNorm(n_coord_sets, norm=coord_norm)
        self.lig_coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.pro_coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(
        self,
        lig_coord_sets,
        messages,
        adj_matrix,
        lig_node_mask,
        pro_coord_sets=None,
        pro_node_mask=None,
    ):
        """Compute an attention update for coordinate sets

        Args:
            lig_coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_lig_nodes, 3]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_lig_nodes, n_lig_nodes + n_pro_nodes, proj_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_lig_nodes, n_lig_nodes + n_pro_nodes]
            lig_node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_lig_nodes], 1 for real, 0 otherwise
            pro_coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_pro_nodes, 3]
            pro_node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_pro_nodes], 1 for real, 0 otherwise
        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """
        _, _, n_lig_nodes, _ = lig_coord_sets.shape

        if pro_coord_sets is not None:
            lig_coord_sets, pro_coord_sets = self.complex_coord_norm(
                lig_coord_sets, lig_node_mask, pro_coord_sets, pro_node_mask
            )
        else:
            lig_coord_sets = self.complex_coord_norm(lig_coord_sets, lig_node_mask)

        proj_lig_coord_sets = self.lig_coord_proj(lig_coord_sets.transpose(1, -1))

        if pro_coord_sets is not None:
            proj_pro_coord_sets = self.pro_coord_proj(pro_coord_sets.transpose(1, -1))
            # [B, 3, N_lig + N_pro, P]
            proj_coord_sets = torch.cat((proj_lig_coord_sets, proj_pro_coord_sets), dim=2)
        else:
            proj_coord_sets = proj_lig_coord_sets

        # proj_coord_sets shape [B, 3, N_lig + N_pro, P]
        # norm_dists shape [B, 3, N_lig + N_pro, N_lig + N_pro, P]
        vec_dists = proj_coord_sets.unsqueeze(3) - proj_coord_sets.unsqueeze(2)
        lengths = torch.linalg.vector_norm(vec_dists, dim=1, keepdim=True)
        norm_dists = vec_dists / (lengths + self.eps)
        norm_dists = norm_dists[:, :, :n_lig_nodes, :, :]

        attn_mask = semla.adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Dim 1 is currently 1 on dists so we need to unsqueeze attentions
        updates = norm_dists * attentions.unsqueeze(1)
        updates = updates.sum(dim=3)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=2))
        updates = updates * weights.unsqueeze(1)

        # updates shape [B, 3, N, P] -> [B, S, N, 3]
        updates = self.attn_proj(updates).transpose(1, -1)
        return updates


class ComplexEdgeMessages(nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        d_out,
        d_protein,
        n_coord_sets,
        d_ff=None,
        d_edge=None,
        d_protein_message=None,
        d_ff_protein=None,
        eps=1e-6,
        no_coord_norm=False,
    ):
        super().__init__()
        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff
        d_protein_message = d_message // 2 if d_protein_message is None else d_protein_message
        d_ff_protein = d_ff // 2 if d_ff_protein is None else d_ff_protein

        lig_lig_in_feats = (d_message * 2) + edge_feats + n_coord_sets
        pro_lig_in_feats = (d_protein_message * 2) + n_coord_sets

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.complex_coord_norm = semla.ComplexCoordNorm(n_coord_sets, norm="off" if no_coord_norm else "none")

        self.ligand_node_norm = torch.nn.LayerNorm(d_model)
        self.protein_node_norm = torch.nn.LayerNorm(d_protein)
        self.ligand_edge_norm = torch.nn.LayerNorm(d_edge) if d_edge is not None else None

        self.ligand_node_proj = torch.nn.Linear(d_model, d_message)
        self.ligand_node_proj_for_protein = torch.nn.Linear(d_model, d_protein_message)
        self.protein_node_proj = torch.nn.Linear(d_protein, d_protein_message)

        self.ligand_ligand_message_mlp = torch.nn.Sequential(
            torch.nn.Linear(lig_lig_in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_out),
        )
        self.protein_ligand_message_mlp = torch.nn.Sequential(
            torch.nn.Linear(pro_lig_in_feats, d_ff_protein),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff_protein, d_out),
        )

    def forward(
        self,
        lig_coords,
        lig_node_feats,
        lig_node_mask,
        lig_edge_feats=None,
        pro_coords=None,
        pro_node_feats=None,
        pro_node_mask=None,
    ):
        """Compute edge messages with optional protein complex
        Args:
            lig_coords (torch.Tensor): Ligand coordinate tensor, shape [batch_size, n_sets, n_lig_nodes, 3]
            lig_node_feats (torch.Tensor): Ligand Node features, shape [batch_size, n_lig_nodes, d_model]
            lig_node_mask (torch.Tensor): Mask for ligand nodes, shape [batch_size, n_sets, n_lig_nodes], 1 for real, 0 otherwise
            lig_edge_feats (Optional[torch.Tensor]): Incoming ligand edge features, shape [batch_size, n_nodes, n_lig_nodes, d_edge]
            pro_coords (Optional[torch.Tensor]): Protein coordinate tensor, shape [batch_size, n_sets, n_pro_nodes, 3]
            pro_node_feats (Optional[torch.Tensor]): Protein node features, shape [batch_size, n_pro_nodes, d_model]
            pro_node_mask (Optional[torch.Tensor]): Mask for protein nodes, shape [batch_size, n_sets, n_pro_nodes], 1 for real, 0 otherwise
        Returns:
        torch.Tensor: Edge messages tensor, shape [batch_size, n_nodes, n_nodes, d_out]
        """
        if not validate_all_or_none(pro_coords, pro_node_feats, pro_node_mask):
            raise ValueError("All protein inputs must be provided or none")

        if lig_edge_feats is not None and self.d_edge is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge as None.")

        if lig_edge_feats is None and self.d_edge is not None:
            raise ValueError("The model was initialised with d_edge but no edge feats were provided to forward fn.")

        # *** Normalise the node features ***
        batch_size, n_lig_nodes, _ = tuple(lig_node_feats.size())
        lig_node_feats = self.ligand_node_norm(lig_node_feats)
        if pro_coords is not None:
            _, n_pro_nodes, _ = tuple(pro_node_feats.size())
            pro_node_feats = self.protein_node_norm(pro_node_feats)

        # *** Normalise the coordinates ***
        if pro_coords is not None:
            lig_coords, pro_coords = self.complex_coord_norm(lig_coords, lig_node_mask, pro_coords, pro_node_mask)
            lig_coords = lig_coords.flatten(0, 1)
            pro_coords = pro_coords.flatten(0, 1)
        else:
            lig_coords = self.complex_coord_norm(lig_coords, lig_node_mask).flatten(0, 1)

        # *** Compute ligand ligand dot products of normalized coordinates ***
        # [B*S, N_lig, D] * [B*S, D, N_lig] -> [B*S, N_lig, N_lig]
        lig_lig_coord_dotprods = torch.bmm(lig_coords, lig_coords.transpose(1, 2))
        # [B, N_lig, N_lig, S]
        lig_lig_coord_feats = lig_lig_coord_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(1, -1)

        # *** Compute ligand protein dot products of normalized coordinates ***
        if pro_coords is not None:
            # [B*S, N_lig, D] * [B*S, D, N_pro] -> [B*S, N_lig, N_pro]
            lig_pro_coord_dotprods = torch.bmm(lig_coords, pro_coords.transpose(1, 2))
            # [B, N_lig, N_pro, S]
            lig_pro_coord_feats = lig_pro_coord_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(1, -1)

        # *** Project the ligand node features and build pair features ***
        # [B, N_lig, D] -> [B, N_lig, M]
        lig_node_feats_for_ligand = self.ligand_node_proj(lig_node_feats)
        # [B, N_lig, N_lig, M]
        lig_lig_node_feats_start = lig_node_feats_for_ligand.unsqueeze(2).expand(
            batch_size, n_lig_nodes, n_lig_nodes, -1
        )
        # [B, N_lig, N_lig, M]
        lig_lig_node_feats_end = lig_node_feats_for_ligand.unsqueeze(1).expand(batch_size, n_lig_nodes, n_lig_nodes, -1)
        # [B, N_lig, N_lig, M*2]
        lig_lig_node_pairs = torch.cat((lig_lig_node_feats_start, lig_lig_node_feats_end), dim=-1)
        # [B, N_lig, N_lig, M*2 + S]
        lig_lig_in_feats = torch.cat((lig_lig_node_pairs, lig_lig_coord_feats), dim=3)
        if lig_edge_feats is not None:
            lig_edge_feats = self.ligand_edge_norm(lig_edge_feats)
            # [B, N_lig, N_lig, M*2 + S + E]
            lig_lig_in_feats = torch.cat((lig_lig_in_feats, lig_edge_feats), dim=-1)

        lig_lig_message = self.ligand_ligand_message_mlp(lig_lig_in_feats)

        # *** Project the protein node features and build ligand-protein pair features ***
        if pro_node_feats is not None:
            # Create a more memory efficient version of the ligand-protein dot products
            lig_node_feats_for_protein = self.ligand_node_proj_for_protein(lig_node_feats)

            # [B, N_pro, D] -> [B, N_pro, M]
            pro_node_feats = self.protein_node_proj(pro_node_feats)
            # [B, N_lig, N_pro, M]
            lig_pro_node_feats_start = lig_node_feats_for_protein.unsqueeze(2).expand(
                batch_size, n_lig_nodes, n_pro_nodes, -1
            )
            # [B, N_lig, N_pro, M]
            lig_pro_node_feats_end = pro_node_feats.unsqueeze(1).expand(batch_size, n_lig_nodes, n_pro_nodes, -1)
            # [B, N_lig, N_pro, M*2]
            lig_pro_node_pairs = torch.cat((lig_pro_node_feats_start, lig_pro_node_feats_end), dim=-1)
            # [B, N_lig, N_pro, M*2 + S]
            lig_pro_in_feats = torch.cat((lig_pro_node_pairs, lig_pro_coord_feats), dim=-1)
            lig_pro_message = self.protein_ligand_message_mlp(lig_pro_in_feats)

            return torch.cat((lig_lig_message, lig_pro_message), dim=2)
        else:
            return lig_lig_message


class ComplexEquiMessagePassingLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge_in=None,
        d_edge_out=None,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}")

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_message_out = n_attn_heads + self.d_coord_message
        d_message_out = d_message_out + d_edge_out if d_edge_out is not None else d_message_out

        self.node_ff = semla.NodeFeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
        )
        self.message_ff = ComplexEdgeMessages(
            d_model,
            d_message,
            d_message_out,
            d_protein=d_model,
            n_coord_sets=n_coord_sets,
            d_ff=d_message_hidden,
            d_edge=d_edge_in,
            eps=eps,
            no_coord_norm=coord_norm == "off",
        )
        self.coord_attn = ComplexCoordAttention(n_coord_sets, self.d_coord_message, coord_norm=coord_norm, eps=eps)
        self.node_attn = semla.NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,  #
            "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(
        self,
        lig_coords,
        lig_node_feats,
        lig_adj_matrix,
        lig_node_mask,
        lig_edge_feats=None,
        pro_coords=None,
        pro_node_feats=None,
        pro_node_mask=None,
    ):
        """Pass data through the layer

        Args:
            lig_coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_lig_nodes, 3]
            lig_node_feats (torch.Tensor): Node features, shape [batch_size, n_lig_nodes, d_model]
            lig_adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_lig_nodes, n_lig_nodes]
            lig_node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_lig_nodes], 1 for real, 0 otherwise
            lig_edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_lig_nodes, n_lig_nodes, d_edge_in]
            pro_coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_pro_nodes, 3]
            pro_node_feats (torch.Tensor): Node features, shape [batch_size, n_pro_nodes, d_model]
            pro_node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_pro_nodes], 1 for real, 0 otherwise

        Returns:
            Either a two-tuple of the new node coordinates and the new node features, or a three-tuple of the new
            node coords, new node features and new edge features.
        """
        _, n_lig_nodes, _ = tuple(lig_node_feats.size())

        if lig_edge_feats is not None and self.d_edge_in is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge_in as None.")

        if lig_edge_feats is None and self.d_edge_in is not None:
            raise ValueError("The model was initialised with d_edge_in but no edge feats were provided to forward.")

        lig_coord_updates, lig_node_updates = self.node_ff(lig_coords, lig_node_feats, lig_node_mask)
        lig_coords = lig_coords + lig_coord_updates
        lig_node_feats = lig_node_feats + lig_node_updates

        messages = self.message_ff(
            lig_coords,
            lig_node_feats,
            lig_node_mask,
            lig_edge_feats=lig_edge_feats,
            pro_coords=pro_coords,
            pro_node_feats=pro_node_feats,
            pro_node_mask=pro_node_mask,
        )
        node_messages = messages[:, :, :, : self.n_attn_heads]
        coord_messages = messages[:, :, :, self.n_attn_heads : (self.n_attn_heads + self.d_coord_message)]

        if pro_coords is not None:
            cat_node_feats = torch.cat((lig_node_feats, pro_node_feats), dim=1)
            # [B, N_lig + N_pro, N_lig + N_pro]
            adj_matrix_from_node_mask = smolF.adj_from_node_mask(
                torch.cat((lig_node_mask[:, 0, :], pro_node_mask[:, 0, :]), dim=-1)
            )
            # [B, N_lig, N_pro]
            pro_lig_adj_matrix = adj_matrix_from_node_mask[:, :n_lig_nodes, n_lig_nodes:]
            # Get the original adjacency matrix for ligands
            # [B, N_lig, N_lig + N_pro]
            adj_matrix = torch.cat((lig_adj_matrix, pro_lig_adj_matrix), dim=2)
        else:
            cat_node_feats = lig_node_feats
            adj_matrix = lig_adj_matrix

        lig_node_feats = lig_node_feats + self.node_attn(cat_node_feats, node_messages, adj_matrix)
        lig_coords = lig_coords + self.coord_attn(
            lig_coords,
            coord_messages,
            adj_matrix,
            lig_node_mask,
            pro_coord_sets=pro_coords,
            pro_node_mask=pro_node_mask,
        )

        if self.d_edge_out is not None:
            edge_out = messages[:, :, :n_lig_nodes, (self.n_attn_heads + self.d_coord_message) :]
            edge_out = lig_edge_feats + edge_out if lig_edge_feats is not None else edge_out
            return lig_coords, lig_node_feats, edge_out

        return lig_coords, lig_node_feats


class ComplexEquiInvDynamics(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_layers,
        n_pro_layers,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge=None,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
        eps=1e-6,
        debug=False,
    ):
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("n_layers is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}")

        if n_pro_layers > n_layers:
            raise ValueError("n_pro_layers must be less than or equal to n_layers.")

        self._hparams = {
            "d_model": d_model,
            "d_message": d_message,
            "n_coord_sets": n_coord_sets,
            "n_layers": n_layers,
            "n_pro_layers": n_pro_layers,
            "n_attn_heads": n_attn_heads,
            "d_message_hidden": d_message_hidden,
            "d_edge": d_edge,
            "bond_refine": bond_refine,
            "self_cond": self_cond,
            "coord_norm": coord_norm,
            "eps": eps,
        }

        self.d_model = d_model
        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.bond_refine = bond_refine and d_edge is not None
        self.self_cond = self_cond
        self.n_layers = n_layers
        self.n_pro_layers = n_pro_layers
        self.debug = debug

        core_layer = ComplexEquiMessagePassingLayer(
            d_model,
            d_message,
            n_coord_sets,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            coord_norm=coord_norm,
            eps=eps,
        )
        layers = self._get_clones(core_layer, n_layers - extra_layers)

        if d_edge is not None:
            # Pass d_message_hidden as None so that these layers will have the same feats as their output
            in_layer = ComplexEquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_in=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            out_layer = ComplexEquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_out=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            layers = [in_layer] + layers + [out_layer]

        self.layers = torch.nn.ModuleList(layers)

        self.final_ff_block = semla.NodeFeedForward(d_model, n_coord_sets, coord_norm=coord_norm)
        self.coord_norm = semla.CoordNorm(n_coord_sets, norm=coord_norm)
        self.feat_norm = torch.nn.LayerNorm(d_model)

        in_coord_sets = 2 if self_cond else 1

        self.lig_coord_proj = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.pro_coord_proj = torch.nn.Linear(1, n_coord_sets, bias=False)

        self.coord_head = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = torch.nn.LayerNorm(d_edge)

        if self.bond_refine:
            self.refine_layer = semla.BondRefine(d_model, d_message, d_edge)

    @property
    def hparams(self):
        return self._hparams

    def forward(
        self,
        lig_coords,
        lig_inv_feats,
        lig_adj_matrix,
        lig_atom_mask=None,
        lig_edge_feats=None,
        lig_cond_coords=None,
        pro_coords=None,
        pro_inv_feats=None,
        pro_atom_mask=None,
    ):
        """Generate molecular coordinates and atom features

        Args:
            lig_coords (torch.Tensor): Input coordinates, shape [batch_size, n_lig_atoms, 3]
            lig_inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_lig_atoms, d_model]
            lig_adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_lig_atoms, n_lig_atoms], 1 for connected
            lig_atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_lig_atoms], 1 for real atoms
            lig_edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_lig_nodes, n_lig_nodes, d_edge]
            lig_cond_coords (torch.Tensor, Optional): Conditional coords, shape [batch_size, n_lig_nodes, 3]
            pro_coords (torch.Tensor, Optional): Input coordinates, shape [batch_size, n_pro_atoms, 3]
            pro_inv_feats (torch.Tensor, Optional): Invariant atom features, shape [batch_size, n_pro_atoms, d_model]
            pro_atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_pro_atoms], 1 for real atoms

        Returns:
            (coords, atom feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        if lig_edge_feats is not None and self.d_edge is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge as None.")

        if lig_edge_feats is None and self.d_edge is not None:
            raise ValueError("The model was initialised with d_edge but no edge feats were provided to forward.")

        if lig_cond_coords is not None and not self.self_cond:
            raise ValueError("cond_coords was provided but the model was initialised with self_cond as False.")

        if lig_cond_coords is None and self.self_cond:
            raise ValueError("The model was initialsed with self_cond but cond_coords was not provided.")

        # Project single coord set into a multiple learnable coord sets, while maintaining equivariance
        lig_coords = (
            torch.stack((lig_coords, lig_cond_coords)) if lig_cond_coords is not None else lig_coords.unsqueeze(0)
        )
        lig_coords = self.lig_coord_proj(lig_coords.movedim(0, -1)).movedim(-1, 1)

        lig_atom_mask = lig_atom_mask.unsqueeze(1).expand(-1, self.n_coord_sets, -1)
        lig_coords = lig_coords * lig_atom_mask.unsqueeze(-1)

        if pro_coords is not None:
            pro_coords = self.pro_coord_proj(pro_coords.unsqueeze(0).movedim(0, -1)).movedim(-1, 1)
            pro_atom_mask = pro_atom_mask.unsqueeze(1).expand(-1, self.n_coord_sets, -1)
            pro_coords = pro_coords * pro_atom_mask.unsqueeze(-1)

        # Update coords and node feats using the model layers
        for i, layer in enumerate(self.layers):
            if i >= (self.n_layers - self.n_pro_layers):
                out = layer(
                    lig_coords,
                    lig_inv_feats,
                    lig_adj_matrix,
                    lig_atom_mask,
                    lig_edge_feats=lig_edge_feats,
                    pro_coords=pro_coords,
                    pro_node_feats=pro_inv_feats,
                    pro_node_mask=pro_atom_mask,
                )
            else:
                out = layer(
                    lig_coords,
                    lig_inv_feats,
                    lig_adj_matrix,
                    lig_atom_mask,
                    lig_edge_feats=lig_edge_feats,
                )

            if len(out) == 2:
                lig_coords, lig_inv_feats = out
                lig_edge_feats = None

            elif len(out) == 3:
                lig_coords, lig_inv_feats, lig_edge_feats = out

        # Apply a final feedforward block and project coord sets to single coord set
        # HACK: update here instead of setting the values
        if self.debug:
            lig_coords, lig_inv_feats = self.final_ff_block(lig_coords, lig_inv_feats, lig_atom_mask)
            lig_out_coords = self.coord_norm(lig_coords, lig_atom_mask)
        else:
            lig_coords_update, lig_inv_feats_update = self.final_ff_block(lig_coords, lig_inv_feats, lig_atom_mask)
            lig_out_coords = lig_coords + lig_coords_update
            lig_inv_feats = lig_inv_feats + lig_inv_feats_update

        lig_out_coords = self.coord_head(lig_out_coords.transpose(1, -1))
        lig_out_coords = lig_out_coords.transpose(1, -1).squeeze(1)

        if self.bond_refine:
            lig_atom_mask = lig_atom_mask[:, 0, :]
            lig_edge_feats = self.refine_layer(lig_out_coords, lig_inv_feats, lig_atom_mask, lig_edge_feats)

        lig_inv_feats = self.feat_norm(lig_inv_feats)

        if self.d_edge is None:
            return lig_out_coords, lig_inv_feats

        lig_edge_feats = self.bond_norm(lig_edge_feats)
        return lig_out_coords, lig_inv_feats, lig_edge_feats

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class ComplexSemlaGenerator(semla.MolecularGenerator):
    def __init__(
        self,
        d_model,
        dynamics,
        vocab_size,
        n_lig_atom_feats,
        n_pro_atom_feats=None,
        d_edge=None,
        n_edge_types=None,
        self_cond=False,
        size_emb=64,
        max_atoms=256,
        debug=False,
    ):

        hparams = {
            "d_model": d_model,
            "vocab_size": vocab_size,
            "n_lig_atom_feats": n_lig_atom_feats,
            "n_pro_atom_feats": n_pro_atom_feats,
            "d_edge": d_edge,
            "n_edge_types": n_edge_types,
            "self_cond": self_cond,
            "size_emb": size_emb,
            "max_atoms": max_atoms,
            **dynamics.hparams,
        }

        super().__init__(**hparams)

        self.debug = debug
        self.self_cond = self_cond
        self.n_pro_atom_feats = n_pro_atom_feats

        if d_edge is not None or n_edge_types is not None:
            if None in [d_edge, n_edge_types]:
                raise ValueError("If either d_edge or n_edge_types are given both must be provided.")

            edge_in_feats = n_edge_types * 2 if self_cond else n_edge_types

            self.edge_in_proj = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, d_edge),
            )
            self.edge_out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, n_edge_types),
            )

        lig_in_feats = n_lig_atom_feats + vocab_size if self_cond else n_lig_atom_feats
        lig_in_feats = lig_in_feats + size_emb

        self.size_emb = torch.nn.Embedding(max_atoms, size_emb)
        self.lig_feat_proj = torch.nn.Sequential(
            torch.nn.Linear(lig_in_feats, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, d_model),
        )

        if n_pro_atom_feats is not None:
            pro_in_feats = n_pro_atom_feats
            self.pro_feat_proj = torch.nn.Sequential(
                torch.nn.Linear(pro_in_feats, d_model),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_model, d_model),
            )

        self.dynamics = dynamics

        self.atom_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, vocab_size),
        )
        self.charge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, 7),
        )

    def forward(
        self,
        lig_coords,
        lig_inv_feats,
        lig_edge_feats=None,
        lig_cond_coords=None,
        lig_cond_atomics=None,
        lig_cond_bonds=None,
        lig_atom_mask=None,
        pro_coords=None,
        pro_inv_feats=None,
        pro_atom_mask=None,
    ):
        """Predict molecular coordinates and atom types

        Args:
            lig_coords (torch.Tensor): Input coordinates, shape [batch_size, n_lig_atoms, 3]
            lig_inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_lig_atoms, n_feats]
            lig_edge_feats (torch.Tensor): In edge features, shape [batch_size, n_lig_atoms, n_lig_atoms, n_edge_types]
            lig_cond_coords (torch.Tensor): Conditional coords, shape [batch_size, n_lig_atoms, 3]
            lig_cond_atomics (torch.Tensor): Conditional atom type logits, shape [batch_size, n_lig_atoms, n_feats]
            lig_cond_bonds (torch.Tensor): Cond bond type logits, shape [batch_size, n_lig_atoms, n_lig_atoms, n_edge_types]
            lig_atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_lig_atoms], 1 for real atoms
            pro_coords (torch.Tensor): Input coordinates, shape [batch_size, n_pro_atoms, 3]
            pro_inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_pro_atoms, n_feats]
            pro_atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_pro_atoms], 1 for real atoms

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges)
            All torch.Tensor, shapes:
                Coordinates: [batch_size, n_lig_atoms, 3]
                Type logits: [batch_size, n_lig_atoms, vocab_size],
                Bond logits: [batch_size, n_lig_atoms, n_lig_atoms, n_edge_types]
                Charge logits: [batch_size, n_lig_atoms, 7]
        """

        if lig_cond_coords is not None and not self.self_cond:
            raise ValueError("cond_coords was provided but the model was initialised with self_cond as False.")

        if lig_cond_coords is None and self.self_cond:
            raise ValueError("The model was initialsed with self_cond but cond_coords was not provided.")

        if lig_edge_feats is None and lig_cond_bonds is not None:
            raise ValueError("edge_feats must be provided if using bond conditioning.")

        if pro_inv_feats is not None and self.n_pro_atom_feats is None:
            raise ValueError("pro_coords was provided but the model was initialised with n_pro_atom_feats as None.")

        lig_atom_mask = torch.ones_like(lig_coords[..., 0]) if lig_atom_mask is None else lig_atom_mask
        lig_adj_matrix = smolF.edges_from_nodes(lig_coords, node_mask=lig_atom_mask)

        # Embed the number of atoms in a mol into a small vector and concat this to inv feats for each atom
        n_lig_atoms = lig_atom_mask.sum(dim=-1, keepdim=True)
        size_lig_emb = self.size_emb(n_lig_atoms).expand(-1, lig_inv_feats.size(1), -1)

        lig_inv_feats = torch.cat((lig_inv_feats, size_lig_emb), dim=-1)
        if lig_cond_atomics is not None:
            lig_inv_feats = torch.cat((lig_inv_feats, lig_cond_atomics), dim=-1)

        lig_atom_feats = self.lig_feat_proj(lig_inv_feats)

        if lig_edge_feats is not None:
            lig_edge_feats = lig_edge_feats.float()
            lig_edge_feats = (
                torch.cat((lig_edge_feats, lig_cond_bonds), dim=-1) if lig_cond_bonds is not None else lig_edge_feats
            )
            lig_edge_feats = self.edge_in_proj(lig_edge_feats)

        pro_atom_feats = None
        if pro_inv_feats is not None:
            pro_atom_mask = torch.ones_like(pro_coords[..., 0]) if pro_atom_mask is None else pro_atom_mask
            pro_atom_feats = self.pro_feat_proj(pro_inv_feats)

        out = self.dynamics(
            lig_coords,
            lig_atom_feats,
            lig_adj_matrix,
            lig_atom_mask=lig_atom_mask,
            lig_edge_feats=lig_edge_feats,
            lig_cond_coords=lig_cond_coords,
            pro_coords=pro_coords,
            pro_inv_feats=pro_atom_feats,
            pro_atom_mask=pro_atom_mask,
        )

        lig_pred_edges = None
        if len(out) == 2:
            lig_pred_coords, lig_pred_feats = out
        elif len(out) == 3:
            lig_pred_coords, lig_pred_feats, lig_pred_edges = out

        # HACK: If we have pockets we don't zero the com of the ligand
        if self.debug:
            lig_pred_coords = smolF.zero_com(lig_pred_coords, node_mask=lig_atom_mask)

        lig_pred_coords = lig_pred_coords * lig_atom_mask.unsqueeze(-1)

        lig_type_logits = self.atom_classifier_head(lig_pred_feats)
        lig_charge_logits = self.charge_classifier_head(lig_pred_feats)

        # If we are predicting edges ensure that the matrix is symmetrical
        if lig_pred_edges is not None:
            lig_pred_edges = lig_pred_edges + lig_pred_edges.transpose(1, 2)
            lig_edge_logits = self.edge_out_proj(lig_pred_edges)
            return lig_pred_coords, lig_type_logits, lig_edge_logits, lig_charge_logits

        return lig_pred_coords, lig_type_logits, lig_charge_logits
