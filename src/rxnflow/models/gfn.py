import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.envs.action import RxnActionType
from rxnflow.models.nn import init_weight_linear, mlp
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.policy.action_space_subsampling import ActionSubspace, ActionSubspaceForProtocol, SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env

ACT_BLOCK = nn.SiLU
ACT_MDP = nn.SiLU
ACT_TB = nn.LeakyReLU


def placeholder(size: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty(size, dtype=torch.float32, device=device)


def neginf(size: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.full(size, -torch.inf, dtype=torch.float32, device=device)


class RxnFlow(TrajectoryBalanceModel):
    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
    ) -> None:
        super().__init__()
        num_emb = cfg.model.num_emb
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings
        num_layers = cfg.model.num_mlp_layers
        num_emb_block = cfg.model.num_emb_block
        dropout = cfg.model.dropout

        # NOTE: State embedding
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim + env_ctx.num_graph_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.graph_transformer.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )
        # For regular GFN, normalization does not effect to model performance.
        # However, we add normalization to match the scale with reaction embedding
        self.norm_state = nn.LayerNorm(num_glob_final)

        # NOTE: Block embedding
        self.emb_block = BlockEmbedding(
            env_ctx.block_fp_dim,
            env_ctx.block_prop_dim,
            env_ctx.num_block_types,
            num_emb_block,
            num_emb_block,
            cfg.model.num_mlp_layers_block,
            ACT_BLOCK,
            dropout=0.0,
        )

        # NOTE: Markov Decision Process
        mlps = {
            "firstblock": mlp(num_glob_final, num_emb, num_emb_block, num_layers, ACT_MDP, dropout=dropout),
            "birxn": mlp(num_glob_final, num_emb, num_emb_block, num_layers, ACT_MDP, dropout=dropout),
        }
        self.mlp_mdp = nn.ModuleDict(mlps)

        # NOTE: Protocol Embeddings
        embs = {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.protocols}
        self.emb_protocol = nn.ParameterDict(embs)
        self.act_mdp = ACT_MDP()

        # NOTE: Etcs. (e.g., partition function)
        self._emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_layers, ACT_TB, dropout=dropout)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, ACT_TB, dropout=dropout)
        self._logit_scale = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, ACT_TB, dropout=dropout)
        self.reset_parameters()

    def emb2graph_out(self, emb: torch.Tensor) -> torch.Tensor:
        return self._emb2graph_out(emb)

    def logZ(self, cond_info: torch.Tensor) -> torch.Tensor:
        """return log partition funciton"""
        return self._logZ(cond_info)

    def logit_scale(self, cond_info: torch.Tensor) -> torch.Tensor:
        """return non-negative scale"""
        return nn.functional.elu(self._logit_scale(cond_info).view(-1)) + 1  # (-1, inf) -> (0, inf)

    def forward(self, g: gd.Batch, cond: torch.Tensor) -> tuple[RxnActionCategorical, torch.Tensor]:
        """

        Parameters
        ----------
        g : gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond : Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        -------
        RxnActionCategorical
        """
        _, emb = self.transf(g, torch.cat([cond, g.graph_attr], axis=-1))
        emb = self.norm_state(emb)
        protocol_masks = list(torch.unbind(g.protocol_mask, dim=1))  # [Ngraph, Nprotocol]
        logit_scale = self.logit_scale(cond)

        # action subsampling
        ctx: SynthesisEnvContext = get_worker_env("ctx")
        sampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        action_subspace: ActionSubspace = sampler.sample()

        logits = self._calculate_logits(emb, logit_scale, action_subspace, protocol_masks, ctx=ctx)
        fwd_cat = RxnActionCategorical(
            g,
            emb,
            logits,
            logit_scale,
            action_subspace,
            protocol_masks,
            model=self,
            ctx=ctx,
        )
        graph_out = self.emb2graph_out(emb)
        return fwd_cat, graph_out

    def _calculate_logits(
        self,
        emb: torch.Tensor,
        logit_scale: torch.Tensor,
        subspace: ActionSubspace,
        protocol_masks: list[torch.Tensor],
        ctx: SynthesisEnvContext,
    ) -> list[torch.Tensor]:
        # action subsampling
        dev: torch.device = emb.device
        num_graphs: int = emb.shape[0]

        masked_logits: list[torch.Tensor] = []
        for protocol_idx, protocol in enumerate(ctx.protocols):
            subsample = subspace[protocol.name]
            num_actions = subsample.num_actions
            protocol_mask = protocol_masks[protocol_idx]

            # calculate the logit for each action - (state, action)
            # shape: [Nstate, Naction]
            if protocol_mask.all():
                if protocol.action is RxnActionType.FirstBlock:
                    logits = self.hook_firstblock(emb, subsample, ctx, protocol.name)
                elif protocol.action is RxnActionType.BiRxn:
                    logits = self.hook_birxn(emb, subsample, ctx, protocol.name)
                else:
                    raise ValueError(protocol.action)
                logits = logits * logit_scale.view(-1, 1)

            elif protocol_mask.any():
                if protocol.action is RxnActionType.FirstBlock:
                    allowed_logits = self.hook_firstblock(emb[protocol_mask], subsample, ctx, protocol.name)
                elif protocol.action is RxnActionType.BiRxn:
                    allowed_logits = self.hook_birxn(emb[protocol_mask], subsample, ctx, protocol.name)
                else:
                    raise ValueError(protocol.action)
                allowed_logits = allowed_logits * logit_scale[protocol_mask].view(-1, 1)

                # create placeholder first and then insert the calculated.
                logits = neginf((num_graphs, num_actions), device=dev)
                logits[protocol_mask] = allowed_logits

            else:
                logits = neginf((num_graphs, num_actions), device=dev)

            masked_logits.append(logits)
        return masked_logits

    def hook_firstblock(
        self,
        emb: torch.Tensor,
        action_space: ActionSubspaceForProtocol,
        ctx: SynthesisEnvContext,
        protocol: str,
    ) -> torch.Tensor:
        """
        The hook function to be called for the FirstBlock.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        action_space : ActionSubspaceForProtocol
            action subspace.
        ctx: SynthesisEnvContext
            environment context.
        protocol: str
            The name of synthesis protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        # get block embeddings
        dev = emb.device
        block_data_list = [ctx.get_block_data(typ, indices) for typ, indices in action_space.subsamples.items()]
        typs, props, fps = list(zip(*block_data_list, strict=True))
        typ = torch.cat(typs).to(dev, non_blocking=True)
        prop = torch.cat(props).to(dev, non_blocking=True)
        fp = torch.cat(fps).to(dtype=torch.float32, device=dev, non_blocking=True)
        block_data = (typ, prop, fp)
        return self.forward_firstblock(emb, block_data, protocol)

    def forward_firstblock(
        self,
        emb: torch.Tensor,
        block_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        protocol: str,
    ):
        typ, prop, fp = block_data
        emb = emb + self.emb_protocol[protocol].view(1, -1)
        state_emb = self.mlp_mdp["firstblock"](self.act_mdp(emb))
        block_emb = self.emb_block(typ, prop, fp)
        return state_emb @ block_emb.T

    def hook_birxn(
        self,
        emb: torch.Tensor,
        action_space: ActionSubspaceForProtocol,
        ctx: SynthesisEnvContext,
        protocol: str,
    ) -> torch.Tensor:
        """
        The hook function to be called for the BiRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        action_space : ActionSubspaceForProtocol
            action subspace.
        ctx: SynthesisEnvContext
            environment context.
        protocol: str
            The name of synthesis protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        # get block embeddings
        dev = emb.device
        block_data_list = [ctx.get_block_data(typ, indices) for typ, indices in action_space.subsamples.items()]
        typs, props, fps = list(zip(*block_data_list, strict=True))
        typ = torch.cat(typs).to(dev, non_blocking=True)
        prop = torch.cat(props).to(dev, non_blocking=True)
        fp = torch.cat(fps).to(dtype=torch.float32, device=dev, non_blocking=True)
        block_data = (typ, prop, fp)
        return self.forward_birxn(emb, block_data, protocol)

    def forward_birxn(
        self,
        emb: torch.Tensor,
        block_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        protocol: str,
    ):
        typ, prop, fp = block_data
        emb = emb + self.emb_protocol[protocol].view(1, -1)
        state_emb = self.mlp_mdp["birxn"](self.act_mdp(emb))
        block_emb = self.emb_block(typ, prop, fp)
        return state_emb @ block_emb.T

    def reset_parameters(self):
        for m in self.mlp_mdp.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, ACT_MDP)

        for layer in [self._emb2graph_out, self._logZ, self._logit_scale]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    init_weight_linear(m, ACT_TB)


class BlockEmbedding(nn.Module):
    def __init__(
        self,
        fp_dim: int,
        prop_dim: int,
        n_type: int,
        n_hid: int,
        n_out: int,
        n_layers: int,
        act: type[nn.Module],
        dropout: float,
    ):
        super().__init__()
        self.emb_type = nn.Embedding(n_type, n_hid)
        self.lin_fp = nn.Sequential(
            nn.Linear(fp_dim, n_hid),
            nn.LayerNorm(n_hid),
            act(),
            nn.Dropout(dropout),
        )
        self.lin_prop = nn.Sequential(
            nn.Linear(prop_dim, n_hid),
            nn.LayerNorm(n_hid),
            act(),
            nn.Dropout(dropout),
        )
        self.mlp = mlp(3 * n_hid, n_hid, n_out, n_layers, act=act, dropout=dropout)
        self.reset_parameters(act)

    def forward(self, typ: torch.Tensor, prop: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        x_typ = self.emb_type(typ)
        x_prop = self.lin_prop(prop)
        x_fp = self.lin_fp(fp)
        x = torch.cat([x_typ, x_fp, x_prop], dim=-1)
        return self.mlp(x)

    def reset_parameters(self, act):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, act)
