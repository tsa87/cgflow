import torch
import numpy as np


class HarmonicSDE:
    # Code from https://github.com/HannesStark/FlowSite
    def __init__(
        self,
        N=None,
        edges=None,
        antiedges=None,
        a=0.5,
        b=0.3,
        J=None,
        diagonalize=True,
    ):
        self.use_cuda = False
        self.l = 1
        if not diagonalize:
            return
        if J is not None:
            J = J
            self.D, P = np.linalg.eigh(J)
            self.P = P
            self.N = self.D.size
            return

    @staticmethod
    def diagonalize(
        N: int,
        edges: torch.Tensor,
        antiedges: torch.Tensor | None = None,
        a=1,
        b=0.3,
        lamb: float | torch.Tensor = 0.0,
        ptr=None,
    ):
        if antiedges is None:
            antiedges = torch.zeros((2, 0), dtype=torch.long)
        if isinstance(lamb, float | int):
            lamb = torch.tensor([lamb])

        J = torch.zeros((N, N), device=edges.device)  # temporary fix
        for i, j in edges:
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        for i, j in antiedges:
            J[i, i] -= b
            J[j, j] -= b
            J[i, j] = J[j, i] = b
        J += torch.diag(torch.as_tensor(lamb))
        if ptr is None:
            return torch.linalg.eigh(J)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:], strict=False):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            Ds.append(D)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)
