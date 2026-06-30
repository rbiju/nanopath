import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


def newton_schulz(X, steps=5):
    """
    Compute the orthogonal factor of X via Newton-Schulz iteration.
    X must have shape (M, N) with M <= N. Returns a semi-orthogonal matrix
    with orthonormal rows, converging from any full-rank initialization.
    Implemented entirely as matmuls for GPU efficiency.
    """
    X = X / (X.norm() + 1e-8)
    for _ in range(steps):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X
    return X


class PrototypeHead(nn.Module):
    """
    Drop-in replacement for DINOHead that compares bottleneck embeddings
    against a learnable orthonormal prototype bank rather than projecting
    into a large unstructured space.

    Prototypes live in bottleneck_dim-space (same as the MLP output) and
    are kept orthonormal via Newton-Schulz during the forward pass.
    Requires n_prototypes <= bottleneck_dim.
    """

    def __init__(
        self,
        in_dim,
        n_prototypes,
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
        ns_steps=5,
    ):
        assert n_prototypes <= bottleneck_dim, (
            f"PrototypeHead requires n_prototypes ({n_prototypes}) "
            f"<= bottleneck_dim ({bottleneck_dim}) for orthogonality to be well-defined."
        )
        super().__init__()
        self.ns_steps = ns_steps
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim)
        self.apply(self._init_weights)
        self.prototypes = nn.Parameter(torch.empty(n_prototypes, bottleneck_dim))
        trunc_normal_(self.prototypes, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        ortho_prototypes = newton_schulz(self.prototypes, steps=self.ns_steps)
        return x @ ortho_prototypes.T
