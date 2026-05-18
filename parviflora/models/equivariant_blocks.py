"""
Reusable equivariant building blocks for e3nn-based policies.

Two block types share the same gated-nonlinearity design
(Weiler et al. "3D Steerable CNNs") but differ in how they
mix features across channels:

  LinearBlock   – uses o3.Linear  (no coupling across l-types,
                  O(n) parameter count, good for the actor where
                  the equivariant structure alone is expressive).

  TPBlock       – uses o3.FullyConnectedTensorProduct (couples
                  channels of different l via Clebsch-Gordan
                  products, needed wherever vector info must flow
                  into scalar outputs, e.g. the Q-function).
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

# ── Shared hidden irrep specification ────────────────────────────────────────


def make_hidden_irreps(n_scalars: int, n_vectors: int) -> o3.Irreps:
    return o3.Irreps(f"{n_scalars}x0e + {n_vectors}x1o")


def make_gate(n_scalars: int, n_vectors: int) -> Gate:
    """
    Gated nonlinearity for a hidden layer of n_scalars + n_vectors channels.

    The gate needs one additional scalar per vector channel, so the layer
    feeding into this Gate must produce:
        n_scalars × 0e  (pass-through scalars, activated by tanh)
        n_vectors × 0e  (gate scalars, activated by sigmoid)
        n_vectors × 1o  (gated vectors)

    gate.irreps_in  = "(n_scalars + n_vectors)x0e + n_vectors x1o"
    gate.irreps_out = "n_scalars x0e + n_vectors x1o"
    """
    return Gate(
        irreps_scalars=o3.Irreps(f"{n_scalars}x0e"),
        act_scalars=[torch.tanh],
        irreps_gates=o3.Irreps(f"{n_vectors}x0e"),
        act_gates=[torch.sigmoid],
        irreps_gated=o3.Irreps(f"{n_vectors}x1o"),
    )


# ── LinearBlock ──────────────────────────────────────────────────────────────


class LinearBlock(nn.Module):
    """
    Equivariant layer: o3.Linear → Gate.

    o3.Linear mixes channels of the *same* l independently
    (scalars with scalars, vectors with vectors). It cannot
    create new l-types. Sufficient for the actor where the
    final output irreps are also "0e + 1o".

    irreps_in  : arbitrary input irreps
    irreps_out : n_scalars × 0e  +  n_vectors × 1o
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        n_scalars: int = 16,
        n_vectors: int = 8,
    ) -> None:
        super().__init__()
        self.gate = make_gate(n_scalars, n_vectors)
        # Linear must produce exactly what the gate expects as input
        self.linear = o3.Linear(irreps_in, self.gate.irreps_in)

    @property
    def irreps_out(self) -> o3.Irreps:
        return self.gate.irreps_out  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.linear(x))


# ── TPBlock ───────────────────────────────────────────────────────────────────


class TPBlock(nn.Module):
    """
    Equivariant layer: FullyConnectedTensorProduct → Gate.

    The tensor product computes Clebsch-Gordan products between
    all pairs of input channels (self-interaction: tp(x, x)).
    Crucially this generates 0e outputs from 1o⊗1o paths
    (i.e. inner products), allowing vector information to flow
    into scalar channels.  Required for the Q-function whose
    final output is an invariant scalar.

    irreps_in  : arbitrary input irreps
    irreps_out : n_scalars × 0e  +  n_vectors × 1o
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        n_scalars: int = 16,
        n_vectors: int = 8,
    ) -> None:
        super().__init__()
        self.gate = make_gate(n_scalars, n_vectors)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_in, self.gate.irreps_in
        )

    @property
    def irreps_out(self) -> o3.Irreps:
        return self.gate.irreps_out  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.tp(x, x))
