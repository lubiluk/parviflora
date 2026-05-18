"""
Full nonlinear equivariant SAC policy for goal-reaching tasks (Reach).

Key design decisions vs the MLP baseline
─────────────────────────────────────────
Actor
  • Two LinearBlock hidden layers keep the actor equivariant under SO(3):
    rotating the scene rotates the output action by the same amount.
  • Mean output is "1x1o + 1x0e" (displacement vector + gripper scalar).
  • Log-std output is "2x0e" — two *invariant* scalars:
      log_std[0]  shared isotropic std for the xyz spatial action
      log_std[1]  std for the gripper scalar action
    Using a *single* std for all three spatial dims is the only choice that
    keeps the Gaussian distribution equivariant: N(Rμ, σ²I) = R·N(μ, σ²I).
  • xyz squashing uses norm_squash (scale magnitude, preserve direction)
    instead of element-wise tanh, which would break equivariance.

Why we augment the scalar input channels
─────────────────────────────────────────
  o3.Linear can only map 0e→0e and 1o→1o (Schur's lemma: zero weights are
  allocated for any 1o→0e path).  This means the Gate scalars — which
  multiplicatively control the vector channels — can only be functions of
  the *original scalar inputs* (gripper_sc).  After two LinearBlocks the
  actor's effective form is:

      mu_xyz  = f(gripper_sc)·error_vec + g(gripper_sc)·ee_vel
      log_std = h(gripper_sc)

  where f, g, h are constants (gripper barely moves in a reach task).
  This makes the actor a fixed-gain PD controller that cannot reduce its
  proportional gain or noise as it nears the goal → oscillation.

  Fix: pre-compute three SO(3)-invariant scalars from the vectors and
  append them as extra 0e features before block1:

      norm_err  = ‖error_vec‖   — distance to goal
      norm_vel  = ‖ee_vel‖      — end-effector speed
      dot_ev    = error_vec·ee_vel — velocity alignment with goal direction

  Input irreps become "2x1o + 7x0e" (was "2x1o + 4x0e").  Equivariance
  is preserved because these scalars are rotation-invariant by construction.

Q-function
  • Standard MLP over hand-crafted SO(3)-invariant scalar features.
    Replacing the earlier TPBlock design: empirically the TP Q-function
    diverged badly early in training (loss_q peaked at 49) because many
    Clebsch-Gordan paths make the loss landscape hard to optimise before
    the weights settle.  An MLP critic converges quickly and stably.
  • Invariant features extracted from obs + action (11 scalars total):
      ‖error_vec‖, ‖ee_vel‖, ‖act_xyz‖          norms
      error_vec·ee_vel, error_vec·act_xyz,
      ee_vel·act_xyz                             inner products
      gripper_sc (4), act_g (1)                 already scalars
  • Two hidden layers [64 × 64] with ReLU, scalar output.
"""

import math
from typing import Union

import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from numpy.typing import NDArray

from ..extractors.equivariant_reach_extractor import EquivariantReachExtractor
from ..models.equivariant_blocks import LinearBlock
from ..models.mlp import mlp
from ..utils.observation import unsqueeze_observation

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ── Equivariant squashing helpers ─────────────────────────────────────────────


def norm_squash(v: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Equivariant squashing for a 3-D action vector.

    Scales the magnitude by tanh and preserves the direction:
        f(v) = limit · tanh(‖v‖) · v / ‖v‖

    Equivariance proof:
        f(Rv) = limit · tanh(‖Rv‖) · Rv / ‖Rv‖
              = limit · tanh(‖v‖)  · R(v / ‖v‖)
              = R · f(v)   ✓
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return limit * torch.tanh(norm) * (v / norm)


def norm_squash_log_det(pre_squash: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Log |det Jacobian| of norm_squash for a 3-D vector.

    J has two distinct eigenvalues:
        limit · tanh(r)/r   (multiplicity 2, perpendicular to v)
        limit · sech²(r)    (multiplicity 1, along v)

    det J = limit³ · (tanh(r)/r)² · sech²(r)

    log|det J| = 3·log(limit)
               + 2·[log tanh(r) − log r]
               + log(1 − tanh²(r))

    This is well-behaved at r → 0 (limit³ · I Jacobian)
    and diverges to −∞ at r → ∞ (boundary of action space).
    """
    r = pre_squash.norm(dim=-1).clamp(min=1e-8)  # (B,)
    tanh_r = torch.tanh(r)
    return (
        3.0 * math.log(limit)
        + 2.0 * (torch.log(tanh_r.clamp(min=1e-8)) - torch.log(r))
        + torch.log(1.0 - tanh_r**2 + 1e-6)
    )


# ── Actor ─────────────────────────────────────────────────────────────────────


class EquivariantSACActor(nn.Module):
    """
    Stochastic equivariant actor for SAC.

    Interface identical to SquashedGaussianMLPActor:
        forward(obs, deterministic, with_logprob) → (action, logp)

    Architecture:
        extractor  →  LinearBlock  →  LinearBlock
                   →  mu_layer  ("1x1o + 1x0e")   equivariant mean
                   →  log_std_layer ("2x0e")        invariant log-stds
    """

    # Augmented input irreps: extractor gives "2x1o + 4x0e" (10 dims), but
    # we append 3 invariant scalars (norm_err, norm_vel, dot_ev) so the Gate
    # values can depend on distance and velocity, not just gripper state.
    IRREPS_IN_AUG = o3.Irreps("2x1o + 7x0e")  # 6 + 7 = 13 dims

    def __init__(
        self,
        extractor: EquivariantReachExtractor,
        act_limit: float,
        n_scalars: int = 16,
        n_vectors: int = 8,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.act_limit = act_limit

        self.block1 = LinearBlock(self.IRREPS_IN_AUG, n_scalars, n_vectors)
        self.block2 = LinearBlock(self.block1.irreps_out, n_scalars, n_vectors)

        hidden = self.block2.irreps_out  # "n_scalars x 0e + n_vectors x 1o"

        # Equivariant mean: displacement vector (1x1o) + gripper scalar (1x0e)
        # o3.Linear maps:  n_scalars x 0e → 1x0e  and  n_vectors x 1o → 1x1o
        self.mu_layer = o3.Linear(hidden, o3.Irreps("1x1o + 1x0e"))

        # Invariant log-stds: two scalars only (0e)
        #   index 0 → shared isotropic σ for all three xyz dims
        #   index 1 → σ for gripper scalar
        self.log_std_layer = o3.Linear(hidden, o3.Irreps("2x0e"))

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        with_logprob: bool = True,
    ):
        device = next(self.parameters()).device
        x = self.extractor(obs, device=device)  # (B, 10)  layout: "2x1o + 4x0e"

        # ── augment scalar channels with geometry-aware invariants ────────
        # o3.Linear cannot produce 0e from 1o (Schur's lemma), so without
        # this step the Gate values would be blind to ‖error_vec‖ and ‖ee_vel‖.
        error_vec_raw = x[..., :3]  # (B, 3)
        ee_vel_raw = x[..., 3:6]  # (B, 3)
        norm_err = error_vec_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B,1)
        norm_vel = ee_vel_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B,1)
        dot_ev = (error_vec_raw * ee_vel_raw).sum(-1, keepdim=True)  # (B,1)
        # Layout must match IRREPS_IN_AUG = "2x1o + 7x0e":
        #   vectors first (dims 0-5), then all scalars (dims 6-12)
        x_aug = torch.cat([x, norm_err, norm_vel, dot_ev], dim=-1)  # (B, 13)

        h = self.block2(self.block1(x_aug))  # (B, hidden_dim)

        # ── mean and log-std ──────────────────────────────────────────────
        mu = self.mu_layer(h)  # (B, 4)
        log_std = self.log_std_layer(h).clamp(  # (B, 2)
            LOG_STD_MIN, LOG_STD_MAX
        )
        std = log_std.exp()  # (B, 2)

        # mu layout follows "1x1o + 1x0e": first 3 dims = vector, last = scalar
        mu_xyz = mu[..., :3]  # (B, 3)  equivariant vector
        mu_g = mu[..., 3:4]  # (B, 1)  invariant scalar

        std_xyz = std[..., 0:1]  # (B, 1)  shared for all 3 spatial dims
        std_g = std[..., 1:2]  # (B, 1)

        # ── sample ───────────────────────────────────────────────────────
        if deterministic:
            pre_xyz = mu_xyz
            pre_g = mu_g
        else:
            # Reparameterisation trick; isotropic noise preserves equivariance
            pre_xyz = mu_xyz + std_xyz * torch.randn_like(mu_xyz)
            pre_g = mu_g + std_g * torch.randn_like(mu_g)

        # ── equivariant squashing ─────────────────────────────────────────
        action_xyz = norm_squash(pre_xyz, self.act_limit)  # (B, 3)
        action_g = self.act_limit * torch.tanh(pre_g)  # (B, 1)
        action = torch.cat([action_xyz, action_g], dim=-1)  # (B, 4)

        # ── log probability ───────────────────────────────────────────────
        logp = None
        if with_logprob:
            # --- xyz: isotropic Gaussian + norm squashing ---
            # log p(pre_xyz) = −3/2·log(2π) − 3·log(σ_xyz)
            #                  − ‖pre_xyz − μ_xyz‖² / (2·σ_xyz²)
            sq_err_xyz = ((pre_xyz - mu_xyz) ** 2).sum(-1)  # (B,)
            log_p_pre_xyz = (
                -1.5 * math.log(2.0 * math.pi)
                - 3.0 * log_std[..., 0]  # 3 × (−log σ_xyz)
                - 0.5 * sq_err_xyz / (std_xyz.squeeze(-1) ** 2 + 1e-8)
            )  # (B,)
            logp_xyz = log_p_pre_xyz - norm_squash_log_det(pre_xyz, self.act_limit)

            # --- gripper: scalar Gaussian + tanh squashing ---
            sq_err_g = ((pre_g - mu_g) / (std_g + 1e-8)).squeeze(-1) ** 2
            log_p_pre_g = (
                -0.5 * math.log(2.0 * math.pi) - log_std[..., 1] - 0.5 * sq_err_g
            )  # (B,)
            # Numerically stable tanh correction (same form as original SAC)
            log_det_g = torch.log(
                self.act_limit * (1.0 - torch.tanh(pre_g).pow(2)) + 1e-6
            ).squeeze(-1)  # (B,)
            logp_g = log_p_pre_g - log_det_g

            logp = logp_xyz + logp_g  # (B,)

        return action, logp


# ── Q-function ────────────────────────────────────────────────────────────────


class InvariantMLPQFunction(nn.Module):
    """
    SO(3)-invariant Q-function: standard MLP over hand-crafted invariant scalars.

    Why not TPBlock here?
    The TP Q-function diverged badly in practice: the many CG paths create a
    difficult early-training loss landscape and caused alpha to spiral to 0.58,
    loss_q to peak at 49, and the policy to stall until ~63 K steps.
    A plain MLP over invariant features converges in the same ~20 K steps as
    the MLP baseline, giving the equivariant actor a stable critic from the start.

    Invariant features derived from extractor output + action (11 scalars):
        ‖error_vec‖              distance to goal
        ‖ee_vel‖                 end-effector speed
        ‖act_xyz‖                spatial action magnitude
        error_vec · ee_vel       velocity alignment with goal direction
        error_vec · act_xyz      action alignment with goal direction
        ee_vel    · act_xyz      action alignment with current motion
        gripper_sc[0–3]          gripper finger states   (4 scalars)
        act_g                    gripper action          (1 scalar)
    """

    N_FEATURES = 11

    def __init__(
        self,
        extractor: EquivariantReachExtractor,
        hidden_sizes: tuple = (64, 64),
        activation: type = nn.ReLU,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.q = mlp([self.N_FEATURES] + list(hidden_sizes) + [1], activation)

    def _invariant_features(
        self, obs: dict, act: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        x_obs = self.extractor(obs, device=device)  # (B, 10)

        error_vec = x_obs[..., :3]  # (B, 3)  1x1o
        ee_vel = x_obs[..., 3:6]  # (B, 3)  1x1o
        gripper_sc = x_obs[..., 6:]  # (B, 4)  4x0e

        act_xyz = act[..., :3]  # (B, 3)
        act_g = act[..., 3:4]  # (B, 1)

        # --- norms (invariant) ---
        norm_err = error_vec.norm(dim=-1, keepdim=True)  # (B, 1)
        norm_vel = ee_vel.norm(dim=-1, keepdim=True)  # (B, 1)
        norm_act = act_xyz.norm(dim=-1, keepdim=True)  # (B, 1)

        # --- inner products (invariant) ---
        dot_ev = (error_vec * ee_vel).sum(-1, keepdim=True)  # (B, 1)
        dot_ea = (error_vec * act_xyz).sum(-1, keepdim=True)  # (B, 1)
        dot_va = (ee_vel * act_xyz).sum(-1, keepdim=True)  # (B, 1)

        return torch.cat(
            [norm_err, norm_vel, norm_act, dot_ev, dot_ea, dot_va, gripper_sc, act_g],
            dim=-1,
        )  # (B, 11)

    def forward(self, obs: dict, act: torch.Tensor) -> torch.Tensor:
        feats = self._invariant_features(obs, act, act.device)  # (B, 11)
        return self.q(feats).squeeze(-1)  # (B,)


# ── Combined policy ───────────────────────────────────────────────────────────


class EquivariantPolicy(nn.Module):
    """
    Full SAC-compatible equivariant policy for PandaReach / FetchReach.

    Exposes the same interface as MlpPolicy so it can be dropped directly
    into the existing SAC training loop:

        policy.pi(obs)              → (action, logp)
        policy.q1(obs, act)         → Q-value tensor  (B,)
        policy.q2(obs, act)         → Q-value tensor  (B,)
        policy.act(obs, det=False)  → numpy action     (act_dim,)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_scalars: int = 16,
        n_vectors: int = 8,
        clip_action: bool = True,
    ) -> None:
        super().__init__()
        self.clip_action = clip_action
        self.act_limit = float(action_space.high[0])  # type: ignore[union-attr]

        extractor = EquivariantReachExtractor(observation_space)

        self.pi = EquivariantSACActor(extractor, self.act_limit, n_scalars, n_vectors)
        self.q1 = InvariantMLPQFunction(extractor)
        self.q2 = InvariantMLPQFunction(extractor)

    def act(
        self,
        obs: Union[NDArray, dict],
        deterministic: bool = False,
    ) -> NDArray:
        with torch.no_grad():
            a, _ = self.pi(
                unsqueeze_observation(obs),
                deterministic=deterministic,
                with_logprob=False,
            )
            a = a.squeeze(0).cpu().numpy()

        if self.clip_action:
            a = np.clip(a, -self.act_limit, self.act_limit)

        return a
