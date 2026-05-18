# Notes

# Equivariant critic

## Diagnosis: Three linked problems

### 1. Q-function instability (the root cause)

The `loss_q` tells the story most starkly:

| Range | MLP loss_q | Equivariant loss_q |
|---|---|---|
| Step 1K | 2.07 | 1.39 |
| Step 25K | 0.13 | **6.27** |
| Step 38K | 0.07 | **16.86** |
| Step 46K (peak) | — | **49.39** |
| Step 63K | 0.026 | 5.11 |
| Step 76K | 0.024 | 0.09 |

The equivariant Q-function (`TPBlock`-based) catastrophically diverges between steps 13K–63K. The MLP Q-function peaks at 2.07 on step 1 and is below 0.2 by step 13K. The `FullyConnectedTensorProduct` has many more gradient paths (all pairwise CG products) than a linear layer — this makes early-training optimization harder and more prone to runaway gradients before the Clebsch-Gordan weights have settled.

### 2. Alpha spiral (the symptom)

Because the Q-function is giving garbage signal, the actor learns nothing useful, so the policy stays random longer than expected. SAC's auto-alpha interprets low-quality logp values as "policy needs more entropy" and responds by **raising alpha from 0.07 → 0.58 between steps 5K–38K**. The MLP's alpha is below 0.01 by step 13K and never rises again. High alpha amplifies the unstable Q-targets (`backup = r + γ*(Q_targ − α*logp)`), making Q-instability even worse — a feedback loop.

### 3. Persistent underperformance after recovery

Even once the system stabilizes (~63K), the equivariant settles at `test_ep_return` ≈ −2.0 to −2.9, while MLP reaches −1.2 to −2.0. The `loss_pi` also remains consistently negative for equivariant (−0.5 to −0.7) vs near-zero positive for MLP (~+0.15). A persistently negative `loss_pi = α·logp − Q` means the Q-function is dominating very strongly, driving the actor greedy. This is a symptom of the higher residual alpha (0.004 vs 0.0003) and the isotropic Gaussian having a different entropy profile than the element-wise tanh the target entropy of −4 was calibrated for.

---

## The fix

The Q-function is doing the heavy lifting for the equivariant training and it doesn't benefit from geometric structure — it just needs to estimate a scalar. The cleanest solution is to **keep the equivariant actor, replace the TP Q-function with a standard MLP Q-function**, and feed it a set of handcrafted invariant features instead:

```parviflora/parviflora/policies/equivariant_policy.py#L1-10
# Instead of TPBlock Q-function, use an MLP over invariant scalars:
#
#   ‖error_vec‖          distance to goal
#   ‖ee_vel‖             speed
#   error_vec · ee_vel   alignment of velocity with goal
#   error_vec · act_xyz  alignment of action with goal     ← key for Reach
#   ‖act_xyz‖            action magnitude
#   + the 4 gripper scalars
#
# These are all SO(3)-invariant, so the Q-function is invariant by
# construction with zero extra complexity.
```

This sidesteps the TP instability entirely while keeping the actor's equivariance, which is where all the geometric benefit lives anyway. The equivariant mean is what enables the policy to generalise across rotations — the critic just needs stable values.
