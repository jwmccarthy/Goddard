# Manuscript revision: online RL skill acquisition

The source for `/home/bento/rl.pdf` is not available here. These passages are
intended to replace the reward-free low-level claims in its abstract, Sections
1, 5, Algorithms 1–2, and the corresponding discussion in Section B. Sections
3–4 (state-only representation and segment inference) remain applicable.

## Abstract / Introduction

The low-level behavior-conditioned policy learns online in the downstream
environment. Expert state-only trajectories supply requested behavior
embeddings, not action labels. An EMA of the representation encoder evaluates
each executed behavior prefix against its requested embedding, yielding a dense
tracking reward with a differential progress term. On-policy reinforcement
learning optimizes the policy and a behavior-conditioned critic against this
reward. Task rewards remain separate: they fit the high-level latent-plan value
model rather than defining the low-level tracking objective.

## Replacement for Section 5.2: Reward and on-policy optimization

For an expert segment `ŝ[a:b]`, request the entity-specific target embedding
`ẑ_j = μ_{φ̄,j}(ŝ[a:b])`. A subsequent plan sample may also provide a request
from the expert-supported sequence prior. On the resulting downstream rollout,
define the realized prefix alignment, after at least one transition, by

```text
c_{t,j} = μ_{φ̄,j}(s[a:t])ᵀ ẑ_j,     a < t ≤ b,   c_{t,j} ∈ [-1, 1].
```

The encoder reads the *whole prefix* from the start of this request, not just
the most recent state pair. The dense per-transition reward for action `a_t`
leading to state `s_{t+1}` is

```text
Δc_{t+1,j} = c_{t+1,j} - c_{t,j}      (zero for the first transition),
r^track_{t,j} = λ_match c_{t+1,j} + λ_diff Δc_{t+1,j}.
```

The first term rewards realizing the requested behavior throughout execution;
the differential term supplies immediate positive or negative feedback for
moving closer to or farther from it. Both coefficients are nonnegative and
configurable. At a request change, the prefix and previous alignment restart;
at an episode end, scoring uses the final simulator state, not the autoreset
observation. Only actions of controlled entities are optimized.

The shared behavior-conditioned policy `π_ψ(a | o, ẑ_j, τ)` is updated from the
current on-policy batch by clipped PPO with generalized advantage estimation
on `r^track`, a separate learned critic `V^track(o, ẑ_j, τ)`, and entropy
regularization. Request termination or episode termination cuts the skill
return. PPO sees every active transition in the joint rollout; its minibatch
updates therefore scale with the number of collected actor-steps. This is
online reward-based skill learning, not behavioral cloning of the policy's
previous actions. Environment goal rewards do **not** enter this skill
objective; Section 6.2 continues to fit `J_ϑ` on task returns to select plans.

The reward encoder is held fixed during rollout and PPO updates. Only after
those updates may representation learning on collected state windows update
the online encoder and its EMA. No gradient flows through the reward encoder,
the decoder, or the demonstrated target embedding to the policy. Rollout
segments are still inferred in hindsight for the duration hazard and sequence
prior, but do not provide supervised policy action targets.

Log the requested-versus-realized cosine alignment, alignment progress,
tracking reward, PPO policy and critic losses, and true task returns. A small
critic error or reconstruction loss alone cannot establish skill realizability.

## Replacement for Algorithm 2, fast loop

1. Draw an expert segment embedding, achieved embedding, or supported prior
   sample as the requested behavior; initialize the scene and controlled set.
2. Roll out the behavior-conditioned policy; evaluate realized prefix
   embeddings with the frozen EMA and the dense tracking reward above.
3. Update the low-level policy and tracking critic by on-policy PPO/GAE over
   every active controlled transition in the rollout.
4. Segment realized trajectories in hindsight to supervise the duration
   hazard, and add state windows to the representation corpus.

The slow representation and slower sequence-prior refit remain as in the
original algorithm. Replace statements elsewhere that skill acquisition uses
"no reward" or "iterated supervised learning" with the tracking-reward
formulation above; the absence of *expert action labels* still holds.
