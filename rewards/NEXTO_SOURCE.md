# Nexto-era reward source and CARL adaptation

`shaping.py` is a vectorized PyTorch adaptation of the Necto project's
**Nexto-era** reward at commit
`5c56a79910decdfe1298c7cb234d08101a0bf70a` (2022-03-31):
https://github.com/Rolv-Arild/Necto/blob/5c56a79910decdfe1298c7cb234d08101a0bf70a/training/reward.py

Attribution: Rolv-Arild and the Necto contributors. The project currently
publishes its source under **CC BY-NC-SA 4.0**:
https://github.com/Rolv-Arild/Necto/blob/2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca/LICENSE
License text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode .
This adaptation is likewise CC BY-NC-SA 4.0, provided without warranties;
noncommercial and share-alike conditions apply. It is not endorsed by Nexto.

The pin precedes the Necto-v3/Tecko reward experiments in the current master
branch. It is a reproducible Nexto-era baseline, not a claim to know the exact
reward history of every distributed Nexto training checkpoint.

Preserved computations and default weights:

- Goal reward 10; goal speed and conceding-distance bonuses 2.5.
- State goal-distance quality 10; player proximity/alignment qualities 0.5/0.5.
  Use signed, **undiscounted** differences; the state quality has the reference's
  extra factor of 1/2 before team distribution.
- Touch height 1, doubled when airborne; ball acceleration on touches 0.25.
- Square-root boost gain 1 and boost loss 0.9.
- On-ground penalty 0.005 per policy decision, including wall contact, as upstream.
- Demo weight 5, split equally between attacker and victim before team mixing.
- Team spirit 0.6; subtract opposing team mean with weight 1.
- Goal events **replace** ordinary shaping. Preserve the released code's use of
  `goal_dist_bonus_w` for the speed bonus too, despite its separately named,
  unused `goal_speed_bonus_w` argument.

CARL adaptations:

- Batched torch tensors, no CPU/NumPy conversion on the reward path.
- Boost converted from CARL's 0..100 to the reference's 0..1.
- Evaluate qualities directly from each true previous/current transition,
  including terminal transitions; never compare against the next reset state.
- Use `score_delta` for goals. Goal speed and ball location come from the
  **previous** state, matching upstream and avoiding CARL's post-goal ball reset.
- CARL's reward context exposes demo victims, not attacker IDs. Distribute the
  positive half-demo credit equally among opposing players. This matches the
  reference in 1v1; individual attacker attribution differs in 2v2/3v3.
- Keep upstream per-decision weights and reference goal/ceiling geometry.
  CARL's physics, actions and chosen frameskip still differ from Nexto's original
  simulator/training protocol; this does not reproduce its policy performance.

RewardPlus wall, air-dribble, flip-reset and acceleration bonuses are available
separately with `--reward-plus`, not silently combined with Nexto. Nexto remains
the default. PPO uses distinct reward-objective IDs, so checkpoints cannot
silently switch between the two reward definitions.
