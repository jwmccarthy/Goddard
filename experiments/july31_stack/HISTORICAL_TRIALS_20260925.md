# July 31-era training trials (September 25, 2026)

Every candidate uses pinned Goddard `a817186`, CARL `965cba1`, JARL `0c6ed3a`,
seed zero, 8,192 or 16,384 simulations, frameskip eight, and 32-step rollouts.
No scripted or Nexto kickoff controllers are used: both sides act with their
learned policies. A committed `kickoff` reward weight is only a training signal.
Only one trainer runs at a time. The original 10-billion-step target is kept
even for short trials, so learning-rate, entropy, and gamma schedules do not
anneal early. Short trials are stopped after a comparable saved policy snapshot;
their logs and model weights remain backed up on `/workspace`.

`committed_reward_ppo.py --reward-commit <commit>` loads the **entire**
`rewards.py` from the exact Goddard commit; it keeps the pinned trainer and
replay corpus unchanged. Committed alternatives:

| Reward commit (EDT) | Direct touch | Kickoff | Other historical difference |
| --- | ---: | ---: | --- |
| `de8975f` (Jul 31, 10:22) | 0.02 | 0.1 | Earlier goal/boost/reward mechanics; goal weight 5 |
| `f7cab81` (Jul 31, 10:32) | 0 | 0.1 | Nexto scoring and 14 dense occupancy incentives |
| `e53f73e` (Jul 31, 15:00) | 0 | 0 | Removes the 14 occupancy incentives |
| `a817186` (Jul 31, 22:19) | 0 | 0 | Adds goal-time and aerial-sequence incentives |

For `de8975f`, use `--goal-score-weight 5 --goal-score-weight-end 5` as in
its dated trainer: otherwise the later trainer's goal schedule would overwrite
the historical reward's goal weight with 10 at the first step. Each other
reward revision used a goal weight of 10. The original archived actor has no
training manifest; these are **comparisons**, not claims about its provenance.
The user clarified that the polished archived actor had **upwards of 20B
training steps**. Its direct-match dominance over 0.5–3B-step policies
measures the **remaining age-mismatched skill gap**, not whether a reward
revision succeeded or would catch up with enough training. Judge competing
revisions primarily at **matched steps**, including direct policy-vs-policy
games and completed-game timeout trajectories. The 3B point checks for
continued mechanics acquisition, not parity with the 20B+ export.

Single-factor follow-ups use CLI options already present in the July 31
trainer. Contemporaneous commits support a 16-second versus 30-second
no-touch timeout (`4e88f2e`), a 14,400- versus 36,000-tick match limit
(`325fd30`), and the replay sampler's existing 70% mix. Candidate tests also
compare PPO epochs, learning rate, and replay-reset probability without
changing frameskip or rollout length. An earlier July 30 commit `a103309`
capped kickoff-like states in reset corpora; archived older dataset generations
can test this feature separately after reward screening.
The dated `f7cab81:ppo.py` defaults were **16-second no-touch timeout and
14,400-tick maximum**, whereas the later pinned `a817186:ppo.py` defaults
used 30 seconds and 36,000 ticks. A historically matched `f7cab81` trial
with both earlier defaults is a high-priority follow-up to the reward-only
trial; changing them together tests that dated configuration, not an
isolated timeout effect.

Evaluate policy snapshots on identical held-out seeds and both sides against
legal random opponents, reporting touch-game fraction, goals, and no-goal
terminations. Replay-mixed stochastic self-play is evaluated separately.
The stopped `f7cab81` reward-only run appears in the existing TensorBoard
server on port **6006** as `july31-reward-f7cab81-8192x32-20260925`.
In addition to its original `PPO/*` and `Episode/*` training scalars,
`publish_heldout_tensorboard.py` added the full-length seed-9210 results
at the original checkpoint steps under `Heldout/frozen_kickoff_seed9210/*`
and `Heldout/mixed_seed9210/{all,replay,kickoff}/*`. The 30 frozen and
31 mixed evaluations include the 3.015B final-policy checks; completed-game
timeout fractions and censored counts are separate charts. The underlying
JSONL files and one-off evaluations remain the source data on `/workspace`.
For live play, `watch_july31_checkpoint.py` serves the pinned simulator and
the existing 3D checkpoint-viewer frontend on **port 8788**. It starts with
the final `f7cab81` actor (`policy_003014580461.pt`) on both sides;
the viewer's selectors also offer 1.343B and 1.547B policies. The curated
weights are in `/workspace/july31-recreation-20260924/viewer/f7cab81/`;
conversion for the viewer only renames parameter keys and leaves the
original checkpoints unchanged. Forward port 8788 to open
`http://localhost:8788`, then use **Kickoff** for a fresh opening or
**Reset** for another replay-start. The viewer simulation uses the pinned
July 31 CARL/JARL stack, seed 9210, frameskip eight, and the original
reset corpus; it displays deterministic (argmax) actions by default.
The original `a817186` 8,192×32 run is retained as the control, observed
through the roughly 3-billion-step point before sequential trials.

The control was stopped after its 3.005B policy snapshot: 37/256 fixed-seed
kickoff games with a touch, four goals, and 249/256 non-goal terminations.
Its best observed snapshot was 1.864B with 82/256 touches, but that skill was
not sustained. Around 500–600M, neither kickoff nor replay-mixed timeouts had
dropped substantially. A promising trial must show **sustained** contact,
goal-scoring, and lower non-goal termination rates through at least 500M;
an early touch-rate bump alone is insufficient evidence of the reference run.
At a matched 116.6M snapshot, the control in 512 replay-mixed games had
only 138 with any observed post-reset touch, 68 goals (44 with no observed
post-reset touch), and 444 non-goal
terminations. Its zero-step policy is byte-identical to the `f7cab81`
trial's (SHA-256 `493f566440939bffc50e6d24fed641d923dc50a1216b9f204a471cfded2da891`),
so the compared runs began from exactly the same actor weights.

### First reward trial: `f7cab81`

Started September 25 at 04:21 UTC with seed zero and the exact committed
`rewards.py` injected into pinned PPO. Training, mirrored snapshots, and
fixed-seed evaluators run under
`july31-reward-f7cab81-8192x32-20260925`. The frozen kickoff evaluator uses
256 games versus legal random actions; the separate replay-mixed evaluator
uses 512 first episodes with the same policy controlling both cars. At the
first **30.7M** snapshot: 145/256 kickoff games with our touch, 18 goals for,
and 217/256 non-goal terminations. The baseline's nearest snapshot at 39.2M
had 10/256 touch games, three goals, and 252/256 non-goal terminations.
This is an initial observation, not a passed 500M-step gate.

At 61.1M the same fixed kickoff test had 212/256 touch games, 24 goals,
and 161/256 non-goal terminations; at 99.3M it had 238/256 touch games,
32 goals, and 89/256 non-goal terminations. A separate 512-game replay-mixed
stochastic self-play evaluation at 114.5M yielded 462 games with at least
one car touching, 234 goals (33 without any observed touch after reset),
and 108 non-goal terminations. The baseline 3.005B policy evaluated with the
same replay-mixed seed had 212 touch games, 77 goals (38 without a touch),
and 435 non-goal terminations. Replay resets can begin with a ball already
headed towards a goal; goals without an observed post-reset touch should not
be counted as proven touch-and-score plays. A few kickoff-start goals also
have no observed touch in the transition-state samples; with eight simulator
ticks per action, this diagnostic may miss a transient contact, so these
counts do **not** prove a genuinely untouched kickoff goal.

**Censoring matters:** the initial fixed kickoff evaluator stops after 900
actions: 11/256 games were still alive at 30.7M, 124/256 at 99.3M, and
160/256 at 190.5M. The initial replay-mixed evaluator stops after 1,500
actions, with 170/512 games still alive at 114.5M. Dividing observed
non-goal endings by the original game count understates the ultimate timeout
rate as longer-lived games become common. New evaluations use a 5,000-action
horizon and report completed, timed out, goal, and censored counts separately.
At 190.5M in 512 replay-mixed 5,000-action games, 497 completed: 379 goals,
118 non-goal endings, and 15 censored. Among the completed games the non-goal
fraction was 118/497 (23.7%), compared with 444/512 (86.7%) for the control
at a matched 116.6M checkpoint; the latter all completed in the 1,500-action
evaluation.
At 213.2M with a 5,000-action limit, the frozen 256 kickoff games against
legal random opponents had 254 completed, 253 games with our ball contact,
137 goals for, 27 against, 90 non-goal endings, and two censored. The
same-policy replay-mixed 512-game evaluation had 499 completed, 387 goals
(25 with no observed post-reset touch), 112 non-goal endings, and 13
censored. This is a substantial improvement over the control without
relying on the shorter evaluator's censoring artifact.
On the identical full-length kickoff setup, the August 1 reference actor
completed all 256 games with **256 wins, zero losses, zero timeouts**, and
contact in every game. Its 40,889 aggregate action steps versus 319,600
for `f7cab81` at 213.2M show that matching touch coverage is not yet
matching the reference's rapid scoring. Both evaluations have the same
initial observation hash (`0b21ccea3706443e`). The archived viewer export
swapped the input/output JARL module names; the evaluator swaps the parameter
names back before loading the actor into the pinned policy architecture.
At 315.1M the same 5,000-action kickoff test completed all 256 games with
255 touch games, **231 wins, 18 losses, and seven timeouts**. The replay-mixed
512-game test completed 504: 408 goals (24 with no post-reset touch), 96
non-goal endings, and eight censored. This strengthens the sustained early
signal, but the reference still wins all 256 kickoff games and scores more
quickly; it is too early to infer that this reward revision produced it.
The archived actor on the same 512 replay-mixed 5,000-action games completed
all 512, with 485 goals (six without a post-reset touch) and 27 timeouts;
all 160 kickoff starts ended in goals. This provides a second, more stringent
reference trajectory than merely making contact against random opponents.
In a separate balanced 512-game direct matchup (seed 9210, 5,000-action
limit, 70% replay starts), the archived actor scored **500 goals** against
the 315.1M `f7cab81` policy's **five**, with seven non-goal endings. All 160
kickoff-start matchups were goals for the archive. Early progress against a
legal random opponent must not be mistaken for matching archived skill.
At 419.1M the 5,000-action kickoff evaluator again completed 256/256 games:
255 touch games, 231 wins, seven losses, and 18 timeouts. The same-seed
replay-mixed test completed 510/512: 433 goals (16 without a post-reset
touch) and 77 non-goal endings. Its completed-game timeout rate has fallen
from 22.4% at 213.2M, to 19.0% at 315.1M, to 15.1% at 419.1M; the
archive is at 5.3% on this test.
At **522.8M**, the 5,000-action kickoff evaluator completed all 256 games:
256 games with our touch, **249 wins, four losses, three timeouts**, and
149,948 aggregate action steps. The same-seed replay-mixed test completed
511/512 games with 453 goals (17 without a post-reset touch), 58 non-goal
endings, and one censored. Its completed-game non-goal rate is **11.4%**
versus the baseline's 86.7% at ~116M and the archived actor's 5.3%; 159
of 160 kickoff-start self-play games ended in goals, with the last game
censored. This reward-only run **passes the user's 500M early-learning gate**
through sustained goals/contact increases and a substantial decline in
non-goal terminations. It remains uninterrupted to evaluate the later
~3B mechanics/competitive-learning milestone. The archive still scored 256/256
kickoff wins in only 40,889 aggregate steps on this setup.
An independent 256-game frozen kickoff check of the 522.8M policy on seed
9211 also found 249 wins, three losses, four timeouts, and 255 contact
games; the improvement is not limited to the fixed watcher seed 9210.
At 522.8M the same 512-game balanced head-to-head against the archived
actor remained lopsided: **497 archived goals to five from `f7cab81`**,
with ten non-goal endings (all 160 kickoff starts were archived goals).
Passing the early-learning gate is therefore not the same as reproducing
the archived actor's competitive ability.
At 626.0M the trial reached 253/256 kickoff wins, three losses, zero
timeouts, and ball contact in all 256 games. All 512 replay-mixed first
episodes finished: 472 goals and 40 timeouts, approaching the archive's
485 goals and 27 timeouts on this measure. Head-to-head against the archive
must still be checked before calling it comparable.
For diagnostic context, the 500M TensorBoard sample of `PPO/approx_kl` is
roughly 0.0033 in the `f7cab81` run versus roughly 0.0806 in the control;
policy entropy is about 3.78 versus 4.27. These are measured correlates,
not proof that the reward revision alone caused the control's large KL.
At 729.0M the frozen kickoff check still showed 253/256 wins and zero
timeouts, now in 90,307 total actions versus 144,709 at 626M and 40,889
for the archive. Replay-mixed self-play completed all 512 with 477 goals and
35 timeouts. Direct balanced 512-game matchups: the 729M reward trial beat
the **best observed control** (1.864B) by 467 goals to 20, but lost to
the archived actor 496 goals to five. Better early general play does not
yet reproduce the archive's learned competitive advantage.
At 830.4M both same-policy replay-mixed batches completed all 512 games:
the reward trial had **483 goals and 29 timeouts**, versus **485 and 27**
for the archived actor. The reward trial won 253/256 frozen kickoffs in
68,905 aggregate steps, down from 90,307 at 729M but still slower than
the archive's 40,889. Similar self-play first-episode goal counts alone do
not establish comparable head-to-head play.
At 932.7M the 5,000-action kickoff evaluator saw 255/256 wins, one loss,
zero timeouts, and 56,342 aggregate action steps. Replay-mixed self-play
finished all 512 with 492 goals and 20 timeouts, exceeding the archive's
485/27 on **that particular metric**. Yet a direct 512-game matchup still
favored the archive 497 goals to seven, including all 160 kickoff starts.
The matched opponent test remains necessary to assess reference-level play.
At 1.040B the kickoff evaluation remained 255 wins, one loss, no timeouts,
and 256 contact games. Replay-mixed same-policy play ended all 512 games
with 486 goals and 26 timeouts. These aggregate rates remain similar to
the archive, but the last direct head-to-head at 933M was still 497–7
in favor of the archive. Training remains uninterrupted toward ~3B.
At 1.242B the trial itself went 256 wins / zero losses / zero timeouts in
45,391 aggregate actions against the fixed legal-random kickoff opponents;
the reference's 256 wins took 40,889 actions. Replay-mixed self-play had
480 goals and 32 timeouts. The archived actor still beat this trial **494–9**
in 512 direct games, taking all 160 kickoff starts. The same frozen kickoff
evaluator recorded about 19.8 of our contacts per 1,000 active actions for
the trial, versus 51.8 for the archive: a contact-frequency/style gap,
not by itself proof of a particular aerial or flick maneuver.
At 1.343B the reward trial won all 256 fixed random-opponent kickoffs in
36,907 aggregate actions, slightly faster than the archive's 40,889 in the
same setup; its replay-mixed self-play sample had 484 goals and 28 timeouts.
Direct balanced play against the archive was still **491–10** over 512
games, but the trial won **three of the 160 kickoff-start games** for the
first time (versus zero at 933M and 1.242B). This is early evidence of
improvement in the stringent benchmark, not parity.
At 1.547B the reward trial was still 256/256 wins on the fixed kickoff set,
with 46,920 aggregate actions; replay-mixed self-play had 486 goals and
26 timeouts. To avoid a saturated weak-opponent metric, a balanced direct
512-game matchup between its 1.343B and 1.547B versions found the **later
policy ahead 270 goals to 211** (31 non-goal endings); the replay-start
split was 182–139 and kickoff split 88–72 in favor of the later policy.
This points to some continued within-run competitive progress despite
the remaining large gap to the archived actor.
At 1.852B the fixed kickoff batch still went 256–0, but took 85,387
aggregate actions; replay-mixed self-play had 471 goals and 41 timeouts,
slightly worse than around 1.3–1.6B. A direct matchup of the 1.547B and
1.852B versions finished **242–241** (29 non-goal endings), so there is
not yet evidence of a large competitive collapse. At ~1.865B the logged
PPO approximate KL was ~0.021 versus ~0.009 at 1.5B, with policy entropy
~1.83 versus ~2.71; watch these signals for instability rather than
attributing the score variation to them without an ablation.
At 2.054B the reward trial won all 256 frozen kickoffs in 61,777 actions
and completed all 512 replay-mixed self-play episodes with 474 goals and
38 timeouts. Against the archive in 512 balanced games it still lost
**493 goals to ten**; only one of 160 kickoff-start games was its win.
Although its early-contact and completed-game timeout rates remain much
better than the control's, its direct competitive improvement has largely
flattened since ~1.3B. Preserve the stronger intermediate snapshots and
continue to ~3B before the historically matched timeout/match-limit trial.
The gap against the archive is not a one-seed artifact: at 2.362B, a
second 512-game balanced head-to-head on seed 9211 finished **492–12**
for the archive (three trial wins in 159 kickoff starts). The same policy
won all 256 fixed seed-9210 kickoffs against legal random opponents and
scored 480 goals with 32 timeouts in 512 replay-mixed self-play games.
At 2.463B another direct 512-game comparison with the 1.547B version
was nearly tied, **244–237 for the later policy** (31 non-goal endings).
The later same-policy mixed check was 469 goals and 43 timeouts, while the
fixed random-opponent kickoff check was 255 wins and one loss. There is no
clear evidence of a large post-1.5B competitive gain or collapse yet.
At 2.665B the 256-game frozen kickoff batch was still 253 wins, three
losses, zero timeouts, with contact in all 256 games. Replay-mixed
self-play was 467 goals and 45 timeouts versus the best earlier batch's
492/20 at 933M. PPO entropy had fallen to ~0.28 near 2.68B, but
approximate KL was only ~0.009; these observations warrant monitoring,
not a causal claim. The 1.343B and 1.547B policies have matching SHA-256
on `/dev/shm` and their persistent `/workspace` backups.

The reward-only trial was stopped at policy **3,014,580,461** steps; its
policy and final `training_latest.pt` were verified byte-identical in the
RAM-backed and persistent checkpoint directories. Full-length 256-game
kickoffs: 256 touch games, 254 wins, one loss, one timeout, 98,248 aggregate
actions (archive: 256 wins, zero losses/timeouts, 40,889 actions). Full-length
replay-mixed self-play: 465 goals, 46 non-goal endings among 511 completed
games, one censored (archive: 485 goals, 27 timeouts, zero censored).
Balanced direct matchup against the archive: **494–10 archived goals** in
512 games; the reward trial won only two of 160 kickoff starts. Directly
against the 1.547B trial version, the 3.015B version trailed **226–249**
(37 non-goal endings). These results rule out treating its early contact
and timeout success as proof of already matching the **20B+** archive,
but do not imply that longer training of this promising reward will fail.
Its 500M gate passed decisively; relative to same-age alternatives it is
the strongest tested July 31 reward configuration so far.

The archived export's **action-distribution entropy** can be measured even
though its historical training entropy cannot be recovered. On seed-9212
archived-policy stochastic self-play, `measure_policy_entropy.py` evaluated
the archived actor and the `f7cab81` 522.8M and 3.015B actors on **the same
first-episode observations**, carrying each actor's own recurrent state.
There were 256 games (512 actor observations initially), with 70% replay
resets and 30% kickoffs; only the archived actor controlled the simulator.
Entropy is the sum of seven **masked categorical-action entropies, in nats**,
not the entropy coefficient or the entropy of one sampled action:

| Steps into held-out first episode | Alive actors | Archived August 1 | `f7cab81` 522.8M | `f7cab81` 3.015B |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 1.722 | 3.013 | 0.716 |
| 15 | 506 | 2.960 | 3.332 | 0.206 |
| 45 | 482 | 2.977 | 3.515 | 0.192 |
| 90 | 422 | 2.480 | 3.239 | 0.209 |
| 180 | 336 | 2.581 | 3.175 | 0.229 |

The archived actor has appreciably more action entropy than the 3.015B
`f7cab81` actor on this fixed observation history, despite its much greater
skill and age. These states are generated by the archived policy, so the
younger actors' later-step entropy is **off-policy** and should not be read
as their own rollout or training entropy. Kickoff/replay splits, per-action
factor entropies, observation SHA-256 prefix `04fb79e570006c5e`, and all
sample counts are saved in
`/workspace/july31-recreation-20260924/archived_action_entropy_9212.json`.

### Historically matched `f7cab81` episode settings

Next sequential trial retains the exact committed `f7cab81` reward, seed
zero, 8,192 simulations, 32-step rollouts, frameskip eight, and the same
reset corpus, while restoring **both** defaults from `f7cab81:ppo.py`:
`--no-touch-timeout 16 --max-ticks 14400`. All held-out evaluations retain
the **common** 30-second/36,000-tick, 5,000-action setup for apples-to-apples
policy comparisons. There are no scripted or Nexto kickoff actions.
Started September 25 at 11:07 UTC as
`july31-f7-era16s14400-8192x32-20260925` (trainer PID 51418), with a
two-minute persistent-output mirror and 100M-step full-length frozen-kickoff
and replay-mixed self-play evaluators. Its zero-step actor SHA-256 is again
`493f566440939bffc50e6d24fed641d923dc50a1216b9f204a471cfded2da891`,
identical to both preceding seed-zero 8,192×32 actors.
At 107.3M in the **common evaluation environment**, the earlier-episode
trial completed all 512 replay-mixed games with 156 goals and 356 timeouts;
its 256 fixed legal-random-opponent kickoffs completed with 240 contact
games, 49 wins, ten losses, and 197 timeouts. The reward-only trial at a
matched 106.9M snapshot had 301 goals, 201 timeouts, ten censored in
replay-mixed play; its 256 fixed kickoffs had 246 contact games, 65 wins,
27 losses, and 164 timeouts. The shortened historical match limits make
this trial substantially **slower to acquire scoring ability at ~100M**,
but the user-specified judgment point is 500M, so the trial continues.
At 214.3M the earlier-episode trial completed all 256 frozen kickoffs:
254 contact games, 77 wins, 21 losses, 158 timeouts. Replay-mixed
self-play completed 502/512, with 341 goals, 161 timeouts, and ten
censored. The reward-only trial at 213.2M (same eval setup) recorded
137 wins / 27 losses / 90 timeouts against random opponents and
387 goals / 112 timeouts / 13 censored in replay-mixed self-play.
The dated episode settings still trail, though their own completed-game
timeouts are falling substantially between 107M and 214M.
At 320.5M the dated-settings trial finished 254/256 fixed kickoffs,
with 255 contact games, 162 wins, 28 losses, 64 timeouts, and two censored.
Replay-mixed self-play had 390 goals, 107 timeouts, and 15 censored.
The reward-only trial around 315M had 231/256 fixed kickoff wins and
408 goals, 96 timeouts, eight censored in self-play. The mixed-results
gap is narrowing, but the goal-for rate against the weak random opponent
is still much lower with the shorter historical training episodes.
In direct balanced 512-game play at matched ~315M/~320M checkpoints,
the longer-episode reward-only policy scored **293 goals** against the
dated-settings policy's **136**, with 76 non-goal endings and seven
censored. Both used identical zero-step actor weights, the same seed,
rewards, architecture, and replay corpus; the two dated episode-limit
changes together appear detrimental to competitive play at this stage.
At 421.1M the dated-settings trial completed 251/256 fixed kickoffs with
256 contact games, 203 wins, 12 losses, 36 timeouts, and five censored.
Replay-mixed self-play had 421 goals, 79 timeouts, 12 censored. The
reward-only 419.1M snapshot had 231 kickoff wins, seven losses, 18
timeouts, and 433 goals / 77 timeouts / two censored in self-play.
The gap on replay-mixed goals is now smaller, but fixed kickoff wins
and unfinished replay-mixed games still favor longer training episodes.
At **521.2M** the dated-settings trial finished all 256 fixed kickoffs
with 256 contact games, 242 wins, ten losses, and four timeouts. Its
512-game replay-mixed self-play test completed all games with 446 goals
and 66 timeouts. The reward-only trial's matched 522.8M checkpoint had
249 wins / four losses / three timeouts against the same weak opponent
and 453 goals / 58 timeouts / one censored in replay-mixed play.
Crucially, in direct balanced 512-game play, the reward-only policy beat
the dated-settings policy **309–144** (59 non-goal endings), confirming
a substantial competitive advantage despite closer aggregate goal rates.
The archived reference beat the dated-settings policy **500–6**, winning
all 160 kickoff starts. The dated-settings trainer was therefore stopped
after its 500M gate; its weights and optimizer checkpoint are mirrored
to `/workspace` for independent inspection. Both historical episode-limit
changes together are not an improvement for this tested seed/configuration.

### Earlier committed reward `de8975f`

Next, isolate the exact July 31 10:22 reward revision (direct `ball_touch`
weight 0.02, `kickoff` 0.1, goal weight five). Keep the later 30-second
no-touch/36,000-tick episode settings fixed because the matched dated
16-second/14,400-tick settings underperformed with the otherwise identical
`f7cab81` reward. Pass **both** `--goal-score-weight 5` and
`--goal-score-weight-end 5`; the later trainer's default of ten would
otherwise overwrite the earlier revision's weight. The experiment is
reward-isolating rather than a claim that these episode settings were the
historical `de8975f` command. Evaluate through 500M on the same held-out
full-length and direct-match tests before promoting it further.
Started September 25 at 12:22 UTC as
`july31-reward-de8975f-8192x32-20260925` (trainer PID 52715). The source
revision resolved to `de8975ff55277f13c0de39491aa08f1f69a214c9`;
its seed-zero initial actor is byte-for-byte identical to the control and
both `f7cab81` trials (SHA-256 `493f566440939bffc50e6d24fed641d923dc50a1216b9f204a471cfded2da891`).
Mirroring and fixed-seed 100M-step full-length evaluators are active.
At 100.1M the 5,000-action fixed kickoff test finished all 256 games with
239 contact games, 48 wins, 18 losses, and 190 timeouts. Replay-mixed
self-play completed 507/512 games with 249 goals, 258 timeouts, and five
censored. The `f7cab81` reward-only policy near 106.9M had 65 kickoff
wins, 27 losses, 164 timeouts and 301 replay-mixed goals / 201 timeouts
/ ten censored on this same held-out setup. The earlier direct-touch
reward does not improve early scoring, but the 500M gate is still pending.
At 206.4M, `de8975f` had 134 kickoff wins, 33 losses, 86 timeouts,
three censored (the reward-only `f7cab81` trial at 213.2M had 137 wins,
27 losses, 90 timeouts, two censored). In replay-mixed self-play `de8975f`
had 371 goals, 117 timeouts, 24 censored versus 387 goals, 112
timeouts, 13 censored for `f7cab81` near 213M. The earlier reward has
largely closed the early scoring gap; direct head-to-head at later
matched checkpoints is needed to compare skills.
In a balanced 512-game ~213M versus ~206M direct match, `f7cab81`
led `de8975f` **225–169** (105 non-goal endings, 13 censored), including
90–59 on 160 kickoff starts. The earlier reward has not yet improved
competitive play, though this is a single seed and early checkpoint.
At 307.8M, `de8975f` had 193 fixed kickoff wins, 14 losses, 43
timeouts, six censored; replay-mixed self-play had 399 goals,
96 timeouts, 17 censored. The `f7cab81` reward-only checkpoint at
315.1M had 231 kickoff wins / 18 losses / seven timeouts and
408 mixed goals / 96 timeouts / eight censored. Self-play aggregate
scoring is close by ~300M, but kickoff conversion still favors
`f7cab81`.
At 411.3M, `de8975f` completed all 256 frozen kickoffs: 236 wins,
four losses, 16 timeouts, and 256 contact games. Replay-mixed self-play
completed 507/512 with 434 goals, 73 timeouts, five censored. The
`f7cab81` reward-only run at 419.1M had 231 kickoff wins, seven losses,
18 timeouts, and 433 mixed goals / 77 timeouts / two censored. These
small fixed-seed aggregate differences do not settle which policy is
competitively stronger; direct matched play at 500M remains the gate.
At **515.0M** the `de8975f` trial completed all 256 weak-opponent
kickoffs with 256 contact games, 252 wins, three losses, one timeout,
in 179,058 aggregate actions. Replay-mixed self-play completed
510/512: 451 goals, 59 timeouts, two censored. Versus the longer-episode
`f7cab81` reward-only trial at 522.8M, the broad results (249 wins and
453 goals / 58 timeouts / one censored) are close. Balanced direct
512-game play slightly favored `de8975f` **225–216** (69 non-goal
endings, two censored); this difference is small enough to be sampling
noise. The archive beat `de8975f` **499–4** in direct play, taking all
160 kickoff starts. The earlier committed reward meets the user's
500M criterion via increased goals and large completed-game timeout
reductions; it is therefore kept **running uninterrupted toward ~3B**
to test whether this earlier **whole reward revision** produces closer
archived ball-control and competitive behavior later. This comparison
cannot attribute any difference solely to its direct-touch term.
At 617.7M the earlier reward had 252/256 fixed kickoff wins, two losses,
one timeout, and one censored; 469 replay-mixed goals and 43 timeouts
with all 512 completed. The `f7cab81` reward-only run near 626M had
253 wins / three losses / no timeouts and 472 mixed goals / 40
timeouts. In a balanced 512-game cross-policy matchup at these steps,
`f7cab81` led `de8975f` **246–222** (44 non-goal endings), a modest gap
relative to the much larger differences seen against the archived actor.
At 721.2M the earlier reward's frozen kickoffs were 253 wins / one loss /
two timeouts in 110,283 actions, with 471 mixed self-play goals / 41
timeouts, all completed. The `f7cab81` matched-steps policy at 729.0M
had 253 wins / three losses / no timeouts in 90,307 actions and 477
mixed goals / 35 timeouts. **The direct 512-game comparison widened
substantially:** `f7cab81` won **303–178**, with 31 non-goal endings
(101–59 on kickoff starts, 202–119 on replay starts). Simple goals
against a weak opponent and same-policy self-play hide this meaningful
competitive difference. Keep checking later snapshots before drawing a
long-run conclusion.
At 823.1M `de8975f` had 255/256 frozen kickoff wins, zero losses, one
timeout, in 99,032 actions, and 474 replay-mixed goals / 38 timeouts.
The `f7cab81` policy near 830.4M had 253 kickoff wins / three losses
in 68,905 actions and 483 mixed goals / 29 timeouts. A second direct
512-game matchup favored `f7cab81` **286–193** (33 non-goal endings;
98–62 kickoff starts and 188–131 replay starts), confirming that the
weak-opponent win-rate edge of the earlier reward does not yet translate
to competitive superiority. Continue to look for later improvement.
At 925.7M `de8975f` finally swept the 256 fixed random-opponent
kickoffs, but needed 89,280 aggregate actions (archive: 40,889;
`f7cab81` 932.7M: 56,342); its replay-mixed self-play was 477 goals /
35 timeouts, versus 492/20 for the matched `f7cab81` policy. Direct
balanced play clearly favored `f7cab81` **318–168** (26 non-goal
endings, with kickoff starts 121–39). The archive beat `de8975f`
**496–8** (eight non-goal endings; 160–0 kickoff starts), still very
far from the much older actor's competitive play. Continue the trial toward the
~3B skill milestone to determine whether it improves later.
At 1.033B `de8975f` had 254/256 random-opponent kickoff wins, one
loss, one timeout (100,044 aggregate actions), and 480 goals / 32
timeouts in all 512 replay-mixed episodes. The matched `f7cab81`
policy at 1.040B had 255 random-opponent wins and 486 mixed goals /
26 timeouts; its direct head-to-head lead was **322–162** over 512
games (28 non-goal endings, including 119–41 on kickoff starts).
The substantial competitive gap has not recovered by ~1B despite
the earlier reward's completed-game timeouts continuing to decline.

The `de8975f` trainer was subsequently stopped at policy
**1,328,319,849** steps to move to the modern-stack trial. Its final actor
and optimizer checkpoint were verified byte-identical to the persistent
`/workspace/july31-recreation-20260924/live/checkpoints/` backups. Only
one training process was active at a time; the July comparisons remain
available for later matched-step evaluation.
