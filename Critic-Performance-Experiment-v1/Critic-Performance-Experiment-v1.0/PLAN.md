# Offline-to-online RL — measuring critic plasticity (Cal-QL + IQL on Pendulum)

Normal offline-to-online RL pipeline. The research question is
**critic plasticity**: how the critic network's ability to keep learning
degrades across offline pretraining and online fine-tuning. Cal-QL and IQL
run on the same shared offline dataset so results compare backbones on
this axis, but the core deliverable is the plasticity measurement suite
itself, applied to each backbone's critic.

## Backbones

- **Cal-QL**: Tanh-Gaussian actor + twin Q critics. Offline pretraining
  uses a CQL conservative penalty (logsumexp over Q) plus Cal-QL's
  calibration term - lower-bounding Q by a Monte-Carlo return-to-go
  estimate from the dataset so OOD actions aren't overestimated. Online
  fine-tuning reuses the exact same loss.
- **IQL**: same actor/critic skeleton, plus a separate state-value network
  V(s). V is fit via expectile regression against Q (no conservative
  penalty needed), Q is fit via TD against V, and the actor is extracted
  by advantage-weighted regression.

## Phases

**Phase 0 — offline dataset.** Fixed transition buffer from a mediocre
policy on `Pendulum-v1` (partially-trained SAC + action noise), with
per-transition Monte-Carlo return-to-go for Cal-QL's calibration term.
Shared by both backbones.

**Phase 1 — offline pretraining.** Train each backbone on the offline
buffer until offline return saturates.

**Phase 2 — online fine-tuning.** Continue training with real env
interaction, offline buffer retained and mixed into online batches.

**Critic plasticity diagnostics run at every eval checkpoint in BOTH
Phase 1 and Phase 2** - plasticity loss is known to accrue during offline
pretraining itself, so measurement can't start only once online
fine-tuning begins.

## Critic plasticity measurement suite

All of the below operate on the critic (Q-network), not the actor.

### Per-layer activation diagnostics (every hidden layer, both Q1 and Q2)
For a batch of `(s,a)` pairs, post-ReLU activations `A` (batch x units) per
hidden layer `l`:
- **FAU**: `mean(A > 0)` over batch and units.
- **Dormant ratio**: per-unit magnitude score `s_j = mean(|A[:,j]|)`,
  normalized `ŝ_j = s_j / mean_j(s_j)`; unit is dormant if `ŝ_j <= τ`
  (default τ=0.1, configurable). Dormant ratio = fraction of units meeting
  that (Sokar et al. "dormant neuron" definition - magnitude-based, softer
  than FAU).
- **Dead fraction**: units with firing rate `r_j = mean_i(A[i,j] > 0)`
  exactly 0 (never active in the batch). Strict subset of dormant.
- **Linearized fraction**: units with `r_j = 1` (always active, ReLU never
  clips them - the opposite failure mode: loss of nonlinearity rather than
  loss of capacity).

Reported per layer, plus a mean-across-layers rollup per metric.

### SRank (penultimate layer)
Effective rank via SVD (min k s.t. top-k singular values capture
`1-eps` of total mass, eps=0.01) of the second-to-last hidden layer's
activations.

### Weight norm
L2 norm of all parameters, reported per twin (Q1, Q2).

### Probe 1 — random-target fitting (supervised plasticity probe)
1. Frozen randomly-initialized target network `g` (same architecture as a
   Q-network). Targets `y = g(s,a)` on a fixed probe batch from the buffer.
2. **Trained-copy run**: deep-copy the current Q1, train on `(s,a) -> y`
   via MSE for `probe_steps` steps.
3. **Fresh-net run**: freshly-initialized network, identical architecture,
   same `y`, same minibatch sequence (same RNG draws) as the trained-copy
   run.
4. `probe1_ratio = fresh_final_error / trained_copy_final_error`.

### Probe 2 — TD-to-another-policy (RL plasticity probe)
1. A frozen, randomly-initialized actor π' drawn **once per run** (not
   per-checkpoint) and reused for every diagnostic checkpoint, so the
   target task is held fixed and only the critic's ability to learn it
   changes over time.
2. **Trained-copy run**: deep-copy current Q1 + build a lagged target
   copy; train via TD on real `(s,a,r,s')` transitions from the buffer,
   bootstrapping through `π'(s')` instead of the agent's own policy.
3. **Fresh-net run**: identical procedure, fresh network + fresh target,
   same π', same minibatch sequence.
4. `probe2_ratio = fresh_final_td_error / trained_copy_final_td_error`.

Both ratios: ~1 = no plasticity loss, <<1 = critic has lost the ability to
fit new targets relative to a same-architecture fresh network.

### Practical decisions
- Probes run on Q1 only (compute control); per-layer/SRank/weight-norm
  stats are forward-pass-only and reported for both twins.
- `probe_steps` defaults conservatively (~200) and is configurable -
  each probe is itself a short training loop (2 networks x probe_steps),
  run at every checkpoint, so cost adds up across many checkpoints even
  though each call is cheap on these tiny MLPs.

## Research question

Two halves:
1. **Does the critic lose plasticity with progressively longer offline training?** 
   Measured from the offline-phase diagnostics collected at
   every eval checkpoint of a long offline run, across multiple seeds.
2. **Does that limit the online improvement achievable from that point?**
   Branch an independent online fine-tuning run off every saved offline
   checkpoint, holding the exploration seed and probe infrastructure fixed
   across all branches for a given (backbone, seed) - so the critic's
   checkpoint state is the only thing that differs between branches.
   Online improvement (`delta_return`, AUC) is then correlated
   (Pearson/Spearman) against each checkpoint's plasticity metrics -
   this correlation is the actual test of the hypothesis.

## Parameters to set per experiment run

1. **Step budget** - `offline_steps`, `eval_every`, `checkpoint_every` in
   each backbone's config (`checkpoint_every` must be a multiple of
   `eval_every`).
2. **Compute strategy** - how the seeds x backbones training budget will
   actually be run (sequential, parallelized, time-boxed, etc.).
3. **Online branch budget** - `online_steps`/`eval_every` passed to
   `run_online_sweep()` for half 2.
