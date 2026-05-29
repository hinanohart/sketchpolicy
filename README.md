# sketchpolicy

**Simulator-free, CPU-only kinematic demonstration *multiplier* for `LeRobotDataset` v3.0.**

> ⚠️ **Pre-alpha bootstrap scaffold (v0.1.0a1).** This is a research scaffold,
> not a production tool. Every write command requires the explicit flag
> `--i-understand-this-is-a-bootstrap-scaffold`. Read the
> [Scope & honest limitations](#scope--honest-limitations) section before using.

`sketchpolicy` takes a *valid* `LeRobotDataset` v3.0 episode whose actions are
end-effector poses, and produces additional schema-valid episodes by applying
**deterministic kinematic transforms** (viewpoint / object-pose / planar
position / time-warp) sampled with a **low-discrepancy Sobol sequence** and
filtered by a **reject-and-resample feasibility test** (out-of-reach and
self-intersection are discarded). No simulator, no neural network, no GPU.

It also ships an *experimental* "sketch" adapter that turns a monocular video
of a **human hand** into an end-effector trajectory using MediaPipe Hands
(Apache-2.0, CPU, 21 landmarks). "Film your hand → sketch a trajectory."

## Install

```bash
pip install sketchpolicy                # core: numpy / scipy / pyarrow (torch-free)
pip install "sketchpolicy[replay]"      # + headless pybullet sanity replay
pip install "sketchpolicy[ingest]"      # + experimental hand-video adapter (mediapipe)
pip install "sketchpolicy[lerobot]"     # + live-reader cross-check (heavy: pulls torch+CUDA)
```

## Quickstart

```bash
# Multiply an EE-action dataset: 1 source episode -> N feasible kinematic variants
sketchpolicy augment ./my_dataset --out ./my_dataset_x10 --n 10 --seed 0 \
    --i-understand-this-is-a-bootstrap-scaffold

# Read-only: render a kinematic replay of one episode (needs [replay])
sketchpolicy replay ./my_dataset --episode 0

# Environment / dependency report
sketchpolicy doctor

# Experimental: hand video -> EE trajectory episode (needs [ingest])
sketchpolicy sketch ./hand_clip.mp4 --out ./sketched \
    --i-understand-this-is-a-bootstrap-scaffold
```

## What it measures

The numbers below come from `bench_results/v0.1.0a1.json`, produced by
`python scripts/measure.py` on the build machine (Linux WSL2, Python 3.12,
numpy 2.4 / scipy 1.17 / pyarrow 24). They are **operational** metrics —
round-trip validity, determinism, the feasibility filter, CPU throughput and a
replay smoke test. They are **not** accuracy or policy-success metrics (there is
no robot ground truth in this benchmark). Re-run `measure.py` to regenerate.

| metric | value |
| --- | --- |
| augment round-trip schema-valid | ✓ (recovered actions within 1e-6) |
| same-seed bit-exact determinism | ✓ |
| reject-resample discards infeasible | ✓ (3 of 8 draws rejected on the boundary fixture) |
| MediaPipe Hands CPU throughput | 66.4 fps (480×640, synthetic-frame throughput, not detection accuracy) |
| pybullet replay smoke | ✓ (max IK residual 0.01 m on a reachable clip) |

## Scope & honest limitations

**What sketchpolicy CLAIMS (and tests):**

- Given a valid EE-action `LeRobotDataset` v3.0 input, it emits **schema-valid**
  v3.0 output that round-trips against a pinned schema fixture.
- Transforms are **deterministic**: the same `--seed` produces bit-exact output.
- The feasibility filter **rejects** out-of-reach / self-intersecting variants
  and resamples to hit the requested count.
- It runs on **CPU** with no torch in the core install.

**What sketchpolicy does NOT claim (hard limits):**

- ❌ It does **not** improve any policy's success rate. It is a data-shaping
  tool; downstream training quality is out of scope and untested here.
- ❌ It does **not** recover absolute metric scale or full 6-DoF accuracy from a
  monocular hand video. The sketch adapter is experimental and relative.
- ❌ It does **not** do sim-to-real, and it does **not** transform camera video;
  v0.1 augments the **proprioceptive / action stream only** (EE pose + gripper).
  Output episodes are action/state-only.
- ❌ Parallel-jaw gripper only in v0.1 (no dexterous / bimanual).

It depends on **MediaPipe Hands only** for hand pose. It does **not** use MANO,
HaMeR, or WiLoR (non-commercial / GPU-ViT — see `NOTICE`).

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
