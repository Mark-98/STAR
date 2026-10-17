# Diffusion Policy validation

Date: **2026-10-11** · [Guide](DIFFUSION_POLICY.md) · [Machine-readable receipt](diffusion_validation.json)

The Can/Square implementation passes bounded installation, data, training and
evaluation checks. **This is not full retraining to published scores.**

## What passed

| Check | Evidence | Limit |
| --- | --- | --- |
| Fresh dependency environment | New Conda prefix, editable project install, CUDA import, freshly compiled MuJoCo-py, clean `pip check` | Same host/driver; OSMesa rendering |
| Official prepared data | Both Can/Square HDF5s downloaded with byte ranges; ZIP CRC verified; SHA-256 matches existing experiment data | Only PH absolute-action image data |
| Base → STAR updates | Both tasks: real replay loading, one update per stage, frozen parameters unchanged, saved/reloaded predictions bitwise equal | Reduced UNet; four denoising steps; tiny dataset/Places fixtures |
| Public training CLI | Can: Base and STAR each complete one epoch, one train/validation update, save checkpoints and run a rollout | Eight-step debugging rollout, not a complete benchmark episode |
| Fresh-policy evaluation | Both tasks × two models × four conditions = **16 complete episodes** in the new environment | Newly initialized/reduced models are not trained agents |
| Existing checkpoint evaluation | Both tasks × Base/STAR × four conditions = **16 complete episodes**, no inference-step override | Existing DP dependency stack and checkpoints, not newly trained weights |
| Regression/distribution | **29 tests** including existing RL tests, scene cycling and downloader range validation; source-only Git export runs the Can model check | Not a complete test of every optional legacy branch |

Complete evaluations use the public `star.evaluation.diffusion` entrypoint,
`--episodes 1 --n-envs 1`, demo150 and the first fixed visual definition per
difficulty. Each episode reaches its configured 400-step horizon. Scene IDs are
not supplied manually. The tiny training CLI rollout is **not** counted among
the 32 complete evaluation episodes.

Both fresh model checks use the first 32 rows of two demonstrations. The image
overlay fixture contains 16 real JPEGs from the public Places365 **validation**
archive, arranged in a separate test-only ImageFolder. This tests loading and
augmentation, not the full Places365 training distribution. The Git-export Can
check and training CLI use the newly downloaded public Can data, not an author's
data path baked into source. No original research files or checkpoints were edited.

## Checkpoint filename update

Base and STAR now save `checkpoints/last_snapshot.ckpt` by default, without
score/epoch-named files. The public training CLI was rechecked for **both Can and
Square, both stages**, using one epoch, one train/validation update, a reduced
UNet and an eight-step rollout. All four runs produced only `last_snapshot.ckpt`
with no checkpoint-setting overrides, including STAR's final-epoch save.

Both tasks also passed real-data model checks with the new filename: frozen
parameters stayed unchanged and saved/reloaded action predictions were bitwise
equal. Six new checkpoint tests passed in the DP environment, covering synchronous
and asynchronous saves, explicit paths, legacy `latest.ckpt` loading and new-file
precedence. The existing 29 RL/shared tests passed in the separate RL environment;
the six optional DP checkpoint tests are skipped there because `dill` is absent.
These bounded checks do not establish full retraining or benchmark scores.

## Installation corrections discovered during testing

The first new environment exposed a missing `pandas` dependency when importing
the workspace logger. Installing `pandas==1.5.3` fixed it; that dependency is now
in the environment recipe. The pinned Robosuite fork requires Numba ≤0.53.1,
so its pip installation had downgraded the initial Conda 0.56.4 selection. The
final recipe explicitly pins pip Numba 0.53.1 and removes the conflicting pin.

The full new environment was created from the initial recipe and corrected in
that **new environment only**; the final edited recipe also passed a separate
Conda dry-run. It was not independently recreated a second time after those
two recipe edits. All reported fresh-runtime tests and `pip check` ran after
the corrections, with no system-site-packages dependency sharing.

`free-mujoco-py==2.1.6` supplies its own MuJoCo 2.1 binaries. Its extension was
built inside the new environment; old compiled extensions were not borrowed
for the fresh-environment checks. Explicit external MuJoCo path variables were
removed for the fresh model/evaluation checks. The earlier existing-stack
checkpoint checks used a private copy of that stack's extension to avoid
writing its build lock into the original environment.

## Data identity and conversion

The downloader selected two members of the official 84,754,919,326-byte archive,
without downloading/storing the entire ZIP:

| Prepared file | Bytes | SHA-256 |
| --- | ---: | --- |
| `can/ph/image_abs.hdf5` | 2,010,694,944 | `6eb74b811a47c94a4409aa6f8f070b172dd8bd2d5190d020209b515b97573e03` |
| `square/ph/image_abs.hdf5` | 2,603,429,840 | `08aa2d987061ed6e30da20e64ad8f56c0b8524bf55b516181754b22c271d4f1e` |

The delta→absolute conversion command also ran on 16 state/action rows per task
with unchanged inputs and finite 7-D outputs. **The first converted action did
not match the prepared archive.** In the recorded check the maximum differences
were ~0.10143 for Can and ~0.03521 for Square; after the first row, differences
were below 3e-8. The first-row result also varied across conversion invocations.
This is why the guide recommends downloading the prepared absolute-action files
for historical data reproduction. Conversion completion alone is not data identity.

## What remains unverified

- Full 3,050-epoch baseline and 20-epoch STAR retraining, multi-seed scores and
  numerical agreement with a paper table.
- Full Places365 training data, every demonstration and every rendered visual
  scene. All frozen scene-definition hashes are checked, but only the first
  scene per difficulty/task is rendered in these bounded evaluations.
- A clean operating-system install, a different host/driver, and GPU/EGL rendering.

The v6 visual definitions are **outcome-selected** and demo150–199 were reused
during calibration. They are not an untouched held-out test set. One episode per
condition is an execution check, not a meaningful success-rate estimate.
