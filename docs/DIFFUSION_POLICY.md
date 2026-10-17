# Diffusion Policy + STAR

[← Main README](../README.md) · [Install](#1-install-a-separate-environment) · [Data](#2-download-demonstrations) · [Train](#4-train-base--star) · [Evaluate](#5-evaluate-by-episode-count)

**Can and Square · proficient-human demonstrations · image observations**

Train a baseline image-conditioned diffusion action policy, then adapt its visual
encoder with per-camera STAR MLPs while freezing the teacher encoder and diffusion
action model. This is imitation learning, **not the RL diffusion/consistency variant**.

| Stage | Learns | Input |
| --- | --- | --- |
| Base | Image encoder and diffusion action model | Absolute-action demonstrations |
| STAR | Student encoder and per-view feature adapters | Base checkpoint, demonstrations, Places365 overlays |
| Evaluate | Nothing; Base and STAR are evaluated as a pair | Checkpoints, demonstration initial states, automatic visual conditions |

## 1. Install a separate environment

Use Linux with an NVIDIA GPU. **Do not activate the RL `.venv` or install anything
under `vendor/` here.** The stacks use incompatible simulator APIs.

```bash
git clone https://github.com/Mark-98/STAR.git
cd STAR
# Skip cloning if you already have this checkout.

# Ubuntu prerequisites (package names may differ on other distributions):
sudo apt install build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
conda env create -f environments/diffusion.yaml
conda activate star-diffusion
python -m pip install --no-deps --no-build-isolation -e .

# Tested headless renderer; policy inference still uses CUDA.
export MUJOCO_PY_FORCE_CPU=1
export MUJOCO_GL=osmesa
export WANDB_MODE=disabled

python -m pip check
python -c "import torch, robosuite, mujoco_py; print(torch.cuda.is_available(), robosuite.__version__)"
```

Expected key versions: Python 3.9, PyTorch 1.12.1, Robomimic 0.2.0,
Robosuite **1.2.0**, diffusers 0.11.1. The pinned Robosuite fork supports the
original demonstration environments. The recipe follows the
[upstream simulation stack](https://github.com/real-stanford/diffusion_policy#%EF%B8%8F-installation).
`free-mujoco-py==2.1.6` **bundles MuJoCo 2.1** and sets its own runtime path:
no separate MuJoCo download or `MUJOCO_PY_MUJOCO_PATH` export is required.
It compiles its extension on first import. The pinned Robosuite fork requires
Numba 0.53.1; the recipe pins this explicitly rather than installing a newer
Conda Numba and letting pip silently downgrade it. OSMesa is the validated
renderer here; GPU/EGL rendering is not part of the fresh-install claim.

**Verification scope:** see [current DP validation](DIFFUSION_VALIDATION.md) for
fresh-environment checks, real-data minibatch updates and bounded simulator
evaluations. These checks do not establish full training to published scores.

## 2. Download demonstrations

Use the **prepared absolute-action datasets** from the official
[Diffusion Policy data archive](https://diffusion-policy.cs.columbia.edu/data/training/):
Can and Square, 200 proficient-human demonstrations each. Our downloader uses
HTTP byte ranges to fetch **only those two members** (~4.6 GB uncompressed),
not the entire ~85 GB ZIP. No extra Python package is needed for downloading.

```bash
python scripts/download_diffusion_data.py --dry-run
python scripts/download_diffusion_data.py
export STAR_DATA_ROOT="$PWD/datasets"
```

Output: `datasets/can/ph/image_abs.hdf5` and
`datasets/square/ph/image_abs.hdf5`. To fetch only one, add `--task can` or
`--task square`. Existing files are protected; do not rerun over an existing
dataset. Downloads check ZIP CRC and print SHA-256. Interrupted downloads retain
a `.part` file and cannot be resumed by this script; inspect/remove only that
partial file before retrying, or choose a new output root.

HDF5 actions are 7-D absolute pose commands; the dataset loader converts rotation
to the policy's 10-D rotation-6D representation. No action conversion is needed
for these prepared files. Budget additional space for multi-GB model checkpoints.

<details>
<summary>Alternative: original delta-action data and conversion</summary>

The original [Robomimic v0.1 PH image datasets](https://robomimic.github.io/docs/v0.2/datasets/robomimic_v0.1.html)
are available per task. Choose this route **instead of** the prepared download,
starting with an empty `datasets/` destination:

```bash
mkdir -p datasets/can/ph datasets/square/ph
wget -c https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/image.hdf5 \
  -P datasets/can/ph
wget -c https://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image.hdf5 \
  -P datasets/square/ph

python -m diffusion_policy.scripts.robomimic_dataset_conversion \
  -i datasets/can/ph/image.hdf5 -o datasets/can/ph/image_abs.hdf5 -n 4
python -m diffusion_policy.scripts.robomimic_dataset_conversion \
  -i datasets/square/ph/image.hdf5 -o datasets/square/ph/image_abs.hdf5 -n 4
```

Conversion refuses to overwrite an existing destination. If you already have
the absolute-action HDF5s, skip conversion. This route needs disk space for both
original and converted files. **It is not identical to the prepared archive:**
our 16-row-per-task checks found a differing first action; subsequent rows agreed within
3e-8. For historical dataset reproduction, use the prepared files above, not a
fresh conversion. See [validation details](DIFFUSION_VALIDATION.md).

</details>

## 3. Prepare visual assets and run a quick model check

For visual evaluation, follow the [RL-ViGen download section](../README.md#2-download-benchmark-assets)
and keep `STAR_TEXTURE_ROOT` set. DP uses the **same 40 texture files**; it does
not need DMC assets or RobosuiteVGB background videos. Validate just textures:

```bash
python scripts/check_assets.py --group textures
python -m unittest discover -s tests -p test_diffusion_checkpoint.py -v
python scripts/smoke_diffusion.py --task can --output-dir outputs/dp_smoke_can
python scripts/smoke_diffusion.py --task square --output-dir outputs/dp_smoke_square
```

Each quick check performs one Base and one STAR update, checks frozen parameters,
saves/reloads a fresh checkpoint, and compares action outputs. It uses synthetic
images, a smaller UNet, batch overlays and four denoising steps: **not a simulator
test or a score reproduction**. Choose a new output directory for every run.
The six checkpoint tests cover default filenames, asynchronous saving, reloads
and legacy `latest.ckpt` resume compatibility. Smoke-check checkpoints are saved
under `base/checkpoints/last_snapshot.ckpt` and `star/checkpoints/last_snapshot.ckpt`
inside the chosen output directory.

For real STAR training, download Places365 as described in the
[main training guide](../README.md#download-places365-for-stage-2-appearance-adaptation)
and export `PLACES365_ROOT`. The training entrypoint rejects a missing Places365
directory instead of silently substituting a different overlay source.
Base training and evaluation do not need Places365.

After preparing that data, optionally exercise the real replay loader and Places365
augmentation too (still a reduced-model, single-minibatch check):

```bash
python scripts/smoke_diffusion.py --task can \
  --dataset datasets/can/ph/image_abs.hdf5 --places-root "$PLACES365_ROOT" \
  --output-dir outputs/dp_smoke_real_can
```

This reads the first 32 rows from two demonstrations into a new tiny fixture;
the original HDF5 is opened read-only.

## 4. Train Base → STAR

Run from the STAR repository root with `star-diffusion` active. W&B remains
disabled by the setup above. These commands use the full-size architecture.

```bash
# Can: first train the baseline diffusion policy.
python -m star.training.diffusion --config-name=train_diffusion_unet_hybrid_workspace \
  task=can_image_abs training.seed=0 training.resume=false \
  task.env_runner.n_envs=10 hydra.run.dir=outputs/can/base_seed0

# Then adapt the baseline's visual representation.
python -m star.training.diffusion --config-name=v6_can \
  pretrained_checkpoint="$PWD/outputs/can/base_seed0/checkpoints/last_snapshot.ckpt" \
  hydra.run.dir=outputs/can/star_base0
```

For Square, use `task=square_image_abs`, `--config-name=v6_square` and distinct
Square output paths. Baseline defaults to 3,050 epochs; STAR uses 20 epochs.
The retained STAR recipe uses a three-layer, non-residual, per-view MLP with
pairwise MSE alignment; the diffusion actor stays frozen.

| Setting | Meaning |
| --- | --- |
| Base `training.seed=0` | Baseline training seed; repeat with 1 and 2 for three runs |
| STAR `training.seed=42` | Historical adaptation seed, preserved even for different Base run IDs |
| `task.env_runner.n_envs` | Parallel simulator count; reduce for memory constraints |
| `dataloader.batch_size` | Default 64; changes affect the training recipe |
| `pretrained_checkpoint` | Your matching Base checkpoint, not a bundled weight |

Both Base and STAR save **`checkpoints/last_snapshot.ckpt`**, updated at each
checkpoint interval and at the final epoch. This is the most recent saved state,
not a best-score selection. Score/epoch-named checkpoints are disabled by default;
set `checkpoint.topk.k=1` if you also want a best-score checkpoint.
Older `latest.ckpt` files remain supported for resuming, and explicitly supplied
checkpoint paths still work. The file remains a standard `.ckpt` checkpoint,
not a full-workspace `.pkl` snapshot.
Use fresh output directories: Hydra's training workspace can otherwise reuse
logs/checkpoints. Only use trusted configs/checkpoints; Hydra's `eval` resolver
and legacy pickle checkpoint loading can execute Python code.

## 5. Evaluate by episode count

You supply the task, dataset, two checkpoints and **`--episodes`**. No scene ID is
required or accepted by this public entrypoint. Clean, easy, medium and hard run
automatically, pairing Base and STAR on the same initial states and visual scenes.

```bash
python -m star.evaluation.diffusion \
  --task can --dataset datasets/can/ph/image_abs.hdf5 \
  --base-checkpoint outputs/can/base_seed0/checkpoints/last_snapshot.ckpt \
  --ours-checkpoint outputs/can/star_base0/checkpoints/last_snapshot.ckpt \
  --episodes 50 --n-envs 10 --output-dir outputs/can/eval_base0
```

Use `--dry-run` to inspect counts without importing a simulator or loading weights.
For Square change `--task square`, the dataset and both checkpoints.

- `--episodes N` means **N per model per condition**: the default four conditions
  and two models produce `8 × N` episodes. `--phase clean` or `--phase visual`
  selects a subset. `--num-episodes` remains an alias.
- Ten fixed visual scenes per difficulty cycle automatically. Their IDs are only
  reproducibility metadata, not a user selection step.
- Default resets use `demo_150` onward. With the 200-demo PH data, the default
  range supports at most 50 episodes; an unavailable range fails explicitly.
  `--demo-start` is an advanced override, not automatic reuse of nonexistent data.
- Policy RNG seed is 42; physical initial states come from saved demonstrations.
  Keep `--n-envs` fixed when comparing repeated
  runs: batch size can change stochastic diffusion sampling.
- The retained comparison uses raw **`model` weights** for both checkpoints,
  not automatic Base EMA replacement. `--model-seed` labels paired Base run IDs.
- Results: `<output-dir>/clean/summary.json`, `visual/summary.json`, per-model
  episode records, and `resolved_config.json`. Existing outputs are protected.

> **Protocol caveat:** the included v6 visual scenes were selected using outcomes,
> and demo150–199 were reused during earlier calibration. They are **not an
> untouched held-out test set**. This release preserves that historical protocol;
> it does not support a claim of unbiased generalization from these scores.
> Training-time rollout scenes and this frozen final-evaluation registry are
> different protocols. See [scene provenance](scene_provenance.json).

## Included / excluded

Included: the image-conditioned UNet policy, STAR adaptation, workspace, replay
dataset/conversion, required rotation/normalization/vision modules, simulator
wrappers and Can/Square configs. Legacy `DADPolicy` class names are preserved for
checkpoint compatibility.

Not included: learned weights, demonstration/Places365 data, generated outputs,
PushT/Kitchen/real-robot control stacks, competing policies, or RL Stage-2
diffusion/consistency variants. `diffusion_policy/real_world/video_recorder.py`
is retained only because the simulation runner imports its video utility.
