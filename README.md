<div align="center">

# STAR

**Visual adaptation for reinforcement learning and Diffusion Policy**

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)
![Source only](https://img.shields.io/badge/Release-source%20only-64748B)

[Project Page](https://mark-98.github.io/star-project-page/)

<a href="https://mark-98.github.io/star-project-page/">
  <img src="assets/project-page-qr.svg" width="164" height="164" alt="Scan to visit the STAR project page">
</a>

[Install](#1-install) · [Train](#3-train) · [Evaluate](#4-evaluate) · [Diffusion Policy](docs/DIFFUSION_POLICY.md)

</div>

STAR learns a policy, then adapts its visual representation to changing observations.

| Workflow | Supported tasks | Environment |
| --- | --- | --- |
| Reinforcement learning | DMC: Walker, Cheetah, Finger, Cartpole; Robosuite: Door, TwoArmPegInHole | Python 3.8, Robosuite 1.4, MuJoCo 2.3 |
| Diffusion Policy + STAR | Robomimic Can and Square, PH image demonstrations | Python 3.9, Robosuite 1.2, MuJoCo-py 2.1 |

Use **separate environments** for the two workflows. This repository contains
source, training and evaluation code—not pretrained weights, datasets or experiment outputs.

```text
Stage 1: learn teacher + collect replay → Stage 2: align encoder/MLP → Evaluate
```

Stage 1 collects **1,000 steps**, then takes **one learner update per environment
step**, without gradient clipping.

## Quick start: reinforcement learning

### 1. Install

Requirements: Linux, Python **3.8**, Git, NVIDIA GPU access and system EGL/OpenGL
libraries. Check `nvidia-smi` first.

```bash
git clone https://github.com/Mark-98/STAR.git
cd STAR

python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.0 setuptools==65.6.3 wheel
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .

python scripts/setup/build_dmc_bindings.py
python -m pip install --no-deps --no-build-isolation \
  -e vendor/dm_control -e vendor/robosuite -e vendor/dmc2gym \
  -e vendor/dmcvgb -e vendor/robosuitevgb

export MUJOCO_GL=egl
export WANDB_MODE=disabled
python -m pip check
```

Keep the editable checkout. Do not install a second simulator package over the
retained forks. [Version pins and troubleshooting →](docs/RL_GUIDE.md#1-install)

### 2. Download public benchmark assets

Download the asset folders directly from
[RL-ViGen](https://github.com/gemcollector/RL-ViGen), at the pinned revision.

```bash
mkdir -p datasets
git clone --filter=blob:none --no-checkout \
  https://github.com/gemcollector/RL-ViGen.git datasets/RL-ViGen
git -C datasets/RL-ViGen sparse-checkout init --cone
git -C datasets/RL-ViGen sparse-checkout set envs/DMCVGB/data
git -C datasets/RL-ViGen checkout --detach 90d8b8c40acb63af6f938c1f4cc79a0cfee7d7ec

export STAR_DMC_ASSET_ROOT="$PWD/datasets/RL-ViGen/envs/DMCVGB/data"
python scripts/check_assets.py --group dmc
```

Re-export the path in new shells.
Use RL-ViGen as an asset source only; do not run its installation scripts.
For Robosuite, also prepare the textures and background videos in the
[Robosuite asset instructions](docs/RL_GUIDE.md#2-download-benchmark-assets).

### 3. Train

**Stage 1:** DrQ teacher + forward dynamics + replay collection.
**Stage 2:** encoder/MLP alignment to the frozen teacher.

A complete **DMC Walker Walk** example (seed 300):

```bash
python train_stage1.py --task_name walker_walk --seed 300 \
  --exp_name walker_teacher_s300
```

The [DMC preset](configs/rl/stage1.yaml) sets batch **256**, LR **1e-4**,
forward-loss weight **5**, action repeat **4** and a **500k-step** budget.
Stage 1 prepares merged replay for Stage 2 at the end.

For **Stage 2 appearance adaptation**, first
[download Places365 and set its path](docs/RL_GUIDE.md#download-places365-for-stage-2-appearance-adaptation).
Then:

```bash
python train_stage2.py --task_name walker_walk --seed 300 \
  --exp_name walker_star_s300 \
  --teacher_work_dir logs/walker_teacher_s300/model/snapshot_2000000.pt \
  --buffer_dir logs/walker_teacher_s300/buffer
```

Stage 1 counts **environment steps**; Stage 2 counts **optimizer updates**.
Replay and datasets can be large: use a disk with sufficient space.

[All task overrides, DMC and geometry training →](docs/RL_GUIDE.md#4-train-star)

### 4. Evaluate

Evaluate Walker Walk under the video-hard condition by setting the episode count:

```bash
python -m star.evaluation.rl --env dmc --task walker_walk \
  --checkpoint logs/walker_star_s300/model/snapshot_100000.pt \
  --mode video_hard --action-repeat 4 --seed 1 --episodes 30 \
  --output-dir outputs/walker_video_hard
```

Results: `outputs/walker_video_hard/summary.json`.
DMC supports color, video, rotate and shift conditions. Robosuite selects scenes
automatically; no scene-ID argument is needed.
Do not use `--max-steps` for benchmark scores, or load untrusted pickle checkpoints.

[Evaluation conditions and DMC commands →](docs/RL_GUIDE.md#5-evaluate)

## Diffusion Policy

Use the **[Can/Square guide](docs/DIFFUSION_POLICY.md)** for the independent
environment, public demonstration download, Base → STAR training and evaluation.
Its installation, demonstration datasets and evaluation protocol are separate from RL.

## More information

| Guide | Contents |
| --- | --- |
| [RL guide](docs/RL_GUIDE.md) | Full installation, task settings, Places365 and troubleshooting |
| [Diffusion Policy](docs/DIFFUSION_POLICY.md) | Can/Square data, training and episode-count evaluation |
| [Assets](docs/ASSETS.md) | Required folders, hashes and backgrounds |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Upstream attribution and retained licenses |

Built on DAD, DrQ-v2/RL-ViGen, Diffusion Policy, Robomimic, dm_control, dmc2gym,
Robosuite and SECANT. Original notices remain in `LICENSE`, `licenses/` and
vendor trees. Datasets, custom media and learned weights are not redistributed.
