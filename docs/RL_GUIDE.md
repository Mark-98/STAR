# Reinforcement learning: complete guide

[Back to the quick start](../README.md)

Run the commands below from the STAR repository root. This guide retains the full
installation, asset, training and evaluation instructions.

STAR adapts visual representations while preserving a learned action policy.
Choose one of two independent workflows:

| Workflow | Tasks | Start here |
| --- | --- | --- |
| Reinforcement learning | DMC; Robosuite Door, TwoArmPegInHole | [Steps 1–5 below](#1-install), Python 3.8 |
| Diffusion Policy + STAR | Robomimic Can, Square (PH image demonstrations) | [Separate installation, data, training and evaluation guide](DIFFUSION_POLICY.md), Python 3.9 |

**Do not mix their environments:** RL uses Robosuite 1.4 / MuJoCo 2.3;
Diffusion Policy uses Robosuite 1.2 / MuJoCo-py 2.1.
Diffusion Policy's Base and STAR checkpoints both use
`<run-dir>/checkpoints/last_snapshot.ckpt`; no score/epoch filename lookup is needed.
In the RL workflow below, Stage 1 learns a **DrQ teacher with forward dynamics**;
Stage 2 adapts an image encoder and **four-layer MLP** to its frozen features.

```text
Stage 1                      Stage 2                         Evaluation
Teacher + replay collection → Encoder/MLP feature alignment → Shifted observations
```

| Benchmark | Tasks | Evaluation conditions |
| --- | --- | --- |
| DMC | Walker Walk, Cheetah Run, Finger Spin, Cartpole Swingup | Clean; color, video, rotate and shift × easy/hard |
| Robosuite | Door, TwoArmPegInHole | Clean, easy, medium, hard; automatic scene rotation |

The release includes the **STAR MLP RL core** and **Can/Square Diffusion Policy
with STAR feature adaptation**. Pretrained weights, datasets, competing agents,
RL diffusion/consistency variants and experiment outputs are not included.

> **New here?** Follow steps 1–3 first. The quick check creates its own tiny
> teacher and replay: no private checkpoints or Places365 download are needed.
> It checks execution, not trained-policy quality or paper scores.

## 1. Install

**Before you start:** Linux, Python **3.8**, Git, a working NVIDIA driver and
system OpenGL/EGL libraries. Check that `nvidia-smi` works. This legacy stack
uses PyTorch 1.12.1/CUDA 11.6 and MuJoCo 2.3.0; use a dedicated environment.

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
python -m unittest discover -s tests -v
```

Expected: `No broken requirements found` and a passing unit suite. Optional
Diffusion Policy tests are skipped when its separate dependencies are absent.
Check GPU access with `python -c "import torch; print(torch.cuda.is_available())"`.
All following commands run from the **STAR root**, with this environment active.

<details>
<summary>Version pins and installation notes</summary>

- `environments/validated-pip-freeze.txt` also pins transitive dependencies. It
  can replace `requirements.txt` after the PyTorch installation above.
- `environments/rl.yaml` is an alternative Conda recipe, not a separately tested
  solver result. If Python 3.8 is absent, provision it before running the venv command.
- Keep the editable checkout: simulator XML/meshes and configs are repository
  resources, not standalone-wheel resources.
- Do not install a second dm_control/Robosuite or RL-ViGen environment over this
  one. STAR's retained forks include benchmark-specific behavior.
- Python 3.8 and Gym are legacy dependencies retained for compatibility, not a
  recommendation for a new general-purpose software stack.

</details>

## 2. Download benchmark assets

The public [RL-ViGen repository](https://github.com/gemcollector/RL-ViGen) already
contains DMC colors/videos and Robosuite textures. Download its asset directories
at the verified revision—**no original experiment directory is needed**:

```bash
mkdir -p datasets
git clone --filter=blob:none --no-checkout https://github.com/gemcollector/RL-ViGen.git datasets/RL-ViGen
git -C datasets/RL-ViGen sparse-checkout init --cone
git -C datasets/RL-ViGen sparse-checkout set \
  envs/DMCVGB/data \
  third_party/robosuite/robosuite/models/assets/textures \
  envs/robosuiteVGB/robosuitevgb/assets/video
git -C datasets/RL-ViGen checkout --detach 90d8b8c40acb63af6f938c1f4cc79a0cfee7d7ec

export STAR_DMC_ASSET_ROOT="$PWD/datasets/RL-ViGen/envs/DMCVGB/data"
export STAR_TEXTURE_ROOT="$PWD/datasets/RL-ViGen/third_party/robosuite/robosuite/models/assets/textures"
export STAR_ROBOSUITE_VIDEO_ROOT="$PWD/datasets/RL-ViGen/envs/robosuiteVGB/robosuitevgb/assets/video"
python scripts/check_assets.py --group dmc --group textures --group robosuite_hard
```

Expected: **168 verified files, no errors**. A full clone at the same revision
works too. Use RL-ViGen as an asset source only; **do not run its install scripts**.
Downloaded data stay under Git-ignored `datasets/`.

> **Robosuite hard needs videos.** The environment factory enables backgrounds
> for hard mode even though the base YAML says `video_background: false`.
> Both training stages also construct the hard evaluation environment, so set
> the video path before training. See [asset details](ASSETS.md).

## 3. Run a short end-to-end check

Before launching a long experiment, verify the complete path:

```bash
python scripts/smoke_pipeline.py --run-name smoke_dmc
```

This uses Cheetah and geometry augmentation, so it does **not** need Places365.
It collects fresh replay, performs **10 teacher updates and 2 student updates**,
reloads the saved models, and runs **one complete evaluation episode**.

| Output | Location |
| --- | --- |
| Teacher checkpoint and replay | `logs/smoke_dmc_teacher/` |
| Student checkpoint | `logs/smoke_dmc_student/model/snapshot.pt` |
| Logs and pass/fail receipt | `outputs/smoke_dmc/` |

Use a new `--run-name` when rerunning; existing outputs are protected.
After preparing Places365 below, check the appearance path or Robosuite with:

```bash
python scripts/smoke_pipeline.py --run-name smoke_appearance --overlay-type random
python scripts/smoke_pipeline.py --task Door --run-name smoke_door --overlay-type random
python scripts/smoke_pipeline.py --task TwoArmPegInHole --run-name smoke_twoarm --overlay-type random
```

These intentionally tiny runs are **execution checks, not benchmark results**.
See [what was actually tested](REPRODUCIBILITY.md).

## 4. Train STAR

### Download Places365 for Stage 2 appearance adaptation

Random-overlay alignment uses the Places365 easy-format archive linked by
[RL-ViGen](https://github.com/gemcollector/RL-ViGen#extra-datasets). It is about
**26.7 GB / 24.9 GiB**, plus extraction space. Respect the
[dataset provider's terms](http://places2.csail.mit.edu/download.html).

```bash
mkdir -p datasets/places
wget -c https://data.csail.mit.edu/places/places365/places365standard_easyformat.tar \
  -P datasets/places
tar -xf datasets/places/places365standard_easyformat.tar -C datasets/places
export PLACES365_ROOT="$PWD/datasets/places"
test -d "$PLACES365_ROOT/places365_standard/train"
```

Required layout: `PLACES365_ROOT/places365_standard/train/<class>/*.jpg`.
Geometry-only alignment and standalone evaluation do not need Places365.
Re-export the asset paths in each new shell; [asset setup](ASSETS.md)
explains how to keep them in a local shell file.

### Stage 1 → Stage 2

Stage 1 collects **1,000 random-action steps** (`num_seed_frames=1000`), then
makes **one learner update per environment step**, including at `init_steps=1000`.
There is **no initial 1,000-update burst and no gradient clipping**. Actor and
target update frequencies remain 2; a learner update need not update the actor.
This intentionally changes the historical initial-update schedule. It is not a
claim that all historical training trajectories or scores have been reproduced.

Example: **Walker Walk, seed 300**. Stage 2 consumes the teacher and merged replay
created by Stage 1:

```bash
# Stage 1: teacher and replay
python train_stage1.py --task_name walker_walk --seed 300 \
  --exp_name walker_teacher_s300

# Stage 2: appearance-robust encoder/MLP, frozen teacher
python train_stage2.py --task_name walker_walk --seed 300 \
  --exp_name walker_star_s300 \
  --teacher_work_dir logs/walker_teacher_s300/model/snapshot_2000000.pt \
  --buffer_dir logs/walker_teacher_s300/buffer
```

For geometry alignment, add `--overlay_type random_rotate_shift` to Stage 2.
Inspect defaults without a GPU using `python train_stage1.py --print-config`
or `python train_stage2.py --print-config`.

| Task | `--env` | `--action_repeat` | `--repr_dim` | `--dyna_alpha` |
| --- | --- | ---: | ---: | ---: |
| `walker_walk`, `cheetah_run` | `dmc` | 4 | 100 | 5 |
| `finger_spin` | `dmc` | 2 | 100 | 5 |
| `cartpole_swingup` | `dmc` | 8 | 100 | 5 |
| `Door`, `TwoArmPegInHole` | `robosuite` | 1 | 256 | 1 |

Apply task overrides to **both stages**. Robosuite Stage 2 also needs
`--expected_teacher_dyna_alpha 1`. For example:

<details>
<summary>Complete Door training commands</summary>

```bash
python train_stage1.py --env robosuite --task_name Door --seed 300 \
  --config configs/rl/stage1_door.yaml --exp_name door_teacher_s300
python train_stage2.py --env robosuite --task_name Door --seed 300 \
  --action_repeat 1 --repr_dim 256 --dyna_alpha 1 --expected_teacher_dyna_alpha 1 \
  --exp_name door_star_s300 \
  --teacher_work_dir logs/door_teacher_s300/model/snapshot_500000.pt \
  --buffer_dir logs/door_teacher_s300/buffer
```

</details>

The Door preset uses batch 256, actor/critic/dynamics LR `5e-5`, temperature LR
`1e-4`, forward-loss weight 1, representation dimension 256, LayerNorm and two
replay workers. It skips in-training evaluation and runs a fixed 500k-step budget.
The four-seed study used the same learner settings, but stopped successful runs
early and deferred replay merging; see the [study and its limits](REPRODUCIBILITY.md).
For TwoArmPegInHole, override `--task_name TwoArmPegInHole` in the Door command
and use distinct experiment names for both stages. Performance of the new
single-update schedule has not been verified for that task or for DMC.

**Step counts matter.** Stage 1 defaults to 500,000 **environment steps**; its
checkpoint suffix is `steps × action_repeat`. Stage 2 defaults to 100,000
**optimizer updates**. It copies merged replay before invoking the legacy loader,
so budget extra disk space. Use a unique experiment name for every run.
W&B is disabled by the setup above; online logging requires an explicit choice.

### Keep large outputs on another disk

Both stages save under `logs/<exp_name>`. The trainer protects existing run
directories, so **do not pre-create its exact experiment directory**. Instead
link a new parent namespace on the larger disk and use a nested experiment name:

```bash
mkdir -p /path/to/large-disk/star-runs/new-study logs
ln -s /path/to/large-disk/star-runs/new-study logs/new-study
python train_stage1.py --config configs/rl/stage1_door.yaml \
  --exp_name new-study/door_teacher_s300
```

Use the resulting `logs/new-study/door_teacher_s300/` paths for Stage 2. Choose
a new namespace if the link or destination already exists; do not replace old runs.
Replay merging and Stage 2's protected replay copy require additional disk space.

## 5. Evaluate

Evaluate the student produced above, using matching task/action-repeat settings:

```bash
# DMC: ten full video-hard episodes
python -m star.evaluation.rl --env dmc --task walker_walk \
  --checkpoint logs/walker_star_s300/model/snapshot_100000.pt \
  --mode video_hard --action-repeat 4 --seed 1 --episodes 10 \
  --output-dir outputs/walker_video_hard

# Robosuite: thirty medium episodes, scenes selected automatically
python -m star.evaluation.rl --env robosuite --task Door \
  --checkpoint logs/door_star_s300/model/snapshot_100000.pt \
  --mode medium --action-repeat 1 --seed 1 --episodes 30 \
  --output-dir outputs/door_medium
```

Set `--episodes` to the number of evaluations you want (default: 30); **no scene
argument is needed or accepted**. Robosuite automatically cycles through its ten
scenes: 10 episodes cover each once, 30 cover each three times. With other counts,
the same cycle continues for exactly the requested number of episodes.
Terminal progress shows `Episode 1/30`, return and completion status, not scene IDs.

Results are written to `<output-dir>/summary.json` with returns, seeds and
episode-completion flags. Automatically chosen scene IDs remain in that file for
reproducibility only. Episodes use `seed + episode`. For DMC geometry, select `rotate_easy`, `rotate_hard`, `shift_easy`
or `shift_hard` and the corresponding geometry-trained student.

`--max-steps` truncates episodes for debugging; do not use it for benchmark scores.
Only load trusted checkpoints: PyTorch's legacy pickle format can execute code.

## Validation and troubleshooting

- [Diffusion Policy guide](DIFFUSION_POLICY.md): independent environment,
  public demonstrations, Base → STAR training, and episode-count evaluation.
  Its historical visual scenes are outcome-selected, not an untouched test set.
- [Reproducibility report](REPRODUCIBILITY.md): clean-install checks,
  checkpoint comparisons and explicit limits. **Full retraining to paper scores
  has not been verified.** Historical checkpoints can have different augmentation
  settings; the defaults are recipes, not a claim to reproduce every old score.
- [Asset guide](ASSETS.md): dataset layouts, hash checks and optional backgrounds.
- `python scripts/check_release.py`: check for missing runtime sources, accidental
  data/weights, private paths and common secret patterns.

| Symptom | Check |
| --- | --- |
| CUDA unavailable / EGL initialization fails | NVIDIA driver, GPU access, system EGL libraries, `MUJOCO_GL=egl` |
| Ignored EGL exception after a completed Robosuite trainer run | Check the exit code and checkpoint; the current bounded check records a process-teardown warning in [verification](REPRODUCIBILITY.md#release-checks-on-2026-10-17) |
| Missing DMC assets, Robosuite textures or hard videos | Export all three paths from step 2; run the 168-file hash check |
| Places images cannot be found | Set `PLACES365_ROOT` to the parent of `places365_standard` |
| Existing run/output directory | Choose a fresh `--exp_name`, `--run-name` or `--output-dir` |
| Replay is too short for Stage 2 | Use the smoke script or collect enough complete episodes; legacy merging drops 1,000 rows |

Core learners: `star/rl/algos/`; entrypoints: `star/training/`, `star/evaluation/`;
presets: `configs/rl/` and `configs/diffusion/`; RL simulator forks: `vendor/`;
Diffusion Policy modules: `diffusion_policy/`. Legacy `dad`/`dad2`/`DADPolicy`
names remain for STAR CLI/checkpoint compatibility.

## Acknowledgements and licensing

Built on DAD, DrQ-v2/RL-ViGen, Diffusion Policy, Robomimic, dm_control,
dmc2gym, Robosuite and SECANT.
Original notices are retained in `LICENSE`, `licenses/` and vendor trees; see
[third-party notices](../THIRD_PARTY_NOTICES.md). Datasets, custom media and trained
weights are not redistributed here or covered by a new ownership claim.
