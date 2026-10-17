# Reproducibility scope

Latest recipe update: **2026-10-17**. Earlier installation checks: **2026-10-11**;
see the [clean-install receipts](installation_check.json) and
[earlier worktree receipts](validation.json).

This page describes the **DMC/Robosuite RL workflow**. The separate Can/Square
imitation-learning implementation has its own [Diffusion Policy validation
report](DIFFUSION_VALIDATION.md); RL installation results do not validate that stack.

## Single-update Stage 1 recipe

The published Stage 1 trainer now collects **1,000 random-action environment
steps**, then calls the learner **once per environment step**, including the
first eligible step. It does **not** perform the historical initial burst of
1,000 learner calls. `init_steps=1000` remains a collection threshold, not a
burst count. Actor and target update frequencies remain 2. Gradient clipping
is not installed or called.

Fresh construction also consumes the historical detached decoder's **CPU
initialization RNG**, as in the controlled study. No decoder is retained or
trained. This preserves that constructor's RNG consumption; it does not assert
equivalence of complete historical online training trajectories.

### Completed four-seed Door study

The study ran on **2026-10-16–17**, using fresh teachers and replay for seeds
**300, 400, 500 and 600**, on one **RTX 3090**, with at most two concurrent jobs.
Its scope is **Door, Stage 1, Original training environment**. Returns are native
training logs, **not deterministic standalone evaluation or shifted-scene scores**.
The report uses the last 20 logged completed episodes, 500 steps per episode.

Settings: batch 256, actor/critic/dynamics LR `5e-5`, temperature LR `1e-4`,
forward-loss weight 1, representation dimension 256, hidden dimension 1024,
encoder LayerNorm, `use_target_enc=false`, two replay workers and one CPU thread.
All four runs used the single-initial-update schedule and **no clipping**.

| Seed | Stopping step | Last-20 mean return | Target reached |
| --- | ---: | ---: | --- |
| 300 | 210,000 | 444.6 | Yes |
| 400 | 125,000 | 444.8 | Yes |
| 500 | 500,000 | 400.0 | No |
| 600 | 500,000 | 93.5 | No |

The diagnostic harness stopped a run when its last-20 mean first reached
**427.5**, checking every 5,000 steps from 20,000, or at the **500,000-step** cap.
These are **adaptively selected endpoints, not equal-budget comparisons**.
Both unsuccessful seeds remain in the report. None of the four runs had nonfinite
logged losses. Actual Adam counters at action steps 1000, 1001, 1002 and 1099
were **1, 2, 3 and 100** for the critic/dynamics and **1, 1, 2 and 50** for the
actor/temperature. Final optimizer counters and checkpoint weights were checked.

**Conclusion: 2/4 seeds met the chosen target. Full multi-seed performance
reproduction is not established, and improved stability is not proven.** This
study does not validate Stage 2, DMC, TwoArmPegInHole or Diffusion Policy scores.

### Published preset versus the study harness

The [Door preset](../configs/rl/stage1_door.yaml) exposes the same learner
settings without private paths or runtime monkey patches. It runs a **fixed
500k-step budget**, without score-based stopping, and merges replay at completion
for Stage 2; the study harness stopped successful runs early and deferred merging.
The table above is evidence from that controlled harness, **not four fresh
full-length runs of the newly published checkout**. Tests and bounded source
comparisons must not be presented as such a full rerun.

Weights, replay, datasets and experiment videos are excluded from this source
release. The recipe changes the historical initial-update schedule intentionally.

### Release checks on 2026-10-17

Checks used an export of the files selected for publication, separate from other
uncommitted work, with the existing isolated Python 3.8 environment on the same
host. This was **not a new dependency installation or full retraining**.

- **34 tests passed; 6 optional Diffusion Policy tests skipped** in the RL environment.
  The new tests execute the native action/update block through its first 100
  updates, check warmup and action-before-update ordering, verify CPU RNG
  consumption and exercise the unclipped native critic/dynamics gradient path.
- **All four seeds** matched the controlled harness's constructor parameters,
  CPU/CUDA RNG and **20 fixed-batch updates** bitwise, including optimizer states.
  This bounded comparison used batch size 2 on the RTX 3090, not a full online run.
- The entire Stage 1 training-loop AST matches the controlled schedule after
  changing only the initial burst expression to `1`.
- Fresh **Cheetah geometry** and **Door appearance** smoke pipelines each verified
  **10 teacher updates, 2 student updates, checkpoint reload and one full evaluation
  episode** on GPU 2. Collection thresholds were 1,250 and 1,500 respectively so
  the inherited replay merge had enough data. Door used the same small public
  Places validation-image fixture described below, not the full training dataset.
- `pip check` passed; **168 public asset hashes** passed; the source-only audit
  found no errors in the files selected for publication.

**Known warning:** the Door trainer processes emitted ignored EGL teardown
exceptions (`EGL_NOT_INITIALIZED`) during process cleanup. Both exited with code
0, saved/reloaded checkpoints and passed the full evaluation episode; the
standalone evaluation log had no such exception. The warning is not hidden or
treated as evidence of performance reproduction.

## Clean-checkout installation and pipeline checks

The installation recheck found a **packaging error** in the earlier publication:
broad `.gitignore` rules excluded `star/datasets/` and two simulator `core.py`
files. Earlier local-worktree tests could still see these files, so they did not
establish that a fresh clone could train. Commit `041b042` includes the missing
runtime sources and narrows the ignore rules. The source audit now also rejects
ignored Python modules, and distribution regression tests check these files.

Robosuite's factory also enables backgrounds for hard mode despite the base
YAML's false flag. The README now sets the video root and verifies the **168
public assets** needed for DMC and Robosuite setup. Neither issue was worked
around by changing benchmark behavior or importing the original project.

The fresh-checkout run uses an isolated Python 3.8 venv, no system-site-packages,
fresh editable installs of all simulator forks, regenerated MuJoCo bindings and
assets downloaded directly from the pinned public RL-ViGen revision. It uses
new teacher/student agents and newly collected replay, not private pretrained
checkpoints or the old replay dataset. Run it with:

```bash
# After README installation and benchmark asset setup; no Places needed:
python scripts/smoke_pipeline.py --run-name smoke_dmc
```

The smoke pipeline collects enough complete episodes to survive the legacy
merge's 1,000-row tail drop, then verifies 10 teacher optimizer steps, 2 student
optimizer steps, checkpoint reload and a full evaluation episode. It is an
execution check only: the resulting tiny policies are not useful trained agents.

Confirmed results are recorded separately in [installation_check.json](installation_check.json):

| Fresh pipeline | Environment steps | Teacher updates | Student updates | Full episode |
| --- | ---: | ---: | ---: | --- |
| Cheetah, geometry alignment | 1,260 | 10 | 2 | Rotate-easy |
| Cheetah, appearance alignment | 1,260 | 10 | 2 | Video-hard |
| Door, appearance alignment | 1,510 | 10 | 2 | Medium + separate hard run |
| TwoArmPegInHole, appearance alignment | 1,510 | 10 | 2 | Medium + separate hard run |

The isolated installation passes `pip check` and all **10 unit tests**. All
168 public benchmark assets match the stored hashes. Every pipeline reloads
its newly saved checkpoints; no old learned model or replay is used.

The full 26.7 GB Places archive plus extracted dataset was not prepared in this
new checkout because of disk limits. Appearance smoke tests use 16 real JPEGs
fetched from the public archive's validation partition, in a clearly separate
test-only ImageFolder. This verifies the normal image-loading/augmentation path,
not the full training distribution. Geometry smoke tests need no Places data.
The operating system, NVIDIA driver and base Python interpreter already existed;
this is a fresh Python/project installation on the same host, not a clean OS.

## Episode-count evaluation API

The current evaluator accepts `--episodes` (default 30), not a manual scene
option. Robosuite cycles through ten scenes in order; seeds remain
`seed + episode_index`. Selected scene IDs are retained only in result metadata.
Console progress uses `Episode 1/N`, return, steps and completion status.

That evaluation-API check passed **14 tests**, including 25 automatic scene/seed
assignments, rejection of nonpositive counts and removal of the manual scene
option. GPU checks ran two full episodes each for Cheetah video-hard, Door
medium and TwoArmPegInHole hard, using the fresh smoke-trained checkpoints.
Both Robosuite runs selected scenes 0 then 1 without user input; all six episodes
completed. Render-context cycles are collected between episodes; final runs
exited without renderer teardown exceptions. This verifies execution, not scores.

## Earlier local-worktree checks (historical)

- Seven unit tests: feature MSE, encoder freezing, clean loss, geometric transform
  determinism, legacy augmentation restoration, agent scope, Robosuite argument
  order and simulator-close traversal (some tests cover several combinations).
- All **12 selected real STAR checkpoints** load without the original project's
  imports. Original-vs-release actions match bitwise on four inputs per model.
  Three controlled optimizer updates, student parameters and CUDA RNG state also
  match bitwise; teacher weights are unchanged. Save/reload and a subsequent
  optimizer step match. The checkpoint regression uses a fixed real Places image
  and synthetic observations; it is not an evaluation-return comparison.
- Fresh Stage-1 construction and two updates using **real replay** match the
  original actor/critic/target/dynamics parameters bitwise with controlled seeds.
  The original detached image-decoder branch is excluded from the public learner;
  its compatibility class remains for old pickle files. Thus an identical entire
  historical RNG trajectory is not asserted for a new full training run.
- Fresh Stage-2 construction and two updates in each of `random` and
  `random_rotate_shift` modes succeed on real replay. Random overlay exercises the
  actual Places ImageFolder/DataLoader, not a synthetic replacement. Teacher
  weights remain unchanged; saved checkpoints reload with matching actions.
- The actual Stage-1 CLI completes **260 environment steps**, replay collection,
  **10 DrQ updates** and a final checkpoint. The actual Stage-2 CLI completes two updates
  with four replay workers, protected replay copies and a final checkpoint.
- **38 full evaluation episodes**: four DMC tasks × eight visual/geometry modes,
  plus Door/TwoArmPegInHole × easy/medium/hard. Fixed evaluation seed 1 and the
  previously selected task/model/scene mapping are recorded in `validation.json`.
  An additional two-episode DMC run verifies the updated cleanup path.
- The full 38-condition sweep and 12-checkpoint regressions used a venv sharing
  the existing Python 3.8.20 /
  PyTorch 1.12.1+cu116 runtime's dependency packages. STAR and all five retained
  simulator packages resolve inside this release checkout, not the old project.
- A separate **isolated venv without system-site-packages** installs the pinned
  requirements and CUDA PyTorch wheels from scratch. MuJoCo bindings regenerate,
  `pip check` finds no broken requirements, and all seven unit tests pass.
  Cheetah video-hard and TwoArmPegInHole medium/scene 6 complete full episodes.
  Fresh Stage-1 and both Stage-2 variants also pass the real-replay update,
  teacher-freezing and save/reload checks in this environment. The resolved
  dependency versions are in `environments/validated-pip-freeze.txt`. This is a
  clean Python environment on the same GPU host, not a fresh operating system.
- Source audit rejects secret patterns, author absolute paths, symlinks, learned
  weights, datasets, videos, database files and oversized files. This is a
  heuristic audit, not a guarantee against every possible credential format.

## Important limits

The historical checks above are **bounded implementation/execution checks**.
The newer Door study above trains teachers only; a fresh full-benchmark 500k-step
teacher + 100k-update student study and reproduction of paper averages are not established.
Standalone evaluation uses `seed + episode`, unlike sequential resets in the
historical training evaluator. Historical checkpoints may have different/missing
augmentation metadata. Do not treat these single episodes as paper benchmark
statistics or compare them to a different scene-selection protocol.

The inherited environment's `pip check` reports an unrelated MoviePy/decorator
mismatch; those packages are not part of the core STAR dependency recipe.
Cross-machine/driver/version determinism, full training convergence and all
unlisted environments are not established here.

## Rerunning bounded checks

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/check_assets.py --group dmc --group textures --group robosuite_hard
python scripts/verify_checkpoint.py \
  --checkpoint /path/to/trusted/star.pt \
  --original-source /path/to/original/algos/dad2.py \
  --places-image /path/to/places/image.jpg --output outputs/regression.json
python scripts/smoke_training.py \
  --teacher /path/to/trusted/teacher.pt --replay-file /path/to/merged/chunk.npz \
  --original-stage1 /path/to/original/algos/dad.py --output-dir outputs/training_smoke
```

To repeat the bounded native Stage-1 CLI check (not a training recipe):

```bash
python train_stage1.py --seed 600 --exp_name native_stage1_smoke \
  --num_train_frames 260 --num_seed_frames 250 --init_steps 250 --batch_size 2 \
  --replay_buffer_num_workers 0 --buffer_merge_on_finish false \
  --skip_eval true --wandb_mode disabled
```

Here `num_seed_frames` retains the legacy environment-step interpretation.
The first 250 steps fill replay; steps 250–259 each make one optimizer update.
No offset between the collection and update thresholds is needed in the
single-update trainer.

Only the optional original-vs-release regression scripts require the trusted old
source as an explicit comparison input. Normal training/evaluation does not.

The source-only release intentionally excludes pretrained weights, replay,
Places365, custom textures and background videos. A fresh clone needs these
authorized external assets before benchmark reproduction.
