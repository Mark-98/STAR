# Public asset preparation

Follow the [download commands in the README](../README.md#2-download-benchmark-assets)
from a fresh STAR checkout. No author-specific filesystem paths, original
experiment folders or private media are needed by the supplied task presets.
Use the original benchmark resources and respect each provider's terms; do not
replace missing backgrounds or textures with generated images for comparisons.

## 1. Obtain the benchmark files from RL-ViGen

The README downloads selected directories from the public
[RL-ViGen repository](https://github.com/gemcollector/RL-ViGen/tree/90d8b8c40acb63af6f938c1f4cc79a0cfee7d7ec).
The pinned revision is `90d8b8c40acb63af6f938c1f4cc79a0cfee7d7ec`.
These are ordinary Git-tracked assets, not a separate private archive. A full
clone can be used instead of the sparse checkout; the paths are the same.
Do not install RL-ViGen's Python packages: use STAR's installation instructions.

| Asset | Path inside the RL-ViGen checkout | Needed for |
| --- | --- | --- |
| DMC colors and videos | `envs/DMCVGB/data` | DMC appearance conditions; native DMC training setup |
| Robosuite custom textures | `third_party/robosuite/robosuite/models/assets/textures` | Robosuite visual scenes |
| Robosuite background videos | `envs/robosuiteVGB/robosuitevgb/assets/video` | Hard evaluation and native Robosuite training setup |

The DMC folder contains `color_easy.pt`, `color_hard.pt`, ten
`video_easy/video0.mp4` ... `video9.mp4` files and one hundred
`video_hard/video0.mp4` ... `video99.mp4` files. The `.pt` files are color tables,
not trained neural-network weights. The texture folder includes `custom01.png`
... `custom40.png`, alongside upstream standard textures.

## 2. Set paths to the downloaded files

Run from the STAR repository root, in the shell used for training or evaluation:

```bash
export STAR_DMC_ASSET_ROOT="$PWD/datasets/RL-ViGen/envs/DMCVGB/data"
export STAR_TEXTURE_ROOT="$PWD/datasets/RL-ViGen/third_party/robosuite/robosuite/models/assets/textures"
export STAR_ROBOSUITE_VIDEO_ROOT="$PWD/datasets/RL-ViGen/envs/robosuiteVGB/robosuitevgb/assets/video"
# After downloading/extracting Places365 as described in the README:
export PLACES365_ROOT="$PWD/datasets/places"
```

These variables point to downloaded directories; they do not download or copy
files. If using a full RL-ViGen checkout elsewhere, substitute that checkout's
location. The same downloaded assets can be shared across runs.

To avoid retyping exports, save the lines in a local shell file such as
`assets.local.sh` (Git-ignored). Run `source assets.local.sh` from the STAR root
at the start of each shell. `$PWD` is resolved when that file is sourced.

## 3. Check the required assets

With the STAR Python environment active:

```bash
python scripts/check_assets.py --group dmc --group textures --group robosuite_hard
```

Expected output: `verified_files: 168`, `errors: []`. For DMC only, use
`--group dmc` (112 files). For Robosuite, use `--group textures --group robosuite_hard`
(40 textures and 16 upstream hard-background videos, including scenes 0–9).
Hashes are recorded in [asset_hashes.json](asset_hashes.json). Missing files mean
the download/layout or path variable needs attention; a hash mismatch means
different asset bytes, not something to fix by bypassing the check.

On 2026-10-11 the README's sparse-clone procedure was executed into a new
directory: all 168 files downloaded from upstream passed the hash check.
This check used the fresh public checkout, not the original experiment's assets.

The manifest also preserves a historical Robosuite video inventory. Calling the
checker without `--group` requests **all 220 historical files**, including local
`train/` and `train_2/` videos absent from the public upstream revision. This is
not the fresh-install check and is not required by the released presets.

## Robosuite hard backgrounds and optional extra modes

**Set `STAR_ROBOSUITE_VIDEO_ROOT` for Robosuite training and hard evaluation.**
The base YAML has `video_background: false`, but the environment factory in
`vendor/robosuitevgb/robosuitevgb/utils.py` overrides it to `true` for `eval-hard`.
Both training stages construct a hard evaluation environment during setup,
including short runs. Clean/easy/medium standalone evaluation does not enable
background videos. This corrects the earlier documentation's YAML-only reading;
the original benchmark behavior has not been changed.

The upstream revision supplies `eval-easy/`, `eval-hard/` and `eval-extreme/`
background directories, but no `train/`, `train_2/` or `eval-medium/` directory.
The normal hard preset uses the publicly available `eval-hard/` directory:

```bash
export STAR_ROBOSUITE_VIDEO_ROOT="$PWD/datasets/RL-ViGen/envs/robosuiteVGB/robosuitevgb/assets/video"
```

The wrapper reads `<root>/<mode>/video<scene_id>.mp4`. Setting the root alone
does not enable backgrounds in other modes, make missing modes available or add an
`eval-extreme` mode to STAR's evaluator. Do not rename `eval-hard` to `eval-medium`
or synthesize substitute videos and call it the same benchmark. Additional
background modes require an explicit, documented asset/protocol choice.

## Places365: separate image download

Follow [README: Places365](../README.md#download-places365-for-stage-2-appearance-adaptation).
RL-ViGen links the provider's `places365standard_easyformat.tar`; cloning
RL-ViGen does **not** download Places365. Extract the archive so that the layout is:

```text
datasets/places/
└── places365_standard/
    └── train/
        ├── <class-1>/*.jpg
        └── <class-2>/*.jpg
```

`PLACES365_ROOT` is the parent of `places365_standard`, not the archive path or
the `train` directory. This dataset is used for Stage-2 random-overlay learning,
not standalone evaluation of a trained agent or geometry-only alignment.
Consult the [official distribution](http://places2.csail.mit.edu/download.html)
and [Places365 project](https://github.com/CSAILVision/places365) for usage terms
and availability. The full Places archive was not downloaded/extracted in the
clean-install check because of disk limits. Its HTTPS URL returned HTTP 200 with
a 26,686,689,280-byte archive. A bounded 64 MiB range was fetched on 2026-10-11,
and 16 real JPEGs from the archive's validation partition were placed in a
separate test-only ImageFolder to exercise augmentation and I/O. This fixture
is not the training dataset or a valid substitute for benchmark training.

## Replay and checkpoints: generated by STAR

These do not come from RL-ViGen's asset folders. Stage 1 writes replay under
`logs/<run>/buffer/`; its end-of-run merge produces `buffer/merged/`.
Stage 2 expects that merged replay and validates headers and worker capacity
when `require_full_replay=true`. Manual merge, if needed:

```bash
python -c "from star.rl.read_buffer.split_buffer import merge_npz; merge_npz('/path/to/buffer', '/path/to/new/merged')"
```

Keep original replay immutable. Stage 2 makes its own working copy before using
the legacy replay loader. No learned weights, replay, Places images or downloaded
RL-ViGen media are redistributed in STAR; retain their original terms and record
asset/checkpoint hashes, seeds and scene IDs when reporting results.
