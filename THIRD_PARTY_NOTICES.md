# Third-party notices

This is a derivative source release. Original author identities and per-file
headers are retained; the repository name does not replace their notices.

| Component | Retained location | Original notice/source |
| --- | --- | --- |
| DAD / STAR predecessor | `star/rl/` | `licenses/DAD-MIT.txt` |
| Diffusion Policy | `diffusion_policy/` (retained simulation/image-policy subset) | [upstream](https://github.com/real-stanford/diffusion_policy), `licenses/diffusion-policy-MIT.txt` |
| Robomimic / Robosuite 1.2 fork | External DP environment dependencies, not the RL vendor tree | [Robomimic](https://github.com/ARISE-Initiative/robomimic/tree/v0.2.0), [pinned Robosuite fork](https://github.com/cheng-chi/robosuite/tree/277ab9588ad7a4f4b55cf75508b44aa67ec171f0) |
| DrQ-v2 / RL-ViGen | RL workspaces, DMCVGB, RobosuiteVGB | [upstream license](https://github.com/gemcollector/RL-ViGen/blob/main/LICENSE), `licenses/RL-ViGen-MIT.txt` |
| Robosuite 1.4 | `vendor/robosuite/` | [v1.4.0 license](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/LICENSE), `licenses/robosuite-MIT.txt` |
| SECANT | `vendor/robosuitevgb/robosuitevgb/secant/` | [upstream license](https://github.com/DrJimFan/SECANT/blob/main/LICENSE), `licenses/SECANT-MIT.txt` |
| dm_control | `vendor/dm_control/` | Apache-2.0 in its original `LICENSE` and file headers |
| dmc2gym | `vendor/dmc2gym/` | Its original `LICENSE` |
| Nested dm_control assets | `vendor/dm_control/dm_control/third_party/` | Original per-component licenses retained |

Upstream Robosuite, SECANT and RL-ViGen notices were checked on 2026-10-11.
Local environment forks retain behavioral changes needed by the historical
benchmark; they must not be described as unmodified upstream installations.
The Robosuite notice also identifies Apache-2.0 MuJoCo code; the Apache license
text is retained with dm_control.

Simulator XML and standard mesh resources are runtime dependencies, not learned
weights. Custom visual textures, color tables, background videos, Places365,
replay and learned checkpoints are external and deliberately excluded. Users
must obtain those resources under the original providers' terms; no new rights
in those datasets or media are asserted here.
