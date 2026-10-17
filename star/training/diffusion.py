"""Hydra entrypoint for the retained Robomimic image-policy workspaces."""

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'diffusion'),
    config_name='train_diffusion_unet_hybrid_workspace',
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    if 'teacher_checkpoint_path' in cfg.policy:
        if not cfg.policy.teacher_checkpoint_path:
            raise ValueError('Stage 2 requires pretrained_checkpoint=<trusted base checkpoint>.')
        from diffusion_policy.policy.diffusion_unet_hybrid_image_dad_policy import Places365Overlay
        if cfg.policy.overlay_mode == 'places365' and not Places365Overlay(
                root=cfg.policy.overlay_root, dataset=cfg.policy.overlay_dataset).available():
            raise FileNotFoundError('Set PLACES365_ROOT or policy.overlay_root; refusing silent batch-overlay fallback.')

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
