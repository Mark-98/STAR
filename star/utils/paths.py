"""Repository-relative resources; external data are configured explicitly."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resource(*parts):
    return ROOT.joinpath(*parts)


def texture_root():
    return Path(os.environ.get('STAR_TEXTURE_ROOT', resource('assets', 'private', 'textures'))).expanduser()


def dmc_asset_root():
    return Path(os.environ.get('STAR_DMC_ASSET_ROOT', resource('assets', 'dmc'))).expanduser()


def places_config(key=None):
    root = os.environ.get('PLACES365_ROOT')
    if not root:
        raise ValueError('Set PLACES365_ROOT to your Places365 ImageFolder directory.')
    data = {'datasets': [str(Path(root).expanduser())]}
    return data if key is None else data[key]
