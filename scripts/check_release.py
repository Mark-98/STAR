"""Audit publishable Git files for private paths, secrets, artifacts and syntax."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = sorted(set(p for p in result.decode().split('\0') if p))
    forbidden = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.hdf5', '.mp4', '.zip', '.sqlite3', '.pyc', '.so'}
    secrets = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}')
    # A URL such as example.org/data/training/ is not an author's absolute path.
    author_path = re.compile(r'(?<![A-Za-z0-9/])/(?:home|data|Users)/[A-Za-z0-9_.-]+/')
    errors = []
    # An ignored Python source can work locally but disappear from every clone.
    for folder in ('star', 'diffusion_policy', 'vendor', 'scripts', 'tests'):
        for source in (ROOT / folder).rglob('*.py'):
            relative = source.relative_to(ROOT).as_posix()
            if relative not in paths:
                errors.append([relative, 'Python source excluded from release'])
    for required in ('star/datasets/replay.py',
                     'diffusion_policy/policy/diffusion_unet_hybrid_image_dad_policy.py',
                     'vendor/dm_control/dm_control/mujoco/wrapper/core.py',
                     'vendor/robosuitevgb/robosuitevgb/secant/envs/robosuite/core.py'):
        if required not in paths:
            errors.append([required, 'required runtime source missing'])
    total = 0
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink():
            errors.append([relative, 'symlink'])
            continue
        if not path.is_file():
            errors.append([relative, 'missing'])
            continue
        size = path.stat().st_size
        total += size
        if path.suffix in forbidden or size >= 50 * 1024**2:
            errors.append([relative, 'artifact or large file'])
        if path.name in {'secret_key', 'field_encryption_key'}:
            errors.append([relative, 'private key filename'])
        if path.suffix in {'.py', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.cfg', '.xml', '.sh'}:
            text = path.read_text(errors='replace')
            if secrets.search(text):
                errors.append([relative, 'credential pattern'])
            if author_path.search(text):
                errors.append([relative, 'author absolute path'])
            if path.suffix == '.py':
                try:
                    ast.parse(text)
                except SyntaxError as exc:
                    errors.append([relative, 'syntax: ' + str(exc)])
    report = dict(files=len(paths), source_bytes=total, errors=errors,
                  scope='Heuristic source audit; not a guarantee that every possible secret format is detected.')
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
