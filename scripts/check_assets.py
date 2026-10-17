"""Verify externally supplied benchmark assets against the validation manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['dmc', 'textures', 'robosuite_hard', 'robosuite_video'], action='append')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'docs/asset_hashes.json').read_text())
    keys = dict(dmc='STAR_DMC_ASSET_ROOT', textures='STAR_TEXTURE_ROOT',
                robosuite_hard='STAR_ROBOSUITE_VIDEO_ROOT', robosuite_video='STAR_ROBOSUITE_VIDEO_ROOT')
    errors = []
    checked = 0
    for group in args.group or ('dmc', 'textures', 'robosuite_video'):
        base = os.environ.get(keys[group])
        if not base:
            errors.append('Set ' + keys[group])
            continue
        rows = manifest['robosuite_video'] if group == 'robosuite_hard' else manifest[group]
        if group == 'robosuite_hard':
            rows = [row for row in rows if row['file'].startswith('eval-hard/')]
        for row in rows:
            path = Path(base) / row['file']
            if not path.is_file():
                errors.append(group + '/' + row['file'] + ': missing')
            elif sha256(path) != row['sha256']:
                errors.append(group + '/' + row['file'] + ': hash mismatch')
            else:
                checked += 1
    print(json.dumps(dict(verified_files=checked, errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
