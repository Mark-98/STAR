"""Fetch only Can/Square PH absolute-action members of the official DP archive.

Uses validated HTTP byte ranges, not an 85 GB full-archive download. Standard
library only; existing datasets and partial files are never overwritten.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile

URL = 'https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip'


class RangeReader(io.RawIOBase):
    def __init__(self, url, block_size=8 * 1024**2):
        self.url, self.block_size = url, block_size
        self.position, self.size, self.etag = 0, None, None
        self.cache_start, self.cache = 0, b''
        self.downloaded_bytes = 0
        self._fetch(0, 0)

    def _fetch(self, start, end):
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Match'] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=90) as response:
            expected = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if response.status != 206 or expected is None:
                raise RuntimeError('Server did not honor byte ranges; refusing a full archive download.')
            first, last, total = map(int, expected.groups())
            if (first, last) != (start, end) or (self.size is not None and total != self.size):
                raise RuntimeError('Unexpected range or changed archive size.')
            etag = response.headers.get('ETag')
            if self.etag and etag != self.etag:
                raise RuntimeError('Archive changed during download.')
            payload = response.read(end - start + 2)
            if len(payload) != end - start + 1:
                raise RuntimeError('Incomplete or oversized range response.')
            self.size, self.etag = total, etag
            self.cache_start, self.cache = start, payload
            self.downloaded_bytes += len(payload)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek mode.')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError('Negative seek.')
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if size > 32 * 1024**2:
            raise ValueError('Use streaming reads; refusing an oversized allocation.')
        chunks = []
        while size:
            offset = self.position - self.cache_start
            if not 0 <= offset < len(self.cache):
                self._fetch(self.position, min(self.size - 1, self.position + self.block_size - 1))
                offset = 0
            count = min(size, len(self.cache) - offset)
            chunks.append(self.cache[offset:offset + count])
            self.position += count
            size -= count
        return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['can', 'square'], action='append')
    parser.add_argument('--output-root', type=Path, default=Path('datasets'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    tasks = list(dict.fromkeys(args.task or ['can', 'square']))
    with RangeReader(URL) as remote, zipfile.ZipFile(remote) as archive:
        selected = []
        for task in tasks:
            suffix = f'{task}/ph/image_abs.hdf5'
            matches = [info for info in archive.infolist()
                       if info.filename == suffix or info.filename.endswith('/' + suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one prepared PH image dataset for {task}.')
            info = matches[0]
            if not 0 < info.file_size < 8 * 1024**3:
                raise ValueError('Unexpected dataset size.')
            target = args.output_root / suffix
            partial = target.with_suffix('.hdf5.part')
            if not args.dry_run and (target.exists() or partial.exists()):
                raise FileExistsError(f'Refusing to overwrite {target} or its partial download.')
            selected.append((info, target, partial))
        print(json.dumps({'source': URL, 'archive_bytes': remote.size, 'etag': remote.etag,
                          'files': [{'member': info.filename, 'bytes': info.file_size,
                                     'crc32': f'{info.CRC:08x}', 'target': str(target)}
                                    for info, target, _ in selected]}, indent=2), flush=True)
        if args.dry_run:
            return
        args.output_root.mkdir(parents=True, exist_ok=True)
        required = sum(info.file_size for info, _, _ in selected)
        if shutil.disk_usage(args.output_root).free < required + 512 * 1024**2:
            raise OSError('Insufficient free space for selected datasets plus a 512 MiB reserve.')
        for info, target, partial in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with archive.open(info) as source, partial.open('xb') as output:
                while True:
                    chunk = source.read(1024**2)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            if count != info.file_size:
                raise RuntimeError(f'Incomplete member: {info.filename}; partial retained at {partial}')
            # Link rather than replace: even a concurrent new destination is protected.
            os.link(partial, target)
            partial.unlink()
            print(json.dumps({'saved': str(target), 'bytes': count,
                              'sha256': digest.hexdigest(), 'zip_crc_verified': True,
                              'downloaded_bytes_so_far': remote.downloaded_bytes}), flush=True)


if __name__ == '__main__':
    main()
