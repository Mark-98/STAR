"""Protect immutable offline data from the legacy reader's file-eviction policy."""
import shutil
from pathlib import Path


def validate_full_replay(source, cfg):
    """Check headers and worker capacities without materializing image arrays.

    'Full' means all legacy-valid sampling ranges, not an exhaustive epoch and not
    restoration of transitions discarded when this merged dataset was prepared.
    """
    import zipfile
    from numpy.lib import format as npformat
    filters = ['index', 'start', 'end', 'tail_count', 'start_ratio', 'end_ratio', 'tail_ratio']
    for suffix in filters:
        value = getattr(cfg, 'replay_dataset_' + suffix, -1)
        if value is not None and value >= 0:
            raise ValueError('Full replay disallows --replay_dataset_' + suffix)
    source = Path(source).expanduser().resolve()
    source = source if source.name == 'merged' else source / 'merged'
    files = sorted(source.glob('*.npz'))
    if not files:
        raise FileNotFoundError('Full replay requires prepared merged/*.npz files.')
    workers = max(1, cfg.replay_buffer_num_workers)
    capacity = cfg.replay_buffer_size // workers
    worker_sizes = [0] * workers
    rows = []
    expected_shape = (cfg.frame_stack * 3, 84, 84)
    action_shape = None
    for path in files:
        try:
            _, index, declared_length = path.stem.rsplit('_', 2)
            index, declared_length = int(index), int(declared_length)
        except ValueError:
            raise ValueError('Replay loader cannot parse filename: ' + path.name)
        shapes, dtypes = {}, {}
        with zipfile.ZipFile(path) as archive:
            for field in ('observation', 'action', 'reward', 'discount'):
                with archive.open(field + '.npy') as stream:
                    version = npformat.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = npformat.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = npformat.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported NPY header version: ' + str(version))
                    if dtype.hasobject:
                        raise ValueError('Object arrays are not valid replay tensors.')
                    shapes[field], dtypes[field] = tuple(shape), str(dtype)
        length = shapes['observation'][0]
        # Historical files may encode array length or episode length (minus dummy).
        if declared_length not in (length, length - 1) or length <= 5:
            raise ValueError('Invalid declared length or empty sampling range: ' + path.name)
        if any(shape[0] != length for shape in shapes.values()):
            raise ValueError('Replay fields have different lengths: ' + path.name)
        if shapes['observation'][1:] != expected_shape or dtypes['observation'] != 'uint8':
            raise ValueError('Expected uint8 frame-stacked 84x84 observations: ' + path.name)
        if len(shapes['action']) != 2 or shapes['action'][1] < 1:
            raise ValueError('Expected (T, action_dim) actions: ' + path.name)
        if action_shape is not None and shapes['action'][1:] != action_shape:
            raise ValueError('Action dimensions differ across chunks.')
        action_shape = shapes['action'][1:]
        if shapes['reward'][1:] != (1,) or shapes['discount'][1:] != (1,):
            raise ValueError('Expected (T, 1) reward and discount arrays.')
        # Mirror _try_fetch's filename-based capacity check, conservatively.
        worker = index % workers
        worker_sizes[worker] += max(declared_length - 4, length - 5)
        rows.append({'file': path.name, 'worker': worker, 'array_rows': length,
                     'sample_index_start': 3, 'sample_index_end_exclusive': length - 2,
                     'eligible_samples': length - 5,
                     'observation_bytes': length * expected_shape[0] * 84 * 84,
                     'shapes': shapes, 'dtypes': dtypes})
    if any(size == 0 for size in worker_sizes):
        raise ValueError('Empty replay worker partition; adjust worker count or chunk indices.')
    if any(size > capacity for size in worker_sizes):
        raise ValueError('Replay capacity would omit chunks; increase replay_buffer_size. '
                         f'Worker requirements={worker_sizes}, capacity={capacity}')
    return {'sampling': 'random_with_replacement_over_legacy_valid_ranges',
            'all_chunks_fit': True, 'worker_capacity': capacity, 'worker_requirements': worker_sizes,
            'total_eligible_samples': sum(r['eligible_samples'] for r in rows),
            'uncompressed_observation_bytes': sum(r['observation_bytes'] for r in rows),
            'header_only_validation': True, 'files': rows}


def stage_replay(source, destination):
    source, destination = Path(source).expanduser().resolve(), Path(destination).resolve()
    source = source if source.name == 'merged' else source / 'merged'
    files = sorted(source.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No merged replay files found in {source}. See docs/ASSETS.md.')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Replay source and work directory must be separate trees.')
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.copyfile(path, destination / path.name)
    return destination
