# replay_buffer.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import io
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


def _debug_replay(message):
    if os.environ.get("DAD_DEBUG_REPLAY", "0").lower() in {"1", "true", "yes", "on"}:
        print(f"[REPLAY DEBUG] {message}", flush=True)


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn: Path):
    """
    Atomic save to avoid DataLoader workers reading a partially-written file.
    Writes to a tmp file then os.replace() to commit.
    """
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)

        tmp = fn.with_name(
            fn.name + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}"
        )
        with tmp.open("wb") as f:
            f.write(bs.read())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fn)


def load_episode(fn: Path):
    with fn.open("rb") as f:
        ep = np.load(f)
        return {k: ep[k] for k in ep.keys()}


def _episode_index_and_len(fn: Path):
    parts = fn.stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except Exception:
        return None


def _resolve_replay_sample_bounds(
    eps_len: int,
    start=None,
    end=None,
    tail_count=None,
    start_ratio=None,
    end_ratio=None,
    tail_ratio=None,
):
    sample_start = 3
    sample_end = max(sample_start, eps_len - 1)

    if start_ratio is not None:
        sample_start = max(sample_start, int(np.floor(eps_len * start_ratio)))
    if end_ratio is not None:
        sample_end = min(sample_end, int(np.ceil(eps_len * end_ratio)))

    if start is not None:
        sample_start = max(sample_start, start)
    if end is not None:
        sample_end = min(sample_end, end)

    if tail_ratio is not None:
        window = max(0, sample_end - sample_start)
        count = max(1, int(np.ceil(window * tail_ratio)))
        sample_start = max(sample_start, sample_end - count)
    if tail_count is not None:
        sample_start = max(sample_start, sample_end - tail_count)

    return sample_start, sample_end


def _optional_nonnegative_int(name, value, include_zero=True):
    if value is None or value < 0:
        return None
    if value == 0 and not include_zero:
        raise ValueError(f"{name} must be positive or -1 to disable")
    return int(value)


def _optional_ratio(name, value, include_zero=False):
    if value is None or value < 0:
        return None
    lower_ok = value >= 0.0 if include_zero else value > 0.0
    if not lower_ok or value > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if include_zero else '(0, 1]'} or -1 to disable")
    return float(value)


class ReplayBufferStorage:
    """
    Writer-side storage.
    If you want "evict from disk when evicted from replay",
    keep max_transitions=None (Storage prune OFF).
    """
    def __init__(self, data_specs, replay_dir: Path, max_transitions: int = None):
        self._data_specs = data_specs
        self._replay_dir = Path(replay_dir)
        self._max_transitions = max_transitions
        self._replay_dir.mkdir(exist_ok=True, parents=True)

        self._current_episode = defaultdict(list)
        self._preload()
        self._prune_if_needed()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

        if time_step.last():
            episode = {}
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._episode_fns = sorted(self._replay_dir.glob("*.npz"))
        self._num_transitions = 0
        max_eps_idx = -1

        for fn in self._episode_fns:
            parsed = _episode_index_and_len(fn)
            if parsed is None:
                continue
            eps_idx, eps_len = parsed
            max_eps_idx = max(max_eps_idx, eps_idx)
            self._num_transitions += eps_len

        self._num_episodes = max_eps_idx + 1  # next eps_idx

    def _prune_if_needed(self):
        if self._max_transitions is None:
            return

        while self._num_transitions > self._max_transitions and self._episode_fns:
            old = self._episode_fns.pop(0)
            try:
                _, _, eps_len = old.stem.split("_")
                eps_len = int(eps_len)
            except Exception:
                eps_len = 0
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass
            self._num_transitions -= eps_len

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1

        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_path = self._replay_dir / f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, eps_path)

        self._episode_fns.append(eps_path)
        self._episode_fns.sort()
        self._num_transitions += eps_len
        self._prune_if_needed()


class ReplayBuffer(IterableDataset):
    """
    Reader-side replay used by DataLoader workers.

    Semantics:
      - buffer_save=False: delete each .npz after it has been loaded (no disk history).
      - buffer_save=True: keep on disk until evicted from the replay window;
                          when evicted to satisfy max_size, delete file too.

    IMPORTANT:
      If you use buffer_save=True and want disk contents to exactly match the
      current replay window, do NOT enable Storage-side pruning (max_transitions=None).
    """
    def __init__(self, replay_dir: Path, max_size: int, num_workers: int,
                 nstep: int, discount: float, fetch_every: int,
                 save_snapshot: bool, buffer_save: bool, use_q_adv =False,
                 reward_scale=1.0, replay_dataset_index=None,
                 replay_dataset_start=None, replay_dataset_end=None,
                 replay_dataset_tail_count=None,
                 replay_dataset_start_ratio=None, replay_dataset_end_ratio=None,
                 replay_dataset_tail_ratio=None):
        self._replay_dir = Path(replay_dir)
        self._size = 0
        self._max_size = max_size
        self.use_q_adv = use_q_adv
        self.reward_scale = reward_scale
        self._num_workers = max(1, num_workers)
        self._replay_dataset_index = replay_dataset_index
        self._replay_dataset_start = replay_dataset_start
        self._replay_dataset_end = replay_dataset_end
        self._replay_dataset_tail_count = replay_dataset_tail_count
        self._replay_dataset_start_ratio = replay_dataset_start_ratio
        self._replay_dataset_end_ratio = replay_dataset_end_ratio
        self._replay_dataset_tail_ratio = replay_dataset_tail_ratio
        self._episode_fns = []
        self._episodes = {}

        self._nstep = nstep
        self._discount = discount

        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every

        self._buffer_save = buffer_save
        _ = save_snapshot  # kept for API compatibility

    def _sample_bounds_from_len(self, eps_len):
        return _resolve_replay_sample_bounds(
            eps_len,
            start=self._replay_dataset_start,
            end=self._replay_dataset_end,
            tail_count=self._replay_dataset_tail_count,
            start_ratio=self._replay_dataset_start_ratio,
            end_ratio=self._replay_dataset_end_ratio,
            tail_ratio=self._replay_dataset_tail_ratio,
        )

    def _sample_bounds(self, episode):
        return self._sample_bounds_from_len(episode_len(episode))

    def _effective_episode_len_from_len(self, eps_len):
        start, end = self._sample_bounds_from_len(eps_len)
        return max(0, end - start)

    def _effective_episode_len(self, episode):
        return self._effective_episode_len_from_len(episode_len(episode))

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn: Path) -> bool:
        try:
            t0 = time.time()
            _debug_replay(f"load start pid={os.getpid()} file={eps_fn}")
            episode = load_episode(eps_fn)
            _debug_replay(
                f"load done pid={os.getpid()} file={eps_fn.name} "
                f"dt={time.time() - t0:.2f}s keys={list(episode.keys())}"
            )
        except FileNotFoundError:
            # could be deleted by eviction / cleanup
            return True
        except Exception:
            # corrupt / transient read error: skip it
            return False

        eps_len = self._effective_episode_len(episode)
        if eps_len <= 0:
            return True

        # evict from memory to satisfy this worker's max_size
        while eps_len + self._size > self._max_size:
            if not self._episode_fns:
                # episode itself cannot fit; drop it
                if self._buffer_save:
                    eps_fn.unlink(missing_ok=True)
                return True

            old_fn = self._episode_fns.pop(0)
            old_ep = self._episodes.pop(old_fn)
            self._size -= self._effective_episode_len(old_ep)

            # evict -> delete from disk if we want disk == active replay
            if self._buffer_save:
                old_fn.unlink(missing_ok=True)

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        # if we don't want disk history, delete immediately after load
        if not self._buffer_save:
            eps_fn.unlink(missing_ok=True)

        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0

        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0

        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        total_files = len(eps_fns)
        if self._replay_dataset_index is not None:
            indexed_eps_fns = []
            for eps_fn in eps_fns:
                parsed = _episode_index_and_len(eps_fn)
                if parsed is not None and parsed[0] == self._replay_dataset_index:
                    indexed_eps_fns.append(eps_fn)
            eps_fns = indexed_eps_fns
        _debug_replay(
            f"try_fetch pid={os.getpid()} worker_id={worker_id} "
            f"num_workers={self._num_workers} files={len(eps_fns)}/{total_files} "
            f"replay_dataset_index={self._replay_dataset_index} dir={self._replay_dir}"
        )
        fetched_size = 0

        for eps_fn in eps_fns:
            parsed = _episode_index_and_len(eps_fn)
            if parsed is None:
                continue
            eps_idx, eps_len = parsed

            if self._replay_dataset_index is None and eps_idx % self._num_workers != worker_id:
                continue

            if eps_fn in self._episodes:
                break

            effective_eps_len = self._effective_episode_len_from_len(eps_len)
            if effective_eps_len <= 0:
                continue

            if fetched_size + effective_eps_len > self._max_size:
                break

            fetched_size += effective_eps_len

            ok = self._store_episode(eps_fn)
            if not ok:
                # don't break on one bad file
                continue

    def _sample(self):
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()

        self._samples_since_last_fetch += 1

        # Startup guard: before any episode file exists, wait a bit
        while not self._episode_fns:
            self._samples_since_last_fetch = self._fetch_every  # force fetch
            self._try_fetch()
            if not self._episode_fns:
                time.sleep(0.01)

        episode = self._sample_episode()

        max_start = episode_len(episode)
        idx_start, idx_end = self._sample_bounds_from_len(max_start)
        if max_start <= 0 or idx_end <= idx_start:
            # episode too short; drop from this worker's memory and resample
            self._episodes.pop(self._episode_fns.pop(0), None)
            return self._sample()

        idx = np.random.randint(idx_start, idx_end)

        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        next_action = episode['action'][idx+1]
        reward = np.zeros_like(episode["reward"][idx])
        disc = np.ones_like(episode["discount"][idx])

        step_reward = episode["reward"][idx]
        reward += disc * step_reward * self.reward_scale
        disc *= episode["discount"][idx] * self._discount

        if self.use_q_adv:
            return (obs, action, reward, disc, next_obs, next_action)
        return (obs, action, reward, disc, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id: int):
    # Deterministic per-worker seeds (based on torch DataLoader initial seed)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, buffer_save=True, use_q_adv=True,
                       reward_scale=1.0, replay_dataset_index=-1,
                       replay_dataset_start=-1, replay_dataset_end=-1,
                       replay_dataset_tail_count=-1,
                       replay_dataset_start_ratio=-1.0,
                       replay_dataset_end_ratio=-1.0,
                       replay_dataset_tail_ratio=-1.0):
    """
    Keeps the original call signature used in your Workspace:
      make_replay_loader(dir, replay_buffer_size, batch_size, num_workers,
                         save_snapshot, nstep, discount, buffer_save)
    """
    replay_dir = Path(replay_dir)
    if replay_dataset_index is not None and replay_dataset_index < 0:
        replay_dataset_index = None
    replay_dataset_start = _optional_nonnegative_int("replay_dataset_start", replay_dataset_start)
    replay_dataset_end = _optional_nonnegative_int("replay_dataset_end", replay_dataset_end)
    replay_dataset_tail_count = _optional_nonnegative_int(
        "replay_dataset_tail_count", replay_dataset_tail_count, include_zero=False
    )
    replay_dataset_start_ratio = _optional_ratio(
        "replay_dataset_start_ratio", replay_dataset_start_ratio, include_zero=True
    )
    replay_dataset_end_ratio = _optional_ratio(
        "replay_dataset_end_ratio", replay_dataset_end_ratio, include_zero=True
    )
    replay_dataset_tail_ratio = _optional_ratio(
        "replay_dataset_tail_ratio", replay_dataset_tail_ratio
    )
    if (
        replay_dataset_start is not None
        and replay_dataset_end is not None
        and replay_dataset_end <= replay_dataset_start
    ):
        raise ValueError("replay_dataset_end must be greater than replay_dataset_start")
    if (
        replay_dataset_start_ratio is not None
        and replay_dataset_end_ratio is not None
        and replay_dataset_end_ratio <= replay_dataset_start_ratio
    ):
        raise ValueError("replay_dataset_end_ratio must be greater than replay_dataset_start_ratio")

    replay_slice_enabled = any(
        value is not None
        for value in (
            replay_dataset_start,
            replay_dataset_end,
            replay_dataset_tail_count,
            replay_dataset_start_ratio,
            replay_dataset_end_ratio,
            replay_dataset_tail_ratio,
        )
    )

    if replay_dataset_index is not None:
        parsed_files = [
            (fn, _episode_index_and_len(fn))
            for fn in sorted(replay_dir.glob("*.npz"))
        ]
        available_indices = sorted(
            {parsed[0] for _, parsed in parsed_files if parsed is not None}
        )
        selected_files = [
            fn for fn, parsed in parsed_files
            if parsed is not None and parsed[0] == replay_dataset_index
        ]
        if not selected_files:
            raise FileNotFoundError(
                f"replay_dataset_index={replay_dataset_index} matched no .npz files "
                f"in {replay_dir}. Available indices: {available_indices}"
            )
        selected_sample_counts = []
        for fn, parsed in parsed_files:
            if parsed is None or parsed[0] != replay_dataset_index:
                continue
            sample_start, sample_end = _resolve_replay_sample_bounds(
                parsed[1],
                start=replay_dataset_start,
                end=replay_dataset_end,
                tail_count=replay_dataset_tail_count,
                start_ratio=replay_dataset_start_ratio,
                end_ratio=replay_dataset_end_ratio,
                tail_ratio=replay_dataset_tail_ratio,
            )
            selected_sample_counts.append((fn, parsed[1], sample_start, sample_end))
        if not any(sample_end > sample_start for _, _, sample_start, sample_end in selected_sample_counts):
            details = ", ".join(
                f"{fn.name}: eps_len={eps_len}, range=[{sample_start}, {sample_end})"
                for fn, eps_len, sample_start, sample_end in selected_sample_counts[:4]
            )
            raise ValueError(
                "Replay dataset slice leaves no sampleable transitions. "
                f"Details: {details}"
            )
        if num_workers > 1:
            print(
                f"[ReplayBuffer] replay_dataset_index={replay_dataset_index} uses "
                "num_workers=1 to avoid empty worker partitions."
            )
            num_workers = 1
        print(
            f"[ReplayBuffer] replay_dataset_index={replay_dataset_index} "
            f"selected {len(selected_files)} file(s): "
            f"{', '.join(fn.name for fn in selected_files[:4])}"
        )
        if replay_slice_enabled:
            total_effective = sum(
                max(0, sample_end - sample_start)
                for _, _, sample_start, sample_end in selected_sample_counts
            )
            preview = ", ".join(
                f"{fn.name}: eps_len={eps_len}, sample_idx=[{sample_start}, {sample_end}) "
                f"count={max(0, sample_end - sample_start)}"
                for fn, eps_len, sample_start, sample_end in selected_sample_counts[:4]
            )
            print(
                "[ReplayBuffer] replay dataset slice enabled: "
                f"effective_sample_count={total_effective}; {preview}"
            )

    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir=replay_dir,
        max_size=max_size_per_worker,
        num_workers=num_workers,
        nstep=nstep,
        discount=discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        buffer_save=buffer_save,
        use_q_adv=use_q_adv,
        reward_scale=reward_scale,
        replay_dataset_index=replay_dataset_index,
        replay_dataset_start=replay_dataset_start,
        replay_dataset_end=replay_dataset_end,
        replay_dataset_tail_count=replay_dataset_tail_count,
        replay_dataset_start_ratio=replay_dataset_start_ratio,
        replay_dataset_end_ratio=replay_dataset_end_ratio,
        replay_dataset_tail_ratio=replay_dataset_tail_ratio,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(
            worker_init_fn=_worker_init_fn,
            multiprocessing_context=torch.multiprocessing.get_context("spawn"),
            persistent_workers=True,      #
            prefetch_factor=2,            # I/O   ( 1)
        )

    loader = torch.utils.data.DataLoader(iterable, **loader_kwargs)
    return loader
