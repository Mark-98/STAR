import glob
import numpy as np
from pathlib import Path
from datetime import datetime


def merge_npz(buffer_dir, out_dir):
    """
     .npz  merged  1-pass 4 chunk split .
    -   " 1000(=1 episode)"
    - dummy  ( / )
    - out_dir .npz
    - merged_path  (   )
    """
    num_workers = 4
    drop_tail = 1000  # ✅  1000

    buffer_dir = Path(buffer_dir)
    out_dir = Path(out_dir)

    # ✅ out_dir (.npz)
    if out_dir.exists() and any(out_dir.glob("*.npz")):
        print(f"⏭️  Skip: out_dir already has npz files → {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # =========================
    # 1)   /
    # =========================
    file_paths = sorted(glob.glob(str(buffer_dir / "*.npz")))
    if not file_paths:
        raise FileNotFoundError(f"npz    : {buffer_dir}")

    #   keys + empty chunk dtype/shape
    with np.load(file_paths[0], allow_pickle=True) as first:
        keys = list(first.files)
        templates = {}
        for k in keys:
            arr0 = first[k]
            templates[k] = {
                "dtype": arr0.dtype,
                "tail_shape": arr0.shape[1:],  # (T, D) D
            }

    #    +
    lengths = []
    for path in file_paths:
        with np.load(path, allow_pickle=True) as d:
            missing = set(keys) - set(d.files)
            if missing:
                raise KeyError(f" {path}   : {missing}")
            lengths.append(int(d[keys[0]].shape[0]))

    total_len = int(sum(lengths))
    effective_len = total_len - drop_tail  # ✅  1000

    if effective_len <= 0:
        raise ValueError(
            f"effective_len 0 . total_len={total_len}, drop_tail={drop_tail}"
        )

    # =========================
    # 2)   0..effective_len-1  4
    # =========================
    indices = np.array_split(np.arange(0, effective_len), num_workers)

    #   [start, end)  (slice )
    ranges = []
    for idx_arr in indices:
        if idx_arr.size == 0:
            ranges.append((0, 0))
        else:
            ranges.append((int(idx_arr[0]), int(idx_arr[-1]) + 1))

    chunk_buf = [{k: [] for k in keys} for _ in range(num_workers)]

    # =========================
    # 3) 1-pass    chunk
    # =========================
    offset = 0
    for path, L in zip(file_paths, lengths):
        file_start = offset
        file_end = offset + L  # [file_start, file_end)

        # ✅ effective_len  ,   effective_len
        if file_start >= effective_len:
            break
        capped_file_end = min(file_end, effective_len)

        with np.load(path, allow_pickle=True) as d:
            for ci, (gs, ge) in enumerate(ranges):
                # chunk [gs, ge)  file [file_start, capped_file_end)
                s = max(gs, file_start)
                e = min(ge, capped_file_end)
                if s >= e:
                    continue

                ls = s - file_start
                le = e - file_start

                for k in keys:
                    chunk_buf[ci][k].append(d[k][ls:le])

        offset += L

    # =========================
    # 4) chunk  (dummy prepend )
    # =========================
    for eps_idx in range(num_workers):
        episode = {}
        for k in keys:
            parts = chunk_buf[eps_idx][k]
            if parts:
                episode[k] = np.concatenate(parts, axis=0)
            else:
                #     chunk  " "
                tail_shape = templates[k]["tail_shape"]
                dtype = templates[k]["dtype"]
                episode[k] = np.empty((0,) + tail_shape, dtype=dtype)

        chunk_len = episode[keys[0]].shape[0]
        fn = f"{ts}_{eps_idx}_{chunk_len}.npz"
        out_path = out_dir / fn

        np.savez_compressed(out_path, **episode)
        print(f"[SPLIT] Saved chunk {eps_idx}: {out_path.name}, length={chunk_len}")

    print(f"🎉   ( {drop_tail} , dummy , 1-pass)")
