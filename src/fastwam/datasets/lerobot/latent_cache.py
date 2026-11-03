"""Validated on-disk cache for frozen VAE latents."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

LATENTS_FILE = "latents_u8.bin"
DONE_FILE = "done.bin"
META_FILE = "meta.json"


def fingerprint(dataset) -> str:
    payload = {
        "dataset_dirs": [str(path) for path in getattr(dataset, "dataset_dirs", []) or []],
        "num_frames": int(getattr(dataset, "num_frames", -1)),
        "video_size": [int(value) for value in (getattr(dataset, "video_size", None) or [])],
        "action_video_freq_ratio": int(getattr(dataset, "action_video_freq_ratio", -1)),
        "concat_multi_camera": str(getattr(dataset, "concat_multi_camera", "")),
        "global_sample_stride": int(getattr(dataset, "global_sample_stride", -1)),
        "skip_padding_as_possible": bool(
            getattr(dataset, "skip_padding_as_possible", False)
        ),
        "n": int(len(dataset)),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_dir(root: str, fingerprint_value: str) -> str:
    return os.path.join(str(root), fingerprint_value)


def write_meta(directory: str, n: int, latent_shape, fingerprint_value: str, extra=None):
    item_bytes = int(np.prod(latent_shape)) * 2
    metadata = {
        "n": int(n),
        "latent_shape": [int(value) for value in latent_shape],
        "dtype": "bfloat16_raw_u8",
        "item_bytes": item_bytes,
        "fingerprint": fingerprint_value,
    }
    if extra:
        metadata.update(extra)
    with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def read_meta(directory: str) -> dict:
    with open(os.path.join(directory, META_FILE), "r", encoding="utf-8") as handle:
        return json.load(handle)


def open_cache(directory: str, expect_n: int, expect_fingerprint: str, mode: str = "r"):
    metadata = read_meta(directory)
    if int(metadata["n"]) != int(expect_n):
        raise ValueError(
            f"latent cache entry count {metadata['n']} != dataset length {expect_n}"
        )
    if str(metadata.get("fingerprint", "")) != str(expect_fingerprint):
        raise ValueError(
            f"latent cache fingerprint {metadata.get('fingerprint')} != "
            f"current fingerprint {expect_fingerprint}"
        )
    n = int(metadata["n"])
    item_bytes = int(metadata["item_bytes"])
    latents = np.memmap(
        os.path.join(directory, LATENTS_FILE),
        dtype=np.uint8,
        mode=mode,
        shape=(n, item_bytes),
    )
    done = np.memmap(
        os.path.join(directory, DONE_FILE),
        dtype=np.uint8,
        mode=mode,
        shape=(n,),
    )
    return latents, done, metadata


def status(root: str, dataset) -> dict:
    fingerprint_value = fingerprint(dataset)
    directory = cache_dir(root, fingerprint_value)
    result = {
        "fingerprint": fingerprint_value,
        "dir": directory,
        "n": int(len(dataset)),
        "exists": False,
        "ready": False,
        "done": 0,
        "missing": int(len(dataset)),
    }
    if not os.path.isfile(os.path.join(directory, META_FILE)):
        return result
    result["exists"] = True
    try:
        _, done, _ = open_cache(directory, len(dataset), fingerprint_value, mode="r")
    except (ValueError, OSError) as error:
        result["error"] = str(error)
        return result
    result["done"] = int(np.count_nonzero(done))
    result["missing"] = int(len(dataset)) - result["done"]
    result["ready"] = result["missing"] == 0
    return result
