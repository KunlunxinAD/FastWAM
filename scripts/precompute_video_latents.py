#!/usr/bin/env python3
"""Precompute VAE latents for the training dataset."""

import argparse
import fcntl
import os
import pathlib
import time

import numpy as np
import torch


class IndexedVideoOnly(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # Encode the requested index, even when it contains padding.
        self.dataset.max_padding_retry = 0
        self.dataset.latent_cache_root = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        sample = self.dataset._get(index)
        return {"idx": index, "video": sample["video"]}


def build_dataset(task, stats_path, video_backend):
    from hydra import compose, initialize_config_dir
    from fastwam.runtime import build_datasets
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str(pathlib.Path.cwd() / "configs"), version_base="1.3"
    ):
        config = compose(
            config_name="train",
            overrides=[
                f"task={task}",
                f"+data.train.video_backend={video_backend}",
                f"+data.train.pretrained_norm_stats={stats_path}",
            ],
        )
    dataset, _ = build_datasets(config.data)
    from fastwam.datasets.lerobot import latent_cache

    return dataset, latent_cache.fingerprint(dataset)


def load_vae():
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints")
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

    _, _, vae_config, _ = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    vae_config.download_if_necessary()
    return _load_registered_model(
        vae_config.path, "wan_video_vae", torch_dtype=torch.bfloat16, device="cuda"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/latent_cache")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--stats", default="")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-budget", type=float, default=0.0)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    stats_path = args.stats or str(
        pathlib.Path.cwd() / "data/libero_mujoco3.3.2/dataset_stats.json"
    )
    dataset, fingerprint_value = build_dataset(
        args.task, stats_path, args.video_backend
    )
    from fastwam.datasets.lerobot import latent_cache

    n = len(dataset)
    latent_shape = (
        48,
        (len(dataset.video_sample_indices) - 1) // 4 + 1,
        int(dataset.video_size[0]) // 16,
        int(dataset.video_size[1]) // 16,
    )
    item_bytes = int(np.prod(latent_shape)) * 2
    if args.check:
        state = latent_cache.status(args.root, dataset)
        print(f"[check] fingerprint={state['fingerprint']} dir={state['dir']}")
        print(
            f"[check] n={state['n']} done={state['done']} "
            f"missing={state['missing']} exists={state['exists']} ready={state['ready']}"
        )
        if "error" in state:
            print(f"[check] cache mismatch: {state['error']}")
            raise SystemExit(4)
        raise SystemExit(0 if state["ready"] else 3)

    output = pathlib.Path(latent_cache.cache_dir(args.root, fingerprint_value))
    output.mkdir(parents=True, exist_ok=True)
    with open(output / ".init.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata_path = output / latent_cache.META_FILE
        if not metadata_path.exists():
            latent_cache.write_meta(
                str(output), n, latent_shape, fingerprint_value,
                extra={"task": args.task, "video_backend": args.video_backend},
            )
        for path, shape in (
            (output / latent_cache.LATENTS_FILE, (n, item_bytes)),
            (output / latent_cache.DONE_FILE, (n,)),
        ):
            if not path.exists():
                np.memmap(path, dtype=np.uint8, mode="w+", shape=shape).flush()
    latents, done, _ = latent_cache.open_cache(
        str(output), n, fingerprint_value, mode="r+"
    )
    # Assign ownership before filtering, so concurrent writes cannot shift ranks.
    owned = np.arange(rank, n, world)
    todo = owned[done[owned] == 0]
    if args.limit:
        todo = todo[:args.limit]
    if len(todo) == 0:
        return

    vae = load_vae()
    loader = torch.utils.data.DataLoader(
        IndexedVideoOnly(dataset, todo),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    started = time.time()
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            video = batch["video"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            encoded = vae.model.encode(video, vae.scale)
            raw = encoded.contiguous().view(torch.uint8).cpu().numpy()
            raw = raw.reshape(video.shape[0], -1)
            if tuple(encoded.shape[1:]) != latent_shape:
                raise ValueError(f"Unexpected VAE latent shape: {tuple(encoded.shape)}")
            for offset, index in enumerate(batch["idx"].tolist()):
                latents[index] = raw[offset]
            latents.flush()
            for index in batch["idx"].tolist():
                done[index] = 1
            done.flush()
            seen += video.shape[0]
            if batch_index % 20 == 0:
                elapsed = time.time() - started
                print(
                    f"[rank{rank}] {batch_index} batches, "
                    f"{seen / max(elapsed, 1e-9):.1f} samples/s",
                    flush=True,
                )
            if args.time_budget and time.time() - started > args.time_budget:
                break
    latents.flush()
    done.flush()


if __name__ == "__main__":
    main()
