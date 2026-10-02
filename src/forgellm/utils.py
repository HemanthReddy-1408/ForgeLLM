"""Small shared helpers: seeding, devices, JSONL IO, atomic writes, timing."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16,
          "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def to_dtype(name: str | torch.dtype) -> torch.dtype:
    return name if isinstance(name, torch.dtype) else DTYPES[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_device_cache(device: torch.device) -> None:
    """Release cached accelerator buffers. On Apple MPS this is also the recovery for the allocator-state bug that makes a
    backward pass return NaN gradients for a batch that is fine after the cache is cleared (see Trainer._train_step)."""
    import gc

    gc.collect()
    sync(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def device_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return 0.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with atomic_write(path) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


@contextlib.contextmanager
def atomic_write(path: str | Path, mode: str = "w") -> Iterator[Any]:
    """Write to a temp file in the same directory, then rename — readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as fh:
            yield fh
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_json(path: str | Path, obj: Any) -> None:
    with atomic_write(path) as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def stable_hash(*parts: Any, n: int = 12) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def file_hash(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class Timer:
    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.t0


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}P"
