"""Limit how many ranks hold a FULL model copy in host DRAM at the same time.

The problem
-----------
``AutoModelForCausalLM.from_pretrained`` materialises the whole model in the
calling process. Tensor parallelism runs one process per core and sharding only
happens after the model object exists, so the transient host-DRAM peak is
``ranks x model_size`` -- every rank holding a private full copy at once. On a
trn2.48xlarge (2147 GB of DRAM) that is the wall you hit before HBM:

    Qwen3.5-122B-A10B  250 GB x 32 ranks =  8.0 TB   OOMKilled
    DeepSeek-V4-Flash  319 GB x 64 ranks = 20.4 TB   OOMKilled (137)

Raising tp to get more HBM makes this strictly worse. The two budgets pull in
opposite directions, which is why no tp ever worked for those models.

The fix here
------------
Let only ``concurrency`` ranks be inside the load->shard->to(device) window at a
time. Once a rank has sharded and moved its slice to HBM it releases its host
copy, so the peak becomes::

    concurrency x model  +  (ranks - concurrency) x model/ranks

For 122B at 32 ranks with concurrency=2 that is 500 + 234 = 734 GB, comfortably
inside 2147. The cost is wall-clock: loads serialise into
``ceil(ranks/concurrency)`` waves.

Why a directory semaphore and not ``dist.barrier()``
----------------------------------------------------
A barrier would be shorter, but collectives are the documented wedge risk on this
hardware -- a failed device barrier can orphan ranks and leave the box unusable,
and it has to work before the model is even loaded. Every rank here is a process
in ONE pod, so ``/tmp`` is shared and ``os.mkdir`` is an atomic
test-and-set. No collective, nothing to wedge.

Fails OPEN. If the wait times out, or a slot is orphaned by a rank that died, the
caller proceeds and risks an OOM rather than hanging a run forever.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import time

__all__ = [
    "DEFAULT_SLOT_DIR",
    "concurrency_from_env",
    "acquire_load_slot",
    "release_load_slot",
    "staggered_peak_gb",
]

DEFAULT_SLOT_DIR = "/tmp/trn_opt_load_slots"
_POLL_S = 2.0
_TIMEOUT_S = 5400.0        # 90 min: a 64-wave load of a big model is slow but real


def concurrency_from_env(world: int) -> int:
    """How many ranks may load at once. ``world`` (i.e. no staggering) by default.

    Staggering is opt-in because it costs wall-clock on every model, including the
    small ones that never needed it. ``TRN_OPT_LOAD_CONCURRENCY=2`` is the setting
    that makes a 122B-class model loadable here.
    """
    raw = (os.environ.get("TRN_OPT_LOAD_CONCURRENCY") or "").strip()
    if not raw:
        return max(1, world)
    try:
        n = int(raw)
    except ValueError:
        return max(1, world)
    if n <= 0:
        return max(1, world)
    return min(max(1, n), max(1, world))


def staggered_peak_gb(weight_gb: float, ranks: int, concurrency: int) -> float:
    """Peak host DRAM with at most ``concurrency`` full copies resident.

    Ranks outside the window still hold their own shard, which is what makes this
    ``c*W + (ranks-c)*W/ranks`` rather than just ``c*W``.
    """
    ranks = max(1, ranks)
    c = min(max(1, concurrency), ranks)
    return c * weight_gb + (ranks - c) * (weight_gb / ranks)


def _slot_is_stale(slot: pathlib.Path) -> bool:
    """True if the slot's owner process is gone (crashed while holding it)."""
    try:
        pid = int((slot / "pid").read_text().strip())
    except Exception:  # noqa: BLE001 - no pid file means we cannot tell; leave it
        return False
    if pid == os.getpid():
        return False
    # os.kill(pid, 0) rather than a /proc lookup: /proc does not exist on macOS,
    # where every slot would look stale and live slots would be reclaimed.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False        # alive, just not ours
    except OSError:
        return False
    return False


def acquire_load_slot(
    rank: int,
    world: int,
    concurrency: int | None = None,
    *,
    log=None,
    slot_dir: str = DEFAULT_SLOT_DIR,
    poll_s: float = _POLL_S,
    timeout_s: float = _TIMEOUT_S,
) -> pathlib.Path | None:
    """Block until one of ``concurrency`` load slots is free; return it.

    Returns None when staggering is off, so the caller can pass the result
    straight to ``release_load_slot`` unconditionally.
    """
    c = concurrency_from_env(world) if concurrency is None else concurrency
    if c >= max(1, world) or world <= 1:
        return None

    root = pathlib.Path(slot_dir)
    root.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"load staggering: rank {rank} waiting for 1 of {c} load slots "
            f"(world={world}); host DRAM holds at most {c} full model copies")

    t0 = time.time()
    announced = False
    while True:
        for i in range(c):
            slot = root / f"slot{i}"
            try:
                slot.mkdir()
            except FileExistsError:
                if _slot_is_stale(slot):
                    if log:
                        log(f"load staggering: reclaiming stale slot{i} "
                            f"(owner process gone)")
                    try:
                        (slot / "pid").unlink(missing_ok=True)
                        slot.rmdir()
                    except OSError:
                        pass
                continue
            else:
                (slot / "pid").write_text(str(os.getpid()))
                atexit.register(release_load_slot, slot, None)
                if log:
                    log(f"load staggering: rank {rank} holds slot{i} after "
                        f"{time.time() - t0:.0f}s")
                return slot
        if time.time() - t0 > timeout_s:
            if log:
                log(f"load staggering: rank {rank} gave up after {timeout_s:.0f}s "
                    f"and is loading anyway (fail open -- an OOM risk beats a hang)")
            return None
        if not announced and time.time() - t0 > 60:
            announced = True
            if log:
                log(f"load staggering: rank {rank} still queued after 60s "
                    f"(expected -- loads run {c} at a time)")
        time.sleep(poll_s)


def release_load_slot(slot: pathlib.Path | None, log=None) -> None:
    """Release a slot. Safe to call with None, twice, or after atexit already ran."""
    if slot is None:
        return
    try:
        (slot / "pid").unlink(missing_ok=True)
        slot.rmdir()
    except OSError:
        return
    if log:
        log(f"load staggering: released {slot.name}")
