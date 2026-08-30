"""Make sure a dead worker's rank processes are gone before the next one starts.

The failure this exists to prevent
----------------------------------
Measurement shells out to ``torchrun -> neuron_worker.py``. ``subprocess.run``
with a ``timeout`` kills only the process it started -- torchrun -- and NOT the
rank processes torchrun spawned. Those orphans keep ``/dev/neuron*`` mapped, so
every later candidate fails the moment it tries to acquire a core.

Observed on a trn2.48xlarge (workload T_d5cd4802): one config
(``attn_implementation=sdpa``) ran into the compile wall at 10800s and was killed.
Every one of the ~20 configs after it then died in about 9 seconds each -- at
tp=1, 2, 4, 16, 32 and 64, at cp=2/4/8, at batch=8/32, and on every compiler
rewrite. Instant failure at *every* tp is not memory pressure; the box was wedged
by the orphans from the timed-out config. The whole search was void and the
trusted grader could not remeasure the baseline (``claimed=6 remeasured=0
drift=100%``), so a run that had a working baseline produced nothing publishable.

One dead-ended candidate must not be able to poison the rest of the run.

Design notes
------------
- The worker is launched in its own session (``start_new_session=True``) so the
  whole tree can be signalled with ``killpg`` without also signalling us.
- SIGTERM first with a grace period, then SIGKILL. A rank killed mid-collective
  can take a moment to release its device.
- After signalling, WAIT until no worker processes remain. Returning early is the
  same bug in a smaller window.
- Linux ``/proc`` is the source of truth for stragglers, but ``proc_root`` is
  injectable so this is testable off-box.
- Everything is best-effort and never raises: failing to reap is bad, but turning
  it into an exception would lose the real failure that triggered the reap.
"""

from __future__ import annotations

import os
import pathlib
import signal
import time

__all__ = [
    "WORKER_MARKERS",
    "kill_process_group",
    "surviving_workers",
    "wait_until_clear",
    "reap",
]

# Substrings that identify a measurement worker in a process command line.
WORKER_MARKERS = ("neuron_worker.py", "torch.distributed.run", "torchrun")

_TERM_GRACE_S = 5.0
_CLEAR_TIMEOUT_S = 120.0
_POLL_S = 2.0


def kill_process_group(proc, log=None, grace_s: float = _TERM_GRACE_S) -> bool:
    """SIGTERM then SIGKILL the worker's entire process group.

    Returns True if a signal was delivered. ``proc`` is a ``Popen``; it must have
    been started with ``start_new_session=True`` or its group is OURS and killing
    it would kill the run.
    """
    if proc is None or proc.poll() is not None:
        return False
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return False
    if pgid == os.getpgid(0):
        # Refuse to signal our own group -- that would take down the optimizer.
        if log:
            log("reap: worker shares our process group; not signalling "
                "(launch it with start_new_session=True)")
        return False
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        try:
            os.killpg(pgid, sig)
            if log:
                log(f"reap: sent {name} to process group {pgid}")
        except ProcessLookupError:
            return True                      # already gone
        except OSError as e:
            if log:
                log(f"reap: {name} to {pgid} failed: {e}")
            return False
        deadline = time.time() + (grace_s if sig == signal.SIGTERM else 2.0)
        while time.time() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(0.2)
    return True


def surviving_workers(
    markers: tuple[str, ...] = WORKER_MARKERS,
    proc_root: str = "/proc",
) -> list[int]:
    """PIDs of worker processes still alive, excluding this process.

    Reads ``/proc/<pid>/cmdline`` because a rank orphaned by a killed torchrun is
    reparented to init and is no longer a child of anything we hold a handle to.
    """
    root = pathlib.Path(proc_root)
    if not root.is_dir():
        return []
    me = os.getpid()
    found: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except (OSError, ValueError):
            continue                          # exited while we looked, or not ours
        if any(m in cmd for m in markers):
            found.append(pid)
    return sorted(found)


def wait_until_clear(
    timeout_s: float = _CLEAR_TIMEOUT_S,
    poll_s: float = _POLL_S,
    log=None,
    markers: tuple[str, ...] = WORKER_MARKERS,
    proc_root: str = "/proc",
) -> bool:
    """Block until no worker processes remain. True if clear, False on timeout."""
    t0 = time.time()
    while True:
        alive = surviving_workers(markers, proc_root)
        if not alive:
            waited = time.time() - t0
            if log and waited > poll_s:
                log(f"reap: devices clear after {waited:.0f}s")
            return True
        if time.time() - t0 > timeout_s:
            if log:
                log(f"reap: {len(alive)} worker process(es) still alive after "
                    f"{timeout_s:.0f}s ({alive[:8]}); the next candidate will "
                    f"probably fail to acquire a core")
            return False
        time.sleep(poll_s)


def reap(
    proc=None,
    log=None,
    timeout_s: float = _CLEAR_TIMEOUT_S,
    markers: tuple[str, ...] = WORKER_MARKERS,
    proc_root: str = "/proc",
) -> str:
    """Kill the worker's group and wait for its devices to be released.

    Returns a short note for the failure_reason so the ledger records whether the
    box was left clean -- if it was not, every later candidate in the run is
    suspect and that needs to be visible rather than inferred.
    """
    try:
        kill_process_group(proc, log=log)
        clear = wait_until_clear(timeout_s=timeout_s, log=log, markers=markers,
                                 proc_root=proc_root)
        if clear:
            return "worker reaped, devices released"
        left = surviving_workers(markers, proc_root)
        return (f"REAP INCOMPLETE: {len(left)} worker process(es) still hold the "
                f"devices; later candidates in this run are unreliable")
    except Exception as e:  # noqa: BLE001 - never mask the failure that caused this
        if log:
            log(f"reap: unexpected error {e!r}")
        return f"reap error {e!r}"
