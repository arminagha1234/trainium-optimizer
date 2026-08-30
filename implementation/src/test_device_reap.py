"""Reaping + failure-classification tests.

Both halves of one real incident (trn2.48xlarge, workload T_d5cd4802):

  1. A candidate hit the compile wall at 10800s. Killing it killed torchrun but
     not its ranks, so the orphans kept /dev/neuron* and every one of the ~20
     candidates after it died in ~9s -- at tp=1/2/4/16/32/64, cp=2/4/8,
     batch=8/32, and every compiler rewrite. The search was void and the grader
     could not remeasure the baseline (claimed=6 remeasured=0 drift=100%).

  2. Every one of those failures was reported as "OOM / HBM pressure", because the
     signature "HBM" matched `NEURON_RT_MAP_HBM=1` -- a line from the runtime's
     env dump, printed alongside ANY error. The diagnosis pointed at memory when
     the box was simply wedged.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from backends.device_reap import (
    WORKER_MARKERS,
    kill_process_group,
    reap,
    surviving_workers,
    wait_until_clear,
)
from backends.native_pytorch import _is_noise, _worker_failure_reason


# --- the misclassification ---------------------------------------------------

# Verbatim from the run log.
_INFODUMP = ("2026-Aug-30 00:11:31.312687 2671273:2671273 ERROR   NRT:nrt_infodump"
             "                                NEURON_RT_MAP_HBM=1")


def test_the_runtime_env_dump_is_never_the_diagnosis():
    """`NEURON_RT_MAP_HBM=1` is configuration, not a symptom."""
    assert _is_noise(_INFODUMP)


def test_a_wedged_box_is_not_reported_as_an_oom():
    """The whole point: 'OOM' sends you to shrink the model, which cannot work."""
    tail = (
        "[rank2]: ERROR  NRT:nrt_allocate_neuron_cores  NERR_RESOURCE: "
        "failed to allocate 8 cores\n"
        + _INFODUMP
    )
    reason = _worker_failure_reason(1, tail)
    assert "device busy" in reason
    assert "OOM" not in reason


def test_a_genuine_oom_is_still_reported_as_one():
    """The fix must not blind the classifier to real memory pressure."""
    tail = "[rank0]: RuntimeError: Out of memory: tried to allocate 12.0 GB of HBM"
    assert "OOM" in _worker_failure_reason(1, tail)


def test_an_infodump_alone_does_not_become_an_oom_verdict():
    """A tail of nothing but env dump must not masquerade as a memory failure."""
    assert "OOM" not in _worker_failure_reason(1, _INFODUMP + "\n" + _INFODUMP)


def test_an_unreaped_box_is_classified_as_device_busy():
    """When no rank got a word in, the reap note itself is the best diagnosis.

    (When a rank DID speak, _worker_failure_reason restricts itself to "[rankN]:"
    lines, so measure() appends the note to the reason instead of the tail.)
    """
    reason = _worker_failure_reason(
        1, "REAP INCOMPLETE: 8 worker process(es) still hold the devices; "
           "later candidates in this run are unreliable")
    assert "device busy" in reason
    assert "unreliable" in reason


# --- reaping, against real processes -----------------------------------------

def _spawn_group(n_children: int = 2):
    """A shell in its own session that spawns children, like torchrun does."""
    script = (f"for i in $(seq {n_children}); do sleep 120 & done; "
              f"echo neuron_worker.py-stub; wait")
    return subprocess.Popen(["bash", "-c", script],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_killing_the_group_takes_the_children_too():
    """subprocess.run's timeout kills only the leader -- this is the difference."""
    proc = _spawn_group(2)
    pgid = os.getpgid(proc.pid)
    time.sleep(0.5)
    assert kill_process_group(proc) is True
    proc.wait(timeout=10)
    time.sleep(0.5)
    # Whole group gone: signalling it again finds nothing.
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_killing_the_leader_alone_leaves_orphans():
    """Pins the bug itself, so a regression to proc.kill() fails here."""
    proc = _spawn_group(2)
    pgid = os.getpgid(proc.pid)
    time.sleep(0.5)
    proc.kill()                     # what subprocess.run(timeout=) does
    proc.wait(timeout=10)
    time.sleep(0.5)
    os.killpg(pgid, signal.SIGTERM)  # children still there: does NOT raise
    time.sleep(0.2)


def test_we_refuse_to_signal_our_own_process_group():
    """A worker launched without start_new_session would take the optimizer down."""
    logged = []
    same_group = subprocess.Popen(["sleep", "30"])   # no start_new_session
    try:
        assert kill_process_group(same_group, log=logged.append) is False
        assert logged and "not signalling" in logged[0]
    finally:
        same_group.kill()
        same_group.wait(timeout=10)


def test_an_already_dead_worker_is_a_no_op():
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait(timeout=10)
    assert kill_process_group(proc) is False
    assert kill_process_group(None) is False


# --- straggler detection (injectable /proc so it runs off-box) ----------------

def _fake_proc(tmp_path, pids_and_cmds):
    for pid, cmd in pids_and_cmds.items():
        d = tmp_path / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    return str(tmp_path)


def test_orphaned_ranks_are_found_by_cmdline(tmp_path):
    """An orphan is reparented to init, so a Popen handle cannot find it."""
    root = _fake_proc(tmp_path, {
        11: "python /src/backends/neuron_worker.py --tp 8",
        12: "python /src/backends/neuron_worker.py --tp 8",
        13: "/usr/bin/sshd -D",
    })
    assert surviving_workers(proc_root=root) == [11, 12]


def test_our_own_pid_is_never_reported(tmp_path):
    root = _fake_proc(tmp_path, {os.getpid(): "python neuron_worker.py"})
    assert surviving_workers(proc_root=root) == []


def test_torchrun_itself_counts_as_a_worker(tmp_path):
    root = _fake_proc(tmp_path, {21: "python -m torch.distributed.run --nproc 8"})
    assert surviving_workers(proc_root=root) == [21]
    assert "torchrun" in WORKER_MARKERS


def test_a_clear_box_returns_immediately(tmp_path):
    (tmp_path / "notapid").mkdir()
    assert wait_until_clear(timeout_s=0.5, poll_s=0.05,
                           proc_root=str(tmp_path)) is True


def test_stragglers_that_never_die_time_out_and_say_so(tmp_path):
    root = _fake_proc(tmp_path, {31: "python neuron_worker.py"})
    logged = []
    assert wait_until_clear(timeout_s=0.2, poll_s=0.05, log=logged.append,
                            proc_root=root) is False
    assert logged and "still alive" in logged[0]


def test_reap_reports_whether_the_box_was_left_clean(tmp_path):
    dirty = _fake_proc(tmp_path / "dirty", {41: "python neuron_worker.py"})
    note = reap(None, timeout_s=0.2, proc_root=dirty)
    assert note.startswith("REAP INCOMPLETE")
    assert "unreliable" in note

    (tmp_path / "clean").mkdir()
    assert reap(None, timeout_s=0.2,
                proc_root=str(tmp_path / "clean")) == "worker reaped, devices released"


def test_reap_never_raises_even_on_a_broken_proc_root():
    assert isinstance(reap(None, timeout_s=0.1, proc_root="/nonexistent"), str)


# --- per-candidate compile budget --------------------------------------------
#
# The 10800s baseline budget was correct: the Qwen3.5-35B-A3B baseline genuinely
# took 56 minutes. Handing that same budget to every search candidate is what let
# one dead end (attn_implementation=sdpa) spend 3 hours and then, via stranded
# ranks, take the ~20 candidates after it down with it.

def _backend():
    from backends.native_pytorch import NativePyTorchBackend
    return NativePyTorchBackend(instance_type="trn2.48xlarge")


def test_the_baseline_keeps_the_full_budget():
    """If the baseline is cut short the model is skipped entirely, not degraded."""
    from backends import native_pytorch as npt

    b = _backend()
    art = b.build_baseline("Qwen/Qwen3.5-35B-A3B")
    assert b._config_timeout_s(art.config) == npt._COMPILE_TIMEOUT_S


def test_a_search_candidate_gets_a_smaller_budget():
    from backends import native_pytorch as npt

    b = _backend()
    art = b.build_baseline("Qwen/Qwen3.5-35B-A3B")
    candidate = dict(art.config)
    candidate["attn_implementation"] = "sdpa"     # the config that burned the run
    budget = b._config_timeout_s(candidate)
    assert budget < npt._COMPILE_TIMEOUT_S
    assert budget == min(npt._CONFIG_TIMEOUT_S, npt._COMPILE_TIMEOUT_S)


def test_the_baseline_is_matched_by_value_not_identity():
    """Configs are copied and round-tripped through JSON before measure() sees them."""
    from backends import native_pytorch as npt

    b = _backend()
    art = b.build_baseline("Qwen/Qwen3.5-35B-A3B")
    assert b._config_timeout_s(dict(art.config)) == npt._COMPILE_TIMEOUT_S


def test_an_unknown_baseline_gets_the_generous_budget():
    """A resumed run has no build_baseline call.

    Short-changing the baseline loses the whole model; being generous to one
    candidate now only costs that candidate, because it is reaped either way.
    """
    from backends import native_pytorch as npt

    b = _backend()
    assert getattr(b, "_baseline_config", None) is None
    assert b._config_timeout_s({"tp_degree": 8}) == npt._COMPILE_TIMEOUT_S


def test_the_candidate_budget_never_exceeds_the_baseline_budget():
    """TRN_OPT_CONFIG_TIMEOUT_S set absurdly high must not outrank the wall."""
    from backends import native_pytorch as npt

    b = _backend()
    art = b.build_baseline("Qwen/Qwen3.5-35B-A3B")
    candidate = dict(art.config)
    candidate["batch"] = 32
    assert b._config_timeout_s(candidate) <= npt._COMPILE_TIMEOUT_S


# --- compiler bugs are not model bugs ----------------------------------------

def test_a_compiler_internal_error_is_labelled_as_one():
    """Verbatim from the Qwen3.5-35B-A3B run (compile_mode=compile-default).

    "unsupported op" means rewrite the graph; INTERNAL_ERROR means the compiler
    crashed on a graph it should accept. Conflating them sends the search off
    probing configs around a bug that no config can avoid.
    """
    tail = ("[rank2]: error message=\"COMPILATION FAILED: Command failed "
            "(neuronx-cc compilation) with exit code 70: "
            "Qwen3_5MoeGatedDeltaNet[linear_attn][0]_select.326 [INTERNAL_ERROR] "
            "[NCC_IBCG901] BIRCodeGenLoop assertion err")
    reason = _worker_failure_reason(1, tail)
    assert "INTERNAL ERROR" in reason
    assert "escalate" in reason


def test_an_unsupported_op_is_still_reported_as_unsupported():
    """The distinction has to cut both ways."""
    tail = "[rank0]: NCC_EVRF029: Operation sort is not supported on trn2"
    reason = _worker_failure_reason(1, tail)
    assert "unsupported" in reason.lower()
    assert "INTERNAL ERROR" not in reason
