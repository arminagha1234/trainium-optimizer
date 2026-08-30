"""Load-staggering tests. Pure filesystem -- no torch, no distributed, no device.

The numbers asserted here are the ones that decide whether a model loads at all
on a trn2.48xlarge, so they are pinned to the measured host DRAM (2147 GB) and to
the two models that OOM-killed the pod during load.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from backends.load_stagger import (
    acquire_load_slot,
    concurrency_from_env,
    release_load_slot,
    staggered_peak_gb,
)

HOST_GB = 2147.0          # trn2.48xlarge /proc/meminfo, measured 2026-08-29


# --- the arithmetic that decides whether a model loads -----------------------

def test_staggering_brings_122b_inside_host_dram():
    """250 GB x 32 ranks is 8 TB and OOM-kills; 2 at a time is 734 GB and fits."""
    assert 32 * 250.2 > HOST_GB * 3                  # the observed failure
    peak = staggered_peak_gb(250.2, ranks=32, concurrency=2)
    assert peak < HOST_GB
    assert 700 <= peak <= 760                        # 2*250 + 30*7.8


def test_staggering_brings_deepseek_v4_flash_inside_host_dram():
    """319 GB (fp8 dequantised to bf16) x 64 ranks is 20 TB."""
    assert 64 * 319.2 > HOST_GB * 9
    assert staggered_peak_gb(319.2, ranks=64, concurrency=2) < HOST_GB


def test_ranks_outside_the_window_still_cost_their_shard():
    """The non-loading ranks are not free -- they hold their own slice."""
    naive = 2 * 500.0
    real = staggered_peak_gb(500.0, ranks=64, concurrency=2)
    assert real > naive                              # + 62 shards
    assert real == pytest.approx(2 * 500.0 + 62 * (500.0 / 64))


def test_concurrency_equal_to_world_is_the_unstaggered_peak():
    assert staggered_peak_gb(100.0, ranks=16, concurrency=16) == 1600.0


# --- env contract ------------------------------------------------------------

def test_default_is_no_staggering(monkeypatch):
    """Opt-in: staggering costs wall-clock on every model, including small ones."""
    monkeypatch.delenv("TRN_OPT_LOAD_CONCURRENCY", raising=False)
    assert concurrency_from_env(32) == 32


@pytest.mark.parametrize("raw,expected", [("2", 2), ("1", 1), ("8", 8), (" 4 ", 4)])
def test_env_sets_concurrency(monkeypatch, raw, expected):
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", raw)
    assert concurrency_from_env(32) == expected


@pytest.mark.parametrize("raw", ["", "abc", "0", "-3", "nan"])
def test_unusable_env_values_fall_back_to_no_staggering(monkeypatch, raw):
    """A typo must not silently serialise a 64-rank load into 64 waves."""
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", raw)
    assert concurrency_from_env(32) == 32


def test_concurrency_is_clamped_to_world(monkeypatch):
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", "999")
    assert concurrency_from_env(8) == 8


# --- the semaphore itself ----------------------------------------------------

def test_no_slot_is_taken_when_staggering_is_off(tmp_path):
    assert acquire_load_slot(0, world=8, concurrency=8, slot_dir=str(tmp_path)) is None
    assert acquire_load_slot(0, world=1, concurrency=1, slot_dir=str(tmp_path)) is None


def test_slots_are_exclusive_and_reusable(tmp_path):
    d = str(tmp_path / "slots")
    a = acquire_load_slot(0, world=8, concurrency=2, slot_dir=d)
    b = acquire_load_slot(1, world=8, concurrency=2, slot_dir=d)
    assert a is not None and b is not None and a != b

    # Third rank finds both taken and gives up at the timeout (fail open).
    c = acquire_load_slot(2, world=8, concurrency=2, slot_dir=d,
                          poll_s=0.01, timeout_s=0.05)
    assert c is None

    release_load_slot(a)
    again = acquire_load_slot(2, world=8, concurrency=2, slot_dir=d,
                              poll_s=0.01, timeout_s=5)
    assert again == a           # the freed slot is handed straight back


def test_a_slot_orphaned_by_a_dead_rank_is_reclaimed(tmp_path):
    """Without this, one crashed rank stalls every remaining rank for 90 minutes."""
    d = pathlib.Path(tmp_path / "slots")
    d.mkdir()
    dead = d / "slot0"
    dead.mkdir()
    # A pid that cannot be running. os.kill(_, 0) must report it gone.
    (dead / "pid").write_text("999999")
    got = acquire_load_slot(3, world=8, concurrency=1, slot_dir=str(d),
                            poll_s=0.01, timeout_s=5)
    assert got == dead


def test_a_slot_held_by_a_live_rank_is_not_reclaimed(tmp_path):
    d = pathlib.Path(tmp_path / "slots")
    d.mkdir()
    held = d / "slot0"
    held.mkdir()
    (held / "pid").write_text(str(os.getpid()))     # this very process
    assert acquire_load_slot(3, world=8, concurrency=1, slot_dir=str(d),
                             poll_s=0.01, timeout_s=0.05) is None


def test_release_is_idempotent_and_none_safe(tmp_path):
    d = str(tmp_path / "slots")
    slot = acquire_load_slot(0, world=8, concurrency=2, slot_dir=d)
    release_load_slot(slot)
    release_load_slot(slot)      # atexit may also fire; must not raise
    release_load_slot(None)
