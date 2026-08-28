# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the framework-improvement wiring into InventEngine:
  * fuse_first (fusion) prepends fused megakernel targets to run()
  * attach_tournament wires the engine's _device_race as the tournament measure
  * profiler seam is accepted and defaults to off (behaviour unchanged)
Pure CPU: injected race_fn / complete_fn, no device."""

from __future__ import annotations

import numpy as np

from invent_engine import InventEngine, RaceResult
from invent_kernels import OpSpec


def _spec(name, ref):
    ins = lambda: {"x": np.ones((4, 4), dtype=np.float32)}
    return OpSpec(name=name, family="m", shape_class="s", dtype="bf16",
                  reference=ref, offline_inputs=ins, real_inputs=ins)


def _engine(tmp_path):
    return InventEngine(out_dir=tmp_path)


# --- F4: fuse_first prepends fused targets -----------------------------------

def test_run_fuse_first_authors_a_fused_megakernel(tmp_path):
    specs = [_spec("rmsnorm", lambda inp: inp["x"] * 0.5),
             _spec("flash_attention", lambda inp: inp["x"] * 2.0)]
    eng = _engine(tmp_path)
    # deferred race so nothing needs a device; we only check the fused target
    # entered the author list.
    race_fn = lambda k, s: RaceResult(False, reason="deferred (test)")
    results = eng.run(specs, race_fn=race_fn, fuse_first=True)
    ops = [r.op for r in results]
    assert any(o.startswith("fused_") for o in ops), ops
    assert "fused_rmsnorm_flash_attention" in ops


def test_run_without_fuse_first_has_no_fused_target(tmp_path):
    specs = [_spec("rmsnorm", lambda inp: inp["x"] * 0.5),
             _spec("flash_attention", lambda inp: inp["x"] * 2.0)]
    eng = _engine(tmp_path)
    race_fn = lambda k, s: RaceResult(False, reason="deferred (test)")
    results = eng.run(specs, race_fn=race_fn)          # default: no fusion
    assert not any(r.op.startswith("fused_") for r in results)


# --- F3: attach_tournament wires the engine's race as measure ----------------

def test_attach_tournament_wires_device_race_measure(tmp_path):
    from kernel_tournament import TournamentAuthor
    eng = _engine(tmp_path)
    eng.attach_tournament(lambda p, **kw: "```python\ndef k(x): return x\n```")
    assert isinstance(eng.author, TournamentAuthor)
    # the tournament's measure must be THIS engine's bound _device_race
    assert eng.author._measure == eng._device_race


# --- F2: profiler seam accepted, default off ---------------------------------

def test_profiler_defaults_off(tmp_path):
    assert _engine(tmp_path).profiler is None


def test_profiler_seam_accepts_injected_profiler(tmp_path):
    eng = InventEngine(out_dir=tmp_path, profiler=lambda run: {"dma": 0.9})
    assert eng.profiler is not None
