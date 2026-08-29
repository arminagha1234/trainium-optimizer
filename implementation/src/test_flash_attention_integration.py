# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the FlashAttention kernel banked into the corpus.

Exercises the WHOLE framework path for the first genuinely-useful invented
kernel: registry (retrieve) -> kernel_inject (load/inject) -> run.

The wiring tests run on a plain CPU box (registry manifest read, primitive
routing, forward-factory load, injection into a mock model, and the invent
engine's Harvest reuse) — none of them import torch / neuronxcc. The end-to-end
correctness test (``run_on_device_e2e``) runs the flash kernel on a real
NeuronCore THROUGH the framework's retrieve->inject->run path; it is gated on
``nki_available()`` + ``FLASH_ONDEVICE=1`` so the normal suite never grabs the
device, and is also runnable as ``python test_flash_attention_integration.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kernel_registry import KernelRegistry, kernel_for_primitive
from backends.kernel_inject import KernelDescriptor, inject_kernel, load_kernel_entry
from invent_engine import InventEngine
from invent_kernels import flash_attention_spec, nki_available

# The in-repo kernel corpus dir (ships the FlashAttention manifest + source).
# Kernels live at repo-root knowledge-bank/kernels/ (one home for registry
# + library). src/../.. == repo root.
KERNEL_DIR = Path(__file__).resolve().parents[2] / "knowledge-bank" / "kernels"


# -- mock nn-module-like target ----------------------------------------------
class _FakeAttn:
    def forward(self, *a, **k):
        return "EAGER_ATTN"


class _FakeModel:
    def __init__(self):
        self.self_attn = _FakeAttn()

    def named_modules(self):
        return [("", self), ("self_attn", self.self_attn)]


# -- retrieve: registry + primitive routing ----------------------------------
def test_primitive_maps_to_flash_attention():
    for p in ("flash_attention", "FlashAttention", "flash-attn",
              "attention_long_context", "long_context_attention"):
        assert kernel_for_primitive(p) == "FlashAttention", p
    # unrelated attention primitives are unaffected
    assert kernel_for_primitive("mla") == "MLA"
    assert kernel_for_primitive("plain_dense_attention") is None


def test_registry_resolves_flash_kernel_rank4():
    reg = KernelRegistry(KERNEL_DIR)
    spec = reg.for_primitive("flash_attention")
    assert spec is not None and spec.name == "FlashAttention"
    assert spec.status == "passed-on-device"
    assert spec.rank == 4
    assert spec.usable and spec.hw_ready            # on-device => reusable HW-ready
    assert spec.entry == "adapter:build_flash_forward"
    assert spec.path == "FlashAttention/adapter.py"
    assert spec.tolerances.get("bf16") == 1.0e-2


# -- inject: load the forward-factory + patch a module -----------------------
def test_flash_forward_factory_loads(monkeypatch):
    monkeypatch.setenv("TRN_OPT_KERNEL_DIR", str(KERNEL_DIR))
    reg = KernelRegistry(KERNEL_DIR)
    spec = reg.for_primitive("flash_attention")
    factory = load_kernel_entry(spec.path, spec.entry)   # relative -> $TRN_OPT_KERNEL_DIR
    assert callable(factory)
    fwd = factory(object())                              # factory -> forward (no device)
    assert callable(fwd)


def test_inject_flash_into_model(monkeypatch):
    monkeypatch.setenv("TRN_OPT_KERNEL_DIR", str(KERNEL_DIR))
    reg = KernelRegistry(KERNEL_DIR)
    spec = reg.for_primitive("flash_attention")
    model = _FakeModel()
    desc = KernelDescriptor(target=r"self_attn", entry=spec.entry, path=spec.path)
    swapped, reason = inject_kernel(model, desc)
    assert swapped, reason
    # forward was swapped to the flash forward-factory's closure (a callable);
    # it is no longer the eager forward. (Not invoked here — that needs a device.)
    assert model.self_attn.forward is not _FakeAttn.forward


# -- retrieve THROUGH the invent engine: Harvest reuse -----------------------
def test_invent_engine_harvests_flash_kernel(tmp_path):
    reg = KernelRegistry(KERNEL_DIR)
    eng = InventEngine(out_dir=tmp_path / "run", registry=reg)
    res = eng.run_op(flash_attention_spec())
    assert res.status == "harvested"
    assert "FlashAttention" in res.detail
    assert any("harvested existing FlashAttention" in r.description
               for r in eng.ledger.read())


# -- the offline reference is well-formed ------------------------------------
def test_flash_reference_shapes_and_softmax():
    spec = flash_attention_spec(seqlen=512, d_head=128)
    inp = spec.real_inputs()
    out = spec.reference(inp)
    assert out.shape == (512, 128)          # (seqlen, d_head)
    assert np.isfinite(out).all()


# ===========================================================================
# on-device end-to-end THROUGH the framework (registry -> inject -> run)
# ===========================================================================
def run_on_device_e2e(seqlen: int, d_head: int = 128, seed: int = 22) -> dict:
    """Retrieve the flash kernel from the registry, build its forward via the
    kernel_inject loader, RUN it on a NeuronCore, and score correctness the
    framework's fair way (no worse than the bf16 incumbent). Returns a summary
    dict. Device-only."""
    import torch
    import torch_xla.core.xla_model as xm

    reg = KernelRegistry(KERNEL_DIR)
    kspec = reg.for_primitive("flash_attention")
    assert kspec is not None and kspec.hw_ready

    # retrieve -> load the forward-factory from the manifest's (entry, path)
    os.environ["TRN_OPT_KERNEL_DIR"] = str(KERNEL_DIR)
    factory = load_kernel_entry(kspec.path, kspec.entry)
    assert callable(factory), "flash forward-factory did not load"
    forward = factory(object())              # inject/build the forward

    # real inputs in the kernel's (d_head, seqlen) layout
    ospec = flash_attention_spec(seqlen=seqlen, d_head=d_head)
    inp = ospec.real_inputs()
    q, k, v = inp["q"], inp["k"], inp["v"]

    dev = xm.xla_device()
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(dev)
    got_t = forward(to(q), to(k), to(v))     # RUN the kernel through the framework
    xm.mark_step(); xm.wait_device_ops()
    got = got_t.cpu().to(torch.float32).numpy()

    # fp32 ideal + bf16 incumbent oracle (same unscaled op)
    ref = np.asarray(ospec.reference(inp), dtype=np.float32)
    qb, kb, vb = (torch.from_numpy(x).to(torch.bfloat16) for x in (q, k, v))
    pc = torch.softmax(qb.T.float() @ kb.float(), dim=-1)
    oracle = (pc @ vb.T.float()).numpy()

    def miss(a):
        return int((~np.isclose(a, ref, atol=1e-2, rtol=1e-2)).sum())

    k_miss, o_miss = miss(got), miss(oracle)
    correct = (got.shape == ref.shape
               and np.isfinite(got).all()
               and k_miss <= 2 * o_miss + 1)     # no worse than the bf16 incumbent
    return {
        "seqlen": seqlen, "d_head": d_head, "correct": bool(correct),
        "kernel_vs_fp32_max_err": float(np.abs(got - ref).max()),
        "kernel_vs_bf16_max_err": float(np.abs(got - oracle).max()),
        "kernel_fp32_miss": k_miss, "incumbent_fp32_miss": o_miss,
        "n_elems": int(got.size),
    }


@pytest.mark.skipif(
    not (nki_available() and os.environ.get("FLASH_ONDEVICE") == "1"),
    reason="on-device flash run (set FLASH_ONDEVICE=1 on a Trainium box)")
def test_flash_runs_on_device_through_framework():
    r = run_on_device_e2e(2048)
    assert r["correct"], r


if __name__ == "__main__":
    if not nki_available():
        raise SystemExit("not on a Trainium box (nki unavailable)")
    for S in (2048, 8192):
        res = run_on_device_e2e(S)
        print(f"[flash e2e] {res}")
        assert res["correct"], f"flash incorrect at S={S}: {res}"
    print("[flash e2e] PASS through framework registry->inject->run")
