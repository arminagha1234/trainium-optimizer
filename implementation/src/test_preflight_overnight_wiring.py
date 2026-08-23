"""
Tests for the kernel-registry -> preflight wiring in overnight.run_one.

The point of the wiring: a linear-attention / GatedDeltaNet model no longer
skips with the GENERIC unsupported reason — overnight builds a KernelRegistry
(reads $TRN_OPT_KERNEL_DIR) and threads it into preflight_check, so the skip's
recorded reason NAMES the kernel it needs (DeltaNet) and reports availability.
With --kernels-wired AND a usable kernel registered, the model is allowed to
PROCEED via the kernel path instead of skipping.

Mock-only: MockBackend, an on-disk KnowledgeBank, and a linear-attention config
written to a local <model_dir>/config.json (so the default weight-free config
loader reads it with no transformers dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import overnight
from bank import KnowledgeBank
from kernel_registry import KernelRegistry
from orchestrator import ModelSpec
from preflight import LINEAR_ATTN_REASON, preflight_check


LINEAR_ATTN_CONFIG = {
    "architectures": ["Qwen3_5GatedDeltaNetForCausalLM"],
    "model_type": "qwen3_5_gated_deltanet",
}


def _deltanet_registry(tmp_path: Path, status: str = "passed-on-device") -> KernelRegistry:
    """A KernelRegistry with a single registered DeltaNet kernel manifest."""
    kdir = tmp_path / "kernels" / "DeltaNet"
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text(json.dumps({
        "name": "DeltaNet", "status": status,
        "entry": "gdn_chunked_prefill_nki:gdn_chunked_prefill_kernel",
        "variants": ["gated_deltanet"],
        "tolerances": {"bf16": 8.3e-4},
    }))
    return KernelRegistry(tmp_path / "kernels")


def _linear_attn_spec(tmp_path: Path) -> ModelSpec:
    """A linear-attention ModelSpec whose model_id is a local dir carrying a
    config.json — so the default (weight-free) config loader detects the arch."""
    mdir = tmp_path / "Qwen3.5-4B-GatedDeltaNet"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(json.dumps(LINEAR_ATTN_CONFIG))
    return ModelSpec(model_id=str(mdir), family="hybrid_attention_causal_lm",
                     param_count=4e9, parent="qwen")


def _sink():
    msgs: list[str] = []
    return msgs, (lambda m: msgs.append(m))


# -- overnight.run_one: linear-attn SKIP names the DeltaNet kernel -----------

def test_run_one_skip_reason_names_deltanet(tmp_path: Path):
    """With a registry but kernels NOT wired, run_one skips a linear-attn model
    and the recorded reason (result, ledger row, and anti-pattern lesson) NAMES
    the DeltaNet kernel — the richer reason, not the generic one."""
    out_root = tmp_path / "art"
    out_root.mkdir()
    bank = KnowledgeBank(tmp_path / "bank")
    spec = _linear_attn_spec(tmp_path)
    reg = _deltanet_registry(tmp_path)
    _, log = _sink()

    result = overnight.run_one(
        slug="gdn", spec=spec, backend_name="mock", out_root=out_root,
        bank=bank, sdk_version="2.28.0", log=log, instance_type=None,
        cycle=1, profile_loop=False, preflight=True,
        registry=reg, kernels_wired=False,
    )

    # It was skipped, with a reason that names the kernel (not the generic one).
    assert result.skipped is True
    assert result.ok is False
    assert "DeltaNet" in result.error
    assert result.error != LINEAR_ATTN_REASON

    # The recorded ledger row (Stage.PREFLIGHT skip) carries the richer reason.
    from ledger import Ledger
    rows = Ledger(out_root / "optimization_runs" / "gdn").read()
    skip_rows = [r for r in rows if "preflight skip" in r.description]
    assert skip_rows and "DeltaNet" in skip_rows[0].description

    # And the emitted anti-pattern lesson carries the richer reason too.
    aps = bank.preflight_antipatterns(spec.family, "2.28.0")
    assert aps and "DeltaNet" in aps[0].reason

    # A permanent HISTORY row was written as a skip.
    hist = (out_root / "HISTORY.tsv").read_text()
    assert "skipped" in hist


def test_run_one_proceeds_when_kernel_wired(tmp_path: Path, monkeypatch):
    """With a usable kernel registered AND --kernels-wired, the linear-attn
    model is NOT skipped — it proceeds down the (mock) optimization pipeline."""
    # Keep the box-throughput probe from shelling out during the mock run.
    monkeypatch.setattr(overnight, "_box_throughput", lambda *a, **k: 0.0)

    out_root = tmp_path / "art"
    out_root.mkdir()
    bank = KnowledgeBank(tmp_path / "bank")
    spec = _linear_attn_spec(tmp_path)
    reg = _deltanet_registry(tmp_path, status="passed-on-device")
    _, log = _sink()

    result = overnight.run_one(
        slug="gdn", spec=spec, backend_name="mock", out_root=out_root,
        bank=bank, sdk_version="2.28.0", log=log, instance_type=None,
        cycle=1, profile_loop=False, preflight=True,
        registry=reg, kernels_wired=True,
    )

    assert not getattr(result, "skipped", False)   # kernel path unblocked it
    assert result.ok is True


# -- preflight_check with the exact kwargs overnight passes ------------------

def test_preflight_check_with_overnight_kwargs_names_deltanet(tmp_path: Path):
    """Belt-and-suspenders: the same call overnight makes (registry + optional
    kernels_wired) names DeltaNet when not wired, and proceeds when wired."""
    spec = _linear_attn_spec(tmp_path)
    reg = _deltanet_registry(tmp_path)

    ok, reason = preflight_check(spec, bank=None, sdk_version="2.28.0",
                                 registry=reg, kernels_wired=False)
    assert ok is False and "DeltaNet" in reason

    ok, reason = preflight_check(spec, bank=None, sdk_version="2.28.0",
                                 registry=reg, kernels_wired=True)
    assert ok is True and reason is None
