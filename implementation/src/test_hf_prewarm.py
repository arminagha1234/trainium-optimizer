"""Tests for the HF cache prewarm that lets ranks resolve a checkpoint offline.

The failure this guards against reads exactly like a corrupt cache and is not one:
at tp=16, sixteen worker processes each resolve the checkpoint against the Hub
unauthenticated, and a rate-limited rank reports an already-cached shard as
absent -- a different rank and a different shard every run. The cache in the
observed case was verified complete (14/14 shards, correct sizes, a real read of
each) by the parent process seconds before a run that failed this way.
"""

from __future__ import annotations

import pytest

from backends.native_pytorch import _HF_WEIGHT_PATTERNS, prewarm_hf_cache


class _Recorder:
    """Stands in for huggingface_hub.snapshot_download."""

    def __init__(self, *, local_ok: bool, remote_ok: bool = True):
        self.local_ok = local_ok
        self.remote_ok = remote_ok
        self.calls: list[dict] = []

    def __call__(self, model_id, **kw):
        self.calls.append({"model_id": model_id, **kw})
        if kw.get("local_files_only"):
            if not self.local_ok:
                raise OSError("not fully cached")
            return "/cache/snapshot"
        if not self.remote_ok:
            raise OSError("403 gated repo")
        return "/cache/snapshot"

    @property
    def n_local(self):
        return sum(1 for c in self.calls if c.get("local_files_only"))

    @property
    def n_remote(self):
        return sum(1 for c in self.calls if not c.get("local_files_only"))


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    def install(rec):
        monkeypatch.setattr(huggingface_hub, "snapshot_download", rec)
        return rec
    return install


def test_a_complete_cache_needs_no_network(hub):
    rec = hub(_Recorder(local_ok=True))
    assert prewarm_hf_cache("Qwen/Qwen3.5-35B-A3B") is True
    assert rec.n_local == 1
    assert rec.n_remote == 0        # the whole point: zero Hub contact


def test_an_incomplete_cache_is_filled_once_here(hub):
    """`tp` ranks downloading the same shard is the other half of the problem."""
    rec = hub(_Recorder(local_ok=False, remote_ok=True))
    assert prewarm_hf_cache("Qwen/Qwen3.5-35B-A3B") is True
    assert rec.n_local == 1
    assert rec.n_remote == 1


def test_a_failed_prewarm_leaves_the_workers_online(hub):
    """Fail open. A gated repo or an offline box must not become a hard failure."""
    logged = []
    rec = hub(_Recorder(local_ok=False, remote_ok=False))
    assert prewarm_hf_cache("some/gated-model", log=logged.append) is False
    assert rec.n_remote == 1
    assert logged and "each rank will resolve on its own" in logged[0]


def test_an_explicit_local_path_is_left_alone(hub, tmp_path):
    rec = hub(_Recorder(local_ok=True))
    assert prewarm_hf_cache(str(tmp_path)) is False
    assert rec.calls == []


def test_empty_model_id_is_not_resolved(hub):
    rec = hub(_Recorder(local_ok=True))
    assert prewarm_hf_cache("") is False
    assert rec.calls == []


def test_only_files_transformers_reads_are_required(hub):
    """A repo may ship GGUF/ONNX variants; requiring them would defeat the prewarm."""
    rec = hub(_Recorder(local_ok=True))
    prewarm_hf_cache("Qwen/Qwen3.5-35B-A3B")
    pats = rec.calls[0]["allow_patterns"]
    assert pats == _HF_WEIGHT_PATTERNS
    assert "*.safetensors" in pats
    assert "*" not in pats


def test_a_missing_hub_package_is_not_fatal(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("no hub")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert prewarm_hf_cache("Qwen/Qwen3.5-35B-A3B") is False
