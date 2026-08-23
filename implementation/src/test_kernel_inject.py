"""Tests for the generic kernel-injection hook (backends/kernel_inject.py).

These run on a plain CPU box: the hook is torch-free (duck-typed against
``named_modules()`` / ``module.forward``), so we validate it with a MOCK model +
a FAKE kernel file written to a tmp dir. No torch, no torch_neuronx, no nki.

What must hold:
  - inject_kernel LOADS a kernel from an external on-disk file (by path+entry),
    swaps the target module's forward, and the swapped forward is what runs;
  - a JSON string descriptor (the --kernel CLI form) is accepted;
  - $TRN_OPT_KERNEL_DIR resolves a relative descriptor path;
  - every unmet precondition is a graceful (False, reason) no-op, never a raise:
    no descriptor, unloadable file, no target match;
  - a class-name target and a dotted-name target both select the right module.
"""

from __future__ import annotations

import json
from pathlib import Path

from backends.kernel_inject import (
    KernelDescriptor,
    inject_kernel,
    load_kernel_entry,
)


# -- mock nn-module-like objects --------------------------------------------

class _FakeModule:
    """The minimum an injectable target needs: a ``forward`` and an identity
    (its class name is used for class-name targeting)."""

    def __init__(self, name: str = "eager"):
        self._name = name

    def forward(self, *a, **k):
        return "EAGER_OUTPUT"


class _FakeMoEBlock(_FakeModule):
    pass


class _FakeModel:
    """A stand-in for an nn.Module: exposes ``named_modules()`` like torch does."""

    def __init__(self):
        self.mlp = _FakeMoEBlock()
        self.attn = _FakeModule()

    def named_modules(self):
        return [("", self), ("mlp", self.mlp), ("attn", self.attn)]


# -- a FAKE kernel file, written to a tmp dir (the external-kernel-dir stand-in)

_FAKE_KERNEL_SRC = '''
"""A fake external kernel file. The entry is a forward FACTORY:
build_forward(module) -> new_forward. No torch/nki — pure python."""

def build_forward(module):
    def forward(*args, **kwargs):
        return "KERNEL_OUTPUT"
    return forward
'''


def _write_fake_kernel(tmp_path: Path, name: str = "fake_kernel.py") -> Path:
    p = tmp_path / name
    p.write_text(_FAKE_KERNEL_SRC)
    return p


# -- the happy path ----------------------------------------------------------

def test_inject_swaps_target_forward_and_it_is_called(tmp_path: Path):
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()

    # sanity: eager forward before injection
    assert model.mlp.forward() == "EAGER_OUTPUT"

    swapped, reason = inject_kernel(
        model,
        {"target": "FakeMoEBlock", "entry": "build_forward", "path": str(kfile)},
    )
    assert swapped, reason
    # the target's forward is now the kernel's, and running it returns the kernel
    # output — proving the swapped forward is what executes.
    assert model.mlp.forward() == "KERNEL_OUTPUT"
    # the non-target module was left eager (class-name target only hit the block)
    assert model.attn.forward() == "EAGER_OUTPUT"
    assert "1/1" in reason or "1" in reason


def test_json_string_descriptor_accepted(tmp_path: Path):
    """The --kernel CLI form is a JSON string; inject_kernel must coerce it."""
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()
    desc = json.dumps({"target": "mlp", "entry": "build_forward",
                       "path": str(kfile)})
    swapped, reason = inject_kernel(model, desc)
    assert swapped, reason
    assert model.mlp.forward() == "KERNEL_OUTPUT"


def test_dotted_name_target(tmp_path: Path):
    """Targeting by dotted module name (WHERE it lives) as well as class name."""
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()
    swapped, _ = inject_kernel(
        model, {"target": r"^attn$", "entry": "build_forward", "path": str(kfile)})
    assert swapped
    assert model.attn.forward() == "KERNEL_OUTPUT"
    assert model.mlp.forward() == "EAGER_OUTPUT"


def test_entry_with_colon_form(tmp_path: Path):
    """A registry-style ``entry`` ('stem:func') resolves to the function name."""
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()
    swapped, _ = inject_kernel(
        model, {"target": "mlp", "entry": "fake_kernel:build_forward",
                "path": str(kfile)})
    assert swapped
    assert model.mlp.forward() == "KERNEL_OUTPUT"


def test_kernel_dir_env_resolves_relative_path(tmp_path: Path, monkeypatch):
    """A relative descriptor path resolves against $TRN_OPT_KERNEL_DIR — the
    external (proprietary) kernel dir convention."""
    _write_fake_kernel(tmp_path, "fake_kernel.py")
    monkeypatch.setenv("TRN_OPT_KERNEL_DIR", str(tmp_path))
    model = _FakeModel()
    swapped, reason = inject_kernel(
        model, {"target": "mlp", "entry": "build_forward",
                "path": "fake_kernel.py"})   # relative -> resolved against env
    assert swapped, reason
    assert model.mlp.forward() == "KERNEL_OUTPUT"


# -- graceful no-op preconditions (never raise) ------------------------------

def test_no_descriptor_is_noop():
    model = _FakeModel()
    swapped, reason = inject_kernel(model, "")
    assert not swapped
    assert "no/invalid" in reason
    assert model.mlp.forward() == "EAGER_OUTPUT"


def test_malformed_json_is_noop():
    model = _FakeModel()
    swapped, _ = inject_kernel(model, "{not json")
    assert not swapped
    assert model.mlp.forward() == "EAGER_OUTPUT"


def test_missing_kernel_file_is_noop():
    model = _FakeModel()
    swapped, reason = inject_kernel(
        model, {"target": "mlp", "entry": "build_forward",
                "path": "/no/such/kernel.py"})
    assert not swapped
    assert "not loadable" in reason
    assert model.mlp.forward() == "EAGER_OUTPUT"


def test_no_target_match_is_noop(tmp_path: Path):
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()
    swapped, reason = inject_kernel(
        model, {"target": "NoSuchModuleXYZ", "entry": "build_forward",
                "path": str(kfile)})
    assert not swapped
    assert "no module matched" in reason
    assert model.mlp.forward() == "EAGER_OUTPUT"


def test_missing_entry_symbol_is_noop(tmp_path: Path):
    kfile = _write_fake_kernel(tmp_path)
    model = _FakeModel()
    swapped, reason = inject_kernel(
        model, {"target": "mlp", "entry": "does_not_exist", "path": str(kfile)})
    assert not swapped
    assert "not loadable" in reason


# -- descriptor + loader units ----------------------------------------------

def test_descriptor_from_obj_requires_all_fields():
    assert KernelDescriptor.from_obj(None) is None
    assert KernelDescriptor.from_obj({"target": "x", "entry": "y"}) is None  # no path
    d = KernelDescriptor.from_obj({"target": "x", "entry": "y", "path": "z"})
    assert d and (d.target, d.entry, d.path) == ("x", "y", "z")


def test_load_kernel_entry_returns_callable(tmp_path: Path):
    kfile = _write_fake_kernel(tmp_path)
    fn = load_kernel_entry(str(kfile), "build_forward")
    assert callable(fn)
    fwd = fn(object())          # factory -> forward
    assert callable(fwd) and fwd() == "KERNEL_OUTPUT"
