"""backends/kernel_inject.py — the GENERIC kernel-injection hook.

This is the mechanism that lets ANY registered/authored kernel actually reach a
served model. Until now the only path from "a kernel exists" to "a kernel runs
in the forward" was the ONE hardcoded MoE special-case (`kernels.moe_fused.
swap_moe_forward`): it knew, by name, that it wanted to patch the HF
`Qwen3MoeSparseMoeBlock.forward`. That does not generalize — a DeltaNet /
Mamba2 / RMSNorm kernel had nowhere to plug in. This module is the generalization:

    descriptor (target, entry, path)  ->  load the kernel from its on-disk file
                                      ->  find every module the target matches
                                      ->  monkeypatch that module's .forward

so the measurement spine can inject an arbitrary kernel and then let the
orchestrator's equivalence gate decide whether the swap was a win (exactly the
same contract `swap_moe_forward` already lives under).

## Two load-bearing design choices

1. **Torch-free.** Everything here is duck-typed against ``model.named_modules()``
   and ``module.forward`` — there is not a single ``import torch`` in this file.
   That is deliberate: `neuron_worker.py` imports torch/transformers at module
   top (it only ever runs under `torchrun` on a Trainium box), so the *resolution
   logic* would be untestable if it lived there. Keeping it here means the whole
   inject/resolve path is unit-testable on a CPU box with a mock model and a fake
   kernel file — which is the Phase-1 requirement. `neuron_worker.py` re-exports
   `inject_kernel` and calls it inside `main()`.

2. **IP / public-repo boundary.** The kernel is LOADED by (path, entry) from an
   EXTERNAL directory — the descriptor's ``path`` (resolved against
   ``$TRN_OPT_KERNEL_DIR`` when relative). Kernel SOURCE never lives in this
   repo; we import it from disk with the SAME ``importlib.util.
   spec_from_file_location`` loader `invent_kernels._load_entry_from_file` uses,
   so a proprietary kernel plugs in privately while the public framework carries
   only the orchestration.

## Kernel factory contract

The ``entry`` symbol resolved from the file is a **forward factory**:

    build_forward(module) -> new_forward_callable

i.e. it is handed the target module (so it can capture that module's weights /
config) and returns the replacement ``forward``. `inject_kernel` then does
``module.forward = build_forward(module)`` for every matching module. This
mirrors how `swap_moe_forward` closes over the block it patches, and keeps the
hook agnostic to what the kernel actually computes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class KernelDescriptor:
    """A resolved instruction for one injection: WHICH modules to patch
    (``target``), WHAT symbol builds the replacement forward (``entry``), and
    WHERE the kernel source lives on disk (``path``).

    * ``target`` — a regex matched (via ``re.search``) against BOTH each module's
      dotted qualified name (from ``named_modules()``) and its class name, so a
      caller can target either "the module living at model.layers.3.mlp" or
      "every module whose class is *SparseMoeBlock*".
    * ``entry``  — the forward-factory symbol. Accepts ``"func"`` or
      ``"<stem>:func"`` (the colon form the registry's ``KernelSpec.entry`` uses,
      e.g. ``"gdn_chunked_prefill_nki:gdn_chunked_prefill_kernel"``); only the
      function name after the colon matters here since ``path`` already names the
      file.
    * ``path``   — the ``.py`` file. Relative paths resolve against
      ``$TRN_OPT_KERNEL_DIR`` so an install can point at its private kernel dir.
    """

    target: str
    entry: str
    path: str

    @classmethod
    def from_obj(cls, obj: Any) -> "KernelDescriptor | None":
        """Coerce a JSON string / dict / mapping-ish object into a descriptor.

        Returns None for anything empty or missing a required field, so the
        caller (the worker) can treat "no --kernel requested" and "malformed
        descriptor" identically: a graceful no-op, never a crash.
        """
        if obj is None:
            return None
        if isinstance(obj, str):
            s = obj.strip()
            if not s:
                return None
            try:
                obj = json.loads(s)
            except Exception:  # noqa: BLE001 — a bad JSON descriptor is "no kernel"
                return None
        if isinstance(obj, cls):
            return obj
        # dict or any object exposing target/entry/path
        def _get(name: str) -> str:
            if isinstance(obj, dict):
                return str(obj.get(name, "") or "")
            return str(getattr(obj, name, "") or "")

        target, entry, path = _get("target"), _get("entry"), _get("path")
        if not (target and entry and path):
            return None
        return cls(target=target, entry=entry, path=path)


def _resolve_path(path: str) -> Path:
    """Resolve the kernel file path, honoring ``$TRN_OPT_KERNEL_DIR`` for
    relative descriptors (the external-kernel-dir convention the registry and
    invent engine already use)."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = os.environ.get("TRN_OPT_KERNEL_DIR")
    return (Path(base) / p) if base else p


def _entry_func_name(entry: str) -> str:
    """The function name from an ``entry`` string. ``"stem:func"`` -> ``"func"``;
    a bare ``"func"`` -> ``"func"``. (The stem is redundant here — ``path``
    already identifies the file — but we accept it so a registry KernelSpec's
    ``entry`` can be passed through unchanged.)"""
    return entry.split(":")[-1].strip()


def load_kernel_entry(path: str, entry: str,
                      log: Callable[[str], None] = print) -> Callable | None:
    """Import the kernel file from disk and return its ``entry`` (forward
    factory) callable, or None on any failure.

    This is the SAME on-disk import approach as
    ``invent_kernels._load_entry_from_file`` (real ``__file__`` so the NKI tracer
    can introspect source), reduced to the load half. It NEVER raises: a missing
    file, an import error, or an absent/non-callable symbol all return None so
    the worker cleanly falls back to the un-injected (eager) model — a bad kernel
    is DATA, not a crash. No torch/nki import here; the kernel file itself pulls
    in whatever it needs when executed (which only happens on a real box).
    """
    fpath = _resolve_path(path)
    func = _entry_func_name(entry)
    if not func:
        log(f"inject: empty entry symbol for {fpath}")
        return None
    if not fpath.is_file():
        log(f"inject: kernel file not found: {fpath}")
        return None
    try:
        # Content-neutral but unique module name so re-injecting a different file
        # at the same logical slot does not collide in sys.modules.
        mod_name = "trn_opt_injected_" + re.sub(r"\W+", "_", str(fpath.stem)) \
            + "_" + str(abs(hash(str(fpath))) % (10 ** 8))
        spec = importlib.util.spec_from_file_location(mod_name, str(fpath))
        if spec is None or spec.loader is None:
            log(f"inject: could not build import spec for {fpath}")
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module          # register BEFORE exec
        try:
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001 — device/import failure is data
            sys.modules.pop(mod_name, None)
            log(f"inject: kernel import failed ({fpath}): {e!r}")
            return None
    except Exception as e:  # noqa: BLE001 — filesystem/import wiring failure
        log(f"inject: kernel load wiring failed ({fpath}): {e!r}")
        return None
    fn = getattr(module, func, None)
    if not callable(fn):
        log(f"inject: entry '{func}' not found / not callable in {fpath}")
        return None
    return fn


def _named_modules(model: Any) -> Iterable[tuple[str, Any]]:
    """Yield (dotted_name, module) pairs. Uses ``model.named_modules()`` when
    present (every ``torch.nn.Module`` has it, and so does the CPU mock), else
    falls back to treating ``model`` itself as the single module."""
    nm = getattr(model, "named_modules", None)
    if callable(nm):
        return list(nm())
    return [("", model)]


def _matches(pattern: str, name: str, module: Any) -> bool:
    """A module is a target if the pattern regex-searches its dotted name OR its
    class name. Two spellings so callers can address a module by WHERE it lives
    (``model.layers.0.mlp``) or by WHAT it is (``*SparseMoeBlock*``)."""
    try:
        rx = re.compile(pattern)
    except re.error:
        # Not a valid regex -> fall back to a plain substring test.
        return pattern in name or pattern in type(module).__name__
    return bool(rx.search(name) or rx.search(type(module).__name__))


def inject_kernel(model: Any, spec: Any,
                  log: Callable[[str], None] = print) -> tuple[bool, str]:
    """Inject a registered/authored kernel into ``model``'s forward.

    ``spec`` is a ``KernelDescriptor`` (or a JSON string / dict coercible to one)
    carrying ``(target, entry, path)``. The kernel is imported from ``path`` and
    its ``entry`` forward-factory is applied to EVERY module matching ``target``:
    ``module.forward = build_forward(module)``.

    Returns ``(swapped, reason)`` — the SAME contract as
    ``kernels.moe_fused.swap_moe_forward`` — and NEVER raises. On any unmet
    precondition (no descriptor, unloadable kernel, no matching module, a factory
    that errors on a given module) it patches what it safely can and reports why,
    leaving the rest of the model eager. The equivalence gate then evaluates a
    correct (possibly unchanged) candidate rather than the run crashing.
    """
    desc = KernelDescriptor.from_obj(spec)
    if desc is None:
        return False, "no/invalid kernel descriptor -> eager (no injection)"

    factory = load_kernel_entry(desc.path, desc.entry, log)
    if factory is None:
        return False, f"kernel not loadable (path={desc.path}, entry={desc.entry})"

    swapped = 0
    matched = 0
    for name, module in _named_modules(model):
        if not _matches(desc.target, name, module):
            continue
        matched += 1
        try:
            new_forward = factory(module)
        except Exception as e:  # noqa: BLE001 — a factory failure on one module
            log(f"inject: factory failed on '{name}' ({e!r}); left eager")
            continue
        if not callable(new_forward):
            log(f"inject: factory for '{name}' did not return a callable; left eager")
            continue
        module.forward = new_forward
        swapped += 1
        log(f"inject: swapped forward on '{name}' ({type(module).__name__}) "
            f"via {desc.entry}")

    if matched == 0:
        return False, f"no module matched target /{desc.target}/ -> eager"
    if swapped == 0:
        return False, (f"target /{desc.target}/ matched {matched} module(s) but "
                       f"none could be swapped -> eager")
    return True, f"injected {desc.entry} into {swapped}/{matched} matched module(s)"
