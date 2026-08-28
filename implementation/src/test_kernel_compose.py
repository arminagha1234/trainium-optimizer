# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kernel_compose — decompose-and-compose authoring. Pure CPU: minimal
OpSpecs + a mock complete_fn (no model, no device)."""

from __future__ import annotations

import numpy as np

from invent_kernels import OpSpec
from kernel_author import build_author_prompt
from kernel_compose import (
    ComposingAuthor, DECOMPOSITIONS, Primitive, PrimitiveLibrary,
    compose_section, default_library, make_compose_prompt_fn,
)


def _spec(name: str, family: str = "x") -> OpSpec:
    ref = lambda inp: inp["x"]
    ins = lambda: {"x": np.zeros((8, 8), dtype=np.float32)}
    return OpSpec(name=name, family=family, shape_class="s", dtype="bf16",
                  reference=ref, offline_inputs=ins, real_inputs=ins)


# --- the library -------------------------------------------------------------

def test_default_library_has_verified_primitives():
    lib = default_library()
    for fam in ("attention", "scan", "matmul"):
        prims = lib.for_family(fam)
        assert prims, f"no primitives for {fam}"
        assert all(p.verified for p in prims)


def test_for_family_respects_decomposition_order():
    lib = default_library()
    got = [p.name for p in lib.for_family("attention")]
    assert got == [n for n in DECOMPOSITIONS["attention"] if lib.get(n)]


def test_unverified_primitive_not_offered():
    lib = PrimitiveLibrary([
        Primitive("tiled_psum_matmul", frozenset({"matmul"}), "src", verified=False)])
    assert lib.for_family("matmul") == []      # unverified block is never offered


def test_unknown_family_empty():
    assert default_library().for_family("elementwise") == []


# --- compose_section ---------------------------------------------------------

def test_compose_section_for_attention_lists_blocks():
    sec = compose_section(_spec("flash_attention"), default_library())
    assert "COMPOSE" in sec
    assert "online_softmax_step" in sec and "kv_tile_loop" in sec
    assert "do NOT" in sec.lower() or "do not" in sec.lower()


def test_compose_section_empty_for_easy_op():
    # rmsnorm classifies as normalization -> not a hard family -> no compose block
    assert compose_section(_spec("rmsnorm"), default_library()) == ""


# --- prompt fn: prepends for hard, byte-identical for easy -------------------

def test_prompt_fn_prepends_for_hard_op():
    fn = make_compose_prompt_fn(default_library())
    spec = _spec("flash_attention")
    composed = fn(spec, None, None)
    base = build_author_prompt(spec, None, None)
    assert composed.endswith(base)          # base prompt preserved verbatim...
    assert len(composed) > len(base)        # ...with the compose block prepended
    assert "COMPOSE" in composed


def test_prompt_fn_byte_identical_for_easy_op():
    fn = make_compose_prompt_fn(default_library())
    spec = _spec("rmsnorm")
    assert fn(spec, None, None) == build_author_prompt(spec, None, None)


# --- ComposingAuthor ---------------------------------------------------------

def test_composing_author_sees_blocks_in_prompt_for_hard_op():
    seen = {}
    def complete(prompt, **kw):
        seen["prompt"] = prompt
        return "```python\n@nki.jit\ndef flash_attention_kernel(x):\n    return x\n```"
    author = ComposingAuthor(complete)
    k = author.author(_spec("flash_attention"))
    assert k.entry == "flash_attention_kernel"
    assert "COMPOSE" in seen["prompt"]
    assert "tiled_psum_matmul" in seen["prompt"]


def test_composing_author_no_blocks_for_easy_op():
    seen = {}
    def complete(prompt, **kw):
        seen["prompt"] = prompt
        return "```python\n@nki.jit\ndef rmsnorm_kernel(x):\n    return x\n```"
    ComposingAuthor(complete).author(_spec("rmsnorm"))
    assert "COMPOSE" not in seen["prompt"]     # easy op: plain prompt


def test_composing_author_is_kernel_author_protocol():
    from kernel_author import KernelAuthor
    author = ComposingAuthor(lambda p, **kw: "```python\ndef k(x): return x\n```")
    assert isinstance(author, KernelAuthor)


def test_from_kernel_library_ingests_verified_only():
    class _Lib:
        def primitives(self):
            return [
                {"name": "extra_ok", "families": ["attention"], "nki_src": "s",
                 "verified": True},
                {"name": "extra_bad", "families": ["attention"], "nki_src": "s",
                 "verified": False}]
    lib = default_library().from_kernel_library(_Lib())
    assert lib.get("extra_ok") is not None
    assert lib.get("extra_bad") is None
