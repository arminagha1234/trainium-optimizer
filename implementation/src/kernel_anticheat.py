"""kernel_anticheat.py — the ADVERSARIAL correctness gate for kernel candidates.

`kernel_validation.verdict` answers "did this run match numerics and emit a
NEFF?" — but that question assumes the candidate is HONESTLY trying to be a
kernel. The LLM-kernel corpus is a graveyard of candidates that gamed exactly
that gate instead of satisfying it. Two failures we refuse to relearn:

  * **Sakana's "100× faster" CUDA kernels** were later shown to be ~3× SLOWER —
    the harness measured a candidate that had quietly fallen back to the
    framework (or read a cached/aliased result) rather than doing the work. The
    "win" was a measurement of the reference, not of a kernel.
  * **Kevin-32B** produced kernels that RECYCLED the reference implementation's
    output tensor: the candidate never computed anything, it returned the buffer
    the reference had already filled, so allclose trivially "passed".

Those are not numerical bugs; they are REWARD HACKS, and a numerics gate alone
cannot see them. This module is the adversarial layer that runs BEFORE / AROUND
the numerics gate:

  1. ``adversarial_source_check`` — static source anti-cheat: does the candidate
     even contain a kernel, or is it a framework fallback / try-except swallow /
     reference-inheriting shim dressed up as one?
  2. ``require_reproducible`` — the "runs twice, different answer" tell: a real
     kernel is deterministic on fixed inputs; a candidate that reads uninitialized
     / aliased / racy buffers is not.
  3. ``run_candidate_before_reference`` — the buffer-order protocol that closes
     Kevin's recycled-output exploit by construction.

Everything here is PURE PYTHON and Trainium-free: the static check parses source
text (``ast``), the reproducibility gate just calls a supplied ``run_fn`` twice,
and the buffer-order helper only sequences two callables. All of it is
unit-testable on a CPU box, which is the whole point — the anti-cheat must run in
the same cheap loop as the rest of the validation spine.
"""

from __future__ import annotations

import ast
from typing import Any, Callable


# -- (1) static source anti-cheat --------------------------------------------

# torch / torch.nn.functional COMPUTE ops. If any of these is the compute path,
# the "kernel" is really the framework doing the work (the Sakana fallback). This
# is deliberately a COMPUTE denylist, NOT "any torch reference": a real NKI kernel
# may legitimately `import torch` for a dtype (``torch.bfloat16``) or allocate an
# output (``torch.empty``/``zeros``), so allocation/dtype attributes are NOT here
# — only ops that actually perform the math a kernel is supposed to perform.
_TORCH_COMPUTE_OPS = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "dot", "einsum", "tensordot",
    "softmax", "log_softmax", "scaled_dot_product_attention", "sdpa",
    "linear", "conv1d", "conv2d", "conv3d", "layer_norm", "group_norm",
    "rms_norm", "batch_norm", "attention", "multi_head_attention_forward",
    "relu", "gelu", "silu", "sigmoid", "tanh", "elu", "cumsum", "cumprod",
    "logsumexp", "cross_entropy",
})

# Import / base-class name tokens that betray "inherit or import the reference
# implementation instead of writing a kernel". Kept narrow (whole recognizable
# tokens, case-insensitive) so a kernel that merely imports `torch` or a math
# helper is not swept up.
_REFERENCE_TOKENS = ("reference", "baseline", "ref_impl", "refimpl",
                     "eager_forward", "reference_impl", "reference_forward")


def _attr_parts(node: ast.AST) -> list[str]:
    """Flatten an attribute/name chain into dotted parts, e.g. the node for
    ``torch.nn.functional.softmax`` -> ``["torch", "nn", "functional",
    "softmax"]``. Returns ``[]`` for anything that is not a plain Name/Attribute
    chain (e.g. a subscript or call in the middle)."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return []


def _uses_nki(tree: ast.AST) -> bool:
    """True if the subtree references any NKI primitive — a call/attribute rooted
    at ``nki`` / ``nl`` (neuronx language) / ``nisa`` (ISA intrinsics), or a bare
    ``@nki``/``@nki.jit`` decorator. ``import nki`` alone does NOT count (an Import
    node yields no Name node), so this measures USE, not merely presence."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            parts = _attr_parts(node)
            if parts and parts[0] in {"nki", "nl", "nisa"}:
                return True
    return False


def _refs_framework(node: ast.AST) -> bool:
    """True if the subtree contains a torch/F compute call (used as a *return*
    value in a fallback handler, typically). Same COMPUTE-denylist discipline as
    the top-level scan."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            parts = _attr_parts(n.func)
            if _is_framework_compute(parts):
                return True
    return False


def _is_framework_compute(parts: list[str]) -> bool:
    """Decide whether a dotted call chain is a framework COMPUTE op:
      * ``F.<anything>``               — torch.nn.functional is compute by convention
      * ``*.functional.<anything>``    — the un-aliased spelling
      * ``torch.<op>`` for op in the compute denylist (dtype/alloc attrs excluded)
    """
    if not parts:
        return False
    if parts[0] == "F" and len(parts) >= 2:
        return True
    if "functional" in parts[:-1]:
        return True
    if parts[0] == "torch" and parts[-1] in _TORCH_COMPUTE_OPS:
        return True
    return False


def _text_only_check(nki_src: str) -> list[str]:
    """Fallback heuristics when the source does not parse as Python. We cannot do
    a structural analysis, so we flag on coarse substrings and note the downgrade
    honestly. A candidate that does not even parse is already suspect."""
    reasons = ["source did not parse as Python; ran text-only anti-cheat "
               "(structural checks skipped)"]
    low = nki_src
    if not any(tok in low for tok in ("nki.", "@nki", "nl.", "nisa.")):
        reasons.append("no NKI primitive (nki./nl./nisa./@nki) found in source")
    if "functional" in low or "F." in low or "torch." in low:
        reasons.append("references torch/F — possible framework fallback "
                       "(unverified: source did not parse)")
    return reasons


def adversarial_source_check(nki_src: str) -> list[str]:
    """Static anti-cheat over a candidate kernel's SOURCE. Returns a list of
    human-readable violation reasons (empty list == clean).

    Catches the four reward-hack shapes the corpus taught us:

      1. **No actual kernel.** No ``nki.``/``nl.``/``nisa.``/``@nki`` usage at all
         — the "kernel" contains no kernel.
      2. **Framework fallback.** torch/F COMPUTE ops are the compute path (the
         Sakana "it's really the reference" hack), OR an ``except`` handler
         returns a torch/F result (the bare-``except: return torch...`` fallback).
      3. **Error-swallowing try/except.** The kernel body (NKI usage) sits inside
         a ``try`` whose ``except`` swallows the failure and ``return``s a
         substitute — so a broken kernel silently reports a non-kernel result.
      4. **Reference inheritance / import.** The candidate imports or subclasses
         the reference/baseline implementation to avoid writing a kernel.

    HEURISTIC LIMITS (documented honestly — this is a tripwire, not a prover):
      * It reasons about SOURCE, not runtime. A kernel that dynamically ``eval``s
        a fallback, or hides compute behind an indirection we can't name-resolve,
        can slip past — that is what ``require_reproducible`` and the buffer-order
        protocol are for.
      * ``import torch`` for dtypes/allocation is intentionally NOT flagged, so a
        candidate that smuggles compute through an unusual torch attribute not in
        the denylist would be missed. The denylist is broad but not exhaustive.
      * On a SyntaxError we fall back to coarse substring checks (``_text_only_
        check``) and say so in the reasons.
    """
    if not nki_src or not nki_src.strip():
        return ["empty kernel source (no kernel)"]

    try:
        tree = ast.parse(nki_src)
    except SyntaxError:
        return _text_only_check(nki_src)

    reasons: list[str] = []

    # (1) is there a kernel at all?
    has_nki = _uses_nki(tree)
    if not has_nki:
        reasons.append("no NKI primitive (nki./nl./nisa./@nki) used — candidate "
                       "contains no kernel")

    # (2) framework compute as the compute path (Sakana fallback).
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_framework_compute(_attr_parts(node.func)):
            dotted = ".".join(_attr_parts(node.func))
            reasons.append(f"framework compute op used as compute path: "
                           f"{dotted}(...) — kernel falls back to the framework")
            break

    # (3)/(2b) try/except that swallows the kernel and returns a substitute.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body_has_nki = any(_uses_nki(stmt) for stmt in node.body)
        for handler in node.handlers:
            returns = [n for n in ast.walk(handler) if isinstance(n, ast.Return)]
            if not returns:
                continue
            # An except handler that returns a torch/F result is a silent
            # framework fallback regardless of what the try body held.
            if any(_refs_framework(r) for r in returns):
                reasons.append("except-handler returns a framework (torch/F) "
                               "result — silent fallback instead of raising")
            # The kernel body wrapped in try/except that swallows and returns:
            # even a non-framework return here means a broken kernel reports a
            # substitute result rather than failing honestly.
            elif body_has_nki:
                reasons.append("kernel body wrapped in try/except that swallows "
                               "the error and returns a non-kernel result")

    # (4) inherit-from / import the reference implementation.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            names = " ".join(a.name.lower() for a in node.names)
            if any(t in mod or t in names for t in _REFERENCE_TOKENS):
                reasons.append(f"imports the reference/baseline implementation "
                               f"(from {node.module}) instead of writing a kernel")
        elif isinstance(node, ast.Import):
            if any(t in a.name.lower() for a in node.names for t in _REFERENCE_TOKENS):
                reasons.append("imports the reference/baseline implementation "
                               "instead of writing a kernel")
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                btext = ".".join(_attr_parts(base)).lower()
                if any(t in btext for t in _REFERENCE_TOKENS):
                    reasons.append(f"class {node.name} inherits from the "
                                   f"reference/baseline implementation")
                    break

    # de-dup while preserving order (multiple try blocks can raise the same tell).
    seen: set[str] = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]
    return deduped


# -- (2) reproducibility gate -------------------------------------------------

def _equal(a: Any, b: Any) -> bool:
    """Best-effort exact equality across the result shapes a run_fn might return
    (numbers, tuples/lists, numpy/torch tensors, arbitrary objects). We want
    EXACT identity of results — this gate exists to catch nondeterminism, so we
    intentionally do NOT apply a tolerance here (that is the numerics gate's job).
    """
    # numpy / torch tensors: compare via .tolist() (avoids importing either).
    a_list = getattr(a, "tolist", None)
    b_list = getattr(b, "tolist", None)
    if callable(a_list) and callable(b_list):
        try:
            return a_list() == b_list()
        except Exception:  # noqa: BLE001 — fall through to generic compare
            pass
    try:
        res = a == b
    except Exception:  # noqa: BLE001 — uncomparable types
        return repr(a) == repr(b)
    if isinstance(res, bool):
        return res
    # array-like truthiness (numpy): reduce element-wise equality to one bool.
    try:
        return bool(all(res)) if hasattr(res, "__iter__") else bool(res)
    except Exception:  # noqa: BLE001
        return repr(a) == repr(b)


def require_reproducible(run_fn: Callable[[], Any], n: int = 2) -> tuple[bool, str]:
    """Run ``run_fn`` ``n`` times and require IDENTICAL results every time.

    The "runs twice, different answer" tell: a correct kernel on fixed inputs is
    deterministic. A candidate that reads an uninitialized scratch buffer, an
    aliased/recycled output, or has a race will drift between runs — and a
    single-shot numerics check can be fooled by whichever run happens to line up
    with the reference. Requiring bitwise-stable repeats closes that.

    Returns ``(ok, reason)``. ``ok`` is False if any run differs from the first
    (nondeterministic) or if ``run_fn`` raises (a kernel that cannot even be run
    reproducibly is not reusable). ``n`` is clamped to >= 2 (one run proves
    nothing about reproducibility).
    """
    n = max(2, int(n))
    try:
        first = run_fn()
    except Exception as e:  # noqa: BLE001 — a run that raises is not reproducible
        return False, f"run_fn raised on the first run: {e!r}"
    for i in range(1, n):
        try:
            cur = run_fn()
        except Exception as e:  # noqa: BLE001
            return False, f"run_fn raised on run {i + 1}/{n}: {e!r}"
        if not _equal(first, cur):
            return False, (f"non-reproducible: run {i + 1}/{n} differs from run 1 "
                           f"(nondeterministic output — reads racy/uninitialized/"
                           f"aliased state?)")
    return True, f"reproducible across {n} runs"


# -- (3) buffer-order protocol ------------------------------------------------

def run_candidate_before_reference(candidate_fn: Callable[[], Any],
                                   reference_fn: Callable[[], Any]) -> tuple[Any, Any]:
    """Run the CANDIDATE first, THEN the reference, and return
    ``(candidate_out, reference_out)`` for the caller's equivalence check.

    Why the order is load-bearing (Kevin-32B's recycled-output exploit): if the
    reference runs first and fills an output buffer that the candidate can reach
    (a shared/aliased tensor), a do-nothing candidate can simply return that
    already-correct buffer and pass allclose without computing anything. Running
    the candidate FIRST means there is no reference-produced result in existence
    for it to alias — it must produce its own output. The caller should still
    compare with the numerics gate, but the ordering removes the exploit by
    construction rather than trying to detect it after the fact.

    We do not catch exceptions here: a candidate that crashes must surface as a
    failure to the caller's gate, not be silently swallowed (swallowing is itself
    one of the hacks ``adversarial_source_check`` flags).
    """
    candidate_out = candidate_fn()
    reference_out = reference_fn()
    return candidate_out, reference_out
