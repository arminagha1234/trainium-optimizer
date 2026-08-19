"""
Pre-run REVIEW GATE — "review, then execute" (borrowed from the
NeurIPS-Trainium-Competition organizer design: an LLM reviewer + AST guard vet a
submission *before* it is ever run).

Adapted to this optimizer, whose candidates are (currently) structured configs
plus compiler-flag strings — so the gate validates:
  - config values are within the allowed axes (no malformed candidate reaches a
    5-20 min compile),
  - the NEURON_CC_FLAGS string is on an allowlist and free of shell
    metacharacters (it is passed to a subprocess env — this is the injection
    guard the competition's AST check provides).

`review_kernel_source` is the extension point for when Stage 3/4 start generating
NKI kernel source: it blocks obvious egress / dynamic-exec before the code runs,
exactly like the competition's forbidden-import guard. Not wired to the search
yet (no code is generated today) — present so the hook exists.
"""

from __future__ import annotations

import re

# Allowed values per config axis. A candidate proposing anything outside these is
# malformed and must not reach a compile.
ALLOWED: dict[str, set] = {
    "tp_degree": set(range(1, 65)),
    "weights_dtype": {"bf16", "fp32"},
    "attn_implementation": {"eager", "sdpa"},
    "compile_mode": {"eager", "compile-default"},
    "batch": {1, 2, 4, 8, 16, 32},
}

# Known-safe neuronx-cc tokens the deep stages (2-5) may set. Anything else in a
# cc_flags string is rejected — the compiler flags are attacker-controlled input
# to a subprocess, so we allowlist rather than blocklist.
_SAFE_CC_TOKENS: set[str] = {
    "--optlevel", "1", "2", "3",
    "--model-type", "transformer",
    "--auto-cast", "none",
    "--enable-fast-loading-neuron-binaries",
}
_SHELL_META = re.compile(r"[;&|`$><\n\\]")  # any of these -> reject as injection


def review_config(config: dict) -> tuple[str, str]:
    """Return (verdict, reason). verdict in {'PASS', 'REJECT'}.

    Backend-agnostic and injection-focused: legitimate candidates from any
    backend's proposer always PASS (values come from that backend's own axes).
    We REJECT only genuinely unsafe candidates — string values or compiler-flag
    strings that carry shell metacharacters, and cc_flags outside the allowlist
    (the flags are passed to a subprocess env, so they are the real attack
    surface). `known_axis_ok` is available for a stricter, backend-specific pass."""
    for k, v in config.items():
        if isinstance(v, str) and _SHELL_META.search(v):
            return "REJECT", f"{k} value has shell metacharacters: {v!r}"
    cc = str(config.get("cc_flags", "") or "")
    if cc:
        if _SHELL_META.search(cc):
            return "REJECT", f"cc_flags contains shell metacharacters: {cc!r}"
        unknown = [t for t in cc.split() if t not in _SAFE_CC_TOKENS]
        if unknown:
            return "REJECT", f"cc_flags has non-allowlisted tokens: {unknown}"
    return "PASS", "ok"


def known_axis_ok(config: dict) -> tuple[str, str]:
    """Stricter native-PyTorch-specific value check (opt-in). Rejects values
    outside the known axes — use only for the native backend, not mock."""
    for k, v in config.items():
        if k in ALLOWED and v not in ALLOWED[k]:
            return "REJECT", f"{k}={v!r} outside allowed set"
    return "PASS", "ok"


def review_kernel_source(src: str) -> tuple[str, str]:
    """Fast static gate for Stage 3/4 GENERATED kernel/model source (extension
    point). Codifies the NeurIPS-Trainium-Competition reviewer taxonomy
    (categories A-L / extraction pass E1-E12): a generated artifact is
    ADVERSARIAL until proven honest. This is the cheap deny-list; the deep,
    account-for-every-construct pass is an LLM reviewer (their design) that
    should run before any generated kernel is trusted.

    Blocks the clear-cut CRITICAL patterns:
      egress, subprocess, dynamic-exec, fs discovery, trusted-state mutation,
      unsafe checkpoint (pickle RCE), and eval-mode/scoring detection signals."""
    banned = (
        # E9 dynamic execution / indirection
        "eval(", "exec(", "__import__", "compile(", "importlib",
        # E-egress (network)
        "import socket", "socket.", "import requests", "requests.",
        "urllib", "http.client", "httpx", "curl", "wget",
        # F subprocess / process spawn
        "subprocess", "os.system", "os.popen", "pty.spawn", "os.exec",
        # E8/D filesystem discovery + val peeking
        "os.walk", "rglob(", "glob.glob", "scandir", "listdir",
        # G tampering with trusted/organizer state
        "prepare.", "setattr(prepare", "__dict__[",
        # G checkpoint-as-pickle RCE (require weights_only=True elsewhere)
        "weights_only=False", "pickle.load", "torch.load(",  # inspect: must be weights_only=True
        # E1 eval-mode / scoring detection signals
        "sys.modules", "os.environ", "__file__",
    )
    low = src
    for b in banned:
        if b in low:
            # torch.load is allowed ONLY with weights_only=True (safe tensors)
            if b == "torch.load(" and "weights_only=True" in low and "weights_only=False" not in low:
                continue
            return "REJECT", f"kernel source contains flagged construct: {b!r} (adversarial-review deny-list)"
    return "PASS", "ok"
